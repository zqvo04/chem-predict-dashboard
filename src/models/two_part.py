"""A2: 2부 모델 — P(binder) x E[potency | binder] (ROADMAP 3단계, open question 1).

배포된 Tier 1은 회귀기 출력만으로 랭킹한다. 회귀기는 정량화된 pchembl로만 학습했으므로
패널에 붙은 적 없는 분자에는 학습 평균(JAK ~ 6.3)을 돌려주고, 그 분자는 potency floor를
통과한다. STATE.md 2절이 봉인 414에서 93.0 %의 예측이 그 분자 자신의 측정 상한을 넘는다고
측정한 것이 이 구조다.

2부 모델이 말하는 것은 퍼널이 원하는 값이 "붙는다면 얼마나 센가"가 아니라 "얼마나 셀
것인가"이고, 둘은 정확히 붙을 확률만큼 다르다는 것이다:

    EV(potency) = P(binder) * reg(molecule) + (1 - P(binder)) * FLOOR

FLOOR는 비결합자가 보이는 potency다. 0도 아니고 학습 평균도 아니며, assay가 멈춘 농도다.

**차분에서 floor는 상쇄된다.** 게이트는 패널당 하나라 P가 모든 아이소폼에서 같으므로

    EV(target) - max_off EV(off) = P * (reg(target) - max_off reg(off))

즉 gap에는 새 정보가 들어오지 않는다 — 기존 gap에 게이트 확률을 곱한 것뿐이다.

**그러나 이것이 순위를 보존한다는 뜻은 아니다.** P가 분자마다 다르므로 곱셈은 순서를
바꾼다: gap 2.0 / P 0.50(= 1.00)은 gap 1.2 / P 0.95(= 1.14)에 밀린다. 낮은 P를 강등하는
것이 의도이지만, 그 강등이 선택성 축에서 이득인지 손해인지는 **측정되지 않았다** —
STATE.md 2c는 potency 축만 잰다.

여기서 무엇도 배포하지 않는다. 게이트나 회귀기를 바꾸면 배포 임계와 그 아래 VALIDATION
수치 전체가 움직인다(G0). 이 모듈은 scripts/two_part_audit.py가 쓰는 측정 도구다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..panels import PanelSpec
from .features import morgan_matrix

# 봉인 414을 채점하기 전에 커밋된다 — 상수에 적용한 G8. 5.0은 pchembl 10 uM, 키나아제
# 패널이 보통 멈추는 농도, 즉 assay가 아무것도 못 봤을 때의 potency다. 4.0(100 uM)은
# 민감도 점검용이며, 둘 다 보고할 뿐 그 사이를 탐색하지 않는다.
CENSORED_FLOOR = 5.0
FLOOR_SENSITIVITY = 4.0


@dataclass(frozen=True)
class TwoPart:
    """동결된 게이트 + 아이소폼별 회귀기를 기대 potency로 합친 것."""

    gate: object                                  # .predict_proba(X)[:, 1]
    regressors: dict                              # isoform -> .predict(X)
    floor: float = CENSORED_FLOOR

    def _score(self, smiles: list[str]):
        smiles = list(smiles)
        X, mask = morgan_matrix(smiles)
        p = (self.gate.predict_proba(X)[:, 1] if X.shape[0]
             else np.empty(0, dtype=float))
        return X, mask, p, len(smiles)

    def p_binder(self, smiles: list[str]) -> np.ndarray:
        """P(binder), 입력과 정렬. 파싱 실패는 NaN."""
        _, mask, p, n = self._score(smiles)
        out = np.full(n, np.nan)
        out[mask] = p
        return out

    def expected_potency(self, smiles: list[str]) -> dict[str, np.ndarray]:
        """아이소폼별 EV(pchembl). 각 배열은 입력과 정렬되고 파싱 실패는 NaN."""
        X, mask, p, n = self._score(smiles)
        out = {}
        for iso, model in self.regressors.items():
            ev = np.full(n, np.nan)
            if X.shape[0]:
                ev[mask] = p * model.predict(X) + (1.0 - p) * self.floor
            out[iso] = ev
        return out

    def expected_gap(self, smiles: list[str], panel: PanelSpec) -> np.ndarray:
        """EV(target) - max_off EV(off). floor는 상쇄되므로 이는 P * gap과 같다."""
        ev = self.expected_potency(smiles)
        return ev[panel.target] - np.maximum.reduce([ev[o] for o in panel.offs])
