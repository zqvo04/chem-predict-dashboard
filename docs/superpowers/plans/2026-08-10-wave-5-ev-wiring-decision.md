# Wave 5 — EV 재구성: 배선할지 정하는 두 측정 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ROADMAP의 A2 항목은 "2부 모델 **+ EV 재구성**"이고 Wave 4는 앞의 절반만 했다. 배선 전에 아직 안 잰 것 두 가지를 재고, 그 결과로 배선 여부를 **사전등록된 규칙으로** 정한다.

**Architecture:** 새 모듈을 만들지 않는다. `scripts/two_part_audit.py`에 팔 하나를 더하고, 같은 pre-cut 재적합 위에서 gap 축을 재는 스크립트 하나를 새로 쓴다. 배선은 조건부이며, 조건이 안 맞으면 **배선하지 않는 것이 이 플랜의 정상 종료**다.

**Tech Stack:** Python 3.11 · pandas · numpy · scikit-learn · RDKit · pytest

## 왜 이것이 다음인가 — Wave 4가 남긴 두 구멍

**구멍 1: 감사의 BASELINE에는 게이트가 아예 없었다.**

`scripts/two_part_audit.py`의 BASELINE 팔은 회귀기 예측만 쓴다. 그런데 배포된 퍼널은
게이트를 **Tier 0.5의 하드 필터**로 이미 쓰고 있다(`funnel.screen_library`가
`binder_prob >= gate.threshold`로 자른다). 따라서 Wave 4가 답한 질문은

> "게이트 확률을 쓰는 것이 아예 안 쓰는 것보다 나은가" — **그렇다 (AUC 0.890 → 0.956)**

이고, 배포가 실제로 묻는 질문은

> "게이트 확률을 **곱하는** 것이 **자르는** 것보다 나은가" — **안 쟀다**

이다. 배포 퍼널은 이미 자르고 있으므로, 후자가 답이 없으면 배선의 근거가 없다.

**구멍 2: `P · gap`이 선택성 순위를 바꾸는데 그 효과를 안 쟀다.**

`EV(target) − max EV(off) = P · gap`은 맞지만, `P`가 분자마다 다르므로 곱셈은 순서를
바꾼다 — gap 2.0 / P 0.50(= 1.00)이 gap 1.2 / P 0.95(= 1.14)에 밀린다. Wave 4의 수치는
전부 target 아이소폼 점수, 즉 potency 축이다. **배선하면 shortlist 순서가 바뀌는데 그
변화가 이득인지 손해인지 모른다.**

두 구멍 모두 새 데이터가 필요 없다. 같은 pre-2020 재적합 위에서 잰다.

## Global Constraints

- 브랜치는 `claude/bioinformatics-computational-chemistry-wje9cs`. **PR은 명시 요청 없이 만들지 않는다.**
- **G0** — 이 플랜의 Task 1·2는 배포 자산을 건드리지 않는다. Task 3의 배선은 **조건부**이며 별도 승인 사안이다. 매 커밋에서 `git diff --stat main..HEAD -- assets/models assets/jak/JAK1.parquet assets/jak/JAK2.parquet assets/jak/JAK3.parquet assets/library`가 빈 출력이어야 한다.
- **G6** — 봉인 414는 채점만. 학습·튜닝·임계 선택에 안 쓴다.
- **G8** — 판정 규칙은 Task 1·2를 돌리기 **전에** 이 문서에 적혀 있고, 실행 후 고치지 않는다.
- **G5** — 배선 안 하기로 끝나도 그것을 결과로 기록한다.
- 시간 컷은 `panel_data.EVAL_TIME_CUT`(= 2020) 하나. 재적합 방식은 `two_part_audit.refit`을 그대로 쓴다.
- **런타임이 길다.** 전체 스위트 ~5분(경합 시 13분), `two_part_audit.py` ~25분. 타임아웃 600000 ms, 무거운 작업 두 개를 동시에 돌리지 않는다 — 4코어뿐이라 직렬이 병렬보다 빠르다.
- 오프라인. 커밋된 `assets/`만 읽는다.

---

## 사전등록 판정 규칙 (실행 전 고정)

| 측정 | 배선 조건 |
|---|---|
| **M1** 하드 게이트 대비 (Task 1) | 배포 캐스케이드의 활성 재현율과 **같은 재현율**에서 EV의 봉인 414 통과율이 **더 낮다** |
| **M2** gap 축 (Task 2) | post-cut 교차측정 위에서 `P · gap`의 gap-Spearman이 plain gap보다 **0.02 이상 나쁘지 않다** (동등 허용, 개선이면 더 좋음) |

**둘 다 만족해야 배선한다.** M1만 만족하고 M2가 깨지면 — potency는 좋아지는데 선택성이
나빠지면 — 배선하지 않는다. 이 퍼널의 산출물은 선택성 랭킹이고, potency는 그 랭킹에
들어갈 자격을 거르는 층이다. 자격 심사를 개선하려고 산출물을 나쁘게 만드는 교환은 하지 않는다.

**대안 경로도 미리 적는다.** M1은 만족하고 M2가 깨지면, 다음 후보는 "EV를 랭킹이 아니라
**Tier 1 통과 판정에만** 쓰는 것"이다 — 즉 `meets_floor`를 `EV >= FLOOR`로 바꾸고 정렬은
plain gap으로 남긴다. 그 변형은 Task 2가 M2를 깬 경우에만 열리며, 별도 측정을 요구한다.

---

## File Structure

| 파일 | 책임 | 태스크 |
|---|---|---|
| `scripts/two_part_audit.py` (수정) | 배포 캐스케이드 팔 추가 — 하드 게이트 + potency floor, 같은 재적합 | 1 |
| `scripts/ev_gap_audit.py` (신규) | post-cut 교차측정 위 `gap` vs `P · gap`의 선택성 랭킹 비교 | 2 |
| `STATE.md` · `VALIDATION.md` · `scripts/reproduce.sh` (수정) | 두 측정과 배선 판정 기록 | 3 |
| `src/funnel.py` (조건부 수정) | **M1·M2 둘 다 만족한 경우에만** EV 배선 | 3 |

---

### Task 1: 배포 캐스케이드를 세 번째 팔로 (M1)

**무엇을 재는가.** 배포 퍼널은 `binder_prob >= threshold`로 자르고 살아남은 것에
`pred_target >= POTENCY_FLOOR`를 건다. 이 팔을 감사에 넣으면 세 팔이 **같은 pre-2020
재적합**을 공유하므로 비로소 비교 가능해진다.

**하드 필터를 곡선 위에 올리는 법.** 하드 필터는 점수가 아니라 한 점이다. 게이트가 자른
분자에 `-inf`를 주면 하나의 점수가 되고, 그러면 기존 `at_matched_recall`이 그대로 쓰인다 —
게이트 통과분 안에서는 회귀기 값으로 순위가 매겨지고, 잘린 것은 항상 최하위다. 새 헬퍼가
필요 없다.

**Files:**
- Modify: `scripts/two_part_audit.py`

**Interfaces:**
- Consumes: `binder_gate.youden_threshold` · `binder_gate.proba_aligned` (Wave 4에서 승격됨), `two_part_audit.refit`
- Produces: 없음 (측정 스크립트)

---

- [ ] **Step 1: 재적합 게이트의 임계를 구한다**

`refit`이 게이트를 돌려주지만 임계는 안 준다. `main`이 `refit` 직후에 계산하게 한다.
**측정 음성이나 봉인 414 위에서 임계를 잡지 않는다** — 판정 대상 위에서 고른 임계는
측정이 아니다(G6, 그리고 `youden_threshold`의 docstring이 이미 그 규범을 담고 있다).

`scripts/two_part_audit.py`의 import에 추가:

```python
from src.models.binder_gate import (at_matched_recall, build_gate,      # noqa: E402
                                    proba_aligned, youden_threshold)
```

`refit` 안에서 게이트를 만든 직후, 홀드아웃 추정 음성으로 임계를 잡고 함께 돌려준다.
`refit`의 반환을 `(regressors, gate, threshold)`로 바꾼다:

```python
    positives = sorted(s for s in positive_smiles(panel) if year.get(s, 9999) <= CUT)
    presumed = build_negatives(panel)["smi"].tolist()
    # 추정 음성의 20 %를 임계 산출에만 쓰고 학습에서 뺀다. gate_ab_audit이 같은 이유로
    # 같은 비율을 쓴다: 학습에 쓴 음성 위의 Youden 점은 홀드아웃 점이 아니다.
    # `build_negatives`는 타깃별 프레임을 이어 붙이므로 위치 슬라이스는 홀드아웃을
    # 하드 키나아제에 몰아준다 — 자르기 전에 섞는다.
    shuffled = (build_negatives(panel).sample(frac=1.0, random_state=0)["smi"].tolist())
    cut = int(len(shuffled) * 0.2)
    holdout_presumed, train_presumed = shuffled[:cut], shuffled[cut:]
    gate = build_gate(positives, train_presumed)
    threshold = youden_threshold(gate, positives, holdout_presumed)
    print(f"  gate refit on {len(positives)} pre-{CUT} positives "
          f"+ {len(train_presumed)} presumed negatives, Youden threshold {threshold:.3f}")
    return regressors, gate, threshold
```

`presumed` 변수를 쓰던 줄이 남아 있으면 지운다(내가 만든 고아만 치운다).

`main`의 호출부:

```python
    regressors, gate, gate_threshold = refit(panel, year)
```

- [ ] **Step 2: 캐스케이드 팔을 arms에 넣는다**

`arms` 딕셔너리를 문자열 키에서 `(이름, 종류)`로 바꾸지 않는다 — 종류는 세 가지고 분기
두 개면 충분하다. `arms`에 항목을 추가하고 분기를 하나 더한다:

```python
    arms = {"BASELINE (regressor only)": None,
            "CASCADE (hard gate + regressor, as deployed)": "cascade",
            f"TWO-PART (floor {CENSORED_FLOOR})": CENSORED_FLOOR,
            f"TWO-PART (floor {FLOOR_SENSITIVITY})": FLOOR_SENSITIVITY}
```

루프 안의 분기를 이렇게 만든다:

```python
        if floor is None or floor == "cascade":
            X, mask = morgan_matrix(mols["smi"].tolist())
            per_iso = {}
            for iso in panel.isoforms:
                col = np.full(len(mols), np.nan)
                col[mask] = regressors[iso].predict(X)
                per_iso[iso] = col
            Xa, ma = morgan_matrix(actives["smi"].tolist())
            pos = np.full(len(actives), np.nan)
            pos[ma] = regressors[target].predict(Xa)
            if floor == "cascade":
                # 게이트가 자른 분자는 항상 최하위. 하드 필터를 하나의 점수로 만들어
                # 같은 matched-recall 곡선 위에 올린다.
                p_neg = proba_aligned(gate, mols["smi"].tolist())
                p_pos = proba_aligned(gate, actives["smi"].tolist())
                for iso in panel.isoforms:
                    per_iso[iso] = np.where(p_neg >= gate_threshold, per_iso[iso], -np.inf)
                pos = np.where(p_pos >= gate_threshold, pos, -np.inf)
        else:
            model = TwoPart(gate=gate, regressors=regressors, floor=floor)
            per_iso = model.expected_potency(mols["smi"].tolist())
            pos = model.expected_potency(actives["smi"].tolist())[target]
```

`-inf`가 A(반증률)에 새는 것을 막는다: `score > pchembl_upper`는 `-inf`에서 자동으로
False이므로 그대로 두면 맞다. 다만 중앙값 출력이 `-inf`가 될 수 있으니 그 줄만 고친다:

```python
        finite_neg = neg[np.isfinite(neg)]
        print(f"     median {target} score "
              f"{np.median(finite_neg) if len(finite_neg) else float('nan'):.2f} on sealed "
              f"({len(neg) - len(finite_neg)} gated out), "
              f"{np.median(pos_clean[np.isfinite(pos_clean)]):.2f} on post-cut actives")
```

`neg`/`pos_clean`을 만들 때 `~np.isnan(...)`은 그대로 둔다 — `-inf`는 NaN이 아니므로
남아야 하고, 남아야 matched-recall이 "잘렸다"를 셀 수 있다.

- [ ] **Step 3: VERDICT 블록의 기준선을 캐스케이드로 바꾼다**

배포가 실제로 묻는 질문이 M1이므로, 판정의 기준선은 BASELINE이 아니라 CASCADE다.
BASELINE 행은 Wave 4와의 연속성을 위해 표에 남기되 판정에서는 빼고, 두 기준선을
모두 출력한다:

```python
    print("\n" + "=" * 78)
    print("VERDICT — pre-registered rule (M1: against the DEPLOYED cascade)")
    print("=" * 78)
    ref_neg, ref_pos = curves["CASCADE (hard gate + regressor, as deployed)"]
    ref_auc = _auc(ref_pos[np.isfinite(ref_pos)], ref_neg[np.isfinite(ref_neg)])
    _, ref_rate = at_matched_recall(ref_neg, ref_pos, 0.95)
    print(f"  reference CASCADE: sealed pass at 95% recall {ref_rate:.1%}"
          f"   (finite-score AUC {ref_auc:.3f})")
    for name, (neg, pos_clean) in curves.items():
        if not name.startswith("TWO-PART"):
            continue
        _, rate = at_matched_recall(neg, pos_clean, 0.95)
        print(f"  {name}:  sealed pass at 95% recall {rate:.1%} "
              f"({rate - ref_rate:+.1%})   -> M1 {'PASS' if rate < ref_rate else 'FAIL'}")
    print("\n  M1 alone does not wire anything. Task 2 (M2, the gap axis) must also")
    print("  pass; the rule is in the Wave 5 plan and was fixed before this ran.")
```

AUC를 캐스케이드에 대해 쓸 때 `-inf`를 제외한 것에 주의한다 — `roc_auc_score`는 `-inf`를
받지만, 잘린 분자를 뺀 AUC는 "게이트를 통과한 것들 안에서의 판별력"이라는 다른 질문이다.
그래서 판정은 AUC가 아니라 **matched-recall 한 지표로만** 한다. 위 코드가 그렇게 되어 있다.

- [ ] **Step 4: 돌린다**

Run: `python scripts/two_part_audit.py` (타임아웃 600000 ms, 다른 무거운 작업과 겹치지 않게)

두 눈 검증:
1. CASCADE의 활성 재현율이 100 %가 아니다 — 게이트가 실제 활성 일부를 자른다. 100 %면
   게이트가 아무것도 안 자른 것이므로 임계나 `proba_aligned` 정렬이 틀렸다.
2. CASCADE의 봉인 414 통과율이 BASELINE보다 **낮다**. 안 낮으면 하드 필터가 아무 일도
   안 한 것이다.

둘 중 하나라도 깨지면 커밋하지 말고 보고한다.

- [ ] **Step 5: 커밋**

```bash
git add scripts/two_part_audit.py
git commit -m "Put the deployed cascade on the same curve as the two-part model"
```

---

### Task 2: gap 축 — `P · gap`이 선택성 랭킹에 무엇을 하는가 (M2)

**왜 교차측정 세트를 그냥 쓸 수 없는가.** 게이트의 양성은 "패널 활성 pchembl ≥ 6"이고
교차측정 분자는 대부분 거기 들어간다. 게이트에 물어보면 `P ≈ 1`을 외워서 돌려주므로
곱셈이 아무 일도 안 하는 것처럼 보인다. **컷 이후 분자에만 물어야 한다** — pre-2020
재적합 게이트가 못 본 것들이고, 그 수는 1,078로 이미 `TimeSplitOracle` 모집단과 같다.

**Files:**
- Create: `scripts/ev_gap_audit.py`

**Interfaces:**
- Consumes: `two_part_audit.refit`은 **import하지 않는다** — 스크립트 간 import는 pytest에서
  깨진다(저장소가 이미 두 번 겪었다). 재적합 코드는 짧으므로 이 스크립트가 자기 것을 갖는다.
- Produces: 없음

---

- [ ] **Step 1: 스크립트를 쓴다**

`scripts/ev_gap_audit.py`:

```python
#!/usr/bin/env python3
"""M2: P(binder)를 곱하면 선택성 랭킹이 어떻게 되는가 (Wave 5).

EV(target) - max_off EV(off) = P * gap 이므로 gap에 새 정보는 안 들어온다. 그러나 P가
분자마다 다르므로 곱셈은 순서를 바꾼다 — gap 2.0 / P 0.50(= 1.00)이 gap 1.2 / P 0.95
(= 1.14)에 밀린다. STATE.md 2c는 potency 축만 쟀고, 배선은 이 축을 바꾼다.

컷 이후 분자에만 묻는다. 게이트의 양성은 "패널 활성 pchembl >= 6"이고 교차측정 분자는
대부분 거기 있으므로, 컷 이전 분자에 물으면 외운 P ~ 1이 돌아오고 곱셈이 아무 일도 안
하는 것처럼 보인다.

    python scripts/ev_gap_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import panel_data                                       # noqa: E402
from src.data.negatives import build_negatives, positive_smiles       # noqa: E402
from src.models.binder_gate import build_gate, proba_aligned          # noqa: E402
from src.models.features import morgan_matrix                         # noqa: E402
from src.models.isoform_regressor import _fit                         # noqa: E402
from src.panels import DEFAULT_PANEL                                  # noqa: E402
from src.selectivity import SELECTIVE_GAP, _enrichment                # noqa: E402

CUT = panel_data.EVAL_TIME_CUT


def main() -> None:
    panel = DEFAULT_PANEL
    year = panel_data.year_first(panel)

    print("=" * 78)
    print(f"M2 — gap vs P * gap on cross-measured molecules published after {CUT}")
    print("=" * 78)

    regressors = {}
    for iso in panel.isoforms:
        data = panel_data.build_isoform_dataset(panel, iso)
        pre = data[data["smi"].map(year).fillna(9999) <= CUT]
        X, mask = morgan_matrix(pre["smi"].tolist())
        regressors[iso] = _fit(X, pre["pchembl"].to_numpy()[mask])
        print(f"  {iso}: regressor refit on {int(mask.sum())} pre-{CUT} molecules")

    positives = sorted(s for s in positive_smiles(panel) if year.get(s, 9999) <= CUT)
    gate = build_gate(positives, build_negatives(panel)["smi"].tolist())
    print(f"  gate refit on {len(positives)} pre-{CUT} positives")

    cross = panel_data.build_cross_measured(panel)
    post = cross[cross["smi"].map(year) > CUT].reset_index(drop=True)
    smiles = post["smi"].tolist()
    X, mask = morgan_matrix(smiles)
    post = post[mask].reset_index(drop=True)
    kept = [s for s, keep in zip(smiles, mask) if keep]

    measured_gap = (post[panel.target].to_numpy()
                    - post[list(panel.offs)].max(axis=1).to_numpy())
    preds = {iso: regressors[iso].predict(X) for iso in panel.isoforms}
    gap = preds[panel.target] - np.maximum.reduce([preds[o] for o in panel.offs])
    p = proba_aligned(gate, kept)
    ev_gap = p * gap

    base = float((measured_gap >= SELECTIVE_GAP).mean())
    print(f"\n  post-{CUT} cross-measured n = {len(kept)}"
          f"   >= {SELECTIVE_GAP} log selective: {base:.1%}")
    print(f"  P(binder) on these: median {np.median(p):.3f}, "
          f"below 0.5 in {(p < 0.5).mean():.1%}")
    if np.median(p) > 0.99:
        print("  WARNING: P is saturated, so the multiplication cannot move anything."
              "\n  That is a null result about this population, not about the model.")

    print(f"\n  {'ranking':>12} {'Spearman':>10} {'top-decile enrichment':>23}")
    for label, score in (("gap", gap), ("P * gap", ev_gap)):
        rho = float(spearmanr(measured_gap, score).statistic)
        print(f"  {label:>12} {rho:>10.3f} {_enrichment(measured_gap, score):>22.2f}x")

    moved = pd.DataFrame({"smi": kept, "gap": gap, "ev_gap": ev_gap, "p": p})
    top_plain = set(moved.nlargest(60, "gap")["smi"])
    top_ev = set(moved.nlargest(60, "ev_gap")["smi"])
    print(f"\n  Top-60 shortlist overlap between the two rankings: "
          f"{len(top_plain & top_ev)} / 60")
    print("  That overlap is what a wired deployment would change about the shortlist.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 돌린다**

Run: `python scripts/ev_gap_audit.py` (타임아웃 600000 ms)

두 눈 검증:
1. `post-2020 cross-measured n`이 1,078이다. `TimeSplitOracle` 모집단과 같아야 한다.
2. `P(binder)` 중앙값이 0.99를 넘으면 스크립트가 스스로 WARNING을 찍는다. 그 경우
   Spearman 두 줄이 사실상 같게 나오는 것이 정상이며, **그것은 곱셈이 이 모집단에서
   할 일이 없다는 결과이지 버그가 아니다.** 그대로 기록한다(G5).

- [ ] **Step 3: 커밋**

```bash
git add scripts/ev_gap_audit.py
git commit -m "Measure what multiplying by P(binder) does to the selectivity ranking"
```

---

### Task 3: 판정과 기록 (배선은 조건부)

- [ ] **Step 1: 두 규칙을 대조한다**

| | 통과 조건 | 결과 |
|---|---|---|
| **M1** | TWO-PART(floor 5.0)의 봉인 414 통과율 < CASCADE의 것 (활성 재현율 95 % 고정) | Task 1 출력 |
| **M2** | `P · gap`의 Spearman ≥ plain gap의 Spearman − 0.02 | Task 2 출력 |

- [ ] **Step 2: 배선 — M1·M2 둘 다 통과한 경우에만**

통과했으면 `src/funnel.py`를 고친다. `screen_library`는 이미 `binder_prob`를 계산하고
하드 필터로 쓴 뒤 **버린다.** 그것을 Tier 1으로 넘기는 것이 배선의 전부다.

`_predict`에 인자를 하나 더한다:

```python
def _predict(df, models, isoforms, target, offs, floor_value=None):
    """Tier 1 (cheap): per-isoform predictions, gap S, potency floor.

    `floor_value`가 주어지면 Tier 0.5가 계산한 `binder_prob`를 버리지 않고 기대값으로
    합친다: EV = P * pred + (1 - P) * floor_value. STATE.md 2c / VALIDATION.md 참조.
    """
```

본문 끝의 세 줄을 이렇게 바꾼다:

```python
    for iso in isoforms:
        df[f"pred_{iso}"] = np.concatenate(preds[iso])
    if floor_value is not None:
        p = df["binder_prob"].to_numpy()
        for iso in isoforms:
            df[f"pred_{iso}"] = p * df[f"pred_{iso}"] + (1.0 - p) * floor_value
    df["gap"] = df[f"pred_{target}"] - df[[f"pred_{o}" for o in offs]].max(axis=1)
    df["meets_floor"] = df[f"pred_{target}"] >= POTENCY_FLOOR
```

`screen_library`의 호출부에 `floor_value=CENSORED_FLOOR`를 넘기고, `two_part`에서
그 상수를 import한다.

**그리고 이것은 G0 사건이다.** 배선하면 배포 shortlist, `gap_distribution`, gap 백분위,
그 아래 VALIDATION 수치 전체가 움직인다. 배선 커밋에는 반드시 다음이 함께 들어간다:

- `python -m src.funnel`을 다시 돌려 새 shortlist를 기록
- `funnel.library_gap_distribution`의 provenance 문자열이 바뀌는지 확인하고, 바뀌면
  그 사실을 STATE.md에 적는다
- `python -m pytest tests/ -q` 전체 통과
- 배선 **전후** shortlist를 나란히 기록 — 몇 개가 바뀌었는지가 기록의 핵심이다(G5)

- [ ] **Step 3: 배선하지 않기로 한 경우**

`src/funnel.py`를 건드리지 않는다. STATE.md에 **왜 안 했는지**를 두 측정값과 함께 적는다.
"측정했고 조건이 안 맞아서 안 했다"는 이 저장소에서 정상 종료다 — §2b가 같은 형태다.

- [ ] **Step 4: 문서와 재현 경로**

`scripts/reproduce.sh`의 two-part 블록 **뒤**에 추가:

```bash
echo
echo "== EV vs the deployed cascade, gap axis (Wave 5, M2) =="
echo "What multiplying the gap by P(binder) does to the selectivity ranking on"
echo "molecules published after the time cut."
python scripts/ev_gap_audit.py
```

STATE.md에 `### 2d. EV 배선 판정 (2026-08-__)`를 §2c 뒤에 넣고 다음을 담는다:
재현 명령 두 개, M1·M2 표, 사전등록 규칙과 그 결과, 배선 여부와 근거, 한계.

VALIDATION.md에는 M1·M2 두 표와 top-60 shortlist 중첩 수를 싣는다. 중첩 수가
"배선이 실제로 무엇을 바꾸는가"의 유일한 구체적 답이다.

- [ ] **Step 5: 게이트 확인 후 커밋·푸시**

```bash
python -m pytest tests/ -q
git diff --stat main..HEAD -- assets/models assets/jak/JAK1.parquet \
    assets/jak/JAK2.parquet assets/jak/JAK3.parquet assets/library
git add -A && git commit -m "Decide the EV wiring on the two measurements it needed"
git push -u origin claude/bioinformatics-computational-chemistry-wje9cs
```

배선하지 않았다면 두 번째 명령은 빈 출력이어야 한다. 배선했다면 여전히 빈 출력이어야
한다 — 배선은 코드를 바꾸지 자산을 바꾸지 않는다. 자산이 움직였다면 의도치 않은 재빌드다.

---

## 이 웨이브 이후

- **§8 튜닝 봉쇄가 A2 완료로 풀린다.** 열되, 프로토콜(중첩 CV·봉인 금지·목적함수 선언)을
  먼저 문서에 적는 것이 N16이고 그건 이미 되어 있다.
- 지도 문서 기준 다음은 **W6 카이랄**(N21·N22·N18) 또는 **W7 M축 구매가능성**.
  DMTA 준비도 표의 유일한 "부분"은 여전히 M축이다.
- Wave 4의 미완 1건: `scripts/assay_time_audit.py` 출력이 VALIDATION.md에 아직 없다.
  코드와 검증은 커밋됐고 실행만 남았다.
