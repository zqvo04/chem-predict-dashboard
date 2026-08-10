#!/usr/bin/env python3
"""A2: 2부 모델이 potency 축의 반증을 실제로 줄이는가 (ROADMAP open question 1).

    EV = P(binder) * reg + (1 - P) * FLOOR

두 팔을 **같은 모델**로 잰다. 회귀기와 게이트를 둘 다 시간 컷 이전 데이터로 재적합하므로,
봉인 414(음성)와 컷 이후 실측 활성(양성)이 같은 점수 위에 놓인다. 그것이
funnel_falsification_audit이 스스로 못 한다고 적어 둔 비교이며, 여기서 닫힌다.

세 지표를 보고하고, 판정은 두 번째와 세 번째로만 한다:

  A 반증률          score > 측정 상한.  EV는 P<=1이라 반드시 내려간다 — 참고용
  B matched-recall  같은 활성 재현율에서의 봉인 414 통과율
  C ROC-AUC         활성 vs 봉인 414, 임계 없음

A만 보면 STATE.md 2b가 기록한 함정을 반복한다: 점수 분포가 통째로 내려간 보정 이동이
판별력 개선으로 보고된다.

봉인 414은 채점만 된다. FLOOR는 src/models/two_part.py에 먼저 커밋된 상수이며 여기서
탐색되지 않는다 (G6, G8).

    python scripts/two_part_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import panel_data                                       # noqa: E402
from src.data.negatives import build_negatives, positive_smiles       # noqa: E402
from src.models.binder_gate import at_matched_recall, build_gate      # noqa: E402
from src.models.features import morgan_matrix                         # noqa: E402
from src.models.isoform_regressor import _fit                         # noqa: E402
from src.models.two_part import (CENSORED_FLOOR, FLOOR_SENSITIVITY,   # noqa: E402
                                 TwoPart)
from src.panels import DEFAULT_PANEL                                  # noqa: E402

CUT = panel_data.EVAL_TIME_CUT
ACTIVE_PCHEMBL = 6.0
RECALLS = (0.99, 0.95, 0.90)


def refit(panel, year) -> tuple[dict, object]:
    """컷 이전 데이터로 재적합한 (아이소폼별 회귀기, 게이트).

    `_fit`을 직접 쓰는 이유는 `train_and_cache`가 아무도 읽지 않는 5시드 평가를 덤으로
    돌기 때문이다 — 아이소폼마다 다섯 번의 추가 적합.
    """
    regressors = {}
    for iso in panel.isoforms:
        data = panel_data.build_isoform_dataset(panel, iso)
        pre = data[data["smi"].map(year).fillna(9999) <= CUT]
        X, mask = morgan_matrix(pre["smi"].tolist())
        regressors[iso] = _fit(X, pre["pchembl"].to_numpy()[mask])
        print(f"  {iso}: regressor refit on {int(mask.sum())} pre-{CUT} molecules")

    positives = sorted(s for s in positive_smiles(panel) if year.get(s, 9999) <= CUT)
    presumed = build_negatives(panel)["smi"].tolist()
    gate = build_gate(positives, presumed)
    print(f"  gate refit on {len(positives)} pre-{CUT} positives "
          f"+ {len(presumed)} presumed negatives")
    return regressors, gate


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return float(roc_auc_score(y, np.concatenate([pos, neg])))


def main() -> None:
    panel = DEFAULT_PANEL
    target = panel.target
    year = panel_data.year_first(panel)

    print("=" * 78)
    print(f"A2 — two-part model, both arms on the same pre-{CUT} refit")
    print("=" * 78)
    regressors, gate = refit(panel, year)

    # 커밋된 봉인 파일을 읽는다 — `censored_library_molecules`로 다시 계산하면 증거
    # 저장소가 자랄 때 모집단이 같이 움직이고, 그러면 G8이 요구하는 사전등록이 아니다.
    sealed = pd.read_parquet(panel.data_bundled / "sealed_negatives.parquet")
    mols = sealed[["inchikey", "smi"]].drop_duplicates("inchikey").reset_index(drop=True)

    actives = panel_data.build_isoform_dataset(panel, target)
    actives = actives[(actives["smi"].map(year) > CUT)
                      & (actives["pchembl"] >= ACTIVE_PCHEMBL)].reset_index(drop=True)
    print(f"\n  sealed non-binders {len(mols)} molecules, {len(sealed)} bounds")
    print(f"  post-{CUT} measured actives (pchembl >= {ACTIVE_PCHEMBL}) {len(actives)}")

    arms = {"BASELINE (regressor only)": None,
            f"TWO-PART (floor {CENSORED_FLOOR})": CENSORED_FLOOR,
            f"TWO-PART (floor {FLOOR_SENSITIVITY})": FLOOR_SENSITIVITY}

    curves = {}
    for name, floor in arms.items():
        if floor is None:
            X, mask = morgan_matrix(mols["smi"].tolist())
            per_iso = {}
            for iso in panel.isoforms:
                col = np.full(len(mols), np.nan)
                col[mask] = regressors[iso].predict(X)
                per_iso[iso] = col
            Xa, ma = morgan_matrix(actives["smi"].tolist())
            pos = np.full(len(actives), np.nan)
            pos[ma] = regressors[target].predict(Xa)
        else:
            model = TwoPart(gate=gate, regressors=regressors, floor=floor)
            per_iso = model.expected_potency(mols["smi"].tolist())
            pos = model.expected_potency(actives["smi"].tolist())[target]

        # A — falsification against each molecule's own measured upper bound
        long = pd.concat(
            [pd.DataFrame({"inchikey": mols["inchikey"], "isoform": iso,
                           "score": per_iso[iso]}) for iso in panel.isoforms],
            ignore_index=True)
        paired = sealed.merge(long, on=["inchikey", "isoform"]).dropna(subset=["score"])
        paired["exceeds"] = paired["score"] > paired["pchembl_upper"]
        per_mol = paired.groupby("inchikey")["exceeds"].any()

        neg = per_iso[target][~np.isnan(per_iso[target])]
        pos_clean = pos[~np.isnan(pos)]
        curves[name] = (neg, pos_clean)

        print("\n" + "=" * 78)
        print(name)
        print("=" * 78)
        print(f"  A  falsification rate   {per_mol.sum()} / {len(per_mol)} "
              f"= {per_mol.mean():.1%}   (any isoform over its own bound)")
        print(f"     median {target} score {np.median(neg):.2f} on sealed, "
              f"{np.median(pos_clean):.2f} on post-cut actives")
        print(f"  C  ROC-AUC actives vs sealed 414   {_auc(pos_clean, neg):.3f}")
        print(f"\n  B  {'actives kept':>13} {'threshold':>11} {'sealed 414 pass':>17}")
        for recall in RECALLS:
            t, rate = at_matched_recall(neg, pos_clean, recall)
            print(f"     {recall:12.0%} {t:11.2f} {rate:16.1%}")

    print("\n" + "=" * 78)
    print("VERDICT — pre-registered rule")
    print("=" * 78)
    base_neg, base_pos = curves["BASELINE (regressor only)"]
    base_auc = _auc(base_pos, base_neg)
    _, base_rate = at_matched_recall(base_neg, base_pos, 0.95)
    for name, (neg, pos_clean) in curves.items():
        if name.startswith("BASELINE"):
            continue
        auc = _auc(pos_clean, neg)
        _, rate = at_matched_recall(neg, pos_clean, 0.95)
        adopt = (auc - base_auc >= 0.02) and (rate < base_rate)
        print(f"  {name}:  AUC {auc:.3f} ({auc - base_auc:+.3f})   "
              f"sealed pass at 95% recall {rate:.1%} ({rate - base_rate:+.1%})   "
              f"-> {'ADOPT' if adopt else 'FALSIFIED'}")

    print("\n  Read B and C, not A. EV <= regressor for every molecule because P <= 1,")
    print("  so A improves whether or not discrimination did. STATE.md section 2b is")
    print("  the same mistake caught one wave earlier.")


if __name__ == "__main__":
    main()
