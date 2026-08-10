#!/usr/bin/env python3
"""P4.1: 선택성 gap을 한 논문 안에서 만들면 무엇이 달라지는가 (ROADMAP open question 3).

배포된 라벨은 아이소폼마다 모든 문헌에 걸친 pchembl 중앙값이다. 그래서 gap이 서로 다른
논문, 서로 다른 조건에서 나온 두 값의 차이일 수 있다. 여기서는 같은 지표를 문헌 내
매칭 라벨로 다시 잰다.

이 비교에는 표본 크기 통제가 필요 없다 — 커밋된 저장소에서 두 모집단은 3,624 대 3,614로
사실상 같은 분자들이고, 바뀌는 것은 라벨 값뿐이다. 그것을 이 스크립트가 먼저 출력한다.

`src.selectivity.evaluate_split`을 그대로 쓰므로 숫자는 VALIDATION.md의 Gate 4와 직접
비교된다.

    python scripts/matched_label_audit.py          # 약 3분
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import panel_data                                   # noqa: E402
from src.models.features import morgan_matrix                     # noqa: E402
from src.models.scaffold_split import scaffold_split              # noqa: E402
from src.panels import DEFAULT_PANEL                              # noqa: E402
from src.selectivity import OFFS, SEEDS, TARGET, evaluate_split   # noqa: E402


def _seeds(cross) -> dict:
    smiles = cross["smi"].tolist()
    X, mask = morgan_matrix(smiles)
    cross = cross[mask].reset_index(drop=True)
    kept = [s for s, keep in zip(smiles, mask) if keep]
    gap = cross[TARGET].to_numpy() - cross[list(OFFS)].max(axis=1).to_numpy()
    rows = [evaluate_split(X, cross, gap, *scaffold_split(kept, 0.2, seed=s))
            for s in SEEDS]
    diff, direct, enrich = (np.array([r[i] for r in rows]) for i in range(3))
    return {"n": len(kept), "diff": (diff.mean(), diff.std()),
            "direct": (direct.mean(), direct.std()),
            "enrich": (np.nanmean(enrich), np.nanstd(enrich)),
            "base": float((gap >= 1.0).mean())}


def main() -> None:
    panel = DEFAULT_PANEL
    pooled = panel_data.build_cross_measured(panel)
    matched = panel_data.same_document_cross_measured(panel)

    shared = set(pooled["smi"]) & set(matched["smi"])
    print("=" * 78)
    print("P4.1 — pooled labels vs same-document labels")
    print("=" * 78)
    print(f"\n  pooled  n={len(pooled)}   matched n={len(matched)}   shared {len(shared)}")
    print("  The molecule sets barely differ, so what follows is a label-value")
    print("  comparison, not a sample-size comparison.\n")

    joined = pooled.merge(matched, on="smi", suffixes=("_pooled", "_matched"))
    for iso in panel.isoforms:
        delta = joined[f"{iso}_matched"] - joined[f"{iso}_pooled"]
        print(f"  {iso}: median label shift {delta.median():+.3f}, "
              f"|shift| > 0.5 log in {(delta.abs() > 0.5).mean():.1%} of molecules")

    print(f"\n  {'labels':>16} {'n':>6} {'Spearman':>18} {'direct':>16} "
          f"{'enrichment':>16} {'base':>7}")
    for label, cross in (("pooled", pooled), ("same-document", matched)):
        r = _seeds(cross)
        print(f"  {label:>16} {r['n']:>6} {r['diff'][0]:>11.3f} ± {r['diff'][1]:.3f} "
              f"{r['direct'][0]:>9.3f} ± {r['direct'][1]:.3f} "
              f"{r['enrich'][0]:>10.2f} ± {r['enrich'][1]:.2f} {r['base']:>6.1%}")

    print("\n  How to read this. A HIGHER matched Spearman means the pooled label was")
    print("  carrying cross-study noise the model could not fit. A LOWER one means the")
    print("  pooled median was averaging that noise away and matching threw data out.")
    print("  Neither outcome is a failure; the point is that open question 3 is closed.")


if __name__ == "__main__":
    main()
