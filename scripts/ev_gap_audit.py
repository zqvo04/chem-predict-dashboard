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
SHORTLIST = 60          # funnel.screen_library's default shortlist size


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
    top_plain = set(moved.nlargest(SHORTLIST, "gap")["smi"])
    top_ev = set(moved.nlargest(SHORTLIST, "ev_gap")["smi"])
    print(f"\n  Top-{SHORTLIST} shortlist overlap between the two rankings: "
          f"{len(top_plain & top_ev)} / {SHORTLIST}")
    print("  That overlap is what a wired deployment would change about the shortlist.")

    dropped = moved[moved["smi"].isin(top_plain - top_ev)]
    if len(dropped):
        print(f"  Molecules the multiplication drops out of the top {SHORTLIST}: "
              f"median P {dropped['p'].median():.3f}, median gap {dropped['gap'].median():+.2f}")

    deployed_shortlist_effect(panel)


def deployed_shortlist_effect(panel) -> None:
    """배포 shortlist가 실제로 어떻게 바뀌는가 — 정답이 없으므로 서술만 한다.

    위의 Spearman은 gap이 **실측된** 유일한 모집단에서 잰 것이고, 그 모집단은 전부 세
    아이소폼에서 측정된 기지 활성이라 게이트가 P ~ 1을 준다. 곱셈이 일할 수 없는 곳이다.
    배포 shortlist는 와이드 라이브러리에서 뽑히고 거기서는 P가 포화되지 않는다 — 다만
    그 분자들에는 측정된 gap이 없으므로 어느 순서가 더 옳은지는 **채점할 수 없다.**

    그래서 여기서는 판정하지 않고 크기만 보고한다: 배선이 오늘의 shortlist를 얼마나
    움직이는가. 그것이 배선 결정이 알 수 있는 전부다.
    """
    from src import funnel
    from src.models.two_part import CENSORED_FLOOR

    print("\n" + "=" * 78)
    print("What wiring would change about TODAY's deployed shortlist (descriptive)")
    print("=" * 78)
    print("  No molecule here has a measured gap, so neither order can be scored.")

    picks = funnel.screen_library(panel)
    if picks.empty:
        print("  Empty shortlist; nothing to compare.")
        return

    p = picks["binder_prob"].to_numpy()
    ev = {iso: p * picks[f"pred_{iso}"].to_numpy() + (1.0 - p) * CENSORED_FLOOR
          for iso in panel.isoforms}
    ev_gap = ev[panel.target] - np.maximum.reduce([ev[o] for o in panel.offs])

    order_plain = picks.assign(s=picks["gap"]).sort_values("s", ascending=False)["smi"].tolist()
    order_ev = picks.assign(s=ev_gap).sort_values("s", ascending=False)["smi"].tolist()
    top = min(10, len(picks))
    print(f"\n  shortlist n = {len(picks)}   P(binder) median {np.median(p):.3f}, "
          f"min {p.min():.3f}")
    print(f"  rank correlation between the two orders: "
          f"{spearmanr(picks['gap'], ev_gap).statistic:.3f}")
    print(f"  top-{top} membership unchanged: "
          f"{len(set(order_plain[:top]) & set(order_ev[:top]))} / {top}")
    print(f"  molecule ranked #1 by gap: {'the same' if order_plain[0] == order_ev[0] else 'DIFFERENT'}")

    also_floor = ((ev[panel.target] >= 6.0).sum(), (picks[f"pred_{panel.target}"] >= 6.0).sum())
    print(f"  would still clear a 6.0 potency floor: {also_floor[0]} of {len(picks)} "
          f"under EV, {also_floor[1]} under the plain prediction")


if __name__ == "__main__":
    main()
