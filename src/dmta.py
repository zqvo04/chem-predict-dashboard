"""One turn of the loop: select, ask, analyze.

The three steps live in one module because they change together — a new acquisition
signal needs a new analysis to justify it, and an analysis nobody selects on is a
report rather than a loop.

**The models are frozen for the whole campaign.** They are refit once on the train
fold of the sealed split and never touched again. STATE.md section 4b measured a
70-molecule round as moving JAK1 Spearman by +0.00008 against a seed-to-seed spread
of +-0.0072, so a loop that retrained each round would be reporting noise as
learning. What changes between rounds is acquisition, and the round record states
that `model_ids` are identical throughout rather than leaving it to be inferred.

**Every round has two arms.** The acquired batch and a random batch of the same size
from the same remaining pool, scored identically (G5, N9). A one-armed hit rate
cannot separate a real effect from a moved denominator, and the pool gets harder by
construction as the loop consumes its own best candidates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import labels
from .applicability import ADReference, build_reference, in_domain
from .data import panel_data
from .models.features import morgan_matrix
from .models.isoform_regressor import _fit
from .oracle import TimeSplitOracle
from .panels import PanelSpec
from .selectivity import SELECTIVE_GAP

# Nearest-neighbour similarity bands the falsification rate is reported over. The
# acquisition penalty is calibrated on exactly these bands, so they are defined once.
NN_BUCKETS = ((0.0, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 1.01))

# A prediction is falsified when it overstates the measurement by more than one log
# unit — the funnel's own "typical potency error" line (VALIDATION.md STEP 3), used
# here so the rate is comparable to the falsification audit's exceedance.
OVERSTATEMENT = 1.0
MIN_BUCKET = 5          # below this a per-bucket rate is noise, so it is not reported


@dataclass(frozen=True)
class Context:
    """The frozen scoring apparatus for one campaign."""
    models: dict
    ad_reference: ADReference
    model_ids: dict[str, str]


def build_context(panel: PanelSpec) -> Context:
    """Refit the regressors and the AD reference on the train fold only.

    The deployed models trained on the whole dataset, eval fold included, so scoring
    the loop through them would measure memorisation. This is the same refit the
    falsification audit's positive arm uses.
    """
    from .loop_contract import model_id

    split = panel_data.load_eval_split(panel)
    train_smiles = set(split.loc[split["fold"] == "train", "smi"])

    models, ids = {}, {}
    for iso in panel.isoforms:
        # `_fit` rather than `train_and_cache`: the latter also runs a five-seed
        # scaffold-split evaluation whose metrics nothing here reads, which is five
        # extra fits per isoform on ~8k molecules.
        data = panel_data.build_isoform_dataset(panel, iso)
        pre = data[data["smi"].isin(train_smiles)]
        X, mask = morgan_matrix(pre["smi"].tolist())
        models[iso] = _fit(X, pre["pchembl"].to_numpy()[mask])
        ids[iso] = model_id(panel.chembl_ids[iso], models[iso])

    return Context(models=models,
                   ad_reference=build_reference(sorted(train_smiles)),
                   model_ids=ids)


def score(context: Context, panel: PanelSpec, smiles: list[str]) -> pd.DataFrame:
    """Predicted pchembl per isoform, the gap, and the AD nearest-neighbour similarity."""
    X, mask = morgan_matrix(list(smiles))
    kept = [s for s, keep in zip(smiles, mask) if keep]
    frame = pd.DataFrame({"smi": kept})
    for iso in panel.isoforms:
        frame[f"pred_{iso}"] = context.models[iso].predict(X)
    frame["gap"] = (frame[f"pred_{panel.target}"]
                    - frame[[f"pred_{o}" for o in panel.offs]].max(axis=1))
    frame["nn_sim"] = in_domain(kept, reference=context.ad_reference)["nn_sim"]
    return frame


def _bucket(nn_sim: float) -> str:
    for low, high in NN_BUCKETS:
        if low <= nn_sim < high:
            return f"{low:.2f}-{high:.2f}"
    return "out-of-range"


def _ask(arm: pd.DataFrame, oracle: TimeSplitOracle, panel: PanelSpec) -> pd.DataFrame:
    """Ask the oracle for one arm; rows it will not answer are dropped."""
    rows = []
    for row in arm.itertuples():
        answer = oracle.answer(row.smi)
        if answer is None:
            continue
        rows.append({"smi": row.smi, "pred": getattr(row, f"pred_{panel.target}"),
                     "measured": answer[panel.target], "nn_sim": row.nn_sim,
                     "measured_gap": (answer[panel.target]
                                      - max(answer[o] for o in panel.offs)),
                     "answer": answer})
    return pd.DataFrame(rows)


def _metrics(answered: pd.DataFrame) -> dict:
    if answered.empty:
        return {"n": 0, "hit_rate": 0.0, "falsification_rate": 0.0,
                "median_error": float("nan")}
    error = answered["pred"] - answered["measured"]
    return {"n": int(len(answered)),
            "hit_rate": float((answered["measured_gap"] >= SELECTIVE_GAP).mean()),
            "falsification_rate": float((error > OVERSTATEMENT).mean()),
            "median_error": float(error.median())}


def run_round(panel: PanelSpec, context: Context, oracle: TimeSplitOracle,
              batch: int = 70, penalty: dict[str, float] | None = None,
              seed: int = 0, root=None) -> dict:
    """Select, ask, analyze — one round, both arms.

    `penalty` maps an NN-similarity bucket to a score deduction; it is what the
    previous round's analysis feeds back. None means the first round, which ranks
    on the gap alone.
    """
    pool = oracle.remaining()
    batch = min(batch, len(pool) // 2)
    scored = score(context, panel, pool)
    scored["penalty"] = [0.0 if penalty is None else penalty.get(_bucket(s), 0.0)
                         for s in scored["nn_sim"]]
    scored["acquisition"] = scored["gap"] - scored["penalty"]

    acquired = scored.sort_values("acquisition", ascending=False).head(batch)
    rest = scored[~scored["smi"].isin(set(acquired["smi"]))]
    control = rest.sample(n=min(batch, len(rest)), random_state=seed)

    acq = _ask(acquired, oracle, panel)
    ctl = _ask(control, oracle, panel)
    both = pd.concat([acq, ctl], ignore_index=True)

    for row in both.itertuples():
        for iso, value in row.answer.items():
            labels.append(panel, smi=row.smi, isoform=iso, value=float(value),
                          relation="=", source="TimeSplitOracle",
                          answer_status="answered", root=root)

    by_bucket = {}
    if not both.empty:
        bucket = pd.Series([_bucket(s) for s in both["nn_sim"]])
        overstated = (both["pred"] - both["measured"]) > OVERSTATEMENT
        by_bucket = {b: float(overstated[(bucket == b).to_numpy()].mean())
                     for b in sorted(set(bucket)) if (bucket == b).sum() >= MIN_BUCKET}

    acq_metrics, ctl_metrics = _metrics(acq), _metrics(ctl)
    # The base rate is the CONTROL arm's hit rate, not the two arms pooled. The
    # acquired arm is selected to be enriched, so pooling it in inflates the
    # denominator by exactly the effect being measured — which produced an
    # "enrichment ceiling" below the enrichment it was supposed to cap.
    base_rate = ctl_metrics["hit_rate"]
    return {
        "acquired": acq_metrics,
        "random": ctl_metrics,
        "advantage": acq_metrics["hit_rate"] - ctl_metrics["hit_rate"],
        "falsification_by_bucket": by_bucket,
        "base_rate": base_rate,
        # Raw enrichment is not comparable across rounds: the pool empties of
        # selective molecules as the loop consumes them, so the base rate falls and
        # the ceiling rises. Report the ratio to the ceiling, never the ratio alone.
        "enrichment": (acq_metrics["hit_rate"] / base_rate) if base_rate else float("nan"),
        "enrichment_ceiling": (1.0 / base_rate) if base_rate else float("nan"),
        "oracle_remaining": oracle.population(),
        "model_ids": dict(context.model_ids),
        "asked": sorted(both["smi"]) if len(both) else [],
    }


def penalty_from(round_results: list[dict], scale: float = 1.0) -> dict[str, float]:
    """Acquisition deduction per NN-similarity bucket, from measured falsification.

    This is the loop's feedback channel, and after Wave 1 it is the only defensible
    one. Data volume is saturated on both axes (STATE.md section 4b), so retraining
    cannot be it; the AD is the layer section 2 found earning its keep; and ROADMAP
    N13 asks for exactly this conversion — the applicability domain from a hard
    filter into an acquisition penalty, which keeps exploration from being
    structurally zero while still charging for unreliability.

    The deduction is in log units, directly comparable to the gap it is subtracted
    from: a bucket where every prediction overstated by more than a log costs a full
    log of gap, one where none did costs nothing.
    """
    totals: dict[str, list[float]] = {}
    for result in round_results:
        for bucket, rate in result.get("falsification_by_bucket", {}).items():
            totals.setdefault(bucket, []).append(float(rate))
    return {bucket: scale * float(np.mean(rates)) for bucket, rates in totals.items()}
