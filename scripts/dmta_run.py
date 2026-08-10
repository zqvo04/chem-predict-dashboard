#!/usr/bin/env python3
"""Drive the DMTA loop: N rounds, resumable, both arms recorded every round.

The round history in `rounds.jsonl` is the state. Restarting replays it to rebuild
the acquisition penalty and the set of molecules already asked, then continues — so
stopping after round 2 and re-running produces the same round 3 (G4).

Stopping rules, both recorded rather than silent:
  * the oracle is exhausted (G3), or
  * the acquired arm has failed to beat the random arm for STALL_ROUNDS rounds.

The second one is why the random arm exists. Without it "the hit rate went down"
cannot be told apart from "the remaining pool got harder", which it does by
construction as the loop consumes its own best candidates.

    python scripts/dmta_run.py [panel] [campaign_id] [n_rounds]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import dmta, registry                                      # noqa: E402
from src.oracle import TimeSplitOracle                              # noqa: E402
from src.panels import DEFAULT_PANEL, PanelSpec, get_panel          # noqa: E402

STALL_ROUNDS = 2
BATCH = 70


def run(panel: PanelSpec, campaign_id: str, n_rounds: int = 3, batch: int = BATCH,
        root: Path | None = None, context: dmta.Context | None = None) -> list[dict]:
    """Run until `n_rounds`, the oracle empties, or acquisition stalls.

    `context` is injectable only so a caller running several campaigns back to back
    does not refit the same frozen models each time; omitted, it is built here.
    """
    context = context or dmta.build_context(panel)
    oracle = TimeSplitOracle(panel)

    # Only this loop's own rounds. `registry.rounds` returns every kind, and app.py
    # writes `screen` rounds into whatever campaign the dashboard is on; one of
    # those landing in a DMTA campaign would shift the round numbering, consume a
    # round, and feed a foreign metrics dict to the penalty. Today the ids differ
    # (`jak-default` vs `jak-autoimmune`) so it is latent — but `campaign_id` is a
    # CLI argument, so it is one typo from live.
    recorded = [r for r in registry.rounds(campaign_id, root) if r.kind == "select"]
    history = [r.metrics for r in recorded]
    # Replay: re-asking the recorded molecules advances the oracle to the state the
    # recorded rounds left it in. Their answers are already in the ledger.
    for record in recorded:
        for smi in record.metrics.get("asked", []):
            oracle.answer(smi)

    for index in range(len(history), n_rounds):
        if oracle.exhausted:
            print(f"  round {index}: oracle exhausted, stopping (G3)")
            break
        result = dmta.run_round(panel, context, oracle, batch=batch,
                                penalty=dmta.penalty_from(history), seed=index,
                                root=root)
        history.append(result)
        registry.append_round(campaign_id, kind="select",
                              model_ids=result["model_ids"],
                              n_molecules=result["acquired"]["n"] + result["random"]["n"],
                              metrics=result, root=root)
        print(f"  round {index}: acquired hit {result['acquired']['hit_rate']:.1%} "
              f"vs random {result['random']['hit_rate']:.1%} "
              f"(advantage {result['advantage']:+.1%}), "
              f"falsified {result['acquired']['falsification_rate']:.1%}, "
              f"{result['oracle_remaining']} left")

        recent = history[-STALL_ROUNDS:]
        if len(recent) == STALL_ROUNDS and all(r["advantage"] <= 0 for r in recent):
            print(f"  stopping: acquisition has not beaten random for "
                  f"{STALL_ROUNDS} rounds")
            break

    return history


def main() -> None:
    panel = get_panel(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PANEL
    campaign_id = sys.argv[2] if len(sys.argv) > 2 else "jak-autoimmune"
    n_rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    print(f"DMTA loop — panel {panel.name}, campaign {campaign_id}, {n_rounds} rounds")
    run(panel, campaign_id, n_rounds=n_rounds)


if __name__ == "__main__":
    main()
