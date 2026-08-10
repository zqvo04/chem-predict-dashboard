# Wave 4 — 라벨 정합과 2부 모델 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ROADMAP 3단계의 남은 CPU 항목 넷(P4.1 · §12 Q2 · Q3 · A2/Q1)을 측정으로 닫는다 — potency 축이 **표현을 바꾸지 않고** 얼마나 나아지는지에 대한 답을, 채택이든 반증이든 기록으로 남긴다.

**Architecture:** 세 개의 작은 데이터 함수와 한 개의 모델 래퍼를 추가한다. (1) `panel_data`가 커밋된 증거 저장소에서 **문헌 내 매칭 라벨**과 **assay 종류/연도**를 유도한다 — 커밋된 JAK parquet은 이 컬럼들이 없고 재빌드는 G0 사건이므로. (2) `binder_gate`가 지금 스크립트 안에만 있는 게이트 재적합 헬퍼 넷을 갖는다. (3) `two_part.TwoPart`가 게이트와 회귀기를 `EV = P·reg + (1−P)·floor`로 합친다. 배포 경로는 **한 줄도 건드리지 않는다** — 이 웨이브는 측정이지 배포가 아니다.

**Tech Stack:** Python 3.11 · pandas · numpy · scikit-learn(HistGradientBoosting) · RDKit · pytest

## Global Constraints

- 브랜치는 `claude/bioinformatics-computational-chemistry-wje9cs`. **PR은 명시 요청 없이 만들지 않는다.**
- **G0** — 배포 자산 불변. `assets/models/**`, `assets/jak/JAK{1,2,3}.parquet`, `assets/library/library.parquet`에 바이트 변화가 없어야 한다. 검증: `git diff --stat main..HEAD -- assets/models assets/jak/JAK1.parquet assets/jak/JAK2.parquet assets/jak/JAK3.parquet assets/library`가 빈 출력.
- **G6** — 봉인 414(`assets/jak/sealed_negatives.parquet`)는 어떤 학습·튜닝에도 들어가지 않는다. 이 웨이브에서는 **채점만** 한다.
- **G8** — 상수는 봉인 세트를 보기 전에 커밋된다. `CENSORED_FLOOR`는 Task 3에서 커밋되고 Task 4에서 처음 채점된다. 두 커밋의 순서를 바꾸지 않는다.
- **G9** — 평가 분할은 `assets/jak/eval_split.parquet`에 봉인된 것을 그대로 쓴다. 새로 계산하지 않는다.
- **G5** — 개선과 미개선을 동일 비중으로 보고한다. 반증도 결과다.
- **§8 튜닝 봉쇄** — 하이퍼파라미터는 건드리지 않는다. `max_iter=400, learning_rate=0.08, random_state=0` 그대로.
- 시간 컷은 `panel_data.EVAL_TIME_CUT` (= 2020) 하나만 쓴다. 상수를 스크립트에 다시 적지 않는다.
- 모든 새 코드는 오프라인에서 돌아야 한다 — 커밋된 `assets/` 만 읽는다. 네트워크 호출 금지.
- 전체 스위트(`python -m pytest tests/ -q`)가 매 커밋에서 통과. 현재 218 통과가 기준선.
- **런타임 추정치는 낙관적이다.** 실측: 전체 스위트 ~5분, `matched_label_audit.py` ~13분.
  스크립트 docstring의 분 단위 추정을 믿지 말고 타임아웃을 넉넉히(600000 ms) 잡는다.

---

## File Structure

| 파일 | 책임 | 태스크 |
|---|---|---|
| `src/data/panel_data.py` (수정) | `same_document_cross_measured` — 문헌 내 매칭 교차측정 세트. `evidence_isoform_frame` — 증거 저장소에서 유도한 `(smi, pchembl_kikd, year_first)` | 1, 2 |
| `tests/test_matched_labels.py` (신규) | 위 두 함수의 단위 테스트. 합성 증거 저장소를 `tmp_path`에 짓는다 | 1, 2 |
| `scripts/matched_label_audit.py` (신규) | pooled vs 문헌 내 매칭 라벨의 gap Spearman 비교 (P4.1 / Q3) | 1 |
| `scripts/assay_time_audit.py` (수정) | 없는 컬럼을 증거 저장소에서 채우는 폴백 — 오프라인 복구 (Q2) | 2 |
| `src/models/binder_gate.py` (수정) | `build_gate` · `proba_aligned` · `youden_threshold` · `at_matched_recall` — 지금 `scripts/gate_ab_audit.py` 안에만 있는 재사용 부품 승격 | 3 |
| `scripts/gate_ab_audit.py` (수정) | 위 넷을 import로 대체. 출력은 한 글자도 바뀌지 않아야 한다 | 3 |
| `src/models/two_part.py` (신규) | `TwoPart` — `EV = P·reg + (1−P)·floor`, 그리고 사전등록 상수 | 3 |
| `tests/test_two_part.py` (신규) | EV 산술, floor 상쇄, 파싱 실패 정렬 | 3 |
| `scripts/two_part_audit.py` (신규) | 두 팔 + matched-recall + AUC 감사 (A2 / Q1) | 4 |
| `STATE.md` · `VALIDATION.md` · `scripts/reproduce.sh` (수정) | 판정 기록 | 4 |

---

### Task 1: 문헌 내 매칭 라벨 (P4.1 / ROADMAP §12 Q3)

**배경 — 착수 전에 읽을 것.** 현재 `panel_data.build_cross_measured`는 아이소폼마다 **모든
문헌에 걸친 pchembl 중앙값**을 취한 뒤 세 컬럼을 inner join한다. 그래서 한 분자의 gap이
서로 다른 논문, 서로 다른 ATP 농도에서 나온 값의 차이일 수 있다. P4.1은 그 gap을 **한
논문 안에서** 만든다.

착수 전에 커밋된 증거 저장소로 세 가지를 이미 확인했다:

- 3개 아이소폼을 모두 담은 (분자, 문헌) 쌍 **5,049개**, 분자 **3,614개**.
- 3개 아이소폼을 모두 담은 (분자, **assay**) 쌍 **0개**. ChEMBL의 assay는 타깃 하나에
  속하므로 아이소폼 간 same-assay 매칭은 **구조적으로 불가능**하다. Q3의 "same-assay"는
  assay **종류** 매칭(Ki/Kd only)으로만 존재하고, 그것은 Task 2가 다룬다.
- 기존 `cross_measured` 3,624개 중 **3,605개**가 이미 same-document 분자다. 즉 이
  태스크는 **분자 집합을 거의 안 바꾸고 라벨 값만 바꾼다.** 그것이 이 실험을 깨끗하게
  만든다 — 표본 크기 통제가 필요 없다.

**Files:**
- Modify: `src/data/panel_data.py` (`CENSORED_RELATIONS` 정의 아래에 추가)
- Create: `tests/test_matched_labels.py`
- Create: `scripts/matched_label_audit.py`

**Interfaces:**
- Consumes: `PanelSpec.root`, `PanelSpec.chembl_ids`, `PanelSpec.isoforms` (기존)
- Produces: `panel_data.same_document_cross_measured(panel: PanelSpec) -> pd.DataFrame`,
  컬럼 `["smi", *panel.isoforms]` — `build_cross_measured`와 **동일한 모양**이므로
  `src.selectivity.evaluate_split`이 둘 다 그대로 받는다.

---

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_matched_labels.py`를 만든다. 실제 증거 저장소가 아니라 `tmp_path`에 지은 합성
저장소로 테스트한다 — 커밋된 저장소는 자라기 때문에 그 위의 숫자를 고정하면 언젠가
데이터가 바뀌었을 때 코드가 아닌 데이터가 테스트를 깬다.

```python
"""문헌 내 매칭 라벨 + 증거 저장소 유도 컬럼 (ROADMAP P4.1 / open questions 2, 3).

두 함수 모두 커밋된 증거 저장소를 읽는다. 저장소는 자라므로 그 위의 절대 숫자를 고정하면
데이터가 바뀔 때 코드가 아닌 데이터가 테스트를 깬다. 그래서 여기서는 tmp_path에 합성
저장소를 짓고 규칙 자체만 고정한다.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from src.data import panel_data
from src.panels import DEFAULT_PANEL


def _store(tmp_path, activity_rows):
    """합성 증거 저장소를 지은 panel을 돌려준다."""
    root = tmp_path / "assets" / "evidence"
    root.mkdir(parents=True)
    act = pd.DataFrame(activity_rows)
    act.to_parquet(root / "activity.parquet", index=False)
    keys = sorted(act["inchikey"].unique())
    pd.DataFrame({"inchikey": keys,
                  "parent_smiles": [f"C{'C' * i}O" for i, _ in enumerate(keys)]}
                 ).to_parquet(root / "molecule.parquet", index=False)
    return replace(DEFAULT_PANEL, root=tmp_path)


def _row(key, doc, iso, value, stype="IC50", year=2015):
    return {"inchikey": key, "document_chembl_id": doc,
            "target_chembl_id": DEFAULT_PANEL.chembl_ids[iso],
            "standard_type": stype, "standard_relation": "=",
            "pchembl_value": value, "document_year": year}


def test_same_document_keeps_only_fully_covered_molecules(tmp_path):
    t, o1, o2 = DEFAULT_PANEL.isoforms
    panel = _store(tmp_path, [
        # A: D1 covers all three (and has the most rows) -> chosen
        _row("A", "CHEMBL_D1", t, 8.0), _row("A", "CHEMBL_D1", t, 9.0),
        _row("A", "CHEMBL_D1", o1, 7.0), _row("A", "CHEMBL_D1", o2, 6.0),
        _row("A", "CHEMBL_D2", t, 5.0), _row("A", "CHEMBL_D2", o1, 5.0),
        _row("A", "CHEMBL_D2", o2, 5.0),
        # C: no single document covers all three -> excluded
        _row("C", "CHEMBL_D4", t, 6.0), _row("C", "CHEMBL_D4", o1, 6.0),
    ])
    out = panel_data.same_document_cross_measured(panel)

    assert list(out.columns) == ["smi", *DEFAULT_PANEL.isoforms]
    assert len(out) == 1, "only A is covered by a single document"
    # the chosen document's WITHIN-document median, not the pooled median
    assert out[t].iloc[0] == pytest.approx(8.5)
    assert out[o1].iloc[0] == pytest.approx(7.0)


def test_document_tie_breaks_on_the_smallest_id(tmp_path):
    t, o1, o2 = DEFAULT_PANEL.isoforms
    panel = _store(tmp_path, [
        _row("D", "CHEMBL_D6", t, 9.0), _row("D", "CHEMBL_D6", o1, 9.0),
        _row("D", "CHEMBL_D6", o2, 9.0),
        _row("D", "CHEMBL_D5", t, 3.0), _row("D", "CHEMBL_D5", o1, 3.0),
        _row("D", "CHEMBL_D5", o2, 3.0),
    ])
    out = panel_data.same_document_cross_measured(panel)

    assert len(out) == 1
    assert out[t].iloc[0] == pytest.approx(3.0), "CHEMBL_D5 < CHEMBL_D6"
```

- [ ] **Step 2: 실패를 확인한다**

Run: `python -m pytest tests/test_matched_labels.py -q`
Expected: FAIL — `AttributeError: module 'src.data.panel_data' has no attribute 'same_document_cross_measured'`

- [ ] **Step 3: 최소 구현**

`src/data/panel_data.py`의 `censored_library_molecules` 정의 **바로 앞**에 추가한다
(같은 증거-저장소 계열 함수끼리 모아 둔다):

```python
def same_document_cross_measured(panel: PanelSpec) -> pd.DataFrame:
    """교차측정 세트를 한 논문 안에서 만든 것 (ROADMAP P4.1).

    `build_cross_measured`는 아이소폼마다 모든 문헌에 걸친 중앙값을 취하므로, 한 분자의
    선택성 gap이 서로 다른 논문·서로 다른 ATP 농도에서 나온 두 값의 차이일 수 있다.
    이 함수는 분자마다 **패널 레코드를 가장 많이 담은 단일 문헌** — 동수면 문헌 id가 작은
    쪽, 그래서 결정적 — 을 고르고 그 문헌 안의 아이소폼별 중앙값만 쓴다.

    컬럼은 `build_cross_measured`와 같은 `smi + 아이소폼별 한 컬럼`이라, 두 프레임 모두
    `selectivity.evaluate_split`에 그대로 들어간다. 그것이 비교를 가능하게 하는 조건이다.
    """
    root = panel.root / "assets" / "evidence"
    act = pd.read_parquet(root / "activity.parquet")
    molecule = pd.read_parquet(root / "molecule.parquet")[["inchikey", "parent_smiles"]]

    by_target = {cid: iso for iso, cid in panel.chembl_ids.items()}
    rows = act[act["target_chembl_id"].isin(by_target)
               & act["pchembl_value"].notna()].copy()
    rows["isoform"] = rows["target_chembl_id"].map(by_target)

    grouped = rows.groupby(["inchikey", "document_chembl_id"])
    per_doc = pd.DataFrame({"n_iso": grouped["isoform"].nunique(),
                            "n_rec": grouped.size()}).reset_index()
    covered = per_doc[per_doc["n_iso"] == len(panel.isoforms)]
    if covered.empty:
        return pd.DataFrame(columns=["smi", *panel.isoforms])

    chosen = (covered.sort_values(["inchikey", "n_rec", "document_chembl_id"],
                                 ascending=[True, False, True])
                     .drop_duplicates("inchikey")[["inchikey", "document_chembl_id"]])

    picked = rows.merge(chosen, on=["inchikey", "document_chembl_id"])
    wide = (picked.groupby(["inchikey", "isoform"])["pchembl_value"]
                  .median().unstack("isoform"))
    out = (wide.join(molecule.drop_duplicates("inchikey").set_index("inchikey"))
               .rename(columns={"parent_smiles": "smi"})
               .dropna()
               .reset_index(drop=True))
    return out[["smi", *panel.isoforms]].drop_duplicates("smi").reset_index(drop=True)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_matched_labels.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/data/panel_data.py tests/test_matched_labels.py
git commit -m "Build the cross-measured set from one document per molecule"
```

- [ ] **Step 6: 감사 스크립트를 쓴다**

`scripts/matched_label_audit.py`:

```python
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
```

- [ ] **Step 7: 감사를 돌린다**

Run: `python scripts/matched_label_audit.py`
Expected: 두 줄의 지표 표가 출력된다. pooled Spearman은 VALIDATION.md의 Gate 4 값
(0.798 근방)과 일치해야 한다 — 일치하지 않으면 `_seeds`가 Gate 4와 다른 것을 재고 있는
것이므로 진행하지 말고 원인을 찾는다.

- [ ] **Step 8: 커밋**

```bash
git add scripts/matched_label_audit.py
git commit -m "Measure what the gap looks like when both values come from one paper"
```

---

### Task 2: `assay_time_audit.py` 오프라인 복구 (ROADMAP §12 Q2 + Q3의 assay 절반)

**배경.** `scripts/assay_time_audit.py`는 이미 2015~2020 시간 컷 표와 Ki/Kd 부분집합
비교를 출력하도록 쓰여 있다 — Q2가 요구하는 "오라클 크기 ↔ 모델 강도 트레이드"가 그
표의 `train`/`test` 컬럼이다. **못 도는 이유는 분석이 없어서가 아니라 데이터가 없어서다:**
`_cross`가 `build_isoform_dataset`에서 `pchembl_kikd`와 `year_first`를 읽는데, 커밋된
JAK parquet은 `(smi, pchembl, n_meas)`뿐이다. PI3K parquet에는 그 컬럼들이 있다 —
JAK 번들이 더 오래된 스키마다.

**JAK parquet을 재빌드하지 않는다** — 네트워크가 필요하고 배포 자산을 움직인다(G0).
커밋된 증거 저장소가 같은 provenance를 오프라인으로 갖고 있으므로 거기서 유도한다.
`panel_data.year_first`가 이미 정확히 이 선례를 만들어 뒀다.

착수 전 확인: 증거 저장소에서 유도한 Ki/Kd 3-아이소폼 완전 교차는 **386분자**로,
스크립트의 `MIN_TEST * 5 = 300` 문턱을 넘는다. 즉 Ki/Kd 팔이 "too small"로 죽지 않는다.

**Files:**
- Modify: `src/data/panel_data.py` (`same_document_cross_measured` 바로 뒤)
- Modify: `scripts/assay_time_audit.py` (`_cross` 위)
- Modify: `tests/test_matched_labels.py` (테스트 추가)

**Interfaces:**
- Consumes: Task 1의 합성 저장소 픽스처 `_store` / `_row`
- Produces: `panel_data.evidence_isoform_frame(panel: PanelSpec, isoform: str) -> pd.DataFrame`,
  컬럼 `["smi", "pchembl_kikd", "year_first"]`

---

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_matched_labels.py` 끝에 추가한다:

```python
def test_evidence_isoform_frame_derives_kikd_and_year(tmp_path):
    t, o1, _ = DEFAULT_PANEL.isoforms
    panel = _store(tmp_path, [
        _row("A", "CHEMBL_D1", t, 8.0, stype="IC50", year=2012),
        _row("A", "CHEMBL_D2", t, 7.0, stype="Ki", year=2019),
        _row("A", "CHEMBL_D3", t, 9.0, stype="Kd", year=2020),
        _row("B", "CHEMBL_D1", t, 5.0, stype="IC50", year=2011),   # no Ki/Kd
        _row("A", "CHEMBL_D1", o1, 4.0, stype="Ki", year=2012),    # other isoform
    ])
    out = panel_data.evidence_isoform_frame(panel, t)

    assert list(out.columns) == ["smi", "pchembl_kikd", "year_first"]
    row_a = out[out["pchembl_kikd"].notna()]
    assert len(row_a) == 1, "only A has Ki/Kd on this isoform"
    assert row_a["pchembl_kikd"].iloc[0] == pytest.approx(8.0), "median of Ki 7.0 and Kd 9.0"
    assert row_a["year_first"].iloc[0] == 2012, "earliest year over ALL types, not Ki/Kd only"
```

- [ ] **Step 2: 실패를 확인한다**

Run: `python -m pytest tests/test_matched_labels.py::test_evidence_isoform_frame_derives_kikd_and_year -q`
Expected: FAIL — `AttributeError: ... has no attribute 'evidence_isoform_frame'`

- [ ] **Step 3: 최소 구현**

`src/data/panel_data.py`의 `same_document_cross_measured` 뒤에 함수를 추가한다.

**상수를 새로 만들지 않는다** — `EQUILIBRIUM_TYPES = ("Ki", "Kd")`가 같은 모듈 65번째
줄에 이미 있고 `_collapse`가 `pchembl_kikd`를 만들 때 쓰고 있다. 같은 값을 두 이름으로
두면 언젠가 한쪽만 바뀐다.

```python
def evidence_isoform_frame(panel: PanelSpec, isoform: str) -> pd.DataFrame:
    """증거 저장소에서 유도한 한 아이소폼의 (smi, pchembl_kikd, year_first).

    `build_isoform_dataset`은 네트워크 재빌드 시 이 두 컬럼을 만들지만, 커밋된 JAK 번들은
    그 이전 스키마라 `(smi, pchembl, n_meas)`뿐이다. 배포 자산 재빌드는 G0 사건이므로,
    assay 종류나 출판 연도가 필요한 감사는 같은 provenance를 이쪽에서 읽는다 —
    `year_first`가 이미 만든 선례와 같다.

    `year_first`는 **모든** assay 종류에 걸친 최초 연도다. 분자가 문헌에 등장한 시점이
    질문이지, Ki로 측정된 시점이 아니다.
    """
    root = panel.root / "assets" / "evidence"
    act = pd.read_parquet(root / "activity.parquet")
    molecule = pd.read_parquet(root / "molecule.parquet")[["inchikey", "parent_smiles"]]

    rows = act[(act["target_chembl_id"] == panel.chembl_ids[isoform])
               & act["pchembl_value"].notna()]
    kikd = (rows[rows["standard_type"].isin(EQUILIBRIUM_TYPES)]
            .groupby("inchikey")["pchembl_value"].median().rename("pchembl_kikd"))
    year = rows.groupby("inchikey")["document_year"].min().rename("year_first")

    out = (pd.concat([year, kikd], axis=1)
             .join(molecule.drop_duplicates("inchikey").set_index("inchikey")))
    return (out.dropna(subset=["parent_smiles", "year_first"])
               .rename(columns={"parent_smiles": "smi"})
               .reset_index(drop=True)[["smi", "pchembl_kikd", "year_first"]])
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_matched_labels.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: 감사 스크립트에 폴백을 단다**

`scripts/assay_time_audit.py`의 `_cross` 정의 **바로 앞**에 넣고, `_cross` 본문의
`panel_data.build_isoform_dataset(DEFAULT_PANEL, iso).set_index("smi")`를
`_isoform_frame(iso)`로 바꾼다:

```python
def _isoform_frame(iso: str):
    """한 아이소폼의 smi-인덱스 프레임, `pchembl_kikd`와 `year_first`를 보장한다.

    커밋된 JAK 번들은 `(smi, pchembl, n_meas)`뿐이라 이 감사가 오프라인에서 죽었다.
    재빌드는 네트워크를 요구하고 배포 자산을 움직이므로(G0), 없는 두 컬럼만 커밋된 증거
    저장소에서 채운다. 컬럼이 이미 있는 패널(PI3K)에서는 아무것도 하지 않는다.
    """
    d = panel_data.build_isoform_dataset(DEFAULT_PANEL, iso).set_index("smi")
    if {"pchembl_kikd", "year_first"} <= set(d.columns):
        return d
    derived = panel_data.evidence_isoform_frame(DEFAULT_PANEL, iso).set_index("smi")
    return d.join(derived[["pchembl_kikd", "year_first"]])
```

`_cross` 본문은 이렇게 된다:

```python
def _cross(column: str) -> pd.DataFrame:
    """Cross-measured frame built from one pchembl column, carrying year_first."""
    frames, years = [], []
    for iso in DEFAULT_PANEL.isoforms:
        d = _isoform_frame(iso)
        frames.append(d[column].rename(iso).dropna())
        years.append(d["year_first"].rename(iso))
    cross = pd.concat(frames, axis=1, join="inner")
    # The molecule's arrival date is the earliest year any isoform measured it.
    cross["year_first"] = pd.concat(years, axis=1).min(axis=1).reindex(cross.index)
    return cross.reset_index()
```

- [ ] **Step 6: 감사를 돌린다**

Run: `python scripts/assay_time_audit.py`
Expected: 두 감사가 모두 완주한다. AUDIT 1이 Ki/Kd n = 384를 보고하고 "too small"로 죽지
않는다(문턱은 300). AUDIT 2의 표가 2015~2020 여섯 줄을 채우고, train/test 수가
2016 → 1,227/2,385 · 2018 → 1,720/1,892 · 2020 → 2,534/**1,078**과 일치한다.

**1,078이 가장 중요한 검증이다** — `TimeSplitOracle`의 모집단과 같은 수이고, 오라클도
`year > 2020`을 eval로 쓴다. 두 경로가 서로 모르는 채 같은 수에 도달하면 유도 컬럼이
행을 잃지 않은 것이다. 어긋나면 `_isoform_frame` join이 SMILES 표준화 차이로 행을 잃고
있다는 뜻이고, 그건 데이터 결함이지 감사 결과가 아니다 — 진행하지 않는다.

- [ ] **Step 7: 커밋**

```bash
git add src/data/panel_data.py scripts/assay_time_audit.py tests/test_matched_labels.py
git commit -m "Let the assay and time audits run offline from the evidence store"
```

---

### Task 3: 게이트 헬퍼 승격 + 2부 모델 (A2 구조, 상수 사전등록)

**배경 — 무엇을 만드는가.** 배포된 Tier 1은 회귀기 출력만으로 랭킹한다. 회귀기는
정량화된 pchembl로만 학습했으므로, 패널에 붙은 적 없는 분자에는 학습 평균(JAK ≈ 6.3)을
돌려주고 그 분자는 potency floor를 통과한다. STATE §2가 봉인 414에서 **93.0 %**의
예측이 그 분자 자신의 측정 상한을 넘는다고 측정한 게 이것이다.

2부 모델은 퍼널이 원하는 값이 "붙는다면 얼마나 세게 붙나"가 아니라 "얼마나 세게 붙을
것인가"이고, 둘의 차이는 정확히 **붙을 확률**이라고 말한다:

```
EV(potency) = P(binder) · reg(molecule) + (1 − P(binder)) · FLOOR
```

**먼저 알아 둘 대수(代數) 하나.** 게이트는 패널당 하나이므로 `P`는 아이소폼마다 같다.
따라서

```
EV(target) − max_off EV(off) = P · (reg(target) − max_off reg(off)) = P · gap
```

**floor가 차분에서 상쇄된다.** 즉 2부 모델은 결합자끼리의 순위를 바꾸지 않고, 붙을 것
같지 않은 분자의 gap을 0쪽으로 수축시킨다. 그것이 의도한 전부이며, 미리 적어 두는 이유는
Task 4가 그것을 발견으로 보고하지 않게 하기 위해서다.

**FLOOR는 사전등록 상수다.** 봉인 414에서 고르지 않는다(G6). 5.0은 pchembl 10 µM —
키나아제 패널이 보통 멈추는 농도, 즉 assay가 아무것도 못 봤을 때의 potency다. 4.0(100 µM)은
민감도 점검용이고, 둘 다 보고할 뿐 **탐색하지 않는다.** 이 상수는 이 태스크의 커밋에
들어가고 Task 4에서 처음 채점된다 — G8을 상수에 적용한 것이다.

**헬퍼 승격이 먼저인 이유.** `scripts/gate_ab_audit.py` 안에 게이트를 명시 클래스로
재적합하는 부품 넷이 있고 Task 4가 그것을 전부 필요로 한다. 스크립트 간 import는 pytest
아래에서 깨진다 — 이 저장소가 `censored_library_molecules`에서 이미 겪고 같은 방식으로
고쳤다. import shim이 아니라 모듈로 옮긴다.

**Files:**
- Modify: `src/models/binder_gate.py`
- Modify: `scripts/gate_ab_audit.py`
- Create: `src/models/two_part.py`
- Create: `tests/test_two_part.py`

**Interfaces:**
- Produces:
  - `binder_gate.build_gate(positives: list[str], negatives: list[str]) -> HistGradientBoostingClassifier`
  - `binder_gate.proba_aligned(model, smiles: list[str]) -> np.ndarray` — 입력과 정렬, 파싱 실패는 NaN
  - `binder_gate.youden_threshold(model, positives: list[str], negatives: list[str]) -> float`
  - `binder_gate.at_matched_recall(scores_neg: np.ndarray, scores_pos: np.ndarray, recall: float) -> tuple[float, float]`
  - `two_part.CENSORED_FLOOR: float` = 5.0, `two_part.FLOOR_SENSITIVITY: float` = 4.0
  - `two_part.TwoPart(gate, regressors: dict[str, object], floor: float)` with
    `.p_binder(smiles) -> np.ndarray`, `.expected_potency(smiles) -> dict[str, np.ndarray]`,
    `.expected_gap(smiles, panel) -> np.ndarray` — 셋 다 입력과 정렬, 파싱 실패는 NaN

---

- [ ] **Step 1: 헬퍼 넷을 `binder_gate.py`로 옮긴다**

`src/models/binder_gate.py`의 `_fit` 정의 **바로 뒤**에 붙인다. 본문은
`scripts/gate_ab_audit.py`의 것을 그대로 가져오되, `probabilities`는 정렬을 유지하는
`proba_aligned`로 바꾼다 — 2부 모델은 NaN을 버릴 수 없다.

```python
def build_gate(positives: list[str], negatives: list[str]) -> HistGradientBoostingClassifier:
    """명시된 클래스 멤버로 게이트를 적합한다. 캐시를 읽지도 쓰지도 않는다.

    배포 게이트는 캐시된 파일이고 그 학습 집합은 고정돼 있다. 시간 컷 위에서 재적합한
    게이트가 필요한 감사 — A/B, 2부 모델 — 는 여기를 쓴다.
    """
    smiles = list(positives) + list(negatives)
    y = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    X, mask = morgan_matrix(smiles)
    return _fit(X, y[mask])


def proba_aligned(model, smiles: list[str]) -> np.ndarray:
    """P(binder), 입력과 정렬. 파싱 실패한 SMILES는 NaN.

    `BinderGate.predict_proba`와 같은 계약이되 캐시된 번들이 아니라 날 모델을 받는다.
    """
    smiles = list(smiles)
    X, mask = morgan_matrix(smiles)
    out = np.full(len(smiles), np.nan)
    if X.shape[0]:
        out[mask] = model.predict_proba(X)[:, 1]
    return out


def youden_threshold(model, positives: list[str], negatives: list[str]) -> float:
    """홀드아웃 양성 대 홀드아웃 추정 음성에서의 Youden's J.

    측정 음성 위에서는 일부러 계산하지 않는다 — 판정 대상 모집단에서 조정한 임계는
    측정이 아니다.
    """
    p_pos = proba_aligned(model, positives)
    p_neg = proba_aligned(model, negatives)
    p_pos, p_neg = p_pos[~np.isnan(p_pos)], p_neg[~np.isnan(p_neg)]
    y = np.concatenate([np.ones(len(p_pos)), np.zeros(len(p_neg))])
    fpr, tpr, thresholds = roc_curve(y, np.concatenate([p_pos, p_neg]))
    return float(thresholds[np.argmax(tpr - fpr)])


def at_matched_recall(scores_neg: np.ndarray, scores_pos: np.ndarray,
                      recall: float) -> tuple[float, float]:
    """`recall`만큼의 양성을 남기는 (임계, 음성 통과율).

    각자의 운영점에서 비교한 두 점수는 같은 곡선 위의 다른 위치일 뿐일 수 있다. 양성
    재현율을 먼저 맞추면 그 자유도가 사라진다. 점수의 종류를 묻지 않으므로 게이트 확률에도
    potency 예측에도 같이 쓴다.
    """
    threshold = float(np.quantile(scores_pos, 1.0 - recall))
    return threshold, float((scores_neg >= threshold).mean())
```

- [ ] **Step 2: `gate_ab_audit.py`가 import로 쓰게 한다**

import 줄을 바꾸고,

```python
from src.models.binder_gate import (at_matched_recall, build_gate,   # noqa: E402
                                    proba_aligned, youden_threshold)
```

로컬 `build_gate` · `youden_threshold` · `at_matched_recall` 정의 세 개를 지운다.
`probabilities`는 얇은 래퍼로 남긴다 — 이 스크립트의 모든 호출부가 NaN 없는 배열을
기대하기 때문이다:

```python
def probabilities(model, smiles: list[str]) -> np.ndarray:
    """P(binder), 파싱 실패는 제외. 모집단 통계용이므로 정렬이 필요 없다."""
    p = proba_aligned(model, smiles)
    return p[~np.isnan(p)]
```

`at_matched_recall`의 인자 순서가 `(scores_neg, scores_pos, recall)`로 옮긴 뒤에도
같은지 확인한다 — 원래 시그니처가 `(p_meas, p_pos, recall)`이므로 호출부는 바뀌지 않는다.
`from sklearn.metrics import roc_curve` import가 스크립트에서 더 이상 쓰이지 않으면
지운다(내가 만든 고아만 치운다).

- [ ] **Step 3: 승격이 출력을 바꾸지 않았는지 확인한다**

Run: `python scripts/gate_ab_audit.py`
Expected: STATE §2b의 표와 같은 숫자 — BASELINE ROC-AUC 0.741 / 봉인 414 0.925,
CANDIDATE 0.735 / 0.773. **한 자리라도 다르면 승격이 동작을 바꾼 것이므로 되돌린다.**

- [ ] **Step 4: 커밋**

```bash
git add src/models/binder_gate.py scripts/gate_ab_audit.py
git commit -m "Promote the gate refit helpers out of the audit script"
```

- [ ] **Step 5: 2부 모델의 실패하는 테스트를 쓴다**

`tests/test_two_part.py`:

```python
"""A2: EV = P(binder) * regressor + (1 - P) * floor.

모델을 학습시키지 않는다. 검증 대상은 산술과 정렬이고, 그 둘이 틀리면 감사 결과 전체가
틀린다. 학습된 모델의 성능은 scripts/two_part_audit.py가 잰다.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.models.two_part import CENSORED_FLOOR, TwoPart
from src.panels import DEFAULT_PANEL


class _ConstGate:
    def __init__(self, p: float) -> None:
        self.p = p

    def predict_proba(self, X):
        p = np.full(X.shape[0], self.p)
        return np.column_stack([1.0 - p, p])


class _ConstReg:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, X):
        return np.full(X.shape[0], self.value)


def _model(p: float, values: dict[str, float]) -> TwoPart:
    return TwoPart(gate=_ConstGate(p),
                   regressors={iso: _ConstReg(v) for iso, v in values.items()})


def test_expected_potency_interpolates_between_regressor_and_floor():
    t = DEFAULT_PANEL.target
    model = _model(0.25, {iso: 6.4 for iso in DEFAULT_PANEL.isoforms})
    ev = model.expected_potency(["CCO"])

    expected = 0.25 * 6.4 + 0.75 * CENSORED_FLOOR
    assert ev[t][0] == pytest.approx(expected)


def test_expected_gap_cancels_the_floor():
    t, o1, o2 = DEFAULT_PANEL.isoforms
    model = _model(0.4, {t: 8.0, o1: 6.0, o2: 5.0})
    gap = model.expected_gap(["CCO"], DEFAULT_PANEL)

    # 0.4 * (8.0 - max(6.0, 5.0)); the floor appears in every isoform and cancels
    assert gap[0] == pytest.approx(0.4 * 2.0)


def test_a_certain_binder_reduces_to_the_regressor():
    t = DEFAULT_PANEL.target
    model = _model(1.0, {iso: 7.3 for iso in DEFAULT_PANEL.isoforms})
    assert model.expected_potency(["CCO"])[t][0] == pytest.approx(7.3)


def test_unparseable_smiles_stay_nan_and_keep_alignment():
    t = DEFAULT_PANEL.target
    model = _model(0.5, {iso: 6.0 for iso in DEFAULT_PANEL.isoforms})
    ev = model.expected_potency(["CCO", "not a molecule", "c1ccccc1"])

    assert len(ev[t]) == 3
    assert np.isnan(ev[t][1])
    assert not np.isnan(ev[t][0]) and not np.isnan(ev[t][2])
```

- [ ] **Step 6: 실패를 확인한다**

Run: `python -m pytest tests/test_two_part.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.models.two_part'`

- [ ] **Step 7: 최소 구현**

`src/models/two_part.py`:

```python
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

즉 2부 모델은 결합자끼리의 순위를 바꾸지 않고, 붙을 것 같지 않은 분자의 gap을 0쪽으로
수축시킨다. 그것이 의도한 전부다.

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
```

`field`는 쓰지 않으므로 import에서 뺀다 — 위 코드에 남아 있으면 지운다.

- [ ] **Step 8: 테스트 통과 확인**

Run: `python -m pytest tests/test_two_part.py -q`
Expected: PASS (4 passed)

- [ ] **Step 9: 전체 스위트 확인 후 커밋**

Run: `python -m pytest tests/ -q`
Expected: PASS, **225 passed** — 착수 시 218, Task 1이 2건, Task 2가 1건, 이 태스크가 4건.

```bash
git add src/models/two_part.py tests/test_two_part.py
git commit -m "Add the two-part model and pre-register its censored floor"
```

---

### Task 4: 두 팔 감사와 판정 (A2 / ROADMAP §12 Q1)

**착수 전에 정하는 판정 규칙 (사전등록).** 감사 결과를 본 뒤에 기준을 만들면 그것은
측정이 아니다. 아래를 지금 고정한다.

| 지표 | 채택 조건 |
|---|---|
| **C. ROC-AUC** (post-cut 실측 활성 vs 봉인 414, target 아이소폼 점수) | **+0.02 이상** 개선 |
| **B. matched-recall** (활성 95 % 유지 시 봉인 414 통과율) | 베이스라인보다 **낮다** |
| A. 반증률 (`score > pchembl_upper`) | 참고. **단독 채택 근거가 아님** |

**A가 단독 근거가 아닌 이유가 이 감사의 핵심이다.** EV는 모든 분자에서 베이스라인보다
낮거나 같다(P ≤ 1). 따라서 A는 개선될 수밖에 없고, 그 개선의 대부분은 판별력이 아니라
**보정 이동**이다 — STATE §2b가 고정 임계에서 후보가 이기는 것처럼 보였던 것과 정확히
같은 함정이다. B와 C만이 그 이동을 무효화한다.

**두 팔이 같은 모델을 쓴다.** `scripts/funnel_falsification_audit.py`는 음성 팔에 배포
모델을, 양성 팔에 재적합 모델을 쓰고 그 비교 불가를 스스로 문서화해 뒀다("an honest
Tier 0.5 recall needs the binder gate and the AD reference refit on the same cut,
and neither takes injected data yet"). 여기서는 회귀기와 게이트를 **둘 다** 컷 이전
데이터로 재적합해 두 팔에 같이 쓴다. 그것이 matched-recall 비교를 성립시키는 조건이고,
이 웨이브가 그 한계를 닫는 지점이다.

그래서 여기서 나오는 베이스라인 반증률은 STATE §2의 93.0 %와 **다를 것이다** — 93.0 %는
배포 모델의 수치다. 두 값을 섞어 읽지 않는다.

**Files:**
- Create: `scripts/two_part_audit.py`
- Modify: `STATE.md` (2절 계열에 새 소절), `VALIDATION.md`, `scripts/reproduce.sh`

**Interfaces:**
- Consumes: `two_part.TwoPart` · `two_part.CENSORED_FLOOR` · `two_part.FLOOR_SENSITIVITY`
  · `binder_gate.build_gate` · `binder_gate.at_matched_recall`
  · `panel_data.year_first` · `PanelSpec.data_bundled / "sealed_negatives.parquet"`
  · `isoform_regressor._fit` · `negatives.build_negatives` · `negatives.positive_smiles`
- Produces: 없음 (측정 스크립트). 파일을 쓰지 않는다.

---

- [ ] **Step 1: 감사 스크립트를 쓴다**

`scripts/two_part_audit.py`:

```python
#!/usr/bin/env python3
"""A2: 2부 모델이 potency 축의 반증을 실제로 줄이는가 (ROADMAP open question 1).

    EV = P(binder) * reg + (1 - P) * FLOOR

두 팔을 **같은 모델**로 잰다. 회귀기와 게이트를 둘 다 시간 컷 이전 데이터로 재적합하므로,
봉인 414(음성)와 컷 이후 실측 활성(양성)이 같은 점수 위에 놓인다. 그것이
funnel_falsification_audit이 스스로 못 한다고 적어 둔 비교이며, 여기서 닫힌다.

세 지표를 보고하고, 판정은 두 번째와 세 번째로만 한다:

  A 반증률      score > 측정 상한.  EV는 P<=1이라 반드시 내려간다 — 참고용
  B matched-recall  같은 활성 재현율에서의 봉인 414 통과율
  C ROC-AUC     활성 vs 봉인 414, 임계 없음

A만 보면 STATE.md 2b가 기록한 함정을 반복한다: 점수 분포가 통째로 내려간 보정 이동이
판별력 개선으로 보고된다.

봉인 414은 채점만 된다. FLOOR는 src/models/two_part.py에 먼저 커밋된 상수이며 여기서
탐색되지 않는다 (G6, G8).

    python scripts/two_part_audit.py           # 약 6분
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
from src.models.two_part import CENSORED_FLOOR, FLOOR_SENSITIVITY, TwoPart  # noqa: E402
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

    positives = sorted(s for s in positive_smiles(panel)
                       if year.get(s, 9999) <= CUT)
    presumed = build_negatives(panel)["smi"].tolist()
    gate = build_gate(positives, presumed)
    print(f"  gate refit on {len(positives)} pre-{CUT} positives "
          f"+ {len(presumed)} presumed negatives")
    return regressors, gate


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
        print(f"  C  ROC-AUC actives vs sealed 414   "
              f"{roc_auc_score(np.r_[np.ones(len(pos_clean)), np.zeros(len(neg))], np.r_[pos_clean, neg]):.3f}")
        print(f"\n  B  {'actives kept':>13} {'threshold':>11} {'sealed 414 pass':>17}")
        for recall in RECALLS:
            t, rate = at_matched_recall(neg, pos_clean, recall)
            print(f"     {recall:12.0%} {t:11.2f} {rate:16.1%}")

    print("\n" + "=" * 78)
    print("VERDICT — pre-registered rule")
    print("=" * 78)
    base_neg, base_pos = curves["BASELINE (regressor only)"]
    base_auc = roc_auc_score(
        np.r_[np.ones(len(base_pos)), np.zeros(len(base_neg))], np.r_[base_pos, base_neg])
    _, base_rate = at_matched_recall(base_neg, base_pos, 0.95)
    for name, (neg, pos_clean) in curves.items():
        if name.startswith("BASELINE"):
            continue
        auc = roc_auc_score(
            np.r_[np.ones(len(pos_clean)), np.zeros(len(neg))], np.r_[pos_clean, neg])
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
```

- [ ] **Step 2: 감사를 돌린다**

Run: `python scripts/two_part_audit.py`
Expected: 세 팔의 블록과 VERDICT 두 줄. 다음 두 가지를 먼저 눈으로 확인한다 —
확인 실패 시 결과를 기록하지 않는다:

1. BASELINE의 봉인 414 반증률이 60 %를 넘는다. 넘지 않으면 회귀기 재적합이 컷을 잘못
   적용한 것이다(양성 팔 모델이 음성을 이미 본 상태).
2. `TWO-PART (floor 5.0)`의 A가 BASELINE보다 **낮다**. 낮지 않으면 EV 산술이 틀렸다 —
   P ≤ 1이므로 수학적으로 반드시 낮아야 한다.

- [ ] **Step 3: 커밋**

```bash
git add scripts/two_part_audit.py
git commit -m "Score the two-part model against both arms of the same refit"
```

- [ ] **Step 4: `reproduce.sh`에 네 블록을 추가한다**

`scripts/reproduce.sh`의 "DMTA loop, three rounds" 블록 **앞**에 넣는다:

```bash
echo
echo "== Same-document labels (P4.1, open question 3) =="
echo "The selectivity gap rebuilt from one publication per molecule, against the"
echo "pooled-median labels the deployed models train on. Same metric as Gate 4."
python scripts/matched_label_audit.py

echo
echo "== Two-part model, both arms (A2, open question 1) =="
echo "EV = P(binder) x regressor + (1 - P) x floor, scored against the sealed"
echo "non-binders and post-cut actives through one pre-cut refit."
python scripts/two_part_audit.py
```

`assay_time_audit.py` 블록은 이미 있으므로 추가하지 않고, 그 위의 안내문에서 "네트워크
재빌드가 필요하다"는 취지의 문장이 있으면 지운다 — 이제 오프라인에서 돈다.

- [ ] **Step 5: STATE.md에 판정을 적는다**

§2b(게이트 음성 재정의) **뒤**에 `### 2c. 2부 모델 (A2) — [채택 / 반증됨] (2026-08-10)`을
추가한다. 다음을 포함한다:

- 재현 명령 `python scripts/two_part_audit.py`
- 두 팔이 같은 재적합을 쓴다는 것, 그리고 그래서 여기 베이스라인이 §2의 93.0 %와 다르다는 것
- A/B/C 세 지표 표, floor 5.0과 4.0 두 줄
- **사전등록 판정 규칙과 그 결과** — 반증이면 반증이라고 적는다(G5)
- 한계: 게이트는 추정 음성만으로 재적합됐다(§2b가 측정 음성 투입을 반증했으므로),
  스캐폴드 분할 없음, 튜닝 없음(§8)

§0a의 DMTA 준비도 표에서 **D** 행의 "남은 것"을 갱신한다 — A2가 반증되면 potency 축의
남은 수단은 직교 증거(G-1)뿐이라는 것이 §2b와 함께 두 번째 증거가 된다.

- [ ] **Step 6: VALIDATION.md에 두 절을 추가한다**

같은 형식의 기존 절(최근접이웃 베이스라인, 게이트 A/B)을 본떠 쓴다:

1. **문헌 내 매칭 라벨** — pooled와 same-document의 Spearman·enrichment 표, 그리고
   same-assay 매칭이 **구조적으로 불가능**하다는 측정(아이소폼 3개를 담은 assay 0개).
   ROADMAP §12 Q3이 이 형태로 닫혔다고 적는다.
2. **assay 종류 / 시간 컷** — 이제 오프라인에서 도는 `assay_time_audit.py`의 출력.
   Q2의 트레이드 표(컷별 train/test)를 그대로 싣는다.

- [ ] **Step 7: 게이트를 확인한다**

```bash
python -m pytest tests/ -q
git diff --stat main..HEAD -- assets/models assets/jak/JAK1.parquet \
    assets/jak/JAK2.parquet assets/jak/JAK3.parquet assets/library
```
Expected: 스위트 전부 통과. 두 번째 명령은 **빈 출력** — 배포 자산이 안 움직였다는 G0 증거.

- [ ] **Step 8: 커밋하고 푸시한다**

```bash
git add STATE.md VALIDATION.md scripts/reproduce.sh
git commit -m "Record what matched labels and the two-part model did to the potency axis"
git push -u origin claude/bioinformatics-computational-chemistry-wje9cs
```

푸시가 네트워크 오류로 실패하면 2s → 4s → 8s → 16s 백오프로 최대 4회 재시도한다.
**PR은 만들지 않는다** (명시 요청 없음).

---

## 이 웨이브가 닫는 것

| ROADMAP | 어디서 |
|---|---|
| P4.1 assay-matched 라벨 | Task 1 |
| §12 Q3 (same-document vs same-assay) | Task 1 + Task 2 — same-assay는 **불가능**으로 닫힌다 |
| §12 Q2 (시간 컷 2018 vs 2020) | Task 2 |
| A2 2부 모델 + EV 재구성 | Task 3 + Task 4 |
| §12 Q1 (2부 모델 vs 검열 손실 회귀) | Task 4 — 2부 모델이 반증되면 Tobit/AFT가 다음 후보로 남는다 |
| §8 "튜닝은 A2 이후" 봉쇄 | Task 4 이후 해제 가능 |

## 이 웨이브가 닫지 않는 것

- **배포는 바뀌지 않는다.** 채택 판정이 나와도 게이트·회귀기 교체는 임계와 Tier 0.5와 그
  아래 VALIDATION 전체를 움직이는 별도 G0 사안이다 — §2b가 세운 선례를 그대로 따른다.
- **P4.3 라이브러리 누출**은 패널 인자화가 모이는 W8로 미룬다.
- 검열 손실 회귀(Tobit/AFT)는 A2가 반증될 때만 열린다. 지금 둘 다 만들지 않는다.
