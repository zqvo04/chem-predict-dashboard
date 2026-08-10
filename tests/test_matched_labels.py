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
