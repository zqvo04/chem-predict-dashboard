"""Name -> structure resolution (offline; the live lookup self-skips)."""
import pytest
import requests

from src.data import pubchem_client as pc

_RUX = "N#CC[C@H](C1CCCC1)n1cc(-c2ncnc3[nH]ccc23)cn1"


def _payload(*records):
    return {"PropertyTable": {"Properties": list(records)}}


def test_resolve_name_returns_the_standardised_parent(monkeypatch):
    """PubChem serves marketed drugs as salts. Returning the salt would give the
    molecule a fingerprint the models never saw and flag a known JAK inhibitor as
    out of domain, so resolution always hands back the neutral parent."""
    monkeypatch.setattr(pc, "_get", lambda url, timeout=40: _payload(
        {"CID": 25127112, "SMILES": _RUX + ".OP(=O)(O)O", "Title": "Jakafi"}))
    smiles, cid, title = pc.resolve_name("ruxolitinib phosphate")
    assert smiles == _RUX
    assert cid == 25127112 and title == "Jakafi"


def test_resolve_name_skips_records_whose_structure_will_not_parse(monkeypatch):
    monkeypatch.setattr(pc, "_get", lambda url, timeout=40: _payload(
        {"CID": 1, "SMILES": "not_a_smiles", "Title": "broken"},
        {"CID": 2, "SMILES": "CCO", "Title": "ethanol"}))
    assert pc.resolve_name("ethanol") == ("CCO", 2, "ethanol")


def test_unknown_name_raises_name_not_found(monkeypatch):
    response = requests.Response()
    response.status_code = 404

    def _404(url, timeout=40):
        raise requests.HTTPError(response=response)

    monkeypatch.setattr(pc, "_get", _404)
    with pytest.raises(pc.NameNotFound):
        pc.resolve_name("definitely-not-a-drug")


def test_empty_query_raises_without_touching_the_network(monkeypatch):
    monkeypatch.setattr(pc, "_get", lambda *a, **k: pytest.fail("hit the network"))
    with pytest.raises(pc.NameNotFound):
        pc.resolve_name("   ")


def test_a_transport_failure_is_not_reported_as_an_unknown_name(monkeypatch):
    """"You are offline" and "no such drug" are different problems for the caller."""
    def _boom(url, timeout=40):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(pc, "_get", _boom)
    with pytest.raises(requests.RequestException):
        pc.resolve_name("ruxolitinib")


def test_live_name_lookup():
    try:
        smiles, cid, _ = pc.resolve_name("ruxolitinib")
    except requests.RequestException as err:
        pytest.skip(f"PubChem unreachable: {err}")
    assert smiles == _RUX and cid == 25126798
