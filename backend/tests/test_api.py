# tests/test_api.py
import os
import pytest
import requests

BASE = os.getenv("BASE_URL", "http://127.0.0.1:8000")

def _get(path: str):
    r = requests.get(f"{BASE}{path}", timeout=10)
    assert r.status_code == 200, r.text
    return r.json()

def _post(path: str, body):
    r = requests.post(f"{BASE}{path}", json=body, timeout=15)
    assert r.status_code == 200, r.text
    return r.json()

def test_health():
    j = _get("/health")
    assert "status" in j and j["status"] == "operational"
    assert "models_loaded" in j
    assert "knowledge_entries" in j
    # If models loaded, entries should be >= 1
    if j["models_loaded"]:
        assert j["knowledge_entries"] >= 1

@pytest.mark.parametrize("payload,expect_suggestions_keywords", [
    ({"user_input": "I am very anxious about my presentation tomorrow"}, ["breathing", "visualization", "breaks"]),
    ({"user_input": "I have a doctor appointment and I'm nervous about the results"}, ["prepare", "support"]),
])
def test_generate_variants(payload, expect_suggestions_keywords):
    j = _post("/generate", payload)
    assert "response" in j and isinstance(j["response"], str) and len(j["response"]) > 0
    assert "suggestions" in j and isinstance(j["suggestions"], list)
    joined = " ".join(j["suggestions"]).lower()
    assert any(k in joined for k in expect_suggestions_keywords)

def test_generate_empty_input():
    j = _post("/generate", {"user_input": "   "})
    assert "here to listen" in j["response"].lower()

def test_debug_corpus():
    j = _get("/debug/corpus")
    assert "total_entries" in j
    assert "sample_entries" in j
    assert isinstance(j["sample_entries"], list)

@pytest.mark.parametrize("q", ["anxious", "doctor", "overwhelmed"])
def test_debug_search(q):
    j = _get(f"/debug/search/{q}")
    assert j["query"] == q
    assert "results" in j and isinstance(j["results"], list)
    assert "results_found" in j and j["results_found"] == len(j["results"])
