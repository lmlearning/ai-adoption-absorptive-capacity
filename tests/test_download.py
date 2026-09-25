import hashlib
import importlib.util
import io
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("survey_download", Path(__file__).resolve().parents[1] / "data/download_stackoverflow.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_verified_download_and_idempotent_existing_file(tmp_path, monkeypatch):
    payload = b"ResponseId,value\n1,example\n"
    digest = hashlib.sha256(payload).hexdigest()
    calls = []
    def response(*args, **kwargs):
        calls.append(args)
        return io.BytesIO(payload)
    monkeypatch.setattr(module, "urlopen", response)
    target = tmp_path / "survey.csv"
    module.download(target, expected_sha256=digest)
    module.download(target, expected_sha256=digest)
    assert target.read_bytes() == payload
    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == [target]


def test_wrong_checksum_removes_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "urlopen", lambda *a, **kw: io.BytesIO(b"wrong"))
    with pytest.raises(ValueError, match="checksum"):
        module.download(tmp_path / "survey.csv")
    assert list(tmp_path.iterdir()) == []


def test_existing_unexpected_file_is_preserved(tmp_path):
    target = tmp_path / "survey.csv"
    target.write_bytes(b"my data")
    with pytest.raises(ValueError, match="preserved"):
        module.download(target)
    assert target.read_bytes() == b"my data"
