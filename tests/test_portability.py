"""Regression checks for complete, offline public clones."""
import hashlib
import json
import pytest
from scripts.check_setup import verify_files


def test_manifest_detects_incomplete_and_changed_assets(tmp_path):
    asset = tmp_path / 'model.bin'
    asset.write_bytes(b'complete')
    (tmp_path / 'runtime-manifest.json').write_text(json.dumps({
        'files': {'model.bin': hashlib.sha256(b'complete').hexdigest()}}))
    verify_files(tmp_path)
    asset.write_bytes(b'truncated')
    with pytest.raises(ValueError, match='changed or incomplete'):
        verify_files(tmp_path)
    asset.unlink()
    with pytest.raises(ValueError, match='missing model.bin'):
        verify_files(tmp_path)


def test_english_preprocessing_without_downloads_or_home_cache(monkeypatch):
    import nltk
    from src.nlp_processor import NLP_DATA_DIR, ensure_nltk_data, default_preprocess
    def forbidden(*args, **kwargs):
        raise AssertionError('Offline preprocessing attempted a download')
    monkeypatch.setattr(nltk, 'download', forbidden)
    assert nltk.data.path == [str(NLP_DATA_DIR)]
    ensure_nltk_data()
    assert default_preprocess('The accounts were charged twice; I cannot log in.')


def test_shipped_manifest_is_complete():
    verify_files()
