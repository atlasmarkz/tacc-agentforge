"""Index provenance checks for the public ShoppingBench preflight."""
from evaluations.shoppingbench_run import index_digest, verify_index


def test_index_digest_matches_and_detects_changes(tmp_path):
    index = tmp_path / 'indexes'
    index.mkdir()
    (index / 'segments_1').write_text('one', encoding='utf-8')
    (index / 'write.lock').write_text('two', encoding='utf-8')
    manifest = index_digest(index)
    assert manifest['index_files'] == 2
    assert manifest['index_bytes'] == 6
    assert verify_index(index, manifest) == []

    (index / 'segments_1').write_text('changed', encoding='utf-8')
    assert 'index_sha256_mismatch' in verify_index(index, manifest)
    assert 'index_bytes_mismatch' in verify_index(index, manifest)
