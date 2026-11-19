from pathlib import Path
from evaluations.shoppingbench import audit
from evaluations.shoppingbench_run import config, UPSTREAM_COMMIT


def test_public_configuration_keeps_official_roster_and_outputs_separate():
    cfg = config(Path('/public'), Path('/outputs'), 'local-model')
    assert cfg['task'] == 'voucher'
    assert cfg['synthesize_file'] == '/public/data/synthesize_voucher_test.jsonl'
    assert cfg['rollout_file'] == '/outputs/rollouts.jsonl'
    assert cfg['model_config']['model'] == 'local-model'
    assert len(UPSTREAM_COMMIT) == 40


def test_audit_keeps_the_denominator_for_empty_or_failing_output():
    report = audit([{'query': 'a'}, {'query': 'b'}], [[]], lambda *_: {'rule': 1, 'budget': 1})
    assert report['tasks'] == 2 and report['asr'] == 0
    assert report['unidentified_rollouts'] == 1 and report['missing'] == 2

    def fail(*_):
        raise RuntimeError('upstream scorer failed')

    report = audit([{'query': 'a'}], [[{'extra_info': {'query': 'a'}}]], fail)
    assert report['tasks'] == 1 and report['asr'] == 0
    assert report['rows'][0]['error'].startswith('RuntimeError')
    assert 'scoring-error' in report['quality_interpretation']


def test_api_complete_rejects_error_finish_reason():
    from evaluations.shoppingbench_recovery import api_complete
    good = [{'completion': {'content': 'ok', 'reasoning_content': '', 'finish_reason': 'stop'}}]
    bad = [{'completion': {'content': 'rate limited', 'reasoning_content': '', 'finish_reason': 'error'}}]
    assert api_complete(good) and not api_complete(bad)


def test_recovery_does_not_retry_incorrect_but_api_complete_answers(tmp_path):
    import json
    from evaluations.shoppingbench_recovery import prepare
    def row(q, content):
        return [{'extra_info': {'query': q}, 'completion': {'content': content, 'reasoning_content': ''}}]
    prior = tmp_path / 'prior.jsonl'
    records = [row('wrong-answer', 'a completely wrong recommendation'), row('api-failure', '')]
    original = ''.join(json.dumps(r) + '\n' for r in records)
    prior.write_text(original)
    dest = tmp_path / 'next.jsonl'
    m = prepare([{'query': 'wrong-answer'}, {'query': 'api-failure'}], prior, dest)
    assert m['inherited_api_complete'] == 1 and m['retry_queries'] == ['api-failure']
    assert json.loads(dest.read_text())[0]['extra_info']['query'] == 'wrong-answer'
    assert prior.read_text() == original


def test_scoring_error_is_not_counted_as_api_incomplete():
    complete = [{'extra_info': {'query': 'a'},
                 'completion': {'content': 'ok', 'reasoning_content': '', 'finish_reason': 'stop'}}]
    def fail(*_):
        raise RuntimeError('upstream scorer failed')
    report = audit([{'query': 'a'}], [complete], fail)
    assert report['api_incomplete'] == 0
    assert report['scoring_errors'] == 1
    assert report['rows'][0]['api_complete'] is True
    assert 'scoring-error' in report['quality_interpretation']


def test_corpus_hash_is_checked_even_without_the_legacy_flag(tmp_path):
    from evaluations.shoppingbench_run import file_hash, verify_corpus
    corpus = tmp_path / 'documents.jsonl.gz'
    corpus.write_bytes(b'corpus')
    digest, blocker = verify_corpus(corpus, expected_bytes=6, expected_sha256=file_hash(corpus))
    assert blocker is None and digest == file_hash(corpus)
    digest, blocker = verify_corpus(corpus, expected_bytes=6, expected_sha256='0' * 64)
    assert digest == file_hash(corpus) and blocker == 'corpus_hash_mismatch'
    digest, blocker = verify_corpus(corpus, expected_bytes=999, expected_sha256=file_hash(corpus))
    assert digest is None and blocker == 'missing_full_lfs_corpus'
