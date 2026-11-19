from copy import deepcopy
import json

import pytest
from evaluations.review import export, read_jsonl, summarize, weighted_kappa
from evaluations.v1.suite import make_case


def test_blind_export_and_unfilled_labels_rejected(tmp_path):
    case, cat = make_case(2, 'earbuds', 'single', 'dev')
    rec = {'case_id': case['case_id'], 'config': 'B3', 'rep': 0, 'outcome': {'kind': 'error'}}
    out = tmp_path / 'review'
    export([rec, {**rec, 'rep': 1}], [case], {cat['catalog_id']: cat}, out)
    a = read_jsonl(out / 'reviewer-a.jsonl')
    b = read_jsonl(out / 'reviewer-b.jsonl')
    key = read_jsonl(out / 'private-key.jsonl')
    assert len(a) == 1 and 'config' not in a[0] and 'case_id' not in a[0]
    with pytest.raises(ValueError, match='reviewer'):
        summarize(a, b, key)
    for rows, reviewer in ((a, 'alice'), (b, 'bob')):
        rows[0]['label'].update(reviewer=reviewer, not_applicable_reason='model error', claims_complete=True)
    report = summarize(a, b, key)
    assert report['by_config']['B3']['alice']['faithfulness_micro']['rate'] is None
    assert report['quadratic_weighted_kappa'] is None
    b[0]['label']['reviewer'] = 'alice'
    with pytest.raises(ValueError, match='distinct'):
        summarize(a, b, key)


def test_blind_ids_are_random_not_derivable_from_config_case_repeat(tmp_path):
    case, cat = make_case(2, 'earbuds', 'single', 'dev')
    rec = {'case_id': case['case_id'], 'config': 'B3', 'rep': 0, 'outcome': {'kind': 'error'}}
    export([rec], [case], {cat['catalog_id']: cat}, tmp_path / 'one')
    export([rec], [case], {cat['catalog_id']: cat}, tmp_path / 'two')
    first = read_jsonl(tmp_path / 'one' / 'private-key.jsonl')[0]['review_id']
    second = read_jsonl(tmp_path / 'two' / 'private-key.jsonl')[0]['review_id']
    assert first != second and first != __import__('hashlib').sha256(b'17:B3:v1-dev-02:0').hexdigest()[:16]


def test_claim_rates_and_rating_agreement():
    label = {'reviewer': 'one', 'preference_fit': 4, 'claims_complete': True, 'claims': [
        {'text': 'total is 18', 'supported': True, 'requires_citation': True, 'cited': True, 'citation_supported': True},
        {'text': 'all taxes included', 'supported': False, 'requires_citation': True, 'cited': False, 'citation_supported': False}]}
    a = [{'review_id': 'x', 'label': label}]
    b = deepcopy(a)
    b[0]['label']['reviewer'] = 'two'
    r = summarize(a, b, [{'review_id': 'x', 'config': 'B3'}])['by_config']['B3']['one']
    assert r['faithfulness_micro'] == {'n': 1, 'm': 2, 'rate': .5}
    assert r['citation_support']['rate'] == 1
    assert r['citation_coverage']['rate'] == .5
    assert weighted_kappa([(1, 1), (5, 5)]) == 1
    assert weighted_kappa([(1, 5), (5, 1)]) == -1
