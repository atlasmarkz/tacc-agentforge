"""Hand-calculated checks; no model scores or human labels are synthesized."""
from copy import deepcopy
from collections import Counter
import json

import pytest

from evaluations.v1.oracle import funds, inspect, money, score, truth
from evaluations.v1.suite import build, make_case
from evaluations.v1.run import schedule, summarize
from evaluations.shoppingbench import audit


def example():
    return make_case(2, 'earbuds', 'basket', 'dev')


def test_order_shipping_tax_and_exact_boundary():
    c, cat = example()
    items = [{'offer_id': o['offer_id'], 'quantity': 1} for o in cat['offers'] if o['offer_id'].endswith('-1')]
    c['budget_basis'] = 'all_in'
    c['constraints']['purchase_cap'] = '34.00'
    v = inspect(items, c, cat)
    assert v['status'] == 'feasible'
    assert v['costs']['total'] == '34.00'  # 15+1 twice, shared shipping 2 only once
    assert v['remaining'] == '0.00'
    c['constraints']['purchase_cap'] = '33.99'
    assert inspect(items, c, cat)['over_budget']


def test_budget_sources_missing_values_and_negative_balance():
    assert funds({'purchase_cap': '50.00', 'periodic_budget': '100.00', 'spent': '50.00', 'reserve': '20.00'})[0] == 30
    assert funds({'periodic_budget': '100.00', 'spent': '50.00'}) == (None, ['reserve'])
    assert funds({'periodic_budget': '10.00', 'spent': '20.00', 'reserve': '0.00'})[0] == -10
    for value in ('NaN', 'Infinity', '-1.00', '1.001', 1.0):
        with pytest.raises(ValueError):
            money(value)


def test_unknown_and_infeasible_do_not_collapse():
    c, cat = make_case(3, 'earbuds', 'uncertainty')
    assert truth(c, cat)['expected'] == 'insufficient_evidence'
    c['constraints']['purchase_cap'] = '1.00'
    # The known item price alone already proves infeasibility.
    assert truth(c, cat)['expected'] == 'no_feasible_plan'
    for o in cat['offers']:
        o['shipping'] = '0.00'
    assert truth(c, cat)['expected'] == 'no_feasible_plan'
    rec = {'config': 'B0', 'rep': 0, 'outcome': {'kind': 'ask_user', 'required_fields': ['colour']}}
    assert not score(rec, c, cat)['success']


def test_amounts_citations_and_multiple_solutions():
    c, cat = make_case(2, 'earbuds', 'single')
    c['constraints']['purchase_cap'] = '80.00'  # keep multiple feasible solutions
    gt = truth(c, cat)
    assert len(gt['feasible']) == 2
    plan = deepcopy(gt['feasible'][1])
    plan.pop('items')
    items = deepcopy(gt['feasible'][1]['items'])
    for i in items:
        i['evidence_ids'] = [next(o['fact_evidence_id'] for o in cat['offers'] if o['offer_id'] == i['offer_id'])]
    rec = {'config': 'B0', 'rep': 0, 'context_ids': [e['evidence_id'] for e in cat['evidence']],
           'outcome': {'kind': 'plan', 'items': items, **plan}}
    assert score(rec, c, cat)['success']
    rec['outcome']['total'] = '0.00'
    assert score(rec, c, cat)['amount_error']
    rec['outcome']['total'] = plan['total']
    rec['outcome']['items'][0]['evidence_ids'] = [cat['evidence'][0]['evidence_id']]
    assert not score(rec, c, cat)['success']
    rec['outcome']['items'][0]['evidence_ids'] = ['invented']
    assert not score(rec, c, cat)['success']


def test_fixture_covers_boundary_clarify_and_one_cent_infeasible(tmp_path):
    from decimal import Decimal
    build(tmp_path)
    rows = [json.loads(l) for l in (tmp_path / 'cases.jsonl').read_text().splitlines()]
    cats = json.loads((tmp_path / 'catalogs.json').read_text())
    expected = Counter((c['split'], truth(c, cats[c['catalog_id']])['expected']) for c in rows)
    assert expected[('dev', 'clarify')] == 1 and expected[('heldout', 'clarify')] == 1
    boundaries = []
    for case in rows:
        gt = truth(case, cats[case['catalog_id']])
        for plan in gt.get('feasible', []):
            if Decimal(plan['remaining']) <= Decimal('0.01'):
                boundaries.append((case['case_id'], plan['remaining']))
    assert boundaries
    # The 11.99 cap is exactly one cent below the cheapest known total.
    case = next(c for c in rows if c['case_id'] == 'v1-dev-00')
    assert case['constraints']['purchase_cap'] == '11.99'
    assert truth(case, cats[case['catalog_id']])['expected'] == 'no_feasible_plan'


def test_versioned_suite_size_matrix_and_fifth_release(tmp_path):
    build(tmp_path)
    rows = [json.loads(l) for l in (tmp_path / 'cases.jsonl').read_text().splitlines()]
    assert Counter(c['split'] for c in rows) == {'dev': 10, 'heldout': 32}
    assert json.loads((tmp_path / 'manifest.json').read_text())['pending'] == 8
    # Only the gated release CLI may call this in a real evaluation.
    build(tmp_path, fifth=True)
    rows = [json.loads(l) for l in (tmp_path / 'cases.jsonl').read_text().splitlines()]
    held = [r for r in rows if r['split'] == 'heldout']
    assert Counter(c['language'] for c in held) == {'zh': 20, 'en': 20}
    assert set(Counter(c['category'] for c in held).values()) == {8}
    assert Counter(c['task_type'] for c in held) == {'single': 12, 'basket': 8, 'ledger': 6, 'dialogue': 8, 'uncertainty': 6}
    assert len(schedule(held, ['B0', 'B1', 'B2', 'B3', 'A1', 'A2'], 3)) == 720


def test_failed_attempts_count_and_zero_plan_bvr():
    c, cat = make_case(2, 'earbuds', 'single')
    records = [{'case_id': c['case_id'], 'config': 'B0', 'rep': i, 'outcome': {'kind': 'error'}} for i in range(3)]
    _, report = summarize(records, [c], {c['catalog_id']: cat})
    s = report['summary']['B0']
    assert s['tsr']['m'] == 3 and s['tsr']['n'] == 0
    assert s['bvr']['rate'] is None and s['tsr_by_case']['cases'] == 1


def test_external_complete_roster_not_only_generated_outputs():
    tasks = [{'query': 'a'}, {'query': 'b'}]
    rollouts = [[{'extra_info': {'query': 'a'}}]]
    report = audit(tasks, rollouts, lambda *_: {'rule': 1, 'budget': 1, 'finished': True})
    assert report['asr'] == .5 and report['missing'] == 1 and report['completion_rate'] == .5
    with pytest.raises(ValueError):
        audit(tasks, rollouts * 2, lambda *_: {})


def test_scripted_revision_keeps_original_requirements(monkeypatch):
    from evaluations.v1 import run
    c, cat = make_case(4, 'earbuds', 'dialogue', 'dev')
    oid, eid = cat['offers'][1]['offer_id'], cat['offers'][1]['fact_evidence_id']
    payloads = []

    def respond(adapter, payload, timeout):
        payloads.append(payload)
        assert 'requirements' not in payload and 'expected' not in payload
        return {'context_ids': [eid], 'outcome': {'kind': 'plan',
                'items': [{'offer_id': oid, 'quantity': 1, 'evidence_ids': [eid]}],
                'lines': [{'offer_id': oid, 'quantity': 1, 'subtotal': '15.00'}],
                'shipping': '2.00', 'total': '17.00',
                'remaining': '63.00' if len(payloads) == 1 else '13.00'}}
    monkeypatch.setattr(run, 'invoke', respond)
    rec = run.run_episode('unused:adapter', c, cat, 'B0', 10)
    rec['rep'] = 0
    assert len(payloads) == 2 and payloads[1]['history']
    assert score(rec, c, cat)['success']
    assert rec['prompt_tokens'] is None


def always_error(payload):
    """Test-only protocol endpoint, never registered as a benchmark baseline."""
    return {'outcome': {'kind': 'error'}, 'error': 'fixture_error'}


def test_cli_roundtrip_and_interrupted_roster(tmp_path):
    import subprocess
    import sys
    build(tmp_path / 'data')
    caps = tmp_path / 'adapter.json'
    caps.write_text(json.dumps({'identity': 'test-only-error-adapter', 'validator_configs': [], 'cache_policy': 'none'}))
    out = tmp_path / 'run'
    base = [sys.executable, '-m', 'evaluations.v1.run', '--data', str(tmp_path / 'data')]
    subprocess.run(base + ['--adapter', 'tests.test_benchmark_v1:always_error', '--adapter-manifest', str(caps),
                          '--configs', 'B0', '--repeats', '1', '--out', str(out)], check=True, capture_output=True)
    rows = (out / 'records.jsonl').read_text().splitlines()
    assert len(rows) == 10
    assert 'git' in json.loads((out / 'manifest.json').read_text())
    (out / 'records.jsonl').write_text('\n'.join(rows[:-1]) + '\n')
    subprocess.run(base + ['--rescore', str(out)], check=True, capture_output=True)
    report = json.loads((out / 'report.json').read_text())
    assert report['missing_attempts'] == 1
    assert report['summary']['B0']['tsr'] == {'n': 0, 'm': 10, 'rate': 0.0}


def test_freeze_release_cli(tmp_path):
    import subprocess
    import sys
    build(tmp_path)
    base = [sys.executable, '-m', 'evaluations.v1.suite']
    failed = subprocess.run(base + ['release-fifth', '--data', str(tmp_path)], capture_output=True)
    assert failed.returncode != 0
    for action in ('freeze', 'release-fifth', 'verify'):
        subprocess.run(base + [action, '--data', str(tmp_path)], check=True, capture_output=True)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['heldout'] == 40 and manifest['pending'] == 0
    assert (tmp_path / 'release.json').exists()
    assert 'git' in json.loads((tmp_path / 'freeze.json').read_text())


def test_itemized_amounts_allow_order_and_equivalent_decimal_format():
    c, cat = example()
    c['budget_basis'] = 'all_in'
    chosen = truth(c, cat)['feasible'][0]
    items = deepcopy(chosen['items'])
    for item in items:
        item['evidence_ids'] = [next(o['fact_evidence_id'] for o in cat['offers'] if o['offer_id'] == item['offer_id'])]
    out = {'kind': 'plan', **chosen, 'items': items}
    out['lines'] = list(reversed(out['lines']))
    out['shipping'] = '2.0'
    rec = {'config': 'B3', 'rep': 0, 'outcome': out, 'context_ids': [e['evidence_id'] for e in cat['evidence']]}
    assert score(rec, c, cat)['success']


def test_bvr_excludes_unverifiable_costs_from_denominator():
    unknown_case, unknown_cat = make_case(3, 'earbuds', 'uncertainty')
    over_case, over_cat = make_case(0, 'earbuds', 'single')
    unknown_offer = unknown_cat['offers'][0]
    over_offer = over_cat['offers'][0]
    records = [
        {'case_id': unknown_case['case_id'], 'config': 'B0', 'rep': 0,
         'context_ids': [e['evidence_id'] for e in unknown_cat['evidence']],
         'outcome': {'kind': 'plan', 'items': [{'offer_id': unknown_offer['offer_id'], 'quantity': 1,
                    'evidence_ids': [unknown_offer['fact_evidence_id']]}],
                    'lines': [], 'shipping': '0.00', 'total': '0.00', 'remaining': '80.00'}},
        {'case_id': over_case['case_id'], 'config': 'B0', 'rep': 1,
         'context_ids': [e['evidence_id'] for e in over_cat['evidence']],
         'outcome': {'kind': 'plan', 'items': [{'offer_id': over_offer['offer_id'], 'quantity': 1,
                    'evidence_ids': [over_offer['fact_evidence_id']]}],
                    'lines': [], 'shipping': '2.00', 'total': '12.00', 'remaining': '-0.01'}},
    ]
    _, report = summarize(records, [unknown_case, over_case],
                          {unknown_cat['catalog_id']: unknown_cat, over_cat['catalog_id']: over_cat})
    b0 = report['summary']['B0']
    assert b0['unverifiable_amount_plans'] == 1
    assert b0['bvr'] == {'n': 1, 'm': 1, 'rate': 1.0}


def test_missing_context_is_na_not_a_plan_failure():
    c, cat = make_case(2, 'earbuds', 'single')
    c['constraints']['purchase_cap'] = '80.00'
    chosen = truth(c, cat)['feasible'][0]
    items = deepcopy(chosen['items'])
    for item in items:
        item['evidence_ids'] = [next(o['fact_evidence_id'] for o in cat['offers']
                                      if o['offer_id'] == item['offer_id'])]
    record = {'config': 'B0', 'rep': 0, 'outcome': {'kind': 'plan', **chosen, 'items': items}}
    result = score(record, c, cat)
    assert result['success'] and not result['context_checked']
    assert result['ids_not_in_context'] is None


def test_invalid_action_is_counted():
    c, cat = make_case(2, 'earbuds', 'single')
    offer = cat['offers'][1]
    record = {'case_id': c['case_id'], 'config': 'B0', 'rep': 0, 'stop_reason': 'invalid_action',
              'context_ids': [e['evidence_id'] for e in cat['evidence']],
              'outcome': {'kind': 'plan', 'items': [{'offer_id': offer['offer_id'], 'quantity': 2,
                         'evidence_ids': [offer['fact_evidence_id']]}],
                         'lines': [], 'shipping': '2.00', 'total': '32.00', 'remaining': '48.00'}}
    _, report = summarize([record], [c], {c['catalog_id']: cat})
    assert report['summary']['B0']['invalid_actions'] == 1


def test_b2_filters_candidates_while_b1_keeps_the_shared_universe():
    from evaluations.v1.model_adapter import context_for
    c, cat = make_case(1, 'chargers', 'basket')
    payload = {'config': 'B1', 'case_id': c['case_id'], 'message': 'Choose chargers only',
               'history': [], 'constraints': c['initial_constraints'],
               'budget_basis': c['budget_basis'], 'catalog': cat}
    b1, _ = context_for({**payload, 'config': 'B1'})
    b2, _ = context_for({**payload, 'config': 'B2'})
    assert {o['category'] for o in b1['candidates']} == {'chargers', 'keyboards'}
    assert {o['category'] for o in b2['candidates']} == {'chargers'}


def test_rescore_ignores_a_partial_trailing_record(tmp_path):
    import subprocess
    import sys
    build(tmp_path / 'data')
    caps = tmp_path / 'adapter.json'
    caps.write_text(json.dumps({'identity': 'test-error-adapter', 'validator_configs': [], 'cache_policy': 'none'}))
    out = tmp_path / 'run'
    base = [sys.executable, '-m', 'evaluations.v1.run', '--data', str(tmp_path / 'data')]
    subprocess.run(base + ['--adapter', 'tests.test_benchmark_v1:always_error',
                           '--adapter-manifest', str(caps), '--configs', 'B0', '--repeats', '1',
                           '--out', str(out)], check=True, capture_output=True)
    lines = (out / 'records.jsonl').read_text().splitlines()
    (out / 'records.jsonl').write_text('\n'.join(lines[:-1]) + '\n' + lines[-1][:15])
    subprocess.run(base + ['--rescore', str(out)], check=True, capture_output=True)
    report = json.loads((out / 'report.json').read_text())
    assert report['missing_attempts'] == 1
    assert report['malformed_lines_ignored'] == 1


def test_strip_traces_removes_raw_prompt_and_intermediate_state():
    from evaluations.v1.run import strip_traces
    record = {'turns': [{'raw': 'x'}], 'model_trace': [{'reply': 'x'}], 'events': [{'data': 'x'}],
              'parsed_requirements': [{'category': 'x'}], 'unchecked_proposal': {'items': []},
              'outcome': {'kind': 'error'}, 'context_ids': ['e0'], 'model_calls': 1}
    stripped = strip_traces(record)
    assert stripped['outcome'] == {'kind': 'error'} and stripped['context_ids'] == ['e0']
    assert not ({'turns', 'model_trace', 'events', 'parsed_requirements', 'unchecked_proposal'} & set(stripped))


def test_malformed_abstention_is_a_failed_attempt_not_a_scoring_crash():
    c, cat = make_case(3, 'earbuds', 'uncertainty')
    for outcome in ([], {'kind': 'insufficient_evidence', 'reason_codes': [{}]}):
        result = score({'config': 'B0', 'rep': 0, 'outcome': outcome}, c, cat)
        assert not result['success']


def test_report_prompt_has_no_pre_tax_tax_contradiction():
    from evaluations.v1.model_adapter import REPORT_PROMPT
    assert 'budget_basis' in REPORT_PROMPT
    assert 'pre_tax' in REPORT_PROMPT and 'price * quantity' in REPORT_PROMPT
    assert 'all_in' in REPORT_PROMPT and '(price + tax) * quantity' in REPORT_PROMPT
    assert 'unit price plus per-unit tax times quantity' not in REPORT_PROMPT
