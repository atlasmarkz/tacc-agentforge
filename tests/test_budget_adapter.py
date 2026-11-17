"""Business validation and actual AgentRuntime integration; no oracle in the adapter."""
from copy import deepcopy
import json

import pytest

from budget.validator import CatalogValidator, available_funds, cents
from evaluations.v1.agent_adapter import respond
from evaluations.v1.oracle import score, truth
from evaluations.v1.suite import make_case
from inference.client import ChatResult


class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def chat(self, messages, **kwargs):
        self.messages.append(messages)
        assert self.responses, 'unexpected model call'
        return ChatResult(content=json.dumps(self.responses.pop(0)), prompt_tokens=10, completion_tokens=5)


def payload(kind='single', index=2, cfg='B3'):
    case, catalog = make_case(index, 'earbuds', kind, 'dev')
    return case, {'config': cfg, 'case_id': case['case_id'], 'message': case['message'],
        'history': [], 'catalog': catalog, 'constraints': case['initial_constraints'],
        'budget_basis': case['budget_basis']}


def test_business_checker_computes_and_refuses_missing_fees():
    case, p = payload()
    checker = CatalogValidator(p['catalog'], case['requirements'], p['constraints'], 'all_in')
    offer = p['catalog']['offers'][1]
    items = [{'offer_id': offer['offer_id'], 'quantity': 1, 'evidence_ids': [offer['fact_evidence_id']]}]
    check = checker.check(items)
    assert check['total'] == '18.00' and check['remaining'] == '0.00'
    offer['tax'] = None
    assert checker.check(items)['status'] == 'unknown'
    offer['price'] = '100.00'
    assert checker.check(items)['status'] == 'infeasible'
    with pytest.raises(ValueError):
        cents('NaN')
    assert available_funds({'purchase_cap': '30.00', 'periodic_budget': '40.00', 'spent': '10.00', 'reserve': '5.00'})[0] == 2500


def test_b3_native_runtime_repairs_rejected_proposal():
    case, p = payload()
    o = p['catalog']['offers'][1]
    good = {'type': 'propose_plan', 'plan': {'items': [{'offer_id': o['offer_id'], 'quantity': 1,
                                                       'evidence_ids': [o['fact_evidence_id']]}]}}
    bad = deepcopy(good)
    bad['plan']['items'][0]['quantity'] = 2
    client = ScriptedClient([{'requirements': case['requirements']}, bad, good])
    record = respond(p, client=client)
    assert record['model_calls'] == 3
    assert record['validator_enabled'] and record['runtime_status'] == 'completed'
    record.update(config='B3', rep=0)
    assert score(record, case, p['catalog'])['success']
    assert any(e['type'] == 'validation_finished' and e['data']['status'] == 'failed' for e in record['events'])
    assert not client.responses


def test_a1_does_not_retry_after_failed_validation():
    case, p = payload(cfg='A1')
    o = p['catalog']['offers'][1]
    client = ScriptedClient([{'requirements': case['requirements']}, {'type': 'propose_plan', 'plan': {
        'items': [{'offer_id': o['offer_id'], 'quantity': 2, 'evidence_ids': [o['fact_evidence_id']]}]}}])
    record = respond(p, client=client)
    assert record['model_calls'] == 2 and record['outcome']['kind'] == 'error'
    assert record['stop_reason'] == 'plan_repair_limit'


def test_a2_reports_unverified_amounts_and_independent_score_catches_error():
    case, p = payload(cfg='A2')
    o = p['catalog']['offers'][1]
    items = [{'offer_id': o['offer_id'], 'quantity': 1, 'evidence_ids': [o['fact_evidence_id']]}]
    client = ScriptedClient([{'requirements': case['requirements']},
        {'type': 'propose_plan', 'plan': {'items': items}},
        {'outcome': {'kind': 'plan', 'items': items, 'lines': [], 'shipping': '0.00', 'total': '0.00', 'remaining': '80.00'}}])
    record = respond(p, client=client)
    assert not record['validator_enabled'] and record['outcome']['kind'] == 'plan'
    assert record['tool_calls'] == 0
    record.update(config='A2', rep=0)
    assert score(record, case, p['catalog'])['amount_error']


def test_a2_report_may_reorder_items_but_not_change_them():
    case, p = payload(kind='basket', index=1, cfg='A2')
    first, second = p['catalog']['offers'][0], p['catalog']['offers'][3]
    items = [{'offer_id': first['offer_id'], 'quantity': 2, 'evidence_ids': [first['fact_evidence_id']]},
             {'offer_id': second['offer_id'], 'quantity': 2, 'evidence_ids': [second['fact_evidence_id']]}]
    client = ScriptedClient([{'requirements': case['requirements']},
        {'type': 'propose_plan', 'plan': {'items': items}},
        {'outcome': {'kind': 'plan', 'items': list(reversed(items)), 'lines': [],
                     'shipping': '0.00', 'total': '0.00', 'remaining': '0.00'}}])
    record = respond(p, client=client)
    assert record['outcome']['kind'] == 'plan'
    assert 'stop_reason' not in record
    assert record['outcome']['items'] == list(reversed(items))


def test_clarification_is_model_driven_for_agent_configs():
    case, cat = make_case(5, 'chargers', 'single', 'dev')
    p = {'config': 'B3', 'case_id': case['case_id'], 'message': case['message'], 'history': [],
         'catalog': cat, 'constraints': case['initial_constraints'], 'budget_basis': case['budget_basis']}
    client = ScriptedClient([{'requirements': case['requirements']},
        {'type': 'ask_user', 'question': 'What is your purchase cap?', 'required_fields': ['purchase_cap']}])
    record = respond(p, client=client)
    record.update(config='B3', rep=0)
    assert record['model_calls'] == 2 and record['outcome']['kind'] == 'ask_user'
    assert score(record, case, cat)['success']


def test_a2_missing_budget_is_model_driven_without_assessment_tool():
    case, cat = make_case(5, 'chargers', 'single', 'dev')
    p = {'config': 'A2', 'case_id': case['case_id'], 'message': case['message'], 'history': [],
         'catalog': cat, 'constraints': case['initial_constraints'], 'budget_basis': case['budget_basis']}
    client = ScriptedClient([{'requirements': case['requirements']},
        {'type': 'ask_user', 'question': 'What is your purchase cap?', 'required_fields': ['purchase_cap']}])
    record = respond(p, client=client)
    record.update(config='A2', rep=0)
    assert record['tool_calls'] == 0 and record['outcome']['kind'] == 'ask_user'
    assert score(record, case, cat)['success']


def test_a1_missing_budget_maps_to_assessment_fields():
    case, cat = make_case(5, 'chargers', 'single', 'dev')
    p = {'config': 'A1', 'case_id': case['case_id'], 'message': case['message'], 'history': [],
         'catalog': cat, 'constraints': case['initial_constraints'], 'budget_basis': case['budget_basis']}
    client = ScriptedClient([{'requirements': case['requirements']},
        {'type': 'ask_user', 'question': 'What is your purchase cap?', 'required_fields': ['catalog_status']}])
    record = respond(p, client=client)
    record.update(config='A1', rep=0)
    assert record['outcome'] == {'kind': 'ask_user', 'question': 'What is your purchase cap?',
                                 'required_fields': ['purchase_cap']}
    assert score(record, case, cat)['success']


def test_known_costs_with_unknown_attribute_can_prove_over_budget():
    cat = {'catalog_id': 't', 'offers': [
        {'offer_id': 'o0', 'category': 'c', 'currency': 'USD', 'price': '10.00', 'shipping': '0.00',
         'shipping_group': 'g', 'tax': '1.00', 'attributes': {'a': None}, 'fact_evidence_id': 'e0'},
        {'offer_id': 'o1', 'category': 'c', 'currency': 'USD', 'price': None, 'shipping': '0.00',
         'shipping_group': 'g', 'tax': None, 'attributes': {'a': False}, 'fact_evidence_id': 'e1',
         'stale': True}],
        'evidence': [{'evidence_id': 'e0', 'offer_id': 'o0', 'text': 'x'},
                     {'evidence_id': 'e1', 'offer_id': 'o1', 'text': 'x'}]}
    req = [{'category': 'c', 'quantity': 2, 'attributes': {'a': True}}]
    cons = {'currency': 'USD', 'purchase_cap': '20.00'}
    checker = CatalogValidator(cat, req, cons, 'all_in')
    item = [{'offer_id': 'o0', 'quantity': 2, 'evidence_ids': ['e0']}]
    assert checker.check(item)['status'] == 'infeasible'
    assert checker.assess()['kind'] == 'no_feasible_plan'
    assert truth({'constraints': cons, 'requirements': req, 'budget_basis': 'all_in'}, cat)['expected'] == 'no_feasible_plan'


def test_native_catalog_assessment_is_not_arbitrary_abstention():
    case, p = payload(kind='uncertainty', index=3)
    client = ScriptedClient([{'requirements': case['requirements']},
        {'type': 'call_tool', 'tool_name': 'assess_catalog', 'arguments': {}},
        {'type': 'ask_user', 'question': 'Shipping cannot be verified.', 'required_fields': ['catalog_status']}])
    record = respond(p, client=client)
    record.update(config='B3', rep=0)
    assert record['outcome']['kind'] == 'insufficient_evidence'
    assert score(record, case, p['catalog'])['success']


def test_wrong_extracted_requirement_is_still_a_task_failure():
    case, p = payload()
    o = p['catalog']['offers'][0]
    client = ScriptedClient([{'requirements': [{'category': 'earbuds', 'quantity': 1, 'attributes': {}}]},
        {'type': 'propose_plan', 'plan': {'items': [{'offer_id': o['offer_id'], 'quantity': 1,
                                                   'evidence_ids': [o['fact_evidence_id']]}]}}])
    record = respond(p, client=client)
    assert record['runtime_status'] == 'completed'
    record.update(config='B3', rep=0)
    assert not score(record, case, p['catalog'])['success']
