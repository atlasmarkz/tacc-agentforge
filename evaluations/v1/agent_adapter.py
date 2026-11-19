"""Real AgentRuntime integration for the normalized v1 fixture catalog.

This is not the SerpApi KB track. Intent is extracted by the deployed model from
user/history; reference requirements and the independent oracle are never read.
B3 uses the business validator; A1 permits one planning decision; A2 records an
unchecked proposal and asks the model to report its own amounts.
"""
import asyncio
import json
import os
import time
from uuid import uuid4

from pydantic import Field, JsonValue
from agent import (AgentDependencies, AgentRequest, AgentRuntime, ModelReply, RunPolicy,
                   ToolRegistry, ToolSpec)
from agent.contracts import Schema
from budget.validator import CatalogValidator, AgentPlanValidator
from inference.client import ModelClient
from inference.model_port import VllmModelPort, decoder_schema
from .model_adapter import context_for, REPORT_PROMPT, REPORT_SCHEMA


class Requirement(Schema):
    category: str = Field(min_length=1, max_length=100)
    quantity: int = Field(strict=True, ge=1, le=100)
    attributes: dict[str, bool | int | float | str] = Field(default_factory=dict)


class Intent(Schema):
    requirements: list[Requirement] = Field(min_length=1, max_length=3)


class AssessmentArguments(Schema):
    pass


class AssessmentResult(Schema):
    result: dict[str, JsonValue]


class AssessmentTool:
    def __init__(self, checker):
        self.checker, self.last = checker, None

    async def execute(self, arguments, context):
        self.last = self.checker.assess()
        return AssessmentResult(result=self.last)


class MeasuredPort:
    def __init__(self, inner):
        self.inner, self.calls = inner, []

    async def generate(self, request, context):
        start = time.monotonic()
        attempt = {'prompt_tokens': None, 'completion_tokens': None, 'messages': [m.model_dump() for m in request.messages]}
        self.calls.append(attempt)
        reply = await self.inner.generate(request, context)
        attempt.update({'prompt_tokens': reply.prompt_tokens, 'completion_tokens': reply.completion_tokens,
                           'duration_s': time.monotonic() - start, 'finish_reason': reply.finish_reason,
                           'messages': [m.model_dump() for m in request.messages], 'reply': reply.content})
        return reply


def _chat(client, messages, calls, schema=None):
    start = time.monotonic()
    attempt = {'prompt_tokens': None, 'completion_tokens': None, 'messages': messages}
    calls.append(attempt)
    result = client.chat(messages, max_tokens=1024, temperature=0, timeout_s=90,
                         enable_thinking=False, on_delta=lambda *_: None, json_schema=decoder_schema(schema) if schema else None)
    attempt.update({'prompt_tokens': result.prompt_tokens, 'completion_tokens': result.completion_tokens,
                  'duration_s': time.monotonic() - start, 'ttft_s': result.first_token_s,
                  'http_attempts': result.attempts, 'finish_reason': result.finish_reason,
                  'messages': messages, 'reply': result.content})
    if result.finish_reason != 'stop':
        raise ValueError('incomplete model generation')
    return result.content


def extract_intent(client, payload, calls):
    text = _chat(client, [
        {'role': 'system', 'content': 'Extract only the shopper\'s requirements from the conversation. '
            'Return raw JSON {"requirements":[{"category":"...","quantity":1,"attributes":{"name":value}}]}. '
            'Use category and attribute names present in the catalog, preserve prior requirements on revisions. '
            'Do not infer a product requirement merely because candidates have it. Do not choose products. '
            'Use exact boolean/numeric/string types. Do not copy requirements from assistant recommendations.'},
        {'role': 'user', 'content': json.dumps({'message': payload['message'], 'history': payload['history'],
             'schema': {o['category']: {k: type(v).__name__ for k, v in o['attributes'].items() if v is not None}
                        for o in payload['catalog']['offers']}}, ensure_ascii=False)}], calls, Intent.model_json_schema())
    return Intent.model_validate_json(text).model_dump(mode='json')['requirements']


def plan_outcome(plan):
    return {'kind': 'plan', 'items': [i.model_dump(mode='json') for i in plan.items],
            'lines': [{'offer_id': i.offer_id, 'quantity': i.quantity,
                       'subtotal': plan.cost_breakdown[f'line_{n}']} for n, i in enumerate(plan.items)],
            'shipping': plan.cost_breakdown['shipping'], 'total': plan.total_cost, 'remaining': plan.remaining_funds}


def same_item_set(actual, expected):
    """A2's report stage may reorder items or reformat evidence, not change items."""
    def key(item):
        return (item.get('offer_id'), item.get('quantity')) if isinstance(item, dict) else None
    return (isinstance(actual, list) and
            sorted((key(i) for i in actual), key=repr) ==
            sorted((key(i) for i in expected), key=repr))


async def run_agent(client, payload, requirements):
    config = payload['config']
    checker = CatalogValidator(payload['catalog'], requirements, payload['constraints'], payload['budget_basis'])
    assessment = AssessmentTool(checker)
    validator = AgentPlanValidator(checker, disabled=config == 'A2')
    one_pass = config == 'A1'
    use_assessment = config != 'A2'
    common, context_ids = context_for({**payload, 'config': config})
    data = {**common, 'parsed_requirements': requirements,
            'adapter_protocol': 'Use assess_catalog to check the finite catalog when necessary. '
                'If a required budget field is missing, call assess_catalog and ask_user with its required_fields. '
                'If no feasible plan or evidence is insufficient, call assess_catalog then '
                'ask_user with required_fields=["catalog_status"] to report the conclusion. '
                'Otherwise propose a plan and let the validator calculate costs. '
                'Treat parsed requirements as an extraction to follow, not a proven reference answer.'}
    if one_pass:
        assessment.last = checker.assess()
        data['assessment'] = assessment.last
        data['adapter_protocol'] = 'This is a fixed pipeline. Do not call tools. Propose one plan; '
        if assessment.last.get('kind') == 'ask_user':
            data['adapter_protocol'] += 'the assessment reports missing required fields; ask_user with the assessment required_fields.'
        else:
            data['adapter_protocol'] += 'if assessment says no solution or unknown, ask_user with required_fields=["catalog_status"].'
    elif not use_assessment:
        # A2 deliberately has no validator (proposal or catalog assessment).
        data['adapter_protocol'] = ('No catalog assessment tool is available. Propose a plan from the catalog '
            'and your own reasoning; do not claim a validator result. If you cannot establish a feasible '
            'plan, use no_feasible_plan or insufficient_evidence with catalog_id and reason_codes.')
    registry = ToolRegistry([] if one_pass or not use_assessment else [ToolSpec('assess_catalog',
        'Assess the finite catalog under the extracted requirements and known budget. Does not select products.',
        AssessmentArguments, AssessmentResult, assessment)])
    port = MeasuredPort(VllmModelPort(client, temperature=0, enable_thinking=False, on_delta=lambda *_: None))
    runtime = AgentRuntime(AgentDependencies(port, registry, validator), RunPolicy(
        max_model_calls=3 if one_pass else 8, max_tool_calls=0 if one_pass or not use_assessment else 8,
        max_plan_repairs=0 if one_pass else 2, max_context_chars=60000, max_output_tokens=1024,
        deadline_seconds=110, call_timeout_s=90))
    run_id = 'bench-' + uuid4().hex
    request = AgentRequest(run_id=run_id, conversation_id=run_id,
        snapshot_id=payload['catalog']['catalog_id'], message=payload['message'] + '\n' + data['adapter_protocol'],
        constraints={**payload['constraints'], 'requirements': requirements, 'budget_basis': payload['budget_basis'], 'benchmark_context': data})
    events, result = [], None
    async for event in runtime.astream(request):
        events.append({'type': event.type, 'node': event.node, 'data': event.data})
        if event.result:
            result = event.result
    if result is None:
        return {'outcome': {'kind': 'error'}, 'stop_reason': 'missing_result'}, port.calls
    record = {'context_ids': context_ids, 'events': events, 'runtime_status': result.status,
              'validator_enabled': not validator.disabled, 'tool_calls': result.usage.tool_calls}
    if result.verified_plan:
        record['outcome'] = plan_outcome(result.verified_plan)
    elif validator.disabled and validator.proposals and result.stop_reason == 'validation_unavailable':
        # This is explicitly an unverified delivery. It is still independently scored.
        record['unchecked_proposal'] = validator.proposals[-1]
    elif result.clarification:
        fields = list(result.clarification.required_fields)
        if fields == ['catalog_status'] and assessment.last and assessment.last.get('kind') == 'ask_user':
            record['outcome'] = {'kind': 'ask_user', 'question': result.clarification.question,
                                 'required_fields': list(assessment.last['required_fields'])}
        elif fields == ['catalog_status'] and assessment.last and assessment.last['kind'] in ('no_feasible_plan', 'insufficient_evidence'):
            record['outcome'] = {k: assessment.last[k] for k in ('kind', 'catalog_id', 'reason_codes')}
        else:
            record['outcome'] = {'kind': 'ask_user', 'question': result.clarification.question,
                                 'required_fields': fields}
    else:
        record.update(outcome={'kind': 'error'}, stop_reason=result.stop_reason or result.status)
    return record, port.calls


def respond(payload, client=None):
    if payload['config'] in ('B0', 'B1', 'B2'):
        from .model_adapter import respond as single_call
        return single_call(payload, client=client)
    if payload['config'] not in ('B3', 'A1', 'A2'):
        raise ValueError('unknown configuration')
    client = client or ModelClient(base_url=os.environ.get('BENCHMARK_BASE_URL'),
        address_file=os.environ.get('BENCHMARK_ADDRESS_FILE'), model=os.environ.get('BENCHMARK_MODEL', 'qwen3.8-27b'))
    if getattr(client, 'address_file', None) is not None:
        from pathlib import Path
        client.address_file = Path(client.address_file)
    calls = []
    record = {}
    try:
        requirements = extract_intent(client, payload, calls)
        record, agent_calls = asyncio.run(run_agent(client, payload, requirements))
        calls.extend(agent_calls)
        record['parsed_requirements'] = requirements
        if 'unchecked_proposal' in record:
            common, _ = context_for({**payload, 'config': payload['config']})
            text = _chat(client, [{'role': 'system', 'content': REPORT_PROMPT +
                '\nReport the unchecked proposed items exactly, including the quantities. Calculate amounts yourself.'},
                {'role': 'user', 'content': json.dumps({'context': common, 'proposal': record['unchecked_proposal']}, ensure_ascii=False)}], calls, REPORT_SCHEMA)
            value = json.loads(text)
            outcome = value.get('outcome') if isinstance(value, dict) else None
            if not isinstance(outcome, dict):
                raise ValueError('expected outcome object')
            record['outcome'] = outcome
            if outcome.get('kind') != 'plan' or not same_item_set(
                    outcome.get('items'), record['unchecked_proposal']['items']):
                record['stop_reason'] = 'invalid_action'
    except Exception as exc:
        record.update(outcome={'kind': 'error'}, error=type(exc).__name__, error_detail=str(exc)[:500])
    if record.get('stop_reason') == 'invalid_action':
        record['invalid_actions'] = 1
    record.update(model_calls=len(calls), model_trace=calls,
                  prompt_tokens=sum(c.get('prompt_tokens') or 0 for c in calls) if all(c.get('prompt_tokens') is not None for c in calls) else None,
                  completion_tokens=sum(c.get('completion_tokens') or 0 for c in calls) if all(c.get('completion_tokens') is not None for c in calls) else None)
    return record
