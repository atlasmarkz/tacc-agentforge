"""Business-side checker using integer cents, independent of evaluations/oracle.

Inputs are normalized observations. Unknown fees are never filled with zero.
Requirements are extracted by the agent, not read from evaluation ground truth.
"""
import re
from itertools import product


def cents(value):
    if not isinstance(value, str) or not re.fullmatch(r'(0|[1-9][0-9]*)(\.[0-9]{1,2})?', value):
        raise ValueError('expected nonnegative decimal amount with at most two fractional digits')
    whole, _, fraction = value.partition('.')
    return int(whole) * 100 + int(fraction.ljust(2, '0'))


def amount(value):
    return ('-' if value < 0 else '') + f'{abs(value) // 100}.{abs(value) % 100:02d}'


def available_funds(constraints):
    limits, missing = [], []
    if constraints.get('purchase_cap') is not None:
        limits.append(cents(constraints['purchase_cap']))
    if constraints.get('periodic_budget') is not None:
        missing = [key for key in ('spent', 'reserve') if constraints.get(key) is None]
        if not missing:
            limits.append(cents(constraints['periodic_budget']) - cents(constraints['spent']) - cents(constraints['reserve']))
    if not limits and not missing:
        missing = ['purchase_cap']
    return (None if missing else min(limits)), missing


class CatalogValidator:
    """Validate concrete plans and certify a bounded catalog's feasibility.

    A category requirement selects one offer with the specified quantity. This
    contract does not support splitting one category across multiple offers.
    """
    def __init__(self, catalog, requirements, constraints, basis):
        if basis not in ('pre_tax', 'all_in'):
            raise ValueError('unknown budget basis')
        self.catalog, self.requirements = catalog, requirements
        self.constraints, self.basis = constraints, basis
        self.offers = {o['offer_id']: o for o in catalog['offers']}
        if len(self.offers) != len(catalog['offers']):
            raise ValueError('duplicate offer IDs')
        if not requirements or len({r['category'] for r in requirements}) != len(requirements):
            raise ValueError('requirements must contain distinct categories')
        if any(type(r['quantity']) is not int or not 0 < r['quantity'] <= 100 for r in requirements):
            raise ValueError('invalid quantity')

    def check(self, items, require_citations=True):
        failures, unknown = [], []
        funds, missing = available_funds(self.constraints)
        if missing:
            return {'status': 'clarify', 'required_fields': missing, 'reason_codes': []}
        if not items or len({i['offer_id'] for i in items}) != len(items):
            failures.append('empty_or_duplicate_items')
        rows, shipping, known_minimum = [], {}, 0
        minimum_valid = True
        cost_known = True
        selected = []
        evidence = {e['evidence_id']: e for e in self.catalog['evidence']}
        for item in items:
            o, q = self.offers.get(item['offer_id']), item.get('quantity')
            if o is None or type(q) is not int or not 0 < q <= 100:
                failures.append('wrong_category_or_id')
                cost_known = False
                continue
            selected.append((o, q))
            if o['currency'] != self.constraints['currency']:
                unknown.append('currency')
                minimum_valid = False
                cost_known = False
            if require_citations:
                refs = item.get('evidence_ids', [])
                if o['fact_evidence_id'] not in refs or any(e not in evidence or evidence[e]['offer_id'] != o['offer_id'] for e in refs):
                    failures.append('unsupported_citation')
            if o.get('stale') or o.get('conflict'):
                unknown.append('stale' if o.get('stale') else 'conflict')
                minimum_valid = False
                cost_known = False
            if o.get('price') is not None:
                known_minimum += cents(o['price']) * q
            else:
                minimum_valid = False
                cost_known = False
            needed = ['price', 'shipping'] + (['tax'] if self.basis == 'all_in' else [])
            if any(o.get(k) is None for k in needed):
                unknown.append('missing_cost')
                cost_known = False
                continue
            subtotal = (cents(o['price']) + (cents(o['tax']) if self.basis == 'all_in' else 0)) * q
            rows.append({'offer_id': o['offer_id'], 'quantity': q, 'subtotal': amount(subtotal)})
            group, fee = o['shipping_group'], cents(o['shipping'])
            if group in shipping and shipping[group] != fee:
                unknown.append('shipping_conflict')
                cost_known = False
            shipping[group] = fee
        requested = {r['category'] for r in self.requirements}
        if any(o['category'] not in requested for o, _ in selected):
            failures.append('wrong_category_or_id')
        for req in self.requirements:
            matched = [(o, q) for o, q in selected if o['category'] == req['category']]
            if len(matched) != 1 or matched[0][1] != req['quantity']:
                failures.append('quantity_or_category')
            for o, _ in matched:
                for key, expected in req.get('attributes', {}).items():
                    actual = o['attributes'].get(key)
                    if actual is None:
                        unknown.append('attribute:' + key)
                    elif type(actual) is not type(expected) or actual != expected:
                        failures.append('attribute:' + key)
        total = sum(cents(r['subtotal']) for r in rows) + sum(shipping.values())
        # A fully known cost can prove infeasibility even when a non-cost fact
        # (for example an attribute) is still unknown. When any cost component
        # is unknown, fall back to the conservative known item-price bound.
        if (cost_known and total > funds) or (minimum_valid and known_minimum > funds):
            failures.append('over_budget')
        status = 'infeasible' if failures else 'unknown' if unknown else 'feasible'
        result = {'status': status, 'reason_codes': sorted(set(failures + unknown))}
        if not unknown:
            result.update(lines=rows, shipping=amount(sum(shipping.values())), total=amount(total), remaining=amount(funds - total))
        return result

    def assess(self):
        _, missing = available_funds(self.constraints)
        if missing:
            return {'kind': 'ask_user', 'required_fields': missing}
        groups = [[{'offer_id': o['offer_id'], 'quantity': r['quantity']} for o in self.offers.values()
                   if o['category'] == r['category']] for r in self.requirements]
        combinations = 1
        for g in groups:
            combinations *= len(g)
        if combinations > 100000:
            raise ValueError('catalog assessment exceeds bounded search limit')
        feasible, uncertain, reasons = 0, 0, set()
        for items in product(*groups):
            check = self.check(list(items), require_citations=False)
            feasible += check['status'] == 'feasible'
            uncertain += check['status'] == 'unknown'
            reasons.update(check['reason_codes'])
        return {'kind': 'plan' if feasible else 'insufficient_evidence' if uncertain else 'no_feasible_plan',
                'catalog_id': self.catalog['catalog_id'], 'reason_codes': sorted(reasons) or ['no_candidate'],
                'feasible_combinations': feasible, 'unknown_combinations': uncertain}


class AgentPlanValidator:
    """Adapter for the agent.PlanValidator port; business check, never an oracle."""
    def __init__(self, checker, disabled=False):
        self.checker, self.disabled = checker, disabled
        self.proposals = []

    async def validate(self, request, context):
        from agent import Issue, ValidationResult, VerifiedPlan
        self.proposals.append(request.plan.model_dump(mode='json'))
        if self.disabled:
            return ValidationResult(status='unavailable', proposal_fingerprint=request.proposal_fingerprint)
        result = self.checker.check(self.proposals[-1]['items'])
        if result['status'] != 'feasible':
            codes = result.get('reason_codes') or ['input:' + f for f in result.get('required_fields', [])]
            return ValidationResult(status='failed', proposal_fingerprint=request.proposal_fingerprint,
                issues=[Issue(code='invalid_plan', message=', '.join(codes)[:500])])
        plan = VerifiedPlan(snapshot_id=context.snapshot_id, items=request.plan.items,
                            currency=self.checker.constraints['currency'], total_cost=result['total'],
                            remaining_funds=result['remaining'], cost_breakdown={
                                'shipping': result['shipping'],
                                **{f'line_{i}': l['subtotal'] for i, l in enumerate(result['lines'])}})
        return ValidationResult(status='passed', proposal_fingerprint=request.proposal_fingerprint, verified_plan=plan)
