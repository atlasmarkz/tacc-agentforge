"""Independent exhaustive reference evaluator. No production tools or validator imports.

Catalogs in this track are explicitly synthetic frozen fixtures, never live offers.
All amounts are decimal strings, shipping is per order-group, tax per unit.
"""
from decimal import Decimal, InvalidOperation
from itertools import product


def money(value):
    if not isinstance(value, str):
        raise ValueError('amounts must be decimal strings')
    try:
        d = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError('invalid amount') from exc
    if not d.is_finite() or d < 0 or d != d.quantize(Decimal('.01')):
        raise ValueError('amount must be finite, nonnegative and exact to cents')
    return d


def funds(constraints):
    missing = []
    cap = constraints.get('purchase_cap')
    period = constraints.get('periodic_budget')
    if cap is None and period is None:
        return None, ['purchase_cap']
    limits = [money(cap)] if cap is not None else []
    if period is not None:
        missing = [k for k in ('spent', 'reserve') if constraints.get(k) is None]
        if not missing:
            limits.append(money(period) - money(constraints['spent']) - money(constraints['reserve']))
    return (None, missing) if missing else (min(limits), [])


def total(items, catalog, case):
    offers = {o['offer_id']: o for o in catalog['offers']}
    lines, shipping, errors = [], {}, []
    for item in items:
        o = offers.get(item.get('offer_id'))
        q = item.get('quantity')
        if o is None or type(q) is not int or q < 1 or q > 100:
            return None, ['invalid_item']
        if o['currency'] != case['constraints']['currency']:
            return None, ['currency']
        if o.get('conflict') or o.get('stale'):
            errors.append('conflict' if o.get('conflict') else 'stale')
        amounts = [o.get('price'), o.get('shipping')]
        if case['budget_basis'] == 'all_in':
            amounts.append(o.get('tax'))
        if any(v is None for v in amounts):
            errors.append('missing_cost')
            continue
        tax = money(o['tax']) if case['budget_basis'] == 'all_in' else Decimal(0)
        lines.append({'offer_id': o['offer_id'], 'quantity': q,
                      'subtotal': f'{(money(o["price"]) + tax) * q:.2f}'})
        group = o['shipping_group']
        cost = money(o['shipping'])
        if group in shipping and shipping[group] != cost:
            errors.append('shipping_conflict')
        shipping[group] = cost
    if errors:
        return None, sorted(set(errors))
    amount = sum((money(l['subtotal']) for l in lines), Decimal(0)) + sum(shipping.values(), Decimal(0))
    return {'lines': lines, 'shipping': f'{sum(shipping.values(), Decimal(0)):.2f}',
            'total': f'{amount:.2f}'}, []


def inspect(items, case, catalog):
    offers = {o['offer_id']: o for o in catalog['offers']}
    fail, unknown = [], []
    if not items or len({i.get('offer_id') for i in items}) != len(items):
        fail.append('empty_or_duplicate_items')
    categories = {r['category'] for r in case['requirements']}
    if any(i.get('offer_id') not in offers or offers[i['offer_id']]['category'] not in categories for i in items):
        fail.append('wrong_category_or_id')
    for req in case['requirements']:
        chosen = [i for i in items if i.get('offer_id') in offers and offers[i['offer_id']]['category'] == req['category']]
        if len(chosen) != 1 or chosen[0].get('quantity') != req['quantity']:
            fail.append('quantity_or_category')
        for i in chosen:
            attrs = offers[i['offer_id']]['attributes']
            for key, want in req.get('attributes', {}).items():
                if attrs.get(key) is None:
                    unknown.append('attribute:' + key)
                elif type(attrs[key]) is not type(want) or attrs[key] != want:
                    fail.append('attribute:' + key)
    costs, cost_errors = total(items, catalog, case)
    available, missing = funds(case['constraints'])
    unknown.extend(cost_errors + ['input:' + k for k in missing])
    lower_bound = Decimal(0)
    bound_known = True
    for item in items:
        o = offers.get(item.get('offer_id'))
        q = item.get('quantity')
        if (o is None or type(q) is not int or q < 1 or o.get('price') is None
                or o.get('conflict') or o.get('stale') or o.get('currency') != case['constraints']['currency']):
            bound_known = False
            break
        lower_bound += money(o['price']) * q
    over = bool(available is not None and (
        (costs is not None and money(costs['total']) > available) or
        (costs is None and bound_known and lower_bound > available)))
    if over:
        fail.append('over_budget')
    return {'status': 'infeasible' if fail else 'unknown' if unknown else 'feasible',
            'reasons': sorted(set(fail + unknown)), 'failures': sorted(set(fail)), 'unknowns': sorted(set(unknown)), 'costs': costs, 'over_budget': over,
            'remaining': f'{available - money(costs["total"]):.2f}' if costs and available is not None else None}


def truth(case, catalog):
    _, missing = funds(case['constraints'])
    if missing:
        return {'expected': 'clarify', 'required_fields': missing}
    choices = [[{'offer_id': o['offer_id'], 'quantity': r['quantity']} for o in catalog['offers']
                if o['category'] == r['category']] for r in case['requirements']]
    feasible, unknown, reasons = [], [], set()
    for combo in product(*choices):
        verdict = inspect(list(combo), case, catalog)
        reasons.update(verdict['reasons'])
        if verdict['status'] == 'feasible':
            feasible.append({'items': list(combo), **verdict['costs'], 'remaining': verdict['remaining']})
        elif verdict['status'] == 'unknown':
            unknown.append(list(combo))
    return {'expected': 'plan' if feasible else 'insufficient_evidence' if unknown else 'no_feasible_plan',
            'feasible': feasible, 'unknown_combinations': len(unknown), 'reasons': sorted(reasons) or ['no_candidate']}


def score(record, case, catalog):
    gt = truth(case, catalog)
    out = record.get('outcome')
    if not isinstance(out, dict):
        out = {'kind': 'error'}
    kind = out.get('kind')
    context = record.get('context_ids')
    context = context if isinstance(context, list) else None
    result = {'case_id': case['case_id'], 'category': case['category'], 'expected': gt['expected'],
              'config': record['config'], 'rep': record['rep'], 'kind': kind, 'success': False,
              'reason': 'wrong_outcome', 'plan_delivered': kind == 'plan', 'plan_total': None,
              'over_budget': False, 'amount_error': None, 'wall_s': record.get('wall_s'),
              'violates': None, 'unverifiable': None, 'ungrounded': None, 'evidence_mismatch': None,
              'context_checked': context is not None, 'ids_not_in_context': None}
    if kind == 'plan':
        try:
            j = inspect(out.get('items', []), case, catalog)
            result.update(plan_total=j['costs']['total'] if j['costs'] else None, over_budget=j['over_budget'],
                          violates=bool(set(j['failures']) - {'over_budget'}), unverifiable=bool(j['unknowns']))
            def same_amount(a, b):
                if not isinstance(a, str) or not isinstance(b, str):
                    return False
                try:
                    actual, expected = Decimal(a), Decimal(b)
                    return actual.is_finite() and actual == expected
                except InvalidOperation:
                    return False
            lines = out.get('lines')
            line_map = {(l['offer_id'], l['quantity']): l['subtotal'] for l in lines} if isinstance(lines, list) else {}
            amounts_ok = bool(j['costs']) and all(same_amount(out.get(k), v) for k, v in {
                'total': j['costs']['total'], 'shipping': j['costs']['shipping'],
                'remaining': j['remaining']}.items()) and isinstance(lines, list) and len(line_map) == len(lines) == len(j['costs']['lines']) and all(
                    same_amount(line_map.get((l['offer_id'], l['quantity'])), l['subtotal']) for l in j['costs']['lines'])
            result['amount_error'] = (not amounts_ok) if j['costs'] else None
            evidence = {e['evidence_id']: e for e in catalog['evidence']}
            by_offer = {o['offer_id']: o for o in catalog['offers']}
            cited_ok = all(i.get('evidence_ids') and all(
                e in evidence and evidence[e]['offer_id'] == i['offer_id'] and
                (context is None or e in context) for e in i['evidence_ids']) and
                by_offer[i['offer_id']]['fact_evidence_id'] in i['evidence_ids']
                for i in out.get('items', []) if i['offer_id'] in by_offer)
            refs = [(i['offer_id'], e) for i in out.get('items', []) for e in i.get('evidence_ids', [])]
            result['ungrounded'] = any(i['offer_id'] not in by_offer for i in out.get('items', [])) or any(e not in evidence for _, e in refs)
            result['evidence_mismatch'] = any(e in evidence and evidence[e]['offer_id'] != oid for oid, e in refs)
            result['ids_not_in_context'] = None if context is None else any(e not in context for _, e in refs)
            result['success'] = gt['expected'] == 'plan' and j['status'] == 'feasible' and amounts_ok and cited_ok
            result['reason'] = 'ok' if result['success'] else 'invalid_or_unsupported_plan'
        except (ValueError, KeyError, TypeError, AttributeError):
            result['reason'] = 'malformed_plan'
    elif gt['expected'] == 'clarify' and kind == 'ask_user':
        result['success'] = isinstance(out.get('required_fields'), list) and all(isinstance(f, str) for f in out['required_fields']) and set(gt['required_fields']) == set(out['required_fields'])
    elif gt['expected'] in ('no_feasible_plan', 'insufficient_evidence') and kind == gt['expected']:
        result['success'] = out.get('catalog_id') == catalog['catalog_id'] and isinstance(out.get('reason_codes'), list) and bool(out['reason_codes']) and all(isinstance(r, str) for r in out['reason_codes']) and set(out['reason_codes']) <= set(gt['reasons'])
    if result['success']:
        result['reason'] = 'ok'
    if record.get('error') or record.get('stop_reason') in ('timeout', 'limit', 'invalid_action'):
        result.update(success=False, reason=record.get('stop_reason') or 'error')
    return result
