"""Build controlled fixtures; the fifth category is released only after code freeze."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'evaluations/v1/data'
CATEGORIES = ['earbuds', 'chargers', 'keyboards', 'backpacks']
ATTRS = {'earbuds': ('anc', True), 'chargers': ('power_w', 65),
         'keyboards': ('mechanical', True), 'backpacks': ('capacity_l', 25), 'desk_lamps': ('dimmable', True)}
TYPES = ['single'] * 12 + ['basket'] * 8 + ['ledger'] * 6 + ['dialogue'] * 8 + ['uncertainty'] * 6
random.Random(17).shuffle(TYPES)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def fingerprint():
    files = sorted(p for folder in ('agent', 'budget', 'inference', 'knowledge', 'configs', 'evaluations')
                   for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml', '.sql')
                   and not any(x in p.parts for x in ('results', '__pycache__', '.venv')))
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def git_state():
    """Commit and dirty state for provenance; None outside a git checkout."""
    try:
        commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                         text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'],
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return {'commit': None, 'dirty': None, 'status': None}
    return {'commit': commit or None, 'dirty': bool(status), 'status': status.splitlines()[:100]}


def make_case(index, category, kind, split='heldout'):
    lang = 'zh' if index % 2 == 0 else 'en'
    key, val = ATTRS[category]
    cid = f'v1-{split}-{index:02d}'
    cats = [category, CATEGORIES[(index + 1) % 4]] if kind == 'basket' else [category]
    if len(cats) > 1 and len(set(cats)) == 1:
        cats[1] = CATEGORIES[(CATEGORIES.index(category) + 1) % 4]
    offers, evidence = [], []
    for cat in cats:
        attr, value = ATTRS[cat]
        for n in range(3):
            oid = f'{cid}-{cat}-{n}'
            eid = f'ev-{oid}'
            o = {'offer_id': oid, 'category': cat, 'currency': 'USD', 'price': str(10 + n * 5) + '.00',
                 'shipping': '2.00', 'shipping_group': 'merchant-A', 'tax': '1.00',
                 'attributes': {attr: value if n else None}, 'fact_evidence_id': eid}
            if kind == 'uncertainty':
                if index % 3 == 0:
                    o['shipping'] = None
                elif index % 3 == 1:
                    o['stale'] = True
                else:
                    o['conflict'] = True
            offers.append(o)
            evidence.append({'evidence_id': eid, 'offer_id': oid, 'text': json.dumps(o, ensure_ascii=False),
                             'source': 'synthetic://controlled-fixture'})
    basis = 'all_in' if (index // 2) % 2 else 'pre_tax'
    # Explicit clarification cases: no budget source is supplied in the final
    # constraints, so the only correct outcome is ask_user(purchase_cap).
    clarify = kind == 'single' and index in (5, 25)
    if clarify:
        constraints = {'currency': 'USD'}
    else:
        constraints = {'currency': 'USD', 'purchase_cap': '80.00'}
        if kind == 'single':
            if index % 4 == 0:
                # One cent below the cheapest pre-tax total (10.00 + 2.00).
                constraints['purchase_cap'] = '11.99'
            elif index % 4 == 1:
                # Exact boundary for offer 1: 15.00 + tax 1.00 + shipping 2.00.
                constraints['purchase_cap'] = '17.00' if basis == 'pre_tax' else '18.00'
            elif index % 4 == 2:
                constraints['purchase_cap'] = '17.00' if basis == 'pre_tax' else '18.00'
        if kind == 'ledger':
            constraints.update(periodic_budget='100.00', spent='50.00', reserve='20.00')
            if index % 2:
                constraints['purchase_cap'] = '20.00'
    quantity = 2 if kind == 'basket' and index % 2 else 1
    requirements = [{'category': cat, 'quantity': quantity, 'attributes': dict([ATTRS[cat]])} for cat in cats]
    detail = ', '.join(f'{r["quantity"]} x {r["category"]} ({r["attributes"]})' for r in requirements)
    if clarify:
        message = (f'请推荐 {detail}，我还没有确定 purchase_cap，请先询问。' if lang == 'zh' else
                   f'Please recommend {detail}; I have not decided my purchase_cap yet.')
    else:
        message = (f'请推荐 {detail}，预算条件为 {constraints}，请列出费用、余额和证据。' if lang == 'zh' else
                   f'Choose {detail}. Budget: {constraints}. Itemize costs and remaining funds with evidence.')
    turns = []
    if kind == 'dialogue':
        if index % 2:
            initial = {k: v for k, v in constraints.items() if k != 'purchase_cap'}
            message = f'Choose {detail}; ask me for my purchase cap.' if lang == 'en' else f'推荐 {detail}，请先询问购买预算。'
            turns = [{'trigger_fields': ['purchase_cap'], 'answer': {'purchase_cap': '80.00'}}]
        else:
            initial = constraints.copy()
            message = f'请推荐 {detail}，预算80美元。' if lang == 'zh' else f'Choose {detail}, budget 80 USD.'
            turns = [{'revision': '预算改成30美元，保留其余要求。' if lang == 'zh' else 'Change the cap to 30 USD; keep other requirements.',
                      'answer': {'purchase_cap': '30.00'}}]
            constraints['purchase_cap'] = '30.00'
    else:
        initial = constraints.copy()
    case = {'schema_version': 1, 'case_id': cid, 'split': split, 'category': category, 'task_type': kind,
            'language': lang, 'budget_basis': basis, 'message': message,
            'initial_constraints': initial, 'constraints': constraints, 'requirements': requirements,
            'script': turns, 'catalog_id': cid, 'synthetic': True}
    return case, {'catalog_id': cid, 'synthetic': True, 'offers': offers, 'evidence': evidence}


def build(out, fifth=False):
    rows, catalogs = [], {}
    for i in range(10):
        c, cat = make_case(i, CATEGORIES[i % 4], ['single', 'basket', 'ledger', 'dialogue', 'uncertainty'][i % 5], 'dev')
        rows.append(c)
        catalogs[c['catalog_id']] = cat
    for i in range(40 if fifth else 32):
        category = CATEGORIES[i // 8] if i < 32 else 'desk_lamps'
        c, cat = make_case(i, category, TYPES[i])
        rows.append(c)
        catalogs[c['catalog_id']] = cat
    out.mkdir(parents=True, exist_ok=True)
    (out / 'cases.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    dump(out / 'catalogs.json', catalogs)
    dump(out / 'manifest.json', {'version': 1, 'synthetic': True, 'status': 'complete' if fifth else 'awaiting_post_freeze_category',
         'dev': 10, 'heldout': 40 if fifth else 32, 'pending': 0 if fifth else 8,
         'hashes': {name: digest(out / name) for name in ('cases.jsonl', 'catalogs.json')}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['build', 'freeze', 'release-fifth', 'verify'])
    p.add_argument('--data', type=Path, default=BASE)
    a = p.parse_args()
    freeze = a.data / 'freeze.json'
    if a.action == 'build':
        if (a.data / 'manifest.json').exists():
            p.error('dataset exists; choose a new --data directory')
        build(a.data)
    elif a.action == 'freeze':
        if freeze.exists():
            p.error('freeze already exists')
        dump(freeze, {'code': fingerprint(), 'pre_release_manifest': digest(a.data / 'manifest.json'),
                      'git': git_state()})
    elif a.action == 'release-fifth':
        f = json.loads(freeze.read_text(encoding='utf-8'))
        if f['code'] != fingerprint() or f['pre_release_manifest'] != digest(a.data / 'manifest.json'):
            p.error('code or dataset changed after freeze')
        build(a.data, fifth=True)
        dump(a.data / 'release.json', {'freeze_sha256': digest(freeze), 'manifest_sha256': digest(a.data / 'manifest.json')})
    else:
        m = json.loads((a.data / 'manifest.json').read_text(encoding='utf-8'))
        if any(digest(a.data / name) != sha for name, sha in m['hashes'].items()):
            p.error('dataset hash mismatch')
        print(json.dumps(m, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
