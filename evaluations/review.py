"""Blind human-review export and two-rater agreement; never invent annotations.

Human labels stay separate from deterministic scores. Unfilled/incomplete labels
are rejected by score, rather than being silently interpreted as bad or good.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import secrets


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def export(records, cases, catalogs, out, seed=17):
    if out.exists():
        raise ValueError('use a new review output directory')
    by_case = {c['case_id']: c for c in cases}
    rows, key = [], []
    # Use the first repeat only for the course-scale human preference review.
    first = {}
    for rec in records:
        k = (rec['config'], rec['case_id'])
        if k not in first or rec['rep'] < first[k]['rep']:
            first[k] = rec
    for (cfg, cid), rec in sorted(first.items()):
        c = by_case[cid]
        blind_id = secrets.token_hex(8)
        rows.append({'review_id': blind_id, 'user_message': c['message'], 'script': c['script'],
            'outcome': rec.get('outcome'), 'context': catalogs[c['catalog_id']],
            'label': {'reviewer': None, 'preference_fit': None, 'not_applicable_reason': None,
                'claims_complete': None, 'claims': [], 'notes': ''}})
        key.append({'review_id': blind_id, 'config': cfg, 'case_id': cid, 'rep': rec['rep']})
    random.Random(seed).shuffle(rows)
    out.mkdir(parents=True)
    write_jsonl(out / 'reviewer-a.jsonl', rows)
    write_jsonl(out / 'reviewer-b.jsonl', rows)
    write_jsonl(out / 'private-key.jsonl', key)
    (out / 'rubric.md').write_text('''# Human review rubric
Do not consult private-key.jsonl or the other reviewer before submitting labels.
Set reviewer to a stable, distinct ID. preference_fit: 1=unsuitable, 2=major gaps,
3=main preferences met with incomplete explanation, 4=good fit and justified tradeoffs,
5=fit, tradeoffs and costs complete. Use null with not_applicable_reason for an
error/abstention that cannot be preference-rated. Do not compensate budget failures
with preference scores. List every factual claim, set claims_complete=true only
after checking completeness. Each claim needs text, supported (boolean),
requires_citation (boolean), cited (boolean), citation_supported (boolean).
A resolvable citation ID is not sufficient proof of claim support. Assess against
the supplied frozen facts, not today's prices. Empty complete claim lists yield
N/A, not perfect faithfulness. Preserve disagreement until both files are signed.
''', encoding='utf-8')


def weighted_kappa(pairs):
    if not pairs:
        return None
    a, b = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    n = len(pairs)
    observed = sum((x - y) ** 2 for x, y in pairs) / (16 * n)
    expected = sum(a[x] * b[y] * (x - y) ** 2 for x in a for y in b) / (16 * n * n)
    return 1 - observed / expected if expected else None


def validate(rows, expected_ids):
    labels, reviewers = {}, set()
    for row in rows:
        rid, label = row['review_id'], row['label']
        if rid in labels or rid not in expected_ids:
            raise ValueError('duplicate or unknown review ID')
        reviewer = label.get('reviewer')
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError('each label needs a reviewer ID')
        reviewers.add(reviewer)
        rating = label.get('preference_fit')
        if rating is not None and (type(rating) is not int or not 1 <= rating <= 5):
            raise ValueError('preference_fit must be integer 1..5 or N/A')
        if rating is None and not label.get('not_applicable_reason'):
            raise ValueError('unrated preference needs N/A reason')
        if label.get('claims_complete') is not True:
            raise ValueError('claim completeness must be reviewed')
        if not isinstance(label.get('claims'), list):
            raise ValueError('claims must be a list')
        for claim in label['claims']:
            if not claim.get('text') or any(type(claim.get(k)) is not bool for k in
                    ('supported', 'requires_citation', 'cited', 'citation_supported')):
                raise ValueError('each claim needs text and explicit boolean judgments')
            if claim['citation_supported'] and not claim['cited']:
                raise ValueError('uncited claim cannot have supported citation')
        labels[rid] = label
    if len(reviewers) != 1 or set(labels) != expected_ids:
        raise ValueError('one reviewer and the complete review roster required per file')
    return labels, next(iter(reviewers))


def summarize(a_rows, b_rows, key_rows):
    keys = {r['review_id']: r for r in key_rows}
    a, aid = validate(a_rows, set(keys))
    b, bid = validate(b_rows, set(keys))
    if aid == bid:
        raise ValueError('two distinct reviewers required')
    pairs = [(a[k]['preference_fit'], b[k]['preference_fit']) for k in keys
             if a[k]['preference_fit'] is not None and b[k]['preference_fit'] is not None]
    report = {'reviewers': [aid, bid], 'reviewed_outputs': len(keys), 'paired_preference_outputs': len(pairs),
              'quadratic_weighted_kappa': weighted_kappa(pairs), 'by_config': {}}
    for cfg in sorted({k['config'] for k in keys.values()}):
        metrics = {}
        for reviewer, labels in ((aid, a), (bid, b)):
            selected = [labels[k] for k in keys if keys[k]['config'] == cfg]
            ratings = [l['preference_fit'] for l in selected if l['preference_fit'] is not None]
            claims = [c for l in selected for c in l['claims']]
            cited = [c for c in claims if c['cited']]
            required = [c for c in claims if c['requires_citation']]
            def rate(sub, pred):
                n, m = sum(pred(c) for c in sub), len(sub)
                return {'n': n, 'm': m, 'rate': n / m if m else None}
            metrics[reviewer] = {'preference_distribution': dict(Counter(ratings)),
                'preference_mean': sum(ratings) / len(ratings) if ratings else None,
                'faithfulness_micro': rate(claims, lambda c: c['supported']),
                'citation_support': rate(cited, lambda c: c['citation_supported']),
                'citation_coverage': rate(required, lambda c: c['cited'] and c['citation_supported'])}
        report['by_config'][cfg] = metrics
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['export', 'score'])
    p.add_argument('--records', type=Path)
    p.add_argument('--data', type=Path, default=Path('evaluations/v1/data'))
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--a', type=Path)
    p.add_argument('--b', type=Path)
    p.add_argument('--key', type=Path)
    args = p.parse_args()
    if args.action == 'export':
        if not args.records:
            p.error('--records required')
        export(read_jsonl(args.records), read_jsonl(args.data / 'cases.jsonl'),
               json.loads((args.data / 'catalogs.json').read_text(encoding='utf-8')), args.out)
    else:
        if not args.a or not args.b or not args.key:
            p.error('--a, --b, --key required')
        report = summarize(read_jsonl(args.a), read_jsonl(args.b), read_jsonl(args.key))
        if args.out.exists():
            p.error('output exists')
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
