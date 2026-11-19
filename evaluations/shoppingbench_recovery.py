"""Objective API-failure recovery; never select episodes by reward or correctness."""
import hashlib
import json
from pathlib import Path


def api_complete(rollout):
    if not rollout:
        return False
    for step in rollout:
        if not isinstance(step, dict):
            return False
        completion = step.get('completion')
        if not isinstance(completion, dict):
            return False
        if completion.get('error') or completion.get('finish_reason') == 'error':
            return False
        if not any(isinstance(completion.get(k), str) and bool(completion[k])
                   for k in ('content', 'reasoning_content')):
            return False
    return True


def prepare(tasks, prior, destination):
    """Seed a new official rollout file with API-complete episodes only.

    The unmodified upstream producer skips those queries and runs the remaining
    original tasks. Wrong recommendations with nonempty model responses stay in
    the file; there is no reward-based retry. The original input is never edited.
    """
    prior = Path(prior)
    destination = Path(destination)
    if destination.exists():
        raise ValueError('recovery destination already exists')
    roster = {t['query'] for t in tasks}
    rows = [json.loads(l) for l in prior.read_text(encoding='utf-8').splitlines() if l.strip()]
    seen, kept, retry = set(), [], []
    for row in rows:
        if not row:
            raise ValueError('rollout missing identity')
        query = row[0]['extra_info']['query']
        if query not in roster or query in seen:
            raise ValueError('unknown or duplicate query in prior run')
        seen.add(query)
        if api_complete(row):
            kept.append(row)
        else:
            retry.append(query)
    if seen != roster:
        raise ValueError('prior run must cover the complete frozen task roster')
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in kept), encoding='utf-8')
    return {'policy': 'retry only episodes with an empty API completion; preserve all nonempty model failures',
            'source_sha256': hashlib.sha256(prior.read_bytes()).hexdigest(),
            'tasks': len(roster), 'inherited_api_complete': len(kept), 'retry_count': len(retry),
            'retry_queries': retry, 'first_attempt_must_be_reported_separately': True}
