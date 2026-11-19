# Human review rubric
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
