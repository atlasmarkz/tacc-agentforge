"""Pydantic tool adapters for the project's existing agent ports.

This optional module imports the agent dependencies; knowledge.service does not.
The snapshot ID always comes from CallContext, never model-controlled arguments.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from pydantic import Field, JsonValue

from agent.contracts import CapabilityError, Identifier, Schema
from agent.tools import ToolRegistry, ToolSpec
from .service import KnowledgeError, KnowledgeService


class DescribeFiltersArguments(Schema):
    category: str | None = None
    cursor: str | None = Field(default=None, max_length=2048)


class SearchProductsArguments(Schema):
    filters: dict[str, JsonValue] = Field(default_factory=dict)
    limit: int = Field(default=5, strict=True, ge=1, le=20)
    cursor: str | None = Field(default=None, max_length=2048)


class RetrieveEvidenceArguments(Schema):
    query: str = Field(min_length=1, max_length=2000)
    candidate_set_id: Identifier
    k: int = Field(default=5, strict=True, ge=1, le=20)


class GetOfferFactsArguments(Schema):
    offer_ids: list[Identifier] = Field(min_length=1, max_length=100)


class ResolveEvidenceArguments(Schema):
    evidence_ids: list[Identifier] = Field(min_length=1, max_length=100)
    cursor: str | None = Field(default=None, max_length=2048)


class KnowledgeResult(Schema):
    snapshot_id: Identifier
    dataset_id: str
    as_of: str
    responded_at: str
    bytes: int = Field(strict=True, ge=0)
    truncated: bool
    next_cursor: str | None


class FilterAttribute(Schema):
    name: str
    type: str
    operators: list[str]
    unit: str | None
    accepted_units: list[str]


class DescribeFiltersResult(KnowledgeResult):
    category: str | None
    categories: list[str]
    attributes: list[FilterAttribute]


class SearchProductsResult(KnowledgeResult):
    candidate_set_id: Identifier
    total_count: int = Field(strict=True, ge=0)
    scope_complete: bool
    candidates: list[dict[str, JsonValue]]
    filter_digest: str


class RetrieveEvidenceResult(KnowledgeResult):
    candidate_set_id: Identifier
    mode: str
    query: str
    evidence: list[dict[str, JsonValue]]
    requested_k: int = Field(strict=True, ge=1)
    returned_count: int = Field(strict=True, ge=0)


class GetOfferFactsResult(KnowledgeResult):
    offers: list[dict[str, JsonValue]]
    unknown_ids: list[str]
    remaining_offer_ids: list[str]


class ResolveEvidenceResult(KnowledgeResult):
    evidence: list[dict[str, JsonValue]]
    unknown_ids: list[str]


class _Workers:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="knowledge-query")
        self.slots = asyncio.Semaphore(2)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


class KnowledgeToolExecutor:
    def __init__(self, service, method, workers, *, retrieval_mode="filtered"):
        self.service, self.method, self.workers = service, method, workers
        self.retrieval_mode = retrieval_mode

    def close(self):
        """Close shared workers at host shutdown, after active runs finish."""
        self.workers.close()

    async def execute(self, arguments, context):
        deadline = time.monotonic() + context.timeout_s
        cancelled = threading.Event()
        kwargs = arguments.model_dump(mode="python")
        if self.method == "retrieve_evidence":
            kwargs["mode"] = self.retrieval_mode

        def call():
            try:
                with self.service.query_context(deadline=deadline, cancellation=cancelled):
                    return getattr(self.service, self.method)(context.snapshot_id, **kwargs)
            except KnowledgeError as exc:
                raise CapabilityError(exc.code, exc.public_message, recoverable=exc.recoverable) from exc

        # An admission permit belongs to the actual concurrent future, including
        # work that outlives an asyncio timeout. Cancelling an asyncio wrapper is
        # not proof that its underlying worker stopped.
        loop = asyncio.get_running_loop()
        admission = asyncio.create_task(self.workers.slots.acquire())
        cancel_wait = asyncio.create_task(context.cancellation.wait())
        raw_future = None
        future = None

        def release_after_worker(_):
            try:
                loop.call_soon_threadsafe(self.workers.slots.release)
            except RuntimeError:
                # A host may close its event loop while a cooperative worker is
                # finishing. No loop waiters remain; restore the permit itself.
                self.workers.slots.release()

        try:
            done, _ = await asyncio.wait(
                [admission, cancel_wait], timeout=max(0, deadline - time.monotonic()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_wait in done or context.cancellation.is_set():
                raise CapabilityError("cancelled", "Knowledge query was cancelled.", recoverable=True)
            if admission not in done or time.monotonic() >= deadline:
                raise CapabilityError("deadline_exceeded", "Knowledge query exceeded its deadline.", recoverable=True)
            admission.result()
            raw_future = self.workers.pool.submit(call)
            raw_future.add_done_callback(release_after_worker)
            future = asyncio.wrap_future(raw_future, loop=loop)
            # Always observe an eventual exception, including after a timed-out
            # caller has left. This callback does not release admission permits.
            future.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
            done, _ = await asyncio.wait(
                [future, cancel_wait], timeout=max(0, deadline - time.monotonic()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_wait in done:
                cancelled.set()
                raise CapabilityError("cancelled", "Knowledge query was cancelled.", recoverable=True)
            if future not in done:
                cancelled.set()
                raise CapabilityError("deadline_exceeded", "Knowledge query exceeded its deadline.", recoverable=True)
            return await future
        except asyncio.CancelledError:
            cancelled.set()
            raise
        finally:
            cancel_wait.cancel()
            if not admission.done():
                admission.cancel()
            await asyncio.gather(admission, cancel_wait, return_exceptions=True)
            if raw_future is None:
                # Cancellation can race with a successful acquire. Release only
                # a permit that was acquired but never transferred to a worker.
                if not admission.cancelled() and admission.exception() is None and admission.result():
                    self.workers.slots.release()
            elif not raw_future.done():
                cancelled.set()
                raw_future.cancel()  # Pending work can stop; running SQL uses progress_handler.
                if future is not None:
                    future.cancel()


def build_knowledge_tools(service: KnowledgeService, *, retrieval_mode="filtered") -> ToolRegistry:
    """Register schema discovery and four bounded queries over the trusted snapshot."""
    if retrieval_mode not in ("filtered", "keyword"):
        raise ValueError("retrieval_mode must be filtered or keyword")
    workers = _Workers()
    specs = [
        ("describe_filters", DescribeFiltersArguments, DescribeFiltersResult,
         "Discover legal search filters from the current snapshot before guessing names. "
         "Call with {} to list categories; pass category to list its canonical attributes, value types, "
         "allowed operators and units for hard_constraints. Continue with next_cursor and the same category. "
         "Attributes describe the schema, not proof that a product has a known value."),
        ("search_products", SearchProductsArguments, SearchProductsResult,
         "Search the trusted snapshot. Filters: category, type, currency, max_price_minor (inclusive), "
         "min_price_minor, variant_ids, hard_constraints {attribute:{op,value,unit?}}, include_unknown. "
         "Use describe_filters to discover legal category and attribute names. "
         "type is a top-level catalog filter, not a hard_constraints attribute; omit it if unknown. "
         "Missing facts remain unknown. Returns a full-scope candidate_set_id and a compact page."),
        ("retrieve_evidence", RetrieveEvidenceArguments, RetrieveEvidenceResult,
         "Retrieve short cited evidence for an active candidate_set_id in the trusted snapshot. "
         "Top-k evidence does not establish that the full catalog has no feasible plan."),
        ("get_offer_facts", GetOfferFactsArguments, GetOfferFactsResult,
         "Read current offer observations with exact minor-unit prices, fee states and evidence IDs. "
         "Unknown fees, conditions, identity or conflicting costs cannot establish affordability."),
        ("resolve_evidence", ResolveEvidenceArguments, ResolveEvidenceResult,
         "Resolve evidence IDs to text, source URL and archived JSON provenance. Continue with next_cursor "
         "and the same IDs when text or records are paginated."),
    ]
    return ToolRegistry(ToolSpec(
        name=name, description=description, arguments_model=arguments, result_model=result,
        executor=KnowledgeToolExecutor(service, name, workers, retrieval_mode=retrieval_mode),
        version="2", timeout_s=5, max_result_bytes=service.max_result_bytes,
    ) for name, arguments, result, description in specs)
