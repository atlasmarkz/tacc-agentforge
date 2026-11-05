"""Offline end-to-end example. All products, prices and evidence are synthetic.

Run from the repository root: python -m agent.examples.mock_run [--events]
"""

import argparse
import asyncio
import json
from decimal import Decimal, InvalidOperation

from pydantic import Field

from agent import (
    AgentDependencies, AgentRequest, AgentRuntime, Issue, ModelReply,
    ToolRegistry, ToolSpec, ValidationResult, VerifiedPlan,
)
from agent.contracts import Schema

DEMO_OFFER = "demo-offer-001"
DEMO_EVIDENCE = "demo-evidence-001"
DEMO_PRICE = Decimal("49.99")


class SearchArguments(Schema):
    query: str = Field(min_length=1, max_length=200)


class SearchResult(Schema):
    offer_id: str
    title: str
    price: str
    currency: str
    evidence_id: str
    synthetic: bool


class MockSearch:
    async def execute(self, arguments, context):
        return SearchResult(
            offer_id=DEMO_OFFER, title="Synthetic commuting earbuds",
            price=str(DEMO_PRICE), currency="USD", evidence_id=DEMO_EVIDENCE,
            synthetic=True,
        )


class MockModel:
    """Stateless decision fixture driven by observations, safe for independent runs."""

    async def generate(self, request, context):
        observations = [json.loads(m.content)["observation"] for m in request.messages
                        if m.role == "user" and m.content.startswith('{"observation":')]
        if not observations:
            action = {"type": "call_tool", "tool_name": "search_products",
                      "arguments": {"query": "wireless commuting earbuds"}}
        elif observations[-1].get("status") == "ok":
            data = observations[-1]["data"]
            action = {"type": "propose_plan", "plan": {"items": [{
                "offer_id": data["offer_id"], "quantity": 1,
                "evidence_ids": [data["evidence_id"]],
            }]}}
        else:
            action = {"type": "ask_user", "question": "Can you revise the requirements or budget?",
                      "required_fields": ["constraints"]}
        # No invented token measurements: absent usage is counted as unreported.
        return ModelReply(content=json.dumps(action))


class MockPlanValidator:
    """Only validates this synthetic fixture; not the project's real budget algorithm."""

    async def validate(self, request, context):
        def reject(code, message, status="failed"):
            return ValidationResult(
                status=status, proposal_fingerprint=request.proposal_fingerprint,
                issues=[Issue(code=code, message=message)],
            )

        constraints = request.request.constraints
        try:
            cap = Decimal(str(constraints["purchase_cap"]))
        except (KeyError, InvalidOperation):
            return reject("missing_budget", "An explicit purchase cap is required.", "unavailable")
        if not cap.is_finite() or cap < 0 or constraints.get("currency") != "USD":
            return reject("unsupported_budget", "This fixture requires a nonnegative USD cap.", "unavailable")
        for item in request.plan.items:
            if item.offer_id != DEMO_OFFER or item.evidence_ids != [DEMO_EVIDENCE]:
                return reject("unknown_evidence", "Use the synthetic catalog's offer and evidence.")
        total = sum((DEMO_PRICE * item.quantity for item in request.plan.items), Decimal("0"))
        if total > cap:
            return reject("budget_exceeded", "The proposed fixture exceeds the purchase cap.")
        return ValidationResult(
            status="passed", proposal_fingerprint=request.proposal_fingerprint,
            verified_plan=VerifiedPlan(
                snapshot_id=request.request.snapshot_id, items=request.plan.items,
                currency="USD", total_cost=f"{total:.2f}", remaining_funds=f"{cap - total:.2f}",
                cost_breakdown={"items": f"{total:.2f}", "shipping": "0.00", "tax": "0.00"},
            ),
        )


def demo_request(run_id="demo-run-001"):
    return AgentRequest(
        run_id=run_id, conversation_id=f"conversation-{run_id}", snapshot_id="demo-snapshot-v1",
        message="推荐预算内适合通勤的无线耳机。",
        constraints={"currency": "USD", "purchase_cap": "100.00"},
    )


def demo_tools(executor=None):
    return ToolRegistry([ToolSpec(
        name="search_products", description="Search the selected snapshot for candidate offers.",
        arguments_model=SearchArguments, result_model=SearchResult,
        executor=executor if executor is not None else MockSearch(),
    )])


def demo_runtime():
    return AgentRuntime(AgentDependencies(MockModel(), demo_tools(), MockPlanValidator()))


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", action="store_true", help="print live node events as JSON lines")
    args = parser.parse_args()
    runtime, request = demo_runtime(), demo_request()
    result = None
    if args.events:
        async for event in runtime.astream(request):
            print(event.model_dump_json(), flush=True)
            if event.result is not None:
                result = event.result
    else:
        result = await runtime.run(request)
        print(result.model_dump_json(indent=2))
    return 0 if result is not None and result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
