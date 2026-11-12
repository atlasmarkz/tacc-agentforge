"""离线端到端示例，所有商品、价格和证据均为模拟数据。

运行方式：python -m agent.examples.mock_run [--events] [--clarify]
指定 --clarify 后演示挂起，并由示例自动提交模拟的用户回答。
"""

import argparse
import asyncio
import json
from decimal import Decimal, InvalidOperation

from pydantic import Field

from agent import (
    AgentDependencies, AgentRequest, AgentRuntime, EvidenceFact, Issue, ModelReply,
    ToolRegistry, ToolSpec, ValidationResult, VerifiedPlan,
)
from agent.contracts import Schema
from agent.settings import load_runtime_settings

DEMO_OFFER = "demo-offer-001"
DEMO_EVIDENCE = "demo-evidence-001"
DEMO_PRICE = Decimal("49.99")


class SearchArguments(Schema):
    """模拟搜索工具接收的查询参数。"""

    query: str = Field(min_length=1, max_length=200)


class SearchResult(Schema):
    """包含商品、报价和证据引用的模拟搜索结果。"""

    offer_id: str
    title: str
    price: str
    currency: str
    evidence_id: str
    synthetic: bool


class MockSearch:
    """返回固定的本地模拟商品，不执行外部查询。"""

    async def execute(self, arguments, context):
        return SearchResult(
            offer_id=DEMO_OFFER, title="Synthetic commuting earbuds",
            price=str(DEMO_PRICE), currency="USD", evidence_id=DEMO_EVIDENCE,
            synthetic=True,
        )


class MockModel:
    """根据请求和观察结果生成预设动作，不保存各任务共享的对话状态。"""

    async def generate(self, request, context):
        if request.action_schema is None:
            plan = json.loads(request.messages[1].content)["verified_plan"]
            text = f"此方案合计 {plan['total_cost']} {plan['currency']}，剩余预算 {plan['remaining_funds']} {plan['currency']}。[E1]"
            if context.on_delta is not None:
                for offset in range(0, len(text), 5):
                    context.on_delta("content", text[offset:offset + 5])
                    await asyncio.sleep(0)
            return ModelReply(content=text)
        original = json.loads(request.messages[1].content)
        observations = [json.loads(m.content)["observation"] for m in request.messages
                        if m.role == "user" and m.content.startswith('{"observation":')]
        if "purchase_cap" not in original["constraints"]:
            action = {"type": "ask_user", "question": "What is your purchase cap in USD?",
                      "required_fields": ["purchase_cap"]}
        elif not observations:
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
                      "required_fields": ["purchase_cap"]}
        # 不编造令牌测量值；未提供的用量会由运行时记入未报告次数。
        return ModelReply(content=json.dumps(action))


class MockPlanValidator:
    """只验证本示例的模拟目录和购买上限，完整预算算法由正式适配器提供。"""

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
        # 使用十进制计算；示例中的零运费和零税费是预设数据，不代表真实费用。
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
    """构造带显式快照和美元购买上限的模拟请求。"""
    return AgentRequest(
        run_id=run_id, conversation_id=f"conversation-{run_id}", snapshot_id="demo-snapshot-v1",
        message="推荐预算内适合通勤的无线耳机。",
        constraints={"currency": "USD", "purchase_cap": "100.00"},
    )


def demo_tools(executor=None):
    """为示例注册搜索工具，也允许测试替换其执行器。"""
    return ToolRegistry([ToolSpec(
        name="search_products", description="Search the selected snapshot for candidate offers.",
        arguments_model=SearchArguments, result_model=SearchResult,
        executor=executor if executor is not None else MockSearch(),
    )])


class MockEvidenceResolver:
    """只提供本示例目录的显式模拟证据。"""

    async def resolve(self, plan, context):
        return [EvidenceFact(snapshot_id=context.snapshot_id, offer_id=item.offer_id, evidence_id=eid,
                             title="Synthetic commuting earbuds", merchant="Synthetic store",
                             unit_price=str(DEMO_PRICE), currency="USD", excerpt="Synthetic fixture.",
                             source_url="https://example.com/demo")
                for item in plan.items for eid in item.evidence_ids]


def demo_runtime(*, explain=None, config=None, profile=None):
    """显式组装模拟模型、搜索工具和校验器，便于离线运行。"""
    settings = load_runtime_settings(config, profile, default_profile="mock")
    policy = settings.policy if explain is None else settings.policy.model_copy(update={"explain": explain})
    return AgentRuntime(AgentDependencies(MockModel(), demo_tools(), MockPlanValidator(),
                                          evidence_resolver=MockEvidenceResolver()), policy)


async def main():
    """根据参数输出最终结果或事件流，并可自动演示一次问答恢复。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", action="store_true", help="print live node events as JSON lines")
    parser.add_argument("--clarify", action="store_true", help="pause for a missing budget, then simulate an answer")
    parser.add_argument("--config", help="Agent YAML (default: configs/agent.yaml)")
    parser.add_argument("--profile", help="named YAML profile (default: mock)")
    parser.add_argument("--explain", action=argparse.BooleanOptionalAction, default=None, help="override explanation policy")
    args = parser.parse_args()
    runtime, request = demo_runtime(explain=args.explain, config=args.config, profile=args.profile), demo_request()
    if args.clarify:
        request = request.model_copy(update={"constraints": {"currency": "USD"}})
    result = None
    if args.events:
        async for event in runtime.astream(request):
            print(event.model_dump_json(), flush=True)
            if event.result is not None:
                result = event.result
    else:
        result = await runtime.run(request)
        print(result.model_dump_json(indent=2))
    if args.clarify and result is not None and result.status == "waiting_user":
        question_id = result.pending_question.question_id
        # 在应用调用边界显式提交模拟回答；实际接入时由调用方收集用户输入。
        if args.events:
            async for event in runtime.astream_resume(request.run_id, question_id, {"purchase_cap": "100.00"}):
                print(event.model_dump_json(), flush=True)
                if event.result is not None:
                    result = event.result
        else:
            result = await runtime.resume(request.run_id, question_id, {"purchase_cap": "100.00"})
            print(result.model_dump_json(indent=2))
    return 0 if result is not None and result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
