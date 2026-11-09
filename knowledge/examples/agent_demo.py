"""Exercise the real KB through AgentRuntime; the model decisions are scripted.

No model endpoint, API request or fabricated budget approval is used. Unknown
taxes/identity lead to a necessary clarification, not a verified purchase plan.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sqlite3

from agent import AgentDependencies, AgentRequest, AgentRuntime, ModelReply
from knowledge.agent_tools import build_knowledge_tools
from knowledge.service import KnowledgeService
from knowledge.store import verify_snapshot


class ScriptedModel:
    async def generate(self, request, context):
        observations=[]
        for message in request.messages:
            if message.role == "user" and message.content.startswith('{"observation":'):
                observations.append(json.loads(message.content)["observation"])
        if not observations:
            action={"type":"call_tool","tool_name":"search_products",
                    "arguments":{"filters":self.filters,"limit":1}}
        elif observations[-1].get("status") != "ok":
            action={"type":"ask_user","question":"商品证据尚不足，请补充需求或更换候选。",
                    "required_fields":["requirements"]}
        elif observations[-1]["tool"] == "search_products":
            candidates=observations[-1]["data"]["candidates"]
            price=candidates[0].get("selected_price") if candidates else None
            if price and price.get("offer_id"):
                action={"type":"call_tool","tool_name":"get_offer_facts",
                        "arguments":{"offer_ids":[price["offer_id"]]}}
            else:
                action={"type":"ask_user","question":"缺少可核实的商家报价，请选择具体规格和商家。",
                        "required_fields":["offer","variant_identity"]}
        else:
            action={"type":"ask_user","question":"当前报价缺少已确认税费、配送条件和具体规格，需补齐后才能验证预算。",
                    "required_fields":["tax","destination","quantity","variant_identity"]}
        return ModelReply(content=json.dumps(action,ensure_ascii=False))


async def run_demo(snapshots_root:Path,snapshot_id:str):
    snapshot=Path(snapshots_root)/snapshot_id
    verification=verify_snapshot(snapshot)
    if not verification["valid"]:
        raise ValueError("Snapshot verification failed")
    conn=sqlite3.connect((snapshot/'catalog.sqlite').as_uri()+"?mode=ro",uri=True)
    try:
        row=conn.execute("SELECT o.variant_id FROM offers o JOIN variants v USING(variant_id) "
                         "JOIN current_offer_observations ob USING(offer_id) "
                         "WHERE v.category='earbuds' AND ob.currency='USD' AND ob.cost_conflict=0 "
                         "ORDER BY o.offer_id LIMIT 1").fetchone()
    finally:
        conn.close()
    if row is None:
        raise ValueError("This example needs an archived USD earbuds offer")
    service=KnowledgeService(snapshots_root)
    tools=build_knowledge_tools(service)
    model=ScriptedModel()
    model.filters={"category":"earbuds","currency":"USD","variant_ids":[row[0]]}
    runtime=AgentRuntime(AgentDependencies(model=model,tools=tools))
    request=AgentRequest(run_id="kb-integration-demo",conversation_id="kb-demo",snapshot_id=snapshot_id,
        message="请根据已归档证据推荐通勤耳机，含税运费预算为100美元。",
        constraints={"currency":"USD","purchase_cap":"100.00"})
    events=[]
    try:
        async for event in runtime.astream(request):
            events.append(event.model_dump(mode="json"))
    finally:
        tools.get("search_products").executor.close()
    result=next(e["result"] for e in reversed(events) if e.get("result"))
    return {"scripted_model":True,"real_archived_knowledge":True,"external_calls":0,
            "result":result,"events":events}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots-root",type=Path,required=True)
    parser.add_argument("--snapshot-id",required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=asyncio.run(run_demo(args.snapshots_root,args.snapshot_id))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({key:value for key,value in result.items() if key!='events'},ensure_ascii=False,indent=2))
    return 0 if result["result"]["status"] == "waiting_user" else 1


if __name__ == "__main__":
    raise SystemExit(main())
