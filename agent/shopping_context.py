"""索引注入工具提供的候选和证据，不修改工具事实，不访问知识库实现。"""

from copy import deepcopy


SHOPPING_READS = {
    "describe_filters": "category", "search_products": "filters",
    "retrieve_evidence": "candidate_set_id", "get_offer_facts": "offer_ids",
    "resolve_evidence": "evidence_ids",
}


def reusable(tool):
    # 显式标记优先；既有购物工具按名称及参数契约识别，避免缓存任意同名工具。
    properties = tool.arguments_model.model_json_schema().get("properties", {})
    return tool.read_only or SHOPPING_READS.get(tool.name) in properties


def records(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def remember(memory, tool, arguments, data):
    memory = deepcopy(memory or {"candidates": [], "evidence": [], "notes": []})
    candidates = {c["variant_id"]: c for c in memory.get("candidates", [])}
    evidence = {e["evidence_id"]: e for e in memory.get("evidence", [])}
    notes = list(memory.get("notes", []))
    for item in records(data.get("candidates")):
        if not isinstance(item.get("variant_id"), str):
            continue
        candidate = candidates.setdefault(item["variant_id"], {"variant_id": item["variant_id"]})
        for key in ("title", "category", "selected_price", "hard_constraint_status", "attributes"):
            if key in item:
                candidate[key] = deepcopy(item[key])
        price = candidate.get("selected_price") or {}
        price = price if isinstance(price, dict) else {}
        # 金额依然是工具的最小货币单位事实；不计算或伪造 verified total。
        candidate["selected_price"] = {k: price[k] for k in (
            "offer_id", "evidence_id", "currency", "price_minor", "shipping_minor", "shipping_status",
            "tax_status", "cost_conflict", "variant_conflict", "identity_status") if k in price}
        if "candidate_set_id" in data:
            candidate["candidate_set_id"] = data["candidate_set_id"]
        statuses = candidate.get("hard_constraint_status", {})
        statuses = statuses if isinstance(statuses, dict) else {}
        for attr in records(candidate.get("attributes")):
            if attr.get("status") == "unknown" and statuses.get(attr.get("name")) == "satisfied":
                notes.append("Structured constraint and displayed attribute disagree; resolve its evidence before claiming a specification.")
        if "unknown" in statuses.values():
            notes.append("Unknown structured facts need cited text, not repeated strict searches. Do not treat unknown as satisfied.")
        candidates.pop(item["variant_id"], None)
        candidates[item["variant_id"]] = candidate
    for item in records(data.get("evidence")):
        if not isinstance(item.get("evidence_id"), str):
            continue
        # 保留文字片段及位置，不能把两个分页文本无标记地拼成新的来源记录。
        record = {k: deepcopy(item[k]) for k in (
            "evidence_id", "variant_id", "kind", "text", "text_start", "text_end",
            "text_truncated", "scope", "source_url") if k in item}
        record["text"] = str(record.get("text", ""))[:1400]
        old = evidence.get(item["evidence_id"], {})
        excerpts = old.get("excerpts", [])
        if record not in excerpts:
            excerpts = (excerpts + [record])[-2:]
        evidence[item["evidence_id"]] = {"evidence_id": item["evidence_id"], "excerpts": excerpts}
    for offer in records(data.get("offers")):
        if not isinstance(offer.get("variant_id"), str):
            continue
        candidate = candidates.setdefault(offer["variant_id"], {"variant_id": offer["variant_id"]})
        candidate["offer_facts"] = {k: deepcopy(offer[k]) for k in (
            "offer_id", "evidence_id", "currency", "price_minor", "shipping_minor", "shipping_status",
            "tax_status", "cost_conflict", "variant_conflict") if k in offer}
    filters = arguments.get("filters", {})
    filters = filters if isinstance(filters, dict) else {}
    if tool == "search_products" and data.get("total_count") == 0 and filters.get("include_unknown") is False:
        notes.append("This empty search excluded unknowns. If earlier candidates exist, retain unknowns once and resolve relevant specification evidence; keep the user's requirements for final selection.")
    memory.update(candidates=list(candidates.values())[-6:], evidence=list(evidence.values())[-8:],
                  notes=list(dict.fromkeys(notes))[-4:])
    return memory


def prompt_memory(memory, limit):
    """整个摘要有界；优先保留已取得的证据，再保留最近候选。"""
    import json
    memory = deepcopy(memory or {})
    while len(json.dumps(memory, ensure_ascii=False)) > limit:
        if len(memory.get("candidates", [])) > 1:
            memory["candidates"].pop(0)
        elif len(memory.get("evidence", [])) > 1:
            memory["evidence"].pop(0)
        elif memory.get("notes"):
            memory["notes"].pop(0)  # 同样的处理规则已在系统指令里；事实片段优先。
        else:
            excerpts = [excerpt for item in memory.get("evidence", []) for excerpt in item.get("excerpts", [])]
            longest = max(excerpts, key=lambda x: len(x.get("text", "")), default={})
            if len(longest.get("text", "")) <= 512:
                return {}  # 原历史仍受独立预算管理，绝不截断成无效 JSON。
            text = longest["text"][:max(512, len(longest["text"]) * 3 // 4)]
            longest.update(text=text, text_end=longest.get("text_start", 0) + len(text),
                           text_truncated=True, summary_truncated=True)
    return memory
