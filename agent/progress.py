"""有界的查询进展与决策预算；不把来源文本提升为运行指令。"""

import hashlib
import json


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def semantic_result(value):
    """忽略响应时间等元数据；翻页文本、报价和错误变化仍算新信息。"""
    volatile = {"responded_at", "bytes", "price_age_seconds", "price_stale", "candidate_set_id", "filter_digest",
                "next_cursor", "truncated", "rank", "score"}
    if isinstance(value, dict):
        return {k: semantic_result(v) for k, v in value.items() if k not in volatile}
    if isinstance(value, list):
        return [semantic_result(v) for v in value]
    return value


def known_satisfying(candidate):
    """判断候选是否同时具备已知价格和已满足的硬约束。"""
    if not isinstance(candidate, dict):
        return False
    price = candidate.get("selected_price")
    if not isinstance(price, dict) or price.get("price_minor") is None:
        return False
    statuses = candidate.get("hard_constraint_status")
    if isinstance(statuses, dict) and any(status != "satisfied" for status in statuses.values()):
        return False
    return candidate.get("constraint_status") != "unknown"


def page_is_informative(data):
    """没有已知满足条件候选的搜索页不算进展；空结果仍代表一次有效发现。"""
    candidates = data.get("candidates") if isinstance(data, dict) else None
    if not isinstance(candidates, list) or not candidates:
        return True
    return any(known_satisfying(candidate) for candidate in candidates)


def result_facts(tool_name, data, arguments):
    """按事实而不是整页计进展：换句柄、顺序或重叠分页不会制造新信息。"""
    if not isinstance(data, dict):
        return [fingerprint({"tool": tool_name, "result": semantic_result(data)})]
    for key in ("candidates", "evidence", "offers"):
        items = data.get(key)
        if isinstance(items, list) and items:
            return [fingerprint({"kind": key, "fact": semantic_result(item)}) for item in items]
    # 空搜索也是一次发现，但不同筛选的空结果不是同一事实。cursor/limit 只改变展示。
    query = semantic_result({k: v for k, v in (arguments or {}).items() if k not in {"cursor", "limit", "k"}})
    return [fingerprint({"tool": tool_name, "query": query, "result": semantic_result(data)})]


def record_result(state, tool_name, data, *, error=False, arguments=None):
    previous = state.get("progress", {})
    seen = previous.get("seen_results", [])
    facts = result_facts(tool_name, data, arguments)
    # 翻页候选的价格和规格全部未知时，下一页不带来可用于决策的新事实。
    unseen = any(digest not in seen for digest in facts)
    new = unseen and not error and page_is_informative(data)
    # 参数纠正后第一次成功检索应有一次针对性补证据的机会；先前参数错误仍占总决策额度。
    stalled_before = 0 if unseen and not error and previous.get("last_result_error") else previous.get("no_progress_calls", 0)
    progress = {"seen_results": list(dict.fromkeys(seen + facts))[-256:],
                "no_progress_calls": 0 if new else stalled_before + 1,
                "last_result_error": error,
                "force_finalize": previous.get("force_finalize", False)}
    if "resume_evidence_budget" in previous:
        progress["resume_evidence_budget"] = previous["resume_evidence_budget"]
    return progress


def decision_control(state, policy):
    remaining = max(0, policy.max_model_calls - state["usage"]["model_calls"])
    tools_left = max(0, policy.max_tool_calls - state["usage"]["tool_calls"])
    progress = state.get("progress", {})
    stalled = progress.get("no_progress_calls", 0) >= policy.max_no_progress_calls
    # 小预算仍为一次查询留空间，但最后一次绝不执行一个无法再用于提案的工具。
    reserve = min(policy.finalize_reserve_model_calls, max(1, policy.max_model_calls - 1))
    if progress.get("resume_evidence_budget", 0):
        reserve = 1  # 回答后可用预留额度补一次证据，再提交；总调用计数仍不重置。
    finalize = remaining <= reserve or stalled or progress.get("force_finalize", False) or not tools_left
    # 已证实目录内无可支持的方案时可以直接说明，不必先把探索额度耗尽。
    allowed = ["propose_plan", "report_incomplete"]
    if not finalize:
        allowed.append("call_tool")
    # 本次提问后至少留两次，用于补证据及提案。计数跨恢复累计。
    if remaining >= 3:
        allowed.append("ask_user")
    return {"model_calls_remaining": remaining, "tool_calls_remaining": tools_left,
            "phase": "finalize" if finalize else "converge" if remaining == reserve + 1 else "explore",
            "allowed_actions": allowed, "no_progress_calls": progress.get("no_progress_calls", 0)}


def bounded(value, limit=4096):
    encoded = json.dumps(value, ensure_ascii=False)
    return value if len(encoded) <= limit else {"truncated": True, "preview": encoded[:limit]}
