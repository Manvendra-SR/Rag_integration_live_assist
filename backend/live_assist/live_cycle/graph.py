from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from live_assist.core.config import get_settings
from live_assist.core.models import ProductType, QueryResponse, RewriteQuestion, Speaker
from live_assist.core.terminal_log import compact_text, debug_log
from live_assist.live_cycle.state import LiveAssistState
from live_assist.providers.llm.groq import GroqLLM
from live_assist.storage.context_store import context_store

settings = get_settings()
config = settings.workflow_config()
llm = GroqLLM(config)
RAG_INDEXED_PRODUCTS = {
    ProductType.ILTS.value,
    ProductType.FGF.value,
    ProductType.FINRAKSHAK.value,
}
ILTS_ALIASES = (
    "ilts",
    "ielts",
    "index long term strategy",
    "relax",
    "basic",
    "comfort",
    "power booster",
    "dynamic",
    "marathon",
)
FGF_ALIASES = ("fgf", "growth fund", "finideas growth fund")
FINRAKSHAK_ALIASES = ("finrakshak", "fin rakshak", "hedge", "hedging")


def log_text(value: str, limit: int = 400) -> str:
    return compact_text(value, limit)


def get_products_list() -> list[str]:
    return [item.value for item in ProductType]


def get_turn_count(messages: list) -> int:
    def _role_of(message: Any) -> str:
        if isinstance(message, dict):
            return str(message.get("role", ""))
        if isinstance(message, BaseMessage):
            if message.type == "human":
                return "user"
            if message.type == "ai":
                return "assistant"
            return message.type
        return ""

    return len([message for message in messages if _role_of(message) == "user"])


def get_recent_messages(
    messages: list,
    last_n_turns: int = 5,
    include_roles: tuple[str, ...] = ("user", "assistant"),
) -> list[dict[str, str]]:
    def _serialize_message(message: Any) -> dict[str, str] | None:
        if isinstance(message, dict):
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
        elif isinstance(message, BaseMessage):
            if message.type == "human":
                role = "user"
            elif message.type == "ai":
                role = "assistant"
            else:
                role = message.type
            content = str(message.content)
        else:
            return None

        if role not in include_roles:
            return None
        return {"role": role, "content": content}

    conversation = [
        serialized
        for serialized in (_serialize_message(message) for message in messages)
        if serialized is not None
    ]
    if include_roles == ("user", "assistant"):
        return conversation[-(last_n_turns * 2) :]
    return conversation[-last_n_turns:]


def format_summary_turns(summary_turns: list[dict[str, str]]) -> str:
    formatted_turns = []
    for index, turn in enumerate(summary_turns, start=1):
        if "speaker" in turn:
            formatted_turns.append(
                f"TURN {index}\n{turn.get('speaker', '').strip()}: {turn.get('text', '').strip()}".strip()
            )
        else:
            formatted_turns.append(
                "\n".join(
                    [
                        f"TURN {index}",
                        f"QUESTION: {turn.get('question', '').strip()}",
                        f"ASSISTANT: {turn.get('assistant', '').strip()}",
                        f"HUMAN_WORKER: {turn.get('human_worker', '').strip()}",
                    ]
                ).strip()
            )
    return "\n\n".join(turn for turn in formatted_turns if turn.strip())


def summarize_conversation(
    user_id: str,
    session_id: str,
    summary_turns: list[dict[str, str]],
    product: str = "",
) -> dict[str, Any]:
    existing_summary = context_store.get_current_session_summary(user_id, session_id)
    conversation_text = format_summary_turns(summary_turns).strip()
    if not conversation_text:
        return {"status": "summary_skipped_empty"}

    if existing_summary:
        user_prompt = f"""
        ## Existing Summary
        {existing_summary}

        ## New Conversation
        {conversation_text}

        Update the existing summary by incorporating the new conversation.
        Keep it under 200 words. Return updated summary text only.
        """
    else:
        user_prompt = f"""
        ## Conversation
        {conversation_text}

        Summarize this financial advisory conversation.
        Keep it under 200 words. Return summary text only.
        """

    summary = llm.invoke_model(
        system_prompt=config["SUMMARIZATION_PROMPT"],
        user_prompt=user_prompt,
        variables={},
    )
    context_store.save_summary(user_id, session_id, product, summary)
    return {"status": "summary_updated"}


def _merge_product_context(existing: str, detected: str) -> str:
    detected = _normalize_rag_product(detected) or detected
    products = [item.strip() for item in (existing or "").split(",") if item.strip()]
    if detected and detected not in products:
        products.append(detected)
    return ", ".join(products)


def _normalize_rag_product(value: str) -> str:
    lowered = (value or "").strip().lower()
    if not lowered:
        return ""

    for product in RAG_INDEXED_PRODUCTS:
        if lowered == product.lower():
            return product

    if any(alias in lowered for alias in ILTS_ALIASES):
        return ProductType.ILTS.value
    if any(alias in lowered for alias in FGF_ALIASES):
        return ProductType.FGF.value
    if any(alias in lowered for alias in FINRAKSHAK_ALIASES):
        return ProductType.FINRAKSHAK.value
    return ""


def _rag_product_filter(*values: str) -> dict[str, str] | None:
    for value in values:
        product = _normalize_rag_product(value)
        if product:
            return {"product": product}
    return None


def _detect_product(text: str, existing: str = "") -> str:
    lowered = (text or "").lower()
    detected = ""
    for product in get_products_list():
        if product.lower() in lowered:
            detected = product
            break
    if not detected:
        detected = _normalize_rag_product(text)
    return _merge_product_context(existing, detected)


def _format_conversation_turns(turns: list[dict[str, str]]) -> str:
    if not turns:
        return ""
    return "\n".join(
        f"{turn.get('speaker', '').strip()}: {turn.get('text', '').strip()}"
        for turn in turns
        if turn.get("text", "").strip()
    )


def ingest_turn(state: LiveAssistState) -> dict[str, Any]:
    speaker = state.speaker or Speaker.UNKNOWN.value
    text = (state.turn_text or state.question or "").strip()
    last_5_turns = [dict(turn) for turn in state.last_5_turns]
    conversation_turn_count = state.conversation_turn_count

    should_generate = state.manual_question or speaker == Speaker.CUSTOMER.value
    route = "rag_answer" if should_generate else "context_only"

    if text and not state.manual_question:
        if last_5_turns and last_5_turns[-1].get("speaker") == speaker:
            previous = last_5_turns[-1].get("text", "").strip()
            last_5_turns[-1]["text"] = f"{previous} {text}".strip()
        else:
            last_5_turns.append({"speaker": speaker, "text": text})
            conversation_turn_count += 1
        last_5_turns = last_5_turns[-settings.live_feedback_recent_turns :]

    debug_log(
        f"[Conversation Context] call={state.session_id} speaker={speaker} "
        f"route={route} conversation_turns={conversation_turn_count} "
        f"last_5={len(last_5_turns)} text={log_text(text)}"
    )
    return {
        "turn_text": text,
        "question": state.question or text,
        "last_5_turns": last_5_turns,
        "conversation_turn_count": conversation_turn_count,
        "should_generate_answer": should_generate,
        "route": route,
    }


def update_product_context(state: LiveAssistState) -> dict[str, Any]:
    existing = state.product_context or context_store.get_current_product_context(
        state.user_id,
        state.session_id,
    )
    product_context = _detect_product(state.turn_text or state.question, existing)
    if product_context:
        context_store.save_product_context(state.user_id, state.session_id, product_context)
    debug_log(f"[Product Context] call={state.session_id} product_context={log_text(product_context)}")
    return {
        "product_context": product_context,
        "product": state.product or (product_context.split(",")[0].strip() if product_context else ""),
    }


def maybe_update_summary(state: LiveAssistState) -> dict[str, Any]:
    if (
        state.conversation_turn_count < settings.live_feedback_recent_turns
        or state.conversation_turn_count % settings.live_feedback_recent_turns != 0
        or state.summary_turn_count == state.conversation_turn_count
    ):
        running_summary = context_store.get_current_session_summary(state.user_id, state.session_id)
        return {"running_summary": running_summary}

    result = summarize_conversation(
        user_id=state.user_id,
        session_id=state.session_id,
        summary_turns=state.last_5_turns,
        product=state.product_context or state.product,
    )
    running_summary = context_store.get_current_session_summary(state.user_id, state.session_id)
    debug_log(
        f"[Summary Context] call={state.session_id} status={result.get('status')} "
        f"conversation_turns={state.conversation_turn_count}"
    )
    return {
        "running_summary": running_summary,
        "summary_turn_count": state.conversation_turn_count,
    }


def route_by_speaker(state: LiveAssistState) -> str:
    if state.should_generate_answer:
        return "answer"
    return "context_only"


def context_only(state: LiveAssistState) -> dict[str, Any]:
    debug_log(
        f"[Context Only] call={state.session_id} speaker={state.speaker} "
        f"text={log_text(state.turn_text)}"
    )
    return {"answer": "", "route": "context_only"}


def enrich_query(state: LiveAssistState) -> dict[str, Any]:
    # Always perform LLM enrichment — cache is now checked as a separate downstream node
    user_prompt = f"""
    ## User Question
    {state.question}

    ## Known Products
    {get_products_list()}

    ## Last Conversation Turns
    {_format_conversation_turns(state.last_5_turns)}

    ## Running Summary
    {state.running_summary}

    ## Product Context
    {state.product_context}

    {config["REWRITE_QUESTION_USER_PROMPT"]}
    """
    response = llm.invoke_model_with_structured_output(
        schema=RewriteQuestion,
        system_prompt=config["REWRITE_QUESTION_SYSTEM_PROMPT"],
        user_prompt=user_prompt,
        variables={"question": state.question},
    )
    if not isinstance(response, RewriteQuestion):
        response = RewriteQuestion(rewriten_question="", product="")
    # If the LLM returned nothing, keep the original question so retrieval still runs
    enriched_query = (response.rewriten_question or state.question or "").strip()
    debug_log(
        "Enriched query created | "
        f"query={log_text(enriched_query) if enriched_query else 'NO MATCH'}"
    )
    product_context = _merge_product_context(state.product_context, response.product or "")
    if product_context:
        context_store.save_product_context(state.user_id, state.session_id, product_context)
    selected_product = _normalize_rag_product(response.product or "") or _normalize_rag_product(state.product or "") or (
        product_context.split(",")[0].strip() if product_context else ""
    )
    return {
        "rewriten_question": enriched_query,
        "product": selected_product,
        "product_context": product_context,
    }


def check_cache(state: LiveAssistState) -> dict[str, Any]:
    """Semantic cache lookup node — runs after enrichment so the enriched query is used as the key."""
    # Only attempt cache lookup for manual questions when cache is enabled
    if not state.manual_question:
        return {"cache_hit": False, "cache_similarity": None}

    _settings = get_settings()
    if not _settings.rag_cache_enabled:
        return {"cache_hit": False, "cache_similarity": None}

    try:
        from live_assist.rag_pipeline.paths import conversation_cache_dir
        from live_assist.rag_pipeline.semantic_cache import SemanticCache

        cache = SemanticCache(
            conversation_cache_dir(state.session_id),
            similarity_threshold=_settings.rag_cache_similarity_threshold,
            max_age_days=_settings.rag_cache_max_age_days,
        )
        query_key = (state.rewriten_question or state.question or "").strip()
        hit = cache.lookup(query_key)
        similarity = hit.get("similarity")
        sim_str = f"{similarity:.4f}" if similarity is not None else "N/A"

        if hit.get("result") == "HIT":
            debug_log(f"[SemanticCache] HIT | sim={sim_str} | session={state.session_id}")
            cached_response = hit["workflow_response"]
            # Inject cached values back into state so serve_cached_answer can return them
            return {
                "cache_hit": True,
                "cache_similarity": similarity,
                "answer": cached_response.get("answer", ""),
                "context": cached_response.get("context", ""),
                "rag_top_chunks": cached_response.get("rag_top_chunks", []),
                "rag_raw_chunks": cached_response.get("rag_raw_chunks", []),
            }
        else:
            debug_log(f"[SemanticCache] MISS | best_sim={sim_str} | session={state.session_id}")
            return {"cache_hit": False, "cache_similarity": similarity}
    except Exception as exc:
        debug_log(f"[SemanticCache] lookup error (ignored): {exc}")
        return {"cache_hit": False, "cache_similarity": None}


def route_by_cache(state: LiveAssistState) -> str:
    """Conditional edge: short-circuit to cached answer or proceed with full retrieval."""
    return "cache_hit" if state.cache_hit else "cache_miss"


def serve_cached_answer(state: LiveAssistState) -> dict[str, Any]:
    """Return state as-is — answer was already injected by check_cache."""
    debug_log(
        f"[SemanticCache] Serving cached answer | "
        f"sim={state.cache_similarity} | answer_len={len(state.answer)}"
    )
    return {"route": "cache_hit"}


def cache_store(state: LiveAssistState) -> dict[str, Any]:
    """Write the freshly generated answer to the semantic cache."""
    if not state.manual_question:
        return {}

    _settings = get_settings()
    if not _settings.rag_cache_enabled:
        return {}

    answer = (state.answer or "").strip()
    if not answer:
        return {}

    try:
        from live_assist.rag_pipeline.paths import conversation_cache_dir
        from live_assist.rag_pipeline.semantic_cache import SemanticCache

        cache = SemanticCache(
            conversation_cache_dir(state.session_id),
            similarity_threshold=_settings.rag_cache_similarity_threshold,
            max_age_days=_settings.rag_cache_max_age_days,
        )
        query_key = (state.rewriten_question or state.question or "").strip()
        # Build a minimal workflow_response snapshot for the cache entry
        workflow_response = {
            "answer": answer,
            "context": state.context,
            "rag_top_chunks": state.rag_top_chunks,
            "rag_raw_chunks": state.rag_raw_chunks,
            "rewriten_question": query_key,
            "product": state.product,
            "product_context": state.product_context,
        }
        cache.write(query_key, workflow_response)
        debug_log(f"[SemanticCache] Written | session={state.session_id}")
    except Exception as exc:
        debug_log(f"[SemanticCache] write error (ignored): {exc}")
    return {}


def retrieve_knowledge(state: LiveAssistState) -> dict[str, Any]:
    query_text = (state.rewriten_question or state.question or "").strip()
    if not query_text:
        return {
            "context": "",
            "rag_top_chunks": [],
            "rag_raw_chunks": [],
        }

    # ── Call the new configurable retriever ─────────────────────────────────────────────
    try:
        from live_assist.providers.rag.retriever import build_rag_retriever

        retriever = build_rag_retriever(settings)
        user_id = state.user_id if hasattr(state, "user_id") and state.user_id else settings.live_feedback_user_id
        doc_filter = (state.doc_filter or "").strip() or None
        retrieval_mode_override = (state.retrieval_mode or "").strip() or None

        if getattr(settings, "rag_provider", "legacy_chroma") == "advanced":
            pipeline_result = retriever.retrieve(
                query_text,
                user_id=user_id,
                doc_filter=doc_filter,
                mode=retrieval_mode_override,
            )
        else:
            # Fallback for legacy
            pipeline_result = {}
            results = retriever.hybrid_search(query_text, k=settings.rag_top_k)
            if results:
                context = "\n\n".join([doc.page_content for doc, _ in results])
                top_chunks = [doc.page_content[:300] for doc, _ in results[:3]]
                pipeline_result = {
                    "context": context,
                    "rag_top_chunks": top_chunks,
                }
            else:
                pipeline_result = {
                    "context": "",
                    "rag_top_chunks": [],
                }

    except Exception as exc:
        debug_log(f"[RAG Pipeline Error] {type(exc).__name__}: {exc}")
        pipeline_result = {"status": "error", "error": str(exc)}

    if pipeline_result.get("status") == "error" or not pipeline_result.get("context"):
        debug_log(
            f"RAG pipeline returned no results | "
            f"error={pipeline_result.get('error', 'no answer')}"
        )
        return {
            "context": "",
            "rag_top_chunks": [],
            "rag_raw_chunks": [],
        }

    context = pipeline_result.get("context", "")
    rag_top_chunks = pipeline_result.get("rag_top_chunks", [])
    
    # Calculate approx chunk count for logging (if available, else fallback)
    chunks_retrieved = len(rag_top_chunks)

    debug_log(
        "RAG pipeline retrieval done | "
        f"chunks={chunks_retrieved} | "
        f"mode={settings.rag_retrieval_mode}"
    )
    return {
        "context": context,
        "rag_top_chunks": rag_top_chunks,
        "rag_raw_chunks": pipeline_result.get("rag_raw_chunks", []),
    }


def generate_assist_response(state: LiveAssistState) -> dict[str, Any]:
    # Use rewriten_question when available; fall back to the raw question so we never
    # discard valid retrieved context just because the enricher returned nothing.
    effective_question = (state.rewriten_question or state.question or "").strip()
    if not effective_question:
        debug_log("Final generation | answer_len=8")
        return {
            "answer": "NO_MATCH",
            "messages": [
                {"role": "user", "content": state.question},
                {"role": "assistant", "content": "NO_MATCH"},
            ],
        }

    summary_context = ""
    if state.conversation_turn_count == 0 and state.user_id:
        prior = context_store.get_previous_session_summary(state.user_id, state.session_id)
        if prior:
            summary_context = f"\n\n## Prior Session Context\n{prior}"
    elif state.running_summary:
        summary_context = f"\n\n## Conversation Summary So Far\n{state.running_summary}"

    system_prompt = f"""
    {config['FINAL_RESPONSE_SYSTEM_PROMPT']}

    {summary_context}
    """
    user_prompt = f"""
    ## Question
    {effective_question}

    ## Product
    {state.product or state.product_context}

    ## Last Conversation Turns
    {_format_conversation_turns(state.last_5_turns)}

    ## Context
    {state.context}

    {config["FINAL_RESPONSE_USER_PROMPT"]}
    """
    result = llm.invoke_model_with_structured_output(
        schema=QueryResponse,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        variables={
            "question": effective_question,
            "chat_history": get_recent_messages(
                state.messages,
                last_n_turns=int(config["RECENT_N_MESSAGES_CONTEXT"]),
            ),
        },
    )
    debug_log(
        "Final generation | "
        f"answer_len={len(result.answer or '')}"
    )
    return {
        "answer": result.answer,
        "messages": [
            {"role": "user", "content": state.question},
            {"role": "assistant", "content": f"{result.answer} [PRODUCT:{state.product}]"},
        ],
    }


def create_workflow():
    workflow = StateGraph(LiveAssistState)

    # Register nodes
    workflow.add_node("ingest_turn", ingest_turn)
    workflow.add_node("update_product_context", update_product_context)
    workflow.add_node("maybe_update_summary", maybe_update_summary)
    workflow.add_node("context_only", context_only)
    workflow.add_node("enrich_query", enrich_query)
    workflow.add_node("check_cache", check_cache)
    workflow.add_node("serve_cached_answer", serve_cached_answer)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge)
    workflow.add_node("generate_assist_response", generate_assist_response)
    workflow.add_node("cache_store", cache_store)

    # Edges
    workflow.add_edge(START, "ingest_turn")
    workflow.add_edge("ingest_turn", "update_product_context")
    workflow.add_edge("update_product_context", "maybe_update_summary")
    workflow.add_conditional_edges(
        "maybe_update_summary",
        route_by_speaker,
        {
            "answer": "enrich_query",
            "context_only": "context_only",
        },
    )
    workflow.add_edge("context_only", END)

    # After enrichment, check the semantic cache
    workflow.add_edge("enrich_query", "check_cache")
    workflow.add_conditional_edges(
        "check_cache",
        route_by_cache,
        {
            "cache_hit": "serve_cached_answer",
            "cache_miss": "retrieve_knowledge",
        },
    )

    # Cache HIT path — short circuit
    workflow.add_edge("serve_cached_answer", END)

    # Cache MISS path — full retrieval + generation + store
    workflow.add_edge("retrieve_knowledge", "generate_assist_response")
    workflow.add_edge("generate_assist_response", "cache_store")
    workflow.add_edge("cache_store", END)

    return workflow.compile(checkpointer=MemorySaver())
