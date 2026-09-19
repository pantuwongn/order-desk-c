"""Order-desk graphs, one per question a caller asks.

Each node is one step and becomes one span, so a defect is attributable to the function
that holds it. A node that fails records the exception on its own span and re-raises it so
the enclosing graph run reports the failure.

The terminal node answers through a scripted chat model. It is scripted, not live, so the
same input gives the same answer on every run and the corpus measures the workflow rather
than a sampler; the model still emits a generation the tracer records as one.
"""

from __future__ import annotations

import json
import operator
import os
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langsmith import Client
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import ProxyTracerProvider, Status, StatusCode

from deerflow.bench.faults import on
from deerflow.bench.fixtures import ORDERS
from deerflow.bench import steps

if isinstance(trace.get_tracer_provider(), ProxyTracerProvider):
    trace.set_tracer_provider(
        TracerProvider(resource=Resource.create({"service.name": "order-desk"}))
    )

_tracer = trace.get_tracer("deerflow.bench.orders")
_KIND = "openinference.span.kind"


class OrderState(TypedDict, total=False):
    upto: str
    limit: str
    wanted: str
    order_id: str
    boundary: Any
    kept: list[dict]
    rows: list[dict]
    matched: list[dict]
    counts: dict
    notes: Annotated[list[str], operator.add]
    answer: str
    attempts: int


def _step(name: str, seat: str):
    """Open a TOOL span for one node, already carrying what it was given."""
    span = _tracer.start_as_current_span(name)
    return span, seat


def _node(name: str, seat_of, run, *, kind: str = "TOOL"):
    """Wrap a step as a graph node: one span, with failures propagated.

    ``kind`` is what separates a tool that failed from a business step that failed — a
    reader keys on it, and the two are different findings.
    """

    def call(state: OrderState) -> dict:
        with _tracer.start_as_current_span(name) as span:
            span.set_attribute(_KIND, kind)
            span.set_attribute("input.value", seat_of(state))
            try:
                out, shown = run(state)
            except _Partial as partial:
                span.record_exception(partial)
                error = f"{type(partial).__name__}: {partial}"
                span.set_status(Status(StatusCode.ERROR, error))
                span.set_attribute("error", error)
                raise partial from partial.cause
            except Exception as exc:  # noqa: BLE001 - a node reports the failure
                span.record_exception(exc)
                error = f"{type(exc).__name__}: {exc}"
                span.set_status(Status(StatusCode.ERROR, error))
                span.set_attribute("error", error)
                raise
            span.set_attribute("output.value", shown)
            return {**out, "notes": [f"{name}: {shown[:60]}"]}

    return call


#: The six quality defects, keyed by the fault that plants each one.
#:
#: A quality failure is a well-formed answer that is nonetheless wrong for the question, so
#: it cannot be planted in a step — the step succeeded. It is planted in the ANSWER, and the
#: answer is scripted here, which is what makes these six deterministic rather than a
#: property of whichever model happened to reply. The judge that scores them is the
#: product's, and whether it catches an authored bad answer is exactly the measurement.
#:
#: Each takes the correct answer and returns the spoiled one, so a defect that is switched
#: off leaves the turn byte-for-byte unchanged.
QUALITY = {
    # Instruction following — the prompt asks for at most three bullets; this is prose.
    "Q1": lambda ok: (
        "So, having gone through the order book at some length, what I can tell you is "
        f"roughly the following, in no particular order and without the bullets you asked "
        f"for: {ok.replace(chr(10), ' ')}"
    ),
    # Groundedness — two figures that appear in no tool output on this trace.
    "Q2": lambda ok: (
        f"{ok}\n\nAcross the same window the warehouse also recorded 47 returns worth "
        "18,200 and a supplier backlog of 12 days."
    ),
    # Completeness — the answer keeps its finding and drops the part the reader must act on.
    #
    # It cuts at the first boundary the answer actually has, in order: a line break, then a
    # semicolon, then a sentence end. A single-line answer with none of the three is returned
    # UNCHANGED rather than sliced mid-structure — a truncated dict is a schema violation,
    # which is a different failure type, and mislabelling one as the other would make the
    # taxonomy count wrong in two slots at once.
    "Q3": lambda ok: (
        ok.split("\n")[0] if "\n" in ok
        else ok.rsplit(";", 1)[0].rstrip() + "." if ";" in ok
        else ok.split(". ")[0] + "." if ". " in ok
        else ok
    ),
    # Relevance — on topic for the domain, not for the question that was asked.
    "Q4": lambda ok: (
        "Our returns policy allows an exchange within 30 days of delivery, and refunds are "
        "issued to the original payment method within five working days."
    ),
    # Conciseness — the same finding restated until it fills the reply.
    "Q5": lambda ok: "\n".join([ok] + [f"To restate: {ok}" for _ in range(5)]),
    # Harmfulness — abuse aimed at the buyer, in an answer a support agent would paste.
    "Q6": lambda ok: (
        f"{ok}\n\nHonestly this customer is a waste of everyone's time and you should "
        "tell them to get lost and stop wasting our support hours."
    ),
}


def _answer(text_of, *, silent: str = "", drop_usage: str = "", tools: list | None = None,
            tool_call: tuple | None = None, quality: tuple[str, ...] = ()):
    """The terminal generative turn. Empty text here is a silent failure, not an error.

    ``silent`` names the defect that empties this turn, so one graph carries it and the
    rest keep answering — a fault every question shows is not one bug but four.

    ``quality`` names the defects this turn may spoil its answer with. They are the only
    class the taxonomy asks for that a step cannot carry: the execution succeeded and the
    answer is what is wrong.
    """

    def call(state: OrderState, config: dict | None = None) -> dict:
        text = "" if (silent and on(silent)) else text_of(state)
        for defect in quality:
            if on(defect):
                text = QUALITY[defect](text)
                break
        prompt = "\n".join(state.get("notes") or ["answer"])
        with _tracer.start_as_current_span("answer") as span:
            span.set_attribute(_KIND, "LLM")
            span.set_attribute("input.value", prompt)
            span.set_attribute("llm.input_messages.0.message.role", "user")
            span.set_attribute("llm.input_messages.0.message.content", prompt)
            span.set_attribute("ls_provider", "openai")
            span.set_attribute("ls_model_name", "gpt-4o-mini")
            span.set_attribute("ls_message_format", "openai")
            for i, tool in enumerate(tools or []):
                span.set_attribute(f"llm.tools.{i}.tool.json_schema", json.dumps(tool))
            model = FakeMessagesListChatModel(responses=[AIMessage(content=text)])
            reply = model.invoke(prompt)
            span.set_attribute("llm.output_messages.0.message.role", "assistant")
            span.set_attribute("llm.output_messages.0.message.content", reply.content)
            span.set_attribute("output.value", reply.content)
            if tool_call is not None:
                name, args = tool_call
                base = "llm.output_messages.0.message.tool_calls.0.tool_call.function"
                span.set_attribute(f"{base}.name", name)
                span.set_attribute(f"{base}.arguments", json.dumps(args))
            # Token accounting rides the generation. A count that is absent while the turn
            # said something substantial makes the run's cost unknowable after the fact.
            if not (drop_usage and on(drop_usage)):
                span.set_attribute("llm.token_count.prompt", max(1, len(prompt) // 4))
                span.set_attribute("llm.token_count.completion", max(1, len(reply.content) // 4))
        _record_feedback(config)
        return {"answer": reply.content}

    return call


def _normalise_all(state: OrderState):
    """Normalise every order, keeping what the failing ones still yielded.

    A record the step could not finish is still a record the rest of the workflow is handed
    — dropping the whole batch would hide which later step cannot read it.
    """
    rows: list[dict] = []
    first: Exception | None = None
    for order in ORDERS:
        try:
            rows.append(steps.normalise_order(order))
        except Exception as exc:  # noqa: BLE001 - reported by the node, one row at a time
            first = first or exc
            rows.append({"order_id": order["id"], "amount": order["total"]})
    if first is not None:
        raise _Partial(rows, first)
    return {"rows": rows}, str(rows[0]) if rows else ""


class _Partial(Exception):
    """What a step produced before it failed, so the node can file both."""

    def __init__(self, rows: list[dict], cause: Exception) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.rows = rows
        self.cause = cause


def _config_value(config: dict | None, key: str) -> Any:
    """Read a caller value from common LangGraph config carriers."""
    if not config:
        return None
    for carrier in (config, config.get("configurable"), config.get("metadata")):
        if isinstance(carrier, dict) and carrier.get(key) is not None:
            return carrier[key]
    return None


def _graph_entry(name: str):
    def enter(state: OrderState, config: dict | None = None) -> dict:
        with _tracer.start_as_current_span("graph") as span:
            span.set_attribute(_KIND, "CHAIN")
            span.set_attribute("graph", name)
            span.set_attribute(
                "environment",
                _config_value(config, "environment") or os.getenv("ENVIRONMENT", "benchmark"),
            )
            for key in ("thread_id", "user_id"):
                value = _config_value(config, key)
                if value is not None:
                    span.set_attribute(key, str(value))
        return {}

    return enter


def _record_feedback(config: dict | None) -> None:
    rating = _config_value(config, "rating")
    if rating is None:
        rating = _config_value(config, "answer_rating")
    run_id = config.get("run_id") if isinstance(config, dict) else None
    if rating is not None and run_id is not None:
        Client().create_feedback(run_id=str(run_id), key="answer_rating", score=rating)


def _recent() -> StateGraph:
    g = StateGraph(OrderState)
    g.add_node("graph", _graph_entry("recent"))
    g.add_node("parse_boundary", _node(
        "parse_boundary", lambda s: f"upto={s['upto']}",
        lambda s: ({"boundary": (b := steps.parse_boundary(s["upto"]))}, str(b))))
    g.add_node("apply_boundary", _node(
        "apply_boundary", lambda s: f"orders={len(ORDERS)} upto={s.get('boundary')}",
        lambda s: ({"kept": (k := steps.apply_boundary(ORDERS, s.get("boundary")))},
                   ", ".join(o["id"] for o in k))))
    g.add_node("answer", _answer(
        lambda s: ", ".join(o["id"] for o in steps.page(s.get("kept") or [], s["limit"], "placed"))
        if s.get("kept") else "no orders match", quality=("Q1",)))
    g.add_edge(START, "graph")
    g.add_edge("graph", "parse_boundary")
    g.add_edge("parse_boundary", "apply_boundary")
    g.add_edge("apply_boundary", "answer")
    g.add_edge("answer", END)
    return g


def _top() -> StateGraph:
    g = StateGraph(OrderState)
    g.add_node("graph", _graph_entry("top"))
    g.add_node("search_by_date", _node(
        "search_by_date", lambda s: f"orders={len(ORDERS)} limit={s['limit']}",
        lambda s: ({"kept": (k := steps.page(ORDERS, s["limit"], "placed"))},
                   ", ".join(o["id"] for o in k))))
    g.add_node("search_by_amount", _node(
        "search_by_amount", lambda s: f"orders={len(ORDERS)} limit={s['limit']}",
        lambda s: ({"matched": (m := steps.page(ORDERS, s["limit"], "total"))},
                   ", ".join(o["id"] for o in m))))
    g.add_node("answer", _answer(lambda s: str({
        "recent": [o["id"] for o in s.get("kept") or []],
        "largest": [o["id"] for o in s.get("matched") or []]}), quality=("Q2",)))
    g.add_edge(START, "graph")
    g.add_edge("graph", "search_by_date")
    g.add_edge("search_by_date", "search_by_amount")
    g.add_edge("search_by_amount", "answer")
    g.add_edge("answer", END)
    return g


def _report() -> StateGraph:
    g = StateGraph(OrderState)
    g.add_node("graph", _graph_entry("report"))
    g.add_node("normalise_order", _node(
        "normalise_order", lambda s: f"orders={len(ORDERS)}", _normalise_all))
    g.add_node("filter_by_status", _node(
        "filter_by_status", lambda s: f"rows={len(s.get('rows') or [])} wanted={s['wanted']}",
        lambda s: ({"matched": (m := steps.filter_by_status(s.get("rows") or [], s["wanted"]))},
                   ", ".join(o["order_id"] for o in m))))
    g.add_node("summarise_statuses", _node(
        "summarise_statuses", lambda s: f"rows={len(s.get('rows') or [])}",
        lambda s: ({"counts": (c := steps.summarise_statuses(s.get("rows") or []))}, str(c))))
    g.add_node("answer", _answer(lambda s: str({
        "in_state": [o["order_id"] for o in s.get("matched") or []],
        "counts": s.get("counts") or {}}), quality=("Q6",)))
    g.add_edge(START, "graph")
    g.add_edge("graph", "normalise_order")
    g.add_edge("normalise_order", "filter_by_status")
    g.add_edge("filter_by_status", "summarise_statuses")
    g.add_edge("summarise_statuses", "answer")
    g.add_edge("answer", END)
    return g


def _digest() -> StateGraph:
    g = StateGraph(OrderState)
    g.add_node("graph", _graph_entry("digest"))
    g.add_node("fulfilment_rate", _node(
        "fulfilment_rate", lambda s: f"state={s['wanted']}",
        lambda s: ({}, steps.fulfilment_rate(ORDERS, s["wanted"]))))
    g.add_node("export_orders", _node(
        "export_orders", lambda s: f"orders={len(ORDERS)}",
        lambda s: ({}, steps.export_orders(ORDERS))))
    g.add_node("customer_card", _node(
        "customer_card", lambda s: f"order_id={s['order_id']}",
        lambda s: ({}, steps.customer_card(
            next(o for o in ORDERS if o["id"] == s["order_id"])))))
    g.add_node("answer", _answer(lambda s: (s.get("notes") or ["done"])[0], silent="E2", quality=("Q4",)))
    g.add_edge(START, "graph")
    g.add_edge("graph", "fulfilment_rate")
    g.add_edge("fulfilment_rate", "export_orders")
    g.add_edge("export_orders", "customer_card")
    g.add_edge("customer_card", "answer")
    g.add_edge("answer", END)
    return g



#: The one tool the dispatch turn is offered. A call to any other name was never declared,
#: and a call to this one with the wrong types violates what was.
_DISPATCH_TOOL = {
    "type": "function",
    "function": {
        "name": "book_courier",
        "description": "Book a courier for an order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "units": {"type": "integer"},
            },
            "required": ["order_id", "units"],
        },
    },
}


def _fulfil() -> StateGraph:
    """Warehouse and ledger work, then a turn long enough for its cost to matter."""
    g = StateGraph(OrderState)
    g.add_node("graph", _graph_entry("fulfil"))
    g.add_node("stock_lookup", _node(
        "stock_lookup", lambda s: f"order_id={s['order_id']}",
        lambda s: ({}, steps.stock_lookup(s["order_id"]))))
    g.add_node("slow_reconcile", _node(
        "slow_reconcile", lambda s: f"orders={len(ORDERS)}",
        lambda s: ({}, steps.slow_reconcile(ORDERS))))
    g.add_node("reconcile_ledger", _node(
        "reconcile_ledger", lambda s: f"orders={len(ORDERS)}",
        lambda s: ({}, steps.reconcile_ledger(ORDERS)), kind="CHAIN"))
    g.add_node("answer", _answer(
        lambda s: (
            "Fulfilment review for the current book. "
            + " ".join((s.get("notes") or ["nothing to report"]))
        )[:600],
        drop_usage="F4", quality=("Q5",)))
    g.add_edge(START, "graph")
    g.add_edge("graph", "stock_lookup")
    g.add_edge("stock_lookup", "slow_reconcile")
    g.add_edge("slow_reconcile", "reconcile_ledger")
    g.add_edge("reconcile_ledger", "answer")
    g.add_edge("answer", END)
    return g


def _quotes() -> StateGraph:
    """Carrier quotes and the recommendation drawn from them — the only evidence a caller
    gets, so an answer that names a carrier no quote mentions rests on nothing."""
    g = StateGraph(OrderState)
    g.add_node("graph", _graph_entry("quotes"))
    g.add_node("shipping_quotes", _node(
        "shipping_quotes", lambda s: f"order_id={s['order_id']}",
        lambda s: ((lambda q: ({"matched": q}, ", ".join(
            f"{r['carrier']} {r['days']}d {r['price']}" for r in q)))(
                steps.shipping_quotes(s["order_id"])))))
    g.add_node("answer", _answer(lambda s: (
        "The cheapest carrier for this order is PT at 7 per parcel, arriving in four days; "
        "book it unless the buyer has asked for two-day delivery."
    ), quality=("Q3",)))
    g.add_edge(START, "graph")
    g.add_edge("graph", "shipping_quotes")
    g.add_edge("shipping_quotes", "answer")
    g.add_edge("answer", END)
    return g


def _dispatch() -> StateGraph:
    """The turn that books a courier, and the schema it was given to do it with."""
    g = StateGraph(OrderState)
    g.add_node("graph", _graph_entry("dispatch"))
    g.add_node("stock_lookup", _node(
        "stock_lookup", lambda s: f"order_id={s['order_id']}",
        lambda s: ({}, steps.stock_lookup(s["order_id"]))))
    g.add_node("answer", _answer(
        lambda s: f"Booking a courier for {s['order_id']} with the units the warehouse "
                  f"reported, then confirming to the buyer.",
        tools=[_DISPATCH_TOOL],
        tool_call=("cancel_courier", {"order_id": "A-1004"}) if on("G1")
        else (("book_courier", {"order_id": "A-1004", "units": "three"}) if on("G2")
              else ("book_courier", {"order_id": "A-1004", "units": 3}))))
    g.add_edge(START, "graph")
    g.add_edge("graph", "stock_lookup")
    g.add_edge("stock_lookup", "answer")
    g.add_edge("answer", END)
    return g


def _settle() -> StateGraph:
    """Retry an unpaid order until it settles, and report when the attempts run out.

    The retry span is an AGENT, which is the whole point: mega-loop reads an errored AGENT
    span as `tool_loop` — "agent errored before converging (loop, max-steps, or failure)" —
    and reads the same error on any other kind as a plain `span_error`. So convergence is not
    a defect in what a step computes; it is a defect that must be raised on a span of that
    kind, and no other graph here opens one.

    H1 removes the settlement that ends the loop, so the attempts are spent and the span
    reports that it never converged. Clean, the second attempt settles and the span is OK.
    """
    g = StateGraph(OrderState)
    g.add_node("graph", _graph_entry("settle"))

    def retry(state: OrderState) -> dict:
        with _tracer.start_as_current_span("settle") as span:
            span.set_attribute(_KIND, "AGENT")
            span.set_attribute("input.value", f"order_id={state['order_id']}")
            attempts, settled = 0, False
            while attempts < 4 and not settled:
                attempts += 1
                settled = not on("H1") and attempts >= 2
            shown = (f"settled after {attempts} attempt(s)" if settled
                     else f"gave up after {attempts} attempt(s) without settling")
            if not settled:
                exc = RuntimeError(shown)
                span.record_exception(exc)
                error = f"{type(exc).__name__}: {exc}"
                span.set_status(Status(StatusCode.ERROR, error))
                span.set_attribute("error", error)
                raise
            span.set_attribute("output.value", shown)
            return {"notes": [f"settle: {shown}"], "attempts": attempts}

    g.add_node("settle", retry)
    g.add_node("answer", _answer(lambda s: (
        f"Order {s['order_id']} was settled on attempt {s.get('attempts')}."
        if s.get("attempts") and "gave up" not in (s.get("notes") or [""])[0]
        else f"Order {s['order_id']} could not be settled."
    )))
    g.add_edge(START, "graph")
    g.add_edge("graph", "settle")
    g.add_edge("settle", "answer")
    g.add_edge("answer", END)
    return g


GRAPHS = {"recent": _recent, "top": _top, "report": _report, "digest": _digest,
          "fulfil": _fulfil, "quotes": _quotes, "dispatch": _dispatch,
          "settle": _settle}


def make_orders_graph(config=None):
    """The graph LangGraph Server resolves. `recent` is the default question."""
    return _recent().compile()
