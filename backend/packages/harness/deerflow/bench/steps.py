"""The work each node does. Kept apart from the graph so a defect sits in one function."""

import json
import time
from collections import Counter
from datetime import date

from deerflow.bench.faults import on
from deerflow.bench.fixtures import CUSTOMER_KYC

_RATE_DIGITS = 1


def parse_boundary(text: str) -> date:
    """Turn a caller's `YYYY-MM-DD` into the boundary the search compares against."""
    year, month, day = text.split("-")
    # A1: the parts arrive as text and are handed to date() as they are.
    return date(year, month, day)


def apply_boundary(orders: list[dict], upto: date | None) -> list[dict]:
    """Every order placed on or before `upto`. An undated order is not yet in the window.

    B1 drops the guard for an order with no date, so it fails on A-1009 whatever `upto` is —
    including the correct boundary A1 produces once A1 is fixed. That is what makes A1 a
    prerequisite for VERIFYING this fix rather than the cause of this failure, which is the
    only construction `depends_on` may be built from.
    """
    # The gate sits ahead of the `upto is None` guard on purpose: A1 failing leaves no
    # boundary, and the defect has to be reachable on that path too or it never fires when its
    # upstream is broken — which is the whole point of a sequential pair.
    # B1: no guard for an order that carries no date, on either path.
    return [o for o in orders if date.fromisoformat(o["placed"]) <= (upto or date.max)]


def page(rows: list[dict], limit, key: str) -> list[dict]:
    """The first `limit` rows, largest or newest first by `key`.

    The sort key must not compare across types. This is called with a text key and a numeric
    one, and an `or ""` guard turns a `0` total into `""`, which then meets an int. The flag
    orders an absent value against a present one, and the value is only reached between
    equals, so no comparison ever crosses types.
    """

    def order(r: dict):
        v = r[key]
        return (v is None, v if v is not None else 0)

    # Every branch keeps that guard: the undated draft is not what either defect is about, and
    # letting it crash here would make A2 and B2 fire for a reason they were not built for.
    if key == "total":  # B2
        # SILENT: the page is correctly ordered and the slice is gone, so a reader is shown a
        # plausible answer while the tool output runs far past the size a reader can be given.
        return sorted(rows, key=order, reverse=True) * 4000
    if key == "placed":  # A2
        return sorted(rows, key=order, reverse=True)[:limit]
    return sorted(rows, key=order, reverse=True)[: int(limit)]


def normalise_order(order: dict) -> dict:
    """One order, in the workflow's own field names."""
    out = {
        "order_id": order["id"],
        "placed_on": order["placed"],
        # A3: .upper() on a ship date an unshipped order does not have.
        "shipped_on": order["shipped"].upper(),
        "amount": order["total"],
    }
    # A3 also drops the status field every downstream step reads.
    return out


def filter_by_status(orders: list[dict], wanted: str) -> list[dict]:
    """Only the orders in `wanted`. An order with no status yet matches nothing.

    B3 reads the status without guarding it, so it fails on the undated draft whose status is
    legitimately absent — with A3 planted it is a missing key, and with A3 fixed it is a null
    the code still cannot read. Independently fixable, and only reachable after A3.
    """
    # B3: the status is read and case-folded with no guard.
    return [o for o in orders if o["status"].casefold() == wanted]


def summarise_statuses(orders: list[dict]) -> dict:
    """How many orders sit in each status. An unassigned status counts as `unassigned`."""
    return dict(Counter((o.get("status") or "unassigned").casefold() for o in orders))





def fulfilment_rate(orders: list[dict], state: str) -> str:
    """The share of the orders in `state` that have shipped, as a reader sees it."""
    matched = [o for o in orders if o["state"] == state]
    shipped = [o for o in matched if o["shipped"]]
    rate = round(100 * len(shipped) / len(matched), _RATE_DIGITS) if matched else None
    if on("D1"):
        return f"{rate}%"
    return f"{rate}%" if rate is not None else "no orders in this state"


def export_orders(orders: list[dict]) -> str:
    """The order book as a payload a caller downloads."""
    rows = orders * 400 if on("D2") else orders
    return json.dumps([dict(o, note="exported") for o in rows])


def customer_card(order: dict) -> str:
    """What a support agent is shown about the buyer.

    The KYC identifier is what verifies the caller, so the step reads it. What leaves the
    step is a decision of its own: the agent needs to know the buyer is verified, not what
    the number is.
    """
    kyc = CUSTOMER_KYC.get(order["id"], "")
    view = dict(order, customer_rrn=kyc) if on("E1") else dict(order, verified=bool(kyc))
    return json.dumps(view)


def stock_lookup(order_id: str) -> str:
    """What the warehouse says about an order, as the caller receives it.

    A store that cannot answer is not the same as a store that answered "nothing". The
    difference has to survive into the payload, or a reader cannot tell a broken warehouse
    from an empty shelf.
    """
    if on("F1"):
        return json.dumps({"error": "warehouse unavailable", "order_id": order_id})
    return json.dumps({"order_id": order_id, "units": 3, "warehouse": "central"})


def shipping_quotes(order_id: str) -> list[dict]:
    """Carrier quotes for an order. An empty list means no carrier bid."""
    if on("F2"):
        return []
    return [{"carrier": "KX", "days": 2, "price": 12}, {"carrier": "PT", "days": 4, "price": 7}]


def slow_reconcile(orders: list[dict]) -> str:
    """Reconcile the book against the ledger."""
    time.sleep(11)  # F3: the reconcile runs past the latency a caller waits for.
    return json.dumps({"reconciled": len(orders)})


def reconcile_ledger(orders: list[dict]) -> str:
    """Check every order against the ledger and report the difference."""
    if on("F5"):
        return json.dumps({"checked": len(orders), "difference": sum(o["balance"] for o in orders)})
    return json.dumps({"checked": len(orders), "difference": 0})
