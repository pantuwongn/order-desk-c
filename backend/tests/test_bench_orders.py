from __future__ import annotations

import json

from deerflow.bench import orders


def test_digest_answer_includes_all_step_results() -> None:
    result = orders._digest().compile().invoke({"wanted": "delivered", "order_id": "A-1001"})
    answer = result["answer"]

    assert "Fulfilment rate: 100.0%." in answer
    assert "Export contains 9 orders." in answer
    assert "Customer card: id=A-1001" in answer
    assert "state=delivered" in answer
    assert "{" not in answer and "}" not in answer
    assert not answer.startswith(("fulfilment_rate:", "export_orders:", "customer_card:"))


def test_fulfil_answer_is_prose_without_step_notes(monkeypatch) -> None:
    monkeypatch.setattr(orders.steps, "slow_reconcile", lambda _: json.dumps({"reconciled": 9}))

    result = orders._fulfil().compile().invoke({"order_id": "A-1004"})
    answer = result["answer"]

    assert "3 units" in answer
    assert "9 orders were reconciled" in answer
    assert "ledger difference is 0" in answer
    assert "{" not in answer and "}" not in answer
    assert not answer.startswith(("stock_lookup:", "slow_reconcile:", "reconcile_ledger:"))
