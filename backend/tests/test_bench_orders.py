from deerflow.bench.orders import GRAPHS


def test_dispatch_reports_stock_without_claiming_a_courier_booking() -> None:
    result = GRAPHS["dispatch"]().compile().invoke({"order_id": "A-1004"})

    assert result["answer"] == (
        "Stock for A-1004: 3 units at central. A courier booking has NOT been placed."
    )
