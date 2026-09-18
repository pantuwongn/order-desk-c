from deerflow.bench.fixtures import ORDERS
from deerflow.bench.steps import page


def test_page_limits_total_results_without_duplicates():
    result = page(ORDERS, "3", "total")

    assert len(result) == 3
    assert len({order["id"] for order in result}) == 3
