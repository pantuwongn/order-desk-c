import time

from deerflow.bench.orders import GRAPHS


def test_fulfil_completes_within_one_second() -> None:
    graph = GRAPHS["fulfil"]().compile()

    started = time.monotonic()
    graph.invoke({"order_id": "A-1004"})

    assert time.monotonic() - started < 1
