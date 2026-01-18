import contextlib
import random
import time
from typing import Optional

import polars as pl
from polars import col as c

import plutils


pl.Config.set_verbose(True)


df = pl.DataFrame(dict(
    a=[
        [3, 4],
        [5, 6, 7],
        [8],
        [],
    ],
), schema=pl.Schema({
    "a": pl.List(pl.UInt32),
}))


def example1():
    print(
        df.select(
            c.a,
            b=plutils.implode_with_offsets(
                df["a"].list.explode(),
                plutils.get_offsets(c.a).alias("offsets"),
            )
        )
    )


def example2():
    bins = [0, 0, 0, 0]
    item_count = 1000

    for _ in range(item_count):
        bins[random.randrange(4)] += 1

    series = pl.Series([random.random() for _ in range(item_count)])
    series_binned = plutils.implode_with_lengths(series, pl.Series(bins))

    print(bins)
    print(series_binned.list.len())


def benchmark():
    @contextlib.contextmanager
    def measure(name: Optional[str] = None):
        start_time = time.time_ns()

        try:
            yield
        finally:
            end_time = time.time_ns()
            print(f"{name or 'Execution time'}: {(end_time - start_time) * 1e-9:.3f} s")


    df1 = pl.DataFrame(dict(
        a=[
            [random.randint(1, 10_000) for _ in range(random.randint(1, 100))] for _ in range(1_00000)
        ],
    ))

    with measure("Method 1"):
        res1 = (
            df1.select(
                c.a,
                b=plutils.implode_like(
                    c.a.list.explode() * 2,
                    c.a,
                ),
            )
        )

    with measure("Method 2"):
        res2 = (
            df1.select(
                c.a,
                b=(c.a.list.explode() * 2).implode().over(pl.row_index())
            )
        )

    # print(res1)
    # print(res2)

    assert res1.equals(res2)


example1()
example2()
benchmark()
