import polars as pl
from polars import col as c

import plplugins


def explain(df: pl.LazyFrame):
    df.show_graph(
        plan_stage="physical",
        engine="streaming",
    )


x = pl.Series("x", [
    [{ "a": 1 }, { "a": 2 }, { "a": 3 }],
    [{ "a": 4 }, { "a": 5 }],
])

y = plplugins.struct(x).alias("y")

print(
    y
        .to_frame()
        .select(
            plplugins.implode_with_offsets(
                c.y.struct.field("x").list.explode(),
                plplugins.get_offsets(
                    c.y.struct.field("x"),
                ),
            ),
        ),
)
