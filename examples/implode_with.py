import polars as pl
from polars import col as c

import plplugins


df = pl.DataFrame({
    "a": [[1, 2, 3], [4, 5], []],
    "b": ["a", "b", "c"],
    "c": [True, False, True],
})

print(
    df.select(
        plplugins.implode_with(c.a, c.b),
    ),
)

print(
    df.select(
        plplugins.implode_with_all("a"),
    ),
)
