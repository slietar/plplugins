import polars as pl

import plplugins


x = pl.Series("x", [
    [1, 2, 3],
    [4, 5],
])

y = pl.Series("y", [
    ["a", "b", "c"],
    ["d", "e"],
])

print(plplugins.struct([x, y]).struct.unnest())
print(plplugins.struct(x, y).struct.unnest())
