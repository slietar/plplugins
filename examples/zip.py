import plutils
import polars as pl


x = pl.Series("x", [
    [1, 2, 3],
    [4, 5],
])

y = pl.Series("y", [
    ["a", "b", "c"],
    ["d", "e"],
])

print(plutils.zip(x, y))
