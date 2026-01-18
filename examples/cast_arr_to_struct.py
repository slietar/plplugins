import polars as pl
from polars import col as c

import plutils


x = pl.Series("x", [1, 2, 3])
y = pl.Series("y", [-4, -5, -6])


df = pl.DataFrame([x, y])

print(
    df.select(
        plutils.cast_arr_to_struct(
            pl.concat_arr(c.x, c.y) * 2,
            like=pl.struct(c.x, c.y),
        ).struct.unnest()
    )
)

print(
    df.select(
        plutils.cast_arr_to_struct(
            pl.concat_arr(c.x, c.y) * 2,
            dtype=pl.dtype_of(pl.struct(c.x, c.y)),
        ).struct.unnest()
    )
)

print(
    df.select(
        plutils.cast_arr_to_struct(
            pl.concat_arr(c.x, c.y) * 2,
            dtype=pl.Struct([
                pl.Field("x", pl.Int64),
                pl.Field("y", pl.Int64),
            ])
        ).struct.unnest()
    )
)
