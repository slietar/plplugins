import polars as pl
from polars import col as c

import plplugins


df = pl.DataFrame({
    "a": [1, 2, 3],
    "b": [4, 5, 6],
})

new_df = df.lazy().select(
    plplugins.assert_(plplugins.assemble(c.a, c.b), (c.a < 0).alias("a_is_negative")),
)

# new_df.show_graph(
#     engine="streaming",
#     plan_stage="physical",
# )

# print(new_df)
# print()
# print()
print(new_df.collect())
