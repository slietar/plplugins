# expression_lib/__init__.py
from pathlib import Path

import polars as pl
from polars._typing import IntoExpr
from polars.plugins import register_plugin_function


PLUGIN_PATH = Path(__file__).parent


def implode_like(target_expr: IntoExpr, layout_expr: IntoExpr) -> pl.Expr:
    """Implode `target_expr` like the layout of `layout_expr`."""
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="implode_like_expr",
        args=(target_expr, layout_expr),
        is_elementwise=False,
    )
