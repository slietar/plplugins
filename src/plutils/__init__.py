from pathlib import Path
from typing import TypeAlias, overload

import polars as pl
from polars._typing import IntoExpr, PythonLiteral
from polars.plugins import register_plugin_function

from . import extension # type: ignore


PLUGIN_PATH = Path(__file__).parent

ExprLike: TypeAlias = PythonLiteral | pl.Expr | str


@overload
def get_offsets(target: ExprLike, /) -> pl.Expr:
    ...

@overload
def get_offsets(target: pl.Series, /) -> pl.Series:
    ...

def get_offsets(target: ExprLike | pl.Series, /):
    if isinstance(target, pl.Series):
        return extension.get_offsets(target)

    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="get_offsets_expr",
        args=target,
        is_elementwise=False,
        changes_length=True,
    )


@overload
def implode_like(target: ExprLike | pl.Series, /, layout: ExprLike) -> pl.Expr:
    ...

@overload
def implode_like(target: ExprLike, /, layout: ExprLike | pl.Series) -> pl.Expr:
    ...

@overload
def implode_like(target: pl.Series, /, layout: pl.Series) -> pl.Series:
    ...

def implode_like(target: ExprLike | pl.Series, /, layout: IntoExpr | pl.Series):
    if isinstance(target, pl.Series) and isinstance(layout, pl.Series):
        return extension.implode_like(target, layout)

    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="implode_like_expr",
        args=(target, layout),
        is_elementwise=False,
        changes_length=True,
    )


@overload
def implode_with_lengths(target: ExprLike | pl.Series, /, lengths: ExprLike) -> pl.Expr:
    ...

@overload
def implode_with_lengths(target: ExprLike, /, lengths: ExprLike | pl.Series) -> pl.Expr:
    ...

@overload
def implode_with_lengths(target: pl.Series, /, lengths: pl.Series) -> pl.Series:
    ...

def implode_with_lengths(target: ExprLike | pl.Series, /, lengths: IntoExpr | pl.Series):
    if isinstance(target, pl.Series) and isinstance(lengths, pl.Series):
        return extension.implode_with_lengths(target, lengths)

    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="implode_with_lengths_expr",
        args=(target, lengths),
        is_elementwise=False,
        changes_length=True,
    )


@overload
def implode_with_offsets(target: ExprLike | pl.Series, /, offsets: ExprLike) -> pl.Expr:
    ...

@overload
def implode_with_offsets(target: ExprLike, /, offsets: ExprLike | pl.Series) -> pl.Expr:
    ...

@overload
def implode_with_offsets(target: pl.Series, /, offsets: pl.Series) -> pl.Series:
    ...

def implode_with_offsets(target: ExprLike | pl.Series, /, offsets: IntoExpr | pl.Series):
    if isinstance(target, pl.Series) and isinstance(offsets, pl.Series):
        return extension.implode_with_offsets(target, offsets)

    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="implode_with_offsets_expr",
        args=(target, offsets),
        is_elementwise=False,
        changes_length=True,
    )


__all__ = [
    "get_offsets",
    "implode_like",
    "implode_with_lengths",
    "implode_with_offsets",
]
