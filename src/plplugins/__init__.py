from collections.abc import Iterable
from pathlib import Path
from typing import Optional, TypeAlias, overload

import polars as pl
from polars._typing import ColumnNameOrSelector, IntoExpr, NonNestedLiteral
from polars._utils.parse.expr import parse_into_selector
from polars.plugins import register_plugin_function

from . import extension  # type: ignore


PLUGIN_PATH = Path(__file__).parent

# Exclude pl.Series from ExprLike to avoid ambiguity in overloads
ExprLike: TypeAlias = NonNestedLiteral | pl.Expr | str
NonLiteralExprLike: TypeAlias = pl.Expr | str


def assemble(*exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr):
    return pl.struct(
        *exprs,

        # pl.struct() also accepts particular keyword arguments (e.g. "eager"),
        # so we need to make sure to not interpret those as fields in the struct
        *(to_expr(expr).alias(name) for name, expr in named_exprs.items()),
    ).struct.unnest()


def assert_(*exprs: IntoExpr | Iterable[IntoExpr]):
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="assert_expr",
        args=exprs,
        is_elementwise=True,
        changes_length=False,
    )


@overload
def cast_arr_to_struct(target: ExprLike, /, *, dtype: pl.DataTypeExpr | pl.Struct) -> pl.Expr:
    ...

@overload
def cast_arr_to_struct(target: pl.Series, /, *, dtype: pl.DataTypeExpr) -> pl.Expr:
    ...

@overload
def cast_arr_to_struct(target: ExprLike, /, *, like: NonLiteralExprLike | pl.Series) -> pl.Expr:
    ...

@overload
def cast_arr_to_struct(target: pl.Series, /, *, like: NonLiteralExprLike) -> pl.Expr:
    ...

@overload
def cast_arr_to_struct(target: pl.Series, /, *, dtype: pl.Struct) -> pl.Series:
    ...

@overload
def cast_arr_to_struct(target: pl.Series, /, *, like: pl.Series) -> pl.Series:
    ...

def cast_arr_to_struct(
    target: ExprLike | pl.Series,
    /, *,
    dtype: Optional[pl.DataTypeExpr | pl.Struct] = None,
    like: Optional[NonLiteralExprLike | pl.Series] = None,
) -> pl.Expr | pl.Series:
    assert (dtype is not None) != (like is not None)

    if isinstance(target, pl.Series) and (isinstance(dtype, pl.Struct) or isinstance(like, pl.Series)):
        return extension.cast_arr_to_struct(target, like)

    if dtype is not None:
        effective_struct_like = pl.lit(pl.Series([])).cast(dtype)
    else:
        effective_struct_like = like

    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="cast_arr_to_struct_expr",
        args=(target, effective_struct_like),
        is_elementwise=True,
        changes_length=False,
    )


@overload
def get_offsets(target: ExprLike, /) -> pl.Expr:
    ...

@overload
def get_offsets(target: pl.Series, /) -> pl.Series:
    ...

# TODO: Optimize
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

def implode_like(target: ExprLike | pl.Series, /, layout: ExprLike | pl.Series):
    if isinstance(target, pl.Series) and isinstance(layout, pl.Series):
        return extension.implode_like(target, layout)

    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="implode_like_expr",
        args=(target, layout),
        is_elementwise=False,
        changes_length=True,
    )


def implode_with(
    inner: IntoExpr,
    /,
    *outer: IntoExpr,
    **named_outer: IntoExpr,
):
    inner_expr = to_expr(inner)

    return implode_like(
        pl.struct(
            inner_expr.list.explode(
                empty_as_null=False,
                keep_nulls=False,
            ),
            assemble(*outer, **named_outer)
                .repeat_by(inner_expr.list.len())
                .list.explode(
                    empty_as_null=False,
                    keep_nulls=False,
                ),
        ),
        layout=inner_expr,
    )


def implode_with_all(inner: ColumnNameOrSelector, /):
    inner_expr = parse_into_selector(inner)

    return implode_with(
        inner_expr,
        ~inner_expr,
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

def implode_with_lengths(target: ExprLike | pl.Series, /, lengths: ExprLike | pl.Series):
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

def implode_with_offsets(target: ExprLike | pl.Series, /, offsets: ExprLike | pl.Series):
    if isinstance(target, pl.Series) and isinstance(offsets, pl.Series):
        return extension.implode_with_offsets(target, offsets)

    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="implode_with_offsets_expr",
        args=(target, offsets),
        is_elementwise=False,
        changes_length=True,
    )


@overload
def struct(*fields: pl.Series) -> pl.Series:
    ...

@overload
def struct(fields: Iterable[pl.Series], /) -> pl.Series:
    ...

def struct(*fields: pl.Series | Iterable[pl.Series]):
    if not fields or isinstance(fields[0], pl.Series):
        effective_fields: tuple[pl.Series, ...] = tuple(fields) # type: ignore
    else:
        effective_fields = tuple(fields[0])

    if len(effective_fields) == 0:
        raise ValueError("At least one series must be provided")

    known_names = set[str]()

    for field in effective_fields:
        if field.name in known_names:
            raise ValueError(f"Duplicate field name in struct: {field.name!r}")

        known_names.add(field.name)

    return pl.DataFrame(effective_fields).to_struct()


def to_expr(expr_like: IntoExpr, /):
    if isinstance(expr_like, pl.Expr):
        return expr_like

    if isinstance(expr_like, str):
        return pl.col(expr_like)

    return pl.lit(expr_like)


@overload
def zip(*targets: pl.Series) -> pl.Series:
    ...

@overload
def zip(targets: Iterable[pl.Series], /) -> pl.Series:
    ...

def zip(*targets: pl.Series | Iterable[pl.Series]):
    if not targets or isinstance(targets[0], pl.Series):
        effective_targets: tuple[pl.Series, ...] = tuple(targets) # type: ignore
    else:
        effective_targets = tuple(targets[0])

    if len(effective_targets) == 0:
        raise ValueError("At least one series must be provided")


    return implode_like(
        struct(
            target.list.explode(
                empty_as_null=False,
                keep_nulls=False,
            ) for target in effective_targets
        ),
        layout=effective_targets[0],
    )


__all__ = [
    "assemble",
    "assert_",
    "cast_arr_to_struct",
    "get_offsets",
    "implode_like",
    "implode_with_all",
    "implode_with_lengths",
    "implode_with_offsets",
    "implode_with",
    "struct",
    "to_expr",
    "zip",
]
