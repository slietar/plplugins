mod functions;

use polars::prelude::*;
use pyo3::types::{PyModule, PyModuleMethods as _};
use pyo3::{pyfunction, pymodule, Bound, PyResult, Python};
use pyo3_polars::derive::polars_expr;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::PySeries;

#[pyfunction]
fn get_offsets(series: PySeries) -> PyResult<PySeries> {
    Ok(PySeries(
        functions::get_offsets(&series.into()).map_err(PyPolarsErr::from)?,
    ))
}

#[pyfunction]
fn implode_like(target_series: PySeries, layout_series: PySeries) -> PyResult<PySeries> {
    Ok(PySeries(
        functions::implode_like(&target_series.into(), &layout_series.into())
            .map_err(PyPolarsErr::from)?,
    ))
}

#[pyfunction]
fn implode_with_lengths(target_series: PySeries, lengths_series: PySeries) -> PyResult<PySeries> {
    Ok(PySeries(
        functions::implode_with_lengths(&target_series.into(), &lengths_series.into())
            .map_err(PyPolarsErr::from)?,
    ))
}

#[pyfunction]
fn implode_with_offsets(target_series: PySeries, offsets_series: PySeries) -> PyResult<PySeries> {
    Ok(PySeries(
        functions::implode_with_offsets(&target_series.into(), &offsets_series.into())
            .map_err(PyPolarsErr::from)?,
    ))
}

#[pymodule]
fn plutils(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(get_offsets, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(implode_like, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(implode_with_lengths, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(implode_with_offsets, m)?)?;
    Ok(())
}

fn implode_output(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(input_fields[0].clone())
}

#[polars_expr(output_type=Int64)]
fn get_offsets_expr(inputs: &[Series]) -> PolarsResult<Series> {
    functions::get_offsets(&inputs[0])
}

#[polars_expr(output_type_func=implode_output)]
fn implode_like_expr(inputs: &[Series]) -> PolarsResult<Series> {
    functions::implode_like(&inputs[0], &inputs[1])
}

#[polars_expr(output_type_func=implode_output)]
fn implode_with_lengths_expr(inputs: &[Series]) -> PolarsResult<Series> {
    functions::implode_with_lengths(&inputs[0], &inputs[1])
}

#[polars_expr(output_type_func=implode_output)]
fn implode_with_offsets_expr(inputs: &[Series]) -> PolarsResult<Series> {
    functions::implode_with_offsets(&inputs[0], &inputs[1])
}
