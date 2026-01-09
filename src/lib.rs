use polars::prelude::*;
use pyo3::types::{PyModule, PyModuleMethods as _};
use pyo3::{Bound, PyResult, Python, pyfunction, pymodule};
use pyo3_polars::PySeries;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::export::polars_arrow::buffer::Buffer;
use pyo3_polars::export::polars_arrow::offset::OffsetsBuffer;
use pyo3_polars::export::polars_core::utils::{Container, align_chunks_binary_ca_series};
use pyo3_polars::export::polars_arrow::array::ListArray;
use pyo3_polars::derive::polars_expr;


fn get_offsets(series: &Series) -> PolarsResult<Series> {
    let list = series.list()?;
    let name = series.name().clone();

    if series.n_chunks() == 1 {
        let ca = list;
        let first_chunk = ca.downcast_iter().next().unwrap();
        let offsets = first_chunk.offsets();

        if offsets[0] == 0 {
            return Ok(
                Series::new(
                    name,
                    offsets.as_slice()
                )
            );
        }
    }

    let mut offsets = Series::from_vec(name, vec![0]);

    offsets.append(
        &cum_sum(
            &Series::new(
                PlSmallStr::EMPTY,
                list.lst_lengths(),
            ),
            false,
        )?
    )?;

    Ok(offsets)
}

#[pyfunction]
fn get_offsets_py(series: PySeries) -> PyResult<PySeries> {
    Ok(
        PySeries(
            get_offsets(&series.into()).map_err(PyPolarsErr::from)?
        )
    )
}

#[pymodule]
fn expression_lib(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(get_offsets_py, m)?)?;
    Ok(())

}


fn implode_like(
    target_series: &Series,
    layout_series: &Series,
) -> PolarsResult<Series> {
    let layout_ca = layout_series.list()?;

    let (layout_aligned_ca, target_aligned_series) = align_chunks_binary_ca_series(
        layout_ca,
        target_series,
    );

    if layout_ca.inner_length() != target_series.len() {
        return Err(
            PolarsError::ShapeMismatch(
                format!(
                    "Target series length ({}) does not match layout inner length ({})",
                    target_series.len(),
                    layout_ca.inner_length(),
                ).into()
            )
        );
    }

    let new_chunks_iter = target_aligned_series
        .chunks()
        .iter()
        .zip(
            layout_aligned_ca.downcast_iter()
        )
        .map(|(target_chunk, layout_chunk)| {
            let offsets_source = layout_chunk.offsets();

            let offsets = if offsets_source[0] != 0 {
                unsafe {
                    OffsetsBuffer::new_unchecked(
                        Buffer::from_iter(
                            offsets_source.iter().map(|offset| offset - offsets_source[0])
                        )
                    )
                }
            } else {
                offsets_source.clone()
            };

            ListArray::new(
                DataType::List(
                    Box::new(
                        target_series.dtype().clone()
                    )
                ).to_arrow(CompatLevel::newest()),
                offsets,
                target_chunk.clone(),
                layout_chunk.validity().cloned(),
            )
        });

    let new_ca = ListChunked::from_chunk_iter(
        target_series.name().clone(),
        new_chunks_iter,
    );

    Ok(new_ca.into_series())
}


fn implode_like_output(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(input_fields[0].clone())
}

#[polars_expr(output_type_func=implode_like_output)]
fn implode_like_expr(inputs: &[Series]) -> PolarsResult<Series>{
    implode_like(&inputs[0], &inputs[1])
}
