use polars::prelude::*;
use polars_compute::gather::sublist::fixed_size_list::sub_fixed_size_list_get_literal;
use pyo3_polars::export::polars_arrow::array::{ListArray, StructArray};
use pyo3_polars::export::polars_arrow::buffer::Buffer;
use pyo3_polars::export::polars_arrow::offset::OffsetsBuffer;
use pyo3_polars::export::polars_core::utils::{align_chunks_binary_ca_series, Container};

pub fn cast_arr_to_struct(target_series: &Series, struct_like_series: &Series) -> PolarsResult<Series> {
    let arr = target_series.array()?;
    let struct_like = struct_like_series.struct_()?;

    if arr.width() != struct_like.struct_fields().len() {
        return Err(PolarsError::ShapeMismatch(
            format!(
                "Cannot cast array of width {} to struct of width {}",
                arr.width(),
                struct_like.struct_fields().len(),
            ).into(),
        ));
    }

    let new_chunks_iter = arr
        .downcast_iter()
        .map(|chunk| {
            StructArray::new(
                struct_like
                    .dtype()
                    .to_arrow(CompatLevel::newest()),
                arr.len(),
                (0..arr.width())
                    .map(|index| sub_fixed_size_list_get_literal(chunk, index as i64, false).unwrap())
                    .collect::<Vec<_>>(),
                chunk
                    .validity()
                    .cloned(),
            )
        });

    let new_ca = StructChunked::from_chunk_iter(
        target_series.name().clone(),
        new_chunks_iter,
    );

    Ok(new_ca.into_series())
}

pub fn get_offsets(series: &Series) -> PolarsResult<Series> {
    let list = series.list()?;
    let name = series.name().clone();

    if series.n_chunks() == 1 {
        let first_chunk = list.downcast_iter().next().unwrap();
        let offsets = first_chunk.offsets();

        if offsets[0] == 0 {
            return Ok(Series::new(name, offsets.as_slice()));
        }
    }

    let mut offsets = Series::from_vec(name, vec![0]);

    offsets.append(&cum_sum(
        &Series::new(PlSmallStr::EMPTY, list.lst_lengths()),
        false,
    )?)?;

    Ok(offsets)
}

pub fn implode_like(target_series: &Series, layout_series: &Series) -> PolarsResult<Series> {
    let layout_ca = layout_series.list()?;

    let (layout_aligned_ca, target_aligned_series) =
        align_chunks_binary_ca_series(layout_ca, target_series);

    if layout_ca.inner_length() != target_series.len() {
        return Err(PolarsError::ShapeMismatch(
            format!(
                "Target series length ({}) does not match layout inner length ({})",
                target_series.len(),
                layout_ca.inner_length(),
            )
            .into(),
        ));
    }

    let new_chunks_iter = target_aligned_series
        .chunks()
        .iter()
        .zip(layout_aligned_ca.downcast_iter())
        .map(|(target_chunk, layout_chunk)| {
            let offsets_source = layout_chunk.offsets();

            let offsets = if offsets_source[0] != 0 {
                unsafe {
                    OffsetsBuffer::new_unchecked(Buffer::from_iter(
                        offsets_source
                            .iter()
                            .map(|offset| offset - offsets_source[0]),
                    ))
                }
            } else {
                offsets_source.clone()
            };

            ListArray::new(
                DataType::List(Box::new(target_series.dtype().clone()))
                    .to_arrow(CompatLevel::newest()),
                offsets,
                target_chunk.clone(),
                layout_chunk.validity().cloned(),
            )
        });

    let new_ca = ListChunked::from_chunk_iter(target_series.name().clone(), new_chunks_iter);

    Ok(new_ca.into_series())
}

pub fn implode_with_lengths(
    target_series: &Series,
    lengths_series: &Series,
) -> PolarsResult<Series> {
    if lengths_series.has_nulls() {
        return Err(PolarsError::ComputeError(
            "Lengths series must not contain null values".into(),
        ));
    }

    let mut offsets = Series::from_vec(PlSmallStr::EMPTY, vec![0i64]);

    offsets.append(&cum_sum(&lengths_series, false)?)?;

    implode_with_offsets(target_series, &offsets)
}

// See also: trim_lists_to_normalized_offsets()
// https://github.com/pola-rs/polars/blob/778dbb645ccbff8b1e5999a279037571a03c718b/crates/polars-compute/src/trim_lists_to_normalized_offsets.rs#L9

pub fn implode_with_offsets(
    target_series: &Series,
    offsets_series: &Series,
) -> PolarsResult<Series> {
    let cast_offsets_series = offsets_series.cast(&DataType::Int64)?;
    let offsets_ca = cast_offsets_series.i64().unwrap();

    if offsets_ca.first().unwrap() != 0 {
        return Err(PolarsError::ShapeMismatch(
            "Offsets must start at zero".into(),
        ));
    }

    let (offsets_aligned_ca, target_aligned_series) =
        align_chunks_binary_ca_series(offsets_ca, target_series);

    let new_chunks_iter = target_aligned_series
        .chunks()
        .iter()
        .zip(offsets_aligned_ca.downcast_iter())
        .map(|(target_chunk, offsets_chunk)| {
            ListArray::new(
                DataType::List(Box::new(target_series.dtype().clone()))
                    .to_arrow(CompatLevel::newest()),
                unsafe { OffsetsBuffer::new_unchecked(offsets_chunk.values().clone()) },
                target_chunk.clone(),
                offsets_chunk.validity().cloned(),
            )
        });

    let new_ca = ListChunked::from_chunk_iter(target_series.name().clone(), new_chunks_iter);

    Ok(new_ca.into_series())
}
