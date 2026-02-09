use polars::prelude::*;
use polars_compute::gather::sublist::fixed_size_list::sub_fixed_size_list_get_literal;
use pyo3_polars::export::polars_arrow::array::{Array, ListArray, StructArray};
use pyo3_polars::export::polars_arrow::buffer::Buffer;
use pyo3_polars::export::polars_arrow::offset::OffsetsBuffer;
use pyo3_polars::export::polars_core::utils::{align_chunks_binary_ca_series, Container};

pub fn cast_arr_to_struct(
    target_series: &Series,
    struct_like_series: &Series,
) -> PolarsResult<Series> {
    let arr = target_series.array()?;
    let struct_like = struct_like_series.struct_()?;

    if arr.width() != struct_like.struct_fields().len() {
        return Err(PolarsError::ShapeMismatch(
            format!(
                "Cannot cast array of width {} to struct of width {}",
                arr.width(),
                struct_like.struct_fields().len(),
            )
            .into(),
        ));
    }

    let new_chunks_iter = arr.downcast_iter().map(|chunk| {
        StructArray::new(
            struct_like.dtype().to_arrow(CompatLevel::newest()),
            arr.len(),
            (0..arr.width())
                .map(|index| sub_fixed_size_list_get_literal(chunk, index as i64, false).unwrap())
                .collect::<Vec<_>>(),
            chunk.validity().cloned(),
        )
    });

    let new_ca = StructChunked::from_chunk_iter(target_series.name().clone(), new_chunks_iter);

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

    let mut offsets = Series::from_vec(name, vec![0i64]);

    offsets.append(&cum_sum(
        &Series::new(PlSmallStr::EMPTY, list.lst_lengths()).cast(&DataType::Int64)?,
        false,
    )?)?;

    Ok(offsets)
}

pub fn implode_like(target_series: &Series, layout_series: &Series) -> PolarsResult<Series> {
    let layout_ca = layout_series.list()?;

    if layout_series.has_nulls() {
        return Err(PolarsError::ComputeError(
            "Layout series must not contain null values".into(),
        ));
    }

    let (target_array, offsets) = 'a: {
        if target_series.n_chunks() == 1 && layout_ca.n_chunks() == 1 {
            let first_layout_chunk_offsets = layout_ca.downcast_iter().next().unwrap().offsets();

            if first_layout_chunk_offsets[0] == 0 {
                let first_target_chunk = target_series.chunks().first().unwrap();

                break 'a (
                    first_target_chunk.clone(),
                    first_layout_chunk_offsets.clone(),
                );
            }
        }

        let offsets_iter = std::iter::once(0)
            .chain(
                layout_ca
                    .downcast_iter()
                    .flat_map(|chunk| chunk.offsets().lengths()),
            )
            .scan(0, |acc, len| {
                *acc += len as i64;
                Some(*acc)
            });

        let offsets = unsafe { OffsetsBuffer::new_unchecked(Buffer::from_iter(offsets_iter)) };

        // Some amount of rechunking is required if a chunk boundary falls within a
        // list. For simplicity, we rechunk the series fully.
        let target_series_rechunked = target_series.rechunk();

        break 'a (
            target_series_rechunked.chunks().first().unwrap().clone(),
            offsets,
        );
    };

    let dtype = target_series.dtype().clone().implode();

    let array = ListArray::new(
        dtype.to_physical().to_arrow(CompatLevel::newest()),
        offsets,
        target_array,
        None,
    );

    return Ok(Series::from_chunk_and_dtype(
        target_series.name().clone(),
        Box::new(array),
        &dtype,
    )?);
}

pub fn implode_with_lengths(
    target_series: &Series,
    lengths_series: &Series,
) -> PolarsResult<Series> {
    let mut offsets = Series::from_vec(PlSmallStr::EMPTY, vec![0i64]);
    offsets.append(&cum_sum(&lengths_series.cast(&DataType::Int64)?, false)?)?;

    implode_with_offsets(target_series, &offsets)
}

// See also: trim_lists_to_normalized_offsets()
// https://github.com/pola-rs/polars/blob/778dbb645ccbff8b1e5999a279037571a03c718b/crates/polars-compute/src/trim_lists_to_normalized_offsets.rs#L9

pub fn implode_with_offsets(
    target_series: &Series,
    offsets_series: &Series,
) -> PolarsResult<Series> {
    // TODO: Fix
    let target_series = target_series.rechunk();
    let offsets_series = offsets_series.rechunk();

    let cast_offsets_series = offsets_series.cast(&DataType::Int64)?;
    let offsets_ca = cast_offsets_series.i64().unwrap();

    if offsets_ca.has_nulls() {
        return Err(PolarsError::ComputeError(
            "Offsets series must not contain null values".into(),
        ));
    }

    if offsets_ca.len() == 0 || offsets_ca.first().unwrap() != 0 {
        return Err(PolarsError::ShapeMismatch(
            "First offset must be zero".into(),
        ));
    }

    let last_offset = offsets_ca.last().unwrap();

    if last_offset != (target_series.len() as i64) {
        return Err(PolarsError::ShapeMismatch(
            format!(
                "Last offset ({}) must equal target series length ({})",
                last_offset,
                target_series.len(),
            )
            .into(),
        ));
    }

    let (offsets_aligned_ca, target_aligned_series) =
        align_chunks_binary_ca_series(offsets_ca, &target_series);

    let new_chunks_iter = target_aligned_series
        .chunks()
        .iter()
        .zip(offsets_aligned_ca.downcast_iter())
        .map(|(target_chunk, offsets_chunk)| {
            Box::new(ListArray::new(
                target_chunk.dtype().clone().to_large_list(true), // TODO: Nullable?
                unsafe { OffsetsBuffer::new_unchecked(offsets_chunk.values().clone()) },
                target_chunk.clone(),
                offsets_chunk.validity().cloned(),
            )) as Box<dyn Array>
        });

    // TODO: The change was perhaps not required
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(
            target_series.name().clone(),
            new_chunks_iter.collect(),
            &DataType::List(Box::new(target_series.dtype().clone())),
        )
    })
}
