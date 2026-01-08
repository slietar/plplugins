use polars::prelude::*;
use pyo3_polars::export::polars_core::utils::align_chunks_binary;
use pyo3_polars::export::polars_arrow::array::ListArray;
use pyo3_polars::derive::polars_expr;


fn implode_like(
    target_series: &Series,
    layout_series: &Series,
) -> PolarsResult<Series> {
    let target_ca = target_series.unpack::<Float32Type>()?;
    let layout_ca = layout_series.list()?;

    let (target_aligned_ca, layout_aligned_ca) = align_chunks_binary(
        target_ca,
        layout_ca,
    );

    let new_chunks_iter = target_aligned_ca
        .chunks()
        .iter()
        .zip(
            layout_aligned_ca.downcast_iter()
        )
        .map(|(target_chunk, layout_chunk)| {
            ListArray::new(
                DataType::List(
                    Box::new(
                        target_ca.dtype().clone()
                    )
                ).to_arrow(CompatLevel::newest()),
                layout_chunk.offsets().clone(),
                target_chunk.clone(),
                None,
            )
        });

    let new_ca = ListChunked::from_chunk_iter(
        target_series.name().clone(),
        new_chunks_iter,
    );

    Ok(new_ca.into_series())
}



#[polars_expr(output_type=Float32)]
fn implode_like_expr(inputs: &[Series]) -> PolarsResult<Series>{
    implode_like(&inputs[0], &inputs[1])
}
