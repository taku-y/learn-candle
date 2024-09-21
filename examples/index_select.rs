use candle_core::{Device, Tensor};

fn main() -> anyhow::Result<()> {
    let x = Tensor::from_slice(
        &[0f32, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        (4, 3),
        &Device::Cpu,
    )?;
    println!("The original tensor:");
    println!("{}", x);

    let y = vec![1, 3];
    let y = Tensor::from_vec(
        y.iter().map(|&i| i as u32).collect::<Vec<u32>>(),
        y.len(),
        &Device::Cpu,
    )?;
    println!("Indexing the tensor:");
    println!("{}", y);

    let z = x.index_select(&y, 0)?;
    println!("The tensor of selected indices:");
    println!("{}", z);

    Ok(())
}
