use candle_core::{Device, Tensor};

fn main() -> anyhow::Result<()> {
    let x = Tensor::from_slice(
        &[0f32, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        (4, 3),
        &Device::Cpu,
    )?;
    println!("The original tensor:");
    println!("{}", x);

    let y = Tensor::from_slice(&[1f32, 2., 3., 4., 5., 6.], (2, 3), &Device::Cpu)?;
    println!("The tensor to be assigned:");
    println!("{}", y);

    x.slice_set(&y, 0, 1)?;
    println!("The tensor after assignment:");
    println!("{}", x);

    Ok(())
}
