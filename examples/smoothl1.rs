use anyhow::Result;
use candle_core::{Device, Tensor, DType::F32};

fn smooth_l1_loss(x: &Tensor, y: &Tensor) -> Result<Tensor, candle_core::Error> {
    let d = (x - y)?.abs()?;
    let m1 = d.lt(1.0)?.to_dtype(F32)?;
    let m2 = Tensor::try_from(1f32)?.broadcast_sub(&m1)?;
    ((0.5 * m1)? * d.powf(2.0))? + m2 * (d - 0.5)
}

fn main() -> Result<()> {
    let x = Tensor::from_slice(&[1f32, 1.5, 2., 3., 4., 5.], (2, 3), &Device::Cpu)?;
    let y = (0.5 * Tensor::ones_like(&x)?)?;
    let z = smooth_l1_loss(&x, &y)?;
    println!("{}", x);
    println!("{}", (x - y)?);
    println!("{}", z);

    Ok(())
}
