use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    let t = Tensor::from_slice(&[1f32, -0.5, -2.3, 4., 5., 0.5], (3, 2), &Device::Cpu)?;
    println!("{}", t);
    let c = t.clamp(-1.0, 1.0).unwrap();
    println!("{}", t); // not changed the original tensor
    println!("{}", c);

    Ok(())
}
