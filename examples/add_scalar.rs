use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    let a = Tensor::from_slice(&[1f32, -0.5, -2.3, 4., 5., 0.5], (3, 2), &Device::Cpu)?;
    let eps: Tensor = 1f32.try_into()?;
    let b = (&a + eps.broadcast_as(a.shape()))?;

    println!("a");
    println!("{}", a);
    println!("b");
    println!("{}", b);

    // a
    // [[ 1.0000, -0.5000],
    //  [-2.3000,  4.0000],
    //  [ 5.0000,  0.5000]]
    // Tensor[[3, 2], f32]
    // b
    // [[ 2.0000,  0.5000],
    //  [-1.3000,  5.0000],
    //  [ 6.0000,  1.5000]]
    // Tensor[[3, 2], f32]

    Ok(())
}
