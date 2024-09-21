use anyhow::Result;
use candle_core::{Device, Tensor, Var};

fn main() -> Result<()> {
    // let w = Tensor::from_slice(&[1f32, -0.5, -2.3, 4., 5., 0.5], (2, 3), &Device::Cpu)?;
    let w = Var::from_slice(&[1f32, -0.5, -2.3, 4., 5., 0.5], (2, 3), &Device::Cpu)?;
    let x = Tensor::from_slice(&[3f32, 4.0, 5.0], (3, 1), &Device::Cpu)?;
    let y = w.matmul(&x)?.powf(2.0)?.sum_all()?;

    let mut grads = y.backward()?;
    let g_w = grads.get(&w).unwrap();
    println!("{}", g_w);
    println!("{:?}", grads);

    let g_w = g_w.clamp(-1.0, 1.0)?;
    println!("{}", g_w);
    let _ = grads.insert(&w, g_w);
    println!("{:?}", grads);

    Ok(())
}
