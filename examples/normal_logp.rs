use anyhow::Result;
use candle_core::{Device, Tensor};

fn normal_logp(x: &Tensor) -> Result<()> {
    let tmp: Tensor = ((-0.5 * (2.0 * std::f32::consts::PI).ln() as f64) - (0.5 * x.powf(2.0)?)?)?;
    println!("{:?}", tmp.to_vec2::<f32>());
    let tmp = tmp.sum(vec![1])?;
    println!("{:?}", tmp.to_vec1::<f32>());

    Ok(())
}

fn main() -> Result<()> {
    let t = Tensor::from_slice(&[1f32, 2., 3., 4., 5., 6.], (3, 2), &Device::Cpu)?;
    println!("{:?}", t);
    normal_logp(&t)?;

    Ok(())
}
