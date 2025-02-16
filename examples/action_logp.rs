use anyhow::Result;
use candle_core::{Device, Tensor, D};
use std::f32::consts::PI;

fn normal_logp(x: &Tensor) -> Result<Tensor> {
    let logp = ((-0.5 * (2.0 * PI).ln() as f64) - (0.5 * x.powf(2.0)?)?)?;
    Ok(logp)
}

fn atanh(x: &Tensor) -> Result<Tensor> {
    let x = x.clamp(-0.999999, 0.999999)?;
    Ok((0.5 * (((1. + &x)? / (1. - &x)?)?).log()?)?)
}

pub fn log_jacobian_tanh(a: &Tensor) -> Result<Tensor> {
    let a = a.clamp(-0.999999, 0.999999)?;
    Ok((-1f64 * (1f64 - a.powf(2.0)?)?.log()?)?.sum(D::Minus1)?)
}

fn main() -> Result<()> {
    let epsilon = 1e-6;
    let mean = Tensor::from_slice(&[0f32, 0., 0., 0., 0., 0.], (3, 2), &Device::Cpu)?;
    let std = Tensor::from_slice(&[1f32, 1., 1., 1., 1., 1.], (3, 2), &Device::Cpu)?;

    // Sample action with tanh squashing
    let z = Tensor::randn(0f32, 1f32, mean.dims(), &Device::Cpu)?;
    let a = (&z * &std + &mean)?.tanh()?;

    // Log probability of the action
    let log_p = (normal_logp(&z)? - 1f64 * ((1f64 - a.powf(2.0)?)? + epsilon)?.log()?)?
        .sum(D::Minus1)?
        .squeeze(D::Minus1)?;
    println!("{:?}", log_p.to_vec1::<f32>());

    // Log probability with inverse transformation
    let log_p = {
        let x = atanh(&a)?;
        let z = ((&x - &mean)? / &std)?;
        (normal_logp(&z)? - 1f64 * ((1f64 - a.powf(2.0)?)? + epsilon)?.log()?)?
            .sum(D::Minus1)?
            .squeeze(D::Minus1)?
    };
    println!("{:?}", log_p.to_vec1::<f32>());

    // Log probability with inverse transformation
    let log_p = {
        let x = atanh(&a)?;
        let z = ((&x - &mean)? / &std)?;
        (normal_logp(&z)?.sum(D::Minus1)? + log_jacobian_tanh(&a)?)?
            .squeeze(D::Minus1)?
    };
    println!("{:?}", log_p.to_vec1::<f32>());

    Ok(())
}
