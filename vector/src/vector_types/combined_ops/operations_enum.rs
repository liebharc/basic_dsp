use numbers::*;
use std::sync::Arc;

/// An alternative way to define operations on a vector.
#[derive(Clone)]
pub enum Operation<T> {
    // Real Ops
    AddReal(usize, T),
    MultiplyReal(usize, T),
    Abs(usize),
    ToComplex(usize),
    MapReal(usize, Arc<Fn(T, usize) -> T + Send + Sync + 'static>),
    // Complex Ops
    AddComplex(usize, Complex<T>),
    MultiplyComplex(usize, Complex<T>),
    Magnitude(usize),
    MagnitudeSquared(usize),
    ComplexConj(usize),
    ToReal(usize),
    ToImag(usize),
    Phase(usize),
    MultiplyComplexExponential(usize, T, T),
    MapComplex(usize, Arc<Fn(Complex<T>, usize) -> Complex<T> + Send + Sync + 'static>),
    // General Ops
    AddVector(usize, usize),
    SubVector(usize, usize),
    MulVector(usize, usize),
    DivVector(usize, usize),
    Sqrt(usize),
    Square(usize),
    Root(usize, T),
    Powf(usize, T),
    Ln(usize),
    Exp(usize),
    Log(usize, T),
    Expf(usize, T),
    Sin(usize),
    Cos(usize),
    Tan(usize),
    ASin(usize),
    ACos(usize),
    ATan(usize),
    Sinh(usize),
    Cosh(usize),
    Tanh(usize),
    ASinh(usize),
    ACosh(usize),
    ATanh(usize),
    CloneFrom(usize, usize),
    AddPoints(usize),
    SubPoints(usize),
    MulPoints(usize),
    DivPoints(usize),
}


