extern crate basic_dsp_vector;

use basic_dsp_vector::*;

mod mat_impl;
pub use self::mat_impl::*;
mod to_from_mat_conversions;
pub use self::to_from_mat_conversions::*;
mod forwards;
pub use self::forwards::*;

pub struct MatrixMxN<V, T>
    where T: RealNumber,
          V: Vector<T> {
  rows: Vec<V>,
  number_type: std::marker::PhantomData<T>
}

pub struct Matrix2xN<V, T>
    where T: RealNumber,
          V: Vector<T> {
  rows: [V; 2],
  number_type: std::marker::PhantomData<T>
}

pub struct Matrix3xN<V, T>
    where T: RealNumber,
          V: Vector<T> {
  rows: [V; 3],
  number_type: std::marker::PhantomData<T>
}

pub struct Matrix4xN<V, T>
    where T: RealNumber,
          V: Vector<T> {
  rows: [V; 4],
  number_type: std::marker::PhantomData<T>
}

pub type Matrix32xN = MatrixMxN<GenDspVec32, f32>;
pub type Matrix64xN = MatrixMxN<GenDspVec64, f32>;
pub type RealTimeMatrix32xN = MatrixMxN<RealTimeVec32, f32>;
pub type RealTimeMatrix64xN = MatrixMxN<RealTimeVec64, f64>;
pub type ComplexTimeMatrix32xN = MatrixMxN<ComplexTimeVec32, f32>;
pub type ComplexTimeMatrix64xN = MatrixMxN<ComplexTimeVec64, f64>;
pub type RealFreqMatrix32xN = MatrixMxN<RealFreqVec32, f32>;
pub type RealFreqMatrix64xN = MatrixMxN<RealFreqVec64, f64>;
pub type ComplexFreqMatrix32xN = MatrixMxN<ComplexFreqVec32, f32>;
pub type ComplexFreqMatrix64xN = MatrixMxN<ComplexFreqVec64, f64>;

pub type Matrix32x2 = Matrix2xN<GenDspVec32, f32>;
pub type Matrix64x2 = Matrix2xN<GenDspVec64, f32>;
pub type RealTimeMatrix32x2 = Matrix2xN<RealTimeVec32, f32>;
pub type RealTimeMatrix64x2 = Matrix2xN<RealTimeVec64, f64>;
pub type ComplexTimeMatrix32x2 = Matrix2xN<ComplexTimeVec32, f32>;
pub type ComplexTimeMatrix64x2 = Matrix2xN<ComplexTimeVec64, f64>;
pub type RealFreqMatrix32x2 = Matrix2xN<RealFreqVec32, f32>;
pub type RealFreqMatrix64x2 = Matrix2xN<RealFreqVec64, f64>;
pub type ComplexFreqMatrix32x2 = Matrix2xN<ComplexFreqVec32, f32>;
pub type ComplexFreqMatrix64x2 = Matrix2xN<ComplexFreqVec64, f64>;

pub type Matrix32x3 = Matrix3xN<GenDspVec32, f32>;
pub type Matrix64x3 = Matrix3xN<GenDspVec64, f32>;
pub type RealTimeMatrix32x3 = Matrix3xN<RealTimeVec32, f32>;
pub type RealTimeMatrix64x3 = Matrix3xN<RealTimeVec64, f64>;
pub type ComplexTimeMatrix32x3 = Matrix3xN<ComplexTimeVec32, f32>;
pub type ComplexTimeMatrix64x3 = Matrix3xN<ComplexTimeVec64, f64>;
pub type RealFreqMatrix32x3 = Matrix3xN<RealFreqVec32, f32>;
pub type RealFreqMatrix64x3 = Matrix3xN<RealFreqVec64, f64>;
pub type ComplexFreqMatrix32x3 = Matrix3xN<ComplexFreqVec32, f32>;
pub type ComplexFreqMatrix64x3 = Matrix3xN<ComplexFreqVec64, f64>;

pub type Matrix32x4 = Matrix4xN<GenDspVec32, f32>;
pub type Matrix64x4 = Matrix4xN<GenDspVec64, f32>;
pub type RealTimeMatrix32x4 = Matrix4xN<RealTimeVec32, f32>;
pub type RealTimeMatrix64x4 = Matrix4xN<RealTimeVec64, f64>;
pub type ComplexTimeMatrix32x4 = Matrix4xN<ComplexTimeVec32, f32>;
pub type ComplexTimeMatrix64x4 = Matrix4xN<ComplexTimeVec64, f64>;
pub type RealFreqMatrix32x4 = Matrix4xN<RealFreqVec32, f32>;
pub type RealFreqMatrix64x4 = Matrix4xN<RealFreqVec64, f64>;
pub type ComplexFreqMatrix32x4 = Matrix4xN<ComplexFreqVec32, f32>;
pub type ComplexFreqMatrix64x4 = Matrix4xN<ComplexFreqVec64, f64>;
