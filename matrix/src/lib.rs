//! In this lib a matrix is simply a collection of
//! vectors. The idea is that the matrix types can be used to reduce the size
//! of a large matrix and that the return types are basic enough
//! so that other specialized matrix libs can do the rest of the work, e.g.
//! inverting the resulting matrix.

extern crate basic_dsp_vector;
extern crate num_complex;

use basic_dsp_vector::*;
use std::mem;

mod mat_impl;
pub use self::mat_impl::*;
mod to_from_mat_conversions;
pub use self::to_from_mat_conversions::*;
mod general;
pub use self::general::*;
mod rededicate;
pub use self::rededicate::*;
mod complex;
pub use self::complex::*;
mod real;
pub use self::real::*;
mod time_freq;
pub use self::time_freq::*;

/// A matrix which can hold 1 to N vectors.
pub struct MatrixMxN<V, S, T>
    where T: RealNumber,
          S: ToSlice<T>,
          V: Vector<T>
{
    rows: Vec<V>,
    storage_type: std::marker::PhantomData<S>,
    number_type: std::marker::PhantomData<T>,
}

/// A matrix which can hold exactly 2 vectors.
pub struct Matrix2xN<V, S, T>
    where T: RealNumber,
          V: Vector<T>
{
    rows: [V; 2],
    storage_type: std::marker::PhantomData<S>,
    number_type: std::marker::PhantomData<T>,
}

/// A matrix which can hold exactly 3 vectors.
pub struct Matrix3xN<V, S, T>
    where T: RealNumber,
          V: Vector<T>
{
    rows: [V; 3],
    storage_type: std::marker::PhantomData<S>,
    number_type: std::marker::PhantomData<T>,
}

/// A matrix which can hold exactly 4 vectors.
pub struct Matrix4xN<V, S, T>
    where T: RealNumber,
          V: Vector<T>
{
    rows: [V; 4],
    storage_type: std::marker::PhantomData<S>,
    number_type: std::marker::PhantomData<T>,
}

/// A matrix which can hold 1 to N vectors of 32 bit floating point numbers in any number space or domain.
pub type Matrix32xN = MatrixMxN<GenDspVec32, Vec<f32>, f32>;
/// A matrix which can hold 1 to N vectors of 64 bit floating point numbers in any number space or domain.
pub type Matrix64xN = MatrixMxN<GenDspVec64, Vec<f64>, f64>;
/// A matrix which can hold 1 to N vectors of 32 bit floating point numbers in real number space and time domain.
pub type RealTimeMatrix32xN = MatrixMxN<RealTimeVec32, Vec<f32>, f32>;
/// A matrix which can hold 1 to N vectors of 64 bit floating point numbers in real number space and time domain.
pub type RealTimeMatrix64xN = MatrixMxN<RealTimeVec64, Vec<f64>, f64>;
/// A matrix which can hold 1 to N vectors of 32 bit floating point numbers in complex number space and time domain.
pub type ComplexTimeMatrix32xN = MatrixMxN<ComplexTimeVec32, Vec<f32>, f32>;
/// A matrix which can hold 1 to N vectors of 64 bit floating point numbers in complex number space and time domain.
pub type ComplexTimeMatrix64xN = MatrixMxN<ComplexTimeVec64, Vec<f64>, f64>;
/// A matrix which can hold 1 to N vectors of 32 bit floating point numbers in real number space and frequency domain.
pub type RealFreqMatrix32xN = MatrixMxN<RealFreqVec32, Vec<f32>, f32>;
/// A matrix which can hold 1 to N vectors of 64 bit floating point numbers in real number space and frequency domain.
pub type RealFreqMatrix64xN = MatrixMxN<RealFreqVec64, Vec<f64>, f64>;
/// A matrix which can hold 1 to N vectors of 32 bit floating point numbers in complex number space and frequency domain.
pub type ComplexFreqMatrix32xN = MatrixMxN<ComplexFreqVec32, Vec<f32>, f32>;
/// A matrix which can hold 1 to N vectors of 64 bit floating point numbers in complex number space and frequency domain.
pub type ComplexFreqMatrix64xN = MatrixMxN<ComplexFreqVec64, Vec<f64>, f64>;

/// A matrix which can hold exactly 2 vectors of 32 bit floating point numbers in any number space or domain.
pub type Matrix32x2 = Matrix2xN<GenDspVec32, Vec<f32>, f32>;
/// A matrix which can hold exactly 2 vectors of 64 bit floating point numbers in any number space or domain.
pub type Matrix64x2 = Matrix2xN<GenDspVec64, Vec<f64>, f64>;
/// A matrix which can hold exactly 2 vectors of 32 bit floating point numbers in real number space and time domain.
pub type RealTimeMatrix32x2 = Matrix2xN<RealTimeVec32, Vec<f32>, f32>;
/// A matrix which can hold exactly 2 vectors of 64 bit floating point numbers in real number space and time domain.
pub type RealTimeMatrix64x2 = Matrix2xN<RealTimeVec64, Vec<f64>, f64>;
/// A matrix which can hold exactly 2 vectors of 32 bit floating point numbers in complex number space and time domain.
pub type ComplexTimeMatrix32x2 = Matrix2xN<ComplexTimeVec32, Vec<f32>, f32>;
/// A matrix which can hold exactly 2 vectors of 64 bit floating point numbers in complex number space and time domain.
pub type ComplexTimeMatrix64x2 = Matrix2xN<ComplexTimeVec64, Vec<f64>, f64>;
/// A matrix which can hold exactly 2 vectors of 32 bit floating point numbers in real number space and frequency domain.
pub type RealFreqMatrix32x2 = Matrix2xN<RealFreqVec32, Vec<f32>, f32>;
/// A matrix which can hold exactly 2 vectors of 64 bit floating point numbers in real number space and frequency domain.
pub type RealFreqMatrix64x2 = Matrix2xN<RealFreqVec64, Vec<f64>, f64>;
/// A matrix which can hold exactly 2 vectors of 32 bit floating point numbers in complex number space and frequency domain.
pub type ComplexFreqMatrix32x2 = Matrix2xN<ComplexFreqVec32, Vec<f32>, f32>;
/// A matrix which can hold exactly 2 vectors of 64 bit floating point numbers in complex number space and frequency domain.
pub type ComplexFreqMatrix64x2 = Matrix2xN<ComplexFreqVec64, Vec<f64>, f64>;

/// A matrix which can hold exactly 3 vectors of 32 bit floating point numbers in any number space or domain.
pub type Matrix32x3 = Matrix3xN<GenDspVec32, Vec<f32>, f32>;
/// A matrix which can hold exactly 3 vectors of 64 bit floating point numbers in any number space or domain.
pub type Matrix64x3 = Matrix3xN<GenDspVec64, Vec<f64>, f64>;
/// A matrix which can hold exactly 3 vectors of 32 bit floating point numbers in real number space and time domain.
pub type RealTimeMatrix32x3 = Matrix3xN<RealTimeVec32, Vec<f32>, f32>;
/// A matrix which can hold exactly 3 vectors of 64 bit floating point numbers in real number space and time domain.
pub type RealTimeMatrix64x3 = Matrix3xN<RealTimeVec64, Vec<f64>, f64>;
/// A matrix which can hold exactly 3 vectors of 32 bit floating point numbers in complex number space and time domain.
pub type ComplexTimeMatrix32x3 = Matrix3xN<ComplexTimeVec32, Vec<f32>, f32>;
/// A matrix which can hold exactly 3 vectors of 64 bit floating point numbers in complex number space and time domain.
pub type ComplexTimeMatrix64x3 = Matrix3xN<ComplexTimeVec64, Vec<f64>, f64>;
/// A matrix which can hold exactly 3 vectors of 32 bit floating point numbers in real number space and frequency domain.
pub type RealFreqMatrix32x3 = Matrix3xN<RealFreqVec32, Vec<f32>, f32>;
/// A matrix which can hold exactly 4 vectors of 64 bit floating point numbers in real number space and frequency domain.
pub type RealFreqMatrix64x3 = Matrix3xN<RealFreqVec64, Vec<f64>, f64>;
/// A matrix which can hold exactly 4 vectors of 32 bit floating point numbers in complex number space and frequency domain.
pub type ComplexFreqMatrix32x3 = Matrix3xN<ComplexFreqVec32, Vec<f32>, f32>;
/// A matrix which can hold exactly 4 vectors of 64 bit floating point numbers in complex number space and frequency domain.
pub type ComplexFreqMatrix64x3 = Matrix3xN<ComplexFreqVec64, Vec<f64>, f64>;

/// A matrix which can hold exactly 4 vectors of 32 bit floating point numbers in any number space or domain.
pub type Matrix32x4 = Matrix4xN<GenDspVec32, Vec<f32>, f32>;
/// A matrix which can hold exactly 4 vectors of 64 bit floating point numbers in any number space or domain.
pub type Matrix64x4 = Matrix4xN<GenDspVec64, Vec<f64>, f64>;
/// A matrix which can hold exactly 4 vectors of 32 bit floating point numbers in real number space and time domain.
pub type RealTimeMatrix32x4 = Matrix4xN<RealTimeVec32, Vec<f32>, f32>;
/// A matrix which can hold exactly 4 vectors of 64 bit floating point numbers in real number space and time domain.
pub type RealTimeMatrix64x4 = Matrix4xN<RealTimeVec64, Vec<f64>, f64>;
/// A matrix which can hold exactly 4 vectors of 32 bit floating point numbers in complex number space and time domain.
pub type ComplexTimeMatrix32x4 = Matrix4xN<ComplexTimeVec32, Vec<f32>, f32>;
/// A matrix which can hold exactly 4 vectors of 64 bit floating point numbers in complex number space and time domain.
pub type ComplexTimeMatrix64x4 = Matrix4xN<ComplexTimeVec64, Vec<f64>, f64>;
/// A matrix which can hold exactly 4 vectors of 32 bit floating point numbers in real number space and frequency domain.
pub type RealFreqMatrix32x4 = Matrix4xN<RealFreqVec32, Vec<f32>, f32>;
/// A matrix which can hold exactly 4 vectors of 64 bit floating point numbers in real number space and frequency domain.
pub type RealFreqMatrix64x4 = Matrix4xN<RealFreqVec64, Vec<f64>, f64>;
/// A matrix which can hold exactly 4 vectors of 32 bit floating point numbers in complex number space and frequency domain.
pub type ComplexFreqMatrix32x4 = Matrix4xN<ComplexFreqVec32, Vec<f32>, f32>;
/// A matrix which can hold exactly 4 vectors of 64 bit floating point numbers in complex number space and frequency domain.
pub type ComplexFreqMatrix64x4 = Matrix4xN<ComplexFreqVec64, Vec<f64>, f64>;

/// Internal trait to transform a row storage type to another
trait TransformContent<S, D> {
    type Output;
    fn transform<F>(self, conversion: F) -> Self::Output where F: FnMut(S) -> D;
    fn transform_res<F>(self, conversion: F) -> TransRes<Self::Output>
        where F: FnMut(S) -> TransRes<D>;
}

trait IntoFixedLength<T, O> {
    fn into_fixed_length(self) -> O;
}

impl<T> IntoFixedLength<T, Vec<T>> for Vec<T> {
    fn into_fixed_length(self) -> Vec<T> {
        self
    }
}

macro_rules! try_conv {
    ($op: expr, $err: ident) => {
        {
            let res = $op;
            match res {
                Ok(v) => v,
                Err((r, v)) => {
                    $err = Some(r);
                    v
                }
            }
        }
    }
}

impl<S, D> TransformContent<S, D> for Vec<S> {
    type Output = Vec<D>;

    fn transform<F>(mut self, mut conversion: F) -> Self::Output
        where F: FnMut(S) -> D
    {
        let mut rows: Vec<D> = Vec::with_capacity(self.len());
        for _ in 0..self.len() {
            let v: S = self.pop().unwrap();
            rows.push(conversion(v));
        }
        rows.reverse();
        rows
    }

    fn transform_res<F>(mut self, mut conversion: F) -> TransRes<Self::Output>
        where F: FnMut(S) -> TransRes<D>
    {
        let mut rows: Vec<D> = Vec::with_capacity(self.len());
        let mut error = None;
        for _ in 0..self.len() {
            let v: S = self.pop().unwrap();
            rows.push(try_conv!(conversion(v), error));
        }
        rows.reverse();

        match error {
            None => Ok(rows),
            Some(err) => Err((err, rows)),
        }
    }
}

impl<S, D> TransformContent<S, D> for [S; 2] {
    type Output = [D; 2];

    fn transform<F>(mut self, mut conversion: F) -> Self::Output
        where F: FnMut(S) -> D
    {
        unsafe {
            let first = mem::replace(&mut self[0], mem::uninitialized());
            let second = mem::replace(&mut self[1], mem::uninitialized());
            mem::forget(self); // TODO possible memory leak
            let first = conversion(first);
            let second = conversion(second);
            [first, second]
        }
    }

    fn transform_res<F>(mut self, mut conversion: F) -> TransRes<Self::Output>
        where F: FnMut(S) -> TransRes<D>
    {
        unsafe {
            let mut error = None;
            let first = mem::replace(&mut self[0], mem::uninitialized());
            let second = mem::replace(&mut self[1], mem::uninitialized());
            mem::forget(self); // TODO possible memory leak
            let first = try_conv!(conversion(first), error);
            let second = try_conv!(conversion(second), error);

            match error {
                None => Ok([first, second]),
                Some(err) => Err((err, [first, second])),
            }
        }
    }
}

impl<S, D> TransformContent<S, D> for [S; 3] {
    type Output = [D; 3];

    fn transform<F>(mut self, mut conversion: F) -> Self::Output
        where F: FnMut(S) -> D
    {
        unsafe {
            let first = mem::replace(&mut self[0], mem::uninitialized());
            let second = mem::replace(&mut self[1], mem::uninitialized());
            let third = mem::replace(&mut self[2], mem::uninitialized());
            mem::forget(self); // TODO possible memory leak
            let first = conversion(first);
            let second = conversion(second);
            let third = conversion(third);
            [first, second, third]
        }
    }

    fn transform_res<F>(mut self, mut conversion: F) -> TransRes<Self::Output>
        where F: FnMut(S) -> TransRes<D>
    {
        unsafe {
            let mut error = None;
            let first = mem::replace(&mut self[0], mem::uninitialized());
            let second = mem::replace(&mut self[1], mem::uninitialized());
            let third = mem::replace(&mut self[2], mem::uninitialized());
            mem::forget(self); // TODO possible memory leak
            let first = try_conv!(conversion(first), error);
            let second = try_conv!(conversion(second), error);
            let third = try_conv!(conversion(third), error);

            match error {
                None => Ok([first, second, third]),
                Some(err) => Err((err, [first, second, third])),
            }
        }
    }
}

impl<S, D> TransformContent<S, D> for [S; 4] {
    type Output = [D; 4];

    fn transform<F>(mut self, mut conversion: F) -> Self::Output
        where F: FnMut(S) -> D
    {
        unsafe {
            let first = mem::replace(&mut self[0], mem::uninitialized());
            let second = mem::replace(&mut self[1], mem::uninitialized());
            let third = mem::replace(&mut self[2], mem::uninitialized());
            let fourth = mem::replace(&mut self[3], mem::uninitialized());
            mem::forget(self); // TODO possible memory leak
            let first = conversion(first);
            let second = conversion(second);
            let third = conversion(third);
            let fourth = conversion(fourth);
            [first, second, third, fourth]
        }
    }

    fn transform_res<F>(mut self, mut conversion: F) -> TransRes<Self::Output>
        where F: FnMut(S) -> TransRes<D>
    {
        unsafe {
            let mut error = None;
            let first = mem::replace(&mut self[0], mem::uninitialized());
            let second = mem::replace(&mut self[1], mem::uninitialized());
            let third = mem::replace(&mut self[2], mem::uninitialized());
            let fourth = mem::replace(&mut self[3], mem::uninitialized());
            mem::forget(self); // TODO possible memory leak
            let first = try_conv!(conversion(first), error);
            let second = try_conv!(conversion(second), error);
            let third = try_conv!(conversion(third), error);
            let fourth = try_conv!(conversion(fourth), error);

            match error {
                None => Ok([first, second, third, fourth]),
                Some(err) => Err((err, [first, second, third, fourth])),
            }
        }
    }
}

impl<T> IntoFixedLength<T, [T; 2]> for Vec<T> {
    fn into_fixed_length(mut self) -> [T; 2] {
        let second = self.pop().unwrap();
        let first = self.pop().unwrap();
        [first, second]
    }
}

impl<T> IntoFixedLength<T, [T; 3]> for Vec<T> {
    fn into_fixed_length(mut self) -> [T; 3] {
        let third = self.pop().unwrap();
        let second = self.pop().unwrap();
        let first = self.pop().unwrap();
        [first, second, third]
    }
}

impl<T> IntoFixedLength<T, [T; 4]> for Vec<T> {
    fn into_fixed_length(mut self) -> [T; 4] {
        let fourth = self.pop().unwrap();
        let third = self.pop().unwrap();
        let second = self.pop().unwrap();
        let first = self.pop().unwrap();
        [first, second, third, fourth]
    }
}
