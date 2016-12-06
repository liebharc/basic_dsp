//! This module allows to combine certain operations into one operation. Since one
//! many machines the speed of many DSP operations is limited by the memory bus speed
//! this approach may result in better register and cache usage and thus decrease
//! the pressure on the memory bus. As with all performance hints remember
//! rule number 1: Benchmark your code. This is especially true at this very early
//! state of the library.
//!
//! With this approach we change how we operate on vectors. If you perform
//! `M` operations on a vector with the length `N` you iterate wit hall other methods like this:
//!
//! ```no_run
//! // pseudocode:
//! // for m in M:
//! //  for n in N:
//! //    execute m on n
//! ```
//!
//! with this method the pattern is changed slightly:
//!
//! ```no_run
//! // pseudocode:
//! // for n in N:
//! //  for m in M:
//! //    execute m on n
//! ```
//!
//! Both variants have the same complexity however the second one is beneficial since we
//! have increased locality this way. This should help us by making better use of registers and
//! CPU caches.
//!
//! Only operations can be combined where the result of every element in the vector
//! is independent from any other element in the vector.
//!
//! # Examples
//!
//! ```
//! use std::f32::consts::PI;
//! use basic_dsp_vector::*;
//! use basic_dsp_vector::combined_ops::*;
//! # fn close(left: &[f32], right: &[f32]) {
//! #   assert_eq!(left.len(), right.len());
//! #   for i in 0..left.len() {
//! #       assert!((left[i] - right[i]) < 1e-2);
//! #   }
//! # }
//! let a = vec!(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8).to_real_time_vec();
//! let b = vec!(0.0; 8).to_real_time_vec();
//! let ops = multi_ops2(a, b);
//! let ops = ops.add_ops(|mut a, mut b| {
//!     a.scale(2.0 * PI);
//!     b.clone_from(&a);
//!     a.sin();
//!     b.cos();
//!     a.mul(&b);
//!     a.abs();
//!     a.log(10.0);
//!     a.scale(10.0);
//!     (a, b)
//! });
//! let mut buffer = SingleBuffer::new();
//! let (a, b) = ops.get(&mut buffer).expect("Ignoring error handling in examples");
//! close(&[0.80902, 0.30902, -0.30902, -0.80902, -1.00000, -0.80902, -0.30902, 0.30902],
//! 		  &b[..]);
//! close(&[-3.2282, -5.3181, -5.3181, -3.2282, -159.1199, -3.2282, -5.3181, -5.3181],
//! 	      &a[..]);
//! ```
//!

mod operations_enum;
pub use self::operations_enum::*;
mod prepared_ops;
pub use self::prepared_ops::*;
mod multi_ops;
pub use self::multi_ops::*;
mod identifier_ops;
pub use self::identifier_ops::*;

use RealNumber;
use simd_extensions::*;
use multicore_support::*;
use std::ops::Range;
use super::{round_len, ToSliceMut, DspVec, ErrorReason, MetaData, TransRes, Vector,
            RealToComplexTransformsOpsBuffered, ComplexToRealTransformsOps, Buffer, Owner,
            RealOrComplexData, TimeOrFrequencyData, Domain, NumberSpace, DataDomain};
use std::sync::{Arc, Mutex};
use std::fmt;
use num::complex::Complex;
use std::ops::{Add, Sub, Mul, Div};

fn require_complex(is_complex: bool) -> Result<bool, ErrorReason> {
    if is_complex {
        Ok(is_complex)
    } else {
        Err(ErrorReason::InputMustBeComplex)
    }
}

fn complex_to_real(is_complex: bool) -> Result<bool, ErrorReason> {
    if is_complex {
        Ok(false)
    } else {
        Err(ErrorReason::InputMustBeComplex)
    }
}

fn real_to_complex(is_complex: bool) -> Result<bool, ErrorReason> {
    if is_complex {
        Err(ErrorReason::InputMustBeReal)
    } else {
        Ok(true)
    }
}

fn require_real(is_complex: bool) -> Result<bool, ErrorReason> {
    if is_complex {
        Err(ErrorReason::InputMustBeReal)
    } else {
        Ok(is_complex)
    }
}

fn evaluate_number_space_transition<T>(is_complex: bool,
                                           operation: &Operation<T>)
                                           -> Result<bool, ErrorReason>
    where T: RealNumber
{
    match *operation {
        // Real Ops
        Operation::AddReal(_, _) |
        Operation::MultiplyReal(_, _) |
        Operation::Abs(_) |
        Operation::MapReal(_, _)
            => require_real(is_complex),
        Operation::ToComplex(_) => real_to_complex(is_complex),
        // Complex Ops
        Operation::AddComplex(_, _) |
        Operation::MultiplyComplex(_, _) |
        Operation::ComplexConj(_) |
        Operation::MultiplyComplexExponential(_, _, _) |
        Operation::MapComplex(_, _)
            => require_complex(is_complex),
        Operation::Magnitude(_) |
        Operation::MagnitudeSquared(_) |
        Operation::ToReal(_) |
        Operation::ToImag(_) | 
        Operation::Phase(_)
            => complex_to_real(is_complex),
        // General Ops
        Operation::AddPoints(_) |
        Operation::SubPoints(_) |
        Operation::MulPoints(_) |
        Operation::DivPoints(_) |
        Operation::AddVector(_, _) |
        Operation::SubVector(_, _) |
        Operation::MulVector(_, _) |
        Operation::DivVector(_, _) |
        Operation::Sqrt(_) |
        Operation::Square(_) |
        Operation::Root(_, _) |
        Operation::Powf(_, _) |
        Operation::Ln(_) |
        Operation::Exp(_) |
        Operation::Log(_, _) |
        Operation::Expf(_, _) |
        Operation::Sin(_) |
        Operation::Cos(_) |
        Operation::Tan(_) |
        Operation::ASin(_) |
        Operation::ACos(_) |
        Operation::ATan(_) |
        Operation::Sinh(_) |
        Operation::Cosh(_) |
        Operation::Tanh(_) |
        Operation::ASinh(_) |
        Operation::ACosh(_) |
        Operation::ATanh(_) |
        Operation::CloneFrom(_, _)
            => Ok(is_complex)
    }
}

fn get_argument<T>(operation: &Operation<T>) -> usize
    where T: RealNumber
{
    match *operation {
        // Real Ops
        Operation::AddReal(arg, _) |
        Operation::MultiplyReal(arg, _) |
        Operation::Abs(arg) |
        Operation::ToComplex(arg) |
        Operation::MapReal(arg, _) |
        // Complex Ops
        Operation::AddComplex(arg, _) |
        Operation::MultiplyComplex(arg, _) |
        Operation::Magnitude(arg) |
        Operation::MagnitudeSquared(arg) |
        Operation::ComplexConj(arg) |
        Operation::ToReal(arg) |
        Operation::ToImag(arg) |
        Operation::Phase(arg) |
        Operation::MultiplyComplexExponential(arg, _, _) |
        Operation::MapComplex(arg, _) |
        // General Ops
        Operation::AddPoints(arg) |
        Operation::SubPoints(arg) |
        Operation::MulPoints(arg) |
        Operation::DivPoints(arg) |
        Operation::AddVector(arg, _) |
        Operation::SubVector(arg, _) |
        Operation::MulVector(arg, _) |
        Operation::DivVector(arg, _) |
        Operation::Sqrt(arg) |
        Operation::Square(arg) |
        Operation::Root(arg, _) |
        Operation::Powf(arg, _) |
        Operation::Ln(arg) |
        Operation::Exp(arg) |
        Operation::Log(arg, _) |
        Operation::Expf(arg, _) |
        Operation::Sin(arg) |
        Operation::Cos(arg) |
        Operation::Tan(arg) |
        Operation::ASin(arg) |
        Operation::ACos(arg) |
        Operation::ATan(arg) |
        Operation::Sinh(arg) |
        Operation::Cosh(arg) |
        Operation::Tanh(arg) |
        Operation::ASinh(arg) |
        Operation::ACosh(arg) |
        Operation::ATanh(arg) |
        Operation::CloneFrom(arg, _)
            => arg
    }
}

trait PerformOperationSimd<T>
    where T: RealNumber,
          Self: Sized
{
    #[inline]
    fn perform_real_operation(vectors: &mut [Self],
                              operation: &Operation<T>,
                              index: usize,
                              points: usize);
    #[inline]
    fn perform_complex_operation(vectors: &mut [Self],
                                 operation: &Operation<T>,
                                 index: usize,
                                 points: usize);
}

impl<T> PerformOperationSimd<T> for T::Reg
    where T: RealNumber
{
    #[inline]
    fn perform_complex_operation(vectors: &mut [Self],
                                 operation: &Operation<T>,
                                 index: usize,
                                 points: usize) {
        match *operation {
            // Real Ops
            Operation::AddReal(idx, value) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = v.add_real(value);
            }
            Operation::MultiplyReal(idx, value) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = v.scale_real(value);
            }
            Operation::Abs(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_vector(|x: T| x.abs());
            }
            Operation::ToComplex(_) => {
                // Number space conversions should already been done
                // before the operations are executed, so there is nothing
                // to do anymore
            }
            Operation::MapReal(_, _) => {
                panic!("real operation on complex vector indicates a bug");
            }
            // Complex Ops
            Operation::AddComplex(idx, value) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = v.add_complex(value);
            }
            Operation::MultiplyComplex(idx, value) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = v.scale_complex(value);
            }
            Operation::Magnitude(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = v.complex_abs2();
            }
            Operation::MagnitudeSquared(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = v.complex_abs_squared2();
            }
            Operation::ComplexConj(idx) => {
                let arg = T::Reg::from_complex(Complex::<T>::new(T::one(), -T::one()));
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = v.mul(arg);
            }
            Operation::ToReal(idx) => {
                // We don't have to reorganize the data and just need to zero the imag part
                let arg = T::Reg::from_complex(Complex::<T>::new(T::one(), T::zero()));
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = v.mul(arg);
            }
            Operation::ToImag(idx) => {
                let arg = T::Reg::from_complex(Complex::<T>::new(T::zero(), -T::one()));
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                let swapped = v.mul_complex(arg);
                let arg = T::Reg::from_complex(Complex::<T>::new(T::one(), T::zero()));
                *v = swapped.mul(arg);
            }
            Operation::Phase(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_complex_vector(|x| Complex::new(x.arg(), T::zero()));
            }
            Operation::MultiplyComplexExponential(idx, a, b) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                let mut exponential = Complex::<T>::from_polar(&T::one(), &b) *
                                      Complex::<T>::from_polar(&T::one(),
                                                               &(a * T::from(index).unwrap()));
                let increment = Complex::<T>::from_polar(&T::one(), &a);
                *v = v.iter_over_complex_vector(|x| {
                    let res = x * exponential;
                    exponential = exponential * increment;
                    res
                });
            }
            Operation::MapComplex(idx, ref op) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                let mut i = index;
                *v = v.iter_over_complex_vector(|x| {
                    let res = op(x, i);
                    i += 1;
                    res
                });
            }
            // General Ops
            Operation::AddPoints(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = v.add_complex(Complex::<T>::new(T::from(points).unwrap(), T::zero()));
            }
            Operation::SubPoints(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = v.add_complex(Complex::<T>::new(-(T::from(points).unwrap()), T::zero()));
            }
            Operation::MulPoints(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = v.scale_complex(Complex::<T>::new(T::from(points).unwrap(), T::zero()));
            }
            Operation::DivPoints(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = v.scale_complex(Complex::<T>::new(T::one() / (T::from(points).unwrap()),
                                                       T::zero()));
            }
            Operation::AddVector(idx1, idx2) => {
                let v2 = unsafe { *vectors.get_unchecked(idx2) };
                let v1 = unsafe { vectors.get_unchecked_mut(idx1) };
                *v1 = v1.add(v2);
            }
            Operation::SubVector(idx1, idx2) => {
                let v2 = unsafe { *vectors.get_unchecked(idx2) };
                let v1 = unsafe { vectors.get_unchecked_mut(idx1) };
                *v1 = v1.sub(v2);
            }
            Operation::MulVector(idx1, idx2) => {
                let v2 = unsafe { *vectors.get_unchecked(idx2) };
                let v1 = unsafe { vectors.get_unchecked_mut(idx1) };
                *v1 = v1.mul_complex(v2);
            }
            Operation::DivVector(idx1, idx2) => {
                let v2 = unsafe { *vectors.get_unchecked(idx2) };
                let v1 = unsafe { vectors.get_unchecked_mut(idx1) };
                *v1 = v1.div_complex(v2);
            }
            Operation::Sqrt(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_complex_vector(|x: Complex<T>| x.sqrt());
            }
            Operation::Square(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = v.mul_complex(*v);
            }
            Operation::Root(idx, value) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_complex_vector(|x| x.powf(T::one() / value));
            }
            Operation::Powf(idx, value) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_complex_vector(|x| x.powf(value));
            }
            Operation::Ln(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_complex_vector(|x: Complex<T>| x.ln());
            }
            Operation::Exp(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_complex_vector(|x: Complex<T>| x.exp());
            }
            Operation::Log(idx, value) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_complex_vector(|x: Complex<T>| x.log(value));
            }
            Operation::Expf(idx, value) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_complex_vector(|x: Complex<T>| x.expf(value));
            }
            Operation::Sin(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_complex_vector(|x: Complex<T>| x.sin());
            }
            Operation::Cos(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_complex_vector(|x: Complex<T>| x.cos());
            }
            Operation::Tan(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_complex_vector(|x: Complex<T>| x.tan());
            }
            Operation::ASin(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_complex_vector(|x: Complex<T>| x.asin());
            }
            Operation::ACos(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_complex_vector(|x: Complex<T>| x.acos());
            }
            Operation::ATan(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_complex_vector(|x: Complex<T>| x.atan());
            }
            Operation::Sinh(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_complex_vector(|x: Complex<T>| x.sinh());
            }
            Operation::Cosh(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_complex_vector(|x: Complex<T>| x.cosh());
            }
            Operation::Tanh(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_complex_vector(|x: Complex<T>| x.tanh());
            }
            Operation::ASinh(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_complex_vector(|x: Complex<T>| x.asinh());
            }
            Operation::ACosh(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_complex_vector(|x: Complex<T>| x.acosh());
            }
            Operation::ATanh(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_complex_vector(|x: Complex<T>| x.atanh());
            }
            Operation::CloneFrom(idx1, idx2) => {
                let v2 = unsafe { *vectors.get_unchecked(idx2) };
                let v1 = unsafe { vectors.get_unchecked_mut(idx1) };
                *v1 = v2;
            }
        }
    }

    #[inline]
    fn perform_real_operation(vectors: &mut [Self],
                              operation: &Operation<T>,
                              index: usize,
                              points: usize) {
        match *operation {
            // Real Ops
            Operation::AddReal(idx, value) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = v.add_real(value);
            }
            Operation::MultiplyReal(idx, value) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = v.scale_real(value);
            }
            Operation::Abs(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_vector(|x: T| x.abs());
            }
            Operation::ToComplex(_) => {
                panic!("number space conversions should have already been completed");
            }
            Operation::MapReal(idx, ref op) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                let mut i = index;
                *v = v.iter_over_vector(|x| {
                    let res = op(x, i);
                    i += 1;
                    res
                });
            }
            // Complex Ops
            Operation::AddComplex(_, _) => {
                panic!("complex operation on real vector indicates a bug");
            }
            Operation::MultiplyComplex(_, _) => {
                panic!("complex operation on real vector indicates a bug");
            }
            Operation::Magnitude(_) => {
                panic!("complex operation on real vector indicates a bug");
            }
            Operation::MagnitudeSquared(_) => {
                panic!("complex operation on real vector indicates a bug");
            }
            Operation::ComplexConj(_) => {
                panic!("complex operation on real vector indicates a bug");
            }
            Operation::ToReal(_) => {
                panic!("complex operation on real vector indicates a bug");
            }
            Operation::ToImag(_) => {
                panic!("complex operation on real vector indicates a bug");
            }
            Operation::Phase(_) => {
                panic!("complex operation on real vector indicates a bug");
            }
            Operation::MultiplyComplexExponential(_, _, _) => {
                panic!("complex operation on real vector indicates a bug");
            }
            Operation::MapComplex(_, _) => {
                panic!("complex operation on real vector indicates a bug");
            }
            // General Ops
            Operation::AddPoints(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = v.add_real(T::from(points).unwrap());
            }
            Operation::SubPoints(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = v.add_real(-(T::from(points).unwrap()));
            }
            Operation::MulPoints(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = v.scale_real(T::from(points).unwrap());
            }
            Operation::DivPoints(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = v.scale_real(T::one() / T::from(points).unwrap());
            }
            Operation::AddVector(idx1, idx2) => {
                let v2 = unsafe { *vectors.get_unchecked(idx2) };
                let v1 = unsafe { vectors.get_unchecked_mut(idx1) };
                *v1 = v1.add(v2);
            }
            Operation::SubVector(idx1, idx2) => {
                let v2 = unsafe { *vectors.get_unchecked(idx2) };
                let v1 = unsafe { vectors.get_unchecked_mut(idx1) };
                *v1 = v1.sub(v2);
            }
            Operation::MulVector(idx1, idx2) => {
                let v2 = unsafe { *vectors.get_unchecked(idx2) };
                let v1 = unsafe { vectors.get_unchecked_mut(idx1) };
                *v1 = v1.mul(v2);
            }
            Operation::DivVector(idx1, idx2) => {
                let v2 = unsafe { *vectors.get_unchecked(idx2) };
                let v1 = unsafe { vectors.get_unchecked_mut(idx1) };
                *v1 = v1.div(v2);
            }
            Operation::Sqrt(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = v.sqrt();
            }
            Operation::Square(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = v.mul(*v);
            }
            Operation::Root(idx, value) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_vector(|x: T| x.powf(T::one() / value));
            }
            Operation::Powf(idx, value) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_vector(|x: T| x.powf(value));
            }
            Operation::Ln(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_vector(|x: T| x.ln());
            }
            Operation::Exp(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_vector(|x: T| x.exp());
            }
            Operation::Log(idx, value) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_vector(|x: T| x.log(value));
            }
            Operation::Expf(idx, value) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_vector(|x: T| value.powf(x));
            }
            Operation::Sin(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_vector(|x: T| x.sin());
            }
            Operation::Cos(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_vector(|x: T| x.cos());
            }
            Operation::Tan(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_vector(|x: T| x.tan());
            }
            Operation::ASin(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_vector(|x: T| x.asin());
            }
            Operation::ACos(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_vector(|x: T| x.acos());
            }
            Operation::ATan(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_vector(|x: T| x.atan());
            }
            Operation::Sinh(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_vector(|x: T| x.sinh());
            }
            Operation::Cosh(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_vector(|x: T| x.cosh());
            }
            Operation::Tanh(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_vector(|x: T| x.tanh());
            }
            Operation::ASinh(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_vector(|x: T| x.asinh());
            }
            Operation::ACosh(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_vector(|x: T| x.acosh());
            }
            Operation::ATanh(idx) => {
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = (*v).iter_over_vector(|x: T| x.atanh());
            }
            Operation::CloneFrom(idx1, idx2) => {
                let v2 = unsafe { *vectors.get_unchecked(idx2) };
                let v1 = unsafe { vectors.get_unchecked_mut(idx1) };
                *v1 = v2;
            }
        }
    }
}

/// An identifier is just a placeholder for a data type
/// used to ensure already at compile time that operations are valid.
pub struct Identifier<T, N, D>
    where T: RealNumber,
          D: Domain,
          N: NumberSpace
{
    arg: usize,
    ops: Vec<(u64, Operation<T>)>,
    counter: Arc<Mutex<u64>>,
    domain: D,
    number_space: N,
}

impl<T, N, D> fmt::Debug for Identifier<T, N, D>
    where T: RealNumber,
          D: Domain,
          N: NumberSpace
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f,
               "Identifier {{ domain: {:?}, number_space: {:?} }}",
               self.domain,
               self.number_space)
    }
}

impl<T, N, D> Identifier<T, N, D>
    where T: RealNumber,
          D: Domain,
          N: NumberSpace
{
    fn add_op(&mut self, op: Operation<T>) {
        let seq = {
            let mut value = self.counter.lock().unwrap();
            *value += 1;
            *value
        };
        self.ops.push((seq, op));
    }
}

/// An operation on one data vector which has been prepared in
/// advance.
///
/// The type arguments can be read like a function definition: TI1 -> TO1
#[derive(Clone)]
pub struct PreparedOperation1<T, NI, DI, NO, DO>
    where T: RealNumber,
          NI: NumberSpace,
          DI: Domain,
          NO: NumberSpace,
          DO: Domain
{
    number_space_in: NI,
    domain_in: DI,
    number_space_out: NO,
    domain_out: DO,
    ops: Vec<Operation<T>>,
}

/// An operation on two data vectors which has been prepared in
/// advance.
///
/// The type arguments can be read like a function definition: (TI1, TI2) -> (TO1, TO2)
#[derive(Clone)]
pub struct PreparedOperation2<T, NI1, DI1, NI2, DI2, NO1, DO1, NO2, DO2>
    where T: RealNumber,
          NI1: NumberSpace,
          DI1: Domain,
          NI2: NumberSpace,
          DI2: Domain,
          NO1: NumberSpace,
          DO1: Domain,
          NO2: NumberSpace,
          DO2: Domain
{
    number_space_in1: NI1,
    domain_in1: DI1,
    number_space_in2: NI2,
    domain_in2: DI2,
    number_space_out1: NO1,
    domain_out1: DO1,
    number_space_out2: NO2,
    domain_out2: DO2,
    ops: Vec<Operation<T>>,
    swap: bool,
}

fn generic_vector_from_any_vector<S, T, N, D>
    (vec: DspVec<S, T, N, D>)
     -> (N, D, DspVec<S, T, RealOrComplexData, TimeOrFrequencyData>)
    where T: RealNumber + 'static,
          S: ToSliceMut<T> + Owner,
          N: NumberSpace,
          D: Domain
{
    let domain = vec.domain.clone();
    let number_space = vec.number_space.clone();
    let gen = DspVec {
        delta: vec.delta,
        domain: TimeOrFrequencyData { domain_current: vec.domain() },
        number_space: RealOrComplexData { is_complex_current: vec.is_complex() },
        valid_len: vec.valid_len,
        multicore_settings: vec.multicore_settings,
        data: vec.data,
    };
    (number_space, domain, gen)
}

fn generic_vector_back_to_vector<S, T, N, D>(number_space: N,
                                             domain: D,
                                             vec: DspVec<S,
                                                         T,
                                                         RealOrComplexData,
                                                         TimeOrFrequencyData>)
                                             -> DspVec<S, T, N, D>
    where T: RealNumber + 'static,
          S: ToSliceMut<T> + Owner,
          N: NumberSpace,
          D: Domain
{
    let is_complex = vec.is_complex();
    let result_domain = vec.domain();
    let mut vec = DspVec {
        data: vec.data,
        delta: vec.delta,
        domain: domain,
        number_space: number_space,
        valid_len: vec.valid_len,
        multicore_settings: vec.multicore_settings,
    };

    if is_complex {
        vec.number_space.to_complex();
    } else {
        vec.number_space.to_real();
    }

    if result_domain == DataDomain::Time {
        vec.domain.to_time();
    } else {
        vec.domain.to_freq();
    }

    vec
}

fn perform_complex_operations_par<T>(array: &mut Vec<&mut [T]>,
                                     range: Range<usize>,
                                     arguments: (&[Operation<T>], usize))
    where T: RealNumber + 'static
{
    let (operations, points) = arguments;
    let mut vectors = Vec::with_capacity(array.len());
    for _ in 0..array.len() {
        vectors.push(T::Reg::splat(T::zero()));
    }

    let reg_len = T::Reg::len() / 2;
    let mut index = range.start / 2;
    let mut i = 0;
    while i < array[0].len() {
        for j in 0..array.len() {
            unsafe {
                let elem = vectors.get_unchecked_mut(j);
                *elem = T::Reg::load_unchecked(array.get_unchecked(j), i)
            }
        }

        for operation in operations {
            PerformOperationSimd::<T>::perform_complex_operation(&mut vectors,
                                                                 operation,
                                                                 index,
                                                                 points);
        }

        for j in 0..array.len() {
            unsafe {
                vectors.get_unchecked(j).store_unchecked(array.get_unchecked_mut(j), i);
            }
        }

        index += reg_len;
        i += T::Reg::len();
    }
}

fn perform_real_operations_par<T>(array: &mut Vec<&mut [T]>,
                                  range: Range<usize>,
                                  arguments: (&[Operation<T>], usize))
    where T: RealNumber + 'static
{
    let (operations, points) = arguments;
    let mut vectors = Vec::with_capacity(array.len());
    for _ in 0..array.len() {
        vectors.push(T::Reg::splat(T::zero()));
    }

    let reg_len = T::Reg::len();
    let mut index = range.start;
    let mut i = 0;
    while i < array[0].len() {
        for j in 0..array.len() {
            unsafe {
                let elem = vectors.get_unchecked_mut(j);
                *elem = T::Reg::load_unchecked(array.get_unchecked(j), i)
            }
        }

        for operation in operations {
            PerformOperationSimd::<T>::perform_real_operation(&mut vectors,
                                                              operation,
                                                              index,
                                                              points);
        }

        for j in 0..array.len() {
            unsafe {
                vectors.get_unchecked(j).store(array.get_unchecked_mut(j), i);
            }
        }

        index += reg_len;
        i += T::Reg::len();
    }
}

impl<S, T> DspVec<S, T, RealOrComplexData, TimeOrFrequencyData>
    where S: ToSliceMut<T> + Owner,
          T: RealNumber + 'static
{
    fn perform_operations<B>(buffer: &mut B,
                             mut vectors: Vec<Self>,
                             operations: &[Operation<T>])
                             -> TransRes<Vec<Self>>
        where B: Buffer<S, T>
    {
        if vectors.is_empty() {
            return Err((ErrorReason::InvalidNumberOfArgumentsForCombinedOp, vectors));
        }

        let verifcation = Self::verify_ops(&vectors, operations);
        if verifcation.is_err() {
            return Err((verifcation.unwrap_err(), vectors));
        }

        let (any_complex_ops, final_number_space) = verifcation.unwrap();

        if operations.len() == 0 {
            return Ok(vectors);
        }

        // All vectors are required to have the same length
        let first_vec_len = vectors[0].points();
        let mut has_errors = false;
        for v in &vectors {
            if v.points() != first_vec_len {
                has_errors = true;
            }
        }

        if has_errors {
            return Err((ErrorReason::InvalidNumberOfArgumentsForCombinedOp, vectors));
        }

        // If the vectors needs to be complex at some point during the
        // calculation the transform them to complex right away.
        // This might be unnecessary in some cases, but covering
        // complex/real mixed cases will cause more implementation effort
        // later in the process so we better keep it simple for now.
        if any_complex_ops {
            let mut complex_vectors = Vec::with_capacity(vectors.len());
            for _ in 0..vectors.len() {
                let vector = vectors.pop().unwrap();
                let as_complex = if vector.is_complex() {
                    vector
                } else {
                    vector.to_complex_b(buffer)
                };
                complex_vectors.push(as_complex);
            }

            complex_vectors.reverse();
            vectors = complex_vectors;
        }

        let (vectorization_length, multicore_settings, scalar_length) = {
            let first = &vectors[0];
            let data_length = first.len();
            let alloc_len = first.alloc_len();
            let rounded_len = round_len(data_length);
            if rounded_len <= alloc_len {
                (rounded_len, first.multicore_settings, 0)
            } else {
                let scalar_length = data_length % T::Reg::len();
                (data_length - scalar_length, first.multicore_settings, scalar_length)
            }
        };

        if vectorization_length > 0 {
            let complexity = if operations.len() > 5 {
                Complexity::Large
            } else {
                Complexity::Medium
            };
            {
                let mut array: Vec<&mut [T]> = vectors.iter_mut()
                    .map(|v| {
                        let len = v.len();
                        let data = v.data.to_slice_mut();
                        &mut data[0..len]
                    })
                    .collect();
                let range = Range {
                    start: 0,
                    end: vectorization_length,
                };
                if any_complex_ops {
                    Chunk::execute_partial_multidim(complexity,
                                                    &multicore_settings,
                                                    &mut array,
                                                    range,
                                                    T::Reg::len(),
                                                    (operations, first_vec_len),
                                                    perform_complex_operations_par);
                } else {
                    Chunk::execute_partial_multidim(complexity,
                                                    &multicore_settings,
                                                    &mut array,
                                                    range,
                                                    T::Reg::len(),
                                                    (operations, first_vec_len),
                                                    perform_real_operations_par);
                }
            }
        }

        if scalar_length > 0 {
            let mut last_elems = Vec::with_capacity(vectors.len());
            for v in &vectors {
                let reg = T::Reg::splat(T::zero());
                let mut i = 0;
                let reg = reg.iter_over_vector(|_| {
                    let res = if i < scalar_length {
                        v[vectorization_length + i]
                    } else {
                        T::zero()
                    };

                    i += 1;

                    res
                });

                last_elems.push(reg);
            }

            if any_complex_ops {
                for operation in operations {
                    PerformOperationSimd::<T>::perform_complex_operation(&mut last_elems,
                                                                         operation,
                                                                         (vectorization_length /
                                                                          T::Reg::len() *
                                                                          2),
                                                                         first_vec_len);
                }
            } else {
                for operation in operations {
                    PerformOperationSimd::<T>::perform_real_operation(&mut last_elems,
                                                                      operation,
                                                                      (vectorization_length /
                                                                       T::Reg::len()),
                                                                      first_vec_len);
                }
            }

            let mut j = 0;
            for reg in last_elems {
                for i in 0..scalar_length {
                    vectors[j][vectorization_length + i] = reg.extract(i as u32);
                }
                j += 1;
            }
        }

        // In case we converted vectors at the beginning to complex
        // we might now need to convert them back to real.
        if any_complex_ops {
            let mut correct_domain = Vec::with_capacity(vectors.len());
            vectors.reverse();
            for num_space in &final_number_space {
                let vector = vectors.pop().unwrap();
                let right_domain = if *num_space {
                    vector
                } else {
                    vector.to_real()
                };
                correct_domain.push(right_domain);
            }

            vectors = correct_domain;
        }

        Ok(vectors)
    }

    fn verify_ops(vectors: &[Self],
                  operations: &[Operation<T>])
                  -> Result<(bool, Vec<bool>), ErrorReason> {
        let mut complex: Vec<bool> = vectors.iter().map(|v| v.is_complex()).collect();
        let max_arg_num = vectors.len();
        let mut complex_at_any_moment = complex.clone();
        for op in operations {
            let index = get_argument(op);

            if index >= max_arg_num {
                return Err(ErrorReason::InvalidNumberOfArgumentsForCombinedOp);
            }

            let eval = evaluate_number_space_transition(complex[index], op);
            let complex_after_op = match eval {
                Err(reason) => return Err(reason),
                Ok(new_complex) => new_complex,
            };

            complex[index] = complex_after_op;
            if complex_after_op {
                complex_at_any_moment[index] = complex_after_op;
            }
        }

        Ok((complex_at_any_moment.iter().any(|c| *c), complex))
    }
}

#[cfg(test)]
mod tests {
    use num::complex::Complex32;
    use super::super::*;
    use super::*;

    #[test]
    fn prepared_ops1_construction() {
        let ops = prepare32_1(RealData, TimeData);
        let ops = ops.add_ops(|mut x| {
            x.offset(5.0);
            x
        });

        let vec = vec!(0.0; 5).to_real_time_vec();
        let mut buffer = SingleBuffer::new();
        let vec = ops.exec(&mut buffer, vec).unwrap();
        assert_eq!(&vec[..], [5.0, 5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn multi_ops_construction() {
        // This test case tests mainly the syntax and less the
        // runtime results.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].to_real_time_vec();
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].to_real_time_vec();
        let mut buffer = SingleBuffer::new();

        let ops = multi_ops2(a, b);
        let ops = ops.add_ops(|a, b| (a, b));
        let (a, _) = ops.get(&mut buffer).unwrap();

        assert_eq!(&a[..], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn prepared_ops_construction() {
        // This test case tests mainly the syntax and less the
        // runtime results.
        let ops = prepare32_2(RealData, TimeData, RealData, TimeData);
        let ops = ops.add_ops(|a, b| (a, b));
        let mut buffer = SingleBuffer::new();

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].to_real_time_vec();
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].to_real_time_vec();
        let (a, _) = ops.exec(&mut buffer, a, b).unwrap();

        assert_eq!(&a[..], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn swapping() {
        let ops = prepare32_2(ComplexData, TimeData, RealData, TimeData);
        let ops = ops.add_ops(|a, b| (a, b));

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].to_complex_time_vec();
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].to_real_time_vec();
        let mut buffer = SingleBuffer::new();
        assert_eq!(a.is_complex(), true);
        let (a, b) = ops.exec(&mut buffer, a, b).unwrap();
        assert_eq!(a.is_complex(), true);
        assert_eq!(b.is_complex(), false);
        assert_eq!(b.len(), 8);

        let ops = ops.add_ops(|a, b| (b, a));
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].to_complex_time_vec();
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].to_real_time_vec();
        assert_eq!(a.is_complex(), true);
        let (b, a) = ops.exec(&mut buffer, a, b).unwrap();
        assert_eq!(a.is_complex(), true);
        assert_eq!(b.is_complex(), false);
        assert_eq!(b.len(), 8);
    }

    #[test]
    fn swap_twice() {
        let ops = prepare32_2(ComplexData, TimeData, RealData, TimeData);
        let ops = ops.add_ops(|a, b| (b, a));
        let ops = ops.add_ops(|a, b| (b, a));

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].to_complex_time_vec();
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].to_real_time_vec();
        let mut buffer = SingleBuffer::new();
        assert_eq!(a.is_complex(), true);
        let (a, b) = ops.exec(&mut buffer, a, b).unwrap();
        assert_eq!(a.is_complex(), true);
        assert_eq!(b.is_complex(), false);
        assert_eq!(b.len(), 8);
    }

    #[test]
    fn complex_operation_on_real_vector() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .to_gen_dsp_vec(false, DataDomain::Time);
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .to_gen_dsp_vec(false, DataDomain::Time);
        let mut buffer = SingleBuffer::new();

        let ops = multi_ops2(a, b);
        let ops = ops.add_ops(|a, b| (b.magnitude(), a));
        let res = ops.get(&mut buffer);
        assert!(res.is_err());
    }

    #[test]
    fn simple_operation() {
        let ops = prepare32_2(ComplexData, TimeData, RealData, TimeData);
        let ops = ops.add_ops(|a, mut b| {
            b.abs();
            (b, a)
        });

        let a = vec![1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, 1.0, -2.0, 3.0, 4.0, 5.0, -6.0,
                     -7.0, 8.0]
            .to_complex_time_vec();
        let b = vec![-11.0, 12.0, -13.0, 14.0, -15.0, 16.0, -17.0, 18.0].to_real_time_vec();
        let mut buffer = SingleBuffer::new();

        assert_eq!(a.is_complex(), true);
        let (b, a) = ops.exec(&mut buffer, a, b).unwrap();
        assert_eq!(a.is_complex(), true);
        assert_eq!(b.is_complex(), false);
        let expected = [1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, 1.0, -2.0, 3.0, 4.0, 5.0, -6.0,
                        -7.0, 8.0];
        assert_eq!(&b[..], &expected);
        let expected = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0];
        assert_eq!(&a[..], &expected);
    }

    /// Same as `simple_operation` but the arguments are passed in reversed order.
    #[test]
    fn simple_operation2() {
        let ops = prepare32_2(RealData, TimeData, ComplexData, TimeData);
        let ops = ops.add_ops(|mut a, b| {
            a.abs();
            (b, a)
        });

        let a = vec![1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, 1.0, -2.0, 3.0, 4.0, 5.0, -6.0,
                     -7.0, 8.0]
            .to_complex_time_vec();
        let b = vec![-11.0, 12.0, -13.0, 14.0, -15.0, 16.0, -17.0, 18.0].to_real_time_vec();
        let mut buffer = SingleBuffer::new();

        assert_eq!(a.is_complex(), true);
        let (a, b) = ops.exec(&mut buffer, b, a).unwrap();
        assert_eq!(a.is_complex(), true);
        assert_eq!(b.is_complex(), false);
        let expected = [1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, 1.0, -2.0, 3.0, 4.0, 5.0, -6.0,
                        -7.0, 8.0];
        assert_eq!(&b[..], &expected);
        let expected = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0];
        assert_eq!(&a[..], &expected);
    }

    #[test]
    fn map_inplace_real_test() {
        let a = vec![1.0, 2.0, 3.0, 4.0].to_real_time_vec();
        let mut buffer = SingleBuffer::new();
        let ops = multi_ops1(a);
        let ops = ops.add_ops(|mut a| {
            a.map_inplace(|v, i| v * i as f32);
            a
        });
        let a = ops.get(&mut buffer).unwrap();
        let expected = [0.0, 2.0, 6.0, 12.0];
        assert_eq!(&a[..], &expected);
    }

    #[test]
    fn map_inplace_complex_test() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].to_complex_time_vec();
        let mut buffer = SingleBuffer::new();
        let ops = multi_ops1(a);
        let ops = ops.add_ops(|mut a| {
            a.map_inplace(|v, i| v * Complex32::new(i as f32, 0.0));
            a
        });
        let a = ops.get(&mut buffer).unwrap();
        let expected = [0.0, 0.0, 3.0, 4.0, 10.0, 12.0];
        assert_eq!(&a[..], &expected);
    }
}
