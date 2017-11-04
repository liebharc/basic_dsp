use simd_extensions::*;
use super::{Operation, PerformOperationSimd};
use numbers::*;

impl<T, Reg> PerformOperationSimd<T> for Reg
    where T: RealNumber,
          Reg: SimdGeneric<T>
        
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
                let arg = Reg::from_complex(Complex::<T>::new(T::one(), -T::one()));
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = v.mul(arg);
            }
            Operation::ToReal(idx) => {
                // We don't have to reorganize the data and just need to zero the imag part
                let arg = Reg::from_complex(Complex::<T>::new(T::one(), T::zero()));
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                *v = v.mul(arg);
            }
            Operation::ToImag(idx) => {
                let arg = Reg::from_complex(Complex::<T>::new(T::zero(), -T::one()));
                let v = unsafe { vectors.get_unchecked_mut(idx) };
                let swapped = v.mul_complex(arg);
                let arg = Reg::from_complex(Complex::<T>::new(T::one(), T::zero()));
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
                              _index: usize,
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