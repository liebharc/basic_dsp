use numbers::*;
use multicore_support::*;
use simd_extensions::*;
use super::super::{DspVec, MetaData, ToSliceMut, NumberSpace, Domain};
/// Trigonometry methods.
pub trait TrigOps {
    /// Calculates the sine of each element in radians.
    ///
    /// # Example
    ///
    /// ```
    /// use std::f32;
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(f32::consts::PI/2.0, -f32::consts::PI/2.0).to_real_time_vec();
    /// vector.sin();
    /// assert_eq!([1.0, -1.0], vector[..]);
    /// ```
    fn sin(&mut self);

    /// Calculates the cosine of each element in radians.
    ///
    /// # Example
    ///
    /// ```
    /// use std::f32;
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(2.0 * f32::consts::PI, f32::consts::PI).to_real_time_vec();
    /// vector.cos();
    /// assert_eq!([1.0, -1.0], vector[..]);
    /// ```
    fn cos(&mut self);

    /// Calculates the tangent of each element in radians.
    fn tan(&mut self);

    /// Calculates the principal value of the inverse sine of each element in radians.
    fn asin(&mut self);

    /// Calculates the principal value of the inverse cosine of each element in radians.
    fn acos(&mut self);

    /// Calculates the principal value of the inverse tangent of each element in radians.
    fn atan(&mut self);

    /// Calculates the hyperbolic sine each element in radians.
    fn sinh(&mut self);

    /// Calculates the hyperbolic cosine each element in radians.
    fn cosh(&mut self);

    /// Calculates the hyperbolic tangent each element in radians.
    fn tanh(&mut self);

    /// Calculates the principal value of the inverse hyperbolic sine of each element in radians.
    fn asinh(&mut self);

    /// Calculates the principal value of the inverse hyperbolic cosine of each element in radians.
    fn acosh(&mut self);

    /// Calculates the principal value of the inverse hyperbolic
    /// tangent of each element in radians.
    fn atanh(&mut self);
}

/// Roots, powers, exponentials and logarithms.
pub trait PowerOps<T>
    where T: RealNumber
{
    /// Gets the square root of all vector elements.
    ///
    /// The sqrt of a negative number gives NaN and not a complex vector.
    /// # Example
    ///
    /// ```
    /// # use std::f64;
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(1.0, 4.0, 9.0, 16.0, 25.0).to_real_time_vec();
    /// vector.sqrt();
    /// assert_eq!([1.0, 2.0, 3.0, 4.0, 5.0], vector[..]);
    /// let mut vector = vec!(-1.0).to_real_time_vec();
    /// vector.sqrt();
    /// assert!(f64::is_nan(vector[0]));
    /// ```
    fn sqrt(&mut self);

    /// Squares all vector elements.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(1.0, 2.0, 3.0, 4.0, 5.0).to_real_time_vec();
    /// vector.square();
    /// assert_eq!([1.0, 4.0, 9.0, 16.0, 25.0], vector[..]);
    /// ```
    fn square(&mut self);

    /// Calculates the n-th root of every vector element.
    ///
    /// If the result would be a complex number then the vector will contain a NaN instead.
    /// So the vector
    /// will never convert itself to a complex vector during this operation.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(1.0, 8.0, 27.0).to_real_time_vec();
    /// vector.root(3.0);
    /// assert_eq!([1.0, 2.0, 3.0], vector[..]);
    /// ```
    fn root(&mut self, degree: T);

    /// Raises every vector element to a floating point power.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(1.0, 2.0, 3.0).to_real_time_vec();
    /// vector.powf(3.0);
    /// assert_eq!([1.0, 8.0, 27.0], vector[..]);
    /// ```
    fn powf(&mut self, exponent: T);

    /// Computes the principal value of natural logarithm of every element in the vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::f64;
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(2.718281828459045, 7.389056, 20.085537).to_real_time_vec();
    /// vector.ln();
    /// let actual = &vector[0..];
    /// let expected = &[1.0, 2.0, 3.0];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!(f64::abs(actual[i] - expected[i]) < 1e-4);
    /// }
    /// ```
    fn ln(&mut self);

    /// Calculates the natural exponential for every vector element.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::f64;
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(1.0, 2.0, 3.0).to_real_time_vec();
    /// vector.exp();
    /// let actual = &vector[0..];
    /// let expected = &[2.718281828459045, 7.389056, 20.085537];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!(f64::abs(actual[i] - expected[i]) < 1e-4);
    /// }
    /// ```
    fn exp(&mut self);

    /// Calculates the logarithm to the given base for every vector element.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::f64;
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(10.0, 100.0, 1000.0).to_real_time_vec();
    /// vector.log(10.0);
    /// let actual = &vector[0..];
    /// let expected = &[1.0, 2.0, 3.0];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!(f64::abs(actual[i] - expected[i]) < 1e-4);
    /// }
    /// ```
    fn log(&mut self, base: T);

    /// Calculates the exponential to the given base for every vector element.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(1.0, 2.0, 3.0).to_real_time_vec();
    /// vector.expf(10.0);
    /// assert_eq!([10.0, 100.0, 1000.0], vector[..]);
    /// ```
    fn expf(&mut self, base: T);
}

const TRIG_COMPLEXITY: Complexity = Complexity::Medium;
const DEF_POW_COMPLEXITY: Complexity = Complexity::Large;

impl<S, T, N, D> TrigOps for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber,
          N: NumberSpace,
          D: Domain
{
    fn sin(&mut self) {
        if self.is_complex() {
            self.pure_complex_operation(|v, _| v.sin(), (), TRIG_COMPLEXITY);
        } else {
            self.pure_real_operation(|v, _| v.sin(), (), TRIG_COMPLEXITY);
        }
    }

    fn cos(&mut self) {
        if self.is_complex() {
            self.pure_complex_operation(|v, _| v.cos(), (), TRIG_COMPLEXITY);
        } else {
            self.pure_real_operation(|v, _| v.cos(), (), TRIG_COMPLEXITY);
        }
    }

    fn tan(&mut self) {
        if self.is_complex() {
            self.pure_complex_operation(|v, _| v.tan(), (), TRIG_COMPLEXITY);
        } else {
            self.pure_real_operation(|v, _| v.tan(), (), TRIG_COMPLEXITY);
        }
    }

    fn asin(&mut self) {
        if self.is_complex() {
            self.pure_complex_operation(|v, _| v.asin(), (), TRIG_COMPLEXITY);
        } else {
            self.pure_real_operation(|v, _| v.asin(), (), TRIG_COMPLEXITY);
        }
    }

    fn acos(&mut self) {
        if self.is_complex() {
            self.pure_complex_operation(|v, _| v.acos(), (), TRIG_COMPLEXITY);
        } else {
            self.pure_real_operation(|v, _| v.acos(), (), TRIG_COMPLEXITY);
        }
    }

    fn atan(&mut self) {
        if self.is_complex() {
            self.pure_complex_operation(|v, _| v.atan(), (), TRIG_COMPLEXITY);
        } else {
            self.pure_real_operation(|v, _| v.atan(), (), TRIG_COMPLEXITY);
        }
    }

    fn sinh(&mut self) {
        if self.is_complex() {
            self.pure_complex_operation(|v, _| v.sinh(), (), TRIG_COMPLEXITY);
        } else {
            self.pure_real_operation(|v, _| v.sinh(), (), TRIG_COMPLEXITY);
        }
    }

    fn cosh(&mut self) {
        if self.is_complex() {
            self.pure_complex_operation(|v, _| v.cosh(), (), TRIG_COMPLEXITY);
        } else {
            self.pure_real_operation(|v, _| v.cosh(), (), TRIG_COMPLEXITY);
        }
    }

    fn tanh(&mut self) {
        if self.is_complex() {
            self.pure_complex_operation(|v, _| v.tanh(), (), TRIG_COMPLEXITY);
        } else {
            self.pure_real_operation(|v, _| v.tanh(), (), TRIG_COMPLEXITY);
        }
    }

    fn asinh(&mut self) {
        if self.is_complex() {
            self.pure_complex_operation(|v, _| v.asinh(), (), TRIG_COMPLEXITY);
        } else {
            self.pure_real_operation(|v, _| v.asinh(), (), TRIG_COMPLEXITY);
        }
    }

    fn acosh(&mut self) {
        if self.is_complex() {
            self.pure_complex_operation(|v, _| v.acosh(), (), TRIG_COMPLEXITY);
        } else {
            self.pure_real_operation(|v, _| v.acosh(), (), TRIG_COMPLEXITY);
        }
    }

    fn atanh(&mut self) {
        if self.is_complex() {
            self.pure_complex_operation(|v, _| v.atanh(), (), TRIG_COMPLEXITY);
        } else {
            self.pure_real_operation(|v, _| v.atanh(), (), TRIG_COMPLEXITY);
        }
    }
}

impl<S, T, N, D> PowerOps<T> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber,
          N: NumberSpace,
          D: Domain
{
    fn sqrt(&mut self) {
        if self.is_complex() {
            self.pure_complex_operation(|x, _arg| x.sqrt(), (), Complexity::Small);
        } else {
            self.simd_real_operation(|x, _arg| x.sqrt(),
                                     |x, _arg| x.sqrt(),
                                     (),
                                     Complexity::Small);
        }
    }

    fn square(&mut self) {
        if self.is_complex() {
            self.pure_complex_operation(|x, _arg| x * x, (), Complexity::Small);
        } else {
            self.simd_real_operation(|x, _arg| x * x, |x, _arg| x * x, (), Complexity::Small);
        }
    }

    fn root(&mut self, degree: T) {
        self.powf(T::one() / degree);
    }

    fn powf(&mut self, exponent: T) {
        if self.is_complex() {
            self.pure_complex_operation(|x, y| x.powf(y), exponent, DEF_POW_COMPLEXITY);
        } else {
            self.pure_real_operation(|x, y| x.powf(y), exponent, DEF_POW_COMPLEXITY);
        }
    }

    fn ln(&mut self) {
        if self.is_complex() {
            self.pure_complex_operation(|x, _arg| x.ln(), (), DEF_POW_COMPLEXITY);
        } else {
            self.pure_real_operation(|x, _arg| x.ln(), (), DEF_POW_COMPLEXITY);
        }
    }

    fn exp(&mut self) {
        if self.is_complex() {
            self.pure_complex_operation(|x, _arg| x.exp(), (), DEF_POW_COMPLEXITY);
        } else {
            self.pure_real_operation(|x, _arg| x.exp(), (), DEF_POW_COMPLEXITY);
        }
    }

    fn log(&mut self, base: T) {
        if self.is_complex() {
            self.pure_complex_operation(|x, y| x.log(y), base, DEF_POW_COMPLEXITY);
        } else {
            self.pure_real_operation(|x, y| x.log(y), base, DEF_POW_COMPLEXITY);
        }
    }

    fn expf(&mut self, base: T) {
        if self.is_complex() {
            self.pure_complex_operation(|x, y| x.expf(y), base, DEF_POW_COMPLEXITY);
        } else {
            self.pure_real_operation(|x, y| y.powf(x), base, DEF_POW_COMPLEXITY);
        }
    }
}
