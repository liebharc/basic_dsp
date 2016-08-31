use RealNumber;
use multicore_support::Complexity;
use super::{
    DspVec,
    ToSliceMut,
    VoidResult,
    NumberSpace,
    Domain};

/// Trigonometry methods.
pub trait TrigOps {
    /// Calculates the sine of each element in radians.
    ///
    /// # Example
    ///
    /// ```
    /// use std::f32;
    /// use basic_dsp_vector::{RealTimeVector32, GenericVectorOps, DataVec, RealIndex};
    /// let vector = RealTimeVector32::from_array(&[f32::consts::PI/2.0, -f32::consts::PI/2.0]);
    /// let result = vector.sin().expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, -1.0], result.real(0..));
    /// ```
    fn sin(&mut self) -> VoidResult;

    /// Calculates the cosine of each element in radians.
    ///
    /// # Example
    ///
    /// ```
    /// use std::f32;
    /// use basic_dsp_vector::{RealTimeVector32, GenericVectorOps, DataVec, RealIndex};
    /// let vector = RealTimeVector32::from_array(&[2.0 * f32::consts::PI, f32::consts::PI]);
    /// let result = vector.cos().expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, -1.0], result.real(0..));
    /// ```
    fn cos(&mut self) -> VoidResult;

    /// Calculates the tangent of each element in radians.
    fn tan(&mut self) -> VoidResult;

    /// Calculates the principal value of the inverse sine of each element in radians.
    fn asin(&mut self) -> VoidResult;

    /// Calculates the principal value of the inverse cosine of each element in radians.
    fn acos(&mut self) -> VoidResult;

    /// Calculates the principal value of the inverse tangent of each element in radians.
    fn atan(&mut self) -> VoidResult;

    /// Calculates the hyperbolic sine each element in radians.
    fn sinh(&mut self) -> VoidResult;

    /// Calculates the hyperbolic cosine each element in radians.
    fn cosh(&mut self) -> VoidResult;

    /// Calculates the hyperbolic tangent each element in radians.
    fn tanh(&mut self) -> VoidResult;

    /// Calculates the principal value of the inverse hyperbolic sine of each element in radians.
    fn asinh(&mut self) -> VoidResult;

    /// Calculates the principal value of the inverse hyperbolic cosine of each element in radians.
    fn acosh(&mut self) -> VoidResult;

    /// Calculates the principal value of the inverse hyperbolic tangent of each element in radians.
    fn atanh(&mut self) -> VoidResult;
}

impl<S, T, N, D> TrigOps for DspVec<S, T, N, D>
    where
        S: ToSliceMut<T>,
        T: RealNumber,
        N: NumberSpace,
        D: Domain {
    fn sin(&mut self) -> VoidResult {
        if self.is_complex() {
            self.pure_real_operation(|v, _| v.sin(), (), Complexity::Medium);
        } else {
            self.pure_complex_operation(|v, _| v.sin(), (), Complexity::Medium);
        }

        Ok(())
    }

    fn cos(&mut self) -> VoidResult {
        if self.is_complex() {
            self.pure_real_operation(|v, _| v.cos(), (), Complexity::Medium);
        } else {
            self.pure_complex_operation(|v, _| v.cos(), (), Complexity::Medium);
        }

        Ok(())
    }

    fn tan(&mut self) -> VoidResult {
        if self.is_complex() {
            self.pure_real_operation(|v, _| v.tan(), (), Complexity::Medium);
        } else {
            self.pure_complex_operation(|v, _| v.tan(), (), Complexity::Medium);
        }

        Ok(())
    }

    fn asin(&mut self) -> VoidResult {
        if self.is_complex() {
            self.pure_real_operation(|v, _| v.asin(), (), Complexity::Medium);
        } else {
            self.pure_complex_operation(|v, _| v.asin(), (), Complexity::Medium);
        }

        Ok(())
    }

    fn acos(&mut self) -> VoidResult {
        if self.is_complex() {
            self.pure_real_operation(|v, _| v.acos(), (), Complexity::Medium);
        } else {
            self.pure_complex_operation(|v, _| v.acos(), (), Complexity::Medium);
        }

        Ok(())
    }

    fn atan(&mut self) -> VoidResult {
        if self.is_complex() {
            self.pure_real_operation(|v, _| v.atan(), (), Complexity::Medium);
        } else {
            self.pure_complex_operation(|v, _| v.atan(), (), Complexity::Medium);
        }

        Ok(())
    }

    fn sinh(&mut self) -> VoidResult {
        if self.is_complex() {
            self.pure_real_operation(|v, _| v.sinh(), (), Complexity::Medium);
        } else {
            self.pure_complex_operation(|v, _| v.sinh(), (), Complexity::Medium);
        }

        Ok(())
    }

    fn cosh(&mut self) -> VoidResult {
        if self.is_complex() {
            self.pure_real_operation(|v, _| v.cosh(), (), Complexity::Medium);
        } else {
            self.pure_complex_operation(|v, _| v.cosh(), (), Complexity::Medium);
        }

        Ok(())
    }

    fn tanh(&mut self) -> VoidResult {
        if self.is_complex() {
            self.pure_real_operation(|v, _| v.tanh(), (), Complexity::Medium);
        } else {
            self.pure_complex_operation(|v, _| v.tanh(), (), Complexity::Medium);
        }

        Ok(())
    }

    fn asinh(&mut self) -> VoidResult {
        if self.is_complex() {
            self.pure_real_operation(|v, _| v.asinh(), (), Complexity::Medium);
        } else {
            self.pure_complex_operation(|v, _| v.asinh(), (), Complexity::Medium);
        }

        Ok(())
    }

    fn acosh(&mut self) -> VoidResult {
        if self.is_complex() {
            self.pure_real_operation(|v, _| v.acosh(), (), Complexity::Medium);
        } else {
            self.pure_complex_operation(|v, _| v.acosh(), (), Complexity::Medium);
        }

        Ok(())
    }

    fn atanh(&mut self) -> VoidResult {
        if self.is_complex() {
            self.pure_real_operation(|v, _| v.atanh(), (), Complexity::Medium);
        } else {
            self.pure_complex_operation(|v, _| v.atanh(), (), Complexity::Medium);
        }

        Ok(())
    }
}
