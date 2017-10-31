use numbers::*;
use multicore_support::*;
use simd_extensions::*;
use super::super::{Vector, MetaData, DspVec, ToSliceMut, Domain, RealNumberSpace};

/// Operations on real types.
///
/// # Failures
///
/// If one of the methods is called on complex data then `self.len()` will be set to `0`.
/// To avoid this it's recommended to use the `to_real_time_vec`, `to_real_freq_vec`
/// `to_complex_time_vec` and `to_complex_freq_vec` constructor methods since
/// the resulting types will already check at compile time (using the type system)
/// that the data is real.
pub trait RealOps {
    /// Gets the absolute value of all vector elements.
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(1.0, -2.0).to_real_time_vec();
    /// vector.abs();
    /// assert_eq!([1.0, 2.0], vector[..]);
    /// ```
    fn abs(&mut self);
}

/// Operations on real types.
///
/// # Failures
///
/// If one of the methods is called on complex data then `self.len()` will be set to `0`.
/// To avoid this it's recommended to use the `to_real_time_vec`, `to_real_freq_vec`
/// `to_complex_time_vec` and `to_complex_freq_vec` constructor methods since
/// the resulting types will already check at compile time (using the type system)
/// that the data is real.
pub trait ModuloOps<T>
    where T: RealNumber
{
    /// Each value in the vector is dividable by the divisor and the remainder
    /// is stored in the resulting
    /// vector. This the same a modulo operation or to phase wrapping.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0).to_real_time_vec();
    /// vector.wrap(4.0);
    /// assert_eq!([1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0], vector[..]);
    /// ```
    fn wrap(&mut self, divisor: T);

    /// This function corrects the jumps in the given vector which occur due
    /// to wrap or modulo operations.
    /// This will undo a wrap operation only if the deltas are smaller than half the divisor.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.).to_real_time_vec();
    /// vector.unwrap(4.0);
    /// assert_eq!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vector[..]);
    /// ```
    fn unwrap(&mut self, divisor: T);
}

/// Recommended to be only used with the CPU feature flags `sse` or `avx`.
///
/// This trait provides alternative implementations for some standard functions which
/// are less accurate but perform faster. Those approximations are written for SSE2
/// (target feature flag `sse`) or AVX2 (target feature flag `avx`) processors. Without any of those
/// feature flags the standard library functions will be used instead.
///
/// Information on the error of the approximation and their performance are rough numbers.
/// A detailed table can be obtained by running the `approx_accuracy` example.
///
/// # Failures
///
/// If one of the methods is called on complex data then `self.len()` will be set to `0`.
/// To avoid this it's recommended to use the `to_real_time_vec`, `to_real_freq_vec`
/// `to_complex_time_vec` and `to_complex_freq_vec` constructor methods since
/// the resulting types will already check at compile time (using the type system)
/// that the data is real.
pub trait ApproximatedOps<T>
    where T: RealNumber
{
    /// Computes the principal value approximation of natural logarithm of every element in the vector.
    ///
    /// Error should be below `1%` as long as the values in the vector are larger than `1`.
    /// Single core performance should be about `5x` as fast.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::f64;
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(2.718281828459045, 7.389056, 20.085537).to_real_time_vec();
    /// vector.ln_approx();
    /// let actual = &vector[0..];
    /// let expected = &[1.0, 2.0, 3.0];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!(f64::abs(actual[i] - expected[i]) < 1e-2);
    /// }
    /// ```
    fn ln_approx(&mut self);

    /// Calculates the natural exponential approximation for every vector element.
    ///
    /// Error should be less than `1%`` as long as the values in the vector are small
    /// (e.g. in the range between -10 and 10).
    /// Single core performance should be about `50%` faster.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::f64;
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(1.0, 2.0, 3.0).to_real_time_vec();
    /// vector.exp_approx();
    /// let actual = &vector[0..];
    /// let expected = &[2.718281828459045, 7.389056, 20.085537];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!(f64::abs(actual[i] - expected[i]) < 1e-4);
    /// }
    /// ```
    fn exp_approx(&mut self);

    /// Calculates the sine approximation of each element in radians.
    ///
    /// Error should be below `1E-6`.
    /// Single core performance should be about `2x` as fast.
    ///
    /// # Example
    ///
    /// ```
    /// use std::f32;
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(f32::consts::PI/2.0, -f32::consts::PI/2.0).to_real_time_vec();
    /// vector.sin_approx();
    /// assert_eq!([1.0, -1.0], vector[..]);
    /// ```
    fn sin_approx(&mut self);

    /// Calculates the cosine approximation of each element in radians
    ///
    /// Error should be below `1E-6`.
    /// Single core performance should be about `2x` as fast.
    ///
    /// # Example
    ///
    /// ```
    /// use std::f32;
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(2.0 * f32::consts::PI, f32::consts::PI).to_real_time_vec();
    /// vector.cos_approx();
    /// assert_eq!([1.0, -1.0], vector[..]);
    /// ```
    fn cos_approx(&mut self);

    /// Calculates the approximated logarithm to the given base for every vector element.
    ///
    /// Error should be below `1%` as long as the values in the vector are larger than `1`.
    /// Single core performance should be about `5x` as fast.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::f64;
    /// use basic_dsp_vector::*;
    /// let mut vector = vec!(10.0, 100.0, 1000.0).to_real_time_vec();
    /// vector.log_approx(10.0);
    /// let actual = &vector[0..];
    /// let expected = &[1.0, 2.0, 3.0];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!(f64::abs(actual[i] - expected[i]) < 1e-4);
    /// }
    /// ```
    fn log_approx(&mut self, base: T);

    /// Calculates the approximated exponential to the given base for every vector element.
    ///
    /// Error should be less than `1%`` as long as the values in the vector are small
    /// (e.g. in the range between -10 and 10).
    /// Single core performance should be about `5x` as fast.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// use std::f32;
    /// let vector: Vec<f32> = vec!(1.0, 2.0, 3.0);
    /// let mut vector = vector.to_real_time_vec();
    /// vector.expf_approx(10.0);
    /// assert!((vector[0] - 10.0).abs() < 1e-3);
    /// assert!((vector[1] - 100.0).abs() < 1e-3);
    /// assert!((vector[2] - 1000.0).abs() < 1e-3);
    /// ```
    fn expf_approx(&mut self, base: T);

    /// Raises every vector element to approximately a floating point power.
    ///
    /// Error should be less than `1%`` as long as the values in the vector are really small
    /// (e.g. in the range between -0.1 and 0.1).
    /// Single core performance should be about `5x` as fast.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// use std::f32;
    /// let vector: Vec<f32> = vec!(1.0, 2.0, 3.0);
    /// let mut vector = vector.to_real_time_vec();
    /// vector.powf_approx(3.0);
    /// assert!((vector[0] - 1.0).abs() < 1e-3);
    /// assert!((vector[1] - 8.0).abs() < 1e-3);
    /// assert!((vector[2] - 27.0).abs() < 1e-3);
    /// ```
    fn powf_approx(&mut self, exponent: T);
}

macro_rules! assert_real {
    ($self_: ident) => {
        if $self_.is_complex() {
            $self_.valid_len = 0;
            return;
        }
    }
}

impl<S, T, N, D> RealOps for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber,
          N: RealNumberSpace,
          D: Domain
{
    fn abs(&mut self) {
        assert_real!(self);
        self.simd_real_operation(|x, _arg| (x * x).sqrt(),
                                 |x, _arg| x.abs(),
                                 (),
                                 Complexity::Small);
    }
}

impl<S, T, N, D> ModuloOps<T> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber,
          N: RealNumberSpace,
          D: Domain
{
    fn wrap(&mut self, divisor: T) {
        assert_real!(self);
        self.pure_real_operation(|x, y| x % y, divisor, Complexity::Small);
    }

    fn unwrap(&mut self, divisor: T) {
        assert_real!(self);
        let data_length = self.len();
        let data = self.data.to_slice_mut();
        let mut i = 0;
        let mut j = 1;
        let half = divisor / T::from(2.0).unwrap();
        while j < data_length {
            let mut diff = data[j] - data[i];
            if diff > half {
                diff = diff % divisor;
                diff = diff - divisor;
                data[j] = data[i] + diff;
            } else if diff < -half {
                diff = diff % divisor;
                diff = diff + divisor;
                data[j] = data[i] + diff;
            }

            i += 1;
            j += 1;
        }
    }
}

/// Complexity of all approximation functions. Medium, because even if the approximations are faster
/// than the standard version, it seems to be benificial to spawn threads early.
const APPROX_COMPLEXITY: Complexity = Complexity::Medium;

impl<S, T, N, D> ApproximatedOps<T> for DspVec<S, T, N, D>
    where S: ToSliceMut<T>,
          T: RealNumber,
          N: RealNumberSpace,
          D: Domain
{
    fn ln_approx(&mut self) {
        assert_real!(self);
        self.simd_real_operation(|x, _arg| x.ln_approx(),
                                 |x, _arg| x.ln(),
                                 (),
                                 APPROX_COMPLEXITY);
    }

    fn exp_approx(&mut self) {
        assert_real!(self);
        self.simd_real_operation(|x, _arg| x.exp_approx(),
                                 |x, _arg| x.exp(),
                                 (),
                                 APPROX_COMPLEXITY);
    }

    fn sin_approx(&mut self) {
        assert_real!(self);
        self.simd_real_operation(|x, _arg| x.sin_approx(),
                                 |x, _arg| x.sin(),
                                 (),
                                 APPROX_COMPLEXITY);
    }

    fn cos_approx(&mut self) {
        assert_real!(self);
        self.simd_real_operation(|x, _arg| x.cos_approx(),
                                 |x, _arg| x.cos(),
                                 (),
                                 APPROX_COMPLEXITY);
    }

    fn log_approx(&mut self, base: T) {
        assert_real!(self);
        let base_ln = T::Reg::splat(base.ln());
        self.simd_real_operation(|x, b| x.ln_approx() / b,
                                 |x, b| x.ln() / b.extract(0),
                                 (base_ln),
                                 APPROX_COMPLEXITY);
    }

    fn expf_approx(&mut self, base: T) {
        assert_real!(self);
        // Transform base with:
        // x^y = e^(ln(x)*y)
        let base_ln = T::Reg::splat(base.ln());
        self.simd_real_operation(|y, x| (x * y).exp_approx(),
                                 |y, x| (x.extract(0) * y).exp(),
                                 (base_ln),
                                 APPROX_COMPLEXITY);
    }

    fn powf_approx(&mut self, exponent: T) {
        assert_real!(self);
        // Transform base with the same equation as for `expf_approx`
        let exponent = T::Reg::splat(exponent);
        self.simd_real_operation(|x, y| (x.ln_approx() * y).exp_approx(),
                                 |x, y| (x.ln() * y.extract(0)).exp(),
                                 (exponent),
                                 APPROX_COMPLEXITY);

    }
}
