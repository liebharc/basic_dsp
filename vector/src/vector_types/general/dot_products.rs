use {RealNumber, InlineVector};
use num::Complex;
use multicore_support::*;
use simd_extensions::*;
use super::super::{Vector, ScalarResult, ErrorReason, DspVec, ToSlice, MetaData, Domain,
                   RealNumberSpace, ComplexNumberSpace};
use super::kahan_sum;

/// An operation which multiplies each vector element with a constant
pub trait DotProductOps<R, A>: Sized
    where R: Sized
{
    type Output;

    /// Calculates the dot product of self and factor. Self and factor remain unchanged.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// let vector1 = vec!(2.0, 1.0, -1.0, 4.0).to_real_time_vec();
    /// let vector2 = vec!(3.0, 4.0, -1.0, -2.0).to_real_time_vec();
    /// let product = vector1.dot_product(&vector2).expect("Ignoring error handling in examples");
    /// assert_eq!(3.0, product);
    /// ```
    fn dot_product(&self, factor: &A) -> Self::Output;
}

/// An operation which multiplies each vector element with a constant
pub trait PreciseDotProductOps<R, A>: Sized
    where R: Sized
{
    type Output;

    /// Calculates the dot product of self and factor using a more precise
    /// but slower algorithm. Self and factor remain unchanged.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// let vector1 = vec!(2.0, 1.0, -1.0, 4.0).to_real_time_vec();
    /// let vector2 = vec!(3.0, 4.0, -1.0, -2.0).to_real_time_vec();
    /// let product = vector1.dot_product_prec(&vector2).expect("Ignoring error handling in examples");
    /// assert_eq!(3.0, product);
    /// ```
    fn dot_product_prec(&self, factor: &A) -> Self::Output;
}

impl<S, T, N, D> DotProductOps<T, DspVec<S, T, N, D>> for DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber,
          N: RealNumberSpace,
          D: Domain
{
    type Output = ScalarResult<T>;

    fn dot_product(&self, factor: &Self) -> ScalarResult<T> {
        if self.is_complex() {
            return Err(ErrorReason::InputMustBeReal);
        }

        let data_length = self.len();
        let array = self.data.to_slice();
        let (scalar_left, scalar_right, vectorization_length) =
            T::Reg::calc_data_alignment_reqs(&array[0..data_length]);
        let other = factor.data.to_slice();
        let chunks = if vectorization_length > 0 {
            Chunk::get_a_fold_b(Complexity::Small,
                                &self.multicore_settings,
                                &other[0..vectorization_length],
                                T::Reg::len(),
                                &array[0..vectorization_length],
                                T::Reg::len(),
                                |original, range, target| {
                let mut result = T::Reg::splat(T::zero());
                let original = T::Reg::array_to_regs(&original[range]);
                let target = T::Reg::array_to_regs(&target[..]);
                for (a, b) in original.iter().zip(target) {
                    result = result + (*a * *b);
                }

                result.sum_real()
            })
        } else {
            InlineVector::empty()
        };

        let mut i = 0;
        let mut sum = T::zero();
        while i < scalar_left {
            sum = sum + array[i] * other[i];
            i += 1;
        }

        let mut i = scalar_right;
        while i < data_length {
            sum = sum + array[i] * other[i];
            i += 1;
        }

        let chunk_sum: T = (&chunks[..]).iter().fold(T::zero(), |a, b| a + *b);
        Ok(chunk_sum + sum)
    }
}

impl<S, T, N, D> DotProductOps<Complex<T>, DspVec<S, T, N, D>> for DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber,
          N: ComplexNumberSpace,
          D: Domain
{
    type Output = ScalarResult<Complex<T>>;

    fn dot_product(&self, factor: &Self) -> ScalarResult<Complex<T>> {
        if !self.is_complex() {
            return Err(ErrorReason::InputMustBeComplex);
        }

        if !factor.is_complex() || self.domain != factor.domain {
            return Err(ErrorReason::InputMetaDataMustAgree);
        }

        let data_length = self.len();
        let array = self.data.to_slice();
        let (scalar_left, scalar_right, vectorization_length) =
            T::Reg::calc_data_alignment_reqs(&array[0..data_length]);
        let other = factor.data.to_slice();
        let chunks = if vectorization_length > 0 {
            Chunk::get_a_fold_b(Complexity::Small,
                                &self.multicore_settings,
                                &other[scalar_left..vectorization_length],
                                T::Reg::len(),
                                &array[scalar_left..vectorization_length],
                                T::Reg::len(),
                                |original, range, target| {
                let mut result = T::Reg::splat(T::zero());
                let original = T::Reg::array_to_regs(&original[range]);
                let target = T::Reg::array_to_regs(&target[..]);
                for (a, b) in original.iter().zip(target) {
                    result = result + (a.mul_complex(*b));
                }

                result.sum_complex()
            })
        } else {
            InlineVector::empty()
        };

        let mut i = 0;
        let mut sum = Complex::<T>::new(T::zero(), T::zero());
        while i < scalar_left {
            let a = Complex::<T>::new(array[i], array[i + 1]);
            let b = Complex::<T>::new(other[i], other[i + 1]);
            sum = sum + a * b;
            i += 2;
        }

        let mut i = scalar_right;
        while i < data_length {
            let a = Complex::<T>::new(array[i], array[i + 1]);
            let b = Complex::<T>::new(other[i], other[i + 1]);
            sum = sum + a * b;
            i += 2;
        }

        let chunk_sum: Complex<T> = (&chunks[..]).iter()
            .fold(Complex::<T>::new(T::zero(), T::zero()), |a, b| a + b);
        Ok(chunk_sum + sum)
    }
}

impl<S, T, N, D> PreciseDotProductOps<T, DspVec<S, T, N, D>> for DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber,
          N: RealNumberSpace,
          D: Domain
{
    type Output = ScalarResult<T>;

    fn dot_product_prec(&self, factor: &Self) -> ScalarResult<T> {
        if self.is_complex() {
            return Err(ErrorReason::InputMustBeReal);
        }

        let data_length = self.len();
        let array = self.data.to_slice();
        let (scalar_left, scalar_right, vectorization_length) =
            T::Reg::calc_data_alignment_reqs(&array[0..data_length]);
        let other = factor.data.to_slice();
        let chunks = if vectorization_length > 0 {
            Chunk::get_a_fold_b(Complexity::Small,
                                &self.multicore_settings,
                                &other[0..vectorization_length],
                                T::Reg::len(),
                                &array[0..vectorization_length],
                                T::Reg::len(),
                                |original, range, target| {
                let original = T::Reg::array_to_regs(&original[range]);
                let target = T::Reg::array_to_regs(&target[..]);
                kahan_sum(original.iter().zip(target).map(|a|*a.0 * *a.1)).sum_real()
            })
        } else {
            InlineVector::empty()
        };

        let mut i = 0;
        let mut sum = T::zero();
        while i < scalar_left {
            sum = sum + array[i] * other[i];
            i += 1;
        }

        let mut i = scalar_right;
        while i < data_length {
            sum = sum + array[i] * other[i];
            i += 1;
        }

        let chunk_sum: T = (&chunks[..]).iter().fold(T::zero(), |a, b| a + *b);
        Ok(chunk_sum + sum)
    }
}

impl<S, T, N, D> PreciseDotProductOps<Complex<T>, DspVec<S, T, N, D>> for DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber,
          N: ComplexNumberSpace,
          D: Domain
{
    type Output = ScalarResult<Complex<T>>;

    fn dot_product_prec(&self, factor: &Self) -> ScalarResult<Complex<T>> {
        if !self.is_complex() {
            return Err(ErrorReason::InputMustBeComplex);
        }

        if !factor.is_complex() || self.domain != factor.domain {
            return Err(ErrorReason::InputMetaDataMustAgree);
        }

        let data_length = self.len();
        let array = self.data.to_slice();
        let (scalar_left, scalar_right, vectorization_length) =
            T::Reg::calc_data_alignment_reqs(&array[0..data_length]);
        let other = factor.data.to_slice();
        let chunks = if vectorization_length > 0 {
            Chunk::get_a_fold_b(Complexity::Small,
                                &self.multicore_settings,
                                &other[scalar_left..vectorization_length],
                                T::Reg::len(),
                                &array[scalar_left..vectorization_length],
                                T::Reg::len(),
                                |original, range, target| {
                let original = T::Reg::array_to_regs(&original[range]);
                let target = T::Reg::array_to_regs(&target[..]);
                kahan_sum(original.iter().zip(target).map(|a|a.0.mul_complex(*a.1))).sum_complex()
            })
        } else {
            InlineVector::empty()
        };

        let mut i = 0;
        let mut sum = Complex::<T>::new(T::zero(), T::zero());
        while i < scalar_left {
            let a = Complex::<T>::new(array[i], array[i + 1]);
            let b = Complex::<T>::new(other[i], other[i + 1]);
            sum = sum + a * b;
            i += 2;
        }

        let mut i = scalar_right;
        while i < data_length {
            let a = Complex::<T>::new(array[i], array[i + 1]);
            let b = Complex::<T>::new(other[i], other[i + 1]);
            sum = sum + a * b;
            i += 2;
        }

        let chunk_sum: Complex<T> = (&chunks[..]).iter()
            .fold(Complex::<T>::new(T::zero(), T::zero()), |a, b| a + b);
        Ok(chunk_sum + sum)
    }
}
