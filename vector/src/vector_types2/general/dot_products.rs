use RealNumber;
use num::Complex;
use multicore_support::*;
use simd_extensions::*;
use super::super::{
	Vector, ScalarResult, ErrorReason,
    DspVec, ToSlice,
    Domain, RealNumberSpace, ComplexNumberSpace
};

/// An operation which multiplies each vector element with a constant
pub trait DotProductOps<R> : Sized
    where R: Sized {
    /// Calculates the dot product of self and factor. Self and factor remain unchanged.
	///
	/// # Example
	///
    /// ```
    /// use basic_dsp_vector::vector_types2::*;
    /// let vector1 = vec!(2.0, 1.0, -1.0, 4.0).to_real_time_vec();
    /// let vector2 = vec!(3.0, 4.0, -1.0, -2.0).to_real_time_vec();
    /// let product = vector1.dot_product(&vector2).expect("Ignoring error handling in examples");
    /// assert_eq!(3.0, product);
    /// ```
    fn dot_product(&self, factor: &Self) -> ScalarResult<R>;
}

impl<S, T, N, D> DotProductOps<T> for DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber,
          N: RealNumberSpace,
          D: Domain {

    fn dot_product(&self, factor: &Self) -> ScalarResult<T>
    {
        if self.is_complex() {
            return Err(ErrorReason::InputMustBeReal);
        }

        let data_length = self.len();
        let array = self.data.to_slice();
        let (scalar_left, scalar_right, vectorization_length) = T::Reg::calc_data_alignment_reqs(&array[0..data_length]);
        let other = factor.data.to_slice();
        let chunks = if vectorization_length > 0 {
            Chunk::get_a_fold_b(
                Complexity::Small, &self.multicore_settings,
                &other[0..vectorization_length], T::Reg::len(),
                &array[0..vectorization_length], T::Reg::len(),
                |original, range, target| {
                    let mut i = 0;
                    let mut j = range.start;
                    let mut result = T::Reg::splat(T::zero());
                    while i < target.len()
                    {
                        let vector1 = T::Reg::load_unchecked(original, j);
                        let vector2 = T::Reg::load_unchecked(target, i);
                        result = result + (vector2 * vector1);
                        i += T::Reg::len();
                        j += T::Reg::len();
                    }

                    result.sum_real()
            })
        } else {
            Vec::new()
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

        let chunk_sum: T = chunks.iter().fold(T::zero(), |a, b| a + *b);
        Ok(chunk_sum + sum)
    }
}

impl<S, T, N, D> DotProductOps<Complex<T>> for DspVec<S, T, N, D>
    where S: ToSlice<T>,
          T: RealNumber,
          N: ComplexNumberSpace,
          D: Domain {

    fn dot_product(&self, factor: &Self) -> ScalarResult<Complex<T>> {
        if !self.is_complex() {
            return Err(ErrorReason::InputMustBeComplex);
        }

        if !factor.is_complex() ||
            self.domain != factor.domain {
            return Err(ErrorReason::InputMetaDataMustAgree);
        }

        let data_length = self.len();
        let array = self.data.to_slice();
        let (scalar_left, scalar_right, vectorization_length) = T::Reg::calc_data_alignment_reqs(&array[0..data_length]);
        let other = factor.data.to_slice();
        let chunks = if vectorization_length > 0 {
            Chunk::get_a_fold_b(
                Complexity::Small, &self.multicore_settings,
                &other[scalar_left..vectorization_length], T::Reg::len(),
                &array[scalar_left..vectorization_length], T::Reg::len(),
                |original, range, target| {
                    let mut i = 0;
                    let mut j = range.start;
                    let mut result = T::Reg::splat(T::zero());
                    while i < target.len()
                    {
                        let vector1 = T::Reg::load_unchecked(original, j);
                        let vector2 = T::Reg::load_unchecked(target, i);
                        result = result + (vector2.mul_complex(vector1));
                        i += T::Reg::len();
                        j += T::Reg::len();
                    }

                result.sum_complex()
            })
        } else {
            Vec::new()
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

        let chunk_sum: Complex<T> = chunks.iter().fold(Complex::<T>::new(T::zero(), T::zero()), |a, b| a + b);
        Ok(chunk_sum + sum)
    }
}
