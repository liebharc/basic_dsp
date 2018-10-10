use super::super::{
    ComplexNumberSpace, Domain, DspVec, ErrorReason, GetMetaData, MetaData, NumberSpace, PosEq,
    RealNumberSpace, ScalarResult, ToSlice, Vector,
};
use super::kahan_sum;
use inline_vector::InlineVector;
use multicore_support::*;
use numbers::*;
use simd_extensions::*;
use std::ops::*;

/// An operation which multiplies each vector element with a constant
pub trait DotProductOps<A, R, T, N, D>
where
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
    A: GetMetaData<T, N, D>,
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
pub trait PreciseDotProductOps<A, R, T, N, D>
where
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
    A: GetMetaData<T, N, D>,
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

impl<S, T, N, D> DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn dot_product_real<Reg: SimdGeneric<T>, O, NO, DO>(
        &self,
        _: RegType<Reg>,
        factor: &O,
    ) -> ScalarResult<T>
    where
        O: Vector<T> + GetMetaData<T, NO, DO> + Index<RangeFull, Output = [T]>,
        NO: NumberSpace,
        DO: Domain,
    {
        if self.is_complex() {
            return Err(ErrorReason::InputMustBeReal);
        }

        let data_length = self.len();
        let array = self.data.to_slice();
        let (left, right, center) =
            Reg::calc_data_alignment_reqs(&array[0..data_length]);
        let other = &factor[..];
        let chunks = if center.is_some() {
            let vectorization_length = center.unwrap();
            Chunk::get_a_fold_b(
                Complexity::Small,
                self.multicore_settings,
                &other[left..vectorization_length],
                Reg::LEN,
                &array[left..vectorization_length],
                Reg::LEN,
                |original, range, target| {
                    let mut result = Reg::splat(T::zero());
                    let mut i = range.start;
                    let target = Reg::array_to_regs(&target[..]);
                    for a in target {
                        result = result + *a * Reg::load(original, i);
                        i += Reg::LEN;
                    }

                    result.sum_real()
                },
            )
        } else {
            InlineVector::empty()
        };

        let mut i = 0;
        let mut sum = T::zero();
        while i < left {
            sum = sum + array[i] * other[i];
            i += 1;
        }

        let mut i = right;
        while i < data_length {
            sum = sum + array[i] * other[i];
            i += 1;
        }

        let chunk_sum: T = (&chunks[..]).iter().fold(T::zero(), |a, b| a + *b);
        Ok(chunk_sum + sum)
    }

    fn dot_product_complex<Reg: SimdGeneric<T>, O, NO, DO>(
        &self,
        _: RegType<Reg>,
        factor: &O,
    ) -> ScalarResult<Complex<T>>
    where
        O: Vector<T> + GetMetaData<T, NO, DO> + Index<RangeFull, Output = [T]>,
        NO: NumberSpace,
        DO: Domain,
    {
        if !self.is_complex() {
            return Err(ErrorReason::InputMustBeComplex);
        }

        if !factor.is_complex() || self.domain() != factor.domain() {
            return Err(ErrorReason::InputMetaDataMustAgree);
        }

        let data_length = self.len();
        let array = self.data.to_slice();
        let (left, right, center) =
            Reg::calc_data_alignment_reqs(&array[0..data_length]);
        let other = &factor[..];
        let chunks = if center.is_some() {
            let vectorization_length = center.unwrap();
            Chunk::get_a_fold_b(
                Complexity::Small,
                self.multicore_settings,
                &other[left..vectorization_length],
                Reg::LEN,
                &array[left..vectorization_length],
                Reg::LEN,
                |original, range, target| {
                    let mut result = Reg::splat(T::zero());
                    let mut i = range.start;
                    let target = Reg::array_to_regs(&target[..]);
                    for a in target {
                        result = result + a.mul_complex(Reg::load(original, i));
                        i += Reg::LEN;
                    }
                    result.sum_complex()
                },
            )
        } else {
            InlineVector::empty()
        };

        let mut i = 0;
        let mut sum = Complex::<T>::new(T::zero(), T::zero());
        while i < left {
            let a = Complex::<T>::new(array[i], array[i + 1]);
            let b = Complex::<T>::new(other[i], other[i + 1]);
            sum = sum + a * b;
            i += 2;
        }

        let mut i = right;
        while i < data_length {
            let a = Complex::<T>::new(array[i], array[i + 1]);
            let b = Complex::<T>::new(other[i], other[i + 1]);
            sum = sum + a * b;
            i += 2;
        }

        let chunk_sum: Complex<T> = (&chunks[..])
            .iter()
            .fold(Complex::<T>::new(T::zero(), T::zero()), |a, b| a + b);
        Ok(chunk_sum + sum)
    }

    fn dot_product_prec_real<Reg: SimdGeneric<T>, O, NO, DO>(
        &self,
        _: RegType<Reg>,
        factor: &O,
    ) -> ScalarResult<T>
    where
        O: Vector<T> + GetMetaData<T, NO, DO> + Index<RangeFull, Output = [T]>,
        NO: NumberSpace,
        DO: Domain,
    {
        if self.is_complex() {
            return Err(ErrorReason::InputMustBeReal);
        }

        let data_length = self.len();
        let array = self.data.to_slice();
        let (left, right, center) =
            Reg::calc_data_alignment_reqs(&array[0..data_length]);
        let other = &factor[..];
        let chunks = if center.is_some() {
            let vectorization_length = center.unwrap();
            Chunk::get_a_fold_b(
                Complexity::Small,
                self.multicore_settings,
                &other[left..vectorization_length],
                Reg::LEN,
                &array[left..vectorization_length],
                Reg::LEN,
                |original, range, target| {
                    let mut i = range.start;
                    let target = Reg::array_to_regs(&target[..]);
                    kahan_sum(target.iter().map(|a| {
                        let res = *a * Reg::load(original, i);
                        i += Reg::LEN;
                        res
                    }))
                    .sum_real()
                },
            )
        } else {
            InlineVector::empty()
        };

        let mut i = 0;
        let mut sum = T::zero();
        while i < left {
            sum = sum + array[i] * other[i];
            i += 1;
        }

        let mut i = right;
        while i < data_length {
            sum = sum + array[i] * other[i];
            i += 1;
        }

        let chunk_sum: T = (&chunks[..]).iter().fold(T::zero(), |a, b| a + *b);
        Ok(chunk_sum + sum)
    }

    fn dot_product_prec_complex<Reg: SimdGeneric<T>, O, NO, DO>(
        &self,
        _: RegType<Reg>,
        factor: &O,
    ) -> ScalarResult<Complex<T>>
    where
        O: Vector<T> + GetMetaData<T, NO, DO> + Index<RangeFull, Output = [T]>,
        NO: NumberSpace,
        DO: Domain,
    {
        if !self.is_complex() {
            return Err(ErrorReason::InputMustBeComplex);
        }

        if !factor.is_complex() || self.domain() != factor.domain() {
            return Err(ErrorReason::InputMetaDataMustAgree);
        }

        let data_length = self.len();
        let array = self.data.to_slice();
        let (left, right, center) =
            Reg::calc_data_alignment_reqs(&array[0..data_length]);
        let other = &factor[..];
        let chunks = if center.is_some() {
            let vectorization_length = center.unwrap();
            Chunk::get_a_fold_b(
                Complexity::Small,
                self.multicore_settings,
                &other[left..vectorization_length],
                Reg::LEN,
                &array[left..vectorization_length],
                Reg::LEN,
                |original, range, target| {
                    let mut i = range.start;
                    let target = Reg::array_to_regs(&target[..]);
                    kahan_sum(target.iter().map(|a| {
                        let res = a.mul_complex(Reg::load(original, i));
                        i += Reg::LEN;
                        res
                    }))
                    .sum_complex()
                },
            )
        } else {
            InlineVector::empty()
        };

        let mut i = 0;
        let mut sum = Complex::<T>::new(T::zero(), T::zero());
        while i < left {
            let a = Complex::<T>::new(array[i], array[i + 1]);
            let b = Complex::<T>::new(other[i], other[i + 1]);
            sum = sum + a * b;
            i += 2;
        }

        let mut i = right;
        while i < data_length {
            let a = Complex::<T>::new(array[i], array[i + 1]);
            let b = Complex::<T>::new(other[i], other[i + 1]);
            sum = sum + a * b;
            i += 2;
        }

        let chunk_sum: Complex<T> = (&chunks[..])
            .iter()
            .fold(Complex::<T>::new(T::zero(), T::zero()), |a, b| a + b);
        Ok(chunk_sum + sum)
    }
}

impl<S, O, T, N, D, NO, DO> DotProductOps<O, T, T, NO, DO> for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: RealNumberSpace,
    D: Domain,
    O: Vector<T> + GetMetaData<T, NO, DO> + Index<RangeFull, Output = [T]>,
    NO: PosEq<N> + NumberSpace,
    DO: PosEq<D> + Domain,
{
    type Output = ScalarResult<T>;

    fn dot_product(&self, factor: &O) -> ScalarResult<T> {
        sel_reg!(self.dot_product_real::<T>(factor))
    }
}

impl<S, O, T, N, D, NO, DO> DotProductOps<O, Complex<T>, T, NO, DO> for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: ComplexNumberSpace,
    D: Domain,
    O: Vector<T> + GetMetaData<T, NO, DO> + Index<RangeFull, Output = [T]>,
    NO: PosEq<N> + NumberSpace,
    DO: PosEq<D> + Domain,
{
    type Output = ScalarResult<Complex<T>>;

    fn dot_product(&self, factor: &O) -> ScalarResult<Complex<T>> {
        sel_reg!(self.dot_product_complex::<T>(factor))
    }
}

impl<S, O, T, N, D, NO, DO> PreciseDotProductOps<O, T, T, NO, DO> for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: RealNumberSpace,
    D: Domain,
    O: Vector<T> + GetMetaData<T, NO, DO> + Index<RangeFull, Output = [T]>,
    NO: PosEq<N> + NumberSpace,
    DO: PosEq<D> + Domain,
{
    type Output = ScalarResult<T>;

    fn dot_product_prec(&self, factor: &O) -> ScalarResult<T> {
        sel_reg!(self.dot_product_prec_real::<T>(factor))
    }
}

impl<S, O, T, N, D, NO, DO> PreciseDotProductOps<O, Complex<T>, T, NO, DO> for DspVec<S, T, N, D>
where
    S: ToSlice<T>,
    T: RealNumber,
    N: ComplexNumberSpace,
    D: Domain,
    O: Vector<T> + GetMetaData<T, NO, DO> + Index<RangeFull, Output = [T]>,
    NO: PosEq<N> + NumberSpace,
    DO: PosEq<D> + Domain,
{
    type Output = ScalarResult<Complex<T>>;

    fn dot_product_prec(&self, factor: &O) -> ScalarResult<Complex<T>> {
        sel_reg!(self.dot_product_prec_complex::<T>(factor))
    }
}

// The test cases for dot product check mainly that the compiler accepts certain combinations of vectors
// and less about the result of the dot product.
#[cfg(test)]
mod tests {
    use super::super::super::*;
    use num_complex::Complex32;

    #[test]
    fn real_and_real() {
        let vec1: Vec<f32> = vec![1.0, 2.0, 3.0];
        let vec2: Vec<f32> = vec![1.0, 2.0, 3.0];
        let dsp1 = vec1.to_real_time_vec();
        let dsp2 = vec2.to_real_time_vec();
        let res = dsp1.dot_product(&dsp2).unwrap();
        assert_eq!(res, 14.0);
    }

    #[test]
    fn real_and_gen() {
        let vec1: Vec<f32> = vec![1.0, 2.0, 3.0];
        let vec2: Vec<f32> = vec![1.0, 2.0, 3.0];
        let dsp1 = vec1.to_real_time_vec();
        let dsp2 = vec2.to_gen_dsp_vec(false, DataDomain::Time);
        let res = dsp1.dot_product(&dsp2).unwrap();
        assert_eq!(res, 14.0);
    }

    #[test]
    fn complex_and_complex() {
        let vec1: Vec<f32> = vec![1.0, 0.0, 3.0, 0.0];
        let vec2: Vec<f32> = vec![1.0, 0.0, 3.0, 0.0];
        let dsp1 = vec1.to_complex_time_vec();
        let dsp2 = vec2.to_complex_time_vec();
        let res = dsp1.dot_product(&dsp2).unwrap();
        assert_eq!(res, Complex32::new(10.0, 0.0));
    }

    #[test]
    fn complex_and_gen() {
        let vec1: Vec<f32> = vec![1.0, 0.0, 3.0, 0.0];
        let vec2: Vec<f32> = vec![1.0, 0.0, 3.0, 0.0];
        let dsp1 = vec1.to_complex_time_vec();
        let dsp2 = vec2.to_gen_dsp_vec(true, DataDomain::Time);
        let res = dsp1.dot_product(&dsp2).unwrap();
        assert_eq!(res, Complex32::new(10.0, 0.0));
    }

    #[test]
    fn freq_and_freq() {
        let vec1: Vec<f32> = vec![1.0, 0.0, 3.0, 0.0];
        let vec2: Vec<f32> = vec![1.0, 0.0, 3.0, 0.0];
        let dsp1 = vec1.to_complex_freq_vec();
        let dsp2 = vec2.to_complex_freq_vec();
        let res = dsp1.dot_product(&dsp2).unwrap();
        assert_eq!(res, Complex32::new(10.0, 0.0));
    }
}
