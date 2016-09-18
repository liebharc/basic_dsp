use multicore_support::{Chunk, Complexity};
use super::definitions::{
    DataVec,
    TransRes,
    VoidResult,
    ErrorReason,
    ScalarResult,
    Statistics,
    ScaleOps,
    VectorIter,
    OffsetOps,
    DotProductOps,
    StatisticsOps,
    ComplexVectorOps};
use super::{
    array_to_complex,
    array_to_complex_mut,
    GenericDataVec};
use super::stats_impl::Stats;
use simd_extensions::*;
use num::complex::Complex;
use std::sync::Arc;

macro_rules! add_complex_impl {
    ($($data_type:ident, $reg:ident);*)
     =>
     {
        $(
            impl ComplexVectorOps<$data_type> for GenericDataVec<$data_type>
            {
                type RealPartner = GenericDataVec<$data_type>;

                fn complex_offset(self, offset: Complex<$data_type>)  -> TransRes<Self>
                {
                    assert_complex!(self);
                    let vector_offset = $reg::from_complex(offset);
                    self.simd_complex_operation(|x,y| x + y, |x,y| x + Complex::<$data_type>::new(y.extract(0), y.extract(1)), vector_offset, Complexity::Small)
                }

                fn complex_scale(self, factor: Complex<$data_type>) -> TransRes<Self>
                {
                    assert_complex!(self);
                    self.simd_complex_operation(|x,y| x.scale_complex(y), |x,y| x * y, factor, Complexity::Small)
                }

                fn multiply_complex_exponential(mut self, a: $data_type, b: $data_type) -> TransRes<Self>
                {
                    assert_complex!(self);
                    {
                        let a = a * self.delta();
                        let data_length = self.len();
                        let scalar_length = data_length % $reg::len();
                        let vectorization_length = data_length - scalar_length;
                        let mut array = &mut self.data;
                        Chunk::execute_with_range(
                            Complexity::Small, &self.multicore_settings,
                            &mut array[0..vectorization_length], $reg::len(),
                            (a, b),
                            move |array, range, args| {
                            let (a, b) = args;
                            let mut exponential =
                                Complex::<$data_type>::from_polar(&1.0, &b)
                                * Complex::<$data_type>::from_polar(&1.0, &(a * range.start as $data_type / 2.0));
                            let increment = Complex::<$data_type>::from_polar(&1.0, &a);
                            let array = array_to_complex_mut(array);
                            for complex in array {
                                *complex = (*complex) * exponential;
                                exponential = exponential * increment;
                            }
                        });
                        let mut exponential =
                            Complex::<$data_type>::from_polar(&1.0, &b)
                            * Complex::<$data_type>::from_polar(&1.0, &(a * vectorization_length as $data_type / 2.0));
                        let increment = Complex::<$data_type>::from_polar(&1.0, &a);
                        let array = array_to_complex_mut(&mut array[vectorization_length..data_length]);
                        for complex in array {
                            *complex = (*complex) * exponential;
                            exponential = exponential * increment;
                        }
                    }

                    Ok(self)
                }

                fn magnitude(self) -> TransRes<Self>
                {
                    assert_complex!(self);
                    self.simd_complex_to_real_operation(|x,_arg| x.complex_abs(), |x,_arg| x.norm(), (), Complexity::Small)
                }

                fn get_magnitude(&self, destination: &mut Self) -> VoidResult
                {
                    if !self.is_complex {
                        return Err(ErrorReason::InputMustBeComplex);
                    }

                    let data_length = self.len();
                    destination.reallocate(data_length / 2);
                    let (scalar_left, scalar_right, vectorization_length) = $reg::calc_data_alignment_reqs(&self.data[0..data_length]);
                    let array = &self.data;
                    let mut temp = &mut destination.data;
                    Chunk::from_src_to_dest(
                        Complexity::Small, &self.multicore_settings,
                        &array[scalar_left..vectorization_length], $reg::len(),
                        &mut temp[scalar_left..vectorization_length/2], $reg::len() / 2, (),
                        move |array, range, target, _arg| {
                            let mut i = 0;
                            let mut j = range.start;
                            while i < target.len()
                            {
                                let vector = $reg::load_unchecked(array, j);
                                let result = vector.complex_abs();
                                result.store_half_unchecked(target, i);
                                j += $reg::len();
                                i += $reg::len() / 2;
                            }
                        });

                    let mut i = 0;
                    while i < scalar_left
                    {
                        temp[i / 2] = (array[i] * array[i] + array[i + 1] * array[i + 1]).sqrt();
                        i += 2;
                    }

                    let mut i = scalar_right;
                    while i < data_length
                    {
                        temp[i / 2] = (array[i] * array[i] + array[i + 1] * array[i + 1]).sqrt();
                        i += 2;
                    }

                    destination.is_complex = false;
                    destination.delta = self.delta;
                    Ok(())
                }

                fn magnitude_squared(self) -> TransRes<Self>
                {
                    assert_complex!(self);
                    self.simd_complex_to_real_operation(|x,_arg| x.complex_abs_squared(), |x,_arg| x.re * x.re + x.im * x.im, (), Complexity::Small)
                }

                fn conj(self) -> TransRes<Self>
                {
                    assert_complex!(self);
                    let factor = $reg::from_complex(Complex::<$data_type>::new(1.0, -1.0));
                    self.simd_complex_operation(|x,y| x * y, |x,_| x.conj(), factor, Complexity::Small)
                }

                fn to_real(self) -> TransRes<Self>
                {
                    assert_complex!(self);
                    self.pure_complex_to_real_operation(|x,_arg|x.re, (), Complexity::Small)
                }

                fn to_imag(self) -> TransRes<Self>
                {
                    assert_complex!(self);
                    self.pure_complex_to_real_operation(|x,_arg|x.im, (), Complexity::Small)
                }

                fn get_real(&self, destination: &mut Self) -> VoidResult
                {
                    if !self.is_complex {
                        return Err(ErrorReason::InputMustBeComplex);
                    }

                    self.pure_complex_into_real_target_operation(destination, |x,_arg|x.re, (), Complexity::Small)
                }

                fn get_imag(&self, destination: &mut Self) -> VoidResult
                {
                    if !self.is_complex {
                        return Err(ErrorReason::InputMustBeComplex);
                    }

                    self.pure_complex_into_real_target_operation(destination, |x,_arg|x.im, (), Complexity::Small)
                }

                fn phase(self) -> TransRes<Self>
                {
                    assert_complex!(self);
                    self.pure_complex_to_real_operation(|x,_arg|x.arg(), (), Complexity::Small)
                }

                fn get_phase(&self, destination: &mut Self) -> VoidResult
                {
                    if !self.is_complex {
                        return Err(ErrorReason::InputMustBeComplex);
                    }

                    self.pure_complex_into_real_target_operation(destination, |x,_arg|x.arg(), (), Complexity::Small)
                }

                fn map_inplace_complex<A, F>(mut self, argument: A, f: F) -> TransRes<Self>
                    where A: Sync + Copy + Send,
                          F: Fn(Complex<$data_type>, usize, A) -> Complex<$data_type> + 'static + Sync {
                    {
                        assert_complex!(self);
                        let mut array = &mut self.data;
                        let length = array.len();
                        Chunk::execute_with_range(
                            Complexity::Small, &self.multicore_settings,
                            &mut array[0..length], 2, argument,
                            move|array, range, argument| {
                                let mut i = range.start / 2;
                                let array = array_to_complex_mut(array);
                                for num in array {
                                    *num = f(*num, i, argument);
                                    i += 1;
                                }
                            });
                    }
                    Ok(self)
                }

                fn map_aggregate_complex<A, FMap, FAggr, R>(
                    &self,
                    argument: A,
                    map: FMap,
                    aggregate: FAggr) -> ScalarResult<R>
                        where A: Sync + Copy + Send,
                              FMap: Fn(Complex<$data_type>, usize, A) -> R + 'static + Sync,
                              FAggr: Fn(R, R) -> R + 'static + Sync + Send,
                              R: Send {

                    let aggregate = Arc::new(aggregate);
                    let mut result = {
                        if !self.is_complex {
                            return Err(ErrorReason::InputMustBeComplex);
                        }

                        let array = &self.data;
                        let length = array.len();
                        if length == 0 {
                            return Err(ErrorReason::InputMustNotBeEmpty);
                        }
                        let aggregate  = aggregate.clone();
                        Chunk::map_on_array_chunks(
                            Complexity::Small, &self.multicore_settings,
                            &array[0..length], 2, argument,
                            move|array, range, argument| {
                                let aggregate  = aggregate.clone();
                                let array = array_to_complex(array);
                                let mut i = range.start / 2;
                                let mut sum: Option<R> = None;
                                for num in array {
                                    let res = map(*num, i, argument);
                                    sum = match sum {
                                        None => Some(res),
                                        Some(s) => Some(aggregate(s, res))
                                    };
                                    i += 1;
                                }
                                sum
                            })
                    };
                    let aggregate  = aggregate.clone();
                    // Would be nicer if we could use iter().fold(..) but we need
                    // the value of R and not just a reference so we can't user an iter
                    let mut only_valid_options = Vec::with_capacity(result.len());
                    for _ in 0..result.len() {
                        let elem = result.pop().unwrap();
                        match elem {
                            None => (),
                            Some(e) => only_valid_options.push(e)
                        };
                    }

                    if only_valid_options.len() == 0 {
                        return Err(ErrorReason::InputMustNotBeEmpty);
                    }
                    let mut aggregated = only_valid_options.pop().unwrap();
                    for _ in 0..only_valid_options.len() {
                        aggregated = aggregate(aggregated, only_valid_options.pop().unwrap());
                    }
                    Ok(aggregated)
                }

                fn complex_dot_product(&self, factor: &Self) -> ScalarResult<Complex<$data_type>>
                {
                    if !self.is_complex {
                        return Err(ErrorReason::InputMustBeComplex);
                    }

                    if !factor.is_complex ||
                        self.domain != factor.domain {
                        return Err(ErrorReason::InputMetaDataMustAgree);
                    }

                    let data_length = self.len();
                    let (scalar_left, scalar_right, vectorization_length) = $reg::calc_data_alignment_reqs(&self.data[0..data_length]);
                    let array = &self.data;
                    let other = &factor.data;
                    let chunks = Chunk::get_a_fold_b(
                        Complexity::Small, &self.multicore_settings,
                        &other[scalar_left..vectorization_length], $reg::len(),
                        &array[scalar_left..vectorization_length], $reg::len(),
                        |original, range, target| {
                            let mut i = 0;
                            let mut j = range.start;
                            let mut result = $reg::splat(0.0);
                            while i < target.len()
                            {
                                let vector1 = $reg::load_unchecked(original, j);
                                let vector2 = $reg::load_unchecked(target, i);
                                result = result + (vector2.mul_complex(vector1));
                                i += $reg::len();
                                j += $reg::len();
                            }

                        result.sum_complex()
                    });

                    let mut i = 0;
                    let mut sum = Complex::<$data_type>::new(0.0, 0.0);
                    while i < scalar_left {
                        let a = Complex::<$data_type>::new(array[i], array[i + 1]);
                        let b = Complex::<$data_type>::new(other[i], other[i + 1]);
                        sum = sum + a * b;
                        i += 2;
                    }

                    let mut i = scalar_right;
                    while i < data_length {
                        let a = Complex::<$data_type>::new(array[i], array[i + 1]);
                        let b = Complex::<$data_type>::new(other[i], other[i + 1]);
                        sum = sum + a * b;
                        i += 2;
                    }

                    let chunk_sum: Complex<$data_type> = chunks.iter().fold(Complex::<$data_type>::new(0.0, 0.0), |a, b| a + b);
                    Ok(chunk_sum + sum)
                }

                fn complex_statistics(&self) -> Statistics<Complex<$data_type>> {
                    let data_length = self.len();
                    let array = &self.data;
                    let chunks = Chunk::get_chunked_results(
                        Complexity::Small, &self.multicore_settings,
                        &array[0..data_length], 2, (),
                        |array, range, _arg| {
                            let mut stat = Statistics::<Complex<$data_type>>::empty();
                            let mut j = range.start / 2;
                            let array = array_to_complex(array);
                            for num in array {
                                stat.add(*num, j);
                                j += 1;
                            }
                            stat
                    });

                    Statistics::merge(&chunks)
                }

                fn complex_statistics_splitted(&self, len: usize) -> Vec<Statistics<Complex<$data_type>>> {
                    if len == 0 {
                        return Vec::new();
                    }

                    let data_length = self.len();
                    let array = &self.data;
                    let chunks = Chunk::get_chunked_results (
                        Complexity::Small, &self.multicore_settings,
                        &array[0..data_length], 2, len,
                        |array, range, len| {
                            let mut results = Statistics::<Complex<$data_type>>::empty_vec(len);
                            let mut j = range.start / 2;
                            let array = array_to_complex(array);
                            for num in array {
                                let stat = &mut results[j % len];
                                stat.add(*num, j / len);
                                j += 1;
                            }

                            results
                    });

                    Statistics::merge_cols(&chunks)
                }

                fn get_real_imag(&self, real: &mut Self::RealPartner, imag: &mut Self::RealPartner) -> VoidResult {
                    let data_length = self.len();
                    real.reallocate(data_length / 2);
                    imag.reallocate(data_length / 2);
                    let data = &self.data;
                    for i in 0..data_length {
                        if i % 2 == 0 {
                            real[i / 2] = data[i];
                        } else {
                            imag[i / 2] = data[i];
                        }
                    }

                    Ok(())
                }

                fn get_mag_phase(&self, mag: &mut Self::RealPartner, phase: &mut Self::RealPartner) -> VoidResult {
                    let data_length = self.len();
                    mag.reallocate(data_length / 2);
                    phase.reallocate(data_length / 2);
                    let data = &self.data;
                    let mut i = 0;
                    while i < data_length {
                        let c = Complex::<$data_type>::new(data[i], data[i + 1]);
                        let (m, p) = c.to_polar();
                        mag[i / 2] = m;
                        phase[i / 2] = p;
                        i += 2;
                    }

                    Ok(())
                }

                fn set_real_imag(mut self, real: &Self::RealPartner, imag: &Self::RealPartner) -> TransRes<Self> {
                    {
                        reject_if!(self, real.len() != imag.len(), ErrorReason::InvalidArgumentLength);
                        self.reallocate(2 * real.len());
                        let data_length = self.len();
                        let data = &mut self.data;
                        for i in 0..data_length {
                            if i % 2 == 0 {
                                data[i] = real[i / 2];
                            } else {
                                data[i] = imag[i / 2];
                            }
                        }
                    }

                    Ok(self)
                }

                fn set_mag_phase(mut self, mag: &Self::RealPartner, phase: &Self::RealPartner) -> TransRes<Self> {
                    {
                        reject_if!(self, mag.len() != phase.len(), ErrorReason::InvalidArgumentLength);
                        self.reallocate(2 * mag.len());
                        let data_length = self.len();
                        let data = &mut self.data;
                        let mut i = 0;
                        while i < data_length {
                            let c = Complex::<$data_type>::from_polar(&mag[i / 2], &phase[i / 2]);
                            data[i] = c.re;
                            data[i + 1] = c.im;
                            i += 2;
                        }
                    }

                    Ok(self)
                }

                fn complex_sum(&self) -> Complex<$data_type> {
                    let data_length = self.len();
                    let (scalar_left, scalar_right, vectorization_length) = $reg::calc_data_alignment_reqs(&self.data[0..data_length]);
                    let array = &self.data;
                    let mut sum =
                        if vectorization_length > 0 {
                            let chunks = Chunk::get_chunked_results(
                                Complexity::Small, &self.multicore_settings,
                                &array[scalar_left..vectorization_length], $reg::len(), (),
                                move |array, _, _| {
                                let array = $reg::array_to_regs(array);
                                let mut sum = $reg::splat(0.0);
                                for reg in array {
                                    sum = sum + *reg;
                                }
                                sum
                            });
                            chunks.iter()
                                .map(|v|v.sum_complex())
                                .fold(Complex::<$data_type>::new(0.0, 0.0), |acc, x| acc + x)
                        }
                        else {
                            Complex::<$data_type>::new(0.0, 0.0)
                        };
                    for num in array_to_complex(&array[0..scalar_left])
                    {
                        sum = sum + *num;
                    }
                    for num in array_to_complex(&array[scalar_right..data_length])
                    {
                        sum = sum + *num;
                    }
                    sum
                }

                fn complex_sum_sq(&self) -> Complex<$data_type> {
                    let data_length = self.len();
                    let (scalar_left, scalar_right, vectorization_length) = $reg::calc_data_alignment_reqs(&self.data[0..data_length]);
                    let array = &self.data;
                    let mut sum =
                        if vectorization_length > 0 {
                            let chunks = Chunk::get_chunked_results(
                                Complexity::Small, &self.multicore_settings,
                                &array[scalar_left..vectorization_length], $reg::len(), (),
                                move |array, _, _| {
                                let array = $reg::array_to_regs(array);
                                let mut sum = $reg::splat(0.0);
                                for reg in array {
                                    sum = sum + reg.mul_complex(*reg);
                                }
                                sum
                            });
                            chunks.iter()
                                .map(|v|v.sum_complex())
                                .fold(Complex::<$data_type>::new(0.0, 0.0), |acc, x| acc + x)
                        }
                        else {
                            Complex::<$data_type>::new(0.0, 0.0)
                        };
                    for num in array_to_complex(&array[0..scalar_left])
                    {
                        sum = sum + *num * *num;
                    }
                    for num in array_to_complex(&array[scalar_right..data_length])
                    {
                        sum = sum + *num * *num;
                    }
                    sum
                }
            }

            impl GenericDataVec<$data_type> {
                fn pure_complex_into_real_target_operation<A, F>(&self, destination: &mut Self, op: F, argument: A, complexity: Complexity) -> VoidResult
                    where A: Sync + Copy + Send,
                          F: Fn(Complex<$data_type>, A) -> $data_type + 'static + Sync {
                    let len = self.len();
                    destination.reallocate(len / 2);
                    destination.delta = self.delta;
                    destination.is_complex = false;
                    let mut array = &mut destination.data;
                    let source = &self.data;
                    Chunk::from_src_to_dest(
                        complexity, &self.multicore_settings,
                        &source[0..len], 2,
                        &mut array[0..len/2], 1, argument,
                        move|original, range, target, argument| {
                            let mut i = range.start;
                            let mut j = 0;
                            while j < target.len()
                            {
                                let complex = Complex::<$data_type>::new(original[i], original[i + 1]);
                                target[j] = op(complex, argument);
                                i += 2;
                                j += 1;
                            }
                        });
                    Ok(())
                }
            }

            impl ScaleOps<Complex<$data_type>> for GenericDataVec<$data_type> {
                fn scale(self, offset: Complex<$data_type>) -> TransRes<Self> {
                    self.complex_scale(offset)
                }
            }

            impl OffsetOps<Complex<$data_type>> for GenericDataVec<$data_type> {
                fn offset(self, offset: Complex<$data_type>) -> TransRes<Self> {
                    self.complex_offset(offset)
                }
            }

            impl DotProductOps<Complex<$data_type>> for GenericDataVec<$data_type> {
                type SumResult = ScalarResult<Complex<$data_type>>;
                fn dot_product(&self, factor: &Self) -> ScalarResult<Complex<$data_type>> {
                    self.complex_dot_product(factor)
                }
            }

            impl StatisticsOps<Complex<$data_type>> for GenericDataVec<$data_type> {
                fn statistics(&self) -> Statistics<Complex<$data_type>> {
                    self.complex_statistics()
                }

                fn statistics_splitted(&self, len: usize) -> Vec<Statistics<Complex<$data_type>>> {
                    self.complex_statistics_splitted(len)
                }

                fn sum(&self) -> Complex<$data_type> {
                    self.complex_sum()
                }

                fn sum_sq(&self) -> Complex<$data_type> {
                    self.complex_sum_sq()
                }
            }

            impl VectorIter<Complex<$data_type>> for GenericDataVec<$data_type> {
                fn map_inplace<A, F>(self, argument: A, map: F) -> TransRes<Self>
                    where A: Sync + Copy + Send,
                          F: Fn(Complex<$data_type>, usize, A) -> Complex<$data_type> + 'static + Sync {
                    self.map_inplace_complex(argument, map)
                }

                fn map_aggregate<A, FMap, FAggr, R>(
                    &self,
                    argument: A,
                    map: FMap,
                    aggregate: FAggr) -> ScalarResult<R>
                where A: Sync + Copy + Send,
                      FMap: Fn(Complex<$data_type>, usize, A) -> R + 'static + Sync,
                      FAggr: Fn(R, R) -> R + 'static + Sync + Send,
                      R: Send {
                    self.map_aggregate_complex(argument, map, aggregate)
                }
            }
        )*
     }
}
add_complex_impl!(f32, Reg32; f64, Reg64);
