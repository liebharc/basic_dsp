use multicore_support::{Chunk, Complexity};
use super::definitions::{
	DataVector,
    VecResult,
    VoidResult,
    ErrorReason,
    ScalarResult,
    Statistics,
    Scale,
    Offset,
	ComplexVectorOperations};
use super::GenericDataVector;
use super::stats_impl::Stats;
use simd_extensions::{Simd,Reg32,Reg64};
use num::complex::Complex;
use num::traits::Float;

macro_rules! add_complex_impl {
    ($($data_type:ident, $reg:ident);*)
	 =>
	 {	 
        $(
            impl ComplexVectorOperations<$data_type> for GenericDataVector<$data_type>
            {
                type RealPartner = GenericDataVector<$data_type>;
                
                fn complex_data(&self) -> &[Complex<$data_type>] {
                    let len = self.len();
                    Self::array_to_complex(&self.data[0..len])
                }
                
                fn complex_offset(self, offset: Complex<$data_type>)  -> VecResult<Self>
                {
                    assert_complex!(self);
                    let vector_offset = $reg::from_complex(offset);
                    self.simd_complex_operation(|x,y| x + y, |x,y| x + Complex::<$data_type>::new(y.extract(0), y.extract(1)), vector_offset, Complexity::Small)
                }
                
                fn complex_scale(self, factor: Complex<$data_type>) -> VecResult<Self>
                {
                    assert_complex!(self);
                    self.simd_complex_operation(|x,y| x.scale_complex(y), |x,y| x * y, factor, Complexity::Small)
                }
                
                fn multiply_complex_exponential(mut self, a: $data_type, b: $data_type) -> VecResult<Self>
                {
                    assert_complex!(self);
                    {
                        let a = a * self.delta();
                        let length = self.len();
                        let array = &mut self.data;
                        let array = Self::array_to_complex_mut(&mut array[0..length]);
                        let mut exponential = Complex::<$data_type>::from_polar(&1.0, &b);
                        let increment = Complex::<$data_type>::from_polar(&1.0, &a);
                        for complex in array {
                            *complex = (*complex) * exponential;
                            exponential = exponential * increment;
                        }
                    }
                    
                    Ok(self)
                }
                
                fn magnitude(self) -> VecResult<Self>
                {
                    assert_complex!(self);
                    self.simd_complex_to_real_operation(|x,_arg| x.complex_abs(), |x,_arg| x.norm(), (), Complexity::Small)
                }
                
                fn get_magnitude(&self, destination: &mut Self) -> VoidResult
                {
                    if !self.is_complex {
                        return Err(ErrorReason::VectorMustBeComplex);
                    }
                    
                    let data_length = self.len();
                    destination.reallocate(data_length / 2);
                    let scalar_length = data_length % $reg::len();
                    let vectorization_length = data_length - scalar_length;
                    let array = &self.data;
                    let mut temp = &mut destination.data;
                    Chunk::from_src_to_dest(
                        Complexity::Small, &self.multicore_settings,
                        &array, vectorization_length, $reg::len(), 
                        &mut temp, vectorization_length / 2, $reg::len() / 2, (),
                        move |array, range, target, _arg| {
                            let mut i = 0;
                            let mut j = range.start;
                            while i < target.len()
                            { 
                                let vector = $reg::load(array, j);
                                let result = vector.complex_abs();
                                result.store_half(target, i);
                                j += $reg::len();
                                i += $reg::len() / 2;
                            }
                        });
                    let mut i = vectorization_length;
                    while i + 1 < data_length
                    {
                        temp[i / 2] = (array[i] * array[i] + array[i + 1] * array[i + 1]).sqrt();
                        i += 2;
                    }
                    
                    destination.is_complex = false;
                    destination.delta = self.delta;
                    Ok(())
                }
                
                fn magnitude_squared(self) -> VecResult<Self>
                {
                    assert_complex!(self);
                    self.simd_complex_to_real_operation(|x,_arg| x.complex_abs_squared(), |x,_arg| x.re * x.re + x.im * x.im, (), Complexity::Small)
                }
                
                fn complex_conj(mut self) -> VecResult<Self>
                {
                    assert_complex!(self);
                    {
                        let data_length = self.len();
                        if data_length == 0 {
                            return Ok(self);
                        }
                        
                        let unroll_len = 8;
                        let scalar_length = data_length % unroll_len;
                        let unrolled_length = data_length - scalar_length;           
                        let mut array = &mut self.data;
                        Chunk::execute_partial(
                            Complexity::Small, &self.multicore_settings,
                            &mut array, unrolled_length, unroll_len, (), 
                            move |array, _| {
                                unsafe {
                                    let array_len = array.len();
                                    if array_len == 0 {
                                        return;
                                    }
                                    let end = &mut array[array_len - 1] as *mut $data_type;
                                    let mut array = &mut array[1] as *mut $data_type;
                                    while array <= end {
                                        *array = -(*array);
                                        array = array.offset(2);
                                        *array = -(*array);
                                        array = array.offset(2);
                                        *array = -(*array);
                                        array = array.offset(2);
                                        *array = -(*array);
                                        array = array.offset(2);
                                    }
                                }
                            });
                        if data_length > unrolled_length {
                            let end = &mut array[data_length - 1] as *mut $data_type;
                            let mut array = &mut array[unrolled_length + 1] as *mut $data_type;
                            unsafe {
                                while array <= end {
                                    *array = -(*array);
                                    array = array.offset(2);
                                }
                            }
                        }
                    }
                    Ok(self)
                }
                
                fn to_real(self) -> VecResult<Self>
                {
                    assert_complex!(self);
                    self.pure_complex_to_real_operation(|x,_arg|x.re, (), Complexity::Small)
                }
            
                fn to_imag(self) -> VecResult<Self>
                {
                    assert_complex!(self);
                    self.pure_complex_to_real_operation(|x,_arg|x.im, (), Complexity::Small)
                }	
                        
                fn get_real(&self, destination: &mut Self) -> VoidResult
                {
                    if !self.is_complex {
                        return Err(ErrorReason::VectorMustBeComplex);
                    }
                    
                    self.pure_complex_into_real_target_operation(destination, |x,_arg|x.re, (), Complexity::Small)
                }
                
                fn get_imag(&self, destination: &mut Self) -> VoidResult
                {
                    if !self.is_complex {
                        return Err(ErrorReason::VectorMustBeComplex);
                    }
                    
                    self.pure_complex_into_real_target_operation(destination, |x,_arg|x.im, (), Complexity::Small)
                }
                
                fn phase(self) -> VecResult<Self>
                {
                    assert_complex!(self);
                    self.pure_complex_to_real_operation(|x,_arg|x.arg(), (), Complexity::Small)
                }
                
                fn get_phase(&self, destination: &mut Self) -> VoidResult
                {
                    if !self.is_complex {
                        return Err(ErrorReason::VectorMustBeComplex);
                    }
                    
                    self.pure_complex_into_real_target_operation(destination, |x,_arg|x.arg(), (), Complexity::Small)
                }
                
                fn complex_dot_product(&self, factor: &Self) -> ScalarResult<Complex<$data_type>>
                {
                    if !self.is_complex {
                        return Err(ErrorReason::VectorMustBeComplex);
                    }
                    
                    if !factor.is_complex ||
                        self.domain != factor.domain {
                        return Err(ErrorReason::VectorMetaDataMustAgree);
                    }
                    
                    let data_length = self.len();
                    let scalar_length = data_length % $reg::len();
                    let vectorization_length = data_length - scalar_length;
                    let array = &self.data;
                    let other = &factor.data;
                    let chunks = Chunk::get_a_fold_b(
                        Complexity::Small, &self.multicore_settings,
                        &other, vectorization_length, $reg::len(), 
                        &array, vectorization_length, $reg::len(), 
                        |original, range, target| {
                            let mut i = 0;
                            let mut j = range.start;
                            let mut result = $reg::splat(0.0);
                            while i < target.len()
                            { 
                                let vector1 = $reg::load(original, j);
                                let vector2 = $reg::load(target, i);
                                result = result + (vector2.mul_complex(vector1));
                                i += $reg::len();
                                j += $reg::len();
                            }
                        
                        result.sum_complex()        
                    });
                    let mut i = vectorization_length;
                    let mut sum = Complex::<$data_type>::new(0.0, 0.0);
                    while i < data_length
                    {
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
                        &array, data_length, 2, (),
                        |array, range, _arg| {
                            let mut stat = Statistics::<Complex<$data_type>>::empty();
                            let mut i = 0;
                            let mut j = range.start / 2;
                            while i < array.len()
                            { 
                                let value = Complex::<$data_type>::new(array[i], array[i + 1]);
                                stat.add(value, j);
                                i += 2;
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
                        &array, data_length, 2, len,
                        |array, range, len| {
                            let mut results = Statistics::<Complex<$data_type>>::empty_vec(len);
                            let mut i = 0;
                            let mut j = range.start / 2;
                            while i < array.len()
                            { 
                                let stat = &mut results[(i / 2) % len];
                                let value = Complex::<$data_type>::new(array[i], array[i + 1]);
                                stat.add(value, j / len);
                                i += 2;
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
                
                fn set_real_imag(mut self, real: &Self::RealPartner, imag: &Self::RealPartner) -> VecResult<Self> {
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
                
                fn set_mag_phase(mut self, mag: &Self::RealPartner, phase: &Self::RealPartner) -> VecResult<Self> {
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
            }
            
            impl GenericDataVector<$data_type> {
                fn pure_complex_into_real_target_operation<A, F>(&self, destination: &mut Self, op: F, argument: A, complexity: Complexity) -> VoidResult 
                    where A: Sync + Copy,
                          F: Fn(Complex<$data_type>, A) -> $data_type + 'static + Sync {
                    let len = self.len();
                    destination.reallocate(len / 2);
                    destination.delta = self.delta;
                    destination.is_complex = false;
                    let mut array = &mut destination.data;
                    let source = &self.data;
                    Chunk::from_src_to_dest(
                        complexity, &self.multicore_settings,
                        &source, len, 2, 
                        &mut array, len / 2, 1, argument,
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
            
            impl Scale<Complex<$data_type>> for GenericDataVector<$data_type> {
                fn scale(self, offset: Complex<$data_type>) -> VecResult<Self> {
                    self.complex_scale(offset)
                }
            }
            
            impl Offset<Complex<$data_type>> for GenericDataVector<$data_type> {
                fn offset(self, offset: Complex<$data_type>) -> VecResult<Self> {
                    self.complex_offset(offset)
                }
            }
        )*
     }
}
add_complex_impl!(f32, Reg32; f64, Reg64);