use multicore_support::{Chunk, Complexity};
use super::definitions::{
	DataVector,
    VecResult,
    VoidResult,
    ScalarResult,
	ComplexVectorOperations};
use super::GenericDataVector;
use simd_extensions::{Simd,Reg32,Reg64};
use num::complex::Complex;
use num::traits::Float;
use std::ops::Range;

macro_rules! add_complex_impl {
    ($($data_type:ident, $reg:ident);*)
	 =>
	 {	 
        $(
            #[inline]
            impl ComplexVectorOperations<$data_type> for GenericDataVector<$data_type>
            {
                type RealPartner = GenericDataVector<$data_type>;
                fn complex_offset(mut self, offset: Complex<$data_type>)  -> VecResult<Self>
                {
                    {
                        let data_length = self.len();
                        let scalar_length = data_length % $reg::len();
                        let vectorization_length = data_length - scalar_length;
                        let mut array = &mut self.data;
                        let vector_offset = $reg::from_complex(offset);
                        Chunk::execute_partial_with_arguments(Complexity::Small, &mut array, vectorization_length, $reg::len(), vector_offset, |array, v| {
                            let mut i = 0;
                            while i < array.len()
                            {
                                let data = $reg::load(array, i);
                                let result = data + v;
                                result.store(array, i);
                                i += $reg::len();
                            }
                        });
                        
                        let mut i = vectorization_length;
                        while i < data_length
                        {
                            array[i] += offset.re;
                            array[i + 1] += offset.im;
                            i += 2;
                        }
                    }
                    
                    Ok(self)
                }
                
                fn complex_scale(mut self, factor: Complex<$data_type>) -> VecResult<Self>
                {
                    {
                        let data_length = self.len();
                        let scalar_length = data_length % $reg::len();
                        let vectorization_length = data_length - scalar_length;
                        let mut array = &mut self.data;
                        Chunk::execute_partial_with_arguments(Complexity::Small, &mut array, vectorization_length, $reg::len(), factor, |array, value| {
                            let mut i = 0;
                            while i < array.len()
                            { 
                                let vector = $reg::load(array, i);
                                let scaled = vector.scale_complex(value);
                                scaled.store(array, i);
                                i += $reg::len();
                            }
                        });
                        let mut i = vectorization_length;
                        while i < data_length
                        {
                            let complex = Complex::<$data_type>::new(array[i], array[i + 1]);
                            let result = complex * factor;
                            array[i] = result.re;
                            array[i + 1] = result.im;
                            i += 2;
                        }
                    }
                    Ok(self)
                }
                
                fn complex_abs(mut self) -> VecResult<Self>
                {
                    {
                        let data_length = self.len();
                        let scalar_length = data_length % $reg::len();
                        let vectorization_length = data_length - scalar_length;
                        let array = &self.data;
                        let mut temp = &mut self.temp;
                        Chunk::execute_original_to_target(Complexity::Small, &array, vectorization_length, $reg::len(), &mut temp, vectorization_length / 2, $reg::len() / 2, Self::complex_abs_simd);
                        let mut i = vectorization_length;
                        while i + 1 < data_length
                        {
                            temp[i / 2] = (array[i] * array[i] + array[i + 1] * array[i + 1]).sqrt();
                            i += 2;
                        }
                        self.is_complex = false;
                        self.valid_len = self.valid_len / 2;
                    }
                    
                    Ok(self.swap_data_temp())
                }
                
                fn get_complex_abs(&self, destination: &mut Self) -> VoidResult
                {
                    let data_length = self.len();
                    destination.reallocate(data_length / 2);
                    let scalar_length = data_length % $reg::len();
                    let vectorization_length = data_length - scalar_length;
                    let array = &self.data;
                    let mut temp = &mut destination.data;
                    Chunk::execute_original_to_target(Complexity::Small, &array, vectorization_length, $reg::len(), &mut temp, vectorization_length / 2, $reg::len() / 2, Self::complex_abs_simd);
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
                
                fn complex_abs_squared(mut self) -> VecResult<Self>
                {
                    {
                        let data_length = self.len();
                        let scalar_length = data_length % $reg::len();
                        let vectorization_length = data_length - scalar_length;
                        let array = &mut self.data;
                        let mut temp = &mut self.temp;
                        Chunk::execute_original_to_target(Complexity::Small, &array, vectorization_length,  $reg::len(), &mut temp, vectorization_length / 2, $reg::len() / 2, |array, range, target| {
                            let mut i = range.start;
                            let mut j = 0;
                            while j < target.len()
                            { 
                                let vector = $reg::load(array, i);
                                let result = vector.complex_abs_squared();
                                result.store_half(target, j);
                                i += $reg::len();
                                j += $reg::len() / 2;
                            }
                        });
                        let mut i = vectorization_length;
                        while i + 1 < data_length
                        {
                            temp[i / 2] = array[i] * array[i] + array[i + 1] * array[i + 1];
                            i += 2;
                        }
                        self.is_complex = false;
                        self.valid_len = self.valid_len / 2;
                    }
                    
                    Ok(self.swap_data_temp())
                }
                
                fn complex_conj(mut self) -> VecResult<Self>
                {
                    {
                        let data_length = self.len();
                        let scalar_length = data_length % $reg::len();
                        let vectorization_length = data_length - scalar_length;
                        let mut array = &mut self.data;
                        Chunk::execute_partial(Complexity::Small, &mut array, $reg::len(), vectorization_length, |array| {
                            let multiplicator = $reg::from_complex(Complex::<$data_type>::new(1.0, -1.0));
                            let mut i = 0;
                            while i < array.len() {
                                let vector = $reg::load(array, i);
                                let result = vector * multiplicator;
                                result.store(array, i);
                                i += $reg::len();
                            }
                        });
                        
                        let mut i = vectorization_length;
                        while i + 2 < data_length
                        {
                            array[i] = -array[i];
                            i += 2;
                        }
                    }
                    
                    Ok(self)
                }
                
                fn to_real(mut self) -> VecResult<Self>
                {
                    {
                        let len = self.len();
                        let mut array = &mut self.temp;
                        let source = &self.data;
                        Chunk::execute_original_to_target(Complexity::Small, &source, len, 2, &mut array, len / 2, 1, |original, range, target| {
                            let mut i = range.start;
                            let mut j = 0;
                            while j < target.len()
                            { 
                                target[j] = original[i];
                                i += 2;
                                j += 1;
                            }
                        });
                    }
                    
                    self.is_complex = false;
                    self.valid_len = self.valid_len / 2;
                    Ok(self.swap_data_temp())
                }
            
                fn to_imag(mut self) -> VecResult<Self>
                {
                   {
                        let len = self.len();
                        let mut array = &mut self.temp;
                        let source = &self.data;
                        Chunk::execute_original_to_target(Complexity::Small, &source, len, 2, &mut array, len / 2, 1, |original, range, target| {
                            let mut i = range.start + 1;
                            let mut j = 0;
                            while j < target.len()
                            { 
                                target[j] = original[i];
                                i += 2;
                                j += 1;
                            }
                        });
                    }
                    
                    self.is_complex = false;
                    self.valid_len = self.valid_len / 2;
                    Ok(self.swap_data_temp())
                }	
                        
                fn get_real(&self, destination: &mut Self) -> VoidResult
                {
                    let len = self.len();
                    destination.reallocate(len / 2);
                    destination.delta = self.delta;
                    destination.is_complex = false;
                    let mut array = &mut destination.data;
                    let source = &self.data;
                    Chunk::execute_original_to_target(Complexity::Small, &source, len, 2, &mut array, len / 2, 1, |original, range, target| {
                        let mut i = range.start;
                        let mut j = 0;
                        while j < target.len()
                        { 
                            target[j] = original[i];
                            i += 2;
                            j += 1;
                        }
                    });
                    
                    Ok(())
                }
                
                fn get_imag(&self, destination: &mut Self) -> VoidResult
                {
                    let len = self.len();
                    destination.reallocate(len / 2);
                    destination.delta = self.delta;
                    destination.is_complex = false;
                    let mut array = &mut destination.data;
                    let source = &self.data;
                    Chunk::execute_original_to_target(Complexity::Small, &source, len, 2, &mut array, len / 2, 1,  |original, range, target| {
                        let mut i = range.start + 1;
                        let mut j = 0;
                        while j < target.len()
                        { 
                            target[j] = original[i];
                            i += 2;
                            j += 1;
                        }
                    });
                    
                    Ok(())
                }
                
                fn phase(mut self) -> VecResult<Self>
                {
                    {
                        let len = self.len();
                        let mut array = &mut self.temp;
                        let source = &self.data;
                        Chunk::execute_original_to_target(Complexity::Small, &source, len, 2, &mut array, len / 2, 1,  Self::phase_par);
                    }
                    
                    self.is_complex = false;
                    self.valid_len = self.valid_len / 2;
                    Ok(self.swap_data_temp())
                }
                
                fn get_phase(&self, destination: &mut Self) -> VoidResult
                {
                    let len = self.len();
                    destination.reallocate(len / 2);
                    destination.delta = self.delta;
                    destination.is_complex = false;
                    let mut array = &mut destination.data;
                    let source = &self.data;
                    Chunk::execute_original_to_target(Complexity::Small, &source, len, 2, &mut array, len / 2, 1,  Self::phase_par);
                    Ok(())
                }
                
                fn complex_dot_product(&self, factor: &Self) -> ScalarResult<Complex<$data_type>>
                {
                    let data_length = self.len();
                    let scalar_length = data_length % $reg::len();
                    let vectorization_length = data_length - scalar_length;
                    let array = &self.data;
                    let other = &factor.data;
                    let chunks = Chunk::get_a_fold_b(Complexity::Small, &other, vectorization_length, $reg::len(), &array, vectorization_length, $reg::len(), |original, range, target| {
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
            }
            
            impl GenericDataVector<$data_type> {
                fn complex_abs_simd(original: &[$data_type], range: Range<usize>, target: &mut [$data_type])
                {
                    let mut i = 0;
                    let mut j = range.start;
                    while i < target.len()
                    { 
                        let vector = $reg::load(original, j);
                        let result = vector.complex_abs();
                        result.store_half(target, i);
                        j += $reg::len();
                        i += $reg::len() / 2;
                    }
                }
                
                fn phase_par(original: &[$data_type], range: Range<usize>, target: &mut [$data_type])
                {
                    let mut i = range.start;
                    let mut j = 0;
                    while j < target.len()
                    { 
                        let complex = Complex::<$data_type>::new(original[i], original[i + 1]);
                        target[j] = complex.arg();
                        i += 2;
                        j += 1;
                    }
                }
            }
        )*
     }
}
add_complex_impl!(f32, Reg32; f64, Reg64);