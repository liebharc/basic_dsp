use multicore_support::{Chunk, Complexity};
use super::definitions::{
	DataVector,
    VecResult,
    GenericVectorOperations,
	RealVectorOperations};
use super::GenericDataVector;    
use simd_extensions::{Simd, Reg32, Reg64};
use num::traits::Float;

macro_rules! add_real_impl {
    ($($data_type:ident, $reg:ident);*)
	 =>
	 {	 
        $(
            #[inline]
            impl RealVectorOperations<$data_type> for GenericDataVector<$data_type>
            {
                type ComplexPartner = Self;
                
                fn real_offset(mut self, offset: $data_type) -> VecResult<Self>
                {
                    {
                        let data_length = self.len();
                        let scalar_length = data_length % $reg::len();
                        let vectorization_length = data_length - scalar_length;           
                        let mut array = &mut self.data;
                        Chunk::execute_partial_with_arguments(Complexity::Small, &mut array, vectorization_length, $reg::len(), offset, |array, value| {
                            let mut i = 0;
                            while i < array.len()
                            { 
                                let vector = $reg::load(array, i);
                                let scaled = vector.add_real(value);
                                scaled.store(array, i);
                                i += $reg::len();
                            }
                        });
                        for i in vectorization_length..data_length
                        {
                            array[i] = array[i] + offset;
                        }
                    }
                    Ok(self)
                }
                
                fn real_scale(mut self, factor: $data_type) -> VecResult<Self>
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
                                let scaled = vector.scale_real(value);
                                scaled.store(array, i);
                                i += $reg::len();
                            }
                        });
                        for i in vectorization_length..data_length
                        {
                            array[i] = array[i] * factor;
                        }
                    }
                    Ok(self)
                }
                
                fn real_abs(mut self) -> VecResult<Self>
                {
                    {
                        let data_length = self.len();
                        let scalar_length = data_length % $reg::len();
                        let vectorization_length = data_length - scalar_length;
                        let mut array = &mut self.data;
                        Chunk::execute_partial(Complexity::Small, &mut array, vectorization_length, $reg::len(), |array| {
                            let mut i = 0;
                            while i < array.len()
                            { 
                                let vector = $reg::load(array, i);
                                let abs = (vector * vector).sqrt();
                                abs.store(array, i);
                                i += $reg::len();
                            }
                        });
                        for i in vectorization_length..data_length
                        {
                            array[i] = array[i].abs();
                        }
                    }
                    Ok(self)
                }
                
                fn to_complex(self) -> VecResult<Self>
                {
                    let result = self.zero_interleave();
                    match result {
                        Ok(mut vec) => { 
                            vec.is_complex = true;
                            Ok(vec)
                        },
                        Err((r, mut vec)) => {
                            vec.is_complex = true;
                            Err((r, vec))
                        }
                    }
                }
                
                fn wrap(mut self, divisor: $data_type) -> VecResult<Self>
                {
                    {
                        let mut array = &mut self.data;
                        let length = array.len();
                        Chunk::execute_partial_with_arguments(Complexity::Small, &mut array, length, 1, divisor, |array, value| {
                            let mut i = 0;
                            while i < array.len() {
                                array[i] = array[i] % value;
                                i += 1;
                            }
                        });
                    }
                    Ok(self)
                }
                
                fn unwrap(mut self, divisor: $data_type) -> VecResult<Self>
                {
                    {
                        let data_length = self.len();
                        let mut data = &mut self.data;
                        let mut i = 0;
                        let mut j = 1;
                        let half = divisor / 2.0;
                        while j < data_length {
                            let mut diff = data[j] - data[i];
                            if diff > half {
                                diff = diff % divisor;
                                diff -= divisor;
                                data[j] = data[i] + diff;
                            }
                            else if diff < -half {
                                diff = diff % divisor;
                                diff += divisor;
                                data[j] = data[i] + diff;
                            }
                                            
                            i += 1;
                            j += 1;
                        }
                    }
                    Ok(self)
                }
            }
        )*
     }
}
add_real_impl!(f32, Reg32; f64, Reg64);