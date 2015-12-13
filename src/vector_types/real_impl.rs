use multicore_support::{Chunk, Complexity};
use super::definitions::{
	DataVector,
    VecResult,
    ScalarResult,
    Statistics,
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
                
                fn real_dot_product(&self, factor: &Self) -> ScalarResult<$data_type>
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
                            result = result + (vector2 * vector1);
                            i += $reg::len();
                            j += $reg::len();
                        }
                        
                        result.sum_real()        
                    });
                    let mut i = vectorization_length;
                    let mut sum = 0.0;
                    while i < data_length
                    {
                        sum += array[i] * other[i];
                        i += 1;
                    }
                    
                    let chunk_sum: $data_type = chunks.iter().fold(0.0, |a, b| a + b);
                    Ok(chunk_sum + sum)
                }
                
                fn real_statistics(&self) -> Statistics<$data_type> {
                    let data_length = self.len();
                    let array = &self.data;
                    let chunks = Chunk::get_chunked_results(Complexity::Small, &array, data_length, 1, |array, range| {
                        let mut i = 0;
                        let mut sum = 0.0;
                        let mut sum_squared = 0.0;
                        let mut max = array[0];
                        let mut min = array[0];
                        let mut max_index = 0;
                        let mut min_index = 0;
                        while i < array.len()
                        { 
                            sum += array[i];
                            sum_squared += array[i] * array[i];
                            if array[i] > max {
                                max = array[i];
                                max_index = i + range.start;
                            }
                            else if array[i] < min {
                                min = array[i];
                                min_index = i + range.start;
                            }
                            
                            i += 1;
                        }
                        
                        Statistics {
                            sum: sum,
                            count: array.len(),
                            average: 0.0, 
                            min: min,
                            max: max, 
                            rms: sum_squared, // this field therefore has a different meaning inside this function
                            min_index: min_index,
                            max_index: max_index,
                        }    
                    });
                    
                    let mut sum = 0.0;
                    let mut max = chunks[0].max;
                    let mut min = chunks[0].min;
                    let mut max_index = chunks[0].max_index;
                    let mut min_index = chunks[0].min_index;
                    let mut sum_squared = 0.0;
                    for stat in chunks {
                        sum += stat.sum;
                        sum_squared += stat.rms; // We stored sum_squared in the field rms
                        if stat.max > max {
                            max = stat.max;
                            max_index = stat.max_index;
                        }
                        else if stat.min > min {
                            min = stat.min;
                            min_index = stat.min_index;
                        }
                    }
                    Statistics {
                        sum: sum,
                        count: array.len(),
                        average: sum / (array.len() as $data_type),
                        min: min,
                        max: max,
                        rms: (sum_squared / (array.len() as $data_type)).sqrt(),
                        min_index: min_index,
                        max_index: max_index,
                    }  
                }
            }
        )*
     }
}
add_real_impl!(f32, Reg32; f64, Reg64);