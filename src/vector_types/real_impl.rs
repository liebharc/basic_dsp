use multicore_support::{Chunk, Complexity};
use super::definitions::{
	DataVector,
    VecResult,
    ErrorReason,
    ScalarResult,
    Statistics,
    GenericVectorOperations,
	RealVectorOperations};
use super::GenericDataVector;    
use super::stats_impl::Stats;
use simd_extensions::{Simd, Reg32, Reg64};
use num::traits::Float;

macro_rules! add_real_impl {
    ($($data_type:ident, $reg:ident);*)
	 =>
	 {	 
        $(
            impl RealVectorOperations<$data_type> for GenericDataVector<$data_type>
            {
                type ComplexPartner = Self;
                
                fn real_offset(self, offset: $data_type) -> VecResult<Self>
                {
                    assert_real!(self);
                    self.simd_real_operation(|x, y| x.add_real(y), |x, y| x + y, offset, Complexity::Small)
                }
                
                fn real_scale(self, factor: $data_type) -> VecResult<Self>
                {
                    assert_real!(self);
                    self.simd_real_operation(|x, y| x.scale_real(y), |x, y| x * y, factor, Complexity::Small)
                }
                
                fn real_abs(self) -> VecResult<Self>
                {
                    assert_real!(self);
                    self.simd_real_operation(|x, _arg| (x * x).sqrt(), |x, _arg| x.abs(), (), Complexity::Small)
                }
                
                fn to_complex(self) -> VecResult<Self>
                {
                    assert_real!(self);
                    let result = self.zero_interleave(2);
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
                
                fn wrap(self, divisor: $data_type) -> VecResult<Self>
                {
                    assert_real!(self);
                    self.pure_real_operation(|x, y| x % y, divisor, Complexity::Small)
                }
                
                fn unwrap(mut self, divisor: $data_type) -> VecResult<Self>
                {
                    {
                        assert_real!(self);
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
                    if self.is_complex {
                        return Err(ErrorReason::VectorMustBeReal);
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
                    let chunks = Chunk::get_chunked_results(
                        Complexity::Small, &self.multicore_settings,
                        &array, data_length, 1, (),
                        |array, range, _arg| {
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
                    
                    Statistics::merge(&chunks)
                }
                
                fn real_statistics_splitted(&self, len: usize) -> Vec<Statistics<$data_type>> {
                    if len == 0 {
                        return Vec::new();
                    }
                    
                    let data_length = self.len();
                    let array = &self.data;
                    let chunks = Chunk::get_chunked_results(
                        Complexity::Small, &self.multicore_settings,
                        &array, data_length, 1, len,
                        |array, range, len| {
                            let mut results = Vec::with_capacity(len);
                            for _ in 0..len {
                                let stats = Statistics::<$data_type>::empty();
                                results.push(stats);
                            }
                            
                            let mut i = 0;
                            while i < array.len() {
                                let stats = &mut results[i % len];
                                stats.sum += array[i];
                                stats.count += 1;
                                stats.rms += array[i] * array[i];
                                if array[i] > stats.max {
                                    stats.max = array[i];
                                    stats.max_index = (i + range.start) / len;
                                }
                                else if array[i] < stats.min {
                                    stats.min = array[i];
                                    stats.min_index = (i + range.start) / len;
                                }
                                
                                i += 1;
                            }
                            
                            results 
                    });
                    
                    let mut results = Vec::with_capacity(len);
                    for i in 0..len {
                        let mut reordered = Vec::with_capacity(chunks.len());
                        for j in 0..chunks.len()
                        {
                            reordered.push(chunks[j][i]);
                        }
                        
                        let stats = Statistics::merge(&reordered);
                        results.push(stats);
                    }
                    
                    results
                }
            }
        )*
     }
}
add_real_impl!(f32, Reg32; f64, Reg64);