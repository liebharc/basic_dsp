use multicore_support::{Chunk, Complexity};
use super::definitions::{
    DataVector,
    VecResult,
    ErrorReason,
    ScalarResult,
    Statistics,
    ScaleOps,
    OffsetOps,
    DotProductOps,
    StatisticsOps,
    GenericVectorOps,
    RealVectorOps,
    VectorIter};
use super::GenericDataVector;    
use super::stats_impl::Stats;
use simd_extensions::*;
use std::sync::Arc;

macro_rules! add_real_impl {
    ($($data_type:ident, $reg:ident);*)
     =>
     {     
        $(
            impl RealVectorOps<$data_type> for GenericDataVector<$data_type>
            {
                type ComplexPartner = Self;
                
                fn real_offset(self, offset: $data_type) -> VecResult<Self>
                {
                    assert_real!(self);
                    self.simd_real_operation(|x, y| x.add_real(y), |x, y| x + y, offset, Complexity::Small)
                }
                
                fn real_scale(self, factor: $data_type) -> VecResult<Self>
                {
                    // This operation actually also works for complex vectors so we allow that
                    self.simd_real_operation(|x, y| x.scale_real(y), |x, y| x * y, factor, Complexity::Small)
                }
                
                fn abs(self) -> VecResult<Self>
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
                
                fn map_inplace_real<A, F>(mut self, argument: A, f: F) -> VecResult<Self>
                    where A: Sync + Copy + Send,
                          F: Fn($data_type, usize, A) -> $data_type + 'static + Sync {
                    {
                        assert_real!(self);
                        let mut array = &mut self.data;
                        let length = array.len();
                        Chunk::execute_with_range(
                            Complexity::Small, &self.multicore_settings,
                            &mut array[0..length], 1, argument,
                            move|array, range, argument| {
                                let mut i = range.start;
                                for num in array {
                                    *num = f(*num, i, argument);
                                    i += 1;
                                }
                            });
                    }
                    Ok(self)
                }
                
                fn map_aggregate_real<A, FMap, FAggr, R>(
                    &self, 
                    argument: A, 
                    map: FMap,
                    aggregate: FAggr) -> ScalarResult<R>
                        where A: Sync + Copy + Send,
                              FMap: Fn($data_type, usize, A) -> R + 'static + Sync,
                              FAggr: Fn(R, R) -> R + 'static + Sync + Send,
                              R: Send {
                    
                    let aggregate = Arc::new(aggregate);
                    let mut result = {
                        if self.is_complex {
                            return Err(ErrorReason::VectorMustBeReal);
                        }
                        
                        let array = &self.data;
                        let length = array.len();
                        if length == 0 {
                            return Err(ErrorReason::VectorMustNotBeEmpty);
                        }
                        let aggregate  = aggregate.clone();
                        Chunk::map_on_array_chunks(
                            Complexity::Small, &self.multicore_settings,
                            &array[0..length], 1, argument,
                            move|array, range, argument| {
                                let aggregate  = aggregate.clone();
                                let mut i = range.start;
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
                        return Err(ErrorReason::VectorMustNotBeEmpty);
                    }
                    let mut aggregated = only_valid_options.pop().unwrap();
                    for _ in 0..only_valid_options.len() {
                        aggregated = aggregate(aggregated, only_valid_options.pop().unwrap());
                    }
                    Ok(aggregated)
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
                        &other[0..vectorization_length], $reg::len(), 
                        &array[0..vectorization_length], $reg::len(), 
                        |original, range, target| {
                            let mut i = 0;
                            let mut j = range.start;
                            let mut result = $reg::splat(0.0);
                            while i < target.len()
                            { 
                                let vector1 = $reg::load_unchecked(original, j);
                                let vector2 = $reg::load_unchecked(target, i);
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
                        &array[0..data_length], 1, (),
                        |array, range, _arg| {
                            let mut stats = Statistics::empty();
                            let mut j = range.start;
                            for num in array { 
                                stats.add(*num, j);
                                j += 1;
                            }
                            stats
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
                        &array[0..data_length], 1, len,
                        |array, range, len| {
                            let mut results = Statistics::empty_vec(len);
                            let mut j = range.start;
                            for num in array {
                                let stats = &mut results[j % len];
                                stats.add(*num, j / len);
                                j += 1;
                            }
                            
                            results 
                    });
                    
                    Statistics::merge_cols(&chunks)
                }
                
                fn real_sum(&self) -> $data_type {
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
                                .map(|v|v.sum_real())
                                .sum()
                        } 
                        else {
                            0.0
                        };
                    for num in &array[0..scalar_left]
                    {
                        sum = sum + *num;
                    }
                    for num in &array[scalar_right..data_length]
                    {
                        sum = sum + *num;
                    }
                    sum
                }
                
                fn real_sum_sq(&self) -> $data_type {
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
                                    sum = sum + *reg * *reg;
                                }
                                sum
                            });
                            chunks.iter()
                                .map(|v|v.sum_real())
                                .sum()
                        } 
                        else {
                            0.0
                        };
                    for num in &array[0..scalar_left]
                    {
                        sum = sum + *num * *num;
                    }
                    for num in &array[scalar_right..data_length]
                    {
                        sum = sum + *num * *num;
                    }
                    sum
                }
            }
            
            impl ScaleOps<$data_type> for GenericDataVector<$data_type> {
                fn scale(self, offset: $data_type) -> VecResult<Self> {
                    self.real_scale(offset)
                }
            }
            
            impl OffsetOps<$data_type> for GenericDataVector<$data_type> {
                fn offset(self, offset: $data_type) -> VecResult<Self> {
                    self.real_offset(offset)
                }
            }
            
            impl DotProductOps<$data_type> for GenericDataVector<$data_type> {
                fn dot_product(&self, factor: &Self) -> ScalarResult<$data_type> {
                    self.real_dot_product(factor)
                }
            }
            
            impl StatisticsOps<$data_type> for GenericDataVector<$data_type> {
                fn statistics(&self) -> Statistics<$data_type> {
                    self.real_statistics()
                }
                
                fn statistics_splitted(&self, len: usize) -> Vec<Statistics<$data_type>> {
                    self.real_statistics_splitted(len)
                }
                
                fn sum(&self) -> $data_type {
                    self.real_sum()
                }
                
                fn sum_sq(&self) -> $data_type {
                    self.real_sum_sq()
                }
            }
            
            impl VectorIter<$data_type> for GenericDataVector<$data_type> {
                fn map_inplace<A, F>(self, argument: A, map: F) -> VecResult<Self>
                    where A: Sync + Copy + Send,
                          F: Fn($data_type, usize, A) -> $data_type + 'static + Sync {
                    self.map_inplace_real(argument, map)
                }
                
                fn map_aggregate<A, FMap, FAggr, R>(
                    &self, 
                    argument: A, 
                    map: FMap,
                    aggregate: FAggr) -> ScalarResult<R>
                where A: Sync + Copy + Send,
                      FMap: Fn($data_type, usize, A) -> R + 'static + Sync,
                      FAggr: Fn(R, R) -> R + 'static + Sync + Send,
                      R: Send {
                    self.map_aggregate_real(argument, map, aggregate)  
                }
            }
        )*
     }
}
add_real_impl!(f32, Reg32; f64, Reg64);