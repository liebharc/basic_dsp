#![feature(test)]
#![feature(box_syntax)] 
extern crate test;
extern crate basic_dsp;
extern crate num;

pub mod tools;

#[cfg(test)]
mod bench {
    use test::Bencher;
    use basic_dsp::{
        DataVec,
        DataVecDomain,
        GenericVectorOps,
        RealVectorOps,
        DataVec32,
        RealTimeVector32,
        RealTimeVector64};
    use basic_dsp::combined_ops::*;
    use tools::{VectorBox, DEFAULT_DATA_SIZE, Size};
    
    #[inline(never)]
    pub fn add_offset_reference32(mut array: Vec<f32>, offset: f32) -> Vec<f32>
    {
        let mut i = 0;
        while i < array.len()
        {
            array[i] = array[i] + offset;
            i += 1;
        }
        
        array
    }
    
    #[inline(never)]
    pub fn add_offset_reference64(mut array: Vec<f64>, offset: f64) -> Vec<f64>
    {
        let mut i = 0;
        while i < array.len()
        {
            array[i] = array[i] + offset;
            i += 1;
        }
        
        array
    }
    
    #[bench]
    fn real_offset_32s_reference(b: &mut Bencher)
    {
        let mut vector = VectorBox::<Vec<f32>>::new(Size::Small);
        b.iter(|| {
            vector.execute(|v|  { add_offset_reference32(v, 100.0) } )
            });
    }
    
    #[bench]
    fn real_offset_64s_reference(b: &mut Bencher)
    {
        let mut vector = VectorBox::<Vec<f64>>::new(Size::Small);
        b.iter(|| {
            vector.execute(|v|  { add_offset_reference64(v, 100.0) } )
            })
    }
    
    #[inline(never)]
    pub fn real_logn_reference32(mut array: Vec<f32>) -> Vec<f32>
    {
        let mut i = 0;
        while i < array.len()
        {
            array[i] = array[i].ln();
            i += 1;
        }
        
        array
    }
    
    #[bench]
    fn vector_creation_32_benchmark(b: &mut Bencher)
    {
        b.iter(|| {
            let data = vec![0.0; DEFAULT_DATA_SIZE];
            let result = DataVec32::from_array_no_copy(false, DataVecDomain::Time, data);
            return result.delta();;
            });
    }
       
    #[bench]
    fn multi_operations_2ops1_vector_32_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<DataVec32>::new(Size::Small, false);
        b.iter(|| {
            vector.execute(|v|  
                {
                    let mut ops = multi_ops1(v);
                    ops.add_enum_op(Operation::Log(0, 10.0));
                    ops.add_enum_op(Operation::MultiplyReal(0, 10.0));
                    ops.get().unwrap()
                })
        });
    }
    
    #[bench]
    fn multi_operations_2ops1_vector_32_reference(b: &mut Bencher)
    {
        let mut vector = VectorBox::<DataVec32>::new(Size::Small, true);
        b.iter(|| {
            vector.execute_res(|v|  { 
                v.log(10.0)
                    .and_then(|v| v.real_scale(10.0)) 
            } )
        });
    }
    
    #[bench]
    fn multi_operations_3ops1_vector_32_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<DataVec32>::new(Size::Small, false);
        b.iter(|| {
            vector.execute(|v|  
                {
                    let mut ops = multi_ops1(v);
                    ops.add_enum_op(Operation::Log(0, 10.0));
                    ops.add_enum_op(Operation::MultiplyReal(0, 10.0));
                    ops.add_enum_op(Operation::Sqrt(0));
                    ops.get().unwrap()
                })
        });
    }
    
    #[bench]
    fn multi_operations_3ops1_vector_32_reference(b: &mut Bencher)
    {
        let mut vector = VectorBox::<DataVec32>::new(Size::Small, true);
        b.iter(|| {
            vector.execute_res(|v|  { 
                v.log(10.0)
                    .and_then(|v| v.real_scale(10.0)) 
                    .and_then(|v| v.sqrt()) 
            } )
        });
    }
    
    #[bench]
    fn multi_operations_3ops2_vector_32_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<DataVec32>::new(Size::Small, false);
        b.iter(|| {
            vector.execute(|v|  
                {
                    let len = v.len(); 
                    let operand = DataVec32::new(false, DataVecDomain::Time, 2.0 * 3.0, len, 1.0);
                    let ops = multi_ops2(v, operand);
                    let ops = ops.add_ops(|v, o| {
                        let v = v.abs()
                         .mul(&o)
                         .sin();
                        (v, o)
                    });
                    let (v, _) = ops.get().unwrap();
                    v
                })
        });
    }
    
    #[bench]
    fn multi_operations_6ops1_vector_32_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<DataVec32>::new(Size::Small, false);
        b.iter(|| {
            vector.execute(|v|  
                {
                    let ops = multi_ops1(v);
                    let ops = ops.add_ops(|v| {
                        v.abs()
                         .real_scale(2.0 * 3.0)
                         .sin()
                         .real_scale(10.0)
                         .abs()
                         .sqrt()
                    });
                    ops.get().unwrap()
                })
        });
    }
    
    #[bench]
    fn multi_operations_6ops2_vector_32_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<DataVec32>::new(Size::Small, false);
        b.iter(|| {
            vector.execute(|v|  
                {
                    let len = v.len(); 
                    let operand = DataVec32::new(false, DataVecDomain::Time, 2.0 * 3.0, len, 1.0);
                    let ops = multi_ops2(v, operand);
                    let ops = ops.add_ops(|v, o| {
                        let v = v.abs()
                         .mul(&o)
                         .sin()
                         .real_scale(10.0)
                         .abs()
                         .sqrt();
                        (v, o)
                    });
                    let (v, _) = ops.get().unwrap();
                    v
                })
        });
    }
    
    #[bench]
    fn real_offset_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVector32>::new(Size::Small);
        b.iter(|| {
            vector.execute_res(|v|  { v.real_offset(100.0) } )
        });
    }
    
    #[bench]
    fn real_offset_32m_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVector32>::new(Size::Medium);
        b.iter(|| {
            vector.execute_res(|v|  { v.real_offset(100.0) } )
        });
    }
    
    #[bench]
    fn real_offset_32l_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVector32>::new(Size::Large);
        b.iter(|| {
            vector.execute_res(|v|  { v.real_offset(100.0) } )
        });
    }
    
    #[bench]
    fn real_offset_64s_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVector64>::new(Size::Small);
        b.iter(|| {
            vector.execute_res(|v|  { v.real_offset(100.0) } )
        });
    }
    
    #[bench]
    fn real_scale_32m_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVector32>::new(Size::Medium);
        b.iter(|| {
            vector.execute_res(|v|  { v.real_scale(2.0) } )
        });
    }
    
    #[bench]
    fn real_scale_with_mapping_32m_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVector32>::new(Size::Medium);
        b.iter(|| {
            vector.execute_res(|v|  { v.map_inplace_real((), |v,_,_| 2.0 * v) } )
        });
    }
    
    #[bench]
    fn real_scale_with_multi_ops_mapping_32m_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVector32>::new(Size::Medium);
        b.iter(|| {
            vector.execute_res(|vec|  
                { 
                    let ops = multi_ops1(vec);
                    let ops = ops.add_ops(|vec| vec.map_inplace_real(|v, _| 2.0 * v));
                    ops.get()
                } )
        });
    }
    
    #[bench]
    fn real_abs_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVector32>::new(Size::Small);
        b.iter(|| {
            vector.execute_res(|v|  { v.abs() } )
        });
    }
    
    #[bench]
    fn real_square_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVector32>::new(Size::Small);
        b.iter(|| {
            vector.execute_res(|v|  { v.square() } )
        });
    }
    
    #[bench]
    fn real_root_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVector32>::new(Size::Small);
        b.iter(|| {
            vector.execute_res(|v|  { v.root(3.0) } )
        });
    }
    
    #[bench]
    fn real_power_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVector32>::new(Size::Small);
        b.iter(|| {
            vector.execute_res(|v|  { v.powf(3.0) } )
        });
    }
    
    #[bench]
    fn real_logn_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVector32>::new(Size::Small);
        b.iter(|| {
            vector.execute_res(|v|  { v.ln() } )
        });
    }
    
    #[bench]
    fn real_logn_32s_reference(b: &mut Bencher)
    {
        let mut vector = VectorBox::<Vec<f32>>::new(Size::Small);
        b.iter(|| {
            vector.execute(|v|  { real_logn_reference32(v) } )
            });
    }
    
    #[bench]
    fn real_logn_32m_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVector32>::new(Size::Medium);
        b.iter(|| {
            vector.execute_res(|v|  { v.ln() } )
        });
    }
    
    #[bench]
    fn real_logn_32l_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVector32>::new(Size::Large);
        b.iter(|| {
            vector.execute_res(|v|  { v.ln() } )
        });
    }
    
    #[bench]
    fn real_expn_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVector32>::new(Size::Small);
        b.iter(|| {
            vector.execute_res(|v|  { v.exp() } )
        });
    }
    
    #[bench]
    fn wrap_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVector32>::new(Size::Small);
        b.iter(|| {
            vector.execute_res(|v|  { v.wrap(10.0) } )
        });
    }
    
    #[bench]
    fn unwrap_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVector32>::new(Size::Small);
        b.iter(|| {
            vector.execute_res(|v|  { v.unwrap(10.0) } )
        });
    }
    
    #[bench]
    fn diff_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVector32>::new(Size::Small);
        b.iter(|| {
            vector.execute_res(|v|  { v.diff_with_start() } )
        });
    }
    
    #[bench]
    fn cum_sum_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVector32>::new(Size::Small);
        b.iter(|| {
            vector.execute_res(|v|  { v.cum_sum() } )
        });
    }
    
    #[bench]
    fn real_vector_multiplication_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVector32>::new(Size::Small);
        b.iter(|| {
            vector.execute_res(|v| {
                let len = v.len(); 
                let operand = RealTimeVector32::from_constant(0.0, len);
                v.mul(&operand) 
            } )
        });
    }
    
    #[bench]
    fn real_sin_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVector32>::new(Size::Small);
        b.iter(|| {
            vector.execute_res(|v|  { v.sin() } )
        });
    }
    
    #[bench]
    fn real_sin_32m_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVector32>::new(Size::Medium);
        b.iter(|| {
            vector.execute_res(|v|  { v.sin() } )
        });
    }
    
    #[bench]
    fn real_sin_64s_benchmark(b: &mut Bencher)
    {
        let mut vector = VectorBox::<RealTimeVector64>::new(Size::Small);
        b.iter(|| {
            vector.execute_res(|v|  { v.sin() } )
        });
    }
}