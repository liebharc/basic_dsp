#![feature(test)]
#![feature(box_syntax)]
extern crate test;
extern crate basic_dsp;
extern crate num;

pub mod tools;

#[cfg(test)]
mod real {
    use test::Bencher;
    use basic_dsp::vector_types2::*;
    use basic_dsp::vector_types2::combined_ops::*;
    use tools::*;

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
        let mut vector = VectorBox::<Vec<f32>, f32>::new(Size::Small);
        b.iter(|| {
            vector.execute(|v, _|  { add_offset_reference32(v, 100.0) } )
            });
    }

    #[bench]
    fn real_offset_64s_reference(b: &mut Bencher)
    {
        let mut vector = VectorBox::<Vec<f64>, f64>::new(Size::Small);
        b.iter(|| {
            vector.execute(|v, _|  { add_offset_reference64(v, 100.0) } )
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
            data.to_real_time_vec();
            });
    }

    #[bench]
    fn multi_operations_2ops1_vector_32_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|v, mut buffer|
                {
                    let ops = multi_ops1(v);
                    let ops = ops.add_ops(|mut x| {
                        x.log(10.0);
                        x.scale(10.0);
                        x
                    });
                    ops.get(buffer).unwrap()
                })
        });
    }

    #[bench]
    fn multi_operations_2ops1_vector_32_reference(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _|  {
                v.log(10.0);
                v.scale(10.0);
                v
            } )
        });
    }

    #[bench]
    fn multi_operations_3ops1_vector_32_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|v, mut buffer|
                {
                    let ops = multi_ops1(v);
                    let ops = ops.add_ops(|mut x| {
                        x.log(10.0);
                        x.scale(10.0);
                        x.sqrt();
                        x
                    });
                    ops.get(buffer).unwrap()
                })
        });
    }

    #[bench]
    fn multi_operations_3ops1_vector_32_reference(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _|  {
                v.log(10.0);
                v.scale(10.0);
                v.sqrt();
                v
            } )
        });
    }

    #[bench]
    fn multi_operations_3ops2_vector_32_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|v, mut buffer|
                {
                    let len = v.len();
                    let operand = vec!(6.0; len).to_real_time_vec();
                    let ops = multi_ops2(v, operand);
                    let ops = ops.add_ops(|mut v, o| {
                        v.abs();
                        v.mul(&o).unwrap();
                        v.sin();
                        (v, o)
                    });
                    let (v, _) = ops.get(buffer).unwrap();
                    v
                })
        });
    }

    #[bench]
    fn multi_operations_6ops1_vector_32_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|v, mut buffer|
                {
                    let ops = multi_ops1(v);
                    let ops = ops.add_ops(|mut x| {
                        x.abs();
                        x.scale(6.0);
                        x.sin();
                        x.scale(10.0);
                        x.abs();
                        x.sqrt();
                        x
                    });
                    ops.get(buffer).unwrap()
                })
        });
    }

    #[bench]
    fn multi_operations_6ops2_vector_32_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|v, mut buffer|
                {
                    let len = v.len();
                    let operand = vec!(6.0; len).to_real_time_vec();
                    let ops = multi_ops2(v, operand);
                    let ops = ops.add_ops(|mut v, o| {
                        v.abs();
                        v.mul(&o).unwrap();
                        v.sin();
                        v.scale(10.0);
                        v.abs();
                        v.sqrt();
                        (v, o)
                    });
                    let (v, _) = ops.get(buffer).unwrap();
                    v
                })
        });
    }

    #[bench]
    fn real_offset_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _|
                {
                    v.offset(100.0);
                    v
                })
        });
    }

    #[bench]
    fn real_offset_32m_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Medium);
        b.iter(|| {
            vector.execute(|mut v, _|
                {
                    v.offset(100.0);
                    v
                })
        });
    }

    #[bench]
    fn real_offset_32l_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Large);
        b.iter(|| {
            vector.execute(|mut v, _|
                {
                    v.offset(100.0);
                    v
                })
        });
    }

    #[bench]
    fn real_offset_64s_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime64Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _|
                {
                    v.offset(100.0);
                    v
                })
        });
    }

    #[bench]
    fn real_scale_32m_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Medium);
        b.iter(|| {
            vector.execute(|mut v, _|
                {
                    v.scale(10.0);
                    v
                })
        });
    }

    #[bench]
    fn real_scale_with_mapping_32m_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Medium);
        b.iter(|| {
            vector.execute(|mut v, _|
                {
                    v.map_inplace((), |v,_,_| 2.0 * v);
                    v
                })
        });
    }

    #[bench]
    fn real_scale_with_multi_ops_mapping_32m_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Medium);
        b.iter(|| {
            vector.execute(|v, mut buffer|
                {
                    let len = v.len();
                    let operand = vec!(6.0; len).to_real_time_vec();
                    let ops = multi_ops2(v, operand);
                    let ops = ops.add_ops(|mut v, o| {
                        v.map_inplace(|v, _| 2.0 * v);
                        (v, o)
                    });
                    let (v, _) = ops.get(buffer).unwrap();
                    v
                })
        });
    }

    #[bench]
    fn real_abs_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _|
                {
                    v.abs();
                    v
                })
        });
    }

    #[bench]
    fn real_square_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _|
                {
                    v.square();
                    v
                })
        });
    }

    #[bench]
    fn real_root_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _|
                {
                    v.root(3.0);
                    v
                })
        });
    }

    #[bench]
    fn real_power_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _|
                {
                    v.powf(3.0);
                    v
                })
        });
    }

    #[bench]
    fn real_logn_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _|
                {
                    v.ln();
                    v
                })
        });
    }

    #[bench]
    fn real_logn_32s_reference(b: &mut Bencher)
    {
        let mut vector = VectorBox::<Vec<f32>, f32>::new(Size::Small);
        b.iter(|| {
            vector.execute(|v, _|  { real_logn_reference32(v) } )
            });
    }

    #[bench]
    fn real_logn_32m_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Medium);
        b.iter(|| {
            vector.execute(|mut v, _|
                {
                    v.ln();
                    v
                })
        });
    }

    #[bench]
    fn real_logn_32l_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Large);
        b.iter(|| {
            vector.execute(|mut v, _|
                {
                    v.ln();
                    v
                })
        });
    }

    #[bench]
    fn real_expn_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _|
                {
                    v.exp();
                    v
                })
        });
    }

    #[bench]
    fn wrap_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _|
                {
                    v.wrap(10.0);
                    v
                })
        });
    }

    #[bench]
    fn unwrap_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _|
                {
                    v.unwrap(10.0);
                    v
                })
        });
    }

    #[bench]
    fn diff_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _|
                {
                    v.diff_with_start();
                    v
                })
        });
    }

    #[bench]
    fn cum_sum_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _|
                {
                    v.cum_sum();
                    v
                })
        });
    }

    #[bench]
    fn real_vector_multiplication_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _|
                {
                    let len = v.len();
                    let operand = vec!(0.0; len).to_real_time_vec();
                    v.mul(&operand).unwrap();
                    v
                })
        });
    }

    #[bench]
    fn real_sin_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _|
                {
                    v.sin();
                    v
                })
        });
    }

    #[bench]
    fn real_sin_32m_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Medium);
        b.iter(|| {
            vector.execute(|mut v, _|
                {
                    v.sin();
                    v
                })
        });
    }

    #[bench]
    fn real_sin_64s_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime64Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _|
                {
                    v.sin();
                    v
                })
        });
    }

    #[bench]
    fn swap_halves_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, b|
                {
                    v.swap_halves_b(b);
                    v
                })
        });
    }

    #[bench]
    fn reverse_32s_benchmark(b: &mut Bencher)
    {
        let mut vector = RealTime32Box::new(Size::Small);
        b.iter(|| {
            vector.execute(|mut v, _|
                {
                    v.reverse();
                    v
                })
        });
    }
}
