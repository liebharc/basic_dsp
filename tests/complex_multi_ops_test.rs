extern crate basic_dsp;
extern crate num;
extern crate rand;
pub mod tools;

mod complex_test {
    use basic_dsp::combined_ops::*;
    use basic_dsp::*;
    use num::complex::*;
    use tools::*;

    #[test]
    fn multi_ops_offset_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, 2);
            let a = a.to_complex_time_vec();
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.offset(Complex32::new(b[0], b[1]));
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            let mut a_expected = a;
            a_expected.offset(Complex32::new(b[0], b[1]));
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_scale_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, 2);
            let a = a.to_complex_time_vec();
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.scale(Complex32::new(b[0], b[1]));
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            let mut a_expected = a;
            a_expected.scale(Complex32::new(b[0], b[1]));
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_magnitude_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = a.to_complex_time_vec();
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| a.magnitude());
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            let a_expected = a.magnitude();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_magnitude_squared_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = a.to_complex_time_vec();
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| a.magnitude_squared());
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            let a_expected = a.magnitude_squared();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_conj_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = a.to_complex_time_vec();
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.conj();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            let mut a_expected = a;
            a_expected.conj();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_to_real_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = a.to_complex_time_vec();
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| a.to_real());
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            let a_expected = a.to_real();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_to_imag_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = a.to_complex_time_vec();
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| a.to_imag());
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            let a_expected = a.to_imag();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_phase_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let a = a.to_complex_time_vec();
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|a| a.phase());
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            let a_expected = a.phase();
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_add_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let b = b.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops2(a.clone(), b.clone());
            let ops = ops.add_ops(|mut a, b| {
                a.add(&b).unwrap();
                (a, b)
            });
            let mut buffer = SingleBuffer::new();
            let (a_actual, b_actual) = ops.get(&mut buffer).unwrap();
            a.add(&b).unwrap();
            let a_expected = a;
            let b_expected = b;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
            assert_vector_eq(&b_expected[..], &b_actual[..]);
        });
    }

    #[test]
    fn multi_ops_sub_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let b = b.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops2(a.clone(), b.clone());
            let ops = ops.add_ops(|mut a, b| {
                a.sub(&b).unwrap();
                (a, b)
            });
            let mut buffer = SingleBuffer::new();
            let (a_actual, b_actual) = ops.get(&mut buffer).unwrap();
            a.sub(&b).unwrap();
            let a_expected = a;
            let b_expected = b;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
            assert_vector_eq(&b_expected[..], &b_actual[..]);
        });
    }

    #[test]
    fn multi_ops_mul_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let b = b.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops2(a.clone(), b.clone());
            let ops = ops.add_ops(|mut a, b| {
                a.mul(&b).unwrap();
                (a, b)
            });
            let mut buffer = SingleBuffer::new();
            let (a_actual, b_actual) = ops.get(&mut buffer).unwrap();
            a.mul(&b).unwrap();
            let a_expected = a;
            let b_expected = b;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
            assert_vector_eq(&b_expected[..], &b_actual[..]);
        });
    }

    #[test]
    fn multi_ops_div_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let b = b.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops2(a.clone(), b.clone());
            let ops = ops.add_ops(|mut a, b| {
                a.div(&b).unwrap();
                (a, b)
            });
            let mut buffer = SingleBuffer::new();
            let (a_actual, b_actual) = ops.get(&mut buffer).unwrap();
            a.div(&b).unwrap();
            let a_expected = a;
            let b_expected = b;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
            assert_vector_eq(&b_expected[..], &b_actual[..]);;
        });
    }

    #[test]
    fn multi_ops_sqrt_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511142, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.sqrt();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.sqrt();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_square_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.square();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.square();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_root_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, 1);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.root(b[0]);
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.root(b[0]);
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_powf_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, 1);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.powf(b[0]);
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.powf(b[0]);
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_ln_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.ln();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.ln();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_exp_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.exp();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.exp();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_log_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, 1);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.log(b[0]);
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.log(b[0]);
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_expf_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let b = create_data_with_len(201511141, iteration, 1);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.expf(b[0]);
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.expf(b[0]);
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_sin_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.sin();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.sin();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_cos_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.cos();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.cos();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_tan_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.tan();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.tan();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_asin_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.asin();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.asin();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_acos_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.acos();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.acos();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_atan_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.atan();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.atan();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_sinh_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.sinh();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.sinh();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_cosh_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.cosh();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.cosh();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_tanh_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.tanh();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.tanh();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_asinh_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.asinh();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.asinh();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_acosh_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.acosh();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.acosh();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_atanh_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1000;
            let a = create_data_with_len(201511141, iteration, len);
            let mut a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.atanh();
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            a.atanh();
            let a_expected = a;
            assert_vector_eq(&a_expected[..], &a_actual[..]);
        });
    }

    #[test]
    fn multi_ops_complex_exponential_test() {
        parameterized_vector_test(|iteration, _| {
            let len = 1030;
            let a = create_data_with_len(201511141, iteration, len);
            let args = create_data_with_len(201511141, iteration, 2);
            let a = a.to_gen_dsp_vec(true, DataDomain::Time);
            let ops = multi_ops1(a.clone());
            let ops = ops.add_ops(|mut a| {
                a.multiply_complex_exponential(args[0], args[1]);
                a
            });
            let mut buffer = SingleBuffer::new();
            let a_actual = ops.get(&mut buffer).unwrap();
            let mut a_expected = a;
            a_expected.multiply_complex_exponential(args[0], args[1]);
            assert_vector_eq_with_reason_and_tolerance(&a_expected[..], &a_actual[..], 1e-2, "");
        });
    }
}
