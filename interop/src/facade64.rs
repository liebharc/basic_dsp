//! Auto generated code, change facade32.rs and run facade64_create.pl
//! Functions for 64bit floating point number based vectors.
//! Please refer to the other chapters of the help for documentation of the functions.
use super::*;
use basic_dsp_vector::*;
use basic_dsp_vector::meta;
use basic_dsp_vector::conv_types::*;
use num_complex::*;
use std::slice;
use std::os::raw::c_void;
use std::mem;

pub type VecBuf = InteropVec<f64>;

pub type VecBox = Box<InteropVec<f64>>;

#[no_mangle]
pub extern "C" fn delete_vector64(vector: VecBox) {
    drop(vector);
}


#[no_mangle]
pub extern "C" fn new64(is_complex: i32,
                        domain: i32,
                        init_value: f64,
                        length: usize,
                        delta: f64)
                        -> VecBox {
    let domain = if domain == 0 {
        DataDomain::Time
    } else {
        DataDomain::Frequency
    };

    let mut vector = Box::new(VecBuf {
        vec: vec!(init_value; length).to_gen_dsp_vec(is_complex != 0, domain),
        buffer: SingleBuffer::new(),
    });
    vector.vec.set_delta(delta);
    vector
}

#[no_mangle]
pub extern "C" fn new_with_performance_options64(is_complex: i32,
                                                 domain: i32,
                                                 init_value: f64,
                                                 length: usize,
                                                 delta: f64,
                                                 core_limit: usize)
                                                 -> VecBox {
    let domain = if domain == 0 {
        DataDomain::Time
    } else {
        DataDomain::Frequency
    };

    let mut vector = Box::new(VecBuf {
        vec: vec!(init_value; length).to_gen_dsp_vec(is_complex != 0, domain),
        buffer: SingleBuffer::new(),
    });
    vector.vec.set_delta(delta);
    vector.vec.set_multicore_settings(MultiCoreSettings::new(core_limit));
    vector

}



#[no_mangle]
pub extern "C" fn new_with_detailed_performance_options64(is_complex: i32,
                                                          domain: i32,
                                                          init_value: f64,
                                                          length: usize,
                                                          delta: f64,
                                                          core_limit: usize,
                                                          med_dual_core_threshold: usize,
                                                          med_multi_core_threshold: usize,
                                                          large_dual_core_threshold: usize,
                                                          large_multi_core_threshold: usize)
                                                          -> VecBox {
    let domain = if domain == 0 {
        DataDomain::Time
    } else {
        DataDomain::Frequency
    };

    let mut vector = Box::new(VecBuf {
        vec: vec!(init_value; length).to_gen_dsp_vec(is_complex != 0, domain),
        buffer: SingleBuffer::new(),
    });
    vector.vec.set_delta(delta);
    let multicore_settings =
        MultiCoreSettings::with_thresholds(
            core_limit,
            med_dual_core_threshold,
            med_multi_core_threshold,
            large_dual_core_threshold,
            large_multi_core_threshold);
    vector.vec.set_multicore_settings(multicore_settings);
    vector

}

#[no_mangle]
pub extern "C" fn get_value64(vector: &VecBuf, index: usize) -> f64 {
    vector.vec[index]
}

#[no_mangle]
pub extern "C" fn set_value64(vector: &mut VecBuf, index: usize, value: f64) {
    vector.vec[index] = value;
}

#[no_mangle]
pub extern "C" fn is_complex64(vector: &VecBuf) -> i32 {
    if vector.vec.is_complex() { 1 } else { 0 }
}

/// Returns the vector domain as integer:
///
/// 1. `0` for [`DataVecDomain::Time`](../../enum.DataVecDomain.html)
/// 2. `1` for [`DataVecDomain::Frequency`](../../enum.DataVecDomain.html)
///
/// if the function returns another value then please report a bug.
#[no_mangle]
pub extern "C" fn get_domain64(vector: &VecBuf) -> i32 {
    match vector.vec.domain() {
        DataDomain::Time => 0,
        DataDomain::Frequency => 1,
    }
}

#[no_mangle]
pub extern "C" fn get_len64(vector: &VecBuf) -> usize {
    vector.vec.len()
}

#[no_mangle]
pub extern "C" fn set_len64(vector: &mut VecBuf, len: usize) {
    let _ = vector.vec.resize(len);
}

#[no_mangle]
pub extern "C" fn get_points64(vector: &VecBuf) -> usize {
    vector.vec.points()
}

#[no_mangle]
pub extern "C" fn get_delta64(vector: &VecBuf) -> f64 {
    vector.vec.delta()
}

#[no_mangle]
pub extern "C" fn complex_data64(vector: &VecBuf) -> &[Complex64] {
    vector.vec.complex(..)
}

#[no_mangle]
pub extern "C" fn get_allocated_len64(vector: &VecBuf) -> usize {
    vector.vec.alloc_len()
}

#[no_mangle]
pub extern "C" fn add64(vector: Box<VecBuf>, operand: &VecBuf) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| v.add(&operand.vec))
}

#[no_mangle]
pub extern "C" fn sub64(vector: Box<VecBuf>, operand: &VecBox) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| v.sub(&operand.vec))
}

#[no_mangle]
pub extern "C" fn div64(vector: Box<VecBuf>, operand: &VecBuf) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| v.div(&operand.vec))
}

#[no_mangle]
pub extern "C" fn mul64(vector: Box<VecBuf>, operand: &VecBuf) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| v.mul(&operand.vec))
}

#[no_mangle]
pub extern "C" fn real_dot_product64(vector: &VecBuf,
                                     operand: &VecBuf)
                                     -> ScalarInteropResult<f64> {
    vector.convert_scalar(|v| {
        DotProductOps::<GenDspVec<Vec<f64>, f64>, f64, f64, meta::RealOrComplex, meta::TimeOrFreq>::dot_product(v, &operand.vec)
    },
    0.0)
}

#[no_mangle]
pub extern "C" fn complex_dot_product64(vector: &VecBuf,
                                        operand: &VecBuf)
                                        -> ScalarInteropResult<Complex64> {
    vector.convert_scalar(|v| {
        DotProductOps::<GenDspVec<Vec<f64>, f64>, Complex64, f64, meta::RealOrComplex, meta::TimeOrFreq>::dot_product(v, &operand.vec)
    },
    Complex64::new(0.0, 0.0))
}

#[no_mangle]
pub extern "C" fn real_statistics64(vector: &VecBuf) -> Statistics<f64> {
    let vec = &vector.vec as &StatisticsOps<f64, Result = Statistics<f64>>;
    vec.statistics()
}

#[no_mangle]
pub extern "C" fn complex_statistics64(vector: &VecBuf) -> Statistics<Complex64> {
    let vec = &vector.vec as &StatisticsOps<Complex64, Result = Statistics<Complex64>>;
    vec.statistics()
}

#[no_mangle]
pub extern "C" fn real_sum64(vector: &VecBuf) -> f64 {
    vector.vec.sum()
}

#[no_mangle]
pub extern "C" fn real_sum_sq64(vector: &VecBuf) -> f64 {
    vector.vec.sum_sq()
}

#[no_mangle]
pub extern "C" fn complex_sum64(vector: &VecBuf) -> Complex64 {
    vector.vec.sum()
}

#[no_mangle]
pub extern "C" fn complex_sum_sq64(vector: &VecBuf) -> Complex64 {
    vector.vec.sum_sq()
}
#[no_mangle]
pub extern "C" fn real_dot_product_prec64(vector: &VecBuf,
                                          operand: &VecBuf)
                                          -> ScalarInteropResult<f64> {
    vector.convert_scalar(|v| {
        PreciseDotProductOps::<GenDspVec<Vec<f64>, f64>, f64, f64, meta::RealOrComplex, meta::TimeOrFreq>::dot_product_prec(v, &operand.vec)
    },
    0.0)
}

#[no_mangle]
pub extern "C" fn complex_dot_product_prec64(vector: &VecBuf,
                                             operand: &VecBuf)
                                             -> ScalarInteropResult<Complex64> {
    vector.convert_scalar(|v| {
        PreciseDotProductOps::<GenDspVec<Vec<f64>, f64>, Complex64, f64, meta::RealOrComplex, meta::TimeOrFreq>::dot_product_prec(v, &operand.vec)
    },
    Complex64::new(0.0, 0.0))
}

#[no_mangle]
pub extern "C" fn real_statistics_prec64(vector: &VecBuf) -> Statistics<f64> {
    let vec = &vector.vec as &PreciseStatisticsOps<f64, Result = Statistics<f64>>;
    vec.statistics_prec()
}

#[no_mangle]
pub extern "C" fn complex_statistics_prec64(vector: &VecBuf) -> Statistics<Complex64> {
    let vec = &vector.vec as &PreciseStatisticsOps<Complex64, Result = Statistics<Complex64>>;
    vec.statistics_prec()
}

#[no_mangle]
pub extern "C" fn real_sum_prec64(vector: &VecBuf) -> f64 {
    vector.vec.sum_prec()
}

#[no_mangle]
pub extern "C" fn real_sum_sq_prec64(vector: &VecBuf) -> f64 {
    vector.vec.sum_sq_prec()
}

#[no_mangle]
pub extern "C" fn complex_sum_prec64(vector: &VecBuf) -> Complex64 {
    vector.vec.sum_prec()
}

#[no_mangle]
pub extern "C" fn complex_sum_sq_prec64(vector: &VecBuf) -> Complex64 {
    vector.vec.sum_sq_prec()
}

/// `padding_option` argument is translated to:
/// Returns the vector domain as integer:
///
/// 1. `0` for [`PaddingOption::End`](../../enum.PaddingOption.html)
/// 2. `1` for [`PaddingOption::Surround`](../../enum.PaddingOption.html)
/// 2. `2` for [`PaddingOption::Center`](../../enum.PaddingOption.html)
#[no_mangle]
pub extern "C" fn zero_pad64(vector: Box<VecBuf>,
                             points: usize,
                             padding_option: i32)
                             -> VectorInteropResult<VecBuf> {
    let padding_option = translate_to_padding_option(padding_option);
    vector.convert_vec(|v, b| v.zero_pad_b(b, points, padding_option))
}

#[no_mangle]
pub extern "C" fn zero_interleave64(vector: Box<VecBuf>,
                                    factor: i32)
                                    -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, b| Ok(v.zero_interleave_b(b, factor as u32)))
}

#[no_mangle]
pub extern "C" fn diff64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.diff()))
}

#[no_mangle]
pub extern "C" fn diff_with_start64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.diff_with_start()))
}

#[no_mangle]
pub extern "C" fn cum_sum64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.cum_sum()))
}

#[no_mangle]
pub extern "C" fn real_offset64(vector: Box<VecBuf>, value: f64) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.offset(value)))
}

#[no_mangle]
pub extern "C" fn real_scale64(vector: Box<VecBuf>, value: f64) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.scale(value)))
}

#[no_mangle]
pub extern "C" fn abs64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.abs()))
}

#[no_mangle]
pub extern "C" fn sqrt64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.sqrt()))
}

#[no_mangle]
pub extern "C" fn square64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.square()))
}

#[no_mangle]
pub extern "C" fn root64(vector: Box<VecBuf>, value: f64) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.root(value)))
}

#[no_mangle]
pub extern "C" fn powf64(vector: Box<VecBuf>, value: f64) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.powf(value)))
}

#[no_mangle]
pub extern "C" fn ln64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.ln()))
}

#[no_mangle]
pub extern "C" fn exp64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.exp()))
}

#[no_mangle]
pub extern "C" fn log64(vector: Box<VecBuf>, value: f64) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.log(value)))
}

#[no_mangle]
pub extern "C" fn expf64(vector: Box<VecBuf>, value: f64) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.expf(value)))
}

#[no_mangle]
pub extern "C" fn to_complex64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.trans_vec(|v, b| Ok(v.to_complex_b(b)))
}

#[no_mangle]
pub extern "C" fn sin64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.sin()))
}

#[no_mangle]
pub extern "C" fn cos64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.cos()))
}

#[no_mangle]
pub extern "C" fn tan64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.tan()))
}

#[no_mangle]
pub extern "C" fn asin64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.asin()))
}

#[no_mangle]
pub extern "C" fn acos64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.acos()))
}

#[no_mangle]
pub extern "C" fn atan64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.tan()))
}

#[no_mangle]
pub extern "C" fn sinh64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.sinh()))
}
#[no_mangle]
pub extern "C" fn cosh64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.cosh()))
}

#[no_mangle]
pub extern "C" fn tanh64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.tanh()))
}

#[no_mangle]
pub extern "C" fn asinh64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.asinh()))
}

#[no_mangle]
pub extern "C" fn acosh64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.acosh()))
}

#[no_mangle]
pub extern "C" fn atanh64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.atanh()))
}

#[no_mangle]
pub extern "C" fn ln_approx64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.ln_approx()))
}

#[no_mangle]
pub extern "C" fn exp_approx64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.exp_approx()))
}

#[no_mangle]
pub extern "C" fn sin_approx64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.sin_approx()))
}

#[no_mangle]
pub extern "C" fn cos_approx64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.cos_approx()))
}

#[no_mangle]
pub extern "C" fn log_approx64(vector: Box<VecBuf>, value: f64) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.log_approx(value)))
}

#[no_mangle]
pub extern "C" fn expf_approx64(vector: Box<VecBuf>, value: f64) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.expf_approx(value)))
}

#[no_mangle]
pub extern "C" fn powf_approx64(vector: Box<VecBuf>, value: f64) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.powf_approx(value)))
}

#[no_mangle]
pub extern "C" fn wrap64(vector: Box<VecBuf>, value: f64) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.wrap(value)))
}

#[no_mangle]
pub extern "C" fn unwrap64(vector: Box<VecBuf>, value: f64) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.unwrap(value)))
}

#[no_mangle]
pub extern "C" fn swap_halves64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.swap_halves()))
}

#[no_mangle]
pub extern "C" fn complex_offset64(vector: Box<VecBuf>,
                                   real: f64,
                                   imag: f64)
                                   -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.offset(Complex64::new(real, imag))))
}

#[no_mangle]
pub extern "C" fn complex_scale64(vector: Box<VecBuf>,
                                  real: f64,
                                  imag: f64)
                                  -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.scale(Complex64::new(real, imag))))
}

#[no_mangle]
pub extern "C" fn complex_divide64(vector: Box<VecBuf>,
                                   real: f64,
                                   imag: f64)
                                   -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.scale(Complex64::new(1.0, 0.0) / Complex64::new(real, imag))))
}

#[no_mangle]
pub extern "C" fn magnitude64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.trans_vec(|v, b| Ok(v.magnitude_b(b)))
}

#[no_mangle]
pub extern "C" fn get_magnitude64(vector: Box<VecBuf>, destination: &mut VecBuf) -> i32 {
    convert_void(Ok(vector.vec.get_magnitude(&mut destination.vec)))
}

#[no_mangle]
pub extern "C" fn get_magnitude_squared64(vector: Box<VecBuf>, destination: &mut VecBuf) -> i32 {
    convert_void(Ok(vector.vec.get_magnitude_squared(&mut destination.vec)))
}

#[no_mangle]
pub extern "C" fn magnitude_squared64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.trans_vec(|v, b| Ok(v.magnitude_squared_b(b)))
}

#[no_mangle]
pub extern "C" fn conj64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.conj()))
}

#[no_mangle]
pub extern "C" fn to_real64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.trans_vec(|v, b| Ok(v.to_real_b(b)))
}

#[no_mangle]
pub extern "C" fn to_imag64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.trans_vec(|v, b| Ok(v.to_imag_b(b)))
}

#[no_mangle]
pub extern "C" fn map_inplace_real64(vector: Box<VecBuf>,
                                     map: extern "C" fn(f64, usize) -> f64)
                                     -> VectorInteropResult<VecBuf> {
    let map = move |v, i, _| map(v, i);
    vector.convert_vec(|v, _| Ok(v.map_inplace((), &map)))
}

#[no_mangle]
pub extern "C" fn map_inplace_complex64(vector: Box<VecBuf>,
                                        map: extern "C" fn(Complex64, usize) -> Complex64)
                                        -> VectorInteropResult<VecBuf> {
    let map = move |v, i, _| map(v, i);
    vector.convert_vec(|v, _| Ok(v.map_inplace((), &map)))
}

/// Warning: This function interface heavily works around the Rust type system and the safety
/// it provides. Use with great care!
#[no_mangle]
pub extern "C" fn map_aggregate_real64(vector: &VecBuf,
                                       map: extern "C" fn(f64, usize) -> *const c_void,
                                       aggregate: extern "C" fn(*const c_void, *const c_void)
                                                                -> *const c_void)
                                       -> ScalarResult<*const c_void> {
    unsafe {
        let map = move |v, i, _| mem::transmute(map(v, i));
        let aggr = move |a: usize, b: usize| {
            mem::transmute(aggregate(mem::transmute(a), mem::transmute(b)))
        };

        let result =
            vector.convert_scalar(|v| v.map_aggregate((), &map, &aggr), mem::transmute(0usize));
        mem::transmute(result)
    }
}

/// Warning: This function interface heavily works around the Rust type system and the safety
/// it provides. Use with great care!
#[no_mangle]
pub extern "C" fn map_aggregate_complex64(vector: &VecBuf,
                                          map: extern "C" fn(Complex64, usize) -> *const c_void,
                                          aggregate: extern "C" fn(*const c_void, *const c_void)
                                                                   -> *const c_void)
                                          -> ScalarResult<*const c_void> {
    unsafe {
        let map = move |v, i, _| mem::transmute(map(v, i));
        let aggr = move |a: usize, b: usize| {
            mem::transmute(aggregate(mem::transmute(a), mem::transmute(b)))
        };

        let result =
            vector.convert_scalar(|v| v.map_aggregate((), &map, &aggr), mem::transmute(0usize));
        mem::transmute(result)
    }
}

#[no_mangle]
pub extern "C" fn get_real64(vector: Box<VecBuf>, destination: &mut VecBuf) -> i32 {
    convert_void(Ok(vector.vec.get_real(&mut destination.vec)))
}

#[no_mangle]
pub extern "C" fn get_imag64(vector: Box<VecBuf>, destination: &mut VecBuf) -> i32 {
    convert_void(Ok(vector.vec.get_imag(&mut destination.vec)))
}

#[no_mangle]
pub extern "C" fn phase64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.trans_vec(|v, b| Ok(v.phase_b(b)))
}

#[no_mangle]
pub extern "C" fn get_phase64(vector: Box<VecBuf>, destination: &mut VecBuf) -> i32 {
    convert_void(Ok(vector.vec.get_phase(&mut destination.vec)))
}

#[no_mangle]
pub extern "C" fn plain_fft64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.trans_vec(|v, b| Ok(v.plain_fft(b)))
}

#[no_mangle]
pub extern "C" fn plain_sfft64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.trans_vec(|v, b| v.plain_sfft(b))
}

#[no_mangle]
pub extern "C" fn plain_ifft64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.trans_vec(|v, b| Ok(v.plain_ifft(b)))
}

#[no_mangle]
pub extern "C" fn clone64(vector: Box<VecBuf>) -> Box<VecBuf> {
    Box::new(VecBuf {
        vec: vector.vec.clone(),
        buffer: SingleBuffer::new(),
    })
}

#[no_mangle]
pub extern "C" fn multiply_complex_exponential64(vector: Box<VecBuf>,
                                                 a: f64,
                                                 b: f64)
                                                 -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.multiply_complex_exponential(a, b)))
}

#[no_mangle]
pub extern "C" fn add_vector64(vector: Box<VecBuf>,
                               operand: &VecBuf)
                               -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| v.add(&operand.vec))
}

#[no_mangle]
pub extern "C" fn sub_vector64(vector: Box<VecBuf>,
                               operand: &VecBuf)
                               -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| v.sub(&operand.vec))
}

#[no_mangle]
pub extern "C" fn div_vector64(vector: Box<VecBuf>,
                               operand: &VecBuf)
                               -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| v.div(&operand.vec))
}

#[no_mangle]
pub extern "C" fn mul_vector64(vector: Box<VecBuf>,
                               operand: &VecBuf)
                               -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| v.mul(&operand.vec))
}

#[no_mangle]
pub extern "C" fn add_smaller_vector64(vector: Box<VecBuf>,
                                       operand: &VecBuf)
                                       -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| v.add_smaller(&operand.vec))
}

#[no_mangle]
pub extern "C" fn sub_smaller_vector64(vector: Box<VecBuf>,
                                       operand: &VecBuf)
                                       -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| v.sub_smaller(&operand.vec))
}

#[no_mangle]
pub extern "C" fn div_smaller_vector64(vector: Box<VecBuf>,
                                       operand: &VecBuf)
                                       -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| v.div_smaller(&operand.vec))
}

#[no_mangle]
pub extern "C" fn mul_smaller_vector64(vector: Box<VecBuf>,
                                       operand: &VecBuf)
                                       -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| v.mul_smaller(&operand.vec))
}

#[no_mangle]
pub extern "C" fn get_real_imag64(vector: Box<VecBuf>,
                                  real: &mut VecBuf,
                                  imag: &mut VecBuf)
                                  -> i32 {
    convert_void(Ok(vector.vec.get_real_imag(&mut real.vec, &mut imag.vec)))
}

#[no_mangle]
pub extern "C" fn get_mag_phase64(vector: Box<VecBuf>,
                                  mag: &mut VecBuf,
                                  phase: &mut VecBuf)
                                  -> i32 {
    convert_void(Ok(vector.vec.get_mag_phase(&mut mag.vec, &mut phase.vec)))
}

#[no_mangle]
pub extern "C" fn set_real_imag64(vector: Box<VecBuf>,
                                  real: &VecBuf,
                                  imag: &VecBuf)
                                  -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| v.set_real_imag(&real.vec, &imag.vec))
}

#[no_mangle]
pub extern "C" fn set_mag_phase64(vector: Box<VecBuf>,
                                  mag: &VecBuf,
                                  phase: &VecBuf)
                                  -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| v.set_mag_phase(&mag.vec, &phase.vec))
}

#[no_mangle]
pub extern "C" fn split_into64(vector: &VecBuf, targets: *mut Box<VecBuf>, len: usize) -> i32 {
    unsafe {
        let targets = slice::from_raw_parts_mut(targets, len);
        let mut targets: Vec<&mut GenDspVec<Vec<f64>, f64>> =
            targets.iter_mut().map(|x| &mut x.vec).collect();
        convert_void(vector.vec.split_into(&mut targets))
    }
}

#[no_mangle]
pub extern "C" fn merge64(vector: Box<VecBuf>,
                          sources: *const Box<VecBuf>,
                          len: usize)
                          -> VectorInteropResult<VecBuf> {
    unsafe {
        let sources = slice::from_raw_parts(sources, len);
        let sources: Vec<&GenDspVec<Vec<f64>, f64>> = sources.iter().map(|x| &x.vec).collect();
        vector.convert_vec(|v, _| v.merge(&sources))
    }
}

#[no_mangle]
pub extern "C" fn overwrite_data64(mut vector: Box<VecBuf>,
                                   data: *const f64,
                                   len: usize)
                                   -> VectorInteropResult<VecBuf> {
    let data = unsafe { slice::from_raw_parts(data, len) };
    if len < vector.vec.len() {
        vector.vec[0..len].clone_from_slice(&data);
        VectorInteropResult {
            result_code: 0,
            vector: vector,
        }
    } else {
        VectorInteropResult {
            result_code: translate_error(ErrorReason::InvalidArgumentLength),
            vector: vector,
        }
    }
}

#[no_mangle]
pub extern "C" fn real_statistics_split64(vector: &VecBuf,
                                          data: *mut Statistics<f64>,
                                          len: usize)
                                          -> i32 {
    let data = unsafe { slice::from_raw_parts_mut(data, len) };
    let vec = &vector.vec as &StatisticsSplitOps<f64, Result = StatsVec<Statistics<f64>>>;
    let stats = vec.statistics_split(data.len());
    match stats {
        Ok(s) => {
            for i in 0..s.len() {
                data[i] = s[i];
            }

            0
        }
        Err(r) => translate_error(r),
    }
}

#[no_mangle]
pub extern "C" fn complex_statistics_split64(vector: &VecBuf,
                                             data: *mut Statistics<Complex64>,
                                             len: usize)
                                             -> i32 {
    let data = unsafe { slice::from_raw_parts_mut(data, len) };
    let vec = &vector.vec as &StatisticsSplitOps<Complex64,
                                                 Result = StatsVec<Statistics<Complex64>>>;
    let stats = vec.statistics_split(data.len());
    match stats {
        Ok(s) => {
            for i in 0..s.len() {
                data[i] = s[i];
            }

            0
        }
        Err(r) => translate_error(r),
    }
}

#[no_mangle]
pub extern "C" fn real_statistics_split_prec64(vector: &VecBuf,
                                               data: *mut Statistics<f64>,
                                               len: usize)
                                               -> i32 {
    let data = unsafe { slice::from_raw_parts_mut(data, len) };
    let vec = &vector.vec as &PreciseStatisticsSplitOps<f64, Result = StatsVec<Statistics<f64>>>;
    let stats = vec.statistics_split_prec(data.len());
    match stats {
        Ok(s) => {
            for i in 0..s.len() {
                data[i] = s[i];
            }

            0
        }
        Err(r) => translate_error(r),
    }
}

#[no_mangle]
pub extern "C" fn complex_statistics_split_prec64(vector: &VecBuf,
                                                  data: *mut Statistics<Complex64>,
                                                  len: usize)
                                                  -> i32 {
    let data = unsafe { slice::from_raw_parts_mut(data, len) };
    let vec = &vector.vec as &PreciseStatisticsSplitOps<Complex64,
                                                        Result = StatsVec<Statistics<Complex64>>>;
    let stats = vec.statistics_split_prec(data.len());
    match stats {
        Ok(s) => {
            for i in 0..s.len() {
                data[i] = s[i];
            }

            0
        }
        Err(r) => translate_error(r),
    }
}

#[no_mangle]
pub extern "C" fn fft64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.trans_vec(|v, b| Ok(v.fft(b)))
}

#[no_mangle]
pub extern "C" fn sfft64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.trans_vec(|v, b| v.sfft(b))
}

#[no_mangle]
pub extern "C" fn ifft64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.trans_vec(|v, b| Ok(v.ifft(b)))
}

#[no_mangle]
pub extern "C" fn plain_sifft64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.trans_vec(|v, b| v.plain_sifft(b))
}

#[no_mangle]
pub extern "C" fn sifft64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.trans_vec(|v, b| v.sifft(b))
}

#[no_mangle]
pub extern "C" fn mirror64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, b| Ok(v.mirror(b)))
}

pub extern "C" fn fft_shift64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.fft_shift()))
}

pub extern "C" fn ifft_shift64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.ifft_shift()))
}

/// `window` argument is translated to:
///
/// 1. `0` to [`TriangularWindow`](../../window_functions/struct.TriangularWindow.html)
/// 2. `1` to [`HammingWindow`](../../window_functions/struct.TriangularWindow.html)
#[no_mangle]
pub extern "C" fn apply_window64(vector: Box<VecBuf>, window: i32) -> VectorInteropResult<VecBuf> {
    let window = translate_to_window_function(window);
    vector.convert_vec(|v, _| Ok(v.apply_window(window.as_ref())))
}

/// See [`apply_window64`](fn.apply_window64.html) for a description of the `window` parameter.
#[no_mangle]
pub extern "C" fn unapply_window64(vector: Box<VecBuf>,
                                   window: i32)
                                   -> VectorInteropResult<VecBuf> {
    let window = translate_to_window_function(window);
    vector.convert_vec(|v, _| Ok(v.unapply_window(window.as_ref())))
}

/// See [`apply_window64`](fn.apply_window64.html) for a description of the `window` parameter.
#[no_mangle]
pub extern "C" fn windowed_fft64(vector: Box<VecBuf>, window: i32) -> VectorInteropResult<VecBuf> {
    let window = translate_to_window_function(window);
    vector.trans_vec(|v, b| Ok(v.windowed_fft(b, window.as_ref())))
}

/// See [`apply_window64`](fn.apply_window64.html) for a description of the `window` parameter.
#[no_mangle]
pub extern "C" fn windowed_sfft64(vector: Box<VecBuf>, window: i32) -> VectorInteropResult<VecBuf> {
    let window = translate_to_window_function(window);
    vector.trans_vec(|v, b| v.windowed_sfft(b, window.as_ref()))
}

/// See [`apply_window64`](fn.apply_window64.html) for a description of the `window` parameter.
#[no_mangle]
pub extern "C" fn windowed_ifft64(vector: Box<VecBuf>, window: i32) -> VectorInteropResult<VecBuf> {
    let window = translate_to_window_function(window);
    vector.trans_vec(|v, b| Ok(v.windowed_ifft(b, window.as_ref())))
}

/// See [`apply_window64`](fn.apply_window64.html) for a description of the `window` parameter.
#[no_mangle]
pub extern "C" fn windowed_sifft64(vector: Box<VecBuf>,
                                   window: i32)
                                   -> VectorInteropResult<VecBuf> {
    let window = translate_to_window_function(window);
    vector.trans_vec(|v, b| v.windowed_sifft(b, window.as_ref()))
}

/// Creates a window from the function `window` and the void pointer `window_data`.
/// The `window_data` pointer is passed to the `window`
/// function at every call and can be used to store parameters.
#[no_mangle]
pub extern "C" fn apply_custom_window64(vector: Box<VecBuf>,
                                        window: extern "C" fn(*const c_void, usize, usize) -> f64,
                                        window_data: *const c_void,
                                        is_symmetric: bool)
                                        -> VectorInteropResult<VecBuf> {
    unsafe {
        let window = ForeignWindowFunction {
            window_function: window,
            window_data: mem::transmute(window_data),
            is_symmetric: is_symmetric,
        };
        vector.convert_vec(|v, _| Ok(v.apply_window(&window)))
    }
}

/// See [`apply_custom_window64`](fn.apply_custom_window64.html) for a description of the
/// `window` and `window_data` parameter.
#[no_mangle]
pub extern "C" fn unapply_custom_window64(vector: Box<VecBuf>,
                                          window: extern "C" fn(*const c_void, usize, usize) -> f64,
                                          window_data: *const c_void,
                                          is_symmetric: bool)
                                          -> VectorInteropResult<VecBuf> {
    unsafe {
        let window = ForeignWindowFunction {
            window_function: window,
            window_data: mem::transmute(window_data),
            is_symmetric: is_symmetric,
        };
        vector.convert_vec(|v, _| Ok(v.unapply_window(&window)))
    }
}

/// See [`apply_custom_window64`](fn.apply_custom_window64.html) for a description of the
/// `window` and `window_data` parameter.
#[no_mangle]
pub extern "C" fn windowed_custom_fft64(vector: Box<VecBuf>,
                                        window: extern "C" fn(*const c_void, usize, usize) -> f64,
                                        window_data: *const c_void,
                                        is_symmetric: bool)
                                        -> VectorInteropResult<VecBuf> {
    unsafe {
        let window = ForeignWindowFunction {
            window_function: window,
            window_data: mem::transmute(window_data),
            is_symmetric: is_symmetric,
        };
        vector.trans_vec(|v, b| Ok(v.windowed_fft(b, &window)))
    }
}

/// See [`apply_custom_window64`](fn.apply_custom_window64.html) for a description of the
/// `window` and `window_data` parameter.
#[no_mangle]
pub extern "C" fn windowed_custom_sfft64(vector: Box<VecBuf>,
                                         window: extern "C" fn(*const c_void, usize, usize) -> f64,
                                         window_data: *const c_void,
                                         is_symmetric: bool)
                                         -> VectorInteropResult<VecBuf> {
    unsafe {
        let window = ForeignWindowFunction {
            window_function: window,
            window_data: mem::transmute(window_data),
            is_symmetric: is_symmetric,
        };
        vector.trans_vec(|v, b| v.windowed_sfft(b, &window))
    }
}

/// See [`apply_custom_window64`](fn.apply_custom_window64.html) for a description of the
/// `window` and `window_data` parameter.
#[no_mangle]
pub extern "C" fn windowed_custom_ifft64(vector: Box<VecBuf>,
                                         window: extern "C" fn(*const c_void, usize, usize) -> f64,
                                         window_data: *const c_void,
                                         is_symmetric: bool)
                                         -> VectorInteropResult<VecBuf> {
    unsafe {
        let window = ForeignWindowFunction {
            window_function: window,
            window_data: mem::transmute(window_data),
            is_symmetric: is_symmetric,
        };
        vector.trans_vec(|v, b| Ok(v.windowed_ifft(b, &window)))
    }
}

/// See [`apply_custom_window64`](fn.apply_custom_window64.html) for a description of the
/// `window` and `window_data` parameter.
#[no_mangle]
pub extern "C" fn windowed_custom_sifft64(vector: Box<VecBuf>,
                                          window: extern "C" fn(*const c_void, usize, usize) -> f64,
                                          window_data: *const c_void,
                                          is_symmetric: bool)
                                          -> VectorInteropResult<VecBuf> {
    unsafe {
        let window = ForeignWindowFunction {
            window_function: window,
            window_data: mem::transmute(window_data),
            is_symmetric: is_symmetric,
        };
        vector.trans_vec(|v, b| v.windowed_sifft(b, &window))
    }
}

#[no_mangle]
pub extern "C" fn reverse64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.reverse()))
}

#[no_mangle]
pub extern "C" fn decimatei64(vector: Box<VecBuf>,
                              decimation_factor: u32,
                              delay: u32)
                              -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, _| Ok(v.decimatei(decimation_factor, delay)))
}

#[no_mangle]
pub extern "C" fn prepare_argument64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.trans_vec(|v, b| Ok(v.prepare_argument(b)))
}

#[no_mangle]
pub extern "C" fn prepare_argument_padded64(vector: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
    vector.trans_vec(|v, b| Ok(v.prepare_argument_padded(b)))
}

#[no_mangle]
pub extern "C" fn correlate64(vector: Box<VecBuf>, other: &VecBuf) -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, b| v.correlate(b, &other.vec))
}

#[no_mangle]
pub extern "C" fn convolve_signal64(vector: Box<VecBuf>,
                                    impulse_response: &VecBuf)
                                    -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, b| v.convolve_signal(b, &impulse_response.vec))
}

/// Convolves the vector with an impulse response defined by `impulse_response` and
/// the void pointer `impulse_response_data`.
/// The `impulse_response_data` pointer is passed to the `impulse_response`
/// function at every call and can be used to store parameters.
#[no_mangle]
pub extern "C" fn convolve_real64(vector: Box<VecBuf>,
                                  impulse_response: extern "C" fn(*const c_void, f64) -> f64,
                                  impulse_response_data: *const c_void,
                                  is_symmetric: bool,
                                  ratio: f64,
                                  len: usize)
                                  -> VectorInteropResult<VecBuf> {
    unsafe {
        let function: &RealImpulseResponse<f64> = &ForeignRealConvolutionFunction {
            conv_function: impulse_response,
            conv_data: mem::transmute(impulse_response_data),
            is_symmetric: is_symmetric,
        };
        vector.convert_vec(|v, b| Ok(v.convolve(b, function, ratio, len)))
    }
}

/// Convolves the vector with an impulse response defined by `impulse_response` and the
/// void pointer `impulse_response_data`.
/// The `impulse_response_data` pointer is passed to the `impulse_response`
/// function at every call and can be used to store parameters.
#[no_mangle]
pub extern "C" fn convolve_complex64(vector: Box<VecBuf>,
                                     impulse_response: extern "C" fn(*const c_void, f64)
                                                                     -> Complex64,
                                     impulse_response_data: *const c_void,
                                     is_symmetric: bool,
                                     ratio: f64,
                                     len: usize)
                                     -> VectorInteropResult<VecBuf> {
    unsafe {
        let function: &ComplexImpulseResponse<f64> = &ForeignComplexConvolutionFunction {
            conv_function: impulse_response,
            conv_data: mem::transmute(impulse_response_data),
            is_symmetric: is_symmetric,
        };
        vector.convert_vec(|v, b| Ok(v.convolve(b, function, ratio, len)))
    }
}

/// `impulse_response` argument is translated to:
///
/// 1. `0` to [`SincFunction`](../../conv_types/struct.SincFunction.html)
/// 2. `1` to [`RaisedCosineFunction`](../../conv_types/struct.RaisedCosineFunction.html)
///
/// `rolloff` is only used if this is a valid parameter for the selected `impulse_response`
#[no_mangle]
pub extern "C" fn convolve64(vector: Box<VecBuf>,
                             impulse_response: i32,
                             rolloff: f64,
                             ratio: f64,
                             len: usize)
                             -> VectorInteropResult<VecBuf> {
    let function = translate_to_real_convolution_function(impulse_response, rolloff);
    vector.convert_vec(|v, b| Ok(v.convolve(b, function.as_ref(), ratio, len)))
}

/// Convolves the vector with an impulse response defined by `frequency_response` and
/// the void pointer `frequency_response_data`.
/// The `frequency_response_data` pointer is passed to the `frequency_response`
/// function at every call and can be used to store parameters.
#[no_mangle]
pub extern fn multiply_frequency_response_real64(vector: Box<VecBuf>,
    frequency_response: extern fn(*const c_void, f64) -> f64,
    frequency_response_data: *const c_void,
    is_symmetric: bool,
    ratio: f64) -> VectorInteropResult<VecBuf> {
    unsafe {
        let function: &RealFrequencyResponse<f64> = &ForeignRealConvolutionFunction {
            conv_function: frequency_response,
            conv_data: mem::transmute(frequency_response_data),
            is_symmetric: is_symmetric,
        };
        vector.convert_vec(|v, _| Ok(v.multiply_frequency_response(function, ratio)))
    }
}

/// Convolves the vector with an impulse response defined by `frequency_response`
/// and the void pointer `frequency_response_data`.
/// The `frequency_response` pointer is passed to the `frequency_response`
/// function at every call and can be used to store parameters.
#[no_mangle]
pub extern fn multiply_frequency_response_complex64(vector: Box<VecBuf>,
    frequency_response: extern fn(*const c_void, f64) -> Complex64,
    frequency_response_data: *const c_void,
    is_symmetric: bool,
    ratio: f64) -> VectorInteropResult<VecBuf> {
    unsafe {
        let function: &ComplexFrequencyResponse<f64> = &ForeignComplexConvolutionFunction {
            conv_function: frequency_response,
            conv_data: mem::transmute(frequency_response_data),
            is_symmetric: is_symmetric,
        };
        vector.convert_vec(|v, _| Ok(v.multiply_frequency_response(function, ratio)))
    }
}

/// `frequency_response` argument is translated to:
///
/// 1. `0` to [`SincFunction`](../../conv_types/struct.SincFunction.html)
/// 2. `1` to [`RaisedCosineFunction`](../../conv_types/struct.RaisedCosineFunction.html)
///
/// `rolloff` is only used if this is a valid parameter for the selected `frequency_response`
#[no_mangle]
pub extern "C" fn multiply_frequency_response64(vector: Box<VecBuf>,
                                                frequency_response: i32,
                                                rolloff: f64,
                                                ratio: f64)
                                                -> VectorInteropResult<VecBuf> {
    let function = translate_to_real_frequency_response(frequency_response, rolloff);
    vector.convert_vec(|v, _| Ok(v.multiply_frequency_response(function.as_ref(), ratio)))
}

/// Convolves the vector with an impulse response defined by `impulse_response` and
/// the void pointer `impulse_response_data`.
/// The `impulse_response_data` pointer is passed to the `impulse_response`
/// function at every call and can be used to store parameters.
#[no_mangle]
pub extern "C" fn interpolatef_custom64(vector: Box<VecBuf>,
                                        impulse_response: extern "C" fn(*const c_void, f64) -> f64,
                                        impulse_response_data: *const c_void,
                                        is_symmetric: bool,
                                        interpolation_factor: f64,
                                        delay: f64,
                                        len: usize)
                                        -> VectorInteropResult<VecBuf> {
    unsafe {
        let function: &RealImpulseResponse<f64> = &ForeignRealConvolutionFunction {
            conv_function: impulse_response,
            conv_data: mem::transmute(impulse_response_data),
            is_symmetric: is_symmetric,
        };
        vector.convert_vec(|v, b| Ok(v.interpolatef(b, function, interpolation_factor, delay, len)))
    }
}

/// `impulse_response` argument is translated to:
///
/// 1. `0` to [`SincFunction`](../../conv_types/struct.SincFunction.html)
/// 2. `1` to [`RaisedCosineFunction`](../../conv_types/struct.RaisedCosineFunction.html)
///
/// `rolloff` is only used if this is a valid parameter for the selected `impulse_response`
#[no_mangle]
pub extern "C" fn interpolatef64(vector: Box<VecBuf>,
                                 impulse_response: i32,
                                 rolloff: f64,
                                 interpolation_factor: f64,
                                 delay: f64,
                                 len: usize)
                                 -> VectorInteropResult<VecBuf> {
    let function = translate_to_real_convolution_function(impulse_response, rolloff);
    vector.convert_vec(|v, b| {
        Ok(v.interpolatef(b, function.as_ref(), interpolation_factor, delay, len))
    })
}

/// Convolves the vector with an frequency response defined by `frequency_response` and
/// the void pointer `frequency_response_data`.
/// The `frequency_response_data` pointer is passed to the `frequency_response`
/// function at every call and can be used to store parameters.
#[no_mangle]
pub extern "C" fn interpolate_custom64(vector: Box<VecBuf>,
                                       frequency_response: extern "C" fn(*const c_void, f64) -> f64,
                                       frequency_response_data: *const c_void,
                                       is_symmetric: bool,
                                       dest_points: usize,
                                       delay: f64)
                                       -> VectorInteropResult<VecBuf> {
    unsafe {
        let function: &RealFrequencyResponse<f64> = &ForeignRealConvolutionFunction {
            conv_function: frequency_response,
            conv_data: mem::transmute(frequency_response_data),
            is_symmetric: is_symmetric,
        };
        vector.convert_vec(|v, b| v.interpolate(b, Some(function), dest_points, delay))
    }
}

/// `frequency_response` argument is translated to:
///
/// 1. `0` to [`SincFunction`](../../conv_types/struct.SincFunction.html)
/// 2. `1` to [`RaisedCosineFunction`](../../conv_types/struct.RaisedCosineFunction.html)
///
/// `rolloff` is only used if this is a valid parameter for the selected `impulse_response`
#[no_mangle]
pub extern "C" fn interpolate64(vector: Box<VecBuf>,
                                frequency_response: i32,
                                rolloff: f64,
                                dest_points: usize,
                                delay: f64)
                                -> VectorInteropResult<VecBuf> {
    let function = translate_to_real_frequency_response(frequency_response, rolloff);
    vector.convert_vec(|v, b| v.interpolate(b, Some(function.as_ref()), dest_points, delay))
}

#[no_mangle]
pub extern "C" fn interpft64(vector: Box<VecBuf>,
                             dest_points: usize)
                             -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, b| Ok(v.interpft(b, dest_points)))
}

/// Convolves the vector with an impulse response defined by `frequency_response` and
/// the void pointer `frequency_response_data`.
/// The `frequency_response_data` pointer is passed to the `frequency_response`
/// function at every call and can be used to store parameters.
#[no_mangle]
pub extern "C" fn interpolatei_custom64(vector: Box<VecBuf>,
                                        frequency_response: extern "C" fn(*const c_void, f64)
                                                                          -> f64,
                                        frequency_response_data: *const c_void,
                                        is_symmetric: bool,
                                        interpolation_factor: i32)
                                        -> VectorInteropResult<VecBuf> {
    unsafe {
        let function: &RealFrequencyResponse<f64> = &ForeignRealConvolutionFunction {
            conv_function: frequency_response,
            conv_data: mem::transmute(frequency_response_data),
            is_symmetric: is_symmetric,
        };
        vector.convert_vec(|v, b| v.interpolatei(b, function, interpolation_factor as u32))
    }
}

/// `frequency_response` argument is translated to:
///
/// 1. `0` to [`SincFunction`](../../conv_types/struct.SincFunction.html)
/// 2. `1` to [`RaisedCosineFunction`](../../conv_types/struct.RaisedCosineFunction.html)
///
/// `rolloff` is only used if this is a valid parameter for the selected `frequency_response`
#[no_mangle]
pub extern "C" fn interpolatei64(vector: Box<VecBuf>,
                                 frequency_response: i32,
                                 rolloff: f64,
                                 interpolation_factor: i32)
                                 -> VectorInteropResult<VecBuf> {
    let function = translate_to_real_frequency_response(frequency_response, rolloff);
    vector.convert_vec(|v, b| v.interpolatei(b, function.as_ref(), interpolation_factor as u32))
}

#[no_mangle]
pub extern "C" fn interpolate_lin64(vector: Box<VecBuf>,
                                    interpolation_factor: f64,
                                    delay: f64)
                                    -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, b| Ok(v.interpolate_lin(b, interpolation_factor, delay)))
}

#[no_mangle]
pub extern "C" fn interpolate_hermite64(vector: Box<VecBuf>,
                                        interpolation_factor: f64,
                                        delay: f64)
                                        -> VectorInteropResult<VecBuf> {
    vector.convert_vec(|v, b| Ok(v.interpolate_hermite(b, interpolation_factor, delay)))
}
