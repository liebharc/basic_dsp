// Auto generated code, change facade32.rs and run facade64_create.pl
//! Functions for 64bit floating point number based vectors. Please refer to the other chapters of the help for documentation of the functions.
use super::*;
use super::super::*;
use window_functions::*;
use conv_types::*;
use num::complex::Complex64;
use std::slice;
use std::os::raw::c_void;
use std::mem;

#[no_mangle]
pub extern fn delete_vector64(vector: Box<DataVector64>) {
    drop(vector);
}
 
#[no_mangle]
pub extern fn new64(is_complex: i32, domain: i32, init_value: f64, length: usize, delta: f64) -> Box<DataVector64> {
    let domain = if domain == 0 {
            DataVectorDomain::Time
        }
        else {
            DataVectorDomain::Frequency
        };
        
    let vector = Box::new(DataVector64::new(is_complex != 0, domain, init_value, length, delta));
    vector
}

#[no_mangle]
pub extern fn new_with_performance_options64(is_complex: i32, domain: i32, init_value: f64, length: usize, delta: f64, core_limit: usize, early_temp_allocation: bool) -> Box<DataVector64> {
    let domain = if domain == 0 {
            DataVectorDomain::Time
        }
        else {
            DataVectorDomain::Frequency
        };
        
    let vector = Box::new(DataVector64::new_with_options(is_complex != 0, domain, init_value, length, delta, MultiCoreSettings::new(core_limit, early_temp_allocation)));
    vector
}

#[no_mangle]
pub extern fn get_value64(vector: &DataVector64, index: usize) -> f64 {
    vector[index]
}

#[no_mangle]
pub extern fn set_value64(vector: &mut DataVector64, index: usize, value : f64) {
    vector[index] = value;
}

#[no_mangle]
pub extern fn is_complex64(vector: &DataVector64) -> i32 {
    if vector.is_complex() {
        1
    } 
    else {
        0
    }
}

/// Returns the vector domain as integer:
///
/// 1. `0` for [`DataVectorDomain::Time`](../../enum.DataVectorDomain.html) 
/// 2. `1` for [`DataVectorDomain::Frequency`](../../enum.DataVectorDomain.html)
/// 
/// if the function returns another value then please report a bug.
#[no_mangle]
pub extern fn get_domain64(vector: &DataVector64) -> i32 {
    match vector.domain() {
        DataVectorDomain::Time => 0,
        DataVectorDomain::Frequency => 1,
    }
}

#[no_mangle]
pub extern fn get_len64(vector: &DataVector64) -> usize {
    vector.len()
}

#[no_mangle]
pub extern fn set_len64(vector: &mut DataVector64, len: usize) {
    vector.set_len(len)
}

#[no_mangle]
pub extern fn get_points64(vector: &DataVector64) -> usize {
    vector.points()
}

#[no_mangle]
pub extern fn get_delta64(vector: &DataVector64) -> f64 {
    vector.delta()
}

#[no_mangle]
pub extern fn complex_data64(vector: &DataVector64) -> &[Complex64] {
    vector.complex_data()
}

#[no_mangle]
pub extern fn get_allocated_len64(vector: &DataVector64) -> usize {
    vector.allocated_len()
}

#[no_mangle]
pub extern fn add_vector64(vector: Box<DataVector64>, operand: &DataVector64) -> VectorResult<DataVector64> {
    convert_vec!(vector.add_vector(operand))
}

#[no_mangle]
pub extern fn subtract_vector64(vector: Box<DataVector64>, operand: &DataVector64) -> VectorResult<DataVector64> {
    convert_vec!(vector.subtract_vector(operand))
}

#[no_mangle]
pub extern fn divide_vector64(vector: Box<DataVector64>, operand: &DataVector64) -> VectorResult<DataVector64> {
    convert_vec!(vector.divide_vector(operand))
}

#[no_mangle]
pub extern fn multiply_vector64(vector: Box<DataVector64>, operand: &DataVector64) -> VectorResult<DataVector64> {
    convert_vec!(vector.multiply_vector(operand))
}

#[no_mangle]
pub extern fn real_dot_product64(vector: &DataVector64, operand: &DataVector64) -> ScalarResult<f64> {
    convert_scalar!(vector.real_dot_product(operand), 0.0)
}

#[no_mangle]
pub extern fn complex_dot_product64(vector: &DataVector64, operand: &DataVector64) -> ScalarResult<Complex64> {
    convert_scalar!(vector.complex_dot_product(operand), Complex64::new(0.0, 0.0))
}

#[no_mangle]
pub extern fn real_statistics64(vector: &DataVector64) -> Statistics<f64> {
    vector.real_statistics()
}

#[no_mangle]
pub extern fn complex_statistics64(vector: &DataVector64) -> Statistics<Complex64> {
    vector.complex_statistics()
}

/// `padding_option` argument is translated to:
/// Returns the vector domain as integer:
///
/// 1. `0` for [`PaddingOption::End`](../../enum.PaddingOption.html)
/// 2. `1` for [`PaddingOption::Surround`](../../enum.PaddingOption.html)
/// 2. `2` for [`PaddingOption::Center`](../../enum.PaddingOption.html)
#[no_mangle]
pub extern fn zero_pad64(vector: Box<DataVector64>, points: usize, padding_option: i32) -> VectorResult<DataVector64> {
    let padding_option = translate_to_padding_option(padding_option);
    convert_vec!(vector.zero_pad(points, padding_option))
}

#[no_mangle]
pub extern fn zero_interleave64(vector: Box<DataVector64>, factor: i32) -> VectorResult<DataVector64> {
    convert_vec!(vector.zero_interleave(factor as u32))
}

#[no_mangle]
pub extern fn diff64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.diff())
}

#[no_mangle]
pub extern fn diff_with_start64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.diff_with_start())
}

#[no_mangle]
pub extern fn cum_sum64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.cum_sum())
}

#[no_mangle]
pub extern fn real_offset64(vector: Box<DataVector64>, value: f64) -> VectorResult<DataVector64> {
    convert_vec!(vector.real_offset(value))
}

#[no_mangle]
pub extern fn real_scale64(vector: Box<DataVector64>, value: f64) -> VectorResult<DataVector64> {
    convert_vec!(vector.real_scale(value))
}

#[no_mangle]
pub extern fn abs64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.abs())
}

#[no_mangle]
pub extern fn sqrt64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.sqrt())
}

#[no_mangle]
pub extern fn square64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.square())
}

#[no_mangle]
pub extern fn root64(vector: Box<DataVector64>, value: f64) -> VectorResult<DataVector64> {
    convert_vec!(vector.root(value))
}

#[no_mangle]
pub extern fn powf64(vector: Box<DataVector64>, value: f64) -> VectorResult<DataVector64> {
    convert_vec!(vector.powf(value))
}

#[no_mangle]
pub extern fn ln64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.ln())
}

#[no_mangle]
pub extern fn exp64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.exp())
}

#[no_mangle]
pub extern fn log64(vector: Box<DataVector64>, value: f64) -> VectorResult<DataVector64> {
    convert_vec!(vector.log(value))
}

#[no_mangle]
pub extern fn expf64(vector: Box<DataVector64>, value: f64) -> VectorResult<DataVector64> {
    convert_vec!(vector.expf(value))
}

#[no_mangle]
pub extern fn to_complex64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.to_complex())
}

#[no_mangle]
pub extern fn sin64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.sin())
}

#[no_mangle]
pub extern fn cos64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.cos())
}

#[no_mangle]
pub extern fn tan64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.tan())
}

#[no_mangle]
pub extern fn asin64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.asin())
}

#[no_mangle]
pub extern fn acos64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.acos())
}

#[no_mangle]
pub extern fn atan64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.tan())
}

#[no_mangle]
pub extern fn sinh64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.sinh())
}
#[no_mangle]
pub extern fn cosh64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.cosh())
}

#[no_mangle]
pub extern fn tanh64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.tanh())
}

#[no_mangle]
pub extern fn asinh64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.asinh())
}

#[no_mangle]
pub extern fn acosh64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.acosh())
}

#[no_mangle]
pub extern fn atanh64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.atanh())
}

#[no_mangle]
pub extern fn wrap64(vector: Box<DataVector64>, value: f64) -> VectorResult<DataVector64> {
    convert_vec!(vector.wrap(value))
}

#[no_mangle]
pub extern fn unwrap64(vector: Box<DataVector64>, value: f64) -> VectorResult<DataVector64> {
    convert_vec!(vector.unwrap(value))
}

#[no_mangle]
pub extern fn swap_halves64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.swap_halves())
}

#[no_mangle]
pub extern fn complex_offset64(vector: Box<DataVector64>, real: f64, imag: f64) -> VectorResult<DataVector64> {
    convert_vec!(vector.complex_offset(Complex64::new(real, imag)))
}

#[no_mangle]
pub extern fn complex_scale64(vector: Box<DataVector64>, real: f64, imag: f64) -> VectorResult<DataVector64> {
    convert_vec!(vector.complex_scale(Complex64::new(real, imag)))
}

#[no_mangle]
pub extern fn complex_divide64(vector: Box<DataVector64>, real: f64, imag: f64) -> VectorResult<DataVector64> {
    convert_vec!(vector.complex_scale(Complex64::new(1.0, 0.0) / Complex64::new(real, imag)))
}

#[no_mangle]
pub extern fn magnitude64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.magnitude())
}

#[no_mangle]
pub extern fn get_magnitude64(vector: Box<DataVector64>, destination: &mut DataVector64) -> i32 {
    convert_void!(vector.get_magnitude(destination))
}

#[no_mangle]
pub extern fn magnitude_squared64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.magnitude_squared())
}

#[no_mangle]
pub extern fn conj64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.conj())
}

#[no_mangle]
pub extern fn to_real64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.to_real())
}

#[no_mangle]
pub extern fn to_imag64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.to_imag())
}

#[no_mangle]
pub extern fn get_real64(vector: Box<DataVector64>, destination: &mut DataVector64) -> i32 {
    convert_void!(vector.get_real(destination))
}

#[no_mangle]
pub extern fn get_imag64(vector: Box<DataVector64>, destination: &mut DataVector64) -> i32 {
    convert_void!(vector.get_imag(destination))
}

#[no_mangle]
pub extern fn phase64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.phase())
}

#[no_mangle]
pub extern fn get_phase64(vector: Box<DataVector64>, destination: &mut DataVector64) -> i32 {
    convert_void!(vector.get_phase(destination))
}

#[no_mangle]
pub extern fn plain_fft64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.plain_fft())
}

#[no_mangle]
pub extern fn plain_sfft64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.plain_sfft())
}

#[no_mangle]
pub extern fn plain_ifft64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.plain_ifft())
}

#[no_mangle]
pub extern fn clone64(vector: Box<DataVector64>) -> Box<DataVector64> {
    vector.clone()
}

#[no_mangle]
pub extern fn multiply_complex_exponential64(vector: Box<DataVector64>, a: f64, b: f64) -> VectorResult<DataVector64> {
    convert_vec!(vector.multiply_complex_exponential(a, b))
}

#[no_mangle]
pub extern fn add_smaller_vector64(vector: Box<DataVector64>, operand: &DataVector64) -> VectorResult<DataVector64> {
    convert_vec!(vector.add_smaller_vector(operand))
}

#[no_mangle]
pub extern fn subtract_smaller_vector64(vector: Box<DataVector64>, operand: &DataVector64) -> VectorResult<DataVector64> {
    convert_vec!(vector.subtract_smaller_vector(operand))
}

#[no_mangle]
pub extern fn divide_smaller_vector64(vector: Box<DataVector64>, operand: &DataVector64) -> VectorResult<DataVector64> {
    convert_vec!(vector.divide_smaller_vector(operand))
}

#[no_mangle]
pub extern fn multiply_smaller_vector64(vector: Box<DataVector64>, operand: &DataVector64) -> VectorResult<DataVector64> {
    convert_vec!(vector.multiply_smaller_vector(operand))
}

#[no_mangle]
pub extern fn get_real_imag64(vector: Box<DataVector64>, real: &mut DataVector64, imag: &mut DataVector64) -> i32 {
    convert_void!(vector.get_real_imag(real, imag))
}

#[no_mangle]
pub extern fn get_mag_phase64(vector: Box<DataVector64>, mag: &mut DataVector64, phase: &mut DataVector64) -> i32 {
    convert_void!(vector.get_mag_phase(mag, phase))
}

#[no_mangle]
pub extern fn set_real_imag64(vector: Box<DataVector64>, real: &DataVector64, imag: &DataVector64) -> VectorResult<DataVector64> {
    convert_vec!(vector.set_real_imag(real, imag))
}

#[no_mangle]
pub extern fn set_mag_phase64(vector: Box<DataVector64>, mag: &DataVector64, phase: &DataVector64) -> VectorResult<DataVector64> {
    convert_vec!(vector.set_mag_phase(mag, phase))
}

#[no_mangle]
pub extern fn split_into64(vector: &DataVector64, targets: *mut Box<DataVector64>, len: usize) -> i32 {
    unsafe {
        let targets = slice::from_raw_parts_mut(targets, len);
        convert_void!(vector.split_into(targets))
    }
}

#[no_mangle]
pub extern fn merge64(vector: Box<DataVector64>, sources: *const Box<DataVector64>, len: usize) -> VectorResult<DataVector64> {
    unsafe {
        let sources = slice::from_raw_parts(sources, len);
        convert_vec!(vector.merge(sources))
    }
}

#[no_mangle]
pub extern fn override_data64(vector: Box<DataVector64>, data: *const f64, len: usize) -> VectorResult<DataVector64> {
    let data = unsafe { slice::from_raw_parts(data, len) };
    convert_vec!(vector.override_data(data))
}

#[no_mangle]
pub extern fn real_statistics_splitted64(vector: &DataVector64, data: *mut Statistics<f64>, len: usize) -> i32 {
    let mut data = unsafe { slice::from_raw_parts_mut(data, len) };
    let stats = vector.real_statistics_splitted(data.len());
    for i in 0..stats.len() {
        data[i] = stats[i];
    }
    
    0
}

#[no_mangle]
pub extern fn complex_statistics_splitted64(vector: &DataVector64, data: *mut Statistics<Complex64>, len: usize) -> i32 {
    let mut data = unsafe { slice::from_raw_parts_mut(data, len) };
    let stats = vector.complex_statistics_splitted(data.len());
    for i in 0..stats.len() {
        data[i] = stats[i];
    }
    
    0
}

#[no_mangle]
pub extern fn fft64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.fft())
}

#[no_mangle]
pub extern fn sfft64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.sfft())
}

#[no_mangle]
pub extern fn ifft64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.ifft())
}

#[no_mangle]
pub extern fn plain_sifft64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.plain_sifft())
}

#[no_mangle]
pub extern fn sifft64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.sifft())
}

#[no_mangle]
pub extern fn mirror64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.mirror())
}

pub extern fn fft_shift64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.fft_shift())
}

pub extern fn ifft_shift64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.ifft_shift())
}

/// `window` argument is translated to:
/// 
/// 1. `0` to [`TriangularWindow`](../../window_functions/struct.TriangularWindow.html)
/// 2. `1` to [`HammingWindow`](../../window_functions/struct.TriangularWindow.html)
#[no_mangle]
pub extern fn apply_window64(vector: Box<DataVector64>, window: i32) -> VectorResult<DataVector64> {
    let window = translate_to_window_function(window);
    convert_vec!(vector.apply_window(window.as_ref()))
}

/// See [`apply_window64`](fn.apply_window64.html) for a description of the `window` parameter.
#[no_mangle]
pub extern fn unapply_window64(vector: Box<DataVector64>, window: i32) -> VectorResult<DataVector64> {
    let window = translate_to_window_function(window);
    convert_vec!(vector.unapply_window(window.as_ref()))
}

/// See [`apply_window64`](fn.apply_window64.html) for a description of the `window` parameter.
#[no_mangle]
pub extern fn windowed_fft64(vector: Box<DataVector64>, window: i32) -> VectorResult<DataVector64> {
    let window = translate_to_window_function(window);
    convert_vec!(vector.windowed_fft(window.as_ref()))
}

/// See [`apply_window64`](fn.apply_window64.html) for a description of the `window` parameter.
#[no_mangle]
pub extern fn windowed_sfft64(vector: Box<DataVector64>, window: i32) -> VectorResult<DataVector64> {
    let window = translate_to_window_function(window);
    convert_vec!(vector.windowed_sfft(window.as_ref()))
}

/// See [`apply_window64`](fn.apply_window64.html) for a description of the `window` parameter.
#[no_mangle]
pub extern fn windowed_ifft64(vector: Box<DataVector64>, window: i32) -> VectorResult<DataVector64> {
    let window = translate_to_window_function(window);
    convert_vec!(vector.windowed_ifft(window.as_ref()))
}

/// See [`apply_window64`](fn.apply_window64.html) for a description of the `window` parameter.
#[no_mangle]
pub extern fn windowed_sifft64(vector: Box<DataVector64>, window: i32) -> VectorResult<DataVector64> {
    let window = translate_to_window_function(window);
    convert_vec!(vector.windowed_sifft(window.as_ref()))
}

/// Creates a window from the function `window` and the void pointer `window_data`. The `window_data` pointer is passed to the `window`
/// function at every call and can be used to store parameters.
#[no_mangle]
pub extern fn apply_custom_window64(
    vector: Box<DataVector64>, 
    window: extern fn(*const c_void, usize, usize) -> f64, 
    window_data: *const c_void,
    is_symmetric: bool) -> VectorResult<DataVector64> {
    unsafe {
        let window = ForeignWindowFunction { window_function: window, window_data: mem::transmute(window_data), is_symmetric: is_symmetric };
        convert_vec!(vector.apply_window(&window))
    }
}

/// See [`apply_custom_window64`](fn.apply_custom_window64.html) for a description of the `window` and `window_data` parameter.
#[no_mangle]
pub extern fn unapply_custom_window64(
    vector: Box<DataVector64>, 
    window: extern fn(*const c_void, usize, usize) -> f64, 
    window_data: *const c_void,
    is_symmetric: bool) -> VectorResult<DataVector64> {
    unsafe {
        let window = ForeignWindowFunction { window_function: window, window_data: mem::transmute(window_data), is_symmetric: is_symmetric };
        convert_vec!(vector.unapply_window(&window))
    }
}

/// See [`apply_custom_window64`](fn.apply_custom_window64.html) for a description of the `window` and `window_data` parameter.
#[no_mangle]
pub extern fn windowed_custom_fft64(
    vector: Box<DataVector64>, 
    window: extern fn(*const c_void, usize, usize) -> f64, 
    window_data: *const c_void,
    is_symmetric: bool) -> VectorResult<DataVector64> {
    unsafe {
        let window = ForeignWindowFunction { window_function: window, window_data: mem::transmute(window_data), is_symmetric: is_symmetric };
        convert_vec!(vector.windowed_fft(&window))
    }
}

/// See [`apply_custom_window64`](fn.apply_custom_window64.html) for a description of the `window` and `window_data` parameter.
#[no_mangle]
pub extern fn windowed_custom_sfft64(
    vector: Box<DataVector64>, 
    window: extern fn(*const c_void, usize, usize) -> f64, 
    window_data: *const c_void,
    is_symmetric: bool) -> VectorResult<DataVector64> {
    unsafe {
        let window = ForeignWindowFunction { window_function: window, window_data: mem::transmute(window_data), is_symmetric: is_symmetric };
        convert_vec!(vector.windowed_sfft(&window))
    }
}

/// See [`apply_custom_window64`](fn.apply_custom_window64.html) for a description of the `window` and `window_data` parameter.
#[no_mangle]
pub extern fn windowed_custom_ifft64(
    vector: Box<DataVector64>, 
    window: extern fn(*const c_void, usize, usize) -> f64, 
    window_data: *const c_void,
    is_symmetric: bool) -> VectorResult<DataVector64> {
    unsafe {
        let window = ForeignWindowFunction { window_function: window, window_data: mem::transmute(window_data), is_symmetric: is_symmetric };
        convert_vec!(vector.windowed_ifft(&window))
    }
}

/// See [`apply_custom_window64`](fn.apply_custom_window64.html) for a description of the `window` and `window_data` parameter.
#[no_mangle]
pub extern fn windowed_custom_sifft64(
    vector: Box<DataVector64>, 
    window: extern fn(*const c_void, usize, usize) -> f64, 
    window_data: *const c_void,
    is_symmetric: bool) -> VectorResult<DataVector64> {
    unsafe {
        let window = ForeignWindowFunction { window_function: window, window_data: mem::transmute(window_data), is_symmetric: is_symmetric };
        convert_vec!(vector.windowed_sifft(&window))
    }
}

#[no_mangle]
pub extern fn reverse64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.reverse())
}

#[no_mangle]
pub extern fn decimatei64(vector: Box<DataVector64>, decimation_factor: u32, delay: u32) -> VectorResult<DataVector64> {
    convert_vec!(vector.decimatei(decimation_factor, delay))
}

#[no_mangle]
pub extern fn prepare_argument64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.prepare_argument())
}

#[no_mangle]
pub extern fn prepare_argument_padded64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.prepare_argument_padded())
}

#[no_mangle]
pub extern fn correlate64(vector: Box<DataVector64>, other: &DataVector64) -> VectorResult<DataVector64> {
    convert_vec!(vector.correlate(other))
}

#[no_mangle]
pub extern fn convolve_vector64(vector: Box<DataVector64>, impulse_response: &DataVector64) -> VectorResult<DataVector64> {
    convert_vec!(vector.convolve_vector(impulse_response))
}

/// Convolves the vector with an impulse response defined by `impulse_response` and the void pointer `impulse_response_data`. 
/// The `impulse_response_data` pointer is passed to the `impulse_response`
/// function at every call and can be used to store parameters.
#[no_mangle]
pub extern fn convolve_real64(vector: Box<DataVector64>, 
    impulse_response: extern fn(*const c_void, f64) -> f64, 
    impulse_response_data: *const c_void,
    is_symmetric: bool,
    ratio: f64,
    len: usize) -> VectorResult<DataVector64> {
    unsafe {
        let function: &RealImpulseResponse<f64> = &ForeignRealConvolutionFunction { conv_function: impulse_response, conv_data: mem::transmute(impulse_response_data), is_symmetric: is_symmetric };
        convert_vec!(vector.convolve(function, ratio, len))
    }
}

/// Convolves the vector with an impulse response defined by `impulse_response` and the void pointer `impulse_response_data`. 
/// The `impulse_response_data` pointer is passed to the `impulse_response`
/// function at every call and can be used to store parameters.
#[no_mangle]
pub extern fn convolve_complex64(vector: Box<DataVector64>, 
    impulse_response: extern fn(*const c_void, f64) -> Complex64, 
    impulse_response_data: *const c_void,
    is_symmetric: bool,
    ratio: f64,
    len: usize) -> VectorResult<DataVector64> {
    unsafe {
        let function: &ComplexImpulseResponse<f64> = &ForeignComplexConvolutionFunction { conv_function: impulse_response, conv_data: mem::transmute(impulse_response_data), is_symmetric: is_symmetric };
        convert_vec!(vector.convolve(function, ratio, len))
    }
}

/// `impulse_response` argument is translated to:
/// 
/// 1. `0` to [`SincFunction`](../../conv_types/struct.SincFunction.html)
/// 2. `1` to [`RaisedCosineFunction`](../../conv_types/struct.RaisedCosineFunction.html)
///
/// `rolloff` is only used if this is a valid parameter for the selected `impulse_response`
#[no_mangle]    
pub extern fn convolve64(vector: Box<DataVector64>, 
    impulse_response: i32,
    rolloff: f64,
    ratio: f64,
    len: usize) -> VectorResult<DataVector64> {
    let function = translate_to_real_convolution_function(impulse_response, rolloff);
    convert_vec!(vector.convolve(function.as_ref(), ratio, len))
}

/// Convolves the vector with an impulse response defined by `frequency_response` and the void pointer `frequency_response_data`. 
/// The `frequency_response_data` pointer is passed to the `frequency_response`
/// function at every call and can be used to store parameters.
#[no_mangle]
pub extern fn multiply_frequency_response_real64(vector: Box<DataVector64>, 
    frequency_response: extern fn(*const c_void, f64) -> f64, 
    frequency_response_data: *const c_void,
    is_symmetric: bool,
    ratio: f64) -> VectorResult<DataVector64> {
    unsafe {
        let function: &RealFrequencyResponse<f64> = &ForeignRealConvolutionFunction { conv_function: frequency_response, conv_data: mem::transmute(frequency_response_data), is_symmetric: is_symmetric };
        convert_vec!(vector.multiply_frequency_response(function, ratio))
    }
}

/// Convolves the vector with an impulse response defined by `frequency_response` and the void pointer `frequency_response_data`. 
/// The `frequency_response` pointer is passed to the `frequency_response`
/// function at every call and can be used to store parameters.
#[no_mangle]
pub extern fn multiply_frequency_response_complex64(vector: Box<DataVector64>, 
    frequency_response: extern fn(*const c_void, f64) -> Complex64, 
    frequency_response_data: *const c_void,
    is_symmetric: bool,
    ratio: f64) -> VectorResult<DataVector64> {
    unsafe {
        let function: &ComplexFrequencyResponse<f64> = &ForeignComplexConvolutionFunction { conv_function: frequency_response, conv_data: mem::transmute(frequency_response_data), is_symmetric: is_symmetric };
        convert_vec!(vector.multiply_frequency_response(function, ratio))
    }
}

/// `frequency_response` argument is translated to:
/// 
/// 1. `0` to [`SincFunction`](../../conv_types/struct.SincFunction.html)
/// 2. `1` to [`RaisedCosineFunction`](../../conv_types/struct.RaisedCosineFunction.html)
///
/// `rolloff` is only used if this is a valid parameter for the selected `frequency_response`
#[no_mangle]    
pub extern fn multiply_frequency_response64(vector: Box<DataVector64>, 
    frequency_response: i32,
    rolloff: f64,
    ratio: f64) -> VectorResult<DataVector64> {
    let function = translate_to_real_frequency_response(frequency_response, rolloff);
    convert_vec!(vector.multiply_frequency_response(function.as_ref(), ratio))
}

/// Convolves the vector with an impulse response defined by `impulse_response` and the void pointer `impulse_response_data`. 
/// The `impulse_response_data` pointer is passed to the `impulse_response`
/// function at every call and can be used to store parameters.
#[no_mangle]
pub extern fn interpolatef_custom64(vector: Box<DataVector64>, 
    impulse_response: extern fn(*const c_void, f64) -> f64, 
    impulse_response_data: *const c_void,
    is_symmetric: bool,
    interpolation_factor: f64,
    delay: f64,
    len: usize) -> VectorResult<DataVector64> {
    unsafe {
        let function: &RealImpulseResponse<f64> = &ForeignRealConvolutionFunction { conv_function: impulse_response, conv_data: mem::transmute(impulse_response_data), is_symmetric: is_symmetric };
        convert_vec!(vector.interpolatef(function, interpolation_factor, delay, len))
    }
}

/// `impulse_response` argument is translated to:
/// 
/// 1. `0` to [`SincFunction`](../../conv_types/struct.SincFunction.html)
/// 2. `1` to [`RaisedCosineFunction`](../../conv_types/struct.RaisedCosineFunction.html)
///
/// `rolloff` is only used if this is a valid parameter for the selected `impulse_response`
#[no_mangle]    
pub extern fn interpolatef64(vector: Box<DataVector64>, 
    impulse_response: i32,
    rolloff: f64,
    interpolation_factor: f64,
    delay: f64,
    len: usize) -> VectorResult<DataVector64> {
    let function = translate_to_real_convolution_function(impulse_response, rolloff);
    convert_vec!(vector.interpolatef(function.as_ref(), interpolation_factor, delay, len))
}

/// Convolves the vector with an impulse response defined by `frequency_response` and the void pointer `frequency_response_data`. 
/// The `frequency_response_data` pointer is passed to the `frequency_response`
/// function at every call and can be used to store parameters.
#[no_mangle]
pub extern fn interpolatei_custom64(vector: Box<DataVector64>, 
    frequency_response: extern fn(*const c_void, f64) -> f64, 
    frequency_response_data: *const c_void,
    is_symmetric: bool,
    interpolation_factor: i32) -> VectorResult<DataVector64> {
    unsafe {
        let function: &RealFrequencyResponse<f64> = &ForeignRealConvolutionFunction { conv_function: frequency_response, conv_data: mem::transmute(frequency_response_data), is_symmetric: is_symmetric };
        convert_vec!(vector.interpolatei(function, interpolation_factor as u32))
    }
}

/// `frequency_response` argument is translated to:
/// 
/// 1. `0` to [`SincFunction`](../../conv_types/struct.SincFunction.html)
/// 2. `1` to [`RaisedCosineFunction`](../../conv_types/struct.RaisedCosineFunction.html)
///
/// `rolloff` is only used if this is a valid parameter for the selected `frequency_response`
#[no_mangle]    
pub extern fn interpolatei64(vector: Box<DataVector64>, 
    frequency_response: i32,
    rolloff: f64,
    interpolation_factor: i32) -> VectorResult<DataVector64> {
    let function = translate_to_real_frequency_response(frequency_response, rolloff);
    convert_vec!(vector.interpolatei(function.as_ref(), interpolation_factor as u32))
}

#[no_mangle]    
pub extern fn interpolate_lin64(vector: Box<DataVector64>, interpolation_factor: f64, delay: f64) -> VectorResult<DataVector64> {
    convert_vec!(vector.interpolate_lin(interpolation_factor, delay))
}

#[no_mangle]    
pub extern fn interpolate_hermite64(vector: Box<DataVector64>, interpolation_factor: f64, delay: f64) -> VectorResult<DataVector64> {
    convert_vec!(vector.interpolate_hermite(interpolation_factor, delay))
}  
