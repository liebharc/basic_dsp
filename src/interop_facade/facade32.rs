//! Functions for 32bit floating point number based vectors. Please refer to the other chapters of the help for documentation of the functions.
use super::*;
use super::super::*;
use window_functions::WindowFunction;
use conv_types::*;
use num::complex::Complex32;
use std::slice;
use std::os::raw::c_void;
use std::mem;

#[no_mangle]
pub extern fn delete_vector32(vector: Box<DataVector32>) {
    drop(vector);
}
 
#[no_mangle]
pub extern fn new32(is_complex: i32, domain: i32, init_value: f32, length: usize, delta: f32) -> Box<DataVector32> {
    let domain = if domain == 0 {
            DataVectorDomain::Time
        }
        else {
            DataVectorDomain::Frequency
        };
        
	let vector = Box::new(DataVector32::new(is_complex != 0, domain, init_value, length, delta));
    vector
}

#[no_mangle]
pub extern fn get_value32(vector: &DataVector32, index: usize) -> f32 {
    vector[index]
}

#[no_mangle]
pub extern fn set_value32(vector: &mut DataVector32, index: usize, value : f32) {
    vector[index] = value;
}

#[no_mangle]
pub extern fn is_complex32(vector: &DataVector32) -> i32 {
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
pub extern fn get_domain32(vector: &DataVector32) -> i32 {
    match vector.domain() {
        DataVectorDomain::Time => 0,
        DataVectorDomain::Frequency => 1,
    }
}

#[no_mangle]
pub extern fn get_len32(vector: &DataVector32) -> usize {
    vector.len()
}

#[no_mangle]
pub extern fn get_points32(vector: &DataVector32) -> usize {
    vector.points()
}

#[no_mangle]
pub extern fn get_delta32(vector: &DataVector32) -> f32 {
    vector.delta()
}

#[no_mangle]
pub extern fn complex_data32(vector: &DataVector32) -> &[Complex32] {
    vector.complex_data()
}

#[no_mangle]
pub extern fn get_allocated_len32(vector: &DataVector32) -> usize {
    vector.allocated_len()
}

#[no_mangle]
pub extern fn add_vector32(vector: Box<DataVector32>, operand: &DataVector32) -> VectorResult<DataVector32> {
    convert_vec!(vector.add_vector(operand))
}

#[no_mangle]
pub extern fn subtract_vector32(vector: Box<DataVector32>, operand: &DataVector32) -> VectorResult<DataVector32> {
    convert_vec!(vector.subtract_vector(operand))
}

#[no_mangle]
pub extern fn divide_vector32(vector: Box<DataVector32>, operand: &DataVector32) -> VectorResult<DataVector32> {
    convert_vec!(vector.divide_vector(operand))
}

#[no_mangle]
pub extern fn multiply_vector32(vector: Box<DataVector32>, operand: &DataVector32) -> VectorResult<DataVector32> {
    convert_vec!(vector.multiply_vector(operand))
}

#[no_mangle]
pub extern fn real_dot_product32(vector: &DataVector32, operand: &DataVector32) -> ScalarResult<f32> {
    convert_scalar!(vector.real_dot_product(operand), 0.0)
}

#[no_mangle]
pub extern fn complex_dot_product32(vector: &DataVector32, operand: &DataVector32) -> ScalarResult<Complex32> {
    convert_scalar!(vector.complex_dot_product(operand), Complex32::new(0.0, 0.0))
}

#[no_mangle]
pub extern fn real_statistics32(vector: &DataVector32) -> Statistics<f32> {
    vector.real_statistics()
}

#[no_mangle]
pub extern fn complex_statistics32(vector: &DataVector32) -> Statistics<Complex32> {
    vector.complex_statistics()
}

/// `padding_option` argument is translated to:
/// Returns the vector domain as integer:
///
/// 1. `0` for [`PaddingOption::End`](../../enum.PaddingOption.html)
/// 2. `1` for [`PaddingOption::Surround`](../../enum.PaddingOption.html)
/// 2. `2` for [`PaddingOption::Center`](../../enum.PaddingOption.html)
#[no_mangle]
pub extern fn zero_pad32(vector: Box<DataVector32>, points: usize, padding_option: i32) -> VectorResult<DataVector32> {
    let padding_option = translate_to_padding_option(padding_option);
    convert_vec!(vector.zero_pad(points, padding_option))
}

#[no_mangle]
pub extern fn zero_interleave32(vector: Box<DataVector32>, factor: i32) -> VectorResult<DataVector32> {
    convert_vec!(vector.zero_interleave(factor as u32))
}

#[no_mangle]
pub extern fn diff32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.diff())
}

#[no_mangle]
pub extern fn diff_with_start32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.diff_with_start())
}

#[no_mangle]
pub extern fn cum_sum32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.cum_sum())
}

#[no_mangle]
pub extern fn real_offset32(vector: Box<DataVector32>, value: f32) -> VectorResult<DataVector32> {
    convert_vec!(vector.real_offset(value))
}

#[no_mangle]
pub extern fn real_scale32(vector: Box<DataVector32>, value: f32) -> VectorResult<DataVector32> {
    convert_vec!(vector.real_scale(value))
}

#[no_mangle]
pub extern fn abs32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.abs())
}

#[no_mangle]
pub extern fn sqrt32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.sqrt())
}

#[no_mangle]
pub extern fn square32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.square())
}

#[no_mangle]
pub extern fn root32(vector: Box<DataVector32>, value: f32) -> VectorResult<DataVector32> {
    convert_vec!(vector.root(value))
}

#[no_mangle]
pub extern fn power32(vector: Box<DataVector32>, value: f32) -> VectorResult<DataVector32> {
    convert_vec!(vector.power(value))
}

#[no_mangle]
pub extern fn logn32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.logn())
}

#[no_mangle]
pub extern fn expn32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.expn())
}

#[no_mangle]
pub extern fn log_base32(vector: Box<DataVector32>, value: f32) -> VectorResult<DataVector32> {
    convert_vec!(vector.log_base(value))
}

#[no_mangle]
pub extern fn exp_base32(vector: Box<DataVector32>, value: f32) -> VectorResult<DataVector32> {
    convert_vec!(vector.exp_base(value))
}

#[no_mangle]
pub extern fn to_complex32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.to_complex())
}

#[no_mangle]
pub extern fn sin32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.sin())
}

#[no_mangle]
pub extern fn cos32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.cos())
}

#[no_mangle]
pub extern fn tan32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.tan())
}

#[no_mangle]
pub extern fn asin32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.asin())
}

#[no_mangle]
pub extern fn acos32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.acos())
}

#[no_mangle]
pub extern fn atan32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.tan())
}

#[no_mangle]
pub extern fn sinh32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.sinh())
}
#[no_mangle]
pub extern fn cosh32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.cosh())
}

#[no_mangle]
pub extern fn tanh32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.tanh())
}

#[no_mangle]
pub extern fn asinh32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.asinh())
}

#[no_mangle]
pub extern fn acosh32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.acosh())
}

#[no_mangle]
pub extern fn atanh32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.atanh())
}

#[no_mangle]
pub extern fn wrap32(vector: Box<DataVector32>, value: f32) -> VectorResult<DataVector32> {
    convert_vec!(vector.wrap(value))
}

#[no_mangle]
pub extern fn unwrap32(vector: Box<DataVector32>, value: f32) -> VectorResult<DataVector32> {
    convert_vec!(vector.unwrap(value))
}

#[no_mangle]
pub extern fn swap_halves32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.swap_halves())
}

#[no_mangle]
pub extern fn complex_offset32(vector: Box<DataVector32>, real: f32, imag: f32) -> VectorResult<DataVector32> {
    convert_vec!(vector.complex_offset(Complex32::new(real, imag)))
}

#[no_mangle]
pub extern fn complex_scale32(vector: Box<DataVector32>, real: f32, imag: f32) -> VectorResult<DataVector32> {
    convert_vec!(vector.complex_scale(Complex32::new(real, imag)))
}

#[no_mangle]
pub extern fn complex_divide32(vector: Box<DataVector32>, real: f32, imag: f32) -> VectorResult<DataVector32> {
    convert_vec!(vector.complex_scale(Complex32::new(1.0, 0.0) / Complex32::new(real, imag)))
}

#[no_mangle]
pub extern fn magnitude32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.magnitude())
}

#[no_mangle]
pub extern fn get_magnitude32(vector: Box<DataVector32>, destination: &mut DataVector32) -> i32 {
    convert_void!(vector.get_magnitude(destination))
}

#[no_mangle]
pub extern fn magnitude_squared32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.magnitude_squared())
}

#[no_mangle]
pub extern fn complex_conj32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.complex_conj())
}

#[no_mangle]
pub extern fn to_real32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.to_real())
}

#[no_mangle]
pub extern fn to_imag32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.to_imag())
}

#[no_mangle]
pub extern fn get_real32(vector: Box<DataVector32>, destination: &mut DataVector32) -> i32 {
    convert_void!(vector.get_real(destination))
}

#[no_mangle]
pub extern fn get_imag32(vector: Box<DataVector32>, destination: &mut DataVector32) -> i32 {
    convert_void!(vector.get_imag(destination))
}

#[no_mangle]
pub extern fn phase32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.phase())
}

#[no_mangle]
pub extern fn get_phase32(vector: Box<DataVector32>, destination: &mut DataVector32) -> i32 {
    convert_void!(vector.get_phase(destination))
}

#[no_mangle]
pub extern fn plain_fft32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.plain_fft())
}

#[no_mangle]
pub extern fn plain_sfft32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.plain_sfft())
}

#[no_mangle]
pub extern fn plain_ifft32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.plain_ifft())
}

#[no_mangle]
pub extern fn clone32(vector: Box<DataVector32>) -> Box<DataVector32> {
    vector.clone()
}

#[no_mangle]
pub extern fn multiply_complex_exponential32(vector: Box<DataVector32>, a: f32, b: f32) -> VectorResult<DataVector32> {
    convert_vec!(vector.multiply_complex_exponential(a, b))
}

#[no_mangle]
pub extern fn add_smaller_vector32(vector: Box<DataVector32>, operand: &DataVector32) -> VectorResult<DataVector32> {
    convert_vec!(vector.add_smaller_vector(operand))
}

#[no_mangle]
pub extern fn subtract_smaller_vector32(vector: Box<DataVector32>, operand: &DataVector32) -> VectorResult<DataVector32> {
    convert_vec!(vector.subtract_smaller_vector(operand))
}

#[no_mangle]
pub extern fn divide_smaller_vector32(vector: Box<DataVector32>, operand: &DataVector32) -> VectorResult<DataVector32> {
    convert_vec!(vector.divide_smaller_vector(operand))
}

#[no_mangle]
pub extern fn multiply_smaller_vector32(vector: Box<DataVector32>, operand: &DataVector32) -> VectorResult<DataVector32> {
    convert_vec!(vector.multiply_smaller_vector(operand))
}

#[no_mangle]
pub extern fn get_real_imag32(vector: Box<DataVector32>, real: &mut DataVector32, imag: &mut DataVector32) -> i32 {
    convert_void!(vector.get_real_imag(real, imag))
}

#[no_mangle]
pub extern fn get_mag_phase32(vector: Box<DataVector32>, mag: &mut DataVector32, phase: &mut DataVector32) -> i32 {
    convert_void!(vector.get_mag_phase(mag, phase))
}

#[no_mangle]
pub extern fn set_real_imag32(vector: Box<DataVector32>, real: &DataVector32, imag: &DataVector32) -> VectorResult<DataVector32> {
    convert_vec!(vector.set_real_imag(real, imag))
}

#[no_mangle]
pub extern fn set_mag_phase32(vector: Box<DataVector32>, mag: &DataVector32, phase: &DataVector32) -> VectorResult<DataVector32> {
    convert_vec!(vector.set_mag_phase(mag, phase))
}

#[no_mangle]
pub extern fn split_into32(vector: Box<DataVector32>, targets: &mut [Box<DataVector32>]) -> i32 {
    convert_void!(vector.split_into(targets))
}

#[no_mangle]
pub extern fn merge32(vector: Box<DataVector32>, sources: &[Box<DataVector32>]) -> VectorResult<DataVector32> {
    convert_vec!(vector.merge(sources))
}

#[no_mangle]
pub extern fn override_data32(vector: Box<DataVector32>, data: *const f32, len: usize) -> VectorResult<DataVector32> {
    let data = unsafe { slice::from_raw_parts(data, len) };
    convert_vec!(vector.override_data(data))
}

#[no_mangle]
pub extern fn real_statistics_splitted32(vector: &DataVector32, data: *mut Statistics<f32>, len: usize) -> i32 {
    let mut data = unsafe { slice::from_raw_parts_mut(data, len) };
    let stats = vector.real_statistics_splitted(data.len());
    for i in 0..stats.len() {
        data[i] = stats[i];
    }
    
    0
}

#[no_mangle]
pub extern fn complex_statistics_splitted32(vector: &DataVector32, data: *mut Statistics<Complex32>, len: usize) -> i32 {
    let mut data = unsafe { slice::from_raw_parts_mut(data, len) };
    let stats = vector.complex_statistics_splitted(data.len());
    for i in 0..stats.len() {
        data[i] = stats[i];
    }
    
    0
}

#[no_mangle]
pub extern fn fft32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.fft())
}

#[no_mangle]
pub extern fn sfft32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.sfft())
}

#[no_mangle]
pub extern fn ifft32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.ifft())
}

#[no_mangle]
pub extern fn plain_sifft32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.plain_sifft())
}

#[no_mangle]
pub extern fn sifft32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.sifft())
}

#[no_mangle]
pub extern fn mirror32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.mirror())
}

pub extern fn fft_shift32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.fft_shift())
}

pub extern fn ifft_shift32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.ifft_shift())
}

/// `window` argument is translated to:
/// 
/// 1. `0` to [`TriangularWindow`](../../window_functions/struct.TriangularWindow.html)
/// 2. `1` to [`HammingWindow`](../../window_functions/struct.TriangularWindow.html)
#[no_mangle]
pub extern fn apply_window32(vector: Box<DataVector32>, window: i32) -> VectorResult<DataVector32> {
    let window = translate_to_window_function(window);
    convert_vec!(vector.apply_window(window.as_ref()))
}

/// See [`apply_window32`](fn.apply_window32.html) for a description of the `window` parameter.
#[no_mangle]
pub extern fn unapply_window32(vector: Box<DataVector32>, window: i32) -> VectorResult<DataVector32> {
    let window = translate_to_window_function(window);
    convert_vec!(vector.unapply_window(window.as_ref()))
}

/// See [`apply_window32`](fn.apply_window32.html) for a description of the `window` parameter.
#[no_mangle]
pub extern fn windowed_fft32(vector: Box<DataVector32>, window: i32) -> VectorResult<DataVector32> {
    let window = translate_to_window_function(window);
    convert_vec!(vector.windowed_fft(window.as_ref()))
}

/// See [`apply_window32`](fn.apply_window32.html) for a description of the `window` parameter.
#[no_mangle]
pub extern fn windowed_sfft32(vector: Box<DataVector32>, window: i32) -> VectorResult<DataVector32> {
    let window = translate_to_window_function(window);
    convert_vec!(vector.windowed_sfft(window.as_ref()))
}

/// See [`apply_window32`](fn.apply_window32.html) for a description of the `window` parameter.
#[no_mangle]
pub extern fn windowed_ifft32(vector: Box<DataVector32>, window: i32) -> VectorResult<DataVector32> {
    let window = translate_to_window_function(window);
    convert_vec!(vector.windowed_ifft(window.as_ref()))
}

/// See [`apply_window32`](fn.apply_window32.html) for a description of the `window` parameter.
#[no_mangle]
pub extern fn windowed_sifft32(vector: Box<DataVector32>, window: i32) -> VectorResult<DataVector32> {
    let window = translate_to_window_function(window);
    convert_vec!(vector.windowed_sifft(window.as_ref()))
}

pub struct ForeignWindowFunction {
    pub window_function: extern fn(*const c_void, usize, usize) -> f32,
    // Actual data type is a const* c_void, but Rust doesn't allow that becaues it's usafe so we store
    // it as usize and transmute it when necessary. Callers shoulds make very sure safety is guaranteed.
    pub window_data: usize,
    
    pub is_symmetric: bool
}

impl WindowFunction<f32> for ForeignWindowFunction {
    fn is_symmetric(&self) -> bool {
        self.is_symmetric
    }

    fn window(&self, idx: usize, points: usize) -> f32 {
        let fun = self.window_function;
        unsafe { fun(mem::transmute(self.window_data), idx, points) }
    }
}

pub struct ForeignRealConvolutionFunction {
    pub conv_function: extern fn(*const c_void, f32) -> f32,
    // Actual data type is a const* c_void, but Rust doesn't allow that becaues it's usafe so we store
    // it as usize and transmute it when necessary. Callers shoulds make very sure safety is guaranteed.
    pub conv_data: usize,
    
    pub is_symmetric: bool
}

impl RealImpulseResponse<f32> for ForeignRealConvolutionFunction {
    fn is_symmetric(&self) -> bool {
        self.is_symmetric
    }

    fn calc(&self, x: f32) -> f32 {
        let fun = self.conv_function;
        unsafe { fun(mem::transmute(self.conv_data), x) }
    }
}

impl RealFrequencyResponse<f32> for ForeignRealConvolutionFunction {
    fn is_symmetric(&self) -> bool {
        self.is_symmetric
    }

    fn calc(&self, x: f32) -> f32 {
        let fun = self.conv_function;
        unsafe { fun(mem::transmute(self.conv_data), x) }
    }
}

pub struct ForeignComplexConvolutionFunction {
    pub conv_function: extern fn(*const c_void, f32) -> Complex32,
    // Actual data type is a const* c_void, but Rust doesn't allow that becaues it's usafe so we store
    // it as usize and transmute it when necessary. Callers shoulds make very sure safety is guaranteed.
    pub conv_data: usize,
    
    pub is_symmetric: bool
}

impl ComplexImpulseResponse<f32> for ForeignComplexConvolutionFunction {
    fn is_symmetric(&self) -> bool {
        self.is_symmetric
    }

    fn calc(&self, x: f32) -> Complex32 {
        let fun = self.conv_function;
        unsafe { fun(mem::transmute(self.conv_data), x) }
    }
}

impl ComplexFrequencyResponse<f32> for ForeignComplexConvolutionFunction {
    fn is_symmetric(&self) -> bool {
        self.is_symmetric
    }

    fn calc(&self, x: f32) -> Complex32 {
        let fun = self.conv_function;
        unsafe { fun(mem::transmute(self.conv_data), x) }
    }
}

/// Creates a window from the function `window` and the void pointer `window_data`. The `window_data` pointer is passed to the `window`
/// function at every call and can be used to store parameters.
#[no_mangle]
pub extern fn apply_custom_window32(
    vector: Box<DataVector32>, 
    window: extern fn(*const c_void, usize, usize) -> f32, 
    window_data: *const c_void,
    is_symmetric: bool) -> VectorResult<DataVector32> {
    unsafe {
        let window = ForeignWindowFunction { window_function: window, window_data: mem::transmute(window_data), is_symmetric: is_symmetric };
        convert_vec!(vector.apply_window(&window))
    }
}

/// See [`apply_custom_window32`](fn.apply_custom_window32.html) for a description of the `window` and `window_data` parameter.
#[no_mangle]
pub extern fn unapply_custom_window32(
    vector: Box<DataVector32>, 
    window: extern fn(*const c_void, usize, usize) -> f32, 
    window_data: *const c_void,
    is_symmetric: bool) -> VectorResult<DataVector32> {
    unsafe {
        let window = ForeignWindowFunction { window_function: window, window_data: mem::transmute(window_data), is_symmetric: is_symmetric };
        convert_vec!(vector.unapply_window(&window))
    }
}

/// See [`apply_custom_window32`](fn.apply_custom_window32.html) for a description of the `window` and `window_data` parameter.
#[no_mangle]
pub extern fn windowed_custom_fft32(
    vector: Box<DataVector32>, 
    window: extern fn(*const c_void, usize, usize) -> f32, 
    window_data: *const c_void,
    is_symmetric: bool) -> VectorResult<DataVector32> {
    unsafe {
        let window = ForeignWindowFunction { window_function: window, window_data: mem::transmute(window_data), is_symmetric: is_symmetric };
        convert_vec!(vector.windowed_fft(&window))
    }
}

/// See [`apply_custom_window32`](fn.apply_custom_window32.html) for a description of the `window` and `window_data` parameter.
#[no_mangle]
pub extern fn windowed_custom_ifft32(
    vector: Box<DataVector32>, 
    window: extern fn(*const c_void, usize, usize) -> f32, 
    window_data: *const c_void,
    is_symmetric: bool) -> VectorResult<DataVector32> {
    unsafe {
        let window = ForeignWindowFunction { window_function: window, window_data: mem::transmute(window_data), is_symmetric: is_symmetric };
        convert_vec!(vector.windowed_ifft(&window))
    }
}

/// See [`apply_custom_window32`](fn.apply_custom_window32.html) for a description of the `window` and `window_data` parameter.
#[no_mangle]
pub extern fn windowed_custom_sifft32(
    vector: Box<DataVector32>, 
    window: extern fn(*const c_void, usize, usize) -> f32, 
    window_data: *const c_void,
    is_symmetric: bool) -> VectorResult<DataVector32> {
    unsafe {
        let window = ForeignWindowFunction { window_function: window, window_data: mem::transmute(window_data), is_symmetric: is_symmetric };
        convert_vec!(vector.windowed_sifft(&window))
    }
}

#[no_mangle]
pub extern fn reverse32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.reverse())
}

#[no_mangle]
pub extern fn decimatei32(vector: Box<DataVector32>, decimation_factor: u32, delay: u32) -> VectorResult<DataVector32> {
    convert_vec!(vector.decimatei(decimation_factor, delay))
}

#[no_mangle]
pub extern fn prepare_argument32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.prepare_argument())
}

#[no_mangle]
pub extern fn prepare_argument_padded32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.prepare_argument_padded())
}

#[no_mangle]
pub extern fn correlate32(vector: Box<DataVector32>, other: &DataVector32) -> VectorResult<DataVector32> {
    convert_vec!(vector.correlate(other))
}

#[no_mangle]
pub extern fn convolve_vector32(vector: Box<DataVector32>, impulse_response: &DataVector32) -> VectorResult<DataVector32> {
    convert_vec!(vector.convolve_vector(impulse_response))
}

/// Convolves the vector with an impulse response defined by `impulse_response` and the void pointer `impulse_response_data`. 
/// The `impulse_response_data` pointer is passed to the `impulse_response`
/// function at every call and can be used to store parameters.
#[no_mangle]
pub extern fn convolve_real32(vector: Box<DataVector32>, 
    impulse_response: extern fn(*const c_void, f32) -> f32, 
    impulse_response_data: *const c_void,
    is_symmetric: bool,
    ratio: f32,
    len: usize) -> VectorResult<DataVector32> {
    unsafe {
        let function: &RealImpulseResponse<f32> = &ForeignRealConvolutionFunction { conv_function: impulse_response, conv_data: mem::transmute(impulse_response_data), is_symmetric: is_symmetric };
        convert_vec!(vector.convolve(function, ratio, len))
    }
}

/// Convolves the vector with an impulse response defined by `impulse_response` and the void pointer `impulse_response_data`. 
/// The `impulse_response_data` pointer is passed to the `impulse_response`
/// function at every call and can be used to store parameters.
#[no_mangle]
pub extern fn convolve_complex32(vector: Box<DataVector32>, 
    impulse_response: extern fn(*const c_void, f32) -> Complex32, 
    impulse_response_data: *const c_void,
    is_symmetric: bool,
    ratio: f32,
    len: usize) -> VectorResult<DataVector32> {
    unsafe {
        let function: &ComplexImpulseResponse<f32> = &ForeignComplexConvolutionFunction { conv_function: impulse_response, conv_data: mem::transmute(impulse_response_data), is_symmetric: is_symmetric };
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
pub extern fn convolve32(vector: Box<DataVector32>, 
    impulse_response: i32,
    rolloff: f32,
    ratio: f32,
    len: usize) -> VectorResult<DataVector32> {
    let function = translate_to_real_convolution_function(impulse_response, rolloff);
    convert_vec!(vector.convolve(function.as_ref(), ratio, len))
}

/// Convolves the vector with an impulse response defined by `frequency_response` and the void pointer `frequency_response_data`. 
/// The `frequency_response_data` pointer is passed to the `frequency_response`
/// function at every call and can be used to store parameters.
#[no_mangle]
pub extern fn multiply_frequency_response_real32(vector: Box<DataVector32>, 
    frequency_response: extern fn(*const c_void, f32) -> f32, 
    frequency_response_data: *const c_void,
    is_symmetric: bool,
    ratio: f32) -> VectorResult<DataVector32> {
    unsafe {
        let function: &RealFrequencyResponse<f32> = &ForeignRealConvolutionFunction { conv_function: frequency_response, conv_data: mem::transmute(frequency_response_data), is_symmetric: is_symmetric };
        convert_vec!(vector.multiply_frequency_response(function, ratio))
    }
}

/// Convolves the vector with an impulse response defined by `frequency_response` and the void pointer `frequency_response_data`. 
/// The `frequency_response` pointer is passed to the `frequency_response`
/// function at every call and can be used to store parameters.
#[no_mangle]
pub extern fn multiply_frequency_response_complex32(vector: Box<DataVector32>, 
    frequency_response: extern fn(*const c_void, f32) -> Complex32, 
    frequency_response_data: *const c_void,
    is_symmetric: bool,
    ratio: f32) -> VectorResult<DataVector32> {
    unsafe {
        let function: &ComplexFrequencyResponse<f32> = &ForeignComplexConvolutionFunction { conv_function: frequency_response, conv_data: mem::transmute(frequency_response_data), is_symmetric: is_symmetric };
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
pub extern fn multiply_frequency_response32(vector: Box<DataVector32>, 
    frequency_response: i32,
    rolloff: f32,
    ratio: f32) -> VectorResult<DataVector32> {
    let function = translate_to_real_frequency_response(frequency_response, rolloff);
    convert_vec!(vector.multiply_frequency_response(function.as_ref(), ratio))
}

/// Convolves the vector with an impulse response defined by `impulse_response` and the void pointer `impulse_response_data`. 
/// The `impulse_response_data` pointer is passed to the `impulse_response`
/// function at every call and can be used to store parameters.
#[no_mangle]
pub extern fn interpolatef_custom32(vector: Box<DataVector32>, 
    impulse_response: extern fn(*const c_void, f32) -> f32, 
    impulse_response_data: *const c_void,
    is_symmetric: bool,
    interpolation_factor: f32,
    delay: f32,
    len: usize) -> VectorResult<DataVector32> {
    unsafe {
        let function: &RealImpulseResponse<f32> = &ForeignRealConvolutionFunction { conv_function: impulse_response, conv_data: mem::transmute(impulse_response_data), is_symmetric: is_symmetric };
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
pub extern fn interpolatef32(vector: Box<DataVector32>, 
    impulse_response: i32,
    rolloff: f32,
    interpolation_factor: f32,
    delay: f32,
    len: usize) -> VectorResult<DataVector32> {
    let function = translate_to_real_convolution_function(impulse_response, rolloff);
    convert_vec!(vector.interpolatef(function.as_ref(), interpolation_factor, delay, len))
}

/// Convolves the vector with an impulse response defined by `frequency_response` and the void pointer `frequency_response_data`. 
/// The `frequency_response_data` pointer is passed to the `frequency_response`
/// function at every call and can be used to store parameters.
#[no_mangle]
pub extern fn interpolatei_custom32(vector: Box<DataVector32>, 
    frequency_response: extern fn(*const c_void, f32) -> f32, 
    frequency_response_data: *const c_void,
    is_symmetric: bool,
    interpolation_factor: i32) -> VectorResult<DataVector32> {
    unsafe {
        let function: &RealFrequencyResponse<f32> = &ForeignRealConvolutionFunction { conv_function: frequency_response, conv_data: mem::transmute(frequency_response_data), is_symmetric: is_symmetric };
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
pub extern fn interpolatei32(vector: Box<DataVector32>, 
    frequency_response: i32,
    rolloff: f32,
    interpolation_factor: i32) -> VectorResult<DataVector32> {
    let function = translate_to_real_frequency_response(frequency_response, rolloff);
    convert_vec!(vector.interpolatei(function.as_ref(), interpolation_factor as u32))
}

#[no_mangle]    
pub extern fn interpolate_lin32(vector: Box<DataVector32>, interpolation_factor: f32, delay: f32) -> VectorResult<DataVector32> {
    convert_vec!(vector.interpolate_lin(interpolation_factor, delay))
}

#[no_mangle]    
pub extern fn interpolate_hermite32(vector: Box<DataVector32>, interpolation_factor: f32, delay: f32) -> VectorResult<DataVector32> {
    convert_vec!(vector.interpolate_hermite(interpolation_factor, delay))
}  