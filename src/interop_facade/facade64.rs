// Auto generated code, change facade32.rs and run facade64_create.pl
//! Functions for 64bit floating point number based vectors. Please refer to the other chapters of the help for documentation of the functions.
use super::*;
use vector_types:: {
		DataVectorDomain,
		DataVector,
		GenericVectorOperations,
		RealVectorOperations,
		ComplexVectorOperations,
		TimeDomainOperations,
		FrequencyDomainOperations,
        SymmetricFrequencyDomainOperations,
		DataVector64, 
        Statistics};
use window_functions::WindowFunction;
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
pub extern fn get_points64(vector: &DataVector64) -> usize {
    vector.points()
}

#[no_mangle]
pub extern fn get_delta64(vector: &DataVector64) -> f64 {
    vector.delta()
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
pub extern fn power64(vector: Box<DataVector64>, value: f64) -> VectorResult<DataVector64> {
    convert_vec!(vector.power(value))
}

#[no_mangle]
pub extern fn logn64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.logn())
}

#[no_mangle]
pub extern fn expn64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.expn())
}

#[no_mangle]
pub extern fn log_base64(vector: Box<DataVector64>, value: f64) -> VectorResult<DataVector64> {
    convert_vec!(vector.log_base(value))
}

#[no_mangle]
pub extern fn exp_base64(vector: Box<DataVector64>, value: f64) -> VectorResult<DataVector64> {
    convert_vec!(vector.exp_base(value))
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
pub extern fn complex_conj64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.complex_conj())
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
pub extern fn split_into64(vector: Box<DataVector64>, targets: &mut [Box<DataVector64>]) -> i32 {
    convert_void!(vector.split_into(targets))
}

#[no_mangle]
pub extern fn merge64(vector: Box<DataVector64>, sources: &[Box<DataVector64>]) -> VectorResult<DataVector64> {
    convert_vec!(vector.merge(sources))
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

struct ForeignWindowFunction {
    window_function: extern fn(*const c_void, usize, usize) -> f64,
    // Actual data type is a const* c_void, but Rust doesn't allow that becaues it's usafe so we store
    // it as usize and transmute it when necessary. Callers shoulds make very sure safety is guaranteed.
    window_data: usize,
    
    is_symmetric: bool
}

impl WindowFunction<f64> for ForeignWindowFunction {
    fn is_symmetric(&self) -> bool {
        self.is_symmetric
    }

    fn window(&self, idx: usize, points: usize) -> f64 {
        let fun = self.window_function;
        unsafe { fun(mem::transmute(self.window_data), idx, points) }
    }
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

// pub extern fn complex_data32 isn't implemented to avoid to rely to much on the struct layout of Complex<T>
