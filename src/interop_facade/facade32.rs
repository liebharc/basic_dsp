use super::*;
#[allow(unused_imports)]
use vector_types::
	{
		DataVectorDomain,
		DataVector,
        VecResult,
        VoidResult,
        ErrorReason,
		GenericVectorOperations,
		RealVectorOperations,
		ComplexVectorOperations,
		TimeDomainOperations,
		FrequencyDomainOperations,
		DataVector32, 
		RealTimeVector32,
		ComplexTimeVector32, 
		RealFreqVector32,
		ComplexFreqVector32,
		Operation,
        Statistics
	};
use num::complex::Complex32;

#[no_mangle]
pub fn delete_vector32(vector: Box<DataVector32>) {
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
/// 0 for time domain
/// 1 for frequency domain
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

#[no_mangle]
pub extern fn zero_pad32(vector: Box<DataVector32>, points: usize) -> VectorResult<DataVector32> {
    convert_vec!(vector.zero_pad(points))
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
pub extern fn real_abs32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.real_abs())
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
pub extern fn complex_abs32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.magnitude())
}

#[no_mangle]
pub extern fn get_complex_abs32(vector: Box<DataVector32>, destination: &mut DataVector32) -> i32 {
    convert_void!(vector.get_magnitude(destination))
}

#[no_mangle]
pub extern fn complex_abs_squared32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
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
pub extern fn plain_ifft32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.plain_ifft())
}

#[no_mangle]
pub extern fn clone32(vector: Box<DataVector32>) -> Box<DataVector32> {
    vector.clone()
}