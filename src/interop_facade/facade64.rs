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
		DataVector64, 
		RealTimeVector64,
		ComplexTimeVector64, 
		RealFreqVector64,
		ComplexFreqVector64,
		Operation,
        Statistics
	};
use num::complex::Complex64;

#[no_mangle]
pub fn delete_vector64(vector: Box<DataVector64>) {
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
/// 0 for time domain
/// 1 for frequency domain
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

#[no_mangle]
pub extern fn zero_pad64(vector: Box<DataVector64>, points: usize) -> VectorResult<DataVector64> {
    convert_vec!(vector.zero_pad(points))
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
pub extern fn real_abs64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
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
pub extern fn complex_abs64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
    convert_vec!(vector.magnitude())
}

#[no_mangle]
pub extern fn get_complex_abs64(vector: Box<DataVector64>, destination: &mut DataVector64) -> i32 {
    convert_void!(vector.get_magnitude(destination))
}

#[no_mangle]
pub extern fn complex_abs_squared64(vector: Box<DataVector64>) -> VectorResult<DataVector64> {
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