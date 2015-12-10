//! Clients using other programming languages should use the functions
//! in this mod. Please refer to the other chapters of the help for documentation of the functions

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
		DataVector64, 
		RealTimeVector64,
		ComplexTimeVector64, 
		RealFreqVector64,
		ComplexFreqVector64,
		Operation32
	};
use num::complex::Complex32;

/// Result of a vector operation. Check the ```result_code```.
#[repr(C)]
pub struct VectorResult<T> {
    /// This value is zero in case of error. All other values mean that an error
    /// occurred and the data in the vector might be unchanged or invalid. Error codes:
    /// 1. Vectors must have the same size.
    /// all other values are undefined. If you see a value which isn't listed here then
    /// please report a bug.
    pub result_code: i32,
    
    /// A pointer to a data vector.
    pub vector: Box<T>
}
    
fn translate_error(reason: ErrorReason) -> i32 {
    match reason {
        ErrorReason::VectorsMustHaveTheSameSize => 1,
    }
}

macro_rules! convert_vec {
    ($operation: expr) => {
        {
            let result = $operation;
            match result {
                Ok(vec) => VectorResult { result_code: 0, vector: Box::new(vec) },
                Err((res, vec)) => VectorResult { result_code: translate_error(res), vector: Box::new(vec) }
            }
        }
    }
}

macro_rules! convert_void {
    ($operation: expr) => {
        {
            let result = $operation;
            match result {
                Ok(()) => 0,
                Err(res) => translate_error(res) 
            }
        }
    }
}
    
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
pub extern fn zero_pad32(vector: Box<DataVector32>, points: usize) -> VectorResult<DataVector32> {
    convert_vec!(vector.zero_pad(points))
}

#[no_mangle]
pub extern fn zero_interleave32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.zero_interleave())
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
pub extern fn wrap32(vector: Box<DataVector32>, value: f32) -> VectorResult<DataVector32> {
    convert_vec!(vector.wrap(value))
}

#[no_mangle]
pub extern fn unwrap32(vector: Box<DataVector32>, value: f32) -> VectorResult<DataVector32> {
    convert_vec!(vector.unwrap(value))
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
    convert_vec!(vector.complex_abs())
}

#[no_mangle]
pub extern fn get_complex_abs32(vector: Box<DataVector32>, destination: &mut DataVector32) -> i32 {
    convert_void!(vector.get_complex_abs(destination))
}

#[no_mangle]
pub extern fn complex_abs_squared32(vector: Box<DataVector32>) -> VectorResult<DataVector32> {
    convert_vec!(vector.complex_abs_squared())
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