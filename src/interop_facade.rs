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

#[repr(C)]
pub struct VectorResult<T> {
    result_code: i32,
    vector: Box<T>
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
    
#[no_mangle]
pub fn delete_vector32(vector: Box<DataVector32>) {
    drop(vector);
}
 
#[no_mangle]
pub extern fn real_from_constant32(constant: f32, size: usize) -> Box<DataVector32> {
	let vector = Box::new(DataVector32::from_constant(constant, size));
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
pub extern fn real_offset32(vector: Box<DataVector32>, offset: f32) -> VectorResult<DataVector32> {
    convert_vec!(vector.real_offset(offset))
}