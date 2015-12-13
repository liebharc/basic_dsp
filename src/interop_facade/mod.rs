//! Clients using other programming languages should use the functions
//! in this mod. Please refer to the other chapters of the help for documentation of the functions
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

macro_rules! convert_scalar {
    ($operation: expr, $default: expr) => {
        {
            let result = $operation;
            match result {
                Ok(scalar) => ScalarResult { result_code: 0, result: scalar },
                Err(err) => ScalarResult { result_code: translate_error(err), result: $default }
            }
        }
    }
}

pub mod facade32;
pub mod facade64;
use vector_types::ErrorReason;

pub fn translate_error(reason: ErrorReason) -> i32 {
    match reason {
        ErrorReason::VectorsMustHaveTheSameSize => 1,
    }
}

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

/// Result of a vector operation. Check the ```result_code```.
#[repr(C)]
pub struct ScalarResult<T> 
    where T: Sized {
    /// This value is zero in case of error. All other values mean that an error
    /// occurred and the data in the vector might be unchanged or invalid. Error codes:
    /// 1. Vectors must have the same size.
    /// all other values are undefined. If you see a value which isn't listed here then
    /// please report a bug.
    pub result_code: i32,
    
    /// The result
    pub result: T
}