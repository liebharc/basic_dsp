//! Clients using other programming languages should use the functions
//! in this mod. Please refer to the other chapters of the help for documentation of the functions.
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

macro_rules! convert_bin_vec {
    ($operation: expr) => {
        {
            let result = $operation;
            match result {
                Ok((vec1, vec2)) => BinaryVectorResult { result_code: 0, vector1: Box::new(vec1), vector2: Box::new(vec2) },
                Err((res, vec1, vec2)) => BinaryVectorResult { result_code: translate_error(res), vector1: Box::new(vec1), vector2: Box::new(vec2) }
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
use vector_types::{
    PaddingOption,
    ErrorReason};
use window_functions::*;
use conv_types::*;
use RealNumber;

/// Error codes:
///
/// 1. VectorsMustHaveTheSameSize
/// 2. VectorMetaDataMustAgree
/// 3. VectorMustBeComplex
/// 4. VectorMustBeReal
/// 5. VectorMustBeInTimeDomain
/// 6. VectorMustBeInFrquencyDomain
/// 7. InvalidArgumentLength
/// 8. VectorMustBeConjSymmetric
/// 9. VectorMustHaveAnOddLength
/// 10. ArgumentFunctionMustBeSymmetric
/// 11. InvalidNumberOfArgumentsForCombinedOp
/// 12. VectorMustNotBeEmpty
/// 
/// all other values are undefined. If you see a value which isn't listed here then
/// please report a bug.
pub fn translate_error(reason: ErrorReason) -> i32 {
    match reason {
        ErrorReason::VectorsMustHaveTheSameSize => 1,
        ErrorReason::VectorMetaDataMustAgree => 2,
        ErrorReason::VectorMustBeComplex => 3,
        ErrorReason::VectorMustBeReal => 4,
        ErrorReason::VectorMustBeInTimeDomain => 5,
        ErrorReason::VectorMustBeInFrquencyDomain => 6,
        ErrorReason::InvalidArgumentLength => 7,
        ErrorReason::VectorMustBeConjSymmetric => 8,
        ErrorReason::VectorMustHaveAnOddLength => 9,
        ErrorReason::ArgumentFunctionMustBeSymmetric => 10,
        ErrorReason::InvalidNumberOfArgumentsForCombinedOp => 11,
        ErrorReason::VectorMustNotBeEmpty => 12,
    }
}

pub fn translate_to_window_function<T>(value: i32) -> Box<WindowFunction<T>> 
    where T: RealNumber + 'static {
    if value == 0 {
        Box::new(TriangularWindow)
    } else {
        Box::new(HammingWindow::default())
    }
}

pub fn translate_to_real_convolution_function<T>(value: i32, rolloff: T) -> Box<RealImpulseResponse<T>> 
    where T: RealNumber + 'static {
    if value == 0 {
        Box::new(SincFunction::new())
    } else {
        Box::new(RaisedCosineFunction::new(rolloff))
    }
}

pub fn translate_to_real_frequency_response<T>(value: i32, rolloff: T) -> Box<RealFrequencyResponse<T>> 
    where T: RealNumber + 'static {
    if value == 0 {
        Box::new(SincFunction::new())
    } else {
        Box::new(RaisedCosineFunction::new(rolloff))
    }
}

pub fn translate_to_padding_option(value: i32) -> PaddingOption {
    match value {
        0 => PaddingOption::End,
        1 => PaddingOption::Surround,
        _ => PaddingOption::Center
    }
}

/// Result of a vector operation. Check the ```result_code```.
#[repr(C)]
pub struct VectorResult<T> {
    /// This value is zero in case of error. All other values mean that an error
    /// occurred and the data in the vector might be unchanged or invalid. Error codes are described in `translate_error`.
    pub result_code: i32,
    
    /// A pointer to a data vector.
    pub vector: Box<T>
}

/// Result of a vector operation. Check the ```result_code```.
#[repr(C)]
pub struct BinaryVectorResult<T> {
    /// This value is zero in case of error. All other values mean that an error
    /// occurred and the data in the vector might be unchanged or invalid. Error codes are described in `translate_error`.
    pub result_code: i32,
    
    /// A pointer to a data vector.
    pub vector1: Box<T>,
    
    /// A pointer to a data vector.
    pub vector2: Box<T>
}

/// Result of a vector operation. Check the ```result_code```.
#[repr(C)]
pub struct ScalarResult<T> 
    where T: Sized {
    /// This value is zero in case of error. All other values mean that an error
    /// occurred and the data in the vector might be unchanged or invalid. Error codes are described in `translate_error`.
    pub result_code: i32,
    
    /// The result
    pub result: T
}