//! Clients using other programming languages should use the functions
//! in this mod. Please refer to the other chapters of the help for documentation of the functions.
extern crate basic_dsp_vector;
extern crate num_complex;

pub mod facade32;
pub mod combined_ops32;
pub mod facade64;
pub mod combined_ops64;
use basic_dsp_vector::{VoidResult, SingleBuffer, TransRes, PaddingOption, GenDspVec, ScalarResult,
                       ErrorReason};
use basic_dsp_vector::window_functions::*;
use basic_dsp_vector::conv_types::*;
use basic_dsp_vector::numbers::RealNumber;
use num_complex::Complex;

pub struct InteropVec<T>
    where T: RealNumber
{
    buffer: SingleBuffer<T>,
    vec: GenDspVec<Vec<T>, T>,
}

impl<T> InteropVec<T>
    where T: RealNumber
{
    pub fn convert_vec<F>(mut self, op: F) -> VectorInteropResult<InteropVec<T>>
        where F: Fn(&mut GenDspVec<Vec<T>, T>, &mut SingleBuffer<T>) -> VoidResult
    {
        let result = op(&mut self.vec, &mut self.buffer);
        match result {
            Ok(()) => VectorInteropResult{vector: Box::new(self), result_code: 0},
            Err(res) => {
                VectorInteropResult {
                    vector: Box::new(self),
                    result_code: translate_error(res),
                }
            }
        }
    }

    pub fn trans_vec<F>(self, op: F) -> VectorInteropResult<InteropVec<T>>
        where F: Fn(GenDspVec<Vec<T>, T>, &mut SingleBuffer<T>) -> TransRes<GenDspVec<Vec<T>, T>>
    {
        let mut buffer = self.buffer;
        let vec = self.vec;
        let result = op(vec, &mut buffer);
        match result {
            Ok(vec) => {
                VectorInteropResult {
                    vector: Box::new(InteropVec {
                        vec: vec,
                        buffer: buffer,
                    }),
                    result_code: 0,
                }
            }
            Err((err, vec)) => {
                VectorInteropResult {
                    vector: Box::new(InteropVec {
                        vec: vec,
                        buffer: buffer,
                    }),
                    result_code: translate_error(err),
                }
            }
        }
    }

    pub fn convert_scalar<F, TT>(&self, op: F, default: TT) -> ScalarInteropResult<TT>
        where F: Fn(&GenDspVec<Vec<T>, T>) -> ScalarResult<TT>
    {
        let result = op(&self.vec);
        match result {
            Ok(res) => {
                ScalarInteropResult {
                    result: res,
                    result_code: 0,
                }
            }
            Err(res) => {
                ScalarInteropResult {
                    result: default,
                    result_code: translate_error(res),
                }
            }
        }
    }

    pub fn decompose(self) -> (GenDspVec<Vec<T>, T>, SingleBuffer<T>) {
        (self.vec, self.buffer)
    }
}

pub fn convert_void(result: VoidResult) -> i32 {
    match result {
        Ok(()) => 9,
        Err(err) => translate_error(err),
    }
}

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
        ErrorReason::InputMustHaveTheSameSize => 1,
        ErrorReason::InputMetaDataMustAgree => 2,
        ErrorReason::InputMustBeComplex => 3,
        ErrorReason::InputMustBeReal => 4,
        ErrorReason::InputMustBeInTimeDomain => 5,
        ErrorReason::InputMustBeInFrequencyDomain => 6,
        ErrorReason::InvalidArgumentLength => 7,
        ErrorReason::InputMustBeConjSymmetric => 8,
        ErrorReason::InputMustHaveAnOddLength => 9,
        ErrorReason::ArgumentFunctionMustBeSymmetric => 10,
        ErrorReason::InvalidNumberOfArgumentsForCombinedOp => 11,
        ErrorReason::InputMustNotBeEmpty => 12,
        ErrorReason::InputMustHaveAnEvenLength => 13,
        ErrorReason::TypeCanNotResize => 14,
    }
}

pub fn translate_to_window_function<T>(value: i32) -> Box<WindowFunction<T>>
    where T: RealNumber + 'static
{
    if value == 0 {
        Box::new(TriangularWindow)
    } else {
        Box::new(HammingWindow::default())
    }
}

pub fn translate_to_real_convolution_function<T>(value: i32,
                                                 rolloff: T)
                                                 -> Box<RealImpulseResponse<T>>
    where T: RealNumber + 'static
{
    if value == 0 {
        Box::new(SincFunction::new())
    } else {
        Box::new(RaisedCosineFunction::new(rolloff))
    }
}

pub fn translate_to_real_frequency_response<T>(value: i32,
                                               rolloff: T)
                                               -> Box<RealFrequencyResponse<T>>
    where T: RealNumber + 'static
{
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
        _ => PaddingOption::Center,
    }
}

/// Result of a vector operation. Check the ```result_code```.
#[repr(C)]
pub struct VectorInteropResult<T> {
    /// This value is zero in case of error. All other values mean that an error
    /// occurred and the data in the vector might be unchanged or invalid.
    /// Error codes are described in `translate_error`.
    pub result_code: i32,

    /// A pointer to a data vector.
    pub vector: Box<T>,
}

/// Result of a vector operation. Check the ```result_code```.
#[repr(C)]
pub struct BinaryVectorInteropResult<T> {
    /// This value is zero in case of error. All other values mean that an error
    /// occurred and the data in the vector might be unchanged or invalid.
    /// Error codes are described in `translate_error`.
    pub result_code: i32,

    /// A pointer to a data vector.
    pub vector1: Box<T>,

    /// A pointer to a data vector.
    pub vector2: Box<T>,
}

/// Result of a vector operation. Check the ```result_code```.
#[repr(C)]
pub struct ScalarInteropResult<T>
    where T: Sized
{
    /// This value is zero in case of error. All other values mean that an error
    /// occurred and the data in the vector might be unchanged or invalid.
    /// Error codes are described in `translate_error`.
    pub result_code: i32,

    /// The result
    pub result: T,
}

/// A window function which can be constructed outside this crate.
struct ForeignWindowFunction<T>
    where T: RealNumber
{
    /// The window function
    pub window_function: extern "C" fn(*const std::os::raw::c_void, usize, usize) -> T,

    /// The data which is passed to the window function
    ///
    /// Actual data type is a `const* c_void`, but Rust doesn't allow that because it's
    /// unsafe so we store
    /// it as `usize` and transmute it when necessary. Callers should make very sure safety
    /// is guaranteed.
    pub window_data: usize,

    /// Indicates whether this function is symmetric around 0 or not.
    /// Symmetry is defined as `self.window(x) == self.window(-x)`.
    pub is_symmetric: bool,
}

impl<T> WindowFunction<T> for ForeignWindowFunction<T>
    where T: RealNumber
{
    fn is_symmetric(&self) -> bool {
        self.is_symmetric
    }

    fn window(&self, idx: usize, points: usize) -> T {
        let fun = self.window_function;
        fun(self.window_data as *const std::os::raw::c_void, idx, points)
    }
}

/// A real function which can be constructed outside this crate.
struct ForeignRealConvolutionFunction<T>
    where T: RealNumber
{
    /// The function
    pub conv_function: extern "C" fn(*const std::os::raw::c_void, T) -> T,

    /// The data which is passed to the function.
    ///
    /// Actual data type is a `const* c_void`, but Rust doesn't allow that
    /// because it's unsafe so we store
    /// it as `usize` and transmute it when necessary. Callers should make
    /// very sure safety is guaranteed.
    pub conv_data: usize,

    /// Indicates whether this function is symmetric around 0 or not.
    /// Symmetry is defined as `self.calc(x) == self.calc(-x)`.
    pub is_symmetric: bool,
}

impl<T> RealImpulseResponse<T> for ForeignRealConvolutionFunction<T>
    where T: RealNumber
{
    fn is_symmetric(&self) -> bool {
        self.is_symmetric
    }

    fn calc(&self, x: T) -> T {
        let fun = self.conv_function;
        fun(self.conv_data as *const std::os::raw::c_void, x)
    }
}

impl<T> RealFrequencyResponse<T> for ForeignRealConvolutionFunction<T>
    where T: RealNumber
{
    fn is_symmetric(&self) -> bool {
        self.is_symmetric
    }

    fn calc(&self, x: T) -> T {
        let fun = self.conv_function;
        fun(self.conv_data as *const std::os::raw::c_void, x)
    }
}

/// A complex function which can be constructed outside this crate.
struct ForeignComplexConvolutionFunction<T>
    where T: RealNumber
{
    /// The function
    pub conv_function: extern "C" fn(*const std::os::raw::c_void, T) -> Complex<T>,

    /// The data which is passed to the window function
    ///
    /// Actual data type is a `const* c_void`, but Rust doesn't allow that
    /// because it's unsafe so we store
    /// it as `usize` and transmute it when necessary. Callers should make very
    /// sure safety is guaranteed.
    pub conv_data: usize,

    /// Indicates whether this function is symmetric around 0 or not.
    /// Symmetry is defined as `self.calc(x) == self.calc(-x)`.
    pub is_symmetric: bool,
}

impl<T> ComplexImpulseResponse<T> for ForeignComplexConvolutionFunction<T>
    where T: RealNumber
{
    fn is_symmetric(&self) -> bool {
        self.is_symmetric
    }

    fn calc(&self, x: T) -> Complex<T> {
        let fun = self.conv_function;
        fun(self.conv_data as *const std::os::raw::c_void, x)
    }
}

impl<T> ComplexFrequencyResponse<T> for ForeignComplexConvolutionFunction<T>
    where T: RealNumber
{
    fn is_symmetric(&self) -> bool {
        self.is_symmetric
    }

    /// Indicates whether this function is symmetric around 0 or not.
    /// Symmetry is defined as `self.calc(x) == self.calc(-x)`.
    fn calc(&self, x: T) -> Complex<T> {
        let fun = self.conv_function;
        fun(self.conv_data as *const std::os::raw::c_void, x)
    }
}
