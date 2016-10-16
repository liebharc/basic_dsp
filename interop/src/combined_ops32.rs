//! Functions for 32bit floating point number based vectors. Please refer to the other chapters of the help for documentation of the functions.
use super::*;
use basic_dsp_vector::*;
use basic_dsp_vector::combined_ops::*;
use num::complex::Complex32;
use std::sync::Arc;

pub type VecBuf = InteropVec<f32>;

pub type VecBox = Box<InteropVec<f32>>;

pub type PreparedOp1F32 = PreparedOperation1<f32,
                                             RealOrComplexData,
                                             TimeOrFrequencyData,
                                             RealOrComplexData,
                                             TimeOrFrequencyData>;

pub type PreparedOp2F32 = PreparedOperation2<f32,
                                             RealOrComplexData,
                                             TimeOrFrequencyData,
                                             RealOrComplexData,
                                             TimeOrFrequencyData,
                                             RealOrComplexData,
                                             TimeOrFrequencyData,
                                             RealOrComplexData,
                                             TimeOrFrequencyData>;

/// Prepares an operation.
/// multi_ops1 will not be made available in for interop since the same functionality
/// can be created with prepared ops, and internally this is what this lib does too.
#[no_mangle]
pub extern "C" fn prepared_ops1_f32() -> Box<PreparedOp1F32> {
    Box::new(prepare32_1(RealOrComplexData { is_complex_current: false },
                         TimeOrFrequencyData { domain_current: DataDomain::Time }))
}

/// Prepares an operation.
/// multi_ops2 will not be made available in for interop since the same functionality
/// can be created with prepared ops, and internally this is what this lib does too.
#[no_mangle]
pub extern "C" fn prepared_ops2_f32() -> Box<PreparedOp2F32> {
    Box::new(prepare32_2(RealOrComplexData { is_complex_current: false },
                         TimeOrFrequencyData { domain_current: DataDomain::Time },
                         RealOrComplexData { is_complex_current: false },
                         TimeOrFrequencyData { domain_current: DataDomain::Time }))
}

/// Prepares an operation.
/// multi_ops1 will not be made available in for interop since the same functionality
/// can be created with prepared ops, and internally this is what this lib does too.
#[no_mangle]
pub extern "C" fn extend_prepared_ops1_f32(ops: Box<PreparedOp1F32>) -> Box<PreparedOp2F32> {
    Box::new(ops.extend(RealOrComplexData { is_complex_current: false },
                        TimeOrFrequencyData { domain_current: DataDomain::Time }))
}

#[no_mangle]
pub extern "C" fn exec_prepared_ops1_f32(ops: &PreparedOp1F32,
                                         v: Box<VecBuf>)
                                         -> VectorInteropResult<VecBuf> {
    v.trans_vec(|v, b| ops.exec(b, v))
}

#[no_mangle]
pub extern "C" fn exec_prepared_ops2_f32(ops: &PreparedOp2F32,
                                         v1: Box<VecBuf>,
                                         v2: Box<VecBuf>)
                                         -> BinaryVectorInteropResult<VecBuf> {
    let (v1, mut b1) = v1.decompose();
    let (v2, b2) = v2.decompose();
    let result = ops.exec(&mut b1, v1, v2);
    match result {
        Ok((r1, r2)) => {
            let r1 = VecBuf {
                vec: r1,
                buffer: b1,
            };
            let r2 = VecBuf {
                vec: r2,
                buffer: b2,
            };
            BinaryVectorInteropResult {
                result_code: 0,
                vector1: Box::new(r1),
                vector2: Box::new(r2),
            }
        }
        Err((reason, r1, r2)) => {
            let r1 = VecBuf {
                vec: r1,
                buffer: b1,
            };
            let r2 = VecBuf {
                vec: r2,
                buffer: b2,
            };
            BinaryVectorInteropResult {
                result_code: translate_error(reason),
                vector1: Box::new(r1),
                vector2: Box::new(r2),
            }
        }
    }
}

// ----------------------------------------------
// PreparedOp1F32
// ----------------------------------------------
#[no_mangle]
pub extern "C" fn add_real_ops1_f32(ops: &mut PreparedOp1F32, arg: usize, value: f32) {
    ops.add_enum_op(Operation::AddReal(arg, value))
}

#[no_mangle]
pub extern "C" fn multiply_real_ops1_f32(ops: &mut PreparedOp1F32, arg: usize, value: f32) {
    ops.add_enum_op(Operation::MultiplyReal(arg, value))
}

#[no_mangle]
pub extern "C" fn abs_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::Abs(arg))
}

#[no_mangle]
pub extern "C" fn to_complex_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::ToComplex(arg))
}

#[no_mangle]
pub extern "C" fn add_complex_ops1_f32(ops: &mut PreparedOp1F32, arg: usize, re: f32, im: f32) {
    ops.add_enum_op(Operation::AddComplex(arg, Complex32::new(re, im)))
}

#[no_mangle]
pub extern "C" fn multiply_complex_ops1_f32(ops: &mut PreparedOp1F32,
                                            arg: usize,
                                            re: f32,
                                            im: f32) {
    ops.add_enum_op(Operation::MultiplyComplex(arg, Complex32::new(re, im)))
}

#[no_mangle]
pub extern "C" fn magnitude_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::Magnitude(arg))
}

#[no_mangle]
pub extern "C" fn magnitude_squared_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::MagnitudeSquared(arg))
}

#[no_mangle]
pub extern "C" fn complex_conj_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::ComplexConj(arg))
}

#[no_mangle]
pub extern "C" fn to_real_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::ToReal(arg))
}

#[no_mangle]
pub extern "C" fn to_imag_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::ToImag(arg))
}

#[no_mangle]
pub extern "C" fn phase_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::Phase(arg))
}

#[no_mangle]
pub extern "C" fn multiply_complex_exponential_ops1_f32(ops: &mut PreparedOp1F32,
                                                        arg: usize,
                                                        a: f32,
                                                        b: f32) {
    ops.add_enum_op(Operation::MultiplyComplexExponential(arg, a, b))
}

#[no_mangle]
pub extern "C" fn add_vector_ops1_f32(ops: &mut PreparedOp1F32, arg: usize, other: usize) {
    ops.add_enum_op(Operation::AddVector(arg, other))
}

#[no_mangle]
pub extern "C" fn mul_vector_ops1_f32(ops: &mut PreparedOp1F32, arg: usize, other: usize) {
    ops.add_enum_op(Operation::MulVector(arg, other))
}

#[no_mangle]
pub extern "C" fn sub_vector_ops1_f32(ops: &mut PreparedOp1F32, arg: usize, other: usize) {
    ops.add_enum_op(Operation::SubVector(arg, other))
}

#[no_mangle]
pub extern "C" fn div_vector_ops1_f32(ops: &mut PreparedOp1F32, arg: usize, other: usize) {
    ops.add_enum_op(Operation::DivVector(arg, other))
}

#[no_mangle]
pub extern "C" fn square_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::Square(arg))
}

#[no_mangle]
pub extern "C" fn sqrt_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::Sqrt(arg))
}

#[no_mangle]
pub extern "C" fn root_ops1_f32(ops: &mut PreparedOp1F32, arg: usize, value: f32) {
    ops.add_enum_op(Operation::Root(arg, value))
}

#[no_mangle]
pub extern "C" fn powf_ops1_f32(ops: &mut PreparedOp1F32, arg: usize, value: f32) {
    ops.add_enum_op(Operation::Powf(arg, value))
}

#[no_mangle]
pub extern "C" fn ln_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::Ln(arg))
}

#[no_mangle]
pub extern "C" fn exp_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::Exp(arg))
}

#[no_mangle]
pub extern "C" fn log_ops1_f32(ops: &mut PreparedOp1F32, arg: usize, value: f32) {
    ops.add_enum_op(Operation::Log(arg, value))
}

#[no_mangle]
pub extern "C" fn expf_ops1_f32(ops: &mut PreparedOp1F32, arg: usize, value: f32) {
    ops.add_enum_op(Operation::Expf(arg, value))
}

#[no_mangle]
pub extern "C" fn sin_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::Sin(arg))
}

#[no_mangle]
pub extern "C" fn cos_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::Cos(arg))
}

#[no_mangle]
pub extern "C" fn tan_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::Tan(arg))
}

#[no_mangle]
pub extern "C" fn asin_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::ASin(arg))
}

#[no_mangle]
pub extern "C" fn acos_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::ACos(arg))
}

#[no_mangle]
pub extern "C" fn atan_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::ATan(arg))
}

#[no_mangle]
pub extern "C" fn sinh_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::Sinh(arg))
}

#[no_mangle]
pub extern "C" fn cosh_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::Cosh(arg))
}

#[no_mangle]
pub extern "C" fn tanh_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::Tanh(arg))
}

#[no_mangle]
pub extern "C" fn asinh_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::ASinh(arg))
}

#[no_mangle]
pub extern "C" fn acosh_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::ACosh(arg))
}

#[no_mangle]
pub extern "C" fn atanh_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::ATanh(arg))
}

#[no_mangle]
pub extern "C" fn clone_from_ops1_f32(ops: &mut PreparedOp1F32, arg: usize, source: usize) {
    ops.add_enum_op(Operation::CloneFrom(arg, source))
}

#[no_mangle]
pub extern "C" fn add_points_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::AddPoints(arg))
}

#[no_mangle]
pub extern "C" fn sub_points_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::SubPoints(arg))
}

#[no_mangle]
pub extern "C" fn mul_points_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::MulPoints(arg))
}

#[no_mangle]
pub extern "C" fn div_points_ops1_f32(ops: &mut PreparedOp1F32, arg: usize) {
    ops.add_enum_op(Operation::DivPoints(arg))
}

#[no_mangle]
pub extern "C" fn map_real_ops1_f32(ops: &mut PreparedOp1F32,
                                    arg: usize,
                                    map: extern "C" fn(f32, usize) -> f32) {
    ops.add_enum_op(Operation::MapReal(arg, Arc::new(move |v, i| map(v, i))))
}

#[no_mangle]
pub extern "C" fn map_complex_ops1_f32(ops: &mut PreparedOp1F32,
                                       arg: usize,
                                       map: extern "C" fn(Complex32, usize) -> Complex32) {
    ops.add_enum_op(Operation::MapComplex(arg, Arc::new(move |v, i| map(v, i))))
}

#[no_mangle]
pub extern "C" fn delete_ops1_f32(vector: Box<PreparedOp1F32>) {
    drop(vector);
}

// ----------------------------------------------
// PreparedOp2F32
// ----------------------------------------------
#[no_mangle]
pub extern "C" fn add_real_ops2_f32(ops: &mut PreparedOp2F32, arg: usize, value: f32) {
    ops.add_enum_op(Operation::AddReal(arg, value))
}

#[no_mangle]
pub extern "C" fn multiply_real_ops2_f32(ops: &mut PreparedOp2F32, arg: usize, value: f32) {
    ops.add_enum_op(Operation::MultiplyReal(arg, value))
}

#[no_mangle]
pub extern "C" fn abs_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::Abs(arg))
}

#[no_mangle]
pub extern "C" fn to_complex_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::ToComplex(arg))
}

#[no_mangle]
pub extern "C" fn add_complex_ops2_f32(ops: &mut PreparedOp2F32, arg: usize, re: f32, im: f32) {
    ops.add_enum_op(Operation::AddComplex(arg, Complex32::new(re, im)))
}

#[no_mangle]
pub extern "C" fn multiply_complex_ops2_f32(ops: &mut PreparedOp2F32,
                                            arg: usize,
                                            re: f32,
                                            im: f32) {
    ops.add_enum_op(Operation::MultiplyComplex(arg, Complex32::new(re, im)))
}

#[no_mangle]
pub extern "C" fn magnitude_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::Magnitude(arg))
}

#[no_mangle]
pub extern "C" fn magnitude_squared_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::MagnitudeSquared(arg))
}

#[no_mangle]
pub extern "C" fn complex_conj_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::ComplexConj(arg))
}

#[no_mangle]
pub extern "C" fn to_real_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::ToReal(arg))
}

#[no_mangle]
pub extern "C" fn to_imag_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::ToImag(arg))
}

#[no_mangle]
pub extern "C" fn phase_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::Phase(arg))
}

#[no_mangle]
pub extern "C" fn multiply_complex_exponential_ops2_f32(ops: &mut PreparedOp2F32,
                                                        arg: usize,
                                                        a: f32,
                                                        b: f32) {
    ops.add_enum_op(Operation::MultiplyComplexExponential(arg, a, b))
}

#[no_mangle]
pub extern "C" fn add_vector_ops2_f32(ops: &mut PreparedOp2F32, arg: usize, other: usize) {
    ops.add_enum_op(Operation::AddVector(arg, other))
}

#[no_mangle]
pub extern "C" fn mul_vector_ops2_f32(ops: &mut PreparedOp2F32, arg: usize, other: usize) {
    ops.add_enum_op(Operation::MulVector(arg, other))
}

#[no_mangle]
pub extern "C" fn sub_vector_ops2_f32(ops: &mut PreparedOp2F32, arg: usize, other: usize) {
    ops.add_enum_op(Operation::SubVector(arg, other))
}

#[no_mangle]
pub extern "C" fn div_vector_ops2_f32(ops: &mut PreparedOp2F32, arg: usize, other: usize) {
    ops.add_enum_op(Operation::DivVector(arg, other))
}

#[no_mangle]
pub extern "C" fn square_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::Square(arg))
}

#[no_mangle]
pub extern "C" fn sqrt_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::Sqrt(arg))
}

#[no_mangle]
pub extern "C" fn root_ops2_f32(ops: &mut PreparedOp2F32, arg: usize, value: f32) {
    ops.add_enum_op(Operation::Root(arg, value))
}

#[no_mangle]
pub extern "C" fn powf_ops2_f32(ops: &mut PreparedOp2F32, arg: usize, value: f32) {
    ops.add_enum_op(Operation::Powf(arg, value))
}

#[no_mangle]
pub extern "C" fn ln_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::Ln(arg))
}

#[no_mangle]
pub extern "C" fn exp_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::Exp(arg))
}

#[no_mangle]
pub extern "C" fn log_ops2_f32(ops: &mut PreparedOp2F32, arg: usize, value: f32) {
    ops.add_enum_op(Operation::Log(arg, value))
}

#[no_mangle]
pub extern "C" fn expf_ops2_f32(ops: &mut PreparedOp2F32, arg: usize, value: f32) {
    ops.add_enum_op(Operation::Expf(arg, value))
}

#[no_mangle]
pub extern "C" fn sin_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::Sin(arg))
}

#[no_mangle]
pub extern "C" fn cos_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::Cos(arg))
}

#[no_mangle]
pub extern "C" fn tan_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::Tan(arg))
}

#[no_mangle]
pub extern "C" fn asin_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::ASin(arg))
}

#[no_mangle]
pub extern "C" fn acos_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::ACos(arg))
}

#[no_mangle]
pub extern "C" fn atan_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::ATan(arg))
}

#[no_mangle]
pub extern "C" fn sinh_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::Sinh(arg))
}

#[no_mangle]
pub extern "C" fn cosh_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::Cosh(arg))
}

#[no_mangle]
pub extern "C" fn tanh_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::Tanh(arg))
}

#[no_mangle]
pub extern "C" fn asinh_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::ASinh(arg))
}

#[no_mangle]
pub extern "C" fn acosh_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::ACosh(arg))
}

#[no_mangle]
pub extern "C" fn atanh_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::ATanh(arg))
}

#[no_mangle]
pub extern "C" fn clone_from_ops2_f32(ops: &mut PreparedOp2F32, arg: usize, source: usize) {
    ops.add_enum_op(Operation::CloneFrom(arg, source))
}

#[no_mangle]
pub extern "C" fn add_points_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::AddPoints(arg))
}

#[no_mangle]
pub extern "C" fn sub_points_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::SubPoints(arg))
}

#[no_mangle]
pub extern "C" fn mul_points_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::MulPoints(arg))
}

#[no_mangle]
pub extern "C" fn div_points_ops2_f32(ops: &mut PreparedOp2F32, arg: usize) {
    ops.add_enum_op(Operation::DivPoints(arg))
}

#[no_mangle]
pub extern "C" fn map_real_ops2_f32(ops: &mut PreparedOp2F32,
                                    arg: usize,
                                    map: extern "C" fn(f32, usize) -> f32) {
    ops.add_enum_op(Operation::MapReal(arg, Arc::new(move |v, i| map(v, i))))
}

#[no_mangle]
pub extern "C" fn map_complex_ops2_f32(ops: &mut PreparedOp2F32,
                                       arg: usize,
                                       map: extern "C" fn(Complex32, usize) -> Complex32) {
    ops.add_enum_op(Operation::MapComplex(arg, Arc::new(move |v, i| map(v, i))))
}

#[no_mangle]
pub extern "C" fn delete_ops2_f32(vector: Box<PreparedOp2F32>) {
    drop(vector);
}
