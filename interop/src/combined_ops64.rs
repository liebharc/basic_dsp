// Auto generated code, change combined_ops32.rs and run facade64_create.pl
//! Functions for 64bit floating point number based vectors. Please refer to the other chapters of the help for documentation of the functions.
use super::*;
use basic_dsp_vector::vector_types2::*;
use basic_dsp_vector::vector_types2::combined_ops::*;
use num::complex::Complex64;
use std::sync::Arc;

pub type VecBuf = InteropVec<f64>;

pub type VecBox = Box<InteropVec<f64>>;

pub type PreparedOp1F64 =
	PreparedOperation1<f64,
		RealOrComplexData, TimeOrFrequencyData,
		RealOrComplexData, TimeOrFrequencyData>;

pub type PreparedOp2F64 =
	PreparedOperation2<f64,
		RealOrComplexData, TimeOrFrequencyData,
		RealOrComplexData, TimeOrFrequencyData,
		RealOrComplexData, TimeOrFrequencyData,
		RealOrComplexData, TimeOrFrequencyData>;

/// Prepares an operation.
/// multi_ops1 will not be made available in for interop since the same functionality
/// can be created with prepared ops, and internally this is what this lib does too.
#[no_mangle]
pub extern fn prepared_ops1_f64() -> Box<PreparedOp1F64> {
    Box::new(prepare64_1(
		RealOrComplexData{is_complex_current: false},
		TimeOrFrequencyData{domain_current: DataDomain::Time}))
}

/// Prepares an operation.
/// multi_ops2 will not be made available in for interop since the same functionality
/// can be created with prepared ops, and internally this is what this lib does too.
#[no_mangle]
pub extern fn prepared_ops2_f64() -> Box<PreparedOp2F64> {
    Box::new(prepare64_2(
		RealOrComplexData{is_complex_current: false},
		TimeOrFrequencyData{domain_current: DataDomain::Time},
		RealOrComplexData{is_complex_current: false},
		TimeOrFrequencyData{domain_current: DataDomain::Time}))
}

/// Prepares an operation.
/// multi_ops1 will not be made available in for interop since the same functionality
/// can be created with prepared ops, and internally this is what this lib does too.
#[no_mangle]
pub extern fn extend_prepared_ops1_f64(ops: Box<PreparedOp1F64>) -> Box<PreparedOp2F64> {
    Box::new(
		ops.extend(
			RealOrComplexData{is_complex_current: false},
			TimeOrFrequencyData{domain_current: DataDomain::Time}))
}

#[no_mangle]
pub extern fn exec_prepared_ops1_f64(
    ops: &PreparedOp1F64,
    v: Box<VecBuf>) -> VectorInteropResult<VecBuf> {
	v.trans_vec(|v, b|ops.exec(b, v))
}

#[no_mangle]
pub extern fn exec_prepared_ops2_f64(
    ops: &PreparedOp2F64,
    v1: Box<VecBuf>,
    v2: Box<VecBuf>) -> BinaryVectorInteropResult<VecBuf> {
	let (v1, mut b1) = v1.decompose();
	let (v2, b2) = v2.decompose();
    let result = ops.exec(&mut b1, v1, v2);
	match result {
		Ok((r1, r2)) => {
			let r1 = VecBuf{vec: r1, buffer: b1 };
			let r2 = VecBuf{vec: r2, buffer: b2 };
			BinaryVectorInteropResult {
				result_code: 0,
				vector1: Box::new(r1),
				vector2: Box::new(r2)
			}
		},
		Err((reason, r1, r2)) => {
			let r1 = VecBuf{vec: r1, buffer: b1 };
			let r2 = VecBuf{vec: r2, buffer: b2 };
			BinaryVectorInteropResult {
				result_code: translate_error(reason),
				vector1: Box::new(r1),
				vector2: Box::new(r2)
			}
		}
	}
}

//----------------------------------------------
// PreparedOp1F64
//----------------------------------------------
#[no_mangle]
pub extern fn add_real_ops1_f64(ops: &mut PreparedOp1F64, arg: usize, value: f64) {
    ops.add_enum_op(Operation::AddReal(arg, value))
}

#[no_mangle]
pub extern fn multiply_real_ops1_f64(ops: &mut PreparedOp1F64, arg: usize, value: f64) {
    ops.add_enum_op(Operation::MultiplyReal(arg, value))
}

#[no_mangle]
pub extern fn abs_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::Abs(arg))
}

#[no_mangle]
pub extern fn to_complex_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::ToComplex(arg))
}

#[no_mangle]
pub extern fn add_complex_ops1_f64(ops: &mut PreparedOp1F64, arg: usize, re: f64, im: f64) {
    ops.add_enum_op(Operation::AddComplex(arg, Complex64::new(re, im)))
}

#[no_mangle]
pub extern fn multiply_complex_ops1_f64(ops: &mut PreparedOp1F64, arg: usize, re: f64, im: f64) {
    ops.add_enum_op(Operation::MultiplyComplex(arg, Complex64::new(re, im)))
}

#[no_mangle]
pub extern fn magnitude_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::Magnitude(arg))
}

#[no_mangle]
pub extern fn magnitude_squared_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::MagnitudeSquared(arg))
}

#[no_mangle]
pub extern fn complex_conj_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::ComplexConj(arg))
}

#[no_mangle]
pub extern fn to_real_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::ToReal(arg))
}

#[no_mangle]
pub extern fn to_imag_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::ToImag(arg))
}

#[no_mangle]
pub extern fn phase_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::Phase(arg))
}

#[no_mangle]
pub extern fn multiply_complex_exponential_ops1_f64(ops: &mut PreparedOp1F64, arg: usize, a: f64, b: f64) {
    ops.add_enum_op(Operation::MultiplyComplexExponential(arg, a, b))
}

#[no_mangle]
pub extern fn add_vector_ops1_f64(ops: &mut PreparedOp1F64, arg: usize, other: usize) {
    ops.add_enum_op(Operation::AddVector(arg, other))
}

#[no_mangle]
pub extern fn mul_vector_ops1_f64(ops: &mut PreparedOp1F64, arg: usize, other: usize) {
    ops.add_enum_op(Operation::MulVector(arg, other))
}

#[no_mangle]
pub extern fn sub_vector_ops1_f64(ops: &mut PreparedOp1F64, arg: usize, other: usize) {
    ops.add_enum_op(Operation::SubVector(arg, other))
}

#[no_mangle]
pub extern fn div_vector_ops1_f64(ops: &mut PreparedOp1F64, arg: usize, other: usize) {
    ops.add_enum_op(Operation::DivVector(arg, other))
}

#[no_mangle]
pub extern fn square_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::Square(arg))
}

#[no_mangle]
pub extern fn sqrt_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::Sqrt(arg))
}

#[no_mangle]
pub extern fn root_ops1_f64(ops: &mut PreparedOp1F64, arg: usize, value: f64) {
    ops.add_enum_op(Operation::Root(arg, value))
}

#[no_mangle]
pub extern fn powf_ops1_f64(ops: &mut PreparedOp1F64, arg: usize, value: f64) {
    ops.add_enum_op(Operation::Powf(arg, value))
}

#[no_mangle]
pub extern fn ln_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::Ln(arg))
}

#[no_mangle]
pub extern fn exp_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::Exp(arg))
}

#[no_mangle]
pub extern fn log_ops1_f64(ops: &mut PreparedOp1F64, arg: usize, value: f64) {
    ops.add_enum_op(Operation::Log(arg, value))
}

#[no_mangle]
pub extern fn expf_ops1_f64(ops: &mut PreparedOp1F64, arg: usize, value: f64) {
    ops.add_enum_op(Operation::Expf(arg, value))
}

#[no_mangle]
pub extern fn sin_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::Sin(arg))
}

#[no_mangle]
pub extern fn cos_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::Cos(arg))
}

#[no_mangle]
pub extern fn tan_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::Tan(arg))
}

#[no_mangle]
pub extern fn asin_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::ASin(arg))
}

#[no_mangle]
pub extern fn acos_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::ACos(arg))
}

#[no_mangle]
pub extern fn atan_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::ATan(arg))
}

#[no_mangle]
pub extern fn sinh_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::Sinh(arg))
}

#[no_mangle]
pub extern fn cosh_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::Cosh(arg))
}

#[no_mangle]
pub extern fn tanh_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::Tanh(arg))
}

#[no_mangle]
pub extern fn asinh_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::ASinh(arg))
}

#[no_mangle]
pub extern fn acosh_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::ACosh(arg))
}

#[no_mangle]
pub extern fn atanh_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::ATanh(arg))
}

#[no_mangle]
pub extern fn clone_from_ops1_f64(ops: &mut PreparedOp1F64, arg: usize, source: usize) {
    ops.add_enum_op(Operation::CloneFrom(arg, source))
}

#[no_mangle]
pub extern fn add_points_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::AddPoints(arg))
}

#[no_mangle]
pub extern fn sub_points_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::SubPoints(arg))
}

#[no_mangle]
pub extern fn mul_points_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::MulPoints(arg))
}

#[no_mangle]
pub extern fn div_points_ops1_f64(ops: &mut PreparedOp1F64, arg: usize) {
    ops.add_enum_op(Operation::DivPoints(arg))
}

#[no_mangle]
pub extern fn map_real_ops1_f64(ops: &mut PreparedOp1F64, arg: usize, map: extern fn(f64, usize) -> f64) {
    ops.add_enum_op(Operation::MapReal(arg, Arc::new(move|v, i|map(v, i))))
}

#[no_mangle]
pub extern fn map_complex_ops1_f64(ops: &mut PreparedOp1F64, arg: usize, map: extern fn(Complex64, usize) -> Complex64) {
    ops.add_enum_op(Operation::MapComplex(arg, Arc::new(move|v, i|map(v, i))))
}

#[no_mangle]
pub extern fn delete_ops1_f64(vector: Box<PreparedOp1F64>) {
    drop(vector);
}

//----------------------------------------------
// PreparedOp2F64
//----------------------------------------------
#[no_mangle]
pub extern fn add_real_ops2_f64(ops: &mut PreparedOp2F64, arg: usize, value: f64) {
    ops.add_enum_op(Operation::AddReal(arg, value))
}

#[no_mangle]
pub extern fn multiply_real_ops2_f64(ops: &mut PreparedOp2F64, arg: usize, value: f64) {
    ops.add_enum_op(Operation::MultiplyReal(arg, value))
}

#[no_mangle]
pub extern fn abs_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::Abs(arg))
}

#[no_mangle]
pub extern fn to_complex_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::ToComplex(arg))
}

#[no_mangle]
pub extern fn add_complex_ops2_f64(ops: &mut PreparedOp2F64, arg: usize, re: f64, im: f64) {
    ops.add_enum_op(Operation::AddComplex(arg, Complex64::new(re, im)))
}

#[no_mangle]
pub extern fn multiply_complex_ops2_f64(ops: &mut PreparedOp2F64, arg: usize, re: f64, im: f64) {
    ops.add_enum_op(Operation::MultiplyComplex(arg, Complex64::new(re, im)))
}

#[no_mangle]
pub extern fn magnitude_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::Magnitude(arg))
}

#[no_mangle]
pub extern fn magnitude_squared_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::MagnitudeSquared(arg))
}

#[no_mangle]
pub extern fn complex_conj_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::ComplexConj(arg))
}

#[no_mangle]
pub extern fn to_real_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::ToReal(arg))
}

#[no_mangle]
pub extern fn to_imag_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::ToImag(arg))
}

#[no_mangle]
pub extern fn phase_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::Phase(arg))
}

#[no_mangle]
pub extern fn multiply_complex_exponential_ops2_f64(ops: &mut PreparedOp2F64, arg: usize, a: f64, b: f64) {
    ops.add_enum_op(Operation::MultiplyComplexExponential(arg, a, b))
}

#[no_mangle]
pub extern fn add_vector_ops2_f64(ops: &mut PreparedOp2F64, arg: usize, other: usize) {
    ops.add_enum_op(Operation::AddVector(arg, other))
}

#[no_mangle]
pub extern fn mul_vector_ops2_f64(ops: &mut PreparedOp2F64, arg: usize, other: usize) {
    ops.add_enum_op(Operation::MulVector(arg, other))
}

#[no_mangle]
pub extern fn sub_vector_ops2_f64(ops: &mut PreparedOp2F64, arg: usize, other: usize) {
    ops.add_enum_op(Operation::SubVector(arg, other))
}

#[no_mangle]
pub extern fn div_vector_ops2_f64(ops: &mut PreparedOp2F64, arg: usize, other: usize) {
    ops.add_enum_op(Operation::DivVector(arg, other))
}

#[no_mangle]
pub extern fn square_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::Square(arg))
}

#[no_mangle]
pub extern fn sqrt_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::Sqrt(arg))
}

#[no_mangle]
pub extern fn root_ops2_f64(ops: &mut PreparedOp2F64, arg: usize, value: f64) {
    ops.add_enum_op(Operation::Root(arg, value))
}

#[no_mangle]
pub extern fn powf_ops2_f64(ops: &mut PreparedOp2F64, arg: usize, value: f64) {
    ops.add_enum_op(Operation::Powf(arg, value))
}

#[no_mangle]
pub extern fn ln_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::Ln(arg))
}

#[no_mangle]
pub extern fn exp_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::Exp(arg))
}

#[no_mangle]
pub extern fn log_ops2_f64(ops: &mut PreparedOp2F64, arg: usize, value: f64) {
    ops.add_enum_op(Operation::Log(arg, value))
}

#[no_mangle]
pub extern fn expf_ops2_f64(ops: &mut PreparedOp2F64, arg: usize, value: f64) {
    ops.add_enum_op(Operation::Expf(arg, value))
}

#[no_mangle]
pub extern fn sin_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::Sin(arg))
}

#[no_mangle]
pub extern fn cos_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::Cos(arg))
}

#[no_mangle]
pub extern fn tan_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::Tan(arg))
}

#[no_mangle]
pub extern fn asin_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::ASin(arg))
}

#[no_mangle]
pub extern fn acos_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::ACos(arg))
}

#[no_mangle]
pub extern fn atan_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::ATan(arg))
}

#[no_mangle]
pub extern fn sinh_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::Sinh(arg))
}

#[no_mangle]
pub extern fn cosh_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::Cosh(arg))
}

#[no_mangle]
pub extern fn tanh_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::Tanh(arg))
}

#[no_mangle]
pub extern fn asinh_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::ASinh(arg))
}

#[no_mangle]
pub extern fn acosh_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::ACosh(arg))
}

#[no_mangle]
pub extern fn atanh_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::ATanh(arg))
}

#[no_mangle]
pub extern fn clone_from_ops2_f64(ops: &mut PreparedOp2F64, arg: usize, source: usize) {
    ops.add_enum_op(Operation::CloneFrom(arg, source))
}

#[no_mangle]
pub extern fn add_points_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::AddPoints(arg))
}

#[no_mangle]
pub extern fn sub_points_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::SubPoints(arg))
}

#[no_mangle]
pub extern fn mul_points_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::MulPoints(arg))
}

#[no_mangle]
pub extern fn div_points_ops2_f64(ops: &mut PreparedOp2F64, arg: usize) {
    ops.add_enum_op(Operation::DivPoints(arg))
}

#[no_mangle]
pub extern fn map_real_ops2_f64(ops: &mut PreparedOp2F64, arg: usize, map: extern fn(f64, usize) -> f64) {
    ops.add_enum_op(Operation::MapReal(arg, Arc::new(move|v, i|map(v, i))))
}

#[no_mangle]
pub extern fn map_complex_ops2_f64(ops: &mut PreparedOp2F64, arg: usize, map: extern fn(Complex64, usize) -> Complex64) {
    ops.add_enum_op(Operation::MapComplex(arg, Arc::new(move|v, i|map(v, i))))
}

#[no_mangle]
pub extern fn delete_ops2_f64(vector: Box<PreparedOp2F64>) {
    drop(vector);
}
