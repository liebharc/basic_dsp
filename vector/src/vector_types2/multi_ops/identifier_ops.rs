use super::{Identifier, Operation};
use RealNumber;
use num::complex::Complex;
use super::super::{
		RealNumberSpace, ComplexNumberSpace, NumberSpace,
		Domain,
		ScaleOps, OffsetOps, PowerOps, TrigOps
};

// TODO
/*
// Real Ops
Abs(usize),
ToComplex(usize),
MapReal(usize, Arc<Fn(T, usize) -> T + Send + Sync + 'static>),
// Complex Ops
Magnitude(usize),
MagnitudeSquared(usize),
ComplexConj(usize),
ToReal(usize),
ToImag(usize),
Phase(usize),
MultiplyComplexExponential(usize, T, T),
MapComplex(usize, Arc<Fn(Complex<T>, usize) -> Complex<T> + Send + Sync + 'static>),
// General Ops
AddVector(usize, usize),
SubVector(usize, usize),
MulVector(usize, usize),
DivVector(usize, usize),
CloneFrom(usize, usize),
AddPoints(usize),
SubPoints(usize),
MulPoints(usize),
DivPoints(usize)
*/

impl<T, N, D> OffsetOps<T> for Identifier<T, N, D>
 	where T: RealNumber,
		  N: RealNumberSpace,
		  D: Domain{
    fn offset(&mut self, offset: T) {
		let arg = self.arg;
		self.add_op(Operation::AddReal(arg, offset));
	}
}

impl<T, N, D> ScaleOps<T> for Identifier<T, N, D>
 	where T: RealNumber,
		  N: RealNumberSpace,
		  D: Domain{
    fn scale(&mut self, offset: T) {
		let arg = self.arg;
		self.add_op(Operation::MultiplyReal(arg, offset));
	}
}

impl<T, N, D> OffsetOps<Complex<T>> for Identifier<T, N, D>
 	where T: RealNumber,
		  N: ComplexNumberSpace,
		  D: Domain{
    fn offset(&mut self, offset: Complex<T>) {
		let arg = self.arg;
		self.add_op(Operation::AddComplex(arg, offset));
	}
}

impl<T, N, D> ScaleOps<Complex<T>> for Identifier<T, N, D>
 	where T: RealNumber,
		  N: ComplexNumberSpace,
		  D: Domain{
    fn scale(&mut self, offset: Complex<T>) {
		let arg = self.arg;
		self.add_op(Operation::MultiplyComplex(arg, offset));
	}
}

impl<T, N, D> TrigOps for Identifier<T, N, D>
 	where T: RealNumber,
		  N: NumberSpace,
		  D: Domain{
    fn sin(&mut self) {
		let arg = self.arg;
		self.add_op(Operation::Sin(arg));
	}

    fn cos(&mut self) {
		let arg = self.arg;
		self.add_op(Operation::Cos(arg));
	}

    fn tan(&mut self) {
		let arg = self.arg;
		self.add_op(Operation::Tan(arg));
	}

    fn asin(&mut self) {
		let arg = self.arg;
		self.add_op(Operation::ASin(arg));
	}

    fn acos(&mut self) {
		let arg = self.arg;
		self.add_op(Operation::ACos(arg));
	}

    fn atan(&mut self) {
		let arg = self.arg;
		self.add_op(Operation::ATan(arg));
	}

    fn sinh(&mut self) {
		let arg = self.arg;
		self.add_op(Operation::Sinh(arg));
	}

    fn cosh(&mut self) {
		let arg = self.arg;
		self.add_op(Operation::Cosh(arg));
	}

    fn tanh(&mut self) {
		let arg = self.arg;
		self.add_op(Operation::Tanh(arg));
	}

    fn asinh(&mut self) {
		let arg = self.arg;
		self.add_op(Operation::ASinh(arg));
	}

    fn acosh(&mut self) {
		let arg = self.arg;
		self.add_op(Operation::ACosh(arg));
	}

    fn atanh(&mut self) {
		let arg = self.arg;
		self.add_op(Operation::ATanh(arg));
	}
}

impl<T, N, D> PowerOps<T> for Identifier<T, N, D>
 	where T: RealNumber,
		  N: NumberSpace,
		  D: Domain{
    fn sqrt(&mut self) {
		let arg = self.arg;
		self.add_op(Operation::ATanh(arg));
	}

    fn square(&mut self) {
		let arg = self.arg;
		self.add_op(Operation::Square(arg));
	}

    fn root(&mut self, degree: T) {
		let arg = self.arg;
		self.add_op(Operation::Root(arg, degree));
	}

    fn powf(&mut self, exponent: T) {
		let arg = self.arg;
		self.add_op(Operation::Powf(arg, exponent));
	}

    fn ln(&mut self) {
		let arg = self.arg;
		self.add_op(Operation::Ln(arg));
	}

    fn exp(&mut self) {
		let arg = self.arg;
		self.add_op(Operation::Exp(arg));
	}

    fn log(&mut self, base: T) {
		let arg = self.arg;
		self.add_op(Operation::Log(arg, base));
	}

    fn expf(&mut self, base: T) {
		let arg = self.arg;
		self.add_op(Operation::Expf(arg, base));
	}
}
