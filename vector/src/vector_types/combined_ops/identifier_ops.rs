use super::super::{
    ComplexData, ComplexNumberSpace, ComplexOps, ComplexToRealTransformsOps, DataDomain, Domain,
    ElementaryOps, FrequencyData, GetMetaData, MultiCoreSettings, NumberSpace, OffsetOps, PowerOps,
    RealData, RealNumberSpace, RealOps, RealOrComplexData, RealToComplexTransformsOps,
    RededicateForceOps, ScaleOps, TimeData, TimeOrFrequencyData, ToComplexResult, ToRealResult,
    TransRes, TrigOps, TypeMetaData, VoidResult,
};
use super::{Identifier, Operation};
use numbers::*;

/// Operations for all kind of vectors which can be used in combination
/// with multi ops or prepared ops.
pub trait IdentifierOps {
    /// The domain in which the data vector resides. Basically specifies the x-axis and
    /// the type of operations which are valid on this vector.
    ///
    /// The domain can be changed using the `RededicateOps` trait.
    fn domain(&self) -> DataDomain;

    /// Indicates whether the vector contains complex data. This also specifies
    /// the type of operations which are valid on this vector.
    ///
    /// The number space can be changed using the `RededicateOps` trait.
    fn is_complex(&self) -> bool;

    /// Copies data from another vector.
    fn clone_from(&mut self, source: &Self);

    /// Adds its length to the vector elements
    /// # Example
    ///
    /// ```no_run
    /// use basic_dsp_vector::*;
    /// use basic_dsp_vector::combined_ops::*;
    /// let complex = vec!(1.0, 2.0, 3.0, 4.0).to_complex_time_vec();
    /// let ops = multi_ops1(complex);
    /// let ops = ops.add_ops(|mut v| {
    /// 	v.add_points();
    /// 	v
    /// });
    /// let mut buffer = SingleBuffer::new();
    /// let complex = ops.get(&mut buffer).expect("Ignoring error handling in examples");
    /// assert_eq!([3.0, 2.0, 5.0, 4.0], &complex[..]);
    /// ```
    fn add_points(&mut self);

    /// Subtracts its length from the vector elements
    /// # Example
    ///
    /// ```no_run
    /// use basic_dsp_vector::*;
    /// use basic_dsp_vector::combined_ops::*;
    /// let complex = vec!(3.0, 2.0, 5.0, 4.0).to_complex_time_vec();
    /// let ops = multi_ops1(complex);
    /// let ops = ops.add_ops(|mut v| {
    /// 	v.sub_points();
    /// 	v
    /// });
    /// let mut buffer = SingleBuffer::new();
    /// let complex = ops.get(&mut buffer).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 2.0, 3.0, 4.0], &complex[..]);
    /// ```
    fn sub_points(&mut self);

    /// divides the vector elements by its length
    /// Subtracts its length from the vector elements
    /// # Example
    ///
    /// ```no_run
    /// use basic_dsp_vector::*;
    /// use basic_dsp_vector::combined_ops::*;
    /// let complex = vec!(2.0, 4.0, 6.0, 8.0).to_complex_time_vec();
    /// let ops = multi_ops1(complex);
    /// let ops = ops.add_ops(|mut v| {
    /// 	v.div_points();
    /// 	v
    /// });
    /// let mut buffer = SingleBuffer::new();
    /// let complex = ops.get(&mut buffer).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 2.0, 3.0, 4.0], &complex[..]);
    /// ```
    fn div_points(&mut self);

    /// Multiplies the vector elements with its length
    /// # Example
    ///
    /// ```no_run
    /// use basic_dsp_vector::*;
    /// use basic_dsp_vector::combined_ops::*;
    /// let complex = vec!(1.0, 2.0, 3.0, 4.0).to_complex_time_vec();
    /// let ops = multi_ops1(complex);
    /// let ops = ops.add_ops(|mut v| {
    /// 	v.mul_points();
    /// 	v
    /// });
    /// let mut buffer = SingleBuffer::new();
    /// let complex = ops.get(&mut buffer).expect("Ignoring error handling in examples");
    /// assert_eq!([2.0, 4.0, 6.0, 8.0], &complex[..]);
    fn mul_points(&mut self);
}

/// A identifier with real numbers in time domain.
pub type RealTimeIdent<T> = Identifier<T, RealData, TimeData>;
/// A identifier with real numbers in frequency domain.
pub type RealFreqIdent<T> = Identifier<T, RealData, FrequencyData>;
/// A identifier with complex numbers in time domain.
pub type ComplexTimeIdent<T> = Identifier<T, ComplexData, TimeData>;
/// A identifier with complex numbers in frequency domain.
pub type ComplexFreqIdent<T> = Identifier<T, ComplexData, FrequencyData>;
/// A identifier with no information about number space or domain at compile time.
pub type GenDspIdent<T> = Identifier<T, RealOrComplexData, TimeOrFrequencyData>;

impl<T> ToRealResult for ComplexTimeIdent<T>
where
    T: RealNumber,
{
    type RealResult = RealTimeIdent<T>;
}

impl<T> ToRealResult for ComplexFreqIdent<T>
where
    T: RealNumber,
{
    type RealResult = RealFreqIdent<T>;
}

impl<T> ToRealResult for GenDspIdent<T>
where
    T: RealNumber,
{
    type RealResult = GenDspIdent<T>;
}

impl<T> ToComplexResult for RealTimeIdent<T>
where
    T: RealNumber,
{
    type ComplexResult = ComplexTimeIdent<T>;
}

impl<T> ToComplexResult for RealFreqIdent<T>
where
    T: RealNumber,
{
    type ComplexResult = ComplexFreqIdent<T>;
}

impl<T> ToComplexResult for GenDspIdent<T>
where
    T: RealNumber,
{
    type ComplexResult = GenDspIdent<T>;
}

impl<T, N, D> GetMetaData<T, N, D> for Identifier<T, N, D>
where
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn get_meta_data(&self) -> TypeMetaData<T, N, D> {
        TypeMetaData {
            number_space: self.number_space.clone(),
            domain: self.domain.clone(),
            delta: T::one(),
            multicore_settings: MultiCoreSettings::default(),
        }
    }
}

impl<T, N, D> RededicateForceOps<Identifier<T, N, D>> for RealTimeIdent<T>
where
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn rededicate_from_force(origin: Identifier<T, N, D>) -> Self {
        RealTimeIdent {
            arg: origin.arg,
            ops: origin.ops,
            counter: origin.counter,
            domain: TimeData,
            number_space: RealData,
        }
    }

    fn rededicate_with_runtime_data(origin: Identifier<T, N, D>, _: bool, _: DataDomain) -> Self {
        Self::rededicate_from_force(origin)
    }
}

impl<T, N, D> RededicateForceOps<Identifier<T, N, D>> for RealFreqIdent<T>
where
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn rededicate_from_force(origin: Identifier<T, N, D>) -> Self {
        RealFreqIdent {
            arg: origin.arg,
            ops: origin.ops,
            counter: origin.counter,
            domain: FrequencyData,
            number_space: RealData,
        }
    }

    fn rededicate_with_runtime_data(origin: Identifier<T, N, D>, _: bool, _: DataDomain) -> Self {
        Self::rededicate_from_force(origin)
    }
}

impl<T, N, D> RededicateForceOps<Identifier<T, N, D>> for ComplexTimeIdent<T>
where
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn rededicate_from_force(origin: Identifier<T, N, D>) -> Self {
        ComplexTimeIdent {
            arg: origin.arg,
            ops: origin.ops,
            counter: origin.counter,
            domain: TimeData,
            number_space: ComplexData,
        }
    }

    fn rededicate_with_runtime_data(origin: Identifier<T, N, D>, _: bool, _: DataDomain) -> Self {
        Self::rededicate_from_force(origin)
    }
}

impl<T, N, D> RededicateForceOps<Identifier<T, N, D>> for ComplexFreqIdent<T>
where
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn rededicate_from_force(origin: Identifier<T, N, D>) -> Self {
        ComplexFreqIdent {
            arg: origin.arg,
            ops: origin.ops,
            counter: origin.counter,
            domain: FrequencyData,
            number_space: ComplexData,
        }
    }

    fn rededicate_with_runtime_data(origin: Identifier<T, N, D>, _: bool, _: DataDomain) -> Self {
        Self::rededicate_from_force(origin)
    }
}

impl<T, N, D> RededicateForceOps<Identifier<T, N, D>> for GenDspIdent<T>
where
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn rededicate_from_force(origin: Identifier<T, N, D>) -> Self {
        let is_complex = origin.is_complex();
        let domain = origin.domain();
        GenDspIdent {
            arg: origin.arg,
            ops: origin.ops,
            counter: origin.counter,
            domain: TimeOrFrequencyData {
                domain_current: domain,
            },
            number_space: RealOrComplexData {
                is_complex_current: is_complex,
            },
        }
    }

    fn rededicate_with_runtime_data(
        origin: Identifier<T, N, D>,
        is_complex: bool,
        domain: DataDomain,
    ) -> Self {
        let mut result = Self::rededicate_from_force(origin);
        result.number_space.is_complex_current = is_complex;
        result.domain.domain_current = domain;
        result
    }
}

impl<T, N, D> OffsetOps<T> for Identifier<T, N, D>
where
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn offset(&mut self, offset: T) {
        let arg = self.arg;
        if self.is_complex() {
            self.add_op(Operation::AddComplex(
                arg,
                Complex::<T>::new(offset, T::zero()),
            ));
        } else {
            self.add_op(Operation::AddReal(arg, offset));
        }
    }
}

impl<T, N, D> ScaleOps<T> for Identifier<T, N, D>
where
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn scale(&mut self, offset: T) {
        let arg = self.arg;
        if self.is_complex() {
            self.add_op(Operation::MultiplyComplex(
                arg,
                Complex::<T>::new(offset, T::zero()),
            ));
        } else {
            self.add_op(Operation::MultiplyReal(arg, offset));
        }
    }
}

impl<T, N, D> OffsetOps<Complex<T>> for Identifier<T, N, D>
where
    T: RealNumber,
    N: ComplexNumberSpace,
    D: Domain,
{
    fn offset(&mut self, offset: Complex<T>) {
        let arg = self.arg;
        self.add_op(Operation::AddComplex(arg, offset));
    }
}

impl<T, N, D> ScaleOps<Complex<T>> for Identifier<T, N, D>
where
    T: RealNumber,
    N: ComplexNumberSpace,
    D: Domain,
{
    fn scale(&mut self, offset: Complex<T>) {
        let arg = self.arg;
        self.add_op(Operation::MultiplyComplex(arg, offset));
    }
}

impl<T, N, D> TrigOps for Identifier<T, N, D>
where
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
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
where
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn sqrt(&mut self) {
        let arg = self.arg;
        self.add_op(Operation::Sqrt(arg));
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

impl<T, N, D> RealOps for Identifier<T, N, D>
where
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn abs(&mut self) {
        let arg = self.arg;
        self.add_op(Operation::Abs(arg));
    }
}

impl<T, N, D> ComplexOps<T> for Identifier<T, N, D>
where
    T: RealNumber,
    N: ComplexNumberSpace,
    D: Domain,
{
    fn multiply_complex_exponential(&mut self, a: T, b: T) {
        let arg = self.arg;
        self.add_op(Operation::MultiplyComplexExponential(arg, a, b))
    }

    fn conj(&mut self) {
        let arg = self.arg;
        self.add_op(Operation::ComplexConj(arg))
    }
}

impl<T, N, D> ElementaryOps<Identifier<T, N, D>, T, N, D> for Identifier<T, N, D>
where
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn add(&mut self, summand: &Self) -> VoidResult {
        let arg = self.arg;
        let other = summand.arg;
        self.add_arg_op(Operation::AddVector(arg, other), summand);
        Ok(())
    }

    fn sub(&mut self, subtrahend: &Self) -> VoidResult {
        let arg = self.arg;
        let other = subtrahend.arg;
        self.add_arg_op(Operation::SubVector(arg, other), subtrahend);
        Ok(())
    }

    fn mul(&mut self, factor: &Self) -> VoidResult {
        let arg = self.arg;
        let other = factor.arg;
        self.add_arg_op(Operation::MulVector(arg, other), factor);
        Ok(())
    }

    fn div(&mut self, divisor: &Self) -> VoidResult {
        let arg = self.arg;
        let other = divisor.arg;
        self.add_arg_op(Operation::DivVector(arg, other), divisor);
        Ok(())
    }
}

impl<T, N, D> IdentifierOps for Identifier<T, N, D>
where
    T: RealNumber,
    N: NumberSpace,
    D: Domain,
{
    fn domain(&self) -> DataDomain {
        self.domain.domain()
    }

    fn is_complex(&self) -> bool {
        self.number_space.is_complex()
    }

    fn clone_from(&mut self, source: &Self) {
        let arg = self.arg;
        let other = source.arg;
        self.add_arg_op(Operation::CloneFrom(arg, other), source);
    }

    fn add_points(&mut self) {
        let arg = self.arg;
        self.add_op(Operation::AddPoints(arg))
    }

    fn sub_points(&mut self) {
        let arg = self.arg;
        self.add_op(Operation::SubPoints(arg))
    }

    fn div_points(&mut self) {
        let arg = self.arg;
        self.add_op(Operation::DivPoints(arg))
    }

    fn mul_points(&mut self) {
        let arg = self.arg;
        self.add_op(Operation::MulPoints(arg))
    }
}

impl<T, N, D> RealToComplexTransformsOps<T> for Identifier<T, N, D>
where
    Identifier<T, N, D>: ToComplexResult,
    <Identifier<T, N, D> as ToComplexResult>::ComplexResult:
        RededicateForceOps<Identifier<T, N, D>>,
    T: RealNumber,
    N: RealNumberSpace,
    D: Domain,
{
    fn to_complex(mut self) -> TransRes<Self::ComplexResult> {
        let arg = self.arg;
        self.add_op(Operation::ToComplex(arg));
        self.number_space.to_complex();
        Ok(Self::ComplexResult::rededicate_from_force(self))
    }
}

impl<T, N, D> ComplexToRealTransformsOps<T> for Identifier<T, N, D>
where
    Identifier<T, N, D>: ToRealResult,
    <Identifier<T, N, D> as ToRealResult>::RealResult: RededicateForceOps<Identifier<T, N, D>>,
    T: RealNumber,
    N: ComplexNumberSpace,
    D: Domain,
{
    fn magnitude(mut self) -> Self::RealResult {
        let arg = self.arg;
        self.add_op(Operation::Magnitude(arg));
        self.number_space.to_real();
        Self::RealResult::rededicate_from_force(self)
    }

    fn magnitude_squared(mut self) -> Self::RealResult {
        let arg = self.arg;
        self.add_op(Operation::MagnitudeSquared(arg));
        self.number_space.to_real();
        Self::RealResult::rededicate_from_force(self)
    }

    fn to_real(mut self) -> Self::RealResult {
        let arg = self.arg;
        self.add_op(Operation::ToReal(arg));
        self.number_space.to_real();
        Self::RealResult::rededicate_from_force(self)
    }

    fn to_imag(mut self) -> Self::RealResult {
        let arg = self.arg;
        self.add_op(Operation::ToImag(arg));
        self.number_space.to_real();
        Self::RealResult::rededicate_from_force(self)
    }

    fn phase(mut self) -> Self::RealResult {
        let arg = self.arg;
        self.add_op(Operation::Phase(arg));
        self.number_space.to_real();
        Self::RealResult::rededicate_from_force(self)
    }
}
