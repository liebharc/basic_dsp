use std::marker::PhantomData;
use std::result;
use std::ops::Range;
use super::super::RealNumber;
use super::{
    round_len,
    DataVec,
    TransRes,
    ErrorReason,
    RealTimeVector,
    ComplexTimeVector,
    RealFreqVector,
    ComplexFreqVector,
    RealVectorOps,
    ComplexVectorOps,
    GenericDataVec,
    RededicateVector};  
use super::operations_enum::{
    Operation,
    evaluate_number_space_transition,
    get_argument,
    PerformOperationSimd};
use multicore_support::{Chunk, Complexity};
use simd_extensions::*;
use num::Complex;
use std::sync::{Arc, Mutex};

const ARGUMENT1: usize = 0;
const ARGUMENT2: usize = 1;

/// Trait which defines the relation between a vector
/// and an identifier. 
pub trait ToIdentifier<T>
    where T: RealNumber 
{
    type Identifier: Identifier<T>;
}

/// An identifier is just a placeholder for a vector
/// used to ensure already at compile time that operations are valid.
pub trait Identifier<T> : Sized
    where T: RealNumber 
{
    type Vector: DataVec<T> + ToIdentifier<T>;
    fn get_arg(&self) -> usize;
    fn get_ops(self) -> Vec<(u64, Operation<T>)>;
    fn new(arg: usize, counter: Arc<Mutex<u64>>) -> Self;
    fn new_ops(ops: Vec<(u64, Operation<T>)>, arg: usize, counter: Arc<Mutex<u64>>) -> Self;
}

/// Operations for complex vectors which can be used in combination 
/// with multi ops or prepared ops. For a documentation of the specific operations
/// see [`ComplexVectorOps`](../trait.ComplexVectorOps.html).
pub trait ComplexIdentifier<T> : Identifier<T>
    where T: RealNumber 
{
    type RealPartner;
    /// See [`ComplexVectorOps`](../trait.ComplexVectorOps.html#tymethod.complex_offset).
    fn complex_offset(self, offset: Complex<T>) -> Self;
    /// See [`ComplexVectorOps`](../trait.ComplexVectorOps.html#tymethod.complex_scale).
    fn complex_scale(self, factor: Complex<T>) -> Self;
    /// See [`ComplexVectorOps`](../trait.ComplexVectorOps.html#tymethod.magnitude).
    fn magnitude(self) -> Self::RealPartner;
    /// See [`ComplexVectorOps`](../trait.ComplexVectorOps.html#tymethod.magnitude_squared).
    fn magnitude_squared(self) -> Self::RealPartner;
    /// See [`ComplexVectorOps`](../trait.ComplexVectorOps.html#tymethod.conj).
    fn conj(self) -> Self;
    /// See [`ComplexVectorOps`](../trait.ComplexVectorOps.html#tymethod.to_real).
    fn to_real(self) -> Self::RealPartner;
    /// See [`ComplexVectorOps`](../trait.ComplexVectorOps.html#tymethod.to_imag).
    fn to_imag(self) -> Self::RealPartner;
    /// See [`ComplexVectorOps`](../trait.ComplexVectorOps.html#tymethod.phase).
    fn phase(self) -> Self::RealPartner;
    /// See [`ComplexVectorOps`](../trait.ComplexVectorOps.html#tymethod.multiply_complex_exponential).
    fn multiply_complex_exponential(self, a: T, b: T) -> Self;
    /// See [`ComplexVectorOps`](../trait.ComplexVectorOps.html#tymethod.map_inplace_complex).
    fn map_inplace_complex<F>(self, f: F) -> Self
        where F: Fn(Complex<T>, usize) -> Complex<T> + Send + Sync;
}

/// Operations for real vectors which can be used on combination 
/// with multi ops or prepared ops. For a documentation of the specific operations
/// see [`RealVectorOps`](../trait.RealVectorOps.html).
pub trait RealIdentifier<T> : Identifier<T>
    where T: RealNumber 
{
    type ComplexPartner;
    /// See [`RealVectorOps`](../trait.RealVectorOps.html#tymethod.real_offset).
    fn real_offset(self, offset: T) -> Self;
    /// See [`RealVectorOps`](../trait.RealVectorOps.html#tymethod.real_scale).
    fn real_scale(self, factor: T) -> Self;
    /// See [`RealVectorOps`](../trait.RealVectorOps.html#tymethod.abs).
    fn abs(self) -> Self;
    /// See [`RealVectorOps`](../trait.RealVectorOps.html#tymethod.to_complex).
    fn to_complex(self) -> Self::ComplexPartner;
    /// See [`RealVectorOps`](../trait.RealVectorOps.html#tymethod.map_inplace_real).
    fn map_inplace_real<F>(self, f: F) -> Self
        where F: Fn(T, usize) -> T + Send + Sync;
}

/// Operations for all kind of vectors which can be used in combination 
/// with multi ops or prepared ops. For a documentation of the specific operations
/// see [`GenericVectorOps`](../trait.GenericVectorOps.html).
pub trait GeneralIdentifier<T> : Identifier<T>
    where T: RealNumber 
{
    /// See [`GenericVectorOps`](../trait.GenericVectorOps.html#tymethod.add_vector).
    fn add_vector(self, summand: &Self) -> Self;
    /// See [`GenericVectorOps`](../trait.GenericVectorOps.html#tymethod.subtract_vector).
    fn subtract_vector(self, subtrahend: &Self) -> Self;
    /// See [`GenericVectorOps`](../trait.GenericVectorOps.html#tymethod.multiply_vector).
    fn multiply_vector(self, factor: &Self) -> Self;
    /// See [`GenericVectorOps`](../trait.GenericVectorOps.html#tymethod.divide_vector).
    fn divide_vector(self, divisor: &Self) -> Self;
    /// See [`GenericVectorOps`](../trait.GenericVectorOps.html#tymethod.sqrt).
    fn sqrt(self) -> Self;
    /// See [`GenericVectorOps`](../trait.GenericVectorOps.html#tymethod.square).
    fn square(self) -> Self;
    /// See [`GenericVectorOps`](../trait.GenericVectorOps.html#tymethod.root).
    fn root(self, degree: T) -> Self;
    /// See [`GenericVectorOps`](../trait.GenericVectorOps.html#tymethod.powf).
    fn powf(self, exponent: T) -> Self;
    /// See [`GenericVectorOps`](../trait.GenericVectorOps.html#tymethod.ln).
    fn ln(self) -> Self;
    /// See [`GenericVectorOps`](../trait.GenericVectorOps.html#tymethod.exp).
    fn exp(self) -> Self;
    /// See [`GenericVectorOps`](../trait.GenericVectorOps.html#tymethod.log).;
    fn log(self, base: T) -> Self;
    /// See [`GenericVectorOps`](../trait.GenericVectorOps.html#tymethod.expf).
    fn expf(self, base: T) -> Self;
    /// See [`GenericVectorOps`](../trait.GenericVectorOps.html#tymethod.sin).
    fn sin(self) -> Self;
    /// See [`GenericVectorOps`](../trait.GenericVectorOps.html#tymethod.cos).
    fn cos(self) -> Self;
    /// See [`GenericVectorOps`](../trait.GenericVectorOps.html#tymethod.tan).
    fn tan(self) -> Self;
    /// See [`GenericVectorOps`](../trait.GenericVectorOps.html#tymethod.asin).
    fn asin(self) -> Self;
    /// See [`GenericVectorOps`](../trait.GenericVectorOps.html#tymethod.acos).
    fn acos(self) -> Self;
    /// See [`GenericVectorOps`](../trait.GenericVectorOps.html#tymethod.atan).
    fn atan(self) -> Self;
    /// See [`GenericVectorOps`](../trait.GenericVectorOps.html#tymethod.sinh).
    fn sinh(self) -> Self;
    /// See [`GenericVectorOps`](../trait.GenericVectorOps.html#tymethod.cosh).
    fn cosh(self) -> Self;
    /// See [`GenericVectorOps`](../trait.GenericVectorOps.html#tymethod.tanh).
    fn tanh(self) -> Self;
    /// See [`GenericVectorOps`](../trait.GenericVectorOps.html#tymethod.asinh).
    fn asinh(self) -> Self;
    /// See [`GenericVectorOps`](../trait.GenericVectorOps.html#tymethod.acosh).
    fn acosh(self) -> Self;
    /// See [`GenericVectorOps`](../trait.GenericVectorOps.html#tymethod.atanh).
    fn atanh(self) -> Self;
    /// Copies data from another vector.
    fn clone_from(self, &Self) -> Self;
    
    /// Adds its length to the vector elements
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// use basic_dsp_vector::combined_ops::*;    
    /// let complex = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
    /// let ops = multi_ops1(complex);
    /// let ops = ops.add_ops(|v|v.add_points());
    /// let complex = ops.get().expect("Ignoring error handling in examples");
    /// assert_eq!([3.0, 2.0, 5.0, 4.0], complex.data());
    /// ```
    fn add_points(self) -> Self;
    
    /// Subtracts its length from the vector elements
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// use basic_dsp_vector::combined_ops::*;    
    /// let complex = ComplexTimeVector32::from_interleaved(&[3.0, 2.0, 5.0, 4.0]);
    /// let ops = multi_ops1(complex);
    /// let ops = ops.add_ops(|v|v.sub_points());
    /// let complex = ops.get().expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 2.0, 3.0, 4.0], complex.data());
    /// ```
    fn sub_points(self) -> Self;
    
    /// divides the vector elements by its length
    /// Subtracts its length from the vector elements
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// use basic_dsp_vector::combined_ops::*;    
    /// let complex = ComplexTimeVector32::from_interleaved(&[2.0, 4.0, 6.0, 8.0]);
    /// let ops = multi_ops1(complex);
    /// let ops = ops.add_ops(|v|v.div_points());
    /// let complex = ops.get().expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 2.0, 3.0, 4.0], complex.data());
    /// ```
    fn div_points(self) -> Self;
    
    /// Multiplies the vector elements with its length
    /// # Example
    ///
    /// ```
    /// use basic_dsp_vector::*;
    /// use basic_dsp_vector::combined_ops::*;    
    /// let complex = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
    /// let ops = multi_ops1(complex);
    /// let ops = ops.add_ops(|v|v.mul_points());
    /// let complex = ops.get().expect("Ignoring error handling in examples");
    /// assert_eq!([2.0, 4.0, 6.0, 8.0], complex.data());
    fn mul_points(self) -> Self;
}

/// Scale operations to vectors in combination 
/// with multi ops or prepared ops. For a documentation of the specific operations
/// see [`Scale`](../trait.Scale.html).
pub trait Scale<T>: Sized where T: Sized {
    /// See [`Scale`](../trait.Scale.html#tymethod.scale).
    fn scale(self, factor: T) -> Self;
}

/// Offset operations to vectors in combination 
/// with multi ops or prepared ops. For a documentation of the specific operations
/// see [`Offset`](../trait.Offset.html).
pub trait Offset<T>: Sized where T: Sized {
    /// See [`Offset`](../trait.Offset.html#tymethod.offset).
    fn offset(self, offset: T) -> Self;
}

/// Allows to map every vector element.
/// with multi ops or prepared ops. For a documentation of the specific operations
/// see [`VectorIter`](../trait.VectorIter.html).
pub trait IdentifierIter<T>: Sized where T: Sized {
    /// See [`VectorIter`](../trait.VectorIter.html#tymethod.map_inplace).
    fn map_inplace<F>(self, f: F) -> Self
        where F: Fn(T, usize) -> T + Send + Sync + 'static ;
}

macro_rules! create_identfier {
    ($($vector:ident, $name:ident;)*)
     =>
     {   
        $(
            /// Placeholder for a concrete vector.
            pub struct $name<T>
                where T: RealNumber 
            {
                arg: usize,
                ops: Vec<(u64, Operation<T>)>,
                counter: Arc<Mutex<u64>>
            }
            
            impl<T> ToIdentifier<T> for $vector<T>
                where T: RealNumber {
                type Identifier = $name<T>;
            }
            
            impl<T> Identifier<T> for $name<T>
                where T: RealNumber {
                type Vector = $vector<T>;
                fn get_arg(&self) -> usize { self.arg }
                fn get_ops(self) -> Vec<(u64, Operation<T>)> { self.ops }
                fn new(arg: usize, counter: Arc<Mutex<u64>>) -> Self { $name { ops: Vec::new(), arg: arg, counter: counter } }
                fn new_ops(ops: Vec<(u64, Operation<T>)>, arg: usize, counter: Arc<Mutex<u64>>) -> Self { $name { ops: ops, arg: arg, counter: counter } }
            }
            
            impl<T> $name<T> 
                where T: RealNumber {
                fn add_op<R>(self, op: Operation<T>) -> R 
                    where R: Identifier<T> {
                    let mut ops = self.ops;
                    let counter = self.counter;
                    let seq = {
                        let mut value = counter.lock().unwrap();
                        *value += 1;
                        *value
                    };
                    ops.push((seq, op));
                    R::new_ops(ops, self.arg, counter)
                }
            }
        )*
     }
}

create_identfier!(
    GenericDataVec, GenericDataIdentifier; 
    RealTimeVector, RealTimeIdentifier; 
    ComplexTimeVector, ComplexTimeIdentifier; 
    RealFreqVector, RealFreqIdentifier; 
    ComplexFreqVector, ComplexFreqIdentifier;);

/// An operation on one data vector which has been prepared in 
/// advance.
///
/// The type arguments can be read like a function definition: TI1 -> TO1
#[derive(Clone)]
pub struct PreparedOperation1<T, TI1, TO1>
    where T: RealNumber,
         TI1: ToIdentifier<T>, 
         TO1: ToIdentifier<T> {
    a: PhantomData<TI1>,
    b: PhantomData<TO1>,
    ops: Vec<Operation<T>>
}

/// An operation on two data vectors which has been prepared in 
/// advance.
///
/// The type arguments can be read like a function definition: (TI1, TI2) -> (TO1, TO2)
#[derive(Clone)]
pub struct PreparedOperation2<T, TI1, TI2, TO1, TO2> 
    where T: RealNumber,
          TI1: ToIdentifier<T>, TI2: ToIdentifier<T>,
          TO1: ToIdentifier<T>, TO2: ToIdentifier<T> {
    a: PhantomData<TI1>,
    b: PhantomData<TI2>,
    c: PhantomData<TO1>,
    d: PhantomData<TO2>,
    ops: Vec<Operation<T>>,
    swap: bool
}

/// A multi operation which holds a vector and records all changes
/// which need to be done to the vectors. By calling `get` on the struct
/// all operations will be executed in one run.
pub struct MultiOperation1<T, TO> 
    where T: RealNumber,
        TO: ToIdentifier<T> {
    a: GenericDataVec<T>,
    prepared_ops: PreparedOperation1<T, GenericDataVec<T>, TO>
}

/// A multi operation which holds two vectors and records all changes
/// which need to be done to the vectors. By calling `get` on the struct
/// all operations will be executed in one run.
pub struct MultiOperation2<T, TO1, TO2> 
    where T: RealNumber,
        TO1: ToIdentifier<T>, 
        TO2: ToIdentifier<T> {
    a: GenericDataVec<T>,
    b: GenericDataVec<T>,
    prepared_ops: PreparedOperation2<T, GenericDataVec<T>, GenericDataVec<T>, TO1, TO2>
}

macro_rules! add_complex_multi_ops_impl {
    ($data_type:ident, $name: ident, $partner: ident)
     =>
     {    
        impl ComplexIdentifier<$data_type> for $name<$data_type> {
            type RealPartner = $partner<$data_type>;
            
            fn complex_offset(self, offset: Complex<$data_type>) -> Self {
                let arg = self.arg;
                self.add_op(Operation::AddComplex(arg, offset))
            }
            
            fn complex_scale(self, factor: Complex<$data_type>) -> Self {
                let arg = self.arg;
                self.add_op(Operation::MultiplyComplex(arg, factor))
            }
            
            fn magnitude(self) -> Self::RealPartner {
                let arg = self.arg;
                self.add_op(Operation::Magnitude(arg))
            }
            
            fn magnitude_squared(self) -> Self::RealPartner {
                let arg = self.arg;
                self.add_op(Operation::MagnitudeSquared(arg))
            }
            
            fn conj(self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::ComplexConj(arg))
            }
            
            fn to_real(self) -> Self::RealPartner {
                let arg = self.arg;
                self.add_op(Operation::ToReal(arg))
            }
            
            fn to_imag(self) -> Self::RealPartner {
                let arg = self.arg;
                self.add_op(Operation::ToImag(arg))
            }
            
            fn phase(self) -> Self::RealPartner {
                let arg = self.arg;
                self.add_op(Operation::Phase(arg))
            }
            
            fn multiply_complex_exponential(self, a: $data_type, b: $data_type) -> Self {
                let arg = self.arg;
                self.add_op(Operation::MultiplyComplexExponential(arg, a, b))
            }
            
            fn map_inplace_complex<F>(self, f: F) -> Self
                where F: Fn(Complex<$data_type>, usize) -> Complex<$data_type> + Send + Sync + 'static {
                let arg = self.arg;
                self.add_op(Operation::MapComplex(arg, Arc::new(f)))
            }
        }
        
        impl Scale<Complex<$data_type>> for $name<$data_type> {
            fn scale(self, factor: Complex<$data_type>) -> Self {
                let arg = self.arg;
                self.add_op(Operation::MultiplyComplex(arg, factor))
            }
        }
        
        impl Offset<Complex<$data_type>> for $name<$data_type> {
            fn offset(self, offset: Complex<$data_type>) -> Self {
                let arg = self.arg;
                self.add_op(Operation::AddComplex(arg, offset))
            }
        }
        
        impl IdentifierIter<Complex<$data_type>> for $name<$data_type> {
            fn map_inplace<F>(self, f: F) -> Self
                where F: Fn(Complex<$data_type>, usize) -> Complex<$data_type> + Send + Sync + 'static {
                let arg = self.arg;
                self.add_op(Operation::MapComplex(arg, Arc::new(f)))
            }
        }
     }
}     

macro_rules! add_real_multi_ops_impl {
    ($data_type:ident, $name: ident, $partner: ident)
     =>
     {    
        impl RealIdentifier<$data_type> for $name<$data_type> {
            type ComplexPartner = $partner<$data_type>;
            fn real_offset(self, offset: $data_type) -> Self {
                let arg = self.arg;
                self.add_op(Operation::AddReal(arg, offset))
            }
            
            fn real_scale(self, factor: $data_type) -> Self {
                let arg = self.arg;
                self.add_op(Operation::MultiplyReal(arg, factor))
            }
            
            fn abs(self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::Abs(arg))
            }
            
            fn to_complex(self) -> Self::ComplexPartner {
                let arg = self.arg;
                self.add_op(Operation::ToComplex(arg))
            }
            
            fn map_inplace_real<F>(self, f: F) -> Self
                where F: Fn($data_type, usize) -> $data_type + Send + Sync + 'static {
                let arg = self.arg;
                self.add_op(Operation::MapReal(arg, Arc::new(f)))
            }
        }
        
        impl Scale<$data_type> for $name<$data_type> {
            fn scale(self, factor: $data_type) -> Self {
                let arg = self.arg;
                self.add_op(Operation::MultiplyReal(arg, factor))
            }
        }
        
        impl Offset<$data_type> for $name<$data_type> {
            fn offset(self, offset: $data_type) -> Self {
                let arg = self.arg;
                self.add_op(Operation::AddReal(arg, offset))
            }
        }
        
        impl IdentifierIter<$data_type> for $name<$data_type> {
            fn map_inplace<F>(self, f: F) -> Self
                where F: Fn($data_type, usize) -> $data_type + Send + Sync + 'static {
                let arg = self.arg;
                self.add_op(Operation::MapReal(arg, Arc::new(f)))
            }
        }
     }
}   

macro_rules! add_general_multi_ops_impl {
    ($data_type:ident, $name: ident)
     =>
     {    
        impl GeneralIdentifier<$data_type> for $name<$data_type>
                where $data_type: RealNumber 
        {
            fn add_vector(self, summand: &Self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::AddVector(arg, summand.arg))
            }
            
            fn subtract_vector(self, subtrahend: &Self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::SubVector(arg, subtrahend.arg))
            }
            
            fn multiply_vector(self, factor: &Self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::MulVector(arg, factor.arg))
            }
            
            fn divide_vector(self, divisor: &Self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::DivVector(arg, divisor.arg))
            }
            
            fn sqrt(self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::Sqrt(arg))
            }
            
            fn square(self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::Square(arg))
            }
            
            fn root(self, degree: $data_type) -> Self {
                let arg = self.arg;
                self.add_op(Operation::Root(arg, degree))
            }
            
            fn powf(self, exponent: $data_type) -> Self {
                let arg = self.arg;
                self.add_op(Operation::Powf(arg, exponent))
            }
            
            fn ln(self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::Ln(arg))
            }
            
            fn exp(self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::Exp(arg))
            }
            
            fn log(self, base: $data_type) -> Self {
                let arg = self.arg;
                self.add_op(Operation::Log(arg, base))
            }
            
            fn expf(self, base: $data_type) -> Self {
                let arg = self.arg;
                self.add_op(Operation::Expf(arg, base))
            }
            
            fn sin(self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::Sin(arg))
            }
            
            fn cos(self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::Cos(arg))
            }
            
            fn tan(self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::Tan(arg))
            }
            
            fn asin(self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::ASin(arg))
            }
            
            fn acos(self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::ACos(arg))
            }
            
            fn atan(self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::ATan(arg))
            }
            
            fn sinh(self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::Sinh(arg))
            }
            
            fn cosh(self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::Cosh(arg))
            }
            
            fn tanh(self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::Tanh(arg))
            }
            
            fn asinh(self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::ASinh(arg))
            }
            
            fn acosh(self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::ACosh(arg))
            }
            
            fn atanh(self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::ATanh(arg))
            }
            
            fn clone_from(self, source: &Self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::CloneFrom(arg, source.arg))
            }
            
            fn add_points(self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::AddPoints(arg))
            }
            
            fn sub_points(self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::SubPoints(arg))
            }
            
            fn div_points(self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::DivPoints(arg))
            }
            
            fn mul_points(self) -> Self {
                let arg = self.arg;
                self.add_op(Operation::MulPoints(arg))
            }
        }
     }
}   


macro_rules! add_multi_ops_impl {
    ($data_type:ident, $reg:ident)
     =>
     {     
        /// An operation which can be prepared in advance and operates on two
        /// inputs and produces two outputs
        impl<TI1, TI2, TO1, TO2> PreparedOperation2<$data_type, TI1, TI2, TO1, TO2>
            where TI1: ToIdentifier<$data_type> + DataVec<$data_type> + RededicateVector<GenericDataVec<$data_type>>, 
            TI2: ToIdentifier<$data_type> + DataVec<$data_type> + RededicateVector<GenericDataVec<$data_type>>, 
            TO1: ToIdentifier<$data_type> + DataVec<$data_type> + RededicateVector<GenericDataVec<$data_type>>, 
            TO2: ToIdentifier<$data_type> + DataVec<$data_type> + RededicateVector<GenericDataVec<$data_type>>
        {
            /// Adds new operations which will be executed with the next call to `exec`
            /// 
            /// As a background: The function `operation` will be executed immediately. It only operated on `Identifier` types and these serve as
            /// placeholder for vectors. Every operation done to an `Identifier`
            /// is recorded and will be executed on vectors if `exec` is called.
            pub fn add_ops<F, TN1, TN2>(self, operation: F) 
                -> PreparedOperation2<$data_type, TI1, TI2, TN1::Vector, TN2::Vector>
                where F: Fn(TO1::Identifier, TO2::Identifier) -> (TN1, TN2),
                         TN1: Identifier<$data_type>,
                         TN2: Identifier<$data_type>
                
            {
                let mut ops = self.ops;
                let mut new_ops = Vec::new();
                let swap = 
                    {
                        let counter = Arc::new(Mutex::new(0));
                        let t1 = TO1::Identifier::new(if self.swap { ARGUMENT2 } else { ARGUMENT1 }, counter.clone());
                        let t2  = TO2::Identifier::new(if self.swap { ARGUMENT1 } else { ARGUMENT2 }, counter.clone());
                        let (r1, r2) = operation(t1, t2);
                        let r1arg = r1.get_arg();
                        new_ops.append(&mut r1.get_ops());
                        new_ops.append(&mut r2.get_ops());
                        (r1arg == ARGUMENT2)
                    };
                 // In theory the sequence counter could overflow.
                 // In practice if we assume that the counter is increment 
                 // every microsecond than the program needs to run for 500,000
                 // years until an overflow occurs.
                 new_ops.sort_by(|a, b| a.0.cmp(&b.0));
                 for v in new_ops {
                    ops.push(v.1);
                 }
                 PreparedOperation2 
                 { 
                    a: PhantomData,
                    b: PhantomData, 
                    c: PhantomData,
                    d: PhantomData, 
                    ops: ops, 
                    swap: swap
                 }
            }
            
            /// Executes all recorded operations on the input vectors.
            pub fn exec(&self, a: TI1, b: TI2) -> result::Result<(TO1, TO2), (ErrorReason, TO1, TO2)> {
                // First "cast" the vectors to generic vectors. This is done with the
                // the rededicate trait since in contrast to the to_gen method it 
                // can be used in a generic context.
                
                let a: GenericDataVec<$data_type> = a.rededicate();
                let b: GenericDataVec<$data_type> = b.rededicate();
                let mut vec = Vec::new();
                vec.push(a);
                vec.push(b);
                
                // at this point we would execute all ops and cast the result to the right types
                let result = GenericDataVec::<$data_type>::perform_operations(vec, &self.ops);
                
                if result.is_err() {
                    let err = result.unwrap_err();
                    let reason = err.0;
                    let mut vec = err.1;
                    let b = vec.pop().unwrap();
                    let a = vec.pop().unwrap();
                    return Err((
                        reason, 
                        TO1::rededicate_from(b),
                        TO2::rededicate_from(a)));
                }
                let mut vec = result.unwrap();
                let b = vec.pop().unwrap();
                let a = vec.pop().unwrap();
                
                // Convert back
                if self.swap {
                    Ok((TO1::rededicate_from(b), TO2::rededicate_from(a)))
                }
                else {
                    Ok((TO1::rededicate_from(a), TO2::rededicate_from(b)))
                }
            }
        }

        /// An operation which can be prepared in advance and operates on one
        /// input and produces one output
        impl<TI1, TO1> PreparedOperation1<$data_type, TI1, TO1>
            where TI1: ToIdentifier<$data_type> + DataVec<$data_type> + RededicateVector<GenericDataVec<$data_type>>,
                  TO1: ToIdentifier<$data_type> + DataVec<$data_type> + RededicateVector<GenericDataVec<$data_type>> {
               
            /// Extends the operation to operate on one more vector.
            pub fn extend<TI2>(self) 
                -> PreparedOperation2<$data_type, TI1, TI2, TO1, TI2>
                where TI2: ToIdentifier<$data_type> {
                PreparedOperation2 
                { 
                    a: PhantomData, 
                    b: PhantomData, 
                    c: PhantomData,
                    d: PhantomData, 
                    ops: self.ops, 
                    swap: false 
                }
            }
            
            /// Adds new operations which will be executed with the next call to `exec`
            /// 
            /// As a background: The function `operation` will be executed immediately. It only operated on `Identifier` types and these serve as
            /// placeholder for vectors. Every operation done to an `Identifier`
            /// is recorded and will be executed on vectors if `exec` is called.
            pub fn add_ops<F, TN>(self, operation: F) 
                -> PreparedOperation1<$data_type, TI1, TN::Vector>
                where F: Fn(TO1::Identifier) -> TN,
                         TN: Identifier<$data_type>
                
            {
                let mut ops = self.ops;
                let new_ops =
                {
                    let counter = Arc::new(Mutex::new(0));
                    let t1 = TO1::Identifier::new(ARGUMENT1, counter);
                    let r1 = operation(t1);
                    r1.get_ops()
                };
                 for v in new_ops {
                    ops.push(v.1);
                 }
                 PreparedOperation1 
                 { 
                    a: PhantomData,
                    b: PhantomData, 
                    ops: ops
                 }
            }
            
            /// Executes all recorded operations on the input vectors.
            pub fn exec(&self, a: TI1) -> result::Result<TO1, (ErrorReason, TO1)> {
                // First "cast" the vectors to generic vectors. This is done with the
                // the rededicate trait since in contrast to the to_gen method it 
                // can be used in a generic context.
                                   
                let a: GenericDataVec<$data_type> = a.rededicate();
                
                let mut vec = Vec::new();
                vec.push(a);
                
                // at this point we would execute all ops and cast the result to the right types
                let result = GenericDataVec::<$data_type>::perform_operations(vec, &self.ops);
                
                if result.is_err() {
                    let err = result.unwrap_err();
                    let reason = err.0;
                    let mut vec = err.1;
                    let a = vec.pop().unwrap();
                    return Err((
                        reason, 
                        TO1::rededicate_from(a)));
                }
                let mut vec = result.unwrap();
                let a = vec.pop().unwrap();
                
                // Convert back
                Ok(TO1::rededicate_from(a))
            }
        }
        
        add_complex_multi_ops_impl!($data_type, ComplexTimeIdentifier, RealTimeIdentifier);
        add_complex_multi_ops_impl!($data_type, ComplexFreqIdentifier, RealFreqIdentifier);
        add_complex_multi_ops_impl!($data_type, GenericDataIdentifier, GenericDataIdentifier);
        add_real_multi_ops_impl!($data_type, RealTimeIdentifier, ComplexTimeIdentifier);
        add_real_multi_ops_impl!($data_type, RealFreqIdentifier, ComplexFreqIdentifier);
        add_real_multi_ops_impl!($data_type, GenericDataIdentifier, GenericDataIdentifier);
        
        add_general_multi_ops_impl!($data_type, ComplexTimeIdentifier);
        add_general_multi_ops_impl!($data_type, ComplexFreqIdentifier);
        add_general_multi_ops_impl!($data_type, GenericDataIdentifier);
        add_general_multi_ops_impl!($data_type, RealTimeIdentifier);
        add_general_multi_ops_impl!($data_type, RealFreqIdentifier);

        /// Holds two vectors and records all operations which shall be done on the
        /// vectors. A call to `get` then runs all recorded operations on the vectors
        /// and returns them. See the modules description for why this can be beneficial.
        impl<TO1, TO2>  MultiOperation2<$data_type, TO1, TO2> 
            where TO1: ToIdentifier<$data_type> + DataVec<$data_type> + RededicateVector<GenericDataVec<$data_type>>, 
                  TO2: ToIdentifier<$data_type> + DataVec<$data_type> + RededicateVector<GenericDataVec<$data_type>> {
            /// Executes all recorded operations on the stored vector.
            pub fn get(self) -> result::Result<(TO1, TO2), (ErrorReason, TO1, TO2)> {
                self.prepared_ops.exec(self.a, self.b)
            }
            
            /// Adds new operations which will be executed with the next call to `get`
            /// 
            /// As a background: The function `operation` will be executed immediately. It only operated on `Identifier` types and these serve as
            /// placeholder for vectors. Every operation done to an `Identifier`
            /// is recorded and will be executed on vectors if `get` is called.
            pub fn add_ops<F, TN1, TN2>(self, operation: F) 
                -> MultiOperation2<$data_type, TN1::Vector, TN2::Vector>
                where F: Fn(TO1::Identifier, TO2::Identifier) -> (TN1, TN2),
                         TN1: Identifier<$data_type>,
                         TN2: Identifier<$data_type>
                
            {
                let ops = self.prepared_ops.add_ops(operation);
                MultiOperation2 { a: self.a, b: self.b, prepared_ops: ops }
            }
        }
        
        /// Holds a vector and records all operations which shall be done on the
        /// vector. A call to `get` then runs all recorded operations on the vector
        /// and returns it. See the modules description for why this can be beneficial.
        impl<TO>  MultiOperation1<$data_type, TO> 
            where TO: ToIdentifier<$data_type> + DataVec<$data_type> + RededicateVector<GenericDataVec<$data_type>> {
            
            /// Executes all recorded operations on the stored vectors.
            pub fn get(self) -> result::Result<(TO), (ErrorReason, TO)> {
                self.prepared_ops.exec(self.a)
            }
            
            /// Adds new operations which will be executed with the next call to `get`
            /// 
            /// As a background: The function `operation` will be executed immediately. It only operated on `Identifier` types and these serve as
            /// placeholder for vectors. Every operation done to an `Identifier`
            /// is recorded and will be executed on vectors if `get` is called.
            
            pub fn add_ops<F, TN>(self, operation: F) 
                -> MultiOperation1<$data_type, TN::Vector>
                where F: Fn(TO::Identifier) -> TN,
                         TN: Identifier<$data_type>
                
            {
                let ops = self.prepared_ops.add_ops(operation);
                MultiOperation1 { a: self.a, prepared_ops: ops }
            }
            
            /// Extends the operation to operate on one more vector.
            pub fn extend<TI2>(self, vector: TI2) 
                -> MultiOperation2<$data_type, TO, TI2>
                where TI2: ToIdentifier<$data_type> + DataVec<$data_type> + RededicateVector<GenericDataVec<$data_type>>  {
                let ops: PreparedOperation2<$data_type, GenericDataVec<$data_type>, GenericDataVec<$data_type>, TO, TI2> =           
                     PreparedOperation2 
                     { 
                        a: PhantomData,
                        b: PhantomData, 
                        c: PhantomData,
                        d: PhantomData, 
                        ops: self.prepared_ops.ops, 
                        swap: false
                     };
                let a: GenericDataVec<$data_type> = self.a;
                let b: GenericDataVec<$data_type> = vector.rededicate();
                MultiOperation2 { a: a, b: b, prepared_ops: ops }
            }
        }
        
        impl GenericDataVec<$data_type> {
            fn perform_operations(mut vectors: Vec<Self>, operations: &[Operation<$data_type>])
                -> TransRes<Vec<Self>>
            {
                if vectors.len() == 0
                {
                    return Err((ErrorReason::InvalidNumberOfArgumentsForCombinedOp, vectors));
                }
                           
                let verifcation = Self::verify_ops(&vectors, operations);
                if verifcation.is_err() {
                    return Err((verifcation.unwrap_err(), vectors));
                }
                
                let (any_complex_ops, final_number_space) = verifcation.unwrap();
            
                if operations.len() == 0
                {
                    return Ok(vectors);
                }
                
                // All vectors are required to have the same length
                let first_vec_len = vectors[0].points();
                let mut has_errors = false;
                for v in &vectors {
                    if v.points() != first_vec_len {
                        has_errors = true;
                    }
                }
                
                if has_errors {
                    return Err((ErrorReason::InvalidNumberOfArgumentsForCombinedOp, vectors));
                }
                
                // If the vectors needs to be complex at some point during the
                // calculation the transform them to complex right away.
                // This might be unnecessary in some cases, but covering
                // complex/real mixed cases will cause more implementation effort
                // later in the process so we better keep it simple for now.
                if any_complex_ops {
                    let mut complex_vectors = Vec::with_capacity(vectors.len());
                    for _ in 0..vectors.len() {
                        let vector = vectors.pop().unwrap();
                        let as_complex = 
                            if vector.is_complex() { vector }
                            else { vector.to_complex().unwrap() };
                        complex_vectors.push(as_complex);
                    }
                    
                    complex_vectors.reverse();
                    vectors = complex_vectors;
                }
                
                let (vectorization_length, 
                     multicore_settings,
                     scalar_length) =
                    {
                        let first = &vectors[0];
                        let data_length = first.len();
                        let alloc_len = first.allocated_len();
                        let rounded_len = round_len(data_length);
                        if rounded_len <= alloc_len {
                            (rounded_len, first.multicore_settings, 0)
                        }
                        else {
                            let scalar_length = data_length % $reg::len();
                            (data_length - scalar_length, first.multicore_settings, scalar_length)
                        }
                    };
                
                if vectorization_length > 0 {
                    let complexity = if operations.len() > 5 { Complexity::Large } else { Complexity::Medium };
                    {
                        let mut array: Vec<&mut [$data_type]> = vectors.iter_mut().map(|v| {
                            let len = v.len();
                            &mut v.data[0..len]
                        }).collect();
                        let range = Range { start: 0, end: vectorization_length };
                        if any_complex_ops {
                            Chunk::execute_partial_multidim(
                                complexity, &multicore_settings,
                                &mut array, range, $reg::len(), 
                                (operations, first_vec_len), 
                                Self::perform_complex_operations_par);
                        }
                        else {
                            Chunk::execute_partial_multidim(
                                complexity, &multicore_settings,
                                &mut array, range, $reg::len(), 
                                (operations, first_vec_len), 
                                Self::perform_real_operations_par);   
                        }
                    }
                }
                
                if scalar_length > 0 {
                    let mut last_elems = Vec::with_capacity(vectors.len());
                    for j in 0..vectors.len() {
                        let reg = $reg::splat(0.0);
                        let mut reg_array = reg.to_array();
                        for i in 0..scalar_length {
                            reg_array[i] = vectors[j].data[vectorization_length + i];
                        }
                        
                        last_elems.push($reg::from_array(reg_array));
                    }
                    
                    if any_complex_ops {
                        for operation in operations
                        {
                            PerformOperationSimd::<$data_type>::perform_complex_operation(
                                &mut last_elems, 
                                operation,
                                (vectorization_length / $reg::len() * 2),
                                first_vec_len);
                        }
                    }
                    else {
                        for operation in operations
                        {
                            PerformOperationSimd::<$data_type>::perform_real_operation(
                                &mut last_elems, 
                                operation,
                                (vectorization_length / $reg::len()),
                                first_vec_len);
                        }
                    }
                    
                    let mut j = 0;
                    for reg in last_elems {
                        let reg_array = reg.to_array();
                        for i in 0..scalar_length {
                            vectors[j].data[vectorization_length + i] = reg_array[i];
                        }
                        j += 1;
                    }
                }
                
                // In case we converted vectors at the beginning to complex
                // we might now need to convert them back to real.
                if any_complex_ops {
                    let mut correct_domain = Vec::with_capacity(vectors.len());
                    vectors.reverse();
                    for i in 0..vectors.len() {
                        let vector = vectors.pop().unwrap();
                        let right_domain = 
                            if final_number_space[i] { vector }
                            else { vector.to_real().unwrap() };
                        correct_domain.push(right_domain);
                    }
                    
                    vectors = correct_domain;
                }
                
                Ok (vectors)
            }
            
            fn perform_complex_operations_par(
                array: &mut Vec<&mut [$data_type]>, 
                range: Range<usize>,
                arguments: (&[Operation<$data_type>], usize))
            {
                let (operations, points) = arguments;
                let mut vectors = Vec::with_capacity(array.len());
                for _ in 0..array.len() {
                   vectors.push($reg::splat(0.0));
                }
            
                let reg_len = $reg::len() / 2;
                let mut index = range.start / 2;
                let mut i =0;
                while i < array[0].len() {
                    for j in 0..array.len() {
                        unsafe {
                            let elem = vectors.get_unchecked_mut(j);
                            *elem = $reg::load_unchecked(array.get_unchecked(j), i)
                        }
                    }
                
                    for operation in operations
                    {
                        PerformOperationSimd::<$data_type>::perform_complex_operation(
                            &mut vectors, 
                            operation,
                            index,
                            points);
                    }
                    
                    for j in 0..array.len() {
                        unsafe {
                            vectors.get_unchecked(j).store_unchecked(array.get_unchecked_mut(j), i);
                        }
                    }
                    
                    index += reg_len;
                    i += $reg::len();
                }
            }
            
            fn perform_real_operations_par(
                array: &mut Vec<&mut [$data_type]>,
                range: Range<usize>,
                arguments: (&[Operation<$data_type>], usize))
            {
                let (operations, points) = arguments;
                let mut vectors = Vec::with_capacity(array.len());
                for _ in 0..array.len() {
                   vectors.push($reg::splat(0.0));
                }
            
                let reg_len = $reg::len();
                let mut index = range.start;
                let mut i =0;
                while i < array[0].len() {
                for j in 0..array.len() {
                        unsafe {
                            let elem = vectors.get_unchecked_mut(j);
                            *elem = $reg::load_unchecked(array.get_unchecked(j), i)
                        }
                    }
                
                    for operation in operations
                    {
                        PerformOperationSimd::<$data_type>::perform_real_operation(
                            &mut vectors, 
                            operation,
                            index,
                            points);
                    }
                    
                    for j in 0..array.len() {
                        unsafe {
                            vectors.get_unchecked(j).store(array.get_unchecked_mut(j), i);
                        }
                    }
                    
                    index += reg_len;
                    i += $reg::len();
                }
            }
            
            fn verify_ops(vectors: &[Self], operations: &[Operation<$data_type>]) -> Result<(bool, Vec<bool>), ErrorReason> {
                let mut complex: Vec<bool> = vectors.iter().map(|v|v.is_complex()).collect();
                let max_arg_num =  vectors.len();
                let mut complex_at_any_moment = complex.clone();
                for op in operations {
                    let index = get_argument(op);
                    
                    if index >= max_arg_num {
                        return Err(ErrorReason::InvalidNumberOfArgumentsForCombinedOp);
                    }
                    
                    let eval = evaluate_number_space_transition(complex[index], op);
                    let complex_after_op = match eval {
                        Err(reason) => { return Err(reason) }
                        Ok(new_complex) => { new_complex }
                    };
                    
                    complex[index] = complex_after_op;
                    if complex_after_op {
                        complex_at_any_moment[index] = complex_after_op;
                    }
                }
                
                Ok((complex_at_any_moment.iter().any(|c|*c), complex))
            }
        }
     }
}        
add_multi_ops_impl!(f32, Reg32);
add_multi_ops_impl!(f64, Reg64);

impl<T> PreparedOperation1<
    T, 
    GenericDataVec<T>, GenericDataVec<T>>
            where T: RealNumber
{
    /// Allows to directly push an `Operation` enum to a `PreparedOperation1`.
    /// This mainly exists as interop between Rust and other languages.
    pub fn add_enum_op(&mut self, op: Operation<T>) {
        self.ops.push(op);
    }
}

impl<T> PreparedOperation2<
    T, 
    GenericDataVec<T>, GenericDataVec<T>, 
    GenericDataVec<T>, GenericDataVec<T>>
            where T: RealNumber
{
    /// Allows to directly push an `Operation` enum to a `PreparedOperation2`.
    /// This mainly exists as interop between Rust and other languages.
    pub fn add_enum_op(&mut self, op: Operation<T>) {
        self.ops.push(op);
    }
}

impl<T> MultiOperation2<
    T, 
    GenericDataVec<T>, GenericDataVec<T>>
            where T: RealNumber
{
    /// Allows to directly push an `Operation` enum to a `MultiOperation2`.
    /// This mainly exists as interop between Rust and other languages.
    pub fn add_enum_op(&mut self, op: Operation<T>) {
        self.prepared_ops.add_enum_op(op);
    }
}

impl<T> MultiOperation1<
    T, 
    GenericDataVec<T>>
            where T: RealNumber
{
    /// Allows to directly push an `Operation` enum to a `MultiOperation1`.
    /// This mainly exists as interop between Rust and other languages.
    pub fn add_enum_op(&mut self, op: Operation<T>) {
        self.prepared_ops.add_enum_op(op);
    }
}

/// Prepares an operation with one input and one output.
pub fn prepare1<T, A>()
    -> PreparedOperation1<T, A, A> 
    where 
        T: RealNumber,
        A: ToIdentifier<T> {
    PreparedOperation1 
         { 
            a: PhantomData,
            b: PhantomData, 
            ops: Vec::new()
         }
}

/// Prepares an operation with two inputs and two outputs.
pub fn prepare2<T, A, B>()
    -> PreparedOperation2<T, A, B, A, B> 
    where 
        T: RealNumber,
        A: ToIdentifier<T>, 
        B: ToIdentifier<T> {
    PreparedOperation2 
         { 
            a: PhantomData,
            b: PhantomData, 
            c: PhantomData,
            d: PhantomData, 
            ops: Vec::new(), 
            swap: false
         }
}

/// Creates a new multi operation for one vectors.
pub fn multi_ops1<T, A>(a: A)
    -> MultiOperation1<T, A>
    where 
        T: RealNumber,
        A: ToIdentifier<T> + DataVec<T> + RededicateVector<GenericDataVec<T>> {
    let ops: PreparedOperation1<T, GenericDataVec<T>, A> =           
         PreparedOperation1 
         { 
            a: PhantomData,
            b: PhantomData, 
            ops: Vec::new()
         };
    let a: GenericDataVec<T> = a.rededicate();
    MultiOperation1 { a: a, prepared_ops: ops }
}


/// Creates a new multi operation for two vectors.
pub fn multi_ops2<T, A, B>(a: A, b: B)
    -> MultiOperation2<T, A, B>
    where 
        T: RealNumber,
        A: ToIdentifier<T> + DataVec<T> + RededicateVector<GenericDataVec<T>>, 
        B: ToIdentifier<T> + DataVec<T> + RededicateVector<GenericDataVec<T>> {
    let ops: PreparedOperation2<T, GenericDataVec<T>, GenericDataVec<T>, A, B> =           
         PreparedOperation2 
         { 
            a: PhantomData,
            b: PhantomData, 
            c: PhantomData,
            d: PhantomData, 
            ops: Vec::new(), 
            swap: false
         };
    let a: GenericDataVec<T> = a.rededicate();
    let b: GenericDataVec<T> = b.rededicate();
    MultiOperation2 { a: a, b: b, prepared_ops: ops }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    use num::complex::Complex32;

    #[test]
    fn multi_ops_construction()
    {
        // This test case tests mainly the syntax and less the 
        // runtime results.
        let array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a = RealTimeVector32::from_array(&array);
        let b = RealTimeVector32::from_array(&array);
        
        let ops = multi_ops2(a, b);
        let ops = ops.add_ops(|a, b| (a, b));
        let (a, _) = ops.get().unwrap();
        
        assert_eq!(a.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }
    
    #[test]
    fn prepared_ops_construction()
    {
        // This test case tests mainly the syntax and less the 
        // runtime results.
        let ops = prepare2::<f32, RealTimeVector32, RealTimeVector32>();
        let ops = ops.add_ops(|a, b| (a, b));
        
        let array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a = RealTimeVector32::from_array(&array);
        let b = RealTimeVector32::from_array(&array);
        let (a, _) = ops.exec(a, b).unwrap();
        
        assert_eq!(a.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }
    
    #[test]
    fn swapping()
    {
        let ops = prepare2::<f32, ComplexTimeVector32, RealTimeVector32>();
        let ops = ops.add_ops(|a, b| (a, b));
        
        let array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a = ComplexTimeVector32::from_interleaved(&array);
        let b = RealTimeVector32::from_array(&array);
        assert_eq!(a.is_complex(), true);
        let (a, b) = ops.exec(a, b).unwrap();
        // Check the compile time types with the assignments
        let c: ComplexTimeVector32 = a; 
        let d: RealTimeVector32 = b;
        assert_eq!(c.is_complex(), true);
        assert_eq!(d.is_complex(), false);
        assert_eq!(d.len(), 8);
        
        let ops = ops.add_ops(|a, b| (b, a));
        let a = ComplexTimeVector32::from_interleaved(&array);
        let b = RealTimeVector32::from_array(&array);
        assert_eq!(a.is_complex(), true);
        let (b, a) = ops.exec(a, b).unwrap();
        // Check the compile time types with the assignments
        let c: ComplexTimeVector32 = a; 
        let d: RealTimeVector32 = b;
        assert_eq!(c.is_complex(), true);
        assert_eq!(d.is_complex(), false);
        assert_eq!(d.len(), 8);
    }
    
    #[test]
    fn swap_twice()
    {
        let ops = prepare2::<f32, ComplexTimeVector32, RealTimeVector32>();
        let ops = ops.add_ops(|a, b| (b, a));
        let ops = ops.add_ops(|a, b| (b, a));
                
        let array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a = ComplexTimeVector32::from_interleaved(&array);
        let b = RealTimeVector32::from_array(&array);
        assert_eq!(a.is_complex(), true);
        let (a, b) = ops.exec(a, b).unwrap();
        // Check the compile time types with the assignments
        let c: ComplexTimeVector32 = a; 
        let d: RealTimeVector32 = b;
        assert_eq!(c.is_complex(), true);
        assert_eq!(d.is_complex(), false);
        assert_eq!(d.len(), 8);
    }
    
    #[test]
    fn complex_operation_on_real_vector()
    {
        let array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a = DataVec32::from_array(false, DataVecDomain::Time, &array);
        let b = DataVec32::from_array(false, DataVecDomain::Time, &array);
        
        let ops = multi_ops2(a, b);
        let ops = ops.add_ops(|a, b| (b.magnitude(), a));
        let res = ops.get();
        assert!(res.is_err());
    }
    
    #[test]
    fn simple_operation()
    {
        let ops = prepare2::<f32, ComplexTimeVector32, RealTimeVector32>();
        let ops = ops.add_ops(|a, b| (b.abs(), a));
        let ops = ops.add_ops(|a, b| (b, a.abs()));
                
        let array = [1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0,
                     1.0, -2.0, 3.0, 4.0, 5.0, -6.0, -7.0, 8.0];
        let a = ComplexTimeVector32::from_interleaved(&array);
        let array = [-11.0, 12.0, -13.0, 14.0, -15.0, 16.0, -17.0, 18.0];
        let b = RealTimeVector32::from_array(&array);
        assert_eq!(a.is_complex(), true);
        let (a, b) = ops.exec(a, b).unwrap();
        assert_eq!(a.is_complex(), true);
        assert_eq!(b.is_complex(), false);
        let expected = [1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0,
                     1.0, -2.0, 3.0, 4.0, 5.0, -6.0, -7.0, 8.0];
        assert_eq!(a.data(), &expected);   
        let expected = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0];
        assert_eq!(b.data(), &expected);            
    }
    
    /// Same as `simple_operation` but the arguments are passed in reversed order.
    #[test]
    fn simple_operation2()
    {
        let ops = prepare2::<f32, RealTimeVector32, ComplexTimeVector32>();
        let ops = ops.add_ops(|a, b| (b, a.abs()));
        let ops = ops.add_ops(|a, b| (b.abs(), a));
                
        let array = [1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0,
                     1.0, -2.0, 3.0, 4.0, 5.0, -6.0, -7.0, 8.0];
        let a = ComplexTimeVector32::from_interleaved(&array);
        let array = [-11.0, 12.0, -13.0, 14.0, -15.0, 16.0, -17.0, 18.0];
        let b = RealTimeVector32::from_array(&array);
        assert_eq!(a.is_complex(), true);
        let (b, a) = ops.exec(b, a).unwrap();
        assert_eq!(a.is_complex(), true);
        assert_eq!(b.is_complex(), false);
        let expected = [1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0,
                     1.0, -2.0, 3.0, 4.0, 5.0, -6.0, -7.0, 8.0];
        assert_eq!(a.data(), &expected);   
        let expected = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0];
        assert_eq!(b.data(), &expected);            
    }
    
    #[test]
    fn map_inplace_real_test()
    {
        let array = [1.0, 2.0, 3.0, 4.0];
        let a = RealTimeVector32::from_array(&array);
        let ops = multi_ops1(a);
        let ops = ops.add_ops(|a|a.map_inplace_real(|v,i|v * i as f32));
        let a = ops.get().unwrap();
        let expected = [0.0, 2.0, 6.0, 12.0];
        assert_eq!(a.data(), &expected); 
    }
    
    /// This test checks mainly if the closure we store in the Operation enum
    /// can still reference its scope.
    #[test]
    fn multiply_linear_test()
    {
        let array = [1.0, 2.0, 3.0, 4.0];
        let a = RealTimeVector32::from_array(&array);
        let a = multiply_linear(a, 0.5);
        let expected = [0.0, 1.0, 3.0, 6.0];
        assert_eq!(a.data(), &expected); 
    }
    
    fn multiply_linear(a: RealTimeVector32, fac: f32) -> RealTimeVector32 {
        let ops = multi_ops1(a);
        let ops = ops.add_ops(|a|a.map_inplace_real(|v,i|v * i as f32 * fac));
        ops.get().unwrap()
    }
    
    #[test]
    fn map_inplace_complex_test()
    {
        let array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = ComplexTimeVector32::from_interleaved(&array);
        let ops = multi_ops1(a);
        let ops = ops.add_ops(|a|a.map_inplace_complex(|v,i|v * Complex32::new(i as f32, 0.0)));
        let a = ops.get().unwrap();
        let expected = [0.0, 0.0, 3.0, 4.0, 10.0, 12.0];
        assert_eq!(a.data(), &expected); 
    }
}