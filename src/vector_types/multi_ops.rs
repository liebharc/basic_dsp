use std::marker::PhantomData;
use std::result;
use super::super::RealNumber;
use super::{
    round_len,
    DataVector,
    VecResult,
    ErrorReason,
    RealTimeVector,
    ComplexTimeVector,
    RealFreqVector,
    ComplexFreqVector,
    RealVectorOps,
    ComplexVectorOps,
    GenericDataVector,
    RededicateVector};  
use super::operations_enum::{
    Operation,
    evaluate_number_space_transition,
    get_argument,
    PerformOperationSimd};
use multicore_support::{Chunk, Complexity};
use simd_extensions::{Simd, Reg32, Reg64};
use num::Complex;

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
    type Vector: DataVector<T> + ToIdentifier<T>;
    fn get_arg(&self) -> usize;
    fn get_ops(self) -> Vec<(u64, Operation<T>)>;
    fn new(arg: usize) -> Self;
    fn new_ops(ops: Vec<(u64, Operation<T>)>, arg: usize) -> Self;
}

pub trait ComplexIdentifier<T> : Identifier<T>
    where T: RealNumber 
{
    type RealPartner;
    fn complex_offset(self, offset: Complex<T>) -> Self;
    fn complex_scale(self, factor: Complex<T>) -> Self;
    fn magnitude(self) -> Self::RealPartner;
    fn magnitude_squared(self) -> Self::RealPartner;
    fn conj(self) -> Self;
    fn to_real(self) -> Self::RealPartner;
    fn to_imag(self) -> Self::RealPartner;
    fn phase(self) -> Self::RealPartner;
}

pub trait RealIdentifier<T> : Identifier<T>
    where T: RealNumber 
{
    type ComplexPartner;
    fn real_offset(self, offset: T) -> Self;
    fn real_scale(self, factor: T) -> Self;
    fn abs(self) -> Self;
    fn to_complex(self) -> Self::ComplexPartner;
}

pub trait GeneralIdentifier<T> : Identifier<T>
    where T: RealNumber 
{
    fn add_vector(self, summand: &Self) -> Self;
    fn subtract_vector(self, subtrahend: &Self) -> Self;
    fn multiply_vector(self, factor: &Self) -> Self;
    fn divide_vector(self, divisor: &Self) -> Self;
    fn sqrt(self) -> Self;
    fn square(self) -> Self;
    fn root(self, degree: T) -> Self;
    fn powf(self, exponent: T) -> Self;
    fn ln(self) -> Self;
    fn exp(self) -> Self;
    fn log(self, base: T) -> Self;
    fn expf(self, base: T) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn asinh(self) -> Self;
    fn acosh(self) -> Self;
    fn atanh(self) -> Self;
}

pub trait Scale<T>: Sized where T: Sized {
    fn scale(self, factor: T) -> Self;
}

pub trait Offset<T>: Sized where T: Sized {
    fn offset(self, offset: T) -> Self;
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
                ops: Vec<(u64, Operation<T>)>
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
                fn new(arg: usize) -> Self { $name { ops: Vec::new(), arg: arg } }
                fn new_ops(ops: Vec<(u64, Operation<T>)>, arg: usize) -> Self { $name { ops: ops, arg: arg } }
            }
            
            impl<T> $name<T> 
                where T: RealNumber {
                fn add_op<R>(self, op: Operation<T>) -> R 
                    where R: Identifier<T> {
                    let mut ops = self.ops;
                    // TODO: Rethink if this unsafe is sound.
                    // Quick thinking was that the actual value in the sequence
                    // doesn't matter as long as repeated calls by the same thread
                    // produce an increasing value (where the distance between two numbers
                    // never matters). Even with race conditions between multiple threads
                    // this should be okay (counter overflows - while unlikely - need to be 
                    // handled during sorting).
                    let seq = unsafe {
                        OPERATION_SEQ_COUNTER += 1;
                        OPERATION_SEQ_COUNTER
                    };
                    ops.push((seq, op));
                    R::new_ops(ops, self.arg)
                }
            }
        )*
     }
}

create_identfier!(
    GenericDataVector, GenericDataIdentifier; 
    RealTimeVector, RealTimeIdentifier; 
    ComplexTimeVector, ComplexTimeIdentifier; 
    RealFreqVector, RealFreqIdentifier; 
    ComplexFreqVector, ComplexFreqIdentifier;);

static mut OPERATION_SEQ_COUNTER : u64 = 0;

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
/// which need to be done to the vectos. By calling `get` on the struct
/// all operations will be executed in one run.
pub struct MultiOperation1<T, TO> 
    where T: RealNumber,
        TO: ToIdentifier<T> {
    a: GenericDataVector<T>,
    prepared_ops: PreparedOperation1<T, GenericDataVector<T>, TO>
}

/// A multi operation which holds two vectors and records all changes
/// which need to be done to the vectors. By calling `get` on the struct
/// all operations will be executed in one run.
pub struct MultiOperation2<T, TO1, TO2> 
    where T: RealNumber,
        TO1: ToIdentifier<T>, 
        TO2: ToIdentifier<T> {
    a: GenericDataVector<T>,
    b: GenericDataVector<T>,
    prepared_ops: PreparedOperation2<T, GenericDataVector<T>, GenericDataVector<T>, TO1, TO2>
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
        }
     }
}   


macro_rules! add_multi_ops_impl {
    ($data_type:ident, $reg:ident)
     =>
     {     
        impl<TI1, TI2, TO1, TO2> PreparedOperation2<$data_type, TI1, TI2, TO1, TO2>
            where TI1: ToIdentifier<$data_type> + DataVector<$data_type> + RededicateVector<GenericDataVector<$data_type>>, 
            TI2: ToIdentifier<$data_type> + DataVector<$data_type> + RededicateVector<GenericDataVector<$data_type>>, 
            TO1: ToIdentifier<$data_type> + DataVector<$data_type> + RededicateVector<GenericDataVector<$data_type>>, 
            TO2: ToIdentifier<$data_type> + DataVector<$data_type> + RededicateVector<GenericDataVector<$data_type>>
        {
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
                        let t1 = TO1::Identifier::new(if self.swap { ARGUMENT2 } else { ARGUMENT1 });
                        let t2  = TO2::Identifier::new(if self.swap { ARGUMENT1 } else { ARGUMENT2 });
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
            
            pub fn exec(&self, a: TI1, b: TI2) -> result::Result<(TO1, TO2), (ErrorReason, TO1, TO2)> {
                // First "cast" the vectors to generic vectors. This is done with the
                // the rededicate trait since in contrast to the to_gen method it 
                // can be used in a generic context.
                
                let a: GenericDataVector<$data_type> = a.rededicate();
                let b: GenericDataVector<$data_type> = b.rededicate();
                let mut vec = Vec::new();
                vec.push(a);
                vec.push(b);
                
                // at this point we would execute all ops and cast the result to the right types
                let result = GenericDataVector::<$data_type>::perform_operations(vec, &self.ops);
                
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

        impl<TI1, TO1> PreparedOperation1<$data_type, TI1, TO1>
            where TI1: ToIdentifier<$data_type> + DataVector<$data_type> + RededicateVector<GenericDataVector<$data_type>>,
                  TO1: ToIdentifier<$data_type> + DataVector<$data_type> + RededicateVector<GenericDataVector<$data_type>> {
                  
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
            
            pub fn add_ops<F, TN>(self, operation: F) 
                -> PreparedOperation1<$data_type, TI1, TN::Vector>
                where F: Fn(TO1::Identifier) -> TN,
                         TN: Identifier<$data_type>
                
            {
                let mut ops = self.ops;
                let new_ops =
                {
                    let t1 = TO1::Identifier::new(ARGUMENT1);
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
            
            pub fn exec(&self, a: TI1) -> result::Result<TO1, (ErrorReason, TO1)> {
                // First "cast" the vectors to generic vectors. This is done with the
                // the rededicate trait since in contrast to the to_gen method it 
                // can be used in a generic context.
                                   
                let a: GenericDataVector<$data_type> = a.rededicate();
                
                let mut vec = Vec::new();
                vec.push(a);
                
                // at this point we would execute all ops and cast the result to the right types
                let result = GenericDataVector::<$data_type>::perform_operations(vec, &self.ops);
                
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
        add_real_multi_ops_impl!($data_type, GenericDataIdentifier, RealTimeIdentifier);
        
        add_general_multi_ops_impl!($data_type, ComplexTimeIdentifier);
        add_general_multi_ops_impl!($data_type, ComplexFreqIdentifier);
        add_general_multi_ops_impl!($data_type, GenericDataIdentifier);
        add_general_multi_ops_impl!($data_type, RealTimeIdentifier);
        add_general_multi_ops_impl!($data_type, RealFreqIdentifier);

        impl<TO1, TO2>  MultiOperation2<$data_type, TO1, TO2> 
            where TO1: ToIdentifier<$data_type> + DataVector<$data_type> + RededicateVector<GenericDataVector<$data_type>>, 
                  TO2: ToIdentifier<$data_type> + DataVector<$data_type> + RededicateVector<GenericDataVector<$data_type>> {
            pub fn get(self) -> result::Result<(TO1, TO2), (ErrorReason, TO1, TO2)> {
                self.prepared_ops.exec(self.a, self.b)
            }
            
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
        
        impl<TO>  MultiOperation1<$data_type, TO> 
            where TO: ToIdentifier<$data_type> + DataVector<$data_type> + RededicateVector<GenericDataVector<$data_type>> {
            pub fn get(self) -> result::Result<(TO), (ErrorReason, TO)> {
                self.prepared_ops.exec(self.a)
            }
            
            pub fn add_ops<F, TN>(self, operation: F) 
                -> MultiOperation1<$data_type, TN::Vector>
                where F: Fn(TO::Identifier) -> TN,
                         TN: Identifier<$data_type>
                
            {
                let ops = self.prepared_ops.add_ops(operation);
                MultiOperation1 { a: self.a, prepared_ops: ops }
            }
            
            pub fn extend<TI2>(self, vector: TI2) 
                -> MultiOperation2<$data_type, TO, TI2>
                where TI2: ToIdentifier<$data_type> + DataVector<$data_type> + RededicateVector<GenericDataVector<$data_type>>  {
                let ops: PreparedOperation2<$data_type, GenericDataVector<$data_type>, GenericDataVector<$data_type>, TO, TI2> =           
                     PreparedOperation2 
                     { 
                        a: PhantomData,
                        b: PhantomData, 
                        c: PhantomData,
                        d: PhantomData, 
                        ops: self.prepared_ops.ops, 
                        swap: false
                     };
                let a: GenericDataVector<$data_type> = self.a;
                let b: GenericDataVector<$data_type> = vector.rededicate();
                MultiOperation2 { a: a, b: b, prepared_ops: ops }
            }
        }
        
        impl GenericDataVector<$data_type> {
            fn perform_operations(mut vectors: Vec<Self>, operations: &[Operation<$data_type>])
                -> VecResult<Vec<Self>>
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
                
                let (vectorization_length, multicore_settings) =
                    {
                        let first = &vectors[0];
                        let data_length = first.len();
                        let alloc_len = first.allocated_len();
                        let rounded_len = round_len(data_length);
                        if rounded_len <= alloc_len {
                            (rounded_len, first.multicore_settings)
                        }
                        else {
                            let scalar_length = data_length % $reg::len();
                            if scalar_length > 0
                            {
                                panic!("perform_operations requires right now that the array length is dividable by {}", $reg::len())
                            }
                            // TODO We can possibly remove this restriction by copying
                            // the last elements into a $eg
                            // and then we can call perform_operation on it.
                            // That's of course not optimal since it involves a copying operation
                            // and in the worst case we don't care for $reg::len() -1 of the results
                            // however it makes the implementation easy and the performance
                            // loss seems to be tolerable
                            (data_length - scalar_length, first.multicore_settings)
                        }
                    };
                
                let complexity = if operations.len() > 5 { Complexity::Large } else { Complexity::Medium };
                {
                    let mut array: Vec<&mut [$data_type]> = vectors.iter_mut().map(|v| {
                        let len = v.len();
                        &mut v.data[0..len]
                    }).collect();
                    if any_complex_ops {
                        Chunk::execute_partial_multidim(
                            complexity, &multicore_settings,
                            &mut array, vectorization_length, $reg::len(), 
                            operations, 
                            Self::perform_complex_operations_par);
                    }
                    else {
                        Chunk::execute_partial_multidim(
                            complexity, &multicore_settings,
                            &mut array, vectorization_length, $reg::len(), 
                            operations, 
                            Self::perform_real_operations_par);
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
            
            fn perform_complex_operations_par(mut array: Vec<&mut [$data_type]>, operations: &[Operation<$data_type>])
            {
                let mut regs: Vec<&mut [$reg]> = 
                    array
                        .iter_mut()
                        .map(|a|$reg::array_to_regs_mut(a))
                        .collect();
                let mut vectors = Vec::with_capacity(regs.len());
                for j in 0..regs.len() {
                   vectors.push(regs[j][0]);
                }
            
                for i in 0..regs[0].len()
                { 
                    for j in 0..regs.len() {
                        unsafe {
                            let elem = vectors.get_unchecked_mut(j);
                            *elem = *regs.get_unchecked(j).get_unchecked(i);
                        }
                    }
                
                    for operation in operations
                    {
                        PerformOperationSimd::<$data_type>::perform_complex_operation(
                            &mut vectors, 
                            *operation);
                    }
                    
                    for j in 0..regs.len() {
                        unsafe {
                            let elem = regs.get_unchecked_mut(j).get_unchecked_mut(i);
                            *elem = *vectors.get_unchecked(j);
                        }
                    }
                }
            }
            
            fn perform_real_operations_par(mut array: Vec<&mut [$data_type]>, operations: &[Operation<$data_type>])
            {
                let mut regs: Vec<&mut [$reg]> = 
                    array
                        .iter_mut()
                        .map(|a|$reg::array_to_regs_mut(a))
                        .collect();
                let mut vectors = Vec::with_capacity(regs.len());
                for j in 0..regs.len() {
                   vectors.push(regs[j][0]);
                }
            
                for i in 0..regs[0].len()
                { 
                    for j in 0..regs.len() {
                        unsafe {
                            let elem = vectors.get_unchecked_mut(j);
                            *elem = *regs.get_unchecked(j).get_unchecked(i);
                        }
                    }
                
                    for operation in operations
                    {
                        PerformOperationSimd::<$data_type>::perform_real_operation(
                            &mut vectors, 
                            *operation);
                    }
                    
                    for j in 0..regs.len() {
                        unsafe {
                            let elem = regs.get_unchecked_mut(j).get_unchecked_mut(i);
                            *elem = *vectors.get_unchecked(j);
                        }
                    }
                }
            }
            
            fn verify_ops(vectors: &[Self], operations: &[Operation<$data_type>]) -> Result<(bool, Vec<bool>), ErrorReason> {
                let mut complex: Vec<bool> = vectors.iter().map(|v|v.is_complex()).collect();
                let max_arg_num =  vectors.len();
                let mut complex_at_any_moment = complex.clone();
                for op in operations {
                    let index = get_argument(*op);
                    
                    if index >= max_arg_num {
                        return Err(ErrorReason::InvalidNumberOfArgumentsForCombinedOp);
                    }
                    
                    let eval = evaluate_number_space_transition(complex[index], *op);
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
    GenericDataVector<T>, GenericDataVector<T>>
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
    GenericDataVector<T>, GenericDataVector<T>, 
    GenericDataVector<T>, GenericDataVector<T>>
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
    GenericDataVector<T>, GenericDataVector<T>>
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
    GenericDataVector<T>>
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
        A: ToIdentifier<T> + DataVector<T> + RededicateVector<GenericDataVector<T>> {
    let ops: PreparedOperation1<T, GenericDataVector<T>, A> =           
         PreparedOperation1 
         { 
            a: PhantomData,
            b: PhantomData, 
            ops: Vec::new()
         };
    let a: GenericDataVector<T> = a.rededicate();
    MultiOperation1 { a: a, prepared_ops: ops }
}


/// Creates a new multi operation for two vectors.
pub fn multi_ops2<T, A, B>(a: A, b: B)
    -> MultiOperation2<T, A, B>
    where 
        T: RealNumber,
        A: ToIdentifier<T> + DataVector<T> + RededicateVector<GenericDataVector<T>>, 
        B: ToIdentifier<T> + DataVector<T> + RededicateVector<GenericDataVector<T>> {
    let ops: PreparedOperation2<T, GenericDataVector<T>, GenericDataVector<T>, A, B> =           
         PreparedOperation2 
         { 
            a: PhantomData,
            b: PhantomData, 
            c: PhantomData,
            d: PhantomData, 
            ops: Vec::new(), 
            swap: false
         };
    let a: GenericDataVector<T> = a.rededicate();
    let b: GenericDataVector<T> = b.rededicate();
    MultiOperation2 { a: a, b: b, prepared_ops: ops }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::super::operations_enum::*;
    use super::*;

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
    fn argument_index_assignment()
    {
        let ops = prepare2::<f32, ComplexTimeVector32, RealTimeVector32>();
        let ops = ops.add_ops(|a, b| (b.abs(), a));
        let ops = ops.add_ops(|a, b| (b, a.abs()));
        let mut exp_ops = Vec::new();
        exp_ops.push(Operation::Abs(1));
        exp_ops.push(Operation::Abs(1));
        assert_eq!(ops.ops, exp_ops);
    }
    
    #[test]
    fn complex_operation_on_real_vector()
    {
        let array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a = DataVector32::from_array(false, DataVectorDomain::Time, &array);
        let b = DataVector32::from_array(false, DataVectorDomain::Time, &array);
        
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
}