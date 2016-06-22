use std::marker::PhantomData;
use super::super::RealNumber;
use super::{
    round_len,
    DataVector,
    DataVector32,
    RealTimeVector,
    ComplexTimeVector,
    RealFreqVector,
    ComplexFreqVector,
    GenericDataVector,
    RededicateVector};  
use num::complex::Complex;
use multicore_support::{Chunk, Complexity, MultiCoreSettings};
use simd_extensions::{Simd, Reg32, Reg64};

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
    fn get_arg(&self) -> Argument;
    fn get_ops(self) -> Vec<(u64, Operation<T>)>;
    fn new(arg: Argument) -> Self;
}

trait ComplexIdentifier<T> : Identifier<T>
    where T: RealNumber 
{
    type RealPartner;
    fn magnitude(self) -> Self::RealPartner;
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
                arg: Argument,
                ops: Vec<(u64, Operation<T>)>
            }
            
            impl<T> ToIdentifier<T> for $vector<T>
                where T: RealNumber {
                type Identifier = $name<T>;
            }
            
            impl<T> Identifier<T> for $name<T>
                where T: RealNumber {
                type Vector = $vector<T>;
                fn get_arg(&self) -> Argument { self.arg }
                fn get_ops(self) -> Vec<(u64, Operation<T>)> { self.ops }
                fn new(arg: Argument) -> Self { $name { ops: Vec::new(), arg: arg } }
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

/// The argument position.
#[derive(Copy)]
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub enum Argument
{
    /// First argument
    A1,
    
    /// Second argument
    A2
}

static mut OPERATION_SEQ_COUNTER : u64 = 0;

/// An operation on one data vector which has been prepared in 
/// advance.
///
/// The type arguments can be read like a function definition: TI1 -> TO1
#[derive(Clone)]
pub struct PreparedOperation1<TI1, TO1> {
    a: PhantomData<TI1>,
    b: PhantomData<TO1>,
    ops: Vec<Operation<f32>>
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

impl<TI1, TI2, TO1, TO2> PreparedOperation2<f32, TI1, TI2, TO1, TO2>
    where TI1: ToIdentifier<f32> + DataVector<f32> + RededicateVector<GenericDataVector<f32>>, 
    TI2: ToIdentifier<f32> + DataVector<f32> + RededicateVector<GenericDataVector<f32>>, 
    TO1: ToIdentifier<f32> + DataVector<f32> + RededicateVector<GenericDataVector<f32>>, 
    TO2: ToIdentifier<f32> + DataVector<f32> + RededicateVector<GenericDataVector<f32>>
{
    pub fn add_ops<F, TN1, TN2>(self, operation: F) 
        -> PreparedOperation2<f32, TI1, TI2, TN1::Vector, TN2::Vector>
        where F: Fn(TO1::Identifier, TO2::Identifier) -> (TN1, TN2),
                 TN1: Identifier<f32>,
                 TN2: Identifier<f32>
        
    {
        let mut ops = self.ops;
        let mut new_ops = Vec::new();
        let swap = 
            {
                let t1 = TO1::Identifier::new(Argument::A1);
                let t2  = TO2::Identifier::new(Argument::A2);
                let (r1, r2) = operation(t1, t2);
                let r1arg = r1.get_arg();
                new_ops.append(&mut r1.get_ops());
                new_ops.append(&mut r2.get_ops());
                r1arg == Argument::A2
            };
         // TODO: Handle overflows in the sequence
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
    
    pub fn exec(&self, a: TI1, b: TI2) -> (TO1, TO2) {
        // First "cast" the vectors to generic vectors. This is done with the
        // the rededicate trait since in contrast to the to_gen method it 
        // can be used in a generic context.
        let len_a = a.len();
        let len_b = b.len();
        let mut a: GenericDataVector<f32> = a.rededicate();
        a.set_len(len_a);
        let mut b: GenericDataVector<f32> = b.rededicate();
        b.set_len(len_b);
        
        // at this point we would execute all ops and cast the result to the right types
        
        // Convert back
        if self.swap {
            let len_a = a.len();
            let len_b = b.len();
            let mut x = TO1::rededicate_from(b);
            x.set_len(len_b);
            let mut y = TO2::rededicate_from(a);
            y.set_len(len_a);
            (x, y)
        }
        else {
            let len_a = a.len();
            let len_b = b.len();
            let mut x = TO1::rededicate_from(a);
            x.set_len(len_a);
            let mut y = TO2::rededicate_from(b);
            y.set_len(len_b);
            (x, y)
        }
    }
}

impl<TI1, TO1> PreparedOperation1<TI1, TO1>
    where TI1: ToIdentifier<f32> + DataVector<f32> + RededicateVector<GenericDataVector<f32>>,
          TO1: ToIdentifier<f32> + DataVector<f32> + RededicateVector<GenericDataVector<f32>> {
          
    pub fn exec(&self, a: TI1) -> TO1 {
        let len_a = a.len();
        let mut a: GenericDataVector<f32> = a.rededicate();
        a.set_len(len_a);
        
        // at this point we would execute all ops and cast the result to the right types
        
        let len_a = a.len();
        let mut x = TO1::rededicate_from(a);
        x.set_len(len_a);
        x
    }
    
    pub fn extend<TI2>(self) 
        -> PreparedOperation2<f32, TI1, TI2, TO1, TI2>
        where TI2: ToIdentifier<f32> {
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
}

impl ComplexIdentifier<f32> for ComplexTimeIdentifier<f32> {
    type RealPartner = RealTimeIdentifier<f32>;
    fn magnitude(self) -> RealTimeIdentifier<f32> {
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
        ops.push((seq, Operation::AbsComplex));
        RealTimeIdentifier { ops: ops, arg: self.arg }
    }
}

pub fn prepare2<A, B>()
    -> PreparedOperation2<f32, A, B, A, B> 
    where A: ToIdentifier<f32>, B: ToIdentifier<f32> {
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

struct MultiOperation2<TO1, TO2> 
    where TO1: ToIdentifier<f32>, TO2: ToIdentifier<f32> {
    a: GenericDataVector<f32>,
    b: GenericDataVector<f32>,
    preped_ops: PreparedOperation2<f32, GenericDataVector<f32>, GenericDataVector<f32>, TO1, TO2>
}

impl<TO1, TO2>  MultiOperation2<TO1, TO2> 
    where TO1: ToIdentifier<f32> + DataVector<f32> + RededicateVector<GenericDataVector<f32>>, 
          TO2: ToIdentifier<f32> + DataVector<f32> + RededicateVector<GenericDataVector<f32>> {
    fn get(self) -> (TO1, TO2) {
        self.preped_ops.exec(self.a, self.b)
    }
    
    fn add_ops<F, TN1, TN2>(self, operation: F) 
        -> MultiOperation2<TN1::Vector, TN2::Vector>
        where F: Fn(TO1::Identifier, TO2::Identifier) -> (TN1, TN2),
                 TN1: Identifier<f32>,
                 TN2: Identifier<f32>
        
    {
        let ops = self.preped_ops.add_ops(operation);
        MultiOperation2 { a: self.a, b: self.b, preped_ops: ops }
    }
}

fn multi_ops2<A, B>(a: A, b: B)
    -> MultiOperation2<A, B>
    where A: ToIdentifier<f32> + DataVector<f32> + RededicateVector<GenericDataVector<f32>>, 
    B: ToIdentifier<f32> + DataVector<f32> + RededicateVector<GenericDataVector<f32>> {
    let ops: PreparedOperation2<f32, GenericDataVector<f32>, GenericDataVector<f32>, A, B> =           
         PreparedOperation2 
         { 
            a: PhantomData,
            b: PhantomData, 
            c: PhantomData,
            d: PhantomData, 
            ops: Vec::new(), 
            swap: false
         };
    let len_a = a.len();
    let len_b = b.len();
    let mut a: GenericDataVector<f32> = a.rededicate();
    a.set_len(len_a);
    let mut b: GenericDataVector<f32> = b.rededicate();
    b.set_len(len_b);
    MultiOperation2 { a: a, b: b, preped_ops: ops }
}

/// An alternative way to define operations on a vector.
#[derive(Copy)]
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub enum Operation<T>
{
    AddReal(T),
    AddComplex(Complex<T>),
    //AddVector(&'a DataVector32<'a>),
    MultiplyReal(T),
    MultiplyComplex(Complex<T>),
    //MultiplyVector(&'a DataVector32<'a>),
    AbsReal,
    AbsComplex,
    Sqrt,
    Log(T)
}

impl GenericDataVector<f32> {
    /// Perform a set of operations on the given vector. 
    /// Warning: Highly unstable and not even fully implemented right now.
    ///
    /// With this approach we change how we operate on vectors. If you perform
    /// `M` operations on a vector with the length `N` you iterate wit hall other methods like this:
    ///
    /// ```
    /// // pseudocode:
    /// // for m in M:
    /// //  for n in N:
    /// //    execute m on n
    /// ```
    ///
    /// with this method the pattern is changed slighly:
    ///
    /// ```
    /// // pseudocode:
    /// // for n in N:
    /// //  for m in M:
    /// //    execute m on n
    /// ```
    ///
    /// Both variants have the same complexity however the second one is benificial since we
    /// have increased locality this way. This should help us by making better use of registers and 
    /// CPU buffers. This might also help since for large data we might have the chance in future to 
    /// move the data to a GPU, run all operations and get the result back. In this case the GPU is fast
    /// for many operations but the roundtrips on the bus should be minimized to keep the speed advantage.
    pub fn perform_operations(mut self, operations: &[Operation<f32>])
        -> Self
    {
        if operations.len() == 0
        {
            return DataVector32 { data: self.data, .. self };
        }
        
        let data_length = self.len();
        let alloc_len = self.allocated_len();
        let rounded_len = round_len(data_length);
        let vectorization_length = 
            if rounded_len <= alloc_len {
                rounded_len
            }
            else {
                let scalar_length = data_length % Reg32::len();
                if scalar_length > 0
                {
                    panic!("perform_operations requires right now that the array length is dividable by 4")
                }
                data_length - scalar_length
            };
            
        let complexity = if operations.len() > 5 { Complexity::Large } else { Complexity::Medium };
        {
            let mut array = &mut self.data;
            Chunk::execute_partial(
                complexity, &self.multicore_settings,
                &mut array, vectorization_length, Reg32::len(), 
                operations, 
                DataVector32::perform_operations_par);
        }
        DataVector32 { data: self.data, .. self }
    }
    
    fn perform_operations_par(array: &mut [f32], operations: &[Operation<f32>])
    {
        let mut i = 0;
        while i < array.len()
        { 
            let mut vector = Reg32::load(array, i);
            let mut j = 0;
            while j < operations.len()
            {
                let operation = &operations[j];
                match *operation
                {
                    Operation::AddReal(value) =>
                    {
                        vector = vector.add_real(value);
                    }
                    Operation::AddComplex(value) =>
                    {
                        vector = vector.add_complex(value);
                    }
                    /*Operation32::AddVector(value) =>
                    {
                        // TODO
                    }*/
                    Operation::MultiplyReal(value) =>
                    {
                        vector = vector.scale_real(value);
                    }
                    Operation::MultiplyComplex(value) =>
                    {
                        vector = vector.scale_complex(value);
                    }
                    /*Operation32::MultiplyVector(value) =>
                    {
                        // TODO
                    }*/
                    Operation::AbsReal =>
                    {
                        vector.store(array, i);
                        {
                            let mut content = &mut array[i .. i + Reg32::len()];
                            let mut k = 0;
                            while k < Reg32::len()
                            {
                                content[k] = content[k].abs();
                                k = k + 1;
                            }
                        }
                        vector = Reg32::load(array, i);
                    }
                    Operation::AbsComplex =>
                    {
                        vector = vector.complex_abs();
                    }
                    Operation::Sqrt =>
                    {
                        vector.store(array, i);
                        {
                            let mut content = &mut array[i .. i + Reg32::len()];
                            let mut k = 0;
                            while k < Reg32::len()
                            {
                                content[k] = content[k].sqrt();
                                k = k + 1;
                            }
                        }
                        vector = Reg32::load(array, i);
                    }
                    Operation::Log(value) =>
                    {
                        vector.store(array, i);
                        {
                            let mut content = &mut array[i .. i + Reg32::len()];
                            let mut k = 0;
                            while k < Reg32::len()
                            {
                                content[k] = content[k].log(value);
                                k = k + 1;
                            }
                        }
                        vector = Reg32::load(array, i);
                    }
                }
                j += 1;
            }
        
            vector.store(array, i);    
            i += Reg32::len();
        }
    }
}