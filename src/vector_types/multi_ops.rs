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
    GenericDataVector,
    RededicateVector};  
use super::operations_enum::{
    Argument,
    Operation,
    argument_to_index,
    evaluate_number_space_transition,
    get_argument,
    PerformOperationSimd};
use multicore_support::{Chunk, Complexity};
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
    fn new_ops(ops: Vec<(u64, Operation<T>)>, arg: Argument) -> Self;
}

pub trait ComplexIdentifier<T> : Identifier<T>
    where T: RealNumber 
{
    type RealPartner;
    fn magnitude(self) -> Self::RealPartner;
}

pub trait RealIdentifier<T> : Identifier<T>
    where T: RealNumber 
{
    type ComplexPartner;
    fn to_complex(self) -> Self::ComplexPartner;
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
                fn new_ops(ops: Vec<(u64, Operation<T>)>, arg: Argument) -> Self { $name { ops: ops, arg: arg } }
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
            fn magnitude(self) -> Self::RealPartner {
                let arg = self.arg;
                self.add_op(Operation::Magnitude(arg))
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
            fn to_complex(self) -> Self::ComplexPartner {
                let arg = self.arg;
                self.add_op(Operation::ToComplex(arg))
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
                        let t1 = TO1::Identifier::new(Argument::A1);
                        let t2  = TO2::Identifier::new(Argument::A2);
                        let (r1, r2) = operation(t1, t2);
                        let r1arg = r1.get_arg();
                        new_ops.append(&mut r1.get_ops());
                        new_ops.append(&mut r2.get_ops());
                        self.swap != (r1arg == Argument::A2)
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
                    let t1 = TO1::Identifier::new(Argument::A1);
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
        }
        
        impl GenericDataVector<$data_type> {
            fn perform_operations(mut vectors: Vec<Self>, operations: &[Operation<$data_type>])
                -> VecResult<Vec<Self>>
            {
                let errors = Self::verify_ops(&vectors, operations);
                if errors.is_some() {
                    return Err((errors.unwrap(), vectors));
                }
            
                if operations.len() == 0
                {
                    return Ok(vectors);
                }
                
                let vectorization_length =
                    {
                        let first = &vectors[0];
                        let data_length = first.len();
                        let alloc_len = first.allocated_len();
                        let rounded_len = round_len(data_length);
                        if rounded_len <= alloc_len {
                            rounded_len
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
                            data_length - scalar_length
                        }
                    };
                let complexity = if operations.len() > 5 { Complexity::Large } else { Complexity::Medium };
                {
                    let mut i = 0;
                    for v in &mut vectors
                    {
                        let ops: Vec<Operation<$data_type>> = 
                            operations.iter()
                            .filter(|o| {
                                let arg = get_argument(**o);
                                let idx = argument_to_index(arg);
                                idx == i
                            }).
                            map(|o|*o).collect();
                        
                        // Make this conversion to slice explcit, otherwise the type checker and my brain get confused
                        let ops: &[Operation<$data_type>] = &ops;
                        let mut array = &mut v.data;
                        Chunk::execute_partial(
                            complexity, &v.multicore_settings,
                            &mut array, vectorization_length, $reg::len(), 
                            ops, 
                            Self::perform_operations_par);
                        i += 1;
                    }
                }
                Ok (vectors)
            }
            
            fn perform_operations_par(array: &mut [$data_type], operations: &[Operation<$data_type>])
            {
                let mut i = 0;
                while i < array.len()
                { 
                    let mut vector = $reg::load(array, i);
                    for operation in operations
                    {
                        vector = vector.perform_operation(*operation);
                    }
                
                    vector.store(array, i);    
                    i += $reg::len();
                }
            }
            
            fn verify_ops(vectors: &[Self], operations: &[Operation<$data_type>]) -> Option<ErrorReason> {
                let mut complex: Vec<bool> = vectors.iter().map(|v|v.is_complex()).collect();
                for op in operations {
                    let arg = get_argument(*op);
                    let index = argument_to_index(arg);
                    let eval = evaluate_number_space_transition(complex[index], *op);
                    complex[index] = match eval {
                        Err(reason) => { return Some(reason) }
                        Ok(new_complex) => { new_complex }
                    }
                }
                
                None
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
        let a = DataVector32::from_array(false, DataVectorDomain::Time, &array);
        let b = DataVector32::from_array(false, DataVectorDomain::Time, &array);
        
        let ops = multi_ops2(a, b);
        let ops = ops.add_ops(|a, b| (b.magnitude(), a));
        let res = ops.get();
        assert!(res.is_err());
    }
}