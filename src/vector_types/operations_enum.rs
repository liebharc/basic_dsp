use super::super::RealNumber;
use super::ErrorReason;
use num::complex::Complex;
use simd_extensions::{Simd, Reg32, Reg64};

/// The argument position. User internally to keep track of arguments.
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

/// An alternative way to define operations on a vector.
#[derive(Copy)]
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub enum Operation<T>
{
    AddReal(Argument, T),
    AddComplex(Argument, Complex<T>),
    //AddVector(&'a DataVector32<'a>),
    MultiplyReal(Argument, T),
    MultiplyComplex(Argument, Complex<T>),
    //MultiplyVector(&'a DataVector32<'a>),
    Abs(Argument),
    Magnitude(Argument),
    Sqrt(Argument),
    Log(Argument, T),
    ToComplex(Argument)
}

pub fn argument_to_index(arg: Argument) -> usize {
    match arg {
        Argument::A1 => { 0 }
        Argument::A2 => { 1 }
    }
}

pub fn evaluate_number_space_transition<T>(is_complex: bool, operation: Operation<T>) -> Result<bool, ErrorReason> 
    where T: RealNumber {
    match operation
    {
        Operation::AddReal(_, _) =>
        {
            if is_complex { Err(ErrorReason::VectorMustBeReal) }
            else { Ok(is_complex) }
        }
        Operation::AddComplex(_, _) =>
        {
            if is_complex { Ok(is_complex) }
            else { Err(ErrorReason::VectorMustBeComplex) }
        }
        /*Operation32::AddVector(value) =>
        {
            // TODO
        }*/
        Operation::MultiplyReal(_, _) =>
        {
            if is_complex { Err(ErrorReason::VectorMustBeReal) }
            else { Ok(is_complex) }
        }
        Operation::MultiplyComplex(_, _) =>
        {
            if is_complex { Ok(is_complex) }
            else { Err(ErrorReason::VectorMustBeComplex) }
        }
        /*Operation32::MultiplyVector(value) =>
        {
            // TODO
        }*/
        Operation::Abs(_) =>
        {
            if is_complex { Err(ErrorReason::VectorMustBeReal) }
            else { Ok(is_complex) }
        }
        Operation::Magnitude(_) =>
        {
            if is_complex { Ok(false) }
            else { Err(ErrorReason::VectorMustBeComplex) }
        }
        Operation::Sqrt(_) =>
        {
            Ok(is_complex)
        }
        Operation::Log(_, _) =>
        {
            Ok(is_complex)
        }
        Operation::ToComplex(_) =>
        {
            panic!("Type conversion should have already been resolved")
        }
    }
}

pub fn get_argument<T>(operation: Operation<T>) -> Argument
    where T: RealNumber {
    match operation
    {
        Operation::AddReal(arg, _) => { arg }
        Operation::AddComplex(arg, _) => { arg }
        /*Operation32::AddVector(value) =>
        {
            // TODO
        }*/
        Operation::MultiplyReal(arg, _) => { arg }
        Operation::MultiplyComplex(arg, _) => { arg }
        /*Operation32::MultiplyVector(value) =>
        {
            // TODO
        }*/
        Operation::Abs(arg) => { arg }
        Operation::Magnitude(arg) => { arg }
        Operation::Sqrt(arg) => { arg }
        Operation::Log(arg, _) => { arg }
        Operation::ToComplex(_) =>
        {
            panic!("Type conversion should have already been resolved")
        }
    }
}

pub trait PerformOperationSimd<T>
    where T: RealNumber,
          Self: Sized {
    fn perform_operation(
        self,
        operation: Operation<T>) -> Self;
        
    fn iter_over_vector<F>(self, op: F) -> Self
       where F: Fn(T) -> T;
}

macro_rules! add_perform_ops_impl {
    ($data_type:ident, $reg:ident)
     =>
     {
        impl PerformOperationSimd<$data_type> for $reg {
            fn perform_operation(
                self,
                operation: Operation<$data_type>) -> Self
            {
                match operation
                {
                    Operation::AddReal(_, value) =>
                    {
                        self.add_real(value)
                    }
                    Operation::AddComplex(_, value) =>
                    {
                        self.add_complex(value)
                    }
                    /*Operation32::Addself(value) =>
                    {
                        // TODO
                    }*/
                    Operation::MultiplyReal(_, value) =>
                    {
                        self.scale_real(value)
                    }
                    Operation::MultiplyComplex(_, value) =>
                    {
                        self.scale_complex(value)
                    }
                    /*Operation32::Multiplyself(value) =>
                    {
                        // TODO
                    }*/
                    Operation::Abs(_) =>
                    {
                        self.iter_over_vector(|x|x.abs())
                    }
                    Operation::Magnitude(_) =>
                    {
                        self.complex_abs()
                    }
                    Operation::Sqrt(_) =>
                    {
                         self.sqrt()
                    }
                    Operation::Log(_, value) =>
                    {
                        self.iter_over_vector(|x|x.log(value))
                    }
                    Operation::ToComplex(_) =>
                    {
                        panic!("Type conversion should have already been resolved")
                    }
                }
            }
            
            fn iter_over_vector<F>(self, op: F) -> Self
                where F: Fn($data_type) -> $data_type {
                let mut array = self.to_array();
                for n in &mut array {
                    *n = op(*n);
                }
                $reg::from_array(array)
            }
        }
    }
}

add_perform_ops_impl!(f32, Reg32);
add_perform_ops_impl!(f64, Reg64); 