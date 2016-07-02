use super::super::RealNumber;
use super::ErrorReason;
use num::complex::Complex;
use simd_extensions::{Simd, Reg32, Reg64};

/// An alternative way to define operations on a vector.
#[derive(Copy)]
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub enum Operation<T>
{
    AddReal(usize, T),
    AddComplex(usize, Complex<T>),
    //AddVector(&'a DataVector32<'a>),
    MultiplyReal(usize, T),
    MultiplyComplex(usize, Complex<T>),
    //MultiplyVector(&'a DataVector32<'a>),
    Abs(usize),
    Magnitude(usize),
    Sqrt(usize),
    Log(usize, T),
    ToComplex(usize)
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

pub fn get_argument<T>(operation: Operation<T>) -> usize
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
        vectors: &mut [Self],
        operation: Operation<T>);
        
    fn iter_over_vector<F>(self, op: F) -> Self
       where F: Fn(T) -> T;
}

macro_rules! add_perform_ops_impl {
    ($data_type:ident, $reg:ident)
     =>
     {
        impl PerformOperationSimd<$data_type> for $reg {
            fn perform_operation(
                vectors: &mut [Self],
                operation: Operation<$data_type>)
            {
                match operation
                {
                    Operation::AddReal(idx, value) =>
                    {
                        let v = unsafe { vectors.get_unchecked_mut(idx) };
                        *v = v.add_real(value);
                    }
                    Operation::AddComplex(idx, value) =>
                    {
                        let v = unsafe { vectors.get_unchecked_mut(idx) };
                        *v = v.add_complex(value);
                    }
                    /*Operation32::Addself(value) =>
                    {
                        // TODO
                    }*/
                    Operation::MultiplyReal(idx, value) =>
                    {
                        let v = unsafe { vectors.get_unchecked_mut(idx) };
                        *v = v.scale_real(value);
                    }
                    Operation::MultiplyComplex(idx, value) =>
                    {
                        let v = unsafe { vectors.get_unchecked_mut(idx) };
                        *v = v.scale_complex(value);
                    }
                    /*Operation32::Multiplyself(value) =>
                    {
                        // TODO
                    }*/
                    Operation::Abs(idx) =>
                    {
                        let v = unsafe { vectors.get_unchecked_mut(idx) };
                        *v = v.iter_over_vector(|x|x.abs());
                    }
                    Operation::Magnitude(idx) =>
                    {
                        let v = unsafe { vectors.get_unchecked_mut(idx) };
                        *v = v.complex_abs();
                    }
                    Operation::Sqrt(idx) =>
                    {
                        let v = unsafe { vectors.get_unchecked_mut(idx) };
                        *v = v.sqrt();
                    }
                    Operation::Log(idx, value) =>
                    {
                        let v = unsafe { vectors.get_unchecked_mut(idx) };
                        *v = v.iter_over_vector(|x|x.log(value));
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