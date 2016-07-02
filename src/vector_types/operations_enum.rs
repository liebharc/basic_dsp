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
    // Real Ops
    AddReal(usize, T),
    MultiplyReal(usize, T),
    Abs(usize),
    ToComplex(usize),
    
    // Complex Ops
    AddComplex(usize, Complex<T>),
    MultiplyComplex(usize, Complex<T>),
    MultiplyComplexExponential(usize, T, T),
    Magnitude(usize),
    MagnitudeSquared(usize),
    ComplexConj(usize),
    ToReal(usize),
    ToImag(usize),
    Phase(usize),
    
    // General Ops
    AddVector(usize, usize),
    SubVector(usize, usize),
    MulVector(usize, usize),
    DivVector(usize, usize),
    Sqrt(usize),
    Square(usize),
    Root(usize, T),
    Powf(usize, T),
    Ln(usize),
    Exp(usize),
    Log(usize, T),
    Expf(usize, T),
    Sin(usize),
    Cos(usize),
    Tan(usize),
    ASin(usize),
    ACos(usize),
    ATan(usize),
    Sinh(usize),
    Cosh(usize),
    Tanh(usize),
    ASinh(usize),
    ACosh(usize),
    ATanh(usize)
}

fn require_complex(is_complex: bool) -> Result<bool, ErrorReason> {
    if is_complex { Ok(is_complex) }
    else { Err(ErrorReason::VectorMustBeComplex) }
}

fn require_real(is_complex: bool) -> Result<bool, ErrorReason> {
    if is_complex { Err(ErrorReason::VectorMustBeReal) }
    else { Ok(is_complex) }
}

pub fn evaluate_number_space_transition<T>(is_complex: bool, operation: Operation<T>) -> Result<bool, ErrorReason> 
    where T: RealNumber {
    match operation
    {
        // Real Ops
        Operation::AddReal(_, _) => require_real(is_complex),
        Operation::MultiplyReal(_, _) => require_real(is_complex),
        Operation::Abs(_) => require_real(is_complex),
        Operation::ToComplex(_) => require_real(is_complex),
        // Complex Ops
        Operation::AddComplex(_, _) => require_complex(is_complex),
        Operation::MultiplyComplex(_, _) => require_complex(is_complex),
        Operation::MultiplyComplexExponential(_, _, _) => require_complex(is_complex),
        Operation::Magnitude(_) => require_complex(is_complex),
        Operation::MagnitudeSquared(_) => require_complex(is_complex),
        Operation::ComplexConj(_) => require_complex(is_complex),
        Operation::ToReal(_) => require_complex(is_complex),
        Operation::ToImag(_) => require_complex(is_complex),
        Operation::Phase(_) => require_complex(is_complex),
        // General Ops
        Operation::AddVector(_, _) => Ok(is_complex),
        Operation::SubVector(_, _) => Ok(is_complex),
        Operation::MulVector(_, _) => Ok(is_complex),
        Operation::DivVector(_, _) => Ok(is_complex),
        Operation::Sqrt(_) => Ok(is_complex),
        Operation::Square(_) => Ok(is_complex),
        Operation::Root(_, _) => Ok(is_complex),
        Operation::Powf(_, _) => Ok(is_complex),
        Operation::Ln(_) => Ok(is_complex),
        Operation::Exp(_) => Ok(is_complex),
        Operation::Log(_, _) => Ok(is_complex),
        Operation::Expf(_, _) => Ok(is_complex),
        Operation::Sin(_) => Ok(is_complex),
        Operation::Cos(_) => Ok(is_complex),
        Operation::Tan(_) => Ok(is_complex),
        Operation::ASin(_) => Ok(is_complex),
        Operation::ACos(_) => Ok(is_complex),
        Operation::ATan(_) => Ok(is_complex),
        Operation::Sinh(_) => Ok(is_complex),
        Operation::Cosh(_) => Ok(is_complex),
        Operation::Tanh(_) => Ok(is_complex),
        Operation::ASinh(_) => Ok(is_complex),
        Operation::ACosh(_) => Ok(is_complex),
        Operation::ATanh(_) => Ok(is_complex)
    }
}

pub fn get_argument<T>(operation: Operation<T>) -> usize
    where T: RealNumber {
    match operation
    {
        // Real Ops
        Operation::AddReal(arg, _) => arg,
        Operation::MultiplyReal(arg, _) => arg,
        Operation::Abs(arg) => arg,
        Operation::ToComplex(arg) => arg,
        // Complex Ops
        Operation::AddComplex(arg, _) => arg,
        Operation::MultiplyComplex(arg, _) => arg,
        Operation::MultiplyComplexExponential(arg, _, _) => arg,
        Operation::Magnitude(arg) => arg,
        Operation::MagnitudeSquared(arg) => arg,
        Operation::ComplexConj(arg) => arg,
        Operation::ToReal(arg) => arg,
        Operation::ToImag(arg) => arg,
        Operation::Phase(arg) => arg,
        // General Ops
        Operation::AddVector(arg, _) => arg,
        Operation::SubVector(arg, _) => arg,
        Operation::MulVector(arg, _) => arg,
        Operation::DivVector(arg, _) => arg,
        Operation::Sqrt(arg) => arg,
        Operation::Square(arg) => arg,
        Operation::Root(arg, _) => arg,
        Operation::Powf(arg, _) => arg,
        Operation::Ln(arg) => arg,
        Operation::Exp(arg) => arg,
        Operation::Log(arg, _) => arg,
        Operation::Expf(arg, _) => arg,
        Operation::Sin(arg) => arg,
        Operation::Cos(arg) => arg,
        Operation::Tan(arg) => arg,
        Operation::ASin(arg) => arg,
        Operation::ACos(arg) => arg,
        Operation::ATan(arg) => arg,
        Operation::Sinh(arg) => arg,
        Operation::Cosh(arg) => arg,
        Operation::Tanh(arg) => arg,
        Operation::ASinh(arg) => arg,
        Operation::ACosh(arg) => arg,
        Operation::ATanh(arg) => arg
    }
}

pub trait PerformOperationSimd<T>
    where T: RealNumber,
          Self: Sized {
    #[inline]
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
            #[inline]
            fn perform_operation(
                vectors: &mut [Self],
                operation: Operation<$data_type>)
            {
                
            
                match operation
                {
                    // Real Ops
                    Operation::AddReal(idx, value) =>
                    {
                        let v = unsafe { vectors.get_unchecked_mut(idx) };
                        *v = v.add_real(value);
                    }
                    Operation::MultiplyReal(idx, value) =>
                    {
                        let v = unsafe { vectors.get_unchecked_mut(idx) };
                        *v = v.scale_real(value);
                    }
                    Operation::Abs(idx) =>
                    {
                        let v = unsafe { vectors.get_unchecked_mut(idx) };
                        *v = v.iter_over_vector(|x|x.abs());
                    }
                    Operation::ToComplex(idx) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    // Complex Ops
                    Operation::AddComplex(idx, value) =>
                    {
                        let v = unsafe { vectors.get_unchecked_mut(idx) };
                        *v = v.add_complex(value);
                    }
                    Operation::MultiplyComplex(idx, value) =>
                    {
                        let v = unsafe { vectors.get_unchecked_mut(idx) };
                        *v = v.scale_complex(value);
                    }
                    Operation::MultiplyComplexExponential(idx, _, _) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::Magnitude(idx) =>
                    {
                        let v = unsafe { vectors.get_unchecked_mut(idx) };
                        *v = v.complex_abs();
                    }
                    Operation::MagnitudeSquared(idx) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::ComplexConj(idx) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::ToReal(idx) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::ToImag(idx) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::Phase(idx) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    // General Ops
                    Operation::AddVector(idx, _) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::SubVector(idx, _) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::MulVector(idx, _) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::DivVector(idx, _) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::Sqrt(idx) =>
                    {
                        let v = unsafe { vectors.get_unchecked_mut(idx) };
                        *v = v.sqrt();
                    }
                    Operation::Square(idx) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::Root(idx, _) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::Powf(idx, _) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::Ln(idx) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::Exp(idx) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::Log(idx, value) =>
                    {
                        let v = unsafe { vectors.get_unchecked_mut(idx) };
                        *v = v.iter_over_vector(|x|x.log(value));
                    }
                    Operation::Expf(idx, _) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::Sin(idx) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::Cos(idx) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::Tan(idx) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::ASin(idx) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::ACos(idx) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::ATan(idx) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::Sinh(idx) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::Cosh(idx) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::Tanh(idx) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::ASinh(idx) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::ACosh(idx) => 
                    {
                        panic!("Not implemented yet {}", idx)
                    }
                    Operation::ATanh(idx) => 
                    {
                        panic!("Not implemented yet {}", idx)
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