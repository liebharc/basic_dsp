use rand::*;
use basic_dsp::{
    DataVector,
    RealTimeVector,
    RealFreqVector,
    ComplexTimeVector,
    ComplexFreqVector,
    RealNumber,
    DataVectorDomain};
use std::ops::Range;
use num::complex::Complex32;
use std::panic;
use std::fmt::{Display, Debug};

pub fn assert_vector_eq_with_reason<T>(left: &[T], right: &[T], reason: &str) 
    where T: RealNumber + Debug + Display {
    assert_vector_eq_with_reason_and_tolerance(left, right, T::from(1e-6).unwrap(), reason);
}

pub fn assert_vector_eq_with_reason_and_tolerance<T>(left: &[T], right: &[T], tolerance: T, reason: &str)
    where T: RealNumber + Debug + Display  {
    let mut errors = Vec::new();
    if reason.len() > 0
    {
        errors.push(format!("{}:\n", reason));
    }
    
    let size_assert_failed = left.len() != right.len();
    if size_assert_failed
    {
        errors.push(format!("Size difference {} != {}", left.len(), right.len()));
    }
    
    let len = if left.len() < right.len() { left.len() } else { right.len() };
    let mut differences = 0;
    for i in 0 .. len {
        if (left[i] - right[i]).abs() > tolerance
        {
            differences += 1;
            if differences <= 10
            {
                errors.push(format!("Difference {} at index {}, left: {} != right: {}", differences, i, left[i], right[i]));
            }
        }
    }
    
    if differences > 0
    {
        errors.push(format!("Total number of differences: {}/{}={}%", differences, len, differences*100/len));
    }
    
    if differences > 0 || size_assert_failed
    {
        let all_errors = errors.join("\n");
        let header = "-----------------------".to_owned();
        let full_text = format!("\n{}\n{}\n{}\n", header, all_errors, header);
        panic!(full_text);
    }
}
    
pub fn assert_vector_eq<T>(left: &[T], right: &[T]) 
    where T: RealNumber + Debug + Display {
    assert_vector_eq_with_reason(left, right, "");
}

pub fn assert_close(left: f32, right: f32) {
    assert_in_tolerance(left, right, 1e-6);
}

pub fn assert_complex_close(left: Complex32, right: Complex32) {
    assert_complex_in_tolerance(left, right, 1e-6);
}

pub fn assert_in_tolerance(left: f32, right: f32, tol: f32) {
    let tol = tol.abs();
    if (left - right).abs() > tol {
        panic!(format!("{} != {} (tol: {})", left, right, tol));
    }
}

pub fn assert_complex_in_tolerance(left: Complex32, right: Complex32, tol: f32) {
    let tol = tol.abs();
    if (left.re - right.re).abs() > tol ||
       (left.im - right.im).abs() > tol {
        panic!(format!("{} != {} (tol: {})", left, right, tol));
    }
}

pub fn create_data(seed: usize, iteration: usize, from: usize, to: usize) -> Vec<f32>
{
    let len_seed: &[_] = &[seed, iteration];
    let mut rng: StdRng = SeedableRng::from_seed(len_seed);
    let len = if from == to { from }else { rng.gen_range(from, to) };
    create_data_with_len(seed, iteration, len)
}

pub fn create_data_even(seed: usize, iteration: usize, from: usize, to: usize) -> Vec<f32>
{
    let len_seed: &[_] = &[seed, iteration];
    let mut rng: StdRng = SeedableRng::from_seed(len_seed);
    let len = rng.gen_range(from, to);
    let len = len + len % 2;
    create_data_with_len(seed, iteration, len)
}

pub fn create_data_with_len(seed: usize, iteration: usize, len: usize) -> Vec<f32>
{
    let seed: &[_] = &[seed, iteration];
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    let mut data = vec![0.0; len];
    for i in 0..len {
        data[i] = rng.gen_range(-10.0, 10.0);
    }
    data
}

pub fn create_data_even_in_range(seed: usize, iteration: usize, from: usize, to: usize, range_start: f32, range_end: f32) -> Vec<f32>
{
    let len_seed: &[_] = &[seed, iteration];
    let mut rng: StdRng = SeedableRng::from_seed(len_seed);
    let len = rng.gen_range(from, to);
    let len = len + len % 2;
    create_data_in_range_with_len(seed, iteration, len, range_start, range_end)
}

pub fn create_data_simd_len_in_range(seed: usize, iteration: usize, from: usize, to: usize) -> Vec<f32>
{
    let len_seed: &[_] = &[seed, iteration];
    let mut rng: StdRng = SeedableRng::from_seed(len_seed);
    let len = rng.gen_range(from, to);
    let len = (len / 8 + 1) + 8; // Not exact but good enough
    let len = if len > to { to } else { len };
    create_data_with_len(seed, iteration, len)
}

pub fn create_data_in_range_with_len(seed: usize, iteration: usize, len: usize, range_start: f32, range_end: f32) -> Vec<f32>
{
    let seed: &[_] = &[seed, iteration];
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    let mut data = vec![0.0; len];
    for i in 0..len {
        data[i] = rng.gen_range(range_start, range_end);
    }
    data
}

pub fn create_delta(seed: usize, iteration: usize)
    -> f32
{
    let seed: &[_] = &[seed, iteration];
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    rng.gen_range(-10.0, 10.0)
}

pub trait AssertMetaData {
    fn assert_meta_data(&self);
}

impl<T> AssertMetaData for RealTimeVector<T> 
    where T: RealNumber {
    fn assert_meta_data(&self) {
        assert_eq!(self.is_complex(), false);
        assert_eq!(self.domain(), DataVectorDomain::Time);
    }
}

impl<T> AssertMetaData for RealFreqVector<T> 
    where T: RealNumber {
    fn assert_meta_data(&self) {
        assert_eq!(self.is_complex(), false);
        assert_eq!(self.domain(), DataVectorDomain::Frequency);
    }
}

impl<T> AssertMetaData for ComplexTimeVector<T> 
    where T: RealNumber {
    fn assert_meta_data(&self) {
        assert_eq!(self.is_complex(), true);
        assert_eq!(self.domain(), DataVectorDomain::Time);
    }
}

impl<T> AssertMetaData for ComplexFreqVector<T> 
    where T: RealNumber {
    fn assert_meta_data(&self) {
        assert_eq!(self.is_complex(), true);
        assert_eq!(self.domain(), DataVectorDomain::Frequency);
    }
}

use std::sync::{Arc, Mutex};

pub const RANGE_SINGLE_CORE: Range<usize> = Range { start: 10000, end: 100000 };
pub const RANGE_MULTI_CORE: Range<usize> = Range { start: 100001, end: 200000 };

pub fn parameterized_vector_test<F>(test_code: F)
    where F: Fn(usize, Range<usize>) + Send + 'static + Sync
{
    let mut test_errors = Vec::new();
    
    // I don't know if there is a good reason for this, but Rust
    // requires us to lock the test_code function, 
    // does catch_panic spawn really a new thread?
    let test_code = Arc::new(Mutex::new(test_code));
    for iteration in 0 .. 10 {
        if test_errors.len() > 0 {
            break; // Stop on the first error to speed up things
        }
        
        let small_range = RANGE_SINGLE_CORE;
        let test_code = test_code.clone();
        let safe_iteration = panic::AssertUnwindSafe(iteration);
        let small_range = panic::AssertUnwindSafe(small_range);
        let test_code = panic::AssertUnwindSafe(test_code);
        let result = panic::catch_unwind(move|| {
            let test_code = test_code.lock().unwrap();
            test_code(*safe_iteration, (*small_range).clone());
        });
        
        match result {
            Ok(_) => (),
            Err(e) => {
                if let Some(e) = e.downcast_ref::<&'static str>() {
                    test_errors.push(format!("\nSingle threaded execution path failed on iteration {}\nFailure: {}", iteration, e));
                }
                else if let Some(e) = e.downcast_ref::<String>() {
                    test_errors.push(format!("\nSingle threaded execution path failed on iteration {}\nFailure: {}", iteration, e));
                }
                else {
                    test_errors.push(format!("\nSingle threaded execution path failed on iteration {}\nGot an unknown error: {:?}", iteration, e));
                }
            }
        }
    }
    
    for iteration in 0 .. 3 {
        if test_errors.len() > 0 {
            break; // Stop on the first error to speed up things
        }
        
        let large_range = RANGE_MULTI_CORE;
        let test_code = test_code.clone();
        let safe_iteration = panic::AssertUnwindSafe(iteration);
        let large_range = panic::AssertUnwindSafe(large_range);
        let test_code = panic::AssertUnwindSafe(test_code);
        let result = panic::catch_unwind(move|| {
            let test_code = test_code.lock().unwrap();
            test_code(*safe_iteration, (*large_range).clone());
        });
        
        match result {
            Ok(_) => (),
            Err(e) => {
                if let Some(e) = e.downcast_ref::<&'static str>() {
                    test_errors.push(format!("\nMulti threaded execution path failed on iteration {}\nFailure: {}", iteration, e));
                } 
                else if let Some(e) = e.downcast_ref::<String>() {
                    test_errors.push(format!("\nMulti threaded execution path failed on iteration {}\nFailure: {}", iteration, e));
                }
                else {
                    test_errors.push(format!("\nMulti threaded execution path failed on iteration {}\nGot an unknown error: {:?}", iteration, e));
                }
            }
        }
    }
    
    if test_errors.len() > 0 {
        let error_messages = test_errors.join("\n");
        panic!(error_messages);
    }
}