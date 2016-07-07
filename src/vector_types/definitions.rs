use std::result;
use num::complex::Complex;
use super::super::RealNumber;

/// DataVector gives access to the basic properties of all data vectors
///
/// A DataVector allocates memory if necessary. It will however never shrink/free memory unless it's 
/// deleted and dropped.
pub trait DataVector<T> : Sized
    where T : RealNumber
{
    /// Gives direct access to the underlying data sequence. It's recommended to use the `Index functions .
    /// For users outside of Rust: It's discouraged to hold references to this array while executing operations on the vector,
    /// since the vector may decide at any operation to invalidate the array. 
    fn data(&self) -> &[T];
    
    /// The x-axis delta. If `domain` is time domain then `delta` is in `[s]`, in frequency domain `delta` is in `[Hz]`.
    fn delta(&self) -> T;
    
    /// The domain in which the data vector resides. Basically specifies the x-axis and the type of operations which
    /// are valid on this vector.
    fn domain(&self) -> DataVectorDomain;
    
    /// Indicates whether the vector contains complex data. This also specifies the type of operations which are valid
    /// on this vector.
    fn is_complex(&self) -> bool;
    
    /// The number of valid elements in the the vector.
    fn len(&self) -> usize;
    
    /// Sets the vector length to the given length. 
    /// If `self.len() < len` then the value of the new elements is undefined.
    fn set_len(&mut self, len: usize);
    
    /// The number of valid points. If the vector is complex then every valid point consists of two floating point numbers,
    /// while for real vectors every point only consists of one floating point number.
    fn points(&self) -> usize;
    
    /// Gets the number of allocated elements in the underlying vector.
    /// The allocated length may be larger than the length of valid points. 
    /// In most cases you likely want to have `len`or `points` instead.
    fn allocated_len(&self) -> usize;
}

/// The domain of a data vector
#[derive(Copy)]
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub enum DataVectorDomain {
    /// Time domain, the x-axis is in [s]
    Time,
    /// Frequency domain, the x-axis in in [Hz]
    Frequency
}

/// This trait allows to change a vector type. The operations will
/// convert a vector to a different type and set `self.len()` to zero.
/// However `self.allocated_len()` will remain unchanged. The use case for this
/// is to allow to reuse the memory of a vector for different operations.
pub trait RededicateVector<Other> {
    /// Make `self` a `Other`.
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{ComplexFreqVector32, ComplexTimeVector32, ComplexVectorOps, RededicateVector, DataVector, DataVectorDomain};
    /// let complex = ComplexFreqVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
    /// let real = complex.phase().expect("Ignoring error handling in examples");
    /// let complex: ComplexTimeVector32 = real.rededicate();
    /// assert_eq!(true, complex.is_complex());
    /// assert_eq!(DataVectorDomain::Time, complex.domain());
    /// assert_eq!(0, complex.len());
    /// assert_eq!(2, complex.allocated_len());
    /// ```
    fn rededicate(self) -> Other;
    
    /// Make `Other` a `Self`.
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{ComplexFreqVector32, ComplexTimeVector32, ComplexVectorOps, RededicateVector, DataVector, DataVectorDomain};
    /// let complex = ComplexFreqVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
    /// let real = complex.phase().expect("Ignoring error handling in examples");
    /// let complex = ComplexTimeVector32::rededicate_from(real);
    /// assert_eq!(true, complex.is_complex());
    /// assert_eq!(DataVectorDomain::Time, complex.domain());
    /// assert_eq!(0, complex.len());
    /// assert_eq!(2, complex.allocated_len());
    /// ```
    fn rededicate_from(origin: Other) -> Self;
}

/// An operation which multiplies each vector element with a constant
pub trait ScaleOps<T> : Sized
    where T: Sized {
    /// Multiplies the vector element with a scalar.
    fn scale(self, factor: T) -> VecResult<Self>;
}

/// An operation which adds a constant to each vector element
pub trait OffsetOps<T> : Sized
    where T: Sized {
    /// Adds a scalar to each vector element.
    fn offset(self, offset: T) -> VecResult<Self>;
}

/// An operation which multiplies each vector element with a constant
pub trait DotProductOps<T> : Sized
    where T: Sized {
    /// Calculates the dot product of self and factor. Self and factor remain unchanged.
    fn dot_product(&self, factor: &Self) -> ScalarResult<T>;
}

/// Calculates the statistics of the data contained in the vector.
pub trait StatisticsOps<T> : Sized
    where T: Sized {
    /// Calculates the statistics of the data contained in the vector.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp;
    /// # use num::complex::Complex32;
    /// use basic_dsp::{ComplexTimeVector32, StatisticsOps};
    /// # fn main() { 
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let result = vector.statistics();
    /// assert_eq!(result.sum, Complex32::new(9.0, 12.0));
    /// assert_eq!(result.count, 3);
    /// assert_eq!(result.average, Complex32::new(3.0, 4.0));
    /// assert!((result.rms - Complex32::new(3.4027193, 4.3102784)).norm() < 1e-4);
    /// assert_eq!(result.min, Complex32::new(1.0, 2.0));
    /// assert_eq!(result.min_index, 0);
    /// assert_eq!(result.max, Complex32::new(5.0, 6.0));
    /// assert_eq!(result.max_index, 2);
    /// }
    /// ```  
    fn statistics(&self) -> Statistics<T>;
    
    /// Calculates the statistics of the data contained in the vector as if the vector would
    /// have been split into `len` pieces. `self.len` should be devisable by `len` without a remainder,
    /// but this isn't enforced by the implementation.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp;
    /// # use num::complex::Complex32;
    /// use basic_dsp::{ComplexTimeVector32, StatisticsOps};
    /// # fn main() { 
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    /// let result = vector.statistics_splitted(2);
    /// assert_eq!(result[0].sum, Complex32::new(6.0, 8.0));
    /// assert_eq!(result[1].sum, Complex32::new(10.0, 12.0));
    /// }
    /// ```  
    fn statistics_splitted(&self, len: usize) -> Vec<Statistics<T>>;     
}

/// Defines all operations which are valid on all `DataVectors`.
pub trait GenericVectorOps<T>: DataVector<T> 
    where T : RealNumber {
    /// Calculates the sum of `self + summand`. It consumes self and returns the result.
    /// # Failures
    /// VecResult may report the following `ErrorReason` members:
    /// 
    /// 1. `VectorsMustHaveTheSameSize`: `self` and `summand` must have the same size
    /// 2. `VectorMetaDataMustAgree`: `self` and `summand` must be in the same domain and number space
    /// 
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// let vector1 = RealTimeVector32::from_array(&[1.0, 2.0]);
    /// let vector2 = RealTimeVector32::from_array(&[10.0, 11.0]);
    /// let result = vector1.add_vector(&vector2).expect("Ignoring error handling in examples");
    /// assert_eq!([11.0, 13.0], result.data());
    /// ```
    fn add_vector(self, summand: &Self) -> VecResult<Self>;
    
    /// Calculates the sum of `self + summand`. `summand` may be smaller than `self` as long
    /// as `self.len() % summand.len() == 0`. THe result is the same as it would be if 
    /// you would repeat `summand` until it has the same length as `self`.
    /// It consumes self and returns the result.
    /// # Failures
    /// VecResult may report the following `ErrorReason` members:
    /// 
    /// 1. `InvalidArgumentLength`: `self.points()` isn't dividable by `summand.points()`
    /// 2. `VectorMetaDataMustAgree`: `self` and `summand` must be in the same domain and number space
    /// 
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// let vector1 = RealTimeVector32::from_array(&[10.0, 11.0, 12.0, 13.0]);
    /// let vector2 = RealTimeVector32::from_array(&[1.0, 2.0]);
    /// let result = vector1.add_smaller_vector(&vector2).expect("Ignoring error handling in examples");
    /// assert_eq!([11.0, 13.0, 13.0, 15.0], result.data());
    /// ```
    fn add_smaller_vector(self, summand: &Self) -> VecResult<Self>;
    
    /// Calculates the difference of `self - subtrahend`. It consumes self and returns the result.
    /// # Failures
    /// VecResult may report the following `ErrorReason` members:
    /// 
    /// 1. `VectorsMustHaveTheSameSize`: `self` and `subtrahend` must have the same size
    /// 2. `VectorMetaDataMustAgree`: `self` and `subtrahend` must be in the same domain and number space
    /// 
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// let vector1 = RealTimeVector32::from_array(&[1.0, 2.0]);
    /// let vector2 = RealTimeVector32::from_array(&[10.0, 11.0]);
    /// let result = vector1.subtract_vector(&vector2).expect("Ignoring error handling in examples");
    /// assert_eq!([-9.0, -9.0], result.data());
    /// ```
    fn subtract_vector(self, subtrahend: &Self) -> VecResult<Self>;
    
    /// Calculates the sum of `self - subtrahend`. `subtrahend` may be smaller than `self` as long
    /// as `self.len() % subtrahend.len() == 0`. THe result is the same as it would be if 
    /// you would repeat `subtrahend` until it has the same length as `self`.
    /// It consumes self and returns the result.
    /// # Failures
    /// VecResult may report the following `ErrorReason` members:
    /// 
    /// 1. `InvalidArgumentLength`: `self.points()` isn't dividable by `subtrahend.points()`
    /// 2. `VectorMetaDataMustAgree`: `self` and `subtrahend` must be in the same domain and number space
    /// 
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// let vector1 = RealTimeVector32::from_array(&[10.0, 11.0, 12.0, 13.0]);
    /// let vector2 = RealTimeVector32::from_array(&[1.0, 2.0]);
    /// let result = vector1.subtract_smaller_vector(&vector2).expect("Ignoring error handling in examples");
    /// assert_eq!([9.0, 9.0, 11.0, 11.0], result.data());
    /// ```
    fn subtract_smaller_vector(self, summand: &Self) -> VecResult<Self>;
    
    /// Calculates the product of `self * factor`. It consumes self and returns the result.
    /// # Failures
    /// VecResult may report the following `ErrorReason` members:
    /// 
    /// 1. `VectorsMustHaveTheSameSize`: `self` and `factor` must have the same size
    /// 2. `VectorMetaDataMustAgree`: `self` and `factor` must be in the same domain and number space
    /// 
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// let vector1 = RealTimeVector32::from_array(&[1.0, 2.0]);
    /// let vector2 = RealTimeVector32::from_array(&[10.0, 11.0]);
    /// let result = vector1.multiply_vector(&vector2).expect("Ignoring error handling in examples");
    /// assert_eq!([10.0, 22.0], result.data());
    /// ```
    fn multiply_vector(self, factor: &Self) -> VecResult<Self>;
    
    /// Calculates the sum of `self - factor`. `factor` may be smaller than `self` as long
    /// as `self.len() % factor.len() == 0`. THe result is the same as it would be if 
    /// you would repeat `factor` until it has the same length as `self`.
    /// It consumes self and returns the result.
    /// # Failures
    /// VecResult may report the following `ErrorReason` members:
    /// 
    /// 1. `InvalidArgumentLength`: `self.points()` isn't dividable by `factor.points()`
    /// 2. `VectorMetaDataMustAgree`: `self` and `factor` must be in the same domain and number space
    /// 
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// let vector1 = RealTimeVector32::from_array(&[10.0, 11.0, 12.0, 13.0]);
    /// let vector2 = RealTimeVector32::from_array(&[1.0, 2.0]);
    /// let result = vector1.multiply_smaller_vector(&vector2).expect("Ignoring error handling in examples");
    /// assert_eq!([10.0, 22.0, 12.0, 26.0], result.data());
    /// ```
    fn multiply_smaller_vector(self, factor: &Self) -> VecResult<Self>;
    
    /// Calculates the quotient of `self / summand`. It consumes self and returns the result.
    /// # Failures
    /// VecResult may report the following `ErrorReason` members:
    /// 
    /// 1. `VectorsMustHaveTheSameSize`: `self` and `divisor` must have the same size
    /// 2. `VectorMetaDataMustAgree`: `self` and `divisor` must be in the same domain and number space
    /// 
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// let vector1 = RealTimeVector32::from_array(&[10.0, 22.0]);
    /// let vector2 = RealTimeVector32::from_array(&[2.0, 11.0]);
    /// let result = vector1.divide_vector(&vector2).expect("Ignoring error handling in examples");
    /// assert_eq!([5.0, 2.0], result.data());
    /// ```
    fn divide_vector(self, divisor: &Self) -> VecResult<Self>;
    
    /// Calculates the sum of `self - divisor`. `divisor` may be smaller than `self` as long
    /// as `self.len() % divisor.len() == 0`. THe result is the same as it would be if 
    /// you would repeat `divisor` until it has the same length as `self`.
    /// It consumes self and returns the result.
    /// # Failures
    /// VecResult may report the following `ErrorReason` members:
    /// 
    /// 1. `InvalidArgumentLength`: `self.points()` isn't dividable by `divisor.points()`
    /// 2. `VectorMetaDataMustAgree`: `self` and `divisor` must be in the same domain and number space
    /// 
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// let vector1 = RealTimeVector32::from_array(&[10.0, 12.0, 12.0, 14.0]);
    /// let vector2 = RealTimeVector32::from_array(&[1.0, 2.0]);
    /// let result = vector1.divide_smaller_vector(&vector2).expect("Ignoring error handling in examples");
    /// assert_eq!([10.0, 6.0, 12.0, 7.0], result.data());
    /// ```
    fn divide_smaller_vector(self, divisor: &Self) -> VecResult<Self>;
    
    /// Appends zeros add the end of the vector until the vector has the size given in the points argument.
    /// If `points` smaller than the `self.len()` then this operation won't do anything.
    ///
    /// Note: Each point is two floating point numbers if the vector is complex.
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{PaddingOption, RealTimeVector32, ComplexTimeVector32, GenericVectorOps, DataVector};
    /// let vector = RealTimeVector32::from_array(&[1.0, 2.0]);
    /// let result = vector.zero_pad(4, PaddingOption::End).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 2.0, 0.0, 0.0], result.data());
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0]);
    /// let result = vector.zero_pad(2, PaddingOption::End).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 2.0, 0.0, 0.0], result.data());
    /// ```
    fn zero_pad(self, points: usize, option: PaddingOption) -> VecResult<Self>;
    
    /// Reverses the data inside the vector.
    fn reverse(self) -> VecResult<Self>;
    
    /// Ineterleaves zeros `factor - 1`times after every vector element, so that the resulting
    /// vector will have a length of `self.len() * factor`.
    ///
    /// Note: Remember that each complex number consists of two floating points and interleaving 
    /// will take that into account.
    ///
    /// If factor is 0 (zero) then `self` will be returned.
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, ComplexTimeVector32, GenericVectorOps, DataVector};
    /// let vector = RealTimeVector32::from_array(&[1.0, 2.0]);
    /// let result = vector.zero_interleave(2).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 0.0, 2.0, 0.0], result.data());
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = vector.zero_interleave(2).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0], result.data());
    /// ```
    fn zero_interleave(self, factor: u32) -> VecResult<Self>;
    
    /// Calculates the delta of each elements to its previous element. This will decrease the vector length by one point. 
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// let vector = RealTimeVector32::from_array(&[2.0, 3.0, 2.0, 6.0]);
    /// let result = vector.diff().expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, -1.0, 4.0], result.data());
    /// ```
    fn diff(self) -> VecResult<Self>;
    
    /// Calculates the delta of each elements to its previous element. The first element
    /// will remain unchanged.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// let vector = RealTimeVector32::from_array(&[2.0, 3.0, 2.0, 6.0]);
    /// let result = vector.diff_with_start().expect("Ignoring error handling in examples");
    /// assert_eq!([2.0, 1.0, -1.0, 4.0], result.data());
    /// ```
    fn diff_with_start(self) -> VecResult<Self>;
    
    /// Calculates the cumulative sum of all elements. This operation undoes the `diff_with_start`operation.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// let vector = RealTimeVector32::from_array(&[2.0, 1.0, -1.0, 4.0]);
    /// let result = vector.cum_sum().expect("Ignoring error handling in examples");
    /// assert_eq!([2.0, 3.0, 2.0, 6.0], result.data());
    /// ```
    fn cum_sum(self) -> VecResult<Self>;
    
    /// Gets the square root of all vector elements.
    ///
    /// The sqrt of a negative number gives NaN and not a complex vector.  
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// # use std::f32;
    /// let vector = RealTimeVector32::from_array(&[1.0, 4.0, 9.0, 16.0, 25.0]);
    /// let result = vector.sqrt().expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 2.0, 3.0, 4.0, 5.0], result.data());
    /// let vector = RealTimeVector32::from_array(&[-1.0]);
    /// let result = vector.sqrt().expect("Ignoring error handling in examples");
    /// assert!(result[0].is_nan());
    /// ```
    fn sqrt(self) -> VecResult<Self>;
    
    /// Squares all vector elements.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// let vector = RealTimeVector32::from_array(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let result = vector.square().expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 4.0, 9.0, 16.0, 25.0], result.data());
    /// ```
    fn square(self) -> VecResult<Self>;
    
    /// Calculates the n-th root of every vector element.
    ///
    /// If the result would be a complex number then the vector will contain a NaN instead. So the vector
    /// will never convert itself to a complex vector during this operation.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// let vector = RealTimeVector32::from_array(&[1.0, 8.0, 27.0]);
    /// let result = vector.root(3.0).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 2.0, 3.0], result.data());
    /// ```
    fn root(self, degree: T) -> VecResult<Self>;
    
    /// Raises every vector element to a floating point power.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// let vector = RealTimeVector32::from_array(&[1.0, 2.0, 3.0]);
    /// let result = vector.powf(3.0).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 8.0, 27.0], result.data());
    /// ```
    fn powf(self, exponent: T) -> VecResult<Self>;
    
    /// Computes the principal value of natural logarithm of every element in the vector.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// let vector = RealTimeVector32::from_array(&[2.718281828459045    , 7.389056, 20.085537]);
    /// let result = vector.ln().expect("Ignoring error handling in examples");
    /// let actual = result.data();
    /// let expected = &[1.0, 2.0, 3.0];
    /// assert_eq!(actual.len(), expected.len());
    /// for i in 0..actual.len() {
    ///        assert!((actual[i] - expected[i]).abs() < 1e-4);
    /// }
    /// ```
    fn ln(self) -> VecResult<Self>;
    
    /// Calculates the natural exponential for every vector element.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// let vector = RealTimeVector32::from_array(&[1.0, 2.0, 3.0]);
    /// let result = vector.exp().expect("Ignoring error handling in examples");
    /// assert_eq!([2.71828182846, 7.389056, 20.085537], result.data());
    /// ```
    fn exp(self) -> VecResult<Self>;
    
    /// Calculates the logarithm to the given base for every vector element.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// let vector = RealTimeVector32::from_array(&[10.0, 100.0, 1000.0]);
    /// let result = vector.log(10.0).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 2.0, 3.0], result.data());
    /// ```
    fn log(self, base: T) -> VecResult<Self>;
    
    /// Calculates the exponential to the given base for every vector element.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// let vector = RealTimeVector32::from_array(&[1.0, 2.0, 3.0]);
    /// let result = vector.expf(10.0).expect("Ignoring error handling in examples");
    /// assert_eq!([10.0, 100.0, 1000.0], result.data());
    /// ```
    fn expf(self, base: T) -> VecResult<Self>;
    
    /// Calculates the sine of each element in radians.
    ///
    /// # Example
    ///
    /// ```
    /// use std::f32;
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// let vector = RealTimeVector32::from_array(&[f32::consts::PI/2.0, -f32::consts::PI/2.0]);
    /// let result = vector.sin().expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, -1.0], result.data());
    /// ```
    fn sin(self) -> VecResult<Self>;
    
    /// Calculates the cosine of each element in radians.
    ///
    /// # Example
    ///
    /// ```
    /// use std::f32;
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// let vector = RealTimeVector32::from_array(&[2.0 * f32::consts::PI, f32::consts::PI]);
    /// let result = vector.cos().expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, -1.0], result.data());
    /// ```
    fn cos(self) -> VecResult<Self>;
    
    /// Calculates the tangent of each element in radians.
    fn tan(self) -> VecResult<Self>;
    
    /// Calculates the principal value of the inverse sine of each element in radians.
    fn asin(self) -> VecResult<Self>;
    
    /// Calculates the principal value of the inverse cosine of each element in radians.
    fn acos(self) -> VecResult<Self>;
    
    /// Calculates the principal value of the inverse tangent of each element in radians.
    fn atan(self) -> VecResult<Self>;
    
    /// Calculates the hyperbolic sine each element in radians.
    fn sinh(self) -> VecResult<Self>;
    
    /// Calculates the hyperbolic cosine each element in radians.
    fn cosh(self) -> VecResult<Self>;
    
    /// Calculates the hyperbolic tangent each element in radians.
    fn tanh(self) -> VecResult<Self>;
    
    /// Calculates the principal value of the inverse hyperbolic sine of each element in radians.
    fn asinh(self) -> VecResult<Self>;
    
    /// Calculates the principal value of the inverse hyperbolic cosine of each element in radians.
    fn acosh(self) -> VecResult<Self>;
    
    /// Calculates the principal value of the inverse hyperbolic tangent of each element in radians.
    fn atanh(self) -> VecResult<Self>;
    
    /// This function swaps both halves of the vector. This operation is also called fft shift
    /// Use it after a `plain_fft` to get a spectrum which is centered at `0 Hz`.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// let vector = RealTimeVector32::from_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    /// let result = vector.swap_halves().expect("Ignoring error handling in examples");
    /// assert_eq!([5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0], result.data());
    /// ```
    fn swap_halves(self) -> VecResult<Self>;
    
    /// Splits the vector into several smaller vectors. `self.len()` must be dividable by
    /// `targets.len()` without a remainder and this conidition must be true too `targets.len() > 0`.
    /// # Failures
    /// VecResult may report the following `ErrorReason` members:
    /// 
    /// 1. `InvalidArgumentLength`: `self.points()` isn't dividable by `targets.len()`
    ///     
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    /// let merge = RealTimeVector32::from_array(&a);
    /// let mut split = &mut 
    ///     [Box::new(RealTimeVector32::empty()), 
    ///     Box::new(RealTimeVector32::empty())];
    /// merge.split_into(split).unwrap();
    /// assert_eq!([1.0, 3.0, 5.0, 7.0, 9.0], split[0].data());
    /// ```
    fn split_into(&self, targets: &mut [Box<Self>]) -> VoidResult;
    
    /// Merges several vectors into `self`. All vectors must have the same size and
    /// at least one vector must be provided.
    /// # Failures
    /// VecResult may report the following `ErrorReason` members:
    /// 
    /// 1. `InvalidArgumentLength`: if `sources.len() == 0`
    ///    
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// let vector = RealTimeVector32::empty();
    /// let parts = &[
    ///     Box::new(RealTimeVector32::from_array(&[1.0, 2.0])),
    ///     Box::new(RealTimeVector32::from_array(&[1.0, 2.0]))];
    /// let merged = vector.merge(parts).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 1.0, 2.0, 2.0], merged.data());
    /// ```
    fn merge(self, sources: &[Box<Self>]) -> VecResult<Self>;
    
    /// Overrides the data in the vector with the given data. This may also change 
    /// the vectors length (however not the allocated length).
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, GenericVectorOps, DataVector};
    /// let vector = RealTimeVector32::from_array(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = vector.override_data(&[5.0, 7.0]).expect("Ignoring error handling in examples");
    /// assert_eq!(&[5.0, 7.0], result.data());
    /// ```
    fn override_data(self, data: &[T]) -> VecResult<Self>;
}

/// Defines all operations which are valid on `DataVectors` containing real data.
/// # Failures
/// All operations in this trait fail with `VectorMustBeReal` if the vector isn't in the real number space.
pub trait RealVectorOps<T> : DataVector<T> 
    where T : RealNumber {
    type ComplexPartner;
    
    /// Adds a scalar to the vector. See also [`OffsetOps`](trait.OffsetOps.html).
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, RealVectorOps, DataVector};
    /// let vector = RealTimeVector32::from_array(&[1.0, 2.0]);
    /// let result = vector.real_offset(2.0).expect("Ignoring error handling in examples");
    /// assert_eq!([3.0, 4.0], result.data());
    /// ```
    fn real_offset(self, factor: T) -> VecResult<Self>;
    
    /// Multiplies the vector with a scalar. See also [`ScaleOps`](trait.ScaleOps.html).
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, RealVectorOps, DataVector};
    /// let vector = RealTimeVector32::from_array(&[1.0, 2.0]);
    /// let result = vector.real_scale(4.0).expect("Ignoring error handling in examples");
    /// assert_eq!([4.0, 8.0], result.data());
    /// ```
    fn real_scale(self, offset: T) -> VecResult<Self>;
    
    /// Gets the absolute value of all vector elements.
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, RealVectorOps, DataVector};
    /// let vector = RealTimeVector32::from_array(&[1.0, -2.0]);
    /// let result = vector.abs().expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 2.0], result.data());
    /// ```
    fn abs(self) -> VecResult<Self>;
        
    /// Converts the real vector into a complex vector.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, RealVectorOps, DataVector};
    /// let vector = RealTimeVector32::from_array(&[1.0, 2.0]);
    /// let result = vector.to_complex().expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 0.0, 2.0, 0.0], result.data());
    /// ```
    fn to_complex(self) -> VecResult<Self::ComplexPartner>;
    
    /// Each value in the vector is devided by the divisor and the remainder is stored in the resulting 
    /// vector. This the same a modulo operation or to phase wrapping.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, RealVectorOps, DataVector};
    /// let vector = RealTimeVector32::from_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    /// let result = vector.wrap(4.0).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0], result.data());
    /// ```
    fn wrap(self, divisor: T) -> VecResult<Self>;
    
    /// This function corrects the jumps in the given vector which occur due to wrap or modulo operations.
    /// This will undo a wrap operation only if the deltas are smaller than half the divisor.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, RealVectorOps, DataVector};
    /// let vector = RealTimeVector32::from_array(&[1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0]);
    /// let result = vector.unwrap(4.0).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], result.data());
    /// ```
    fn unwrap(self, divisor: T) -> VecResult<Self>;
    
    /// Calculates the dot product of self and factor. Self and factor remain unchanged. See also  [`DotProductOps`](trait.DotProductOps.html).
    /// # Failures
    /// VecResult may report the following `ErrorReason` members:
    /// 
    /// 1. `VectorMetaDataMustAgree`: `self` and `factor` must be in the same domain and number space
    /// 
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, RealVectorOps};
    /// let vector1 = RealTimeVector32::from_array(&[9.0, 2.0, 7.0]);
    /// let vector2 = RealTimeVector32::from_array(&[4.0, 8.0, 10.0]);
    /// let result = vector1.real_dot_product(&vector2).expect("Ignoring error handling in examples");
    /// assert_eq!(122.0, result);
    /// ```  
    fn real_dot_product(&self, factor: &Self) -> ScalarResult<T>;
    
    /// Calculates the statistics of the data contained in the vector. See also  [`StatisticsOps`](trait.StatisticsOps.html).
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, RealVectorOps};
    /// let vector = RealTimeVector32::from_array(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let result = vector.real_statistics();
    /// assert_eq!(result.sum, 15.0);
    /// assert_eq!(result.count, 5);
    /// assert_eq!(result.average, 3.0);
    /// assert!((result.rms - 3.3166).abs() < 1e-4);
    /// assert_eq!(result.min, 1.0);
    /// assert_eq!(result.min_index, 0);
    /// assert_eq!(result.max, 5.0);
    /// assert_eq!(result.max_index, 4);
    /// ```  
    fn real_statistics(&self) -> Statistics<T>;
    
    /// Calculates the statistics of the data contained in the vector as if the vector would
    /// have been split into `len` pieces. `self.len` should be devisable by `len` without a remainder,
    /// but this isn't enforced by the implementation. See also  [`StatisticsOps`](trait.StatisticsOps.html).
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{RealTimeVector32, RealVectorOps};
    /// let vector = RealTimeVector32::from_array(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = vector.real_statistics_splitted(2);
    /// assert_eq!(result[0].sum, 4.0);
    /// assert_eq!(result[1].sum, 6.0);
    /// ```  
    fn real_statistics_splitted(&self, len: usize) -> Vec<Statistics<T>>;
    
    fn map_inplace_real<A, F>(self, argument: A, map: F) -> VecResult<Self>
        where A: Sync + Copy + Send,
              F: Fn(T, usize, A) -> T + 'static + Sync;
    
    fn map_aggregate_real<A, FMap, FAggr, R>(
            &self, 
            argument: A, 
            map: FMap,
            aggregate: FAggr) -> ScalarResult<R>
        where A: Sync + Copy + Send,
              FMap: Fn(T, usize, A) -> R + 'static + Sync,
              FAggr: Fn(R, R) -> R + 'static + Sync,
              R: Send;
}

/// Defines all operations which are valid on `DataVectors` containing complex data.
/// # Failures
/// All operations in this trait fail with `VectorMustBeComplex` if the vector isn't in the complex number space.
pub trait ComplexVectorOps<T> : DataVector<T> 
    where T : RealNumber {
    type RealPartner;
    
    /// Gets `self.data()` as complex array.
    fn complex_data(&self) -> &[Complex<T>];
    
    /// Adds a scalar to the vector. See also  [`OffsetOps`](trait.OffsetOps.html).
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp;
    /// use basic_dsp::{ComplexTimeVector32, ComplexVectorOps, DataVector};
    /// use num::complex::Complex32;
    /// # fn main() { 
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = vector.complex_offset(Complex32::new(-1.0, 2.0)).expect("Ignoring error handling in examples");
    /// assert_eq!([0.0, 4.0, 2.0, 6.0], result.data());
    /// # }
    /// ```
    fn complex_offset(self, offset: Complex<T>) -> VecResult<Self>;
    
    /// Multiplies the vector with a scalar. See also  [`ScaleOps`](trait.ScaleOps.html).
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp;
    /// use basic_dsp::{ComplexTimeVector32, ComplexVectorOps, DataVector};
    /// use num::complex::Complex32;
    /// # fn main() { 
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = vector.complex_scale(Complex32::new(-1.0, 2.0)).expect("Ignoring error handling in examples");
    /// assert_eq!([-5.0, 0.0, -11.0, 2.0], result.data());
    /// # }
    /// ```
    fn complex_scale(self, factor: Complex<T>) -> VecResult<Self>;
    
    /// Multiplies each vector element with `exp(j*(a*idx*self.delta() + b))`
    /// where `a` and `b` are arguments and `idx` is the index of the data points
    /// in the vector ranging from `0 to self.points() - 1`. `j` is the imaginary number and
    /// `exp` the exponential function.
    ///
    /// This method can be used to perform a frequency shift in time domain.
    ///
    /// # Example
    ///
    /// ```
    /// use basic_dsp::{ComplexTimeVector32, ComplexVectorOps, DataVector};
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = vector.multiply_complex_exponential(2.0, 3.0).expect("Ignoring error handling in examples");
    /// let expected = [-1.2722325, -1.838865, 4.6866837, -1.7421241];
    /// let result = result.data();
    /// for i in 0..expected.len() {
    ///     assert!((result[i] - expected[i]).abs() < 1e-4);   
    /// }
    /// ```
    fn multiply_complex_exponential(mut self, a: T, b: T) -> VecResult<Self>;
    
    /// Gets the absolute value, magnitude or norm of all vector elements.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp;
    /// use basic_dsp::{ComplexTimeVector32, ComplexVectorOps, DataVector};
    /// use num::complex::Complex32;
    /// # fn main() { 
    /// let vector = ComplexTimeVector32::from_interleaved(&[3.0, -4.0, -3.0, 4.0]);
    /// let result = vector.magnitude().expect("Ignoring error handling in examples");
    /// assert_eq!([5.0, 5.0], result.data());
    /// # }
    /// ```
    fn magnitude(self) -> VecResult<Self::RealPartner>;
    
    /// Copies the absolute value or magnitude of all vector elements into the given target vector.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp;
    /// use basic_dsp::{ComplexTimeVector32, RealTimeVector32, ComplexVectorOps, DataVector};
    /// # fn main() { 
    /// let vector = ComplexTimeVector32::from_interleaved(&[3.0, -4.0, -3.0, 4.0]);
    /// let mut result = RealTimeVector32::from_array(&[0.0]);
    /// vector.get_magnitude(&mut result).expect("Ignoring error handling in examples");
    /// assert_eq!([5.0, 5.0], result.data());
    /// # }
    /// ```
    fn get_magnitude(&self, destination: &mut Self::RealPartner) -> VoidResult;
    
    /// Gets the square root of the absolute value of all vector elements.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp;
    /// use basic_dsp::{ComplexTimeVector32, ComplexVectorOps, DataVector};
    /// use num::complex::Complex32;
    /// # fn main() { 
    /// let vector = ComplexTimeVector32::from_interleaved(&[3.0, -4.0, -3.0, 4.0]);
    /// let result = vector.magnitude_squared().expect("Ignoring error handling in examples");
    /// assert_eq!([25.0, 25.0], result.data());
    /// # }
    /// ```
    fn magnitude_squared(self) -> VecResult<Self::RealPartner>;
    
    /// Calculates the complex conjugate of the vector. 
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp;
    /// use basic_dsp::{ComplexTimeVector32, ComplexVectorOps, DataVector};
    /// # fn main() { 
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = vector.conj().expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, -2.0, 3.0, -4.0], result.data());
    /// # }
    /// ```
    fn conj(self) -> VecResult<Self>;
    
    /// Gets all real elements.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp;
    /// use basic_dsp::{ComplexTimeVector32, ComplexVectorOps, DataVector};
    /// # fn main() { 
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = vector.to_real().expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 3.0], result.data());
    /// # }
    /// ```
    fn to_real(self) -> VecResult<Self::RealPartner>;
    
    /// Gets all imag elements.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp;
    /// use basic_dsp::{ComplexTimeVector32, ComplexVectorOps, DataVector};
    /// # fn main() { 
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = vector.to_imag().expect("Ignoring error handling in examples");
    /// assert_eq!([2.0, 4.0], result.data());
    /// # }
    /// ```
    fn to_imag(self) -> VecResult<Self::RealPartner>;
    
    /// Copies all real elements into the given vector.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp;
    /// use basic_dsp::{RealTimeVector32, ComplexTimeVector32, ComplexVectorOps, DataVector};
    /// # fn main() { 
    /// let mut result = RealTimeVector32::from_array(&[0.0, 0.0]);
    /// let vector = ComplexTimeVector32::from_real_imag(&[1.0, 3.0], &[2.0, 4.0]);
    /// vector.get_real(&mut result).expect("Ignoring error handling in examples");
    /// assert_eq!([1.0, 3.0], result.data());
    /// # }
    /// ```
    fn get_real(&self, destination: &mut Self::RealPartner) -> VoidResult;
    
    /// Copies all imag elements into the given vector.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp;
    /// use basic_dsp::{RealTimeVector32, ComplexTimeVector32, ComplexVectorOps, DataVector};
    /// # fn main() { 
    /// let mut result = RealTimeVector32::from_array(&[0.0, 0.0]);
    /// let vector = ComplexTimeVector32::from_real_imag(&[1.0, 3.0], &[2.0, 4.0]);
    /// vector.get_imag(&mut result).expect("Ignoring error handling in examples");
    /// assert_eq!([2.0, 4.0], result.data());
    /// # }
    /// ```
    fn get_imag(&self, destination: &mut Self::RealPartner) -> VoidResult;
    
    /// Gets the phase of all elements in [rad].
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp;
    /// use basic_dsp::{ComplexTimeVector32, ComplexVectorOps, DataVector};
    /// # fn main() { 
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 0.0, 0.0, 4.0, -2.0, 0.0, 0.0, -3.0, 1.0, 1.0]);
    /// let result = vector.phase().expect("Ignoring error handling in examples");
    /// assert_eq!([0.0, 1.5707964, 3.1415927, -1.5707964, 0.7853982], result.data());
    /// # }
    /// ```
    fn phase(self) -> VecResult<Self::RealPartner>;
    
    /// Copies the phase of all elements in [rad] into the given vector.
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp;
    /// use basic_dsp::{RealTimeVector32, ComplexTimeVector32, ComplexVectorOps, DataVector};
    /// # fn main() { 
    /// let mut result = RealTimeVector32::from_array(&[0.0, 0.0]);
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 0.0, 0.0, 4.0, -2.0, 0.0, 0.0, -3.0, 1.0, 1.0]);
    /// vector.get_phase(&mut result).expect("Ignoring error handling in examples");
    /// assert_eq!([0.0, 1.5707964, 3.1415927, -1.5707964, 0.7853982], result.data());
    /// # }
    /// ```
    fn get_phase(&self, destination: &mut Self::RealPartner) -> VoidResult;
    
    /// Calculates the dot product of self and factor. Self and factor remain unchanged. See also  [`DotProductOps`](trait.DotProductOps.html).
    /// # Failures
    /// VecResult may report the following `ErrorReason` members:
    /// 
    /// 1. `VectorMetaDataMustAgree`: `self` and `factor` must be in the same domain and number space
    /// 
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp;
    /// # use num::complex::Complex32;
    /// use basic_dsp::{ComplexTimeVector32, ComplexVectorOps};
    /// # fn main() { 
    /// let vector1 = ComplexTimeVector32::from_interleaved(&[9.0, 2.0, 7.0, 1.0]);
    /// let vector2 = ComplexTimeVector32::from_interleaved(&[4.0, 0.0, 10.0, 0.0]);
    /// let result = vector1.complex_dot_product(&vector2).expect("Ignoring error handling in examples");
    /// assert_eq!(Complex32::new(106.0, 18.0), result);
    /// }
    /// ```  
    fn complex_dot_product(&self, factor: &Self) -> ScalarResult<Complex<T>>;
    
    /// Calculates the statistics of the data contained in the vector. See also  [`StatisticsOps`](trait.StatisticsOps.html).
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp;
    /// # use num::complex::Complex32;
    /// use basic_dsp::{ComplexTimeVector32, ComplexVectorOps};
    /// # fn main() { 
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let result = vector.complex_statistics();
    /// assert_eq!(result.sum, Complex32::new(9.0, 12.0));
    /// assert_eq!(result.count, 3);
    /// assert_eq!(result.average, Complex32::new(3.0, 4.0));
    /// assert!((result.rms - Complex32::new(3.4027193, 4.3102784)).norm() < 1e-4);
    /// assert_eq!(result.min, Complex32::new(1.0, 2.0));
    /// assert_eq!(result.min_index, 0);
    /// assert_eq!(result.max, Complex32::new(5.0, 6.0));
    /// assert_eq!(result.max_index, 2);
    /// }
    /// ```  
    fn complex_statistics(&self) -> Statistics<Complex<T>>;
    
    /// Calculates the statistics of the data contained in the vector as if the vector would
    /// have been split into `len` pieces. `self.len` should be devisable by `len` without a remainder,
    /// but this isn't enforced by the implementation. See also  [`StatisticsOps`](trait.StatisticsOps.html).
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate basic_dsp;
    /// # use num::complex::Complex32;
    /// use basic_dsp::{ComplexTimeVector32, ComplexVectorOps};
    /// # fn main() { 
    /// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    /// let result = vector.complex_statistics_splitted(2);
    /// assert_eq!(result[0].sum, Complex32::new(6.0, 8.0));
    /// assert_eq!(result[1].sum, Complex32::new(10.0, 12.0));
    /// }
    /// ```  
    fn complex_statistics_splitted(&self, len: usize) -> Vec<Statistics<Complex<T>>>; 
    
    /// Gets the real and imaginary parts and stores them in the given vectors. 
    /// See also  [`get_phase`](trait.ComplexVectorOps.html#tymethod.get_phase) and
    /// [`get_complex_abs`](trait.ComplexVectorOps.html#tymethod.get_complex_abs) for further
    /// information.
    fn get_real_imag(&self, real: &mut Self::RealPartner, imag: &mut Self::RealPartner) -> VoidResult;
    
    /// Gets the magnitude and phase and stores them in the given vectors.
    /// See also [`get_real`](trait.ComplexVectorOps.html#tymethod.get_real) and
    /// [`get_imag`](trait.ComplexVectorOps.html#tymethod.get_imag) for further
    /// information.
    fn get_mag_phase(&self, mag: &mut Self::RealPartner, phase: &mut Self::RealPartner) -> VoidResult;
    
    /// Overrides the `self` vectors data with the real and imaginary data in the given vectors.
    /// `real` and `imag` must have the same size.
    fn set_real_imag(self, real: &Self::RealPartner, imag: &Self::RealPartner) -> VecResult<Self>;
    
    /// Overrides the `self` vectors data with the magnitude and phase data in the given vectors.
    /// Note that `self` vector will immediately convert the data into a real and imaginary representation
    /// of the complex numbers which is its default format. 
    /// `mag` and `phase` must have the same size.
    fn set_mag_phase(self, mag: &Self::RealPartner, phase: &Self::RealPartner) -> VecResult<Self>;
}

/// Enumeration of all error reasons
#[derive(Copy)]
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub enum ErrorReason {
    /// The operations requires all vectors to have the same size, 
    /// in most cases this means that the following must be true:
    /// `self.len()` == `argument.len()`
    VectorsMustHaveTheSameSize,
    
    /// The operations requires all vectors to have the same meta data
    /// in most cases this means that the following must be true:
    /// `self.is_complex()` == `argument.is_complex()` &&
    /// `self.domain()` == `argument.domain()` &&
    /// `self.delta()`== `argument.domain()`;
    /// Consider to convert one of the vectors so that this conidition is true.
    /// The necessary operations may include FFT/IFFT, complex/real conversion and resampling.
    VectorMetaDataMustAgree,
    
    /// The operation requires the vector to be complex.
    VectorMustBeComplex,
    
    /// The operation requires the vector to be real.
    VectorMustBeReal,
    
    /// The operation requires the vector to be in time domain.
    VectorMustBeInTimeDomain,
    
    /// The operation requires the vector to be in frequency domain.
    VectorMustBeInFrquencyDomain,
    
    /// The arguments have an invalid length to perform the operation. The
    /// operations documentation should have more information about the requirements.
    /// Please open a defect if this isn't the case.
    InvalidArgumentLength,
    
    /// The operations is only valid if the data vector contains half of a symmetric spectrum.
    /// The symmetry definition follows soon however more important is that the element at 0 Hz
    /// which happens to be the first vector element must be real. This is actually violated 
    /// if this error message appears and the rest of the definition is only listed here for
    /// completness.
    /// The required symmetry is that for every point `vector[x].conj() == vector[-x]`(pseudocode)
    /// where `x` is the x-axis position relativ to 0Hz and `conj` is the complex conjungate.
    VectorMustBeConjSymmetric,
    
    /// `self.points()` must be an odd number.
    VectorMustHaveAnOddLength,
    
    /// The function passed as argument must be symmetric
    ArgumentFunctionMustBeSymmetric,
    
    /// The number of arguments passed into a combined operation methods doesn't match
    /// with the number of arguments specified previously via the `add_op` methods.
    InvalidNumberOfArgumentsForCombinedOp,
    
    /// The operation isn't specified for an empty vector.
    VectorMustNotBeEmpty,
}

/// Result contains on success the vector. On failure it contains an error reason and an vector with invalid data
/// which still can be used in order to avoid memory allocation.
pub type VecResult<T> = result::Result<T, (ErrorReason, T)>;

/// Void/nothing in case of success or a reason in case of an error.
pub type VoidResult = result::Result<(), ErrorReason>;

/// Scalar result or a reason in case of an error.
pub type ScalarResult<T> = result::Result<T, ErrorReason>;

/// Statistics about the data in a vector
#[repr(C)]
#[derive(Copy)]
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub struct Statistics<T> {
    pub sum: T,
    pub count: usize,
    pub average: T,
    pub rms: T,
    pub min: T,
    pub min_index: usize,
    pub max: T,
    pub max_index: usize
}

/// An option which defines how a vector should be padded
#[derive(Copy)]
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub enum PaddingOption {
    /// Appends zeros to the end of the vector.
    End,
    /// Surrounds the vector with zeros at the beginning and at the end.
    Surround,
    
    /// Inserts zeros in the center of the vector
    Center,
}