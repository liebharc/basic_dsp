use std::result;
use num::complex::Complex;
use super::super::RealNumber;
use super::{
    ComplexTimeVector,
    RealTimeVector,
    GenericDataVector,
    RealFreqVector,
    ComplexFreqVector  
};

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
pub trait RededicateVector<T> : DataVector<T>
    where T: RealNumber {
    /// Make `self` a complex time vector
    /// # Example
	///
	/// ```
	/// use basic_dsp::{ComplexFreqVector32, ComplexVectorOperations, RededicateVector, DataVector, DataVectorDomain};
	/// let complex = ComplexFreqVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
	/// let real = complex.phase().expect("Ignoring error handling in examples");
    /// let complex = real.rededicate_as_complex_time_vector(2.0);
	/// assert_eq!(true, complex.is_complex());
    /// assert_eq!(DataVectorDomain::Time, complex.domain());
    /// assert_eq!(0, complex.len());
    /// assert_eq!(4, complex.allocated_len());
	/// ```
    fn rededicate_as_complex_time_vector(self, delta: T) -> ComplexTimeVector<T>;
    
    /// Make `self` a complex frequency vector
    fn rededicate_as_complex_freq_vector(self, delta: T) -> ComplexFreqVector<T>;
    
    /// Make `self` a real time vector
    fn rededicate_as_real_time_vector(self, delta: T) -> RealTimeVector<T>;
    
    /// Make `self` a real freq vector
    fn rededicate_as_real_freq_vector(self, delta: T) -> RealFreqVector<T>;
    
    /// Make `self` a generic vector
    fn rededicate_as_generic_vector(self, is_complex: bool, domain: DataVectorDomain, delta: T) -> GenericDataVector<T>;
}

/// Defines all operations which are valid on all `DataVectors`.
pub trait GenericVectorOperations<T>: DataVector<T> 
    where T : RealNumber {
	/// Calculates the sum of `self + summand`. It consumes self and returns the result.
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
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
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
	/// let vector1 = RealTimeVector32::from_array(&[10.0, 11.0, 12.0, 13.0]);
	/// let vector2 = RealTimeVector32::from_array(&[1.0, 2.0]);
	/// let result = vector1.add_smaller_vector(&vector2).expect("Ignoring error handling in examples");
	/// assert_eq!([11.0, 13.0, 13.0, 15.0], result.data());
	/// ```
	fn add_smaller_vector(self, summand: &Self) -> VecResult<Self>;
	
	/// Calculates the difference of `self - subtrahend`. It consumes self and returns the result.
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
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
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
	/// let vector1 = RealTimeVector32::from_array(&[10.0, 11.0, 12.0, 13.0]);
	/// let vector2 = RealTimeVector32::from_array(&[1.0, 2.0]);
	/// let result = vector1.subtract_smaller_vector(&vector2).expect("Ignoring error handling in examples");
	/// assert_eq!([9.0, 9.0, 11.0, 11.0], result.data());
	/// ```
	fn subtract_smaller_vector(self, summand: &Self) -> VecResult<Self>;
	
	/// Calculates the product of `self * factor`. It consumes self and returns the result.
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
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
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
	/// let vector1 = RealTimeVector32::from_array(&[10.0, 11.0, 12.0, 13.0]);
	/// let vector2 = RealTimeVector32::from_array(&[1.0, 2.0]);
	/// let result = vector1.multiply_smaller_vector(&vector2).expect("Ignoring error handling in examples");
	/// assert_eq!([10.0, 22.0, 12.0, 26.0], result.data());
	/// ```
	fn multiply_smaller_vector(self, factor: &Self) -> VecResult<Self>;
	
	/// Calculates the quotient of `self / summand`. It consumes self and returns the result.
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
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
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
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
	/// use basic_dsp::{RealTimeVector32, ComplexTimeVector32, GenericVectorOperations, DataVector};
	/// let vector = RealTimeVector32::from_array(&[1.0, 2.0]);
	/// let result = vector.zero_pad(4).expect("Ignoring error handling in examples");
	/// assert_eq!([1.0, 2.0, 0.0, 0.0], result.data());
	/// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0]);
	/// let result = vector.zero_pad(2).expect("Ignoring error handling in examples");
	/// assert_eq!([1.0, 2.0, 0.0, 0.0], result.data());
	/// ```
	fn zero_pad(self, points: usize) -> VecResult<Self>;
	
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
	/// use basic_dsp::{RealTimeVector32, ComplexTimeVector32, GenericVectorOperations, DataVector};
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
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
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
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
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
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
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
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
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
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
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
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
	/// let vector = RealTimeVector32::from_array(&[1.0, 8.0, 27.0]);
	/// let result = vector.root(3.0).expect("Ignoring error handling in examples");
	/// assert_eq!([1.0, 2.0, 3.0], result.data());
	/// ```
	fn root(self, degree: T) -> VecResult<Self>;
	
	/// Raises every vector element to the given power.
	///
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
	/// let vector = RealTimeVector32::from_array(&[1.0, 2.0, 3.0]);
	/// let result = vector.power(3.0).expect("Ignoring error handling in examples");
	/// assert_eq!([1.0, 8.0, 27.0], result.data());
	/// ```
	fn power(self, exponent: T) -> VecResult<Self>;
	
	/// Calculates the natural logarithm to the base e for every vector element.
	///
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
	/// let vector = RealTimeVector32::from_array(&[2.718281828459045	, 7.389056, 20.085537]);
	/// let result = vector.logn().expect("Ignoring error handling in examples");
	/// let actual = result.data();
	/// let expected = &[1.0, 2.0, 3.0];
	/// assert_eq!(actual.len(), expected.len());
	/// for i in 0..actual.len() {
	///		assert!((actual[i] - expected[i]).abs() < 1e-4);
	/// }
	/// ```
	fn logn(self) -> VecResult<Self>;
	
	/// Calculates the natural exponential to the base e for every vector element.
	///
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
	/// let vector = RealTimeVector32::from_array(&[1.0, 2.0, 3.0]);
	/// let result = vector.expn().expect("Ignoring error handling in examples");
	/// assert_eq!([2.71828182846, 7.389056, 20.085537], result.data());
	/// ```
	fn expn(self) -> VecResult<Self>;
	
	/// Calculates the logarithm to the given base for every vector element.
	///
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
	/// let vector = RealTimeVector32::from_array(&[10.0, 100.0, 1000.0]);
	/// let result = vector.log_base(10.0).expect("Ignoring error handling in examples");
	/// assert_eq!([1.0, 2.0, 3.0], result.data());
	/// ```
	fn log_base(self, base: T) -> VecResult<Self>;
	
	/// Calculates the exponential to the given base for every vector element.
	///
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
	/// let vector = RealTimeVector32::from_array(&[1.0, 2.0, 3.0]);
	/// let result = vector.exp_base(10.0).expect("Ignoring error handling in examples");
	/// assert_eq!([10.0, 100.0, 1000.0], result.data());
	/// ```
	fn exp_base(self, base: T) -> VecResult<Self>;
    
    /// Calculates the sine of each element in radians.
    ///
    /// # Example
	///
	/// ```
    /// use std::f32;
    /// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
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
    /// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
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
    
    /// This function swaps both halves of the vector.
	///
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
	/// let vector = RealTimeVector32::from_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
	/// let result = vector.swap_halves().expect("Ignoring error handling in examples");
	/// assert_eq!([5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0], result.data());
	/// ```
    fn swap_halves(self) -> VecResult<Self>;
    
    /// Splits the vector into several smaller vectors. `self.len()` must be dividable by
    /// `targets.len()` without a remainder and this conidition must be true too `targets.len() > 0`.
    ///
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
    /// let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
	/// let merge = RealTimeVector32::from_array(&a);
    /// let mut split = &mut 
    ///     [Box::new(RealTimeVector32::real_empty()), 
    ///     Box::new(RealTimeVector32::real_empty())];
    /// merge.split_into(split).unwrap();
    /// assert_eq!([1.0, 3.0, 5.0, 7.0, 9.0], split[0].data());
	/// ```
    fn split_into(&self, targets: &mut [Box<Self>]) -> VoidResult;
    
    /// Merges several vectors into `self`. All vectors must have the same size and
    /// at least one vector must be provided.
    ///
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
	/// let vector = RealTimeVector32::real_empty();
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
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
	/// let vector = RealTimeVector32::from_array(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = vector.override_data(&[5.0, 7.0]).expect("Ignoring error handling in examples");
	/// assert_eq!(&[5.0, 7.0], result.data());
	/// ```
    fn override_data(self, data: &[T]) -> VecResult<Self>;
}

/// Defines all operations which are valid on `DataVectors` containing real data.
pub trait RealVectorOperations<T> : DataVector<T> 
    where T : RealNumber {
	type ComplexPartner;
	
	/// Adds a scalar to the vector.
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, RealVectorOperations, DataVector};
	/// let vector = RealTimeVector32::from_array(&[1.0, 2.0]);
	/// let result = vector.real_offset(2.0).expect("Ignoring error handling in examples");
	/// assert_eq!([3.0, 4.0], result.data());
	/// ```
	fn real_offset(self, offset: T) -> VecResult<Self>;
	
	/// Multiplies the vector with a scalar.
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, RealVectorOperations, DataVector};
	/// let vector = RealTimeVector32::from_array(&[1.0, 2.0]);
	/// let result = vector.real_scale(4.0).expect("Ignoring error handling in examples");
	/// assert_eq!([4.0, 8.0], result.data());
	/// ```
	fn real_scale(self, offset: T) -> VecResult<Self>;
	
	/// Gets the absolute value of all vector elements.
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, RealVectorOperations, DataVector};
	/// let vector = RealTimeVector32::from_array(&[1.0, -2.0]);
	/// let result = vector.real_abs().expect("Ignoring error handling in examples");
	/// assert_eq!([1.0, 2.0], result.data());
	/// ```
	fn real_abs(self) -> VecResult<Self>;
		
	/// Converts the real vector into a complex vector.
	///
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, RealVectorOperations, DataVector};
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
	/// use basic_dsp::{RealTimeVector32, RealVectorOperations, DataVector};
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
	/// use basic_dsp::{RealTimeVector32, RealVectorOperations, DataVector};
	/// let vector = RealTimeVector32::from_array(&[1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0]);
	/// let result = vector.unwrap(4.0).expect("Ignoring error handling in examples");
	/// assert_eq!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], result.data());
	/// ```
	fn unwrap(self, divisor: T) -> VecResult<Self>;
    
    /// Calculates the dot product of self and factor. Self and factor remain unchanged.
    /// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, RealVectorOperations};
	/// let vector1 = RealTimeVector32::from_array(&[9.0, 2.0, 7.0]);
	/// let vector2 = RealTimeVector32::from_array(&[4.0, 8.0, 10.0]);
	/// let result = vector1.real_dot_product(&vector2).expect("Ignoring error handling in examples");
	/// assert_eq!(122.0, result);
	/// ```  
    fn real_dot_product(&self, factor: &Self) -> ScalarResult<T>;
    
    /// Calculates the statistics of the data contained in the vector.
    /// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, RealVectorOperations};
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
    /// but this isn't enforced by the implementation.
    /// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, RealVectorOperations};
	/// let vector = RealTimeVector32::from_array(&[1.0, 2.0, 3.0, 4.0]);
	/// let result = vector.real_statistics_splitted(2);
	/// assert_eq!(result[0].sum, 4.0);
    /// assert_eq!(result[1].sum, 6.0);
	/// ```  
    fn real_statistics_splitted(&self, len: usize) -> Vec<Statistics<T>>; 
}

/// Defines all operations which are valid on `DataVectors` containing complex data.
pub trait ComplexVectorOperations<T> : DataVector<T> 
    where T : RealNumber {
	type RealPartner;
	
	/// Adds a scalar to the vector.
	/// # Example
	///
	/// ```
	/// # extern crate num;
	/// # extern crate basic_dsp;
	/// use basic_dsp::{ComplexTimeVector32, ComplexVectorOperations, DataVector};
	/// use num::complex::Complex32;
	/// # fn main() { 
	/// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
	/// let result = vector.complex_offset(Complex32::new(-1.0, 2.0)).expect("Ignoring error handling in examples");
	/// assert_eq!([0.0, 4.0, 2.0, 6.0], result.data());
	/// # }
	/// ```
	fn complex_offset(self, offset: Complex<T>) -> VecResult<Self>;
	
	/// Multiplies the vector with a scalar.
	/// # Example
	///
	/// ```
	/// # extern crate num;
	/// # extern crate basic_dsp;
	/// use basic_dsp::{ComplexTimeVector32, ComplexVectorOperations, DataVector};
	/// use num::complex::Complex32;
	/// # fn main() { 
	/// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
	/// let result = vector.complex_scale(Complex32::new(-1.0, 2.0)).expect("Ignoring error handling in examples");
	/// assert_eq!([-5.0, 0.0, -11.0, 2.0], result.data());
	/// # }
	/// ```
	fn complex_scale(self, factor: Complex<T>) -> VecResult<Self>;
	
	/// Gets the absolute value or magnitude of all vector elements.
	/// # Example
	///
	/// ```
	/// # extern crate num;
	/// # extern crate basic_dsp;
	/// use basic_dsp::{ComplexTimeVector32, ComplexVectorOperations, DataVector};
	/// use num::complex::Complex32;
	/// # fn main() { 
	/// let vector = ComplexTimeVector32::from_interleaved(&[3.0, -4.0, -3.0, 4.0]);
	/// let result = vector.complex_abs().expect("Ignoring error handling in examples");
	/// assert_eq!([5.0, 5.0], result.data());
	/// # }
	/// ```
	fn complex_abs(self) -> VecResult<Self::RealPartner>;
	
	/// Copies the absolute value or magnitude of all vector elements into the given target vector.
	/// # Example
	///
	/// ```
	/// # extern crate num;
	/// # extern crate basic_dsp;
	/// use basic_dsp::{ComplexTimeVector32, RealTimeVector32, ComplexVectorOperations, DataVector};
	/// # fn main() { 
	/// let vector = ComplexTimeVector32::from_interleaved(&[3.0, -4.0, -3.0, 4.0]);
	/// let mut result = RealTimeVector32::from_array(&[0.0]);
	/// vector.get_complex_abs(&mut result).expect("Ignoring error handling in examples");
	/// assert_eq!([5.0, 5.0], result.data());
	/// # }
	/// ```
	fn get_complex_abs(&self, destination: &mut Self::RealPartner) -> VoidResult;
	
	/// Gets the square root of the absolute value of all vector elements.
	/// # Example
	///
	/// ```
	/// # extern crate num;
	/// # extern crate basic_dsp;
	/// use basic_dsp::{ComplexTimeVector32, ComplexVectorOperations, DataVector};
	/// use num::complex::Complex32;
	/// # fn main() { 
	/// let vector = ComplexTimeVector32::from_interleaved(&[3.0, -4.0, -3.0, 4.0]);
	/// let result = vector.complex_abs_squared().expect("Ignoring error handling in examples");
	/// assert_eq!([25.0, 25.0], result.data());
	/// # }
	/// ```
	fn complex_abs_squared(self) -> VecResult<Self::RealPartner>;
	
	/// Calculates the complex conjugate of the vector. 
	/// # Example
	///
	/// ```
	/// # extern crate num;
	/// # extern crate basic_dsp;
	/// use basic_dsp::{ComplexTimeVector32, ComplexVectorOperations, DataVector};
	/// use num::complex::Complex32;
	/// # fn main() { 
	/// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
	/// let result = vector.complex_conj().expect("Ignoring error handling in examples");
	/// assert_eq!([1.0, -2.0, 3.0, -4.0], result.data());
	/// # }
	/// ```
	fn complex_conj(self) -> VecResult<Self>;
	
	/// Gets all real elements.
	/// # Example
	///
	/// ```
	/// # extern crate num;
	/// # extern crate basic_dsp;
	/// use basic_dsp::{ComplexTimeVector32, ComplexVectorOperations, DataVector};
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
	/// use basic_dsp::{ComplexTimeVector32, ComplexVectorOperations, DataVector};
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
	/// use basic_dsp::{RealTimeVector32, ComplexTimeVector32, ComplexVectorOperations, DataVector};
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
	/// use basic_dsp::{RealTimeVector32, ComplexTimeVector32, ComplexVectorOperations, DataVector};
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
	/// use basic_dsp::{ComplexTimeVector32, ComplexVectorOperations, DataVector};
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
	/// use basic_dsp::{RealTimeVector32, ComplexTimeVector32, ComplexVectorOperations, DataVector};
	/// # fn main() { 
	/// let mut result = RealTimeVector32::from_array(&[0.0, 0.0]);
	/// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 0.0, 0.0, 4.0, -2.0, 0.0, 0.0, -3.0, 1.0, 1.0]);
	/// vector.get_phase(&mut result).expect("Ignoring error handling in examples");
	/// assert_eq!([0.0, 1.5707964, 3.1415927, -1.5707964, 0.7853982], result.data());
	/// # }
	/// ```
	fn get_phase(&self, destination: &mut Self::RealPartner) -> VoidResult;
    
    /// Calculates the dot product of self and factor. Self and factor remain unchanged.
    /// # Example
	///
	/// ```
    /// # extern crate num;
	/// # extern crate basic_dsp;
    /// # use num::complex::Complex32;
	/// use basic_dsp::{ComplexTimeVector32, ComplexVectorOperations};
	/// # fn main() { 
	/// let vector1 = ComplexTimeVector32::from_interleaved(&[9.0, 2.0, 7.0, 1.0]);
	/// let vector2 = ComplexTimeVector32::from_interleaved(&[4.0, 0.0, 10.0, 0.0]);
	/// let result = vector1.complex_dot_product(&vector2).expect("Ignoring error handling in examples");
	/// assert_eq!(Complex32::new(106.0, 18.0), result);
    /// }
	/// ```  
    fn complex_dot_product(&self, factor: &Self) -> ScalarResult<Complex<T>>;
    
    /// Calculates the statistics of the data contained in the vector.
    /// # Example
	///
	/// ```
	/// # extern crate num;
	/// # extern crate basic_dsp;
    /// # use num::complex::Complex32;
	/// use basic_dsp::{ComplexTimeVector32, ComplexVectorOperations};
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
    /// but this isn't enforced by the implementation.
    /// # Example
	///
	/// ```
	/// # extern crate num;
	/// # extern crate basic_dsp;
    /// # use num::complex::Complex32;
	/// use basic_dsp::{ComplexTimeVector32, ComplexVectorOperations};
	/// # fn main() { 
	/// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
	/// let result = vector.complex_statistics_splitted(2);
	/// assert_eq!(result[0].sum, Complex32::new(6.0, 8.0));
    /// assert_eq!(result[1].sum, Complex32::new(10.0, 12.0));
    /// }
	/// ```  
    fn complex_statistics_splitted(&self, len: usize) -> Vec<Statistics<Complex<T>>>; 
    
    /// Gets the real and imaginary parts and stores them in the given vectors. 
    /// See [`get_phase`](trait.ComplexVectorOperations.html#tymethod.get_phase) and
    /// [`get_complex_abs`](trait.ComplexVectorOperations.html#tymethod.get_complex_abs) for further
    /// information.
    fn get_real_imag(&self, real: &mut Self::RealPartner, imag: &mut Self::RealPartner) -> VoidResult;
    
    /// Gets the magnitude and phase and stores them in the given vectors.
    /// See [`get_real`](trait.ComplexVectorOperations.html#tymethod.get_real) and
    /// [`get_imag`](trait.ComplexVectorOperations.html#tymethod.get_imag) for further
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
}

/// Result contains on success the vector. On failure it contains an error reason and an vector with invalid data
/// which still can be used in order to avoid memory allocation.
pub type VecResult<T> = result::Result<T, (ErrorReason, T)>;

/// Result or a reason in case of an error.
pub type VoidResult = result::Result<(), ErrorReason>;

/// Result or a reason in case of an error.
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