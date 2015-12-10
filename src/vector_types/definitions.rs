use std::result;

/// DataVector gives access to the basic properties of all data vectors
///
/// A DataVector allocates memory if necessary. It will however never shrink/free memory unless it's 
/// deleted and dropped.
pub trait DataVector : Sized
{
	/// The underlying data type of the vector: `f32` or `f64`. 
	type E;
	
	/// Gives direct access to the underlying data sequence. It's recommended to use the `Index functions .
	/// For users outside of Rust: It's discouraged to hold references to this array while executing operations on the vector,
	/// since the vector may decide at any operation to invalidate the array. 
	fn data(&self) -> &[Self::E];
	
	/// The x-axis delta. If `domain` is time domain then `delta` is in `[s]`, in frequency domain `delta` is in `[Hz]`.
	fn delta(&self) -> Self::E;
	
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

/// Defines all operations which are valid on all `DataVectors`.
pub trait GenericVectorOperations : DataVector {
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
	
	/// Appends zeros add the end of the vector until the vector has the size given in the points argument.
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
	
	/// Ineterleaves zeros afeter every vector element.
	///
	/// Note: Remember that each complex number consists of two floating points and interleaving 
	/// will take that into account.
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, ComplexTimeVector32, GenericVectorOperations, DataVector};
	/// let vector = RealTimeVector32::from_array(&[1.0, 2.0]);
	/// let result = vector.zero_interleave().expect("Ignoring error handling in examples");
	/// assert_eq!([1.0, 0.0, 2.0, 0.0], result.data());
	/// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 2.0, 3.0, 4.0]);
	/// let result = vector.zero_interleave().expect("Ignoring error handling in examples");
	/// assert_eq!([1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0], result.data());
	/// ```
	fn zero_interleave(self) -> VecResult<Self>;
	
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
	fn root(self, degree: Self::E) -> VecResult<Self>;
	
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
	fn power(self, exponent: Self::E) -> VecResult<Self>;
	
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
	fn log_base(self, base: Self::E) -> VecResult<Self>;
	
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
	fn exp_base(self, base: Self::E) -> VecResult<Self>;
    
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
}

/// Defines all operations which are valid on `DataVectors` containing real data.
pub trait RealVectorOperations : DataVector {
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
	fn real_offset(self, offset: Self::E) -> VecResult<Self>;
	
	/// Multiplies the vector with a scalar.
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, RealVectorOperations, DataVector};
	/// let vector = RealTimeVector32::from_array(&[1.0, 2.0]);
	/// let result = vector.real_scale(4.0).expect("Ignoring error handling in examples");
	/// assert_eq!([4.0, 8.0], result.data());
	/// ```
	fn real_scale(self, offset: Self::E) -> VecResult<Self>;
	
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
	fn wrap(self, divisor: Self::E) -> VecResult<Self>;
	
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
	fn unwrap(self, divisor: Self::E) -> VecResult<Self>;
}

/// Defines all operations which are valid on `DataVectors` containing complex data.
pub trait ComplexVectorOperations : DataVector {
	type RealPartner;
	type Complex;
	
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
	fn complex_offset(self, offset: Self::Complex) -> VecResult<Self>;
	
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
	fn complex_scale(self, factor: Self::Complex) -> VecResult<Self>;
	
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
}

/// Defines all operations which are valid on `DataVectors` containing real data.
pub trait TimeDomainOperations : DataVector {
	type FreqPartner;
	
	/// Performs a Fast Fourier Transformation transforming a time domain vector
	/// into a frequency domain vector. 
	/// 
	/// This version of the FFT neither applies a window nor does it scale the 
	/// vector.
	/// # Example
	///
	/// ```
	/// use basic_dsp::{ComplexTimeVector32, TimeDomainOperations, DataVector};
	/// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 0.0, -0.5, 0.8660254, -0.5, -0.8660254]);
	/// let result = vector.plain_fft().expect("Ignoring error handling in examples");
	/// let actual = result.data();
	/// let expected = &[0.0, 0.0, 3.0, 0.0, 0.0, 0.0];
	/// assert_eq!(actual.len(), expected.len());
	/// for i in 0..actual.len() {
	///		assert!((actual[i] - expected[i]).abs() < 1e-4);
	/// }
	/// ```
	fn plain_fft(self) -> VecResult<Self::FreqPartner>;
	
	// TODO add fft method which also applies a window
}

/// Defines all operations which are valid on `DataVectors` containing complex data.
pub trait FrequencyDomainOperations : DataVector {
	type TimePartner;
	
	/// Performs an Inverse Fast Fourier Transformation transforming a frequency domain vector
	/// into a time domain vector.
	/// 
	/// This version of the IFFT neither applies a window nor does it scale the 
	/// vector.
	/// # Example
	///
	/// ```
	/// use basic_dsp::{ComplexFreqVector32, FrequencyDomainOperations, DataVector};
	/// let vector = ComplexFreqVector32::from_interleaved(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
	/// let result = vector.plain_ifft().expect("Ignoring error handling in examples");
	/// let actual = result.data();
	/// let expected = &[1.0, 0.0, -0.5, 0.8660254, -0.5, -0.8660254];
	/// assert_eq!(actual.len(), expected.len());
	/// for i in 0..actual.len() {
	///		assert!((actual[i] - expected[i]).abs() < 1e-4);
	/// }
	/// ```
	fn plain_ifft(self) -> VecResult<Self::TimePartner>;
}

#[derive(Copy)]
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub enum ErrorReason {
	VectorsMustHaveTheSameSize,
}

pub type VecResult<T> = result::Result<T, (ErrorReason, T)>;

pub type VoidResult = result::Result<(), ErrorReason>;