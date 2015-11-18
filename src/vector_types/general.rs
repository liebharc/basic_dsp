/// DataVector gives access to the basic properties of all data vectors
pub trait DataVector
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
	
	/// The allocated length of the vector. The allocated length may be larger than the length of valid points. 
	/// In most cases you likely want to have `points` instead.
	fn len(&self) -> usize;
	
	/// The number of valid points. If the vector is complex then every valid point consists of two floating point numbers,
	/// while for real vectors every point only consists of one floating point number.
	fn points(&self) -> usize;
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
	/// let result = vector1.add_vector(&vector2);
	/// assert_eq!([11.0, 13.0], result.data());
	/// ```
	fn add_vector(self, summand: &Self) -> Self;
	
	/// Calculates the difference of `self - subtrahend`. It consumes self and returns the result.
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
	/// let vector1 = RealTimeVector32::from_array(&[1.0, 2.0]);
	/// let vector2 = RealTimeVector32::from_array(&[10.0, 11.0]);
	/// let result = vector1.subtract_vector(&vector2);
	/// assert_eq!([-9.0, -9.0], result.data());
	/// ```
	fn subtract_vector(self, subtrahend: &Self) -> Self;
	
	/// Calculates the product of `self * factor`. It consumes self and returns the result.
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
	/// let vector1 = RealTimeVector32::from_array(&[1.0, 2.0]);
	/// let vector2 = RealTimeVector32::from_array(&[10.0, 11.0]);
	/// let result = vector1.multiply_vector(&vector2);
	/// assert_eq!([10.0, 22.0], result.data());
	/// ```
	fn multiply_vector(self, factor: &Self) -> Self;
	
	/// Calculates the quotient of `self / summand`. It consumes self and returns the result.
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, GenericVectorOperations, DataVector};
	/// let vector1 = RealTimeVector32::from_array(&[10.0, 22.0]);
	/// let vector2 = RealTimeVector32::from_array(&[2.0, 11.0]);
	/// let result = vector1.divide_vector(&vector2);
	/// assert_eq!([5.0, 2.0], result.data());
	/// ```
	fn divide_vector(self, divisor: &Self) -> Self;
}

/// Defines all operations which are valid on `DataVectors` containing real data.
pub trait RealVectorOperations : DataVector {
	/// Adds a scalar to the vector.
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, RealVectorOperations, DataVector};
	/// let vector = RealTimeVector32::from_array(&[1.0, 2.0]);
	/// let result = vector.real_offset(2.0);
	/// assert_eq!([3.0, 4.0], result.data());
	/// ```
	fn real_offset(self, offset: Self::E) -> Self;
	
	/// Multiplies the vector with a scalar.
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, RealVectorOperations, DataVector};
	/// let vector = RealTimeVector32::from_array(&[1.0, 2.0]);
	/// let result = vector.real_scale(4.0);
	/// assert_eq!([4.0, 8.0], result.data());
	/// ```
	fn real_scale(self, offset: Self::E) -> Self;
	
	/// Gets the absolute value of all vector elements.
	/// # Example
	///
	/// ```
	/// use basic_dsp::{RealTimeVector32, RealVectorOperations, DataVector};
	/// let vector = RealTimeVector32::from_array(&[1.0, -2.0]);
	/// let result = vector.real_abs();
	/// assert_eq!([1.0, 2.0], result.data());
	/// ```
	fn real_abs(self) -> Self;
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
	/// let result = vector.complex_offset(Complex32::new(-1.0, 2.0));
	/// assert_eq!([0.0, 4.0, 2.0, 6.0], result.data());
	/// # }
	/// ```
	fn complex_offset(self, offset: Self::Complex) -> Self;
	
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
	/// let result = vector.complex_scale(Complex32::new(-1.0, 2.0));
	/// assert_eq!([-5.0, 0.0, -11.0, 2.0], result.data());
	/// # }
	/// ```
	fn complex_scale(self, factor: Self::Complex) -> Self;
	
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
	/// let result = vector.complex_abs();
	/// assert_eq!([5.0, 5.0], result.data());
	/// # }
	/// ```
	fn complex_abs(self) -> Self::RealPartner;
	
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
	/// let result = vector.complex_abs_squared();
	/// assert_eq!([25.0, 25.0], result.data());
	/// # }
	/// ```
	fn complex_abs_squared(self) -> Self::RealPartner;
}

/// Defines all operations which are valid on `DataVectors` containing real data.
pub trait TimeDomainOperations : DataVector {
	fn fft(self) -> Self;
}

/// Defines all operations which are valid on `DataVectors` containing complex data.
pub trait FrequencyDomainOperations : DataVector {
	fn ifft(self) -> Self;
}