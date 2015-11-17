/// DataVector gives access to the basic properties of all data vectors
pub trait DataVector
{
	/// The underlying data type of the vector: `f32` or `f64`. 
	type E;
	
	/// Gives direct access to the underlying data sequence. It's highly discouraged to use this function.
	/// It's even more discouraged to hold references to this array while executing operations on the vector,
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