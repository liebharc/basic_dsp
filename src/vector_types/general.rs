/// DataVector gives access to the basic properties of all data vectors
pub trait DataVector
{
	type E;
	fn data(&self) -> &[Self::E];
	fn delta(&self) -> Self::E;
	fn domain(&self) -> DataVectorDomain;
	fn is_complex(&self) -> bool;
	fn len(&self) -> usize;
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