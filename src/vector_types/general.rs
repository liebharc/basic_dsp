pub trait DataVector
{
	type E;
	fn data(&self) -> &[Self::E];
	fn delta(&self) -> Self::E;
	fn domain(&self) -> DataVectorDomain;
	fn is_complex(&self) -> bool;
	fn len(&self) -> usize;
}

#[derive(Copy)]
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub enum DataVectorDomain {
	Time,
    Frequency
}