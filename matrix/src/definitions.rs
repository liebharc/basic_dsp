use basic_dsp_vector::*;

pub trait DataMatrix<T> : Sized
    where T : RealNumber
{
    fn delta(&self) -> T;

    fn domain(&self) -> DataVecDomain;

    fn is_complex(&self) -> bool;
}
