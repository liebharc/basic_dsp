use basic_dsp_vector::*;

pub trait DataMatrix<T> : Sized
    where T : RealNumber
{
    fn get_row(&self, row: usize) -> &[T];

    fn delta(&self) -> T;

    fn domain(&self) -> DataVecDomain;

    fn is_complex(&self) -> bool;
}
