use super::definitions::{
    DataVectorDomain,
    ErrorReason,
    DataVector,
    TransRes,
    PaddingOption};
use RealNumber;
use super::{
    GenericDataVector,
    RealVectorOps,
    GenericVectorOps,
    ComplexVectorOps,
    TimeDomainOperations,
    FrequencyDomainOperations,
    ComplexTimeVector,
    ComplexFreqVector};
    
/// Cross-correlation of data vectors. See also https://en.wikipedia.org/wiki/Cross-correlation
///
/// The correlation is calculated in two steps. This is done to give you more control over two things:
///
/// 1. Should the correlation use zero padding or not? This is done by calling either `prepare_argument`
///    or `prepare_argument_padded`.
/// 2. The lifetime of the argument. The argument needs to be transformed for the correlation and
///    depending on the application that might be just fine, or a clone needs to be created or
///    it's okay to use one argument for multiple correlations.
///
/// To get the same behavior like GNU Octave or MATLAB `prepare_argument_padded` needs to be 
/// called before doing the correlation. See also the example section for how to do this.
/// # Example
///
/// ```
/// use basic_dsp_vector::{ComplexTimeVector32, CrossCorrelation, DataVector};
/// let vector = ComplexTimeVector32::from_interleaved(&[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
/// let argument = ComplexTimeVector32::from_interleaved(&[3.0, 3.0, 2.0, 2.0, 1.0, 1.0]);
/// let argument = argument.prepare_argument_padded().expect("Ignoring error handling in examples");
/// let result = vector.correlate(&argument).expect("Ignoring error handling in examples");
/// let expected = &[2.0, 0.0, 8.0, 0.0, 20.0, 0.0, 24.0, 0.0, 18.0, 0.0];
/// for i in 0..result.len() {
///     assert!((result[i] - expected[i]).abs() < 1e-4);
/// }
/// ```
/// # Unstable
/// This functionality has been recently added in order to find out if the definitions are consistent.
/// However the actual implementation is lacking tests.
/// # Failures
/// TransRes may report the following `ErrorReason` members:
/// 
/// 1. `VectorMustBeComplex`: if `self` is in real number space.
/// 3. `VectorMetaDataMustAgree`: in case `self` and `function` are not in the same number space and same domain.
pub trait CrossCorrelation<T> : DataVector<T> 
    where T : RealNumber {
    type FreqPartner;
    /// Prepares an argument to be used for convolution. Preparing an argument includes two steps:
    /// 
    /// 1. Calculate the plain FFT
    /// 2. Calculate the complex conjugate
    fn prepare_argument(self) -> TransRes<Self::FreqPartner>;
    
    /// Prepares an argument to be used for convolution. The argument is zero padded to length of `2 * self.points() - 1`
    /// and then the same operations are performed as described for `prepare_argument`.
    fn prepare_argument_padded(self) -> TransRes<Self::FreqPartner>;
     
    /// Calculates the correlation between `self` and `other`. `other` needs to be a time vector which
    /// went through one of the prepare functions `prepare_argument` or `prepare_argument_padded`. See also the 
    /// trait description for more details.
    fn correlate(self, other: &Self::FreqPartner) -> TransRes<Self>;
}

macro_rules! define_correlation_impl {
    ($($data_type:ident,$reg:ident);*) => {
        $( 
            impl CrossCorrelation<$data_type> for GenericDataVector<$data_type> {
                type FreqPartner = Self;
                
                fn prepare_argument(self) -> TransRes<Self::FreqPartner> {
                    self.plain_fft()
                    .and_then(|v|v.conj())
                }
    
                fn prepare_argument_padded(self) -> TransRes<Self::FreqPartner> {
                    let points = self.points();
                    self.zero_pad(2 * points - 1, PaddingOption::Surround)
                    .and_then(|v|v.plain_fft())
                    .and_then(|v|v.conj())
                }
                
                fn correlate(self, other: &Self::FreqPartner) -> TransRes<Self> {
                    assert_complex!(self);
                    assert_time!(self);
                    let points = other.points();
                    Ok(self)
                    .and_then(|v|v.zero_pad(points, PaddingOption::Surround))
                    .and_then(|v|v.plain_fft())
                    .and_then(|v|v.multiply_vector(&other))
                    .and_then(|v| {
                        let p = v.points();
                        v.real_scale(1.0 / p as $data_type)
                    })
                    .and_then(|v|v.plain_ifft())
                    .and_then(|v|v.swap_halves())
                }
            }
        )*
    }
}
define_correlation_impl!(f32, Reg32; f64, Reg64);

macro_rules! define_correlation_forward {
    ($($name:ident, $data_type:ident);*) => {
        $( 
            impl CrossCorrelation<$data_type> for $name<$data_type> {
                type FreqPartner = ComplexFreqVector<$data_type>;
                fn prepare_argument(self) -> TransRes<Self::FreqPartner> {
                    Self::FreqPartner::from_genres(self.to_gen().prepare_argument())
                }
    
                fn prepare_argument_padded(self) -> TransRes<Self::FreqPartner> {
                    Self::FreqPartner::from_genres(self.to_gen().prepare_argument_padded())
                }
                
                fn correlate(self, other: &Self::FreqPartner) -> TransRes<Self> {
                    Self::from_genres(self.to_gen().correlate(other.to_gen_borrow()))
                }
            }
        )*
    }
}

define_correlation_forward!(
    ComplexTimeVector, f32; ComplexTimeVector, f64
);

#[cfg(test)]
mod tests {
    use vector_types::*;
    
    #[test]
    fn time_correlation_test() {
        let a = ComplexTimeVector32::from_interleaved(&[0.0800, 0.0, 0.1876, 0.1170, 0.4601, 0.4132, 0.7700, 0.7500, 0.9723, 0.9698, 0.9723, 0.9698, 0.7700, 0.7500, 0.4601, 0.4132, 0.1876, 0.1170, 0.0800, 0.0]);
        let b = ComplexTimeVector32::from_interleaved(&[0.1000, -0.6366, 0.3000, 0.0, 0.5000, 0.2122, 0.7000, 0.0, 0.9000, -0.1273, 0.9000, 0.0, 0.7000, 0.0909, 0.5000, 0.0, 0.3000, -0.0707, 0.1000, 0.0]);
        let c: &[f32] = &[0.0080, 0.0000, 0.0428, 0.0174, 0.1340, 0.0897, 0.3356, 0.2827, 0.7192, 0.6479, 1.3058, 1.1946, 2.0175, 1.8757,
                          2.7047, 2.5665, 3.2186, 3.0874, 3.4409, 3.2994, 3.2291, 3.1287, 2.5801, 2.7264, 1.7085, 2.1882, 0.8637, 1.6369,
                          0.2319, 1.1420, -0.0878, 0.7078, -0.1208, 0.3523, -0.0317, 0.1311, 0.0080, 0.0509];
        let b = b.prepare_argument_padded().unwrap();
        let res: ComplexTimeVector32 = a.correlate(&b).unwrap();
        let res = res.data();
        let tol = 0.1;
        for i in 0..c.len() {
            if (res[i] - c[i]).abs() > tol {
                panic!("assertion failed: {:?} != {:?} at index {}", res, c, i);
            }
        }
    }
    
    #[test]
    fn time_correlation_test2() {
        let a = ComplexTimeVector32::from_interleaved(&[1.0, 1.0, 2.0, 1.0, 3.0, 1.0]);
        let b = ComplexTimeVector32::from_interleaved(&[4.0, 1.0, 5.0, 1.0, 6.0, 1.0]);
        let c: &[f32] = &[7.0, 5.0, 19.0, 8.0, 35.0, 9.0, 25.0, 4.0, 13.0, 1.0];
        let b = b.prepare_argument_padded().unwrap();
        let res: ComplexTimeVector32 = a.correlate(&b).unwrap();
        let res = res.data();
        let tol = 0.1;
        for i in 0..c.len() {
            if (res[i] - c[i]).abs() > tol {
                panic!("assertion failed: {:?} != {:?} at index {}", res, c, i);
            }
        }
    }
}