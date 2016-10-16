/// Enumeration of all error reasons
#[derive(Copy)]
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub enum ErrorReason {
    /// The operations requires all inputs to have the same size,
    /// in most cases this means that the following must be true:
    /// `self.len()` == `argument.len()`
    InputMustHaveTheSameSize,

    /// The operations requires all inputs to have the same meta data.
    /// For a vector this means that the following must be true:
    /// `self.is_complex()` == `argument.is_complex()` &&
    /// `self.domain()` == `argument.domain()` &&
    /// `self.delta()`== `argument.domain()`;
    /// Consider to convert one of the inputs so that this condition is true.
    /// The necessary operations may include FFT/IFFT, complex/real conversion and resampling.
    InputMetaDataMustAgree,

    /// The operation requires the input to be complex.
    InputMustBeComplex,

    /// The operation requires the input to be real.
    InputMustBeReal,

    /// The operation requires the input to be in time domain.
    InputMustBeInTimeDomain,

    /// The operation requires the input to be in frequency domain.
    InputMustBeInFrequencyDomain,

    /// The arguments have an invalid length to perform the operation. The
    /// operations documentation should have more information about the requirements.
    /// Please open a defect if this isn't the case.
    InvalidArgumentLength,

    /// The operations is only valid if the data input contains half of a symmetric spectrum.
    /// The symmetry definition follows soon however more important is that the element at 0 Hz
    /// which happens to be the first vector element must be real. The error message is raised if this
    /// is violated, the rest of the definition is only listed here for completeness snce it can't
    /// be checked.
    /// The required symmetry for a vector is that for every point `vector[x].conj() == vector[-x]`(pseudocode)
    /// where `x` is the x-axis position relative to 0 Hz and `conj` is the complex conjugate.
    InputMustBeConjSymmetric,

    /// `self.points()` must be an odd number.
    InputMustHaveAnOddLength,

    /// The function passed as argument must be symmetric
    ArgumentFunctionMustBeSymmetric,

    /// The number of arguments passed into a combined operation methods doesn't match
    /// with the number of arguments specified previously via the `add_op` methods.
    InvalidNumberOfArgumentsForCombinedOp,

    /// The operation isn't specified for an empty vector.
    InputMustNotBeEmpty,

    /// Given input must have an even length.
    InputMustHaveAnEvenLength,

    /// The arguments would require that the type allocates larger memory. But the
    /// type can't do that.
    TypeCanNotResize,
}
