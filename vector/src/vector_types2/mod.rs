use RealNumber;

mod requirements;
pub use self::requirements::*;
mod to_from_vec_conversions;
pub use self::to_from_vec_conversions::*;
mod vec_trait_and_indexers;
pub use self::vec_trait_and_indexers::*;
use vector_types::{
    DataVecDomain};
use multicore_support::MultiCoreSettings;

macro_rules! define_vector_struct {
    ($($name:ident),*) => {
        $(
            #[derive(Debug)]
            /// A 1xN (one times N elements) or Nx1 data vector as used for most digital signal processing (DSP) operations.
            /// All data vector operations consume the vector they operate on and return a new vector. A consumed vector
            /// must not be accessed again.
            ///
            /// Vectors come in different flavors:
            ///
            /// 1. Time or Frequency domain
            /// 2. Real or Complex numbers
            /// 3. 32bit or 64bit floating point numbers
            ///
            /// The first two flavors define meta information about the vector and provide compile time information what
            /// operations are available with the given vector and how this will transform the vector. This makes sure that
            /// some invalid operations are already discovered at compile time. In case that this isn't desired or the information
            /// about the vector isn't known at compile time there are the generic [`DataVec32`](type.DataVec32.html) and [`DataVec64`](type.DataVec64.html) vectors
            /// available.
            ///
            /// 32bit and 64bit flavors trade performance and memory consumption against accuracy. 32bit vectors are roughly
            /// two times faster than 64bit vectors for most operations. But remember that you should benchmark first
            /// before you give away accuracy for performance unless however you are sure that 32bit accuracy is certainly good
            /// enough.
            pub struct $name<S, T>
                where S: ToSlice<T>,
                      T: RealNumber {
                data: S,
                delta: T,
                domain: DataVecDomain,
                is_complex: bool,
                valid_len: usize,
                multicore_settings: MultiCoreSettings
            }
        )*
    }
}
define_vector_struct!(
    GenDspVec,
    RealTimeVec, RealFreqVec,
    ComplexTimeVec, ComplexFreqVec);
