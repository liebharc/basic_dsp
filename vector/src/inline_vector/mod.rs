//! This is the data type `basic_dsp` uses internally.
//! Typically the `std::Vec` is a good choice, however 
//! for really small vectors of small data types it may be 
//! benificial to avoid heap allocation and keep the items on the
//! stack (refer to `http://stackoverflow.com/questions/27859822/alloca-variable-length-arrays-in-rust`
//! for more details). 
//! 
//! If the lib is compiled without the `std` feature then `std::Vec` isn't available
//! and only stack allocation will be used.

#[cfg(feature="std")]
mod dynamic;
#[cfg(feature="std")]
pub use self::dynamic::*;
#[cfg(not(feature="std"))]
mod fixed_size;
#[cfg(not(feature="std"))]
pub use self::fixed_size::*;
