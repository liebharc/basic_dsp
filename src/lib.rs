#[cfg(feature = "interop")]
extern crate basic_dsp_interop;
#[cfg(feature = "matrix")]
extern crate basic_dsp_matrix;
extern crate basic_dsp_vector;

pub use basic_dsp_vector::*;

#[cfg(feature = "matrix")]
pub mod matrix {
    pub use basic_dsp_matrix::*;
}

#[cfg(feature = "interop")]
pub mod interop {
    pub use basic_dsp_interop::*;
}
