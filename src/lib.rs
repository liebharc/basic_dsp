extern crate basic_dsp_vector;
extern crate basic_dsp_matrix;
#[cfg(feature = "basic_dsp_interop")]
extern crate basic_dsp_interop;

pub use basic_dsp_vector::*;

#[cfg(feature = "basic_dsp_matrix")]
pub mod matrix {
    pub use basic_dsp_matrix::*;
}

#[cfg(feature = "basic_dsp_interop")]
pub mod interop {
    pub use basic_dsp_interop::*;
}
