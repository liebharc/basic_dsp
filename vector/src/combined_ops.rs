//! This module allows to combine certain operations into one operation. Since one
//! many machines the speed of many DSP operations is limited by the memory bus speed
//! this approach may result in better register and cache usage and thus decrease
//! the pressure on the memory bus. As with all performance hints remember 
//! rule number 1: Benchmark your code. This is especially true at this very early 
//! state of the library.
//! 
//! With this approach we change how we operate on vectors. If you perform
//! `M` operations on a vector with the length `N` you iterate wit hall other methods like this:
//!
//! ```no_run
//! // pseudocode:
//! // for m in M:
//! //  for n in N:
//! //    execute m on n
//! ```
//!
//! with this method the pattern is changed slightly:
//!
//! ```no_run
//! // pseudocode:
//! // for n in N:
//! //  for m in M:
//! //    execute m on n
//! ```
//!
//! Both variants have the same complexity however the second one is beneficial since we
//! have increased locality this way. This should help us by making better use of registers and 
//! CPU caches.
//!
//! Only operations can be combined where the result of every element in the vector
//! is independent from any other element in the vector.
//!
//! # Examples
//!
//!```
//! use std::f32::consts::PI;
//! use basic_dsp_vector::*;
//! use basic_dsp_vector::combined_ops::*;
//! # fn close(left: &[f32], right: &[f32]) {
//! #   assert_eq!(left.len(), right.len());
//! #   for i in 0..left.len() {
//! #       assert!((left[i] - right[i]) < 1e-2);
//! #   }
//! # }
//! let a = RealTimeVector32::from_array(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
//! let b = RealTimeVector32::from_constant(0.0, 8);
//! let ops = multi_ops2(a, b);
//! let ops = ops.add_ops(|a, b| {
//!     let a = a.scale(2.0 * PI);
//!     let b = b.clone_from(&a);
//!     let a = a.sin();
//!     let b = b.cos();
//!     let a = a.multiply_vector(&b).abs();
//!     let a_db = a.log(10.0).scale(10.0);
//!     (a_db, b)
//! });
//! let (a, b) = ops.get().expect("Ignoring error handling in examples");
//! close(&[0.80902, 0.30902, -0.30902, -0.80902, -1.00000, -0.80902, -0.30902, 0.30902], b.data());
//! close(&[-3.2282, -5.3181, -5.3181, -3.2282, -159.1199, -3.2282, -5.3181, -5.3181], a.data());
//!```
//!

// In this module we just export types from other modules with a deeper nesting
// level. This makes the API as presented to the user flatter.
pub use vector_types::multi_ops::*;  
pub use vector_types::operations_enum::Operation;