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
//! ```
//! // pseudocode:
//! // for m in M:
//! //  for n in N:
//! //    execute m on n
//! ```
//!
//! with this method the pattern is changed slighly:
//!
//! ```
//! // pseudocode:
//! // for n in N:
//! //  for m in M:
//! //    execute m on n
//! ```
//!
//! Both variants have the same complexity however the second one is benificial since we
//! have increased locality this way. This should help us by making better use of registers and 
//! CPU caches.
//!
//! Only operations can be combined where the result of every element in the vector
//! is independent from any other element in the vector.

// In this module we just export types from other modules with a deeper nesting
// level. This makes the API as presented to the user flatter.
pub use vector_types::{
    ToIdentifier,
    Identifier,
    GenericDataIdentifier,
    RealTimeIdentifier,
    ComplexTimeIdentifier,
    RealFreqIdentifier,
    ComplexFreqIdentifier,
    Argument,
    PreparedOperation1,
    PreparedOperation2,
    prepare2,
    MultiOperation2,
    ComplexIdentifier,
    Operation,
    multi_ops2};  