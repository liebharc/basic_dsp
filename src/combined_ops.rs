//! This module allows to combine certain operations into one operation. Since one
//! many machines the speed of many DSP operations is limited by the memory bus speed
//! this approach may result in better register and cache usage and thus decrease
//! the pressure on the memory bus. As with all performance hints remember 
//! rule number 1: Benchmark your code. This is especially true at this very early 
//! state of the library.
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
    Operation,
    multi_ops2};  