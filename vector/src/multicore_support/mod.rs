//! Abstraction of threading. If the vector becomes large then the lib will start
//! to spawn worker threads to perform the calculation. Since spawning threads
//! for tiny tasks like isn't very benificial users might want to implement their
//! algorithm in a multi-threaded design and deactivate threading this "auto-threading"
//! behaviour. The `MultiCoreSettings` mechanism allows to do that on a fine grained level.
//!
//! If compiled without the `std` feature then only the `no_threading` implementation
//! will be used which just implements single threaded loops. We then rely on the Rust
//! compiler to remove the overhead of this implementation - and it seems to do that
//! very well.

#[cfg(feature = "std")]
mod threading;
#[cfg(feature = "std")]
pub use self::threading::*;

#[cfg(not(feature = "std"))]
mod no_threading;
#[cfg(not(feature = "std"))]
pub use self::no_threading::*;

/// Indicates how complex an operation is and determines how many cores
/// will be used since operations with smaller complexity are memory bus bound
/// and not CPU bound
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Complexity {
    Small,
    Medium,
    Large,
}
