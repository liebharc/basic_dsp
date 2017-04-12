
#[cfg(feature="std")]
mod threading;
#[cfg(feature="std")]
pub use self::threading::*;

#[cfg(not(feature="std"))]
mod no_threading;
#[cfg(not(feature="std"))]
pub use self::no_threading::*;

/// Indicates how complex an operation is and determines how many cores
/// will be used since operations with smaller complexity are memory bus bound
/// and not CPU bound
#[derive(Copy)]
#[derive(Clone)]
#[derive(PartialEq)]
#[derive(Debug)]
pub enum Complexity {
    Small,
    Medium,
    Large,
}
