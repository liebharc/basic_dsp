#[cfg(feature="std")]
mod dynamic;
#[cfg(feature="std")]
pub use self::dynamic::*;
#[cfg(not(feature="std"))]
mod fixed_size;
#[cfg(not(feature="std"))]
pub use self::fixed_size::*;
