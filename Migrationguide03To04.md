# Migration Guide v0.3 to v0.4
The API as defined in version 0.3 proofed to be not flexible enough and would prevent new features from being added to this crate. In particular there was no way to operate on slices of a vector since all implementations were based on `std::vec::Vec<T>`. Thus a lot of breaking changes to the API were unavoidable. This open up the opportunity to rethink the current and the result was a quite different approach.

Apologies for the inconvenience caused by the API change. Hopefully users will agree that the new approach is more Rust idiomatic. The next sections give a brief overview about what has changed to make the migration easier.

Pros:

- Support for new storage types such as `[f32]`, `[f64]`, `&[f32]`, `&[f64]`, `&mut [f32]`, `&mut [f64]`, `Box<[f32]>` and `Box[f64]`. The main advantage is that it's no possible to get a slice from a vector and still runs some operations on it.
- Vector types don't have an internal buffer anymore. This makes them more lightweight. It also gives more explicit control over the memory management and allows to use this lib in environments with tight memory limits.
- The new traits can be more easily reused and will likely enable matrix types.

Cons:

- The added flexibility makes the type definitions in the documentation more complex.
- While the explicit buffering gives more control over allocations it at the same time adds more clutter to the API.

## Conversions instead of constructors
DSP vectors are now created using `to_real_time_vec`, `to_real_freq_vec`, `to_complex_time_vec`, `to_complex_freq_vec` and `to_gen_dsp_vec` on `std::vec::Vec<T>` or on slices.

## Favoring statically checked vector types
DSP vectors created with `to_real_time_vec`, `to_real_freq_vec`, `to_complex_time_vec` and `to_complex_freq_vec` prevent certain errors. E.g. they prevent that a method which is only valid on complex data is called on a vector holding real data. With version v0.4 these constructor functions are highly recommended. `to_gen_dsp_vec` only remains mainly for interop scenarios, if they still need to be used then in certain error cases the lib won't return an error but sets `dsp.len()` to `0` instead to indicate an error. This certainly is a poor error handling mechanism by itself, but this enables much easier interfaces and makes vector types created with `to_real_time_vec`, `to_real_freq_vec`, `to_complex_time_vec` and `to_complex_freq_vec` easier to use.

## Signature changes and more fine grained traits
The different methods on vector types are now implemented by a quite a few traits. It might be much easier to therefore import with `use basic_dsp::*`. In v0.3 all methods transformed `self` which is actually a pattern which is more idiomatic to other languages. For Rust it seems to be more straightforward to use a `&self` or `&mut self` reference for methods and so where possible this is now what methods do.

## Buffering
Methods which require a buffer end with the suffix `_b`. There is usually another method available without the suffix which doesn't require a buffer. For some storage types (e.g. `&[T]`) a buffer is necessary since otherwise the operation will fail if it would result in `self.len() > self.alloc_len()`, meaning that it would require more memory than the storage type can provide. In such cases the buffer is used to allocate a new storage. In other cases the buffer might allow for a different and faster implementation of the same operation.

In order to get the same performance as in v0.3 it should be considered to use the buffered methods where possible. Right now there is one buffer type available (called `SingleBuffer` since it buffers only one Vec/slice at a time), but this type should be sufficient for most tasks.
