initSidebarItems({"enum":[["DataVectorDomain","The domain of a data vector"],["ErrorReason","Enumeration of all error reasons"],["EvenOdd","Argument for some operations to determine if the result should have an even or odd number of points."],["Operation","An alternative way to define operations on a vector. Warning: Highly unstable and not even fully implemented right now."]],"mod":[["complex_impl",""],["convolution_impl",""],["correlation_impl",""],["definitions",""],["general_impl",""],["interpolation_impl",""],["real_impl",""],["time_freq_impl",""]],"struct":[["ComplexFreqVector","A 1xN (one times N elements) or Nx1 data vector as used for most digital signal processing (DSP) operations. All data vector operations consume the vector they operate on and return a new vector. A consumed vector must not be accessed again."],["ComplexTimeVector","A 1xN (one times N elements) or Nx1 data vector as used for most digital signal processing (DSP) operations. All data vector operations consume the vector they operate on and return a new vector. A consumed vector must not be accessed again."],["GenericDataVector","A 1xN (one times N elements) or Nx1 data vector as used for most digital signal processing (DSP) operations. All data vector operations consume the vector they operate on and return a new vector. A consumed vector must not be accessed again."],["RealFreqVector","A 1xN (one times N elements) or Nx1 data vector as used for most digital signal processing (DSP) operations. All data vector operations consume the vector they operate on and return a new vector. A consumed vector must not be accessed again."],["RealTimeVector","A 1xN (one times N elements) or Nx1 data vector as used for most digital signal processing (DSP) operations. All data vector operations consume the vector they operate on and return a new vector. A consumed vector must not be accessed again."],["Statistics","Statistics about the data in a vector"]],"trait":[["ComplexVectorOperations","Defines all operations which are valid on `DataVectors` containing complex data. # Failures All operations in this trait fail with `VectorMustBeComplex` if the vector isn't in the complex number space."],["Convolution","Provides a convolution operation for data vectors."],["CrossCorrelation","Cross-correlation of data vectors. See also https://en.wikipedia.org/wiki/Cross-correlation # Unstable This functionality has been recently added in order to find out if the definitions are consistent. However the actual implementation is lacking tests. # Failures VecResult may report the following `ErrorReason` members:"],["DataVector","DataVector gives access to the basic properties of all data vectors"],["FrequencyDomainOperations","Defines all operations which are valid on `DataVectors` containing frequency domain data. # Failures All operations in this trait fail with `VectorMustBeInFrquencyDomain` or `VectorMustBeComplex`  if the vector isn't in frequency domain and complex number space. # Unstable A lot of details about FFTs and IFFTs aren't working out yet. Changes and updates are expected."],["GenericVectorOperations","Defines all operations which are valid on all `DataVectors`."],["Interpolation","Provides a interpolation operation for data vectors. # Unstable This functionality has been recently added in order to find out if the definitions are consistent. However the actual implementation is lacking tests."],["Offset","An operation which adds a constant to each vector element"],["RealVectorOperations","Defines all operations which are valid on `DataVectors` containing real data. # Failures All operations in this trait fail with `VectorMustBeReal` if the vector isn't in the real number space."],["RededicateVector","This trait allows to change a vector type. The operations will convert a vector to a different type and set `self.len()` to zero. However `self.allocated_len()` will remain unchanged. The use case for this is to allow to reuse the memory of a vector for different operations."],["Scale","An operation which multiplies each vector element with a constant"],["SymmetricFrequencyDomainOperations","Defines all operations which are valid on `DataVectors` containing frequency domain data. # Failures All operations in this trait fail with `VectorMustBeInFrquencyDomain` if the vector isn't in frequency domain. # Unstable A lot of details about FFTs and IFFTs aren't working out yet. Changes and updates are expected."],["SymmetricTimeDomainOperations","Defines all operations which are valid on `DataVectors` containing real time domain data. # Failures All operations in this trait fail with `VectorMustBeInTimeDomain` if the vector isn't in time domain. # Unstable A lot of details about FFTs and IFFTs aren't working out yet. Changes and updates are expected."],["TimeDomainOperations","Defines all operations which are valid on `DataVectors` containing time domain data. # Failures All operations in this trait fail with `VectorMustBeInTimeDomain` if the vector isn't in time domain. # Unstable A lot of details about FFTs and IFFTs aren't working out yet. Changes and updates are expected."]],"type":[["ComplexFreqVector32","Specialization of a vector for a certain data type."],["ComplexFreqVector64","Specialization of a vector for a certain data type."],["ComplexTimeVector32","Specialization of a vector for a certain data type."],["ComplexTimeVector64","Specialization of a vector for a certain data type."],["DataVector32","Specialization of a vector for a certain data type."],["DataVector64","Specialization of a vector for a certain data type."],["RealFreqVector32","Specialization of a vector for a certain data type."],["RealFreqVector64","Specialization of a vector for a certain data type."],["RealTimeVector32","Specialization of a vector for a certain data type."],["RealTimeVector64","Specialization of a vector for a certain data type."],["VecResult","Result contains on success the vector. On failure it contains an error reason and an vector with invalid data which still can be used in order to avoid memory allocation."],["VoidResult","Void/nothing in case of success or a reason in case of an error."]]});