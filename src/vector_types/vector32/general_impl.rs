use multicore_support::{Chunk, Complexity};
use super::super::definitions::{
	DataVector,
    DataVectorDomain,
	GenericVectorOperations,
    VecResult,
    ErrorReason};
use super::DataVector32;
use simd::f32x4;
use num::complex::Complex32;
use complex_extensions::ComplexExtensions;
use simd_extensions::SimdExtensions;
use super::super::super::multicore_support::MultiCoreSettings;

impl DataVector32 {
    /// Creates a new generic data vector from the given arguments.
    pub fn new(is_complex: bool, domain: DataVectorDomain, init_value: f32, length: usize, delta: f32) -> DataVector32 {
        DataVector32 
        { 
            data: vec![init_value; length],
            temp: vec![0.0; length],
            delta: delta,
            domain: domain,
            is_complex: is_complex,
            valid_len: length,
            multicore_settings: MultiCoreSettings::new()
        }
    }
}

#[inline]
impl GenericVectorOperations for DataVector32
{
	fn add_vector(mut self, summand: &Self) -> VecResult<Self>
	{
		{
			let len = self.len();
            reject_if!(self, len != summand.len(), ErrorReason::VectorsMustHaveTheSameSize);
			
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			let other = &summand.data;
			Chunk::execute_original_to_target(Complexity::Small, &other, vectorization_length, 4, &mut array, vectorization_length, 4,  |original, range, target| {
                let mut i = 0;
                let mut j = range.start;
                while i < target.len()
                { 
                    let vector1 = f32x4::load(original, j);
                    let vector2 = f32x4::load(target, i);
                    let result = vector1 + vector2;
                    result.store(target, i);
                    i += 4;
                    j += 4;
                }
            });
			let mut i = vectorization_length;
			while i < data_length
			{
				array[i] = array[i] + other[i];
				i += 1;
			}
		}
		
		Ok(self)
	}
	
	fn subtract_vector(mut self, subtrahend: &Self) -> VecResult<Self>
	{
		{
			let len = self.len();
			reject_if!(self, len != subtrahend.len(), ErrorReason::VectorsMustHaveTheSameSize);
				
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			let other = &subtrahend.data;
			Chunk::execute_original_to_target(Complexity::Small, &other, vectorization_length, 4, &mut array, vectorization_length, 4,   |original, range, target| {
                let mut i = 0;
                let mut j = range.start;
                while i < target.len()
                { 
                    let vector1 = f32x4::load(original, j);
                    let vector2 = f32x4::load(target, i);
                    let result = vector2 - vector1;
                    result.store(target, i);
                    i += 4;
                    j += 4;
                }
            });
			let mut i = vectorization_length;
			while i < data_length
			{
				array[i] = array[i] - other[i];
				i += 1;
			}
		}
		
		Ok(self)
	}
	
	fn multiply_vector(self, factor: &Self) -> VecResult<Self>
	{
		let len = self.len();
		reject_if!(self, len != factor.len(), ErrorReason::VectorsMustHaveTheSameSize);
		
		if self.is_complex
		{
			Ok(self.multiply_vector_complex(factor))
		}
		else
		{
			Ok(self.multiply_vector_real(factor))
		}
	}
	
	fn divide_vector(self, divisor: &Self) -> VecResult<Self>
	{
		let len = self.len();
		reject_if!(self, len != divisor.len(), ErrorReason::VectorsMustHaveTheSameSize);
		
		if self.is_complex
		{
			Ok(self.divide_vector_complex(divisor))
		}
		else
		{
			Ok(self.divide_vector_real(divisor))
		}
	}
	
	fn zero_pad(mut self, points: usize) -> VecResult<Self>
	{
		{
			let len_before = self.len();
			let len = if self.is_complex { 2 * points } else { points };
			self.reallocate(len);
			let array = &mut self.data;
			for i in len_before..len
			{
				array[i] = 0.0;
			}
		}
		
		Ok(self)
	}
	
	fn zero_interleave(self) -> VecResult<Self>
	{
		if self.is_complex
		{
			Ok(self.zero_interleave_complex())
		}
		else
		{
			Ok(self.zero_interleave_real())
		}
	}
	
	fn diff(mut self) -> VecResult<Self>
	{
		{
			let data_length = self.len();
			let mut target = &mut self.temp;
			let org = &self.data;
			if self.is_complex {
				self.valid_len -= 2;
				Chunk::execute_original_to_target(Complexity::Small, &org, data_length, 2, &mut target, data_length, 2, |original, range, target| {
                    let mut i = 0;
                    let mut j = range.start;
                    let mut len = target.len();
                    if range.end == original.len() - 1
                    {
                        len -= 2;
                    }
                    
                    while i < len
                    { 
                        target[i] = original[j + 2] - original[i];
                        i += 1;
                        j += 1;
                    }
                });
			}
			else {
				self.valid_len -= 1;
				Chunk::execute_original_to_target(Complexity::Small, &org, data_length, 1, &mut target, data_length, 1, |original, range, target| {
                    let mut i = 0;
                    let mut j = range.start;
                    let mut len = target.len();
                    if range.end >= original.len() - 1
                    {
                        len -= 1;
                    }
                        
                    while i < len
                    { 
                        target[i] = original[j + 1] - original[j];
                        i += 1;
                        j += 1;
                    }
                });
			}
		}
		
		Ok(self.swap_data_temp())
	}
	
	fn diff_with_start(mut self) -> VecResult<Self>
	{
		{
			let data_length = self.len();
			let mut target = &mut self.temp;
			let org = &self.data;
			if self.is_complex {
				Chunk::execute_original_to_target(Complexity::Small, &org, data_length, 2, &mut target, data_length, 2, |original, range, target| {
                    let mut i = 0;
                    let mut j = range.start;
                    if j == 0 {
                        i = 2;
                        j = 2;
                        target[0] = original[0];
                        target[1] = original[1];
                    }
                    
                    while i < target.len()
                    { 
                        target[i] = original[j] - original[j - 2];
                        i += 1;
                        j += 1;
                    }
                });
			}
			else {
				Chunk::execute_original_to_target(Complexity::Small, &org, data_length, 1, &mut target, data_length, 1, |original, range, target| {
                    let mut i = 0;
                    let mut j = range.start;
                    if j == 0 {
                        i = 1;
                        j = 1;
                        target[0] = original[0];
                    }
                    
                    while i < target.len()
                    { 
                        target[i] = original[j] - original[j - 1];
                        i += 1;
                        j += 1;
                    }
                });
			}
		}
		
		Ok(self.swap_data_temp())
	}
	
	fn cum_sum(mut self) -> VecResult<Self>
	{
		{
			let data_length = self.len();
			let mut data = &mut self.data;
			let mut i = 0;
			let mut j = 1;
			if self.is_complex {
				j = 2;
			}
			
			while j < data_length {
				data[j] = data[j] + data[i];
				i += 1;
				j += 1;
			}
		}
		Ok(self)
	}
    
    fn sqrt(self) -> VecResult<Self>
	{
        if self.is_complex() {
          DataVector32::complex_sqrt(self)  
        }
        else {
		  DataVector32::real_sqrt(self)
        }
	}
	
	fn square(self) -> VecResult<Self>
	{
        if self.is_complex() {
          DataVector32::complex_square(self)  
        }
        else {
		  DataVector32::real_square(self)
        }
	}
	
	fn root(self, degree: Self::E) -> VecResult<Self>
	{
        if self.is_complex() {
          DataVector32::complex_root(self, degree)  
        }
        else {
		  DataVector32::real_root(self, degree)
        }
	}
	
	fn power(self, exponent: Self::E) -> VecResult<Self>
	{
        if self.is_complex() {
          DataVector32::complex_power(self, exponent)  
        }
        else {
		  DataVector32::real_power(self, exponent)
        }
	}
	
	fn logn(self) -> VecResult<Self>
	{
        if self.is_complex() {
          DataVector32::complex_logn(self)  
        }
        else {
		  DataVector32::real_logn(self)
        }
	}
	
	fn expn(self) -> VecResult<Self>
	{
        if self.is_complex() {
          DataVector32::complex_expn(self)  
        }
        else {
		  DataVector32::real_expn(self)
        }
	}

	fn log_base(self, base: Self::E) -> VecResult<Self>
	{
        if self.is_complex() {
          DataVector32::complex_log_base(self, base)  
        }
        else {
		  DataVector32::real_log_base(self, base)
        }
	}
	
	fn exp_base(self, base: Self::E) -> VecResult<Self>
	{
        if self.is_complex() {
          DataVector32::complex_exp_base(self, base)  
        }
        else {
		  DataVector32::real_exp_base(self, base)
        }
	}
    
    fn sin(self) -> VecResult<Self>
    {
        if self.is_complex() {
          DataVector32::complex_sin(self)  
        }
        else {
		  DataVector32::real_sin(self)
        }
    }
    
    fn cos(self) -> VecResult<Self>
    {
        if self.is_complex() {
          DataVector32::complex_cos(self)  
        }
        else {
		  DataVector32::real_cos(self)
        }
    }
}

impl DataVector32 {
    fn multiply_vector_complex(mut self, factor: &DataVector32) -> DataVector32
	{
		{
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			let other = &factor.data;
			Chunk::execute_original_to_target(Complexity::Small, &other, vectorization_length, 4, &mut array, vectorization_length, 4, |original, range, target| {
                let mut i = 0;
                let mut j = range.start;
                while i < target.len()
                { 
                    let vector1 = f32x4::load(original, j);
                    let vector2 = f32x4::load(target, i);
                    let result = vector2.mul_complex(vector1);
                    result.store(target, i);
                    i += 4;
                    j += 4;
                }
            });
			let mut i = vectorization_length;
			while i < data_length
			{
				let complex1 = Complex32::new(array[i], array[i + 1]);
				let complex2 = Complex32::new(other[i], other[i + 1]);
				let result = complex1 * complex2;
				array[i] = result.re;
				array[i + 1] = result.im;
				i += 2;
			}
		}
		self
	}
	
	fn multiply_vector_real(mut self, factor: &DataVector32) -> DataVector32
	{
		{
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			let other = &factor.data;
			Chunk::execute_original_to_target(Complexity::Small, &other, vectorization_length, 4, &mut array, vectorization_length, 4, |original, range, target| {
                let mut i = 0;
                let mut j = range.start;
                while i < target.len()
                { 
                    let vector1 = f32x4::load(original, j);
                    let vector2 = f32x4::load(target, i);
                    let result = vector2 * vector1;
                    result.store(target, i);
                    i += 4;
                    j += 4;
                }        
            });
			let mut i = vectorization_length;
			while i < data_length
			{
				array[i] = array[i] * other[i];
				i += 1;
			}
		}
		self
	}
	
	fn divide_vector_complex(mut self, divisor: &DataVector32) -> DataVector32
	{
		{
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			let other = &divisor.data;
			Chunk::execute_original_to_target(Complexity::Small, &other, vectorization_length, 4, &mut array, vectorization_length, 4,  |original, range, target| {
                let mut i = 0;
                let mut j = range.start;
                while i < target.len()
                { 
                    let vector1 = f32x4::load(original, j);
                    let vector2 = f32x4::load(target, i);
                    let result = vector2.div_complex(vector1);
                    result.store(target, i);
                    i += 4;
                    j += 4;
                }
            });
			let mut i = vectorization_length;
			while i < data_length
			{
				let complex1 = Complex32::new(array[i], array[i + 1]);
				let complex2 = Complex32::new(other[i], other[i + 1]);
				let result = complex1 / complex2;
				array[i] = result.re;
				array[i + 1] = result.im;
				i += 2;
			}
		}
		self
	}
	
	fn divide_vector_real(mut self, divisor: &DataVector32) -> DataVector32
	{
		{
			let data_length = self.len();
			let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			let mut array = &mut self.data;
			let other = &divisor.data;
			Chunk::execute_original_to_target(Complexity::Small, &other, vectorization_length, 4, &mut array, vectorization_length, 4,  |original, range, target| {
                let mut i = 0;
                let mut j = range.start;
                while i < target.len()
                { 
                    let vector1 = f32x4::load(original, j);
                    let vector2 = f32x4::load(target, i);
                    let result = vector2 / vector1;
                    result.store(target, i);
                    i += 4;
                    j += 4;
                }
            });
			let mut i = vectorization_length;
			while i < data_length
			{
				array[i] = array[i] / other[i];
				i += 1;
			}
		}
		self
	}
    
    fn zero_interleave_complex(mut self) -> Self
	{
		{
			let new_len = 2 * self.len();
			self.reallocate(new_len);
			let data_length = new_len;
			let mut target = &mut self.temp;
			let source = &self.data;
			Chunk::execute_original_to_target(Complexity::Small, &source, data_length, 4, &mut target, data_length, 4,  |original, range, target| {
                let mut i = 0;
                let mut j = range.start;
                while i < target.len() / 2 {
                    if i % 2 == 0
                    {
                        target[2 * i] = original[j];
                        target[2 * i + 1] = original[j + 1];
                        j += 2;
                    }
                    else
                    {
                        target[2 * i] = 0.0;
                        target[2 * i + 1] = 0.0;
                    }
                    
                    i += 1;
                }
            });
		}
		self.swap_data_temp()
	}
	
	fn zero_interleave_real(mut self) -> Self
	{
		{
			let new_len = 2 * self.len();
			self.reallocate(new_len);
			let data_length = new_len;
			let mut target = &mut self.temp;
			let source = &self.data;
			Chunk::execute_original_to_target(Complexity::Small, &source, data_length, 4, &mut target, data_length, 2,  |original, range, target| {
                let mut i = 0;
                let mut j = range.start;
                while i < target.len() {
                    if i % 2 == 0
                    {
                        target[i] = original[j];
                        j += 1;
                    }
                    else
                    {
                        target[i] = 0.0;
                    }
                    
                    i += 1;
                }
            });
		}
		self.swap_data_temp()
	}
    
    fn real_sqrt(mut self) -> VecResult<Self>
	{
		{
            let data_length = self.len();
			let mut array = &mut self.data;
            let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			Chunk::execute_partial(Complexity::Small, &mut array, vectorization_length, 4, |array| {
                let mut i = 0;
                while i < array.len()
                {
                    let vector = f32x4::load(array, i);
                    let result = vector.sqrt();
                    result.store(array, i);
                    i += 4;
                }
            });
            for i in vectorization_length..data_length
			{
				array[i] = array[i].sqrt();
			}
		}
		Ok(self)
	}
    
    fn complex_sqrt(mut self) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial(Complexity::Medium, &mut array, length, 2, |array| {
                let mut i = 0;
                while i < array.len()
                {
                    let complex = Complex32::new(array[i], array[i + 1]);
                    let result = complex.sqrt();
                    array[i] = result.re;
                    array[i + 1] = result.im;
                    i += 2;
                }
            });
		}
		Ok(self)
	}
	
	fn real_square(mut self) -> VecResult<Self>
	{
		{
			let data_length = self.len();
			let mut array = &mut self.data;
            let scalar_length = data_length % 4;
			let vectorization_length = data_length - scalar_length;
			Chunk::execute_partial(Complexity::Small, &mut array, vectorization_length, 4, |array| {
                let mut i = 0;
                while i < array.len()
                {
                    let a = f32x4::load(array, i);
                    let result = a * a;
                    result.store(array, i);
                    i += 4;
                }
            });
            
            for i in vectorization_length..data_length
			{
				array[i] = array[i] * array[i];
			}
		}
		Ok(self)
	}
    
    fn complex_square(mut self) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial(Complexity::Medium, &mut array, length, 2, |array| {
                let mut i = 0;
                while i < array.len()
                {
                    let complex = Complex32::new(array[i], array[i + 1]);
                    let result = complex * complex;
                    array[i] = result.re;
                    array[i + 1] = result.im;
                    i += 2;
                }
            });
		}
		Ok(self)
	}
	
	fn real_root(mut self, degree: f32) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial_with_arguments(Complexity::Medium, &mut array, length, 1, degree, |array, base| {
                let base = 1.0 / base;
                let mut i = 0;
                while i < array.len()
                {
                    array[i] = array[i].powf(base);
                    i += 1;
                }
            });
		}
		Ok(self)
	}
    
    fn complex_root(mut self, base: f32) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial_with_arguments(Complexity::Medium, &mut array, length, 2, base, |array, base| {
                let base = 1.0 / base;
                let mut i = 0;
                while i < array.len()
                {
                    let complex = Complex32::new(array[i], array[i + 1]);
                    let result = complex.powf(base);
                    array[i] = result.re;
                    array[i + 1] = result.im;
                    i += 2;
                }
            });
		}
		Ok(self)
	}
	
	fn real_power(mut self, exponent: f32) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial_with_arguments(Complexity::Medium, &mut array, length, 1, exponent, |array, base| {
                let mut i = 0;
                while i < array.len()
                {
                    array[i] = array[i].powf(base);
                    i += 1;
                }
            });
		}
		Ok(self)
	}
    
    fn complex_power(mut self, base: f32) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial_with_arguments(Complexity::Medium, &mut array, length, 2, base, |array, base| {
                let mut i = 0;
                while i < array.len()
                {
                    let complex = Complex32::new(array[i], array[i + 1]);
                    let result = complex.powf(base);
                    array[i] = result.re;
                    array[i + 1] = result.im;
                    i += 2;
                }
            });
		}
		Ok(self)
	}
	
	fn real_logn(mut self) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial(Complexity::Medium, &mut array, length, 1, |array| {
                let mut i = 0;
                while i < array.len()
                {
                    array[i] = array[i].ln();
                    i += 1;
                }
            });
		}
		Ok(self)
	}
    
    fn complex_logn(mut self) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial(Complexity::Medium, &mut array, length, 2, |array| {
                let mut i = 0;
                while i < array.len()
                {
                    let complex = Complex32::new(array[i], array[i + 1]);
                    let result = complex.ln();
                    array[i] = result.re;
                    array[i + 1] = result.im;
                    i += 2;
                }
            });
		}
		Ok(self)
	}
	
	fn real_expn(mut self) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial(Complexity::Medium, &mut array, length, 1, |array| {
                let mut i = 0;
                while i < array.len()
                {
                    array[i] = array[i].exp();
                    i += 1;
                }
            });
		}
		Ok(self)
	}
    
    fn complex_expn(mut self) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial(Complexity::Medium, &mut array, length, 2, |array| {
                let mut i = 0;
                while i < array.len()
                {
                    let complex = Complex32::new(array[i], array[i + 1]);
                    let result = complex.expn();
                    array[i] = result.re;
                    array[i + 1] = result.im;
                    i += 2;
                }
            });
		}
		Ok(self)
	}

	fn real_log_base(mut self, base: f32) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial_with_arguments(Complexity::Medium, &mut array, length, 1, base, |array, base| {
                let mut i = 0;
                while i < array.len()
                {
                    array[i] = array[i].log(base);
                    i += 1;
                }
            });
		}
		Ok(self)
	}
    
    fn complex_log_base(mut self, base: f32) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial_with_arguments(Complexity::Medium, &mut array, length, 2, base, |array, base| {
                let mut i = 0;
                while i < array.len()
                {
                    let complex = Complex32::new(array[i], array[i + 1]);
                    let result = complex.log(base);
                    array[i] = result.re;
                    array[i + 1] = result.im;
                    i += 2;
                }
            });
		}
		Ok(self)
	}
	
	fn real_exp_base(mut self, base: f32) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial_with_arguments(Complexity::Medium, &mut array, length, 1, base, |array, base| {
                let mut i = 0;
                while i < array.len()
                {
                    array[i] = base.powf(array[i]);
                    i += 1;
                }
            });
		}
		Ok(self)
	}
    
    fn complex_exp_base(mut self, base: f32) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial_with_arguments(Complexity::Medium, &mut array, length, 2, base, |array, base| {
                let mut i = 0;
                while i < array.len()
                {
                    let complex = Complex32::new(array[i], array[i + 1]);
                    let result = complex.exp(base);
                    array[i] = result.re;
                    array[i + 1] = result.im;
                    i += 2;
                }
            });
		}
		Ok(self)
	}
    
    fn real_sin(mut self) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial(Complexity::Medium, &mut array, length, 1, |array| {
                let mut i = 0;
                while i < array.len()
                {
                    array[i] = array[i].sin();
                    i += 1;
                }
            });
		}
		Ok(self)
	}
    
    fn complex_sin(mut self) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial(Complexity::Medium, &mut array, length, 2, |array| {
                let mut i = 0;
                while i < array.len()
                {
                    let complex = Complex32::new(array[i], array[i + 1]);
                    let result = complex.sin();
                    array[i] = result.re;
                    array[i + 1] = result.im;
                    i += 2;
                }
            });
		}
		Ok(self)
	}
    
    fn real_cos(mut self) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial(Complexity::Medium, &mut array, length, 1, |array| {
                let mut i = 0;
                while i < array.len()
                {
                    array[i] = array[i].cos();
                    i += 1;
                }
            });
		}
		Ok(self)
	}
    
    fn complex_cos(mut self) -> VecResult<Self>
	{
		{
			let mut array = &mut self.data;
			let length = array.len();
			Chunk::execute_partial(Complexity::Medium, &mut array, length, 2, |array| {
                let mut i = 0;
                while i < array.len()
                {
                    let complex = Complex32::new(array[i], array[i + 1]);
                    let result = complex.cos();
                    array[i] = result.re;
                    array[i + 1] = result.im;
                    i += 2;
                }
            });
		}
		Ok(self)
	}
}