use simd::f32x4;

pub struct Complex
{
	pub real: f32,
	pub imag: f32
}

impl Complex
{
	pub fn new(real: f32, imag: f32) -> Complex
	{
		return Complex { real: real, imag: imag };
	}
}

const NUM_CHUNKS: usize = 8;

#[derive(Debug)]
#[derive(PartialEq)]
struct Chunk
{
	pub start: usize,
	pub end: usize,
	pub step_size: usize
}

impl Chunk
{
	fn new(start: usize, end: usize, step_size: usize) -> Chunk
	{
		return Chunk { start: start, end: end, step_size: step_size };
	}
	
	fn partition(array_length: usize, step_size: usize) -> Vec<Chunk>
	{
		let mut result = Vec::new();
		if array_length < 1000
		{
			result.push(Chunk::new(0, array_length - 1, step_size));
			return result;
		}
		else
		{
			let mut chunk_size = array_length / NUM_CHUNKS;
			chunk_size -= chunk_size % step_size;
			let mut i = 0;
			let mut last_start = 0;
			while i < NUM_CHUNKS
			{
				let new_start = 
					if i != NUM_CHUNKS - 1 { last_start + chunk_size } 
					else				     { array_length };
				result.push(Chunk::new(last_start, new_start - 1, step_size));
				last_start = new_start;
				i += 1;
			}
			
			return result;
		}
	}
}

pub struct DataVector<'a> 
{
	pub data: &'a mut[f32],
	length_valids: usize,
	pub delta: f64,
	pub is_complex: bool
}

impl<'a> DataVector<'a>
{
	pub fn new<'b>(data: &'b mut[f32]) -> DataVector<'b>
	{
		let length_valids = data.len();
		return DataVector { data: data, length_valids: length_valids, delta: 1.0, is_complex: false };
	}

    pub fn len(&self) -> usize 
	{
		let array_length = self.allocated_length();
        if  self.length_valids < array_length
		{
			return self.length_valids;
		}
		else
		{
			return array_length;
		}
    }
	
	pub fn allocated_length(&self) -> usize 
	{
		return self.data.len();
	}
	
	pub fn inplace_real_offset(&mut self, offset: f32) 
	{
		let increment_vector = f32x4::splat(offset); 
		let data_length = self.len();
		let scalar_length = data_length % 4;
		let vectorization_length = data_length - scalar_length;
		let array = &mut self.data;
		let mut i = 0;
		while i < vectorization_length
		{ 
			let vector = f32x4::load(array, i);
			let incremented = vector + increment_vector;
			incremented.store(array, i);
			i += 4;
		}
		
		for i in vectorization_length..data_length
		{
			array[i] = array[i] + offset;
		}
	}
	
	pub fn inplace_complex_offset(&mut self, offset: Complex)
	{
		let real_imag = &[offset.real, offset.imag, offset.real, offset.imag];
		let increment_vector = f32x4::load(real_imag, 0); 
		let data_length = self.len();
		let scalar_length = data_length % 4;
		let vectorization_length = data_length - scalar_length;
		let mut array = &mut self.data;
		let mut i = 0;
		while i < vectorization_length
		{ 
			let vector = f32x4::load(array, i);
			let incremented = vector + increment_vector;
			incremented.store(array, i);
			i += 4;
		}
		
		for i in vectorization_length..data_length
		{
			array[i] = array[i] + real_imag[i % 2];
		}
	}
	
	pub fn inplace_real_scale(&mut self, scale: f32) 
	{
		let increment_vector = f32x4::splat(scale); 
		let data_length = self.len();
		let scalar_length = data_length % 4;
		let vectorization_length = data_length - scalar_length;
		let array = &mut self.data;
		let mut i = 0;
		while i < vectorization_length
		{ 
			let vector = f32x4::load(array, i);
			let incremented = vector * increment_vector;
			incremented.store(array, i);
			i += 4;
		}
		
		for i in vectorization_length..data_length
		{
			array[i] = array[i] * scale;
		}
	}
	
	pub fn inplace_complex_scale(&mut self, scale: Complex)
	{
		let real_imag = &[scale.real, scale.imag, scale.real, scale.imag];
		let increment_vector = f32x4::load(real_imag, 0); 
		let data_length = self.len();
		let scalar_length = data_length % 4;
		let vectorization_length = data_length - scalar_length;
		let mut array = &mut self.data;
		let mut i = 0;
		while i < vectorization_length
		{ 
			let vector = f32x4::load(array, i);
			let incremented = vector * increment_vector;
			incremented.store(array, i);
			i += 4;
		}
		
		for i in vectorization_length..data_length
		{
			array[i] = array[i] * real_imag[i % 2];
		}
	}
}

#[test]
fn partition_small_arrays_test()
{
	let result = Chunk::partition(500, 2);
	assert_eq!(result, vec![Chunk::new(0, 499, 2)]);
}

#[test]
fn partition_large_arrays_test()
{
	let result = Chunk::partition(6777, 4);
	// test assumes that the chunk number is set to 8
	assert_eq!(result, 
		vec![ Chunk::new(0, 843, 4), 
		  Chunk::new(844, 1687, 4),
		  Chunk::new(1688, 2531, 4),
		  Chunk::new(2532, 3375, 4),
		  Chunk::new(3376, 4219, 4),
		  Chunk::new(4220, 5063, 4),
		  Chunk::new(5064, 5907, 4),
		  Chunk::new(5908, 6776, 4),
		 ]);
}

#[test]
fn add_real_one_test()
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let mut result = DataVector::new(&mut data);
	result.inplace_real_offset(1.0);
	assert_eq!(result.data, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
}

#[test]
fn add_real_two_test()
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let mut result = DataVector::new(&mut data);
	result.inplace_real_offset(2.0);
	assert_eq!(result.data, [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
}

#[test]
fn add_complex_test()
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let mut result = DataVector::new(&mut data);
	result.inplace_complex_offset(Complex::new(1.0, -1.0));
	assert_eq!(result.data, [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0]);
}

#[test]
fn add_real_one_odd_number_of_elements_test()
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
	let mut result = DataVector::new(&mut data);
	result.inplace_real_offset(1.0);
	assert_eq!(result.data, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn add_complex_odd_number_of_elements_test()
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
	let mut result = DataVector::new(&mut data);
	result.inplace_complex_offset(Complex::new(1.0, -1.0));
	assert_eq!(result.data, [2.0, 1.0, 4.0, 3.0, 6.0, 5.0]);
}

#[test]
fn scale_real_two_test()
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let mut result = DataVector::new(&mut data);
	result.inplace_real_scale(2.0);
	assert_eq!(result.data, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
}

#[test]
fn scale_real_half_test_test()
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let mut result = DataVector::new(&mut data);
	result.inplace_real_scale(0.5);
	assert_eq!(result.data, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);
}

#[test]
fn scale_complex_test()
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
	let mut result = DataVector::new(&mut data);
	result.inplace_complex_scale(Complex::new(2.0, -0.5));
	assert_eq!(result.data, [2.0, -1.0, 6.0, -2.0, 10.0, -3.0, 14.0, -4.0]);
}

#[test]
fn scale_real_two_odd_number_of_elements_test()
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
	let mut result = DataVector::new(&mut data);
	result.inplace_real_scale(2.0);
	assert_eq!(result.data, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]);
}

#[test]
fn scale_complex_odd_number_of_elements_test()
{
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
	let mut result = DataVector::new(&mut data);
	result.inplace_complex_scale(Complex::new(2.0, -0.5));
	assert_eq!(result.data, [2.0, -1.0, 6.0, -2.0, 10.0, -3.0]);
}