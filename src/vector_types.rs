
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

#[allow(dead_code)]
pub struct DataVector<'a> 
{
	pub data: &'a mut[f32],
	length_valids: usize,
	pub delta: f64,
	pub is_complex: bool
}

#[allow(dead_code)]
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
}