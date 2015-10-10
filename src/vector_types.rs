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
}