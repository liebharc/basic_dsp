#[allow(dead_code)]
pub struct RealVector<'a>
{
	pub data: &'a mut[f64],
	length_valids: usize,
	pub delta: f64
}

#[allow(dead_code)]
impl<'a> RealVector<'a>
{
	pub fn new<'b>(data: &'b mut[f64]) -> RealVector<'b>
	{
		let length_valids = data.len();
		return RealVector { data: data, length_valids: length_valids, delta: 1.0 };
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