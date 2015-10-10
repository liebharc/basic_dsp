#[allow(dead_code)]
pub struct RealVector<'a>
{
	data: &'a mut[f64],
	length_valids: usize,
	delta: f64
}

#[allow(dead_code)]
impl<'a> RealVector<'a>
{
	fn new<'b>(data: &'b mut[f64]) -> RealVector<'b>
	{
		let length_valids = data.len();
		return RealVector { data: data, length_valids: length_valids, delta: 1.0 };
	}

    fn len(&self) -> usize 
	{
		let array_length = self.data.len();
        if  self.length_valids < array_length
		{
			return self.length_valids;
		}
		else
		{
			return array_length;
		}
    }
	
	fn allocated_length(&self) -> usize 
	{
		return self.data.len();
	}
}

pub fn add_one(data: &mut RealVector) 
{
	for i in 0..data.len()
	{ 
		data.data[i] = data.data[i] + 1.0;
	}
}

#[test]
fn add_one_test()
{
	let mut result = 
		RealVector 
		{ 
			data: &mut[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 
			length_valids: 10, 
			delta: 1.0
		};
	add_one(&mut result);
	assert_eq!(result.data, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
}

#[test]
fn add_one_test_odd_number_of_elements()
{
	let mut result = 
		RealVector 
		{ 
			data: &mut[1.0, 2.0, 3.0], 
			length_valids: 3, 
			delta: 1.0
		};
	add_one(&mut result);
	assert_eq!(result.data, [2.0, 3.0, 4.0]);
}