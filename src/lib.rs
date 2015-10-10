mod vector_types;
use vector_types::*;

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
	let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
	let mut result = RealVector::new(&mut data);
	add_one(&mut result);
	assert_eq!(result.data, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
}

#[test]
fn add_one_test_odd_number_of_elements()
{
	let mut data = [1.0, 2.0, 3.0];
	let mut result = RealVector::new(&mut data);
	add_one(&mut result);
	assert_eq!(result.data, [2.0, 3.0, 4.0]);
}