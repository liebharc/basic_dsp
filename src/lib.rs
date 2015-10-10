pub fn add_one(data: &mut [f32]) 
{
	for i in 0..10
	{ 
		data[i] = data[i] + 1.0;
	}
}

#[test]
fn add_one_test()
{
	let mut result = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
	add_one(&mut result);
	assert_eq!(result, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
}