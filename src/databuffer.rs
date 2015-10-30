use num_cpus;
use simple_parallel::Pool;

pub struct DataBuffer
{
	// Storing the pool saves a little bit of initialization time
	pool: &'static mut Pool,
	
	// TODO: buffer vectors so that they can be reused
}

impl DataBuffer
{
	fn get_static_pool() -> &'static mut Pool
	{
		use std::sync::{Once, ONCE_INIT};
		use std::mem::transmute;
		unsafe
		{
			static mut pool: *mut Pool = 0 as *mut Pool;
			static mut ONCE: Once = ONCE_INIT;
			ONCE.call_once(||
			{
				pool = transmute::<Box<Pool>, *mut Pool>(box Pool::new(num_cpus::get()));
			});
			
			let mut static_pool = &mut *pool;
			static_pool
		}
	}

	#[allow(unused_variables)]
	pub fn new(name: &str) -> DataBuffer
	{
		let pool = DataBuffer::get_static_pool();
		return DataBuffer { pool: pool };
	}
}

pub trait DataBufferAccess
{
	fn pool(&mut self) -> &mut Pool;
}

impl DataBufferAccess for DataBuffer
{
	fn pool(&mut self) -> &mut Pool
	{
		self.pool
	}
}