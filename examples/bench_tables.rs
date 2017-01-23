extern crate basic_dsp;
extern crate docopt;
extern crate time;
extern crate rand;

use basic_dsp::*;
use std::env;
use time::PreciseTime;
use docopt::Docopt;
use rand::*;
use std::collections::HashMap;
const INIT_VAL_RANGE: (f32, f32) = (-100.0, 100.0);

const USAGE: &'static str = "
Benchmarks various functions over different daza sizes and prints the results to STDOUT in
a table format. Results can be piped to `plot_csv_data.py` to produce a plot.

Usage: bench_tables [--limit=<size>]
       bench_tables (--help)

Options:
    -h, --help  Display usage.
	--limit=<size>     Data sizes tested will not exceed this value. [default: 0]
";

fn main() {
    let argv = env::args();
    let args = Docopt::new(USAGE)
                  .and_then(|d| d.argv(argv.into_iter()).parse())
                  .unwrap_or_else(|e| e.exit());
    if args.get_bool("-h") || args.get_bool("--help") {
        println!("{}", USAGE);
        std::process::exit(0);
    }

	let limit_arg = args.get_str("--limit");
	let limit = if limit_arg != "" {
		match limit_arg.trim().parse::<usize>().ok() {
			 Some(l) => l,
			 _ => {
				 println!("{}", USAGE);
	         	  std::process::exit(1);}
		}
	} else {
		0
	};

	let mut results: HashMap<&'static str, Vec<i64>, _> = HashMap::new();
	results.insert("Offset", Vec::new());
	results.insert("Sin", Vec::new());
	results.insert("Log", Vec::new());

	let data_sizes: Vec<usize> = vec![1000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000,
					  100000, 200000, 500000, 1000000, 10000000, 500000, 100000000];
	let data_sizes: Vec<usize> =
		data_sizes.into_iter()
		.take_while(|x| limit == 0 || *x <= limit)
		.collect();
    for data_set_size in &data_sizes {
		//let mut buffer = SingleBuffer::new();
		let original = {
	        let mut vec = vec![0.0; *data_set_size];
	        let seed: &[_] = &[5798734, 198196];
	        let mut rng: StdRng = SeedableRng::from_seed(seed); // Create repeatable data
	        for n in &mut vec {
	            *n = rng.gen_range(INIT_VAL_RANGE.0, INIT_VAL_RANGE.1);
	        }
	        vec
	    };
		let mut dsp = original.clone().to_real_time_vec();
		let start = PreciseTime::now();
		dsp.offset(5.0);
		let end = PreciseTime::now();
		let duration = start.to(end).num_nanoseconds().unwrap();
		results.get_mut("Offset").unwrap().push(duration);

		let mut dsp = original.clone().to_real_time_vec();
		let start = PreciseTime::now();
		dsp.sin();
		let end = PreciseTime::now();
		let duration = start.to(end).num_nanoseconds().unwrap();
		results.get_mut("Sin").unwrap().push(duration);

		let mut dsp = original.clone().to_real_time_vec();
		let start = PreciseTime::now();
		dsp.log(10.0);
		let end = PreciseTime::now();
		let duration = start.to(end).num_nanoseconds().unwrap();
		results.get_mut("Log").unwrap().push(duration);
	}

	// Print
	print!("Sizes, ");
	for value in &data_sizes {
		print!("{:?}, ", value);
	}
	println!("");
	for (key, values) in results.iter() {
		print!("{:?} [ns], ", key);
		for value in values {
			print!("{:?}, ", value);
		}
		println!("");
	}
}
