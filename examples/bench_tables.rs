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
const INIT_VAL_RANGE: (f64, f64) = (-100.0, 100.0);

const USAGE: &'static str = "
Benchmarks various functions over different daza sizes and prints the results to STDOUT in
a table format. Results can be piped to `plot_csv_data.py` to produce a plot.

Usage: bench_tables [--limit=<size>]
       bench_tables (--help)

Options:
    -h, --help  Display usage.
	--limit=<size>     Data sizes tested will not exceed this value. [default: 0]
";

/// Benchmarks the given function. The results are more rough estimations and should
/// only be interpreted with care and in context of more results.
fn bench_real<F: FnMut(RealTimeVec64)>(
        name: &'static str,
        data: &Vec<f64>,
        results: &mut HashMap<&'static str, Vec<i64>>,
        mut func: F) {
    let mut dsp = data.clone().to_real_time_vec();
    let mut settings = dsp.get_multicore_settings().clone();
    settings.core_limit = 1;
    dsp.set_multicore_settings(settings);
    let start = PreciseTime::now();
    func(dsp);
    let end = PreciseTime::now();
    let duration = start.to(end).num_nanoseconds().unwrap();
    results.entry(name).or_insert(Vec::new()).push(duration);
}

fn create_pseudo_random_data(data_set_size: usize, seed: usize) -> Vec<f64> {
    let mut vec: Vec<f64> = vec![0.0; data_set_size];
    let seed: &[_] = &[seed, 42];
    let mut rng: StdRng = SeedableRng::from_seed(seed); // Create repeatable data
    for n in &mut vec {
        *n = rng.gen_range(INIT_VAL_RANGE.0, INIT_VAL_RANGE.1);
    }
    vec
}

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

	let data_sizes: Vec<usize> = vec![1000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000,
					  100000, 200000, 500000, 1000000, 5000000, 100000000];
	let data_sizes: Vec<usize> =
		data_sizes.into_iter()
		.take_while(|x| limit == 0 || *x <= limit)
		.collect();
    for data_set_size in &data_sizes {
		/*let mut buffer = SingleBuffer::new();
        buffer.free(Vec::with_capacity(*data_set_size));*/
		let original = create_pseudo_random_data(*data_set_size, 5798734);
        //let argument = create_pseudo_random_data(*data_set_size, 1412412).to_real_time_vec();
        //let imp_resp = create_pseudo_random_data(128, 341313).to_real_time_vec();

        /*bench_real("Offset", &original, &mut results, |mut v|v.offset(5.0));
        bench_real("Sin", &original, &mut results, |mut v|v.sin());
        bench_real("Log", &original, &mut results, |mut v|v.log(10.0));
        bench_real("Powf", &original, &mut results, |mut v|v.powf(5.0));
        bench_real("Sqrt", &original, &mut results, |mut v|v.sqrt());
        bench_real("Convolve", &original, &mut results, |mut v|v.convolve_vector(&mut buffer, &imp_resp).unwrap());
        bench_real("Add", &original, &mut results, |mut v|v.add(&argument).unwrap());
        bench_real("Dot Product", &original, &mut results, |v|{ let _ = v.dot_product(&argument).unwrap(); });*/
        bench_real("Approx", &original, &mut results, |mut v|v.log_approx(10.0));
        bench_real("Accurate", &original, &mut results, |mut v|v.log(10.0));
    }

	// Print
	print!("Sizes, ");
	for value in &data_sizes {
		print!("{:?}, ", value);
	}
	println!("");
	let mut sorted: Vec<_> = results.keys().map(|x|x.clone()).collect();
	sorted.sort();
	for key in &sorted {
		print!("{:?} [ns], ", key);
		for value in &results[key] {
			print!("{:?}, ", value);
		}
		println!("");
	}
}
