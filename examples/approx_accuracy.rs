extern crate basic_dsp;
extern crate docopt;

use basic_dsp::*;
use docopt::Docopt;
use std::env;
use std::io::prelude::*;

const USAGE: &'static str = "
Prints the accuracy of the approximation functions over a small range in a format which
is understood by plot_csv_data.py.

Usage: approx_accuracy
    approx_accuracy (--help)

Options:
    -h, --help  Display usage.
";

fn print_diff<F1: FnMut(&mut RealTimeVec64), F2: FnMut(&mut RealTimeVec64)>(
    name: &'static str,
    is_relative: bool,
    x_vec: &RealTimeVec64,
    mut std_func: F1,
    mut approx_func: F2,
) {
    let mut should = x_vec.clone();
    let mut is = x_vec.clone();
    std_func(&mut should);
    approx_func(&mut is);
    should
        .sub(&is)
        .expect("Vectors should have the same length");
    should.abs();
    if is_relative {
        should
            .div(x_vec)
            .expect("Vectors should have the same length");
    }
    print!("{}, ", name);
    for n in should.data(..) {
        print!("{}, ", *n);
    }
    println!("");
    let mut stderr = std::io::stderr();
    writeln!(&mut stderr, "{} max, {}", name, should.statistics().max)
        .expect("Could not write to stderr");
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

    let x_delta = 1e-3;
    let len = 10000;
    let mut x_vec = Vec::with_capacity(len);
    print!("X, ");
    for i in 1..(len + 1) {
        let x = x_delta * i as f64;
        print!("{}, ", x);
        x_vec.push(x);
    }
    println!("");

    let x_vec = x_vec.to_real_time_vec();
    print_diff("Sin", false, &x_vec, |x| x.sin(), |x| x.sin_approx());
    print_diff("Cos", false, &x_vec, |x| x.cos(), |x| x.cos_approx());
    print_diff("Ln", true, &x_vec, |x| x.ln(), |x| x.ln_approx());
    print_diff("Exp", true, &x_vec, |x| x.exp(), |x| x.exp_approx());
    print_diff("Log2", true, &x_vec, |x| x.log(2.0), |x| x.log_approx(2.0));
    print_diff(
        "Expf2",
        true,
        &x_vec,
        |x| x.expf(2.0),
        |x| x.expf_approx(2.0),
    );
    print_diff(
        "Powf2",
        true,
        &x_vec,
        |x| x.powf(2.0),
        |x| x.powf_approx(2.0),
    );
}
