extern crate basic_dsp;
extern crate docopt;

use basic_dsp::*;
use docopt::Docopt;
use std::env;

const USAGE: &'static str = "
This program shows the internal calibration information of basic_dsp.

Usage:
    show_calibration
    show_calibration (--help)

Options:
    -h, --help  Display usage.
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

    println!("{}", print_calibration())
}
