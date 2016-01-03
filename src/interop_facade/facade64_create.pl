#!/usr/bin/env perl
# Rust macros don't allow right now to concat strings to generate method names,
# see https://github.com/rust-lang/rust/issues/12249 for more information.
# As long as there is no way to do that in Rust this Perl script does the copy&paste&replace work for us.
# To keep the build simple facade64.rs is still checked in.

use strict;
use warnings;

open FACADE32, "<", "facade32.rs" or die $!;
open FACADE64, ">", "facade64.rs.tmp" or die $!;
while (<FACADE32>) {
    my $line = $_;
    chomp $line;
    $line =~ s/(\w+)Vector32/$1Vector64/g;
    $line =~ s/f32/f64/g;
    $line =~ s/Complex32/Complex64/g;
    $line =~ s/^pub extern fn (\w+)32/pub extern fn ${1}64/;
    print FACADE64 "$line\n";
}
close FACADE32;
close FACADE64;

if (-f "facade64.rs") {
    unlink "facade64.rs";
}

rename "facade64.rs.tmp", "facade64.rs";