#!/usr/bin/env perl
# Rust macros don't allow right now to concat strings to generate method names,
# see https://github.com/rust-lang/rust/issues/12249 and https://github.com/rust-lang/rust/issues/29599 for more information.
# As long as there is no way to do that in Rust this Perl script does the copy&paste&replace work for us.
# To keep the build simple facade64.rs is still checked in.

use strict;
use warnings;

use Cwd 'abs_path';
use File::Basename;

my $location = dirname(abs_path($0));
print "$location\n";

open FACADE32, "<", "$location/facade32.rs" or die $!;
open FACADE64, ">", "$location/facade64.rs.tmp" or die $!;
print FACADE64 "// Auto generated code, change facade32.rs and run facade64_create.pl\n";
while (<FACADE32>) {
    my $line = $_;
    chomp $line;
    $line =~ s/(\w+)Vector32/$1Vector64/g;
    $line =~ s/f32/f64/g;
    $line =~ s/32bit/64bit/g;
    $line =~ s/Complex32/Complex64/g;
    $line =~ s/^pub extern fn (\w+)32/pub extern fn ${1}64/;
    $line =~ s/fn.(\w+)32.html/fn.${1}64.html/;
    $line =~ s/`(\w+)32`/`${1}64`/;
    print FACADE64 "$line\n";
}
close FACADE32;
close FACADE64;

if (-f "$location/facade64.rs") {
    unlink "$location/facade64.rs";
}

rename "$location/facade64.rs.tmp", "$location/facade64.rs";