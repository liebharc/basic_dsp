#!/usr/bin/env perl
# Checks if facade32.rs has functions for all public trait methods. This script needs to be extended
# if new traits are added.

use strict;
use warnings;

sub parse_definition {
    my ($file, @traits) = @_;
    open DEF, "<", $file or die $!;
    my $traits = join("|", @traits);
    my $level = 0;
    my $record = 0;
    my @methods = ();
    while (<DEF>) {
        my $line = $_;
        chomp $line;
        if ($level == 0 and $line =~ /^pub trait (\S+)/) {
            my $trait = $1;
            $record = $trait =~ /($traits)/;
        }
        if ($line =~ /\{/) {
            $level++;
        }
        if ($line =~ /\}/) {
            $level--;
            if ($level == 0) {
                $record = 0;
            }
        }
        
        if ($level == 1 and $record and $line =~ /fn (\w+)/) {
            my $method = $1;
            push @methods, $method;
        }
    }
    close DEF;
    return @methods;
}

sub parse_facade {
    open FACADE32, "<", "facade32.rs" or die $!;
    my @methods = ();
    while (<FACADE32>) {
        my $line = $_;
        chomp $line;
        if ($line =~ /^(\/\/)?\s*pub extern fn (\w+)32/) {
           push @methods, $2; 
        }
    }
    close FACADE32;
    return @methods;
}

my @definitions = parse_definition("../vector_types/definitions.rs", "GenericVectorOperations", "RealVectorOperations", "ComplexVectorOperations");
push @definitions, parse_definition("../vector_types/time_freq_impl.rs", "TimeDomainOperations", "FrequencyDomainOperations", "SymmetricFrequencyDomainOperations", "SymmetricTimeDomainOperations");
push @definitions, parse_definition("../vector_types/correlation_impl.rs", "CrossCorrelation");
push @definitions, parse_definition("../vector_types/convolution_impl.rs", "Convolution");
my @impl = parse_facade();
my $found = 0;
my $missing = 0;
for my $def (@definitions) {
    if (grep(/^$def$/, @impl)) {
        $found++;
    }
    else {
        print "missing: $def\n";
        $missing++;
    }
}

print "found: $found, missing: $missing\n";