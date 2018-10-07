#!/usr/bin/env perl
# Checks if facade32.rs has functions for all public trait methods. This script needs to be extended
# if new traits are added.

use strict;
use warnings;

sub parse_trait_definition {
    my ($file, @traits) = @_;
    open DEF, "<", $file or die "$! while opening $file";
    my $traits = join("|", @traits);
    my @traits_found;
    my $level = 0;
    my $record = 0;
    my @methods = ();
    while (<DEF>) {
        my $line = $_;
        chomp $line;
        if ($level == 0 and $line =~ /^pub trait (\w+)/) {
            my $trait = $1;
            $record = $trait =~ /^($traits)$/;
            push @traits_found, $trait;
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

    my @union;
    for my $trait (@traits_found)
    {
        if ( grep( /^$trait$/, @traits ) ) {
          push @union, $trait;
        }
    }

    if (scalar @traits ne scalar @union) {
        die "@traits should have been parsed but only @union were found.";
    }

    return @methods;
}

sub parse_facade {
    my ($type, $file) = @_;
    open FACADE32, "<", "$file" or die $!;
    my @methods = ();
    while (<FACADE32>) {
        my $line = $_;
        chomp $line;
        if ($line =~ /^(\/\/)?\s*pub extern "C" fn (\w+)32.*$type/) {
           push @methods, $2;
        }
    }
    close FACADE32;
    return @methods;
}

# DataVector32
my $root = "../../vector/src/vector_types/";
my @definitions = parse_trait_definition("$root/complex/complex_ops.rs", "ComplexOps");
push @definitions, parse_trait_definition("$root/complex/complex_to_real.rs", "ComplexToRealTransformsOps", "ComplexToRealGetterOps", "ComplexToRealSetterOps");
push @definitions, parse_trait_definition("$root/general/data_reorganization.rs", "ReorganizeDataOps", "InsertZerosOps", "SplitOps", "MergeOps");
push @definitions, parse_trait_definition("$root/general/diff_sum.rs", "DiffSumOps");
push @definitions, parse_trait_definition("$root/general/dot_products.rs", "DotProductOps");
push @definitions, parse_trait_definition("$root/general/elementary.rs", "ScaleOps", "OffsetOps", "ElementaryOps", "ElementaryWrapAroundOps");
push @definitions, parse_trait_definition("$root/general/mapping.rs", "MapInplaceOps", "MapAggregateOps");
push @definitions, parse_trait_definition("$root/general/statistics.rs", "StatisticsOps");
push @definitions, parse_trait_definition("$root/general/trigonometry_and_powers.rs", "TrigOps", "PowerOps");
push @definitions, parse_trait_definition("$root/real/real_ops.rs", "RealOps", "ModuloOps");
push @definitions, parse_trait_definition("$root/real/real_to_complex.rs", "RealToComplexTransformsOps");
push @definitions, parse_trait_definition("$root/time_freq/convolution.rs", "ConvolutionOps", "FrequencyMultiplication");
push @definitions, parse_trait_definition("$root/time_freq/correlation.rs", "CrossCorrelationOps");
push @definitions, parse_trait_definition("$root/time_freq/freq.rs", "FrequencyDomainOperations");
push @definitions, parse_trait_definition("$root/time_freq/freq_to_time.rs", "FrequencyToTimeDomainOperations", "SymmetricFrequencyToTimeDomainOperations");
push @definitions, parse_trait_definition("$root/time_freq/interpolation.rs", "InterpolationOps");
push @definitions, parse_trait_definition("$root/time_freq/real_interpolation.rs", "RealInterpolationOps");
push @definitions, parse_trait_definition("$root/time_freq/time.rs", "TimeDomainOperations");
push @definitions, parse_trait_definition("$root/time_freq/time_to_freq.rs", "TimeToFrequencyDomainOperations", "SymmetricTimeToFrequencyDomainOperations");
my @impl = parse_facade("VecBuf", "facade32.rs");
my $found = 0;
my $missing = 0;
for my $def (@definitions) {
    if (grep(/^$def$/, @impl)) {
        $found++;
    }
    elsif (grep(/^real_$def$/, @impl)) {
        $found++;
    }
    elsif (grep(/^complex_$def$/, @impl)) {
        $found++;
    }
    elsif (grep(/^${def}_real$/, @impl)) {
        $found++;
    }
    elsif (grep(/^${def}_complex$/, @impl)) {
        $found++;
    }
    elsif (grep(/^${def}_vector$/, @impl)) {
        $found++;
    }
    else {
        print "missing for DataVector32: $def\n";
        $missing++;
    }
}

print "found: $found, missing: $missing\n";