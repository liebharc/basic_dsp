#!/usr/bin/env perl
# Checks if facade32.rs has functions for all public trait methods. This script needs to be extended
# if new traits are added.

use strict;
use warnings;

sub parse_trait_definition {
    my ($file, @traits) = @_;
    open DEF, "<", $file or die $!;
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
            $record = $trait =~ /($traits)/;
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

sub parse_enum_definition {
    my ($file, @enums) = @_;
    open DEF, "<", $file or die $!;
    my $enums = join("|", @enums);
    my @enums_found;
    my $level = 0;
    my $record = 0;
    my @members = ();
    while (<DEF>) {
        my $line = $_;
        chomp $line;
        if ($level == 0 and $line =~ /^pub enum (\w+)/) {
            my $enum = $1;
            $record = $enum =~ /($enums)/;
            push @enums_found, $enum;
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
        
        if ($level == 1 and $record and $line =~ /(\w+)[\(;]/) {
            my $member = $1;
            push @members, $member;
        }
    }
    close DEF;
    
    my @union;
    for my $enum (@enums_found)
    {
        if ( grep( /^$enum$/, @enums ) ) {
          push @union, $enum;
        }
    }
    
    if (scalar @enums ne scalar @union) {
        die "@enums should have been parsed but only @union were found.";
    }
    
    return @members;
}

sub parse_facade {
    my ($type) = @_;
    open FACADE32, "<", "facade32.rs" or die $!;
    my @methods = ();
    while (<FACADE32>) {
        my $line = $_;
        chomp $line;
        if ($line =~ /^(\/\/)?\s*pub extern fn (\w+)32.*$type/) {
           push @methods, $2; 
        }
    }
    close FACADE32;
    return @methods;
}

sub camel_to_snake {
    my @input = @_;
    my @output;
    for my $input (@input) {
        $input =~ s/([a-z])([A-Z])/"$1_$2"/egx;
        push @output, lc $input;
    }
    
    return @output;
}

# DataVector32
my @definitions = parse_trait_definition("../vector_types/definitions.rs", "GenericVectorOps", "RealVectorOps", "ComplexVectorOps");
push @definitions, parse_trait_definition("../vector_types/time_freq_impl.rs", "TimeDomainOperations", "FrequencyDomainOperations", "SymmetricFrequencyDomainOperations", "SymmetricTimeDomainOperations");
push @definitions, parse_trait_definition("../vector_types/correlation_impl.rs", "CrossCorrelation");
push @definitions, parse_trait_definition("../vector_types/convolution_impl.rs", "Convolution", "VectorConvolution", "FrequencyMultiplication");
push @definitions, parse_trait_definition("../vector_types/interpolation_impl.rs", "Interpolation", "RealInterpolation");
my @impl = parse_facade("DataVector32");
my $found = 0;
my $missing = 0;
for my $def (@definitions) {
    if (grep(/^$def$/, @impl)) {
        $found++;
    }
    else {
        print "missing for DataVector32: $def\n";
        $missing++;
    }
}

# multi-ops
@definitions = parse_enum_definition("../vector_types/operations_enum.rs", "Operation");
@definitions = camel_to_snake(@definitions);
@impl = parse_facade("PreparedOp1F32");
for my $def (@definitions) {
    my $regex = sprintf("%s_ops1_f", $def);
    if (grep(/^$regex$/, @impl)) {
        $found++;
    }
    else {
        print "missing for PreparedOp1F32: $def\n";
        $missing++;
    }
}

@impl = parse_facade("PreparedOp2F32");
for my $def (@definitions) {
    my $regex = sprintf("%s_ops2_f", $def);
    if (grep(/^$regex$/, @impl)) {
        $found++;
    }
    else {
        print "missing for PreparedOp2F32: $def\n";
        $missing++;
    }
}

print "found: $found, missing: $missing\n";