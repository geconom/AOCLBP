#!/usr/bin/perl

my $number = @ARGV[0];
my $limit = $number - 1;

open INFILE, "<", "example-input256_256.txt" or die $!;
open OUTFILE, ">", "example-input".$number."_".$number.".txt" or die $!;

my $line = <INFILE>;
print OUTFILE $number."\n";

while ($line = <INFILE>)
  {
    my $prefix = substr $line, 0, 2;
    if ($prefix == "  ")
      {
        my @values = split (" ", $line);
        for (my $i = 0; $i <= $limit; $i++) {
          print OUTFILE "  ".@values[$i % 256];
        }
        print OUTFILE "\n";
      }
    else {
      print OUTFILE $line;
    }
  }

close INFILE;
close OUTFILE;
