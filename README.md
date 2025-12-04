# CNery
*breseq* copy number variation extension predicts copy numbers using sequencing coverage output from [*breseq*](https://github.com/barricklab/breseq.git). The predictions account for biases in coverage due to sequencing methodologies such as GC content or prokaryotic physiological states during DNA isolation that may bias sequencing coverage towards origin of genome replication.

**Installation:**

Recommended: Create conda python environment.

```
mamba env create -f environment.yml
mamba activate CNery
```
Install breseq-ext-cnv
```
pip install git+https://github.com/barricklab/breseq-ext-cnv.git
```
**Run:**

Run CNery in the *breseq* folder that contains folders 01_.., 02_.., data and output: 

```
CNery [-o <output folder location>] [-w <window>] [-s <step size>] [-f <fragment length>]
```

If running this script from in a different location, specify input .bam file, refrence.fasta file and output paths:
```
CNery [-i <input .bam file>] [-ref <refrence.fasta file location>][-o <output folder location>] [-w <window>] [-s <step size>] [-f <fragment length>]
```

**Run examples:**

Calculate coverage with a window size of 500 and an sliding increment size of 250 to summarize coverage across the genome. Average fragment length of the sequencing platform is 300bp

```
CNery -o <output folder> -w 500 -s 250 -f 300
```
Analyze coverage across the whole genome but output copy number prediction and coverage plots of a specific genomic segment
```
CNery -o <output folder> --region 3497890-3955678 -w 1000 -s 500
```
Correct bias in sequencing coverage due to either one of the two factors GC or OTR (origin-terminus of replication) or no correction:
```
CNery -o <output folder> -w 500 -s 250 --bias otr #only correct for bias in coverage due to replication

CNery -o <output folder> -w 500 -s 250 --bias gc #only correct for bias in coverage due to GC content of the sequence

CNery -o <output folder> -w 500 -s 250 --bias none #no bias correction to be applied before CN prediction
```
**CNery run options**
```
$CNery -h

usage: CNery [-h] [-i I] [-ref REF] [-reg REG] [-o O] [-w W] [-s S] [-ori ORI] [-ter TER] [-f F] [-e E]
             [--bias {all,none,gc,otr}]

CNery is python package extension to breseq that analyzes the sequencing coverage across the genome to predict copy number variation (CNV)

options:
  -h, --help            show this help message and exit
  -i, --input I         input .bam file from breseq output
  -ref REF              select the reference file used for breseq
  -reg REG              select the region of the genome to evaluate
  -o, --output O        output file prefix. Defaults to the CNV_out folder.
  -w, --window W        Define window length to parse through the genome and calculate coverage and GC statistics.
  -s, --step-size S     Define step size (<= window size) for each progression of the window across the genome sequence. Set step-size=window-size if non-overlapping windows.
  -ori, --origin ORI    Genomic coordinate for origin of replication.
  -ter, --terminus TER  Genomic coordinate for terminus of replication.
  -f, --frag_size F     Average fragment size of the sequencing reads.
  -e, --error-rate E    Approximate error rate in sequencing read coverage/refrence alignment.
  --bias {all,none,gc,otr}
                        Select specific bias correction (only OTR or only GC) to run before CN prediction.

Run this script in the breseq output folder that contains 'data' and 'output' folders. 
```

