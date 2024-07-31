# breseq-ext-cnv
*breseq* copy number variation extension accepts .tab coverage output from [*breseq*](https://github.com/barricklab/breseq.git) BAM2COV and predicts copy number variations across the genome after correcting the biases in read counts introduced by variations in sequencing methods.

**Installation:**

Create python environment. 
```
conda create -n <env-name> python>=3.8
conda activate <env-name>
```
Install breseq-ext-cnv
```
pip install git+https://github.com/barricklab/breseq-ext-cnv.git
```
**Run:**

Run BAM2COV on *breseq* output to get the coverage table: 
```
breseq bam2cov -t[--table] --resolution 0 (0=single base resolution) --region <reference:START-END> --output <filename>
```
With the coverage table as the input determine regions of copy number variation using: 

```
breseq-ext-cnv -i <input file> [-o <output folder location>] [-w <window>] [-s <step size>]
```
```
usage: get_CNV.py [-h] -i I [-o O] [-w W] [-s S] [-ori ORI] [-ter TER] [-f F]

The breseq-ext-cnv is python package extension to breseq that analyzes the sequencing coverage across the genome to determine specific regions that have undergone copy number variation (CNV)

options:
  -h, --help            show this help message and exit
  -i I, --input I       input .tab file address from breseq bam2cov.
  -o O, --output O      output file location preference. Defaults to the current folder.
  -w W, --window W      Define window length to parse through the genome and calculate coverage and GC statistics.
  -s S, --step-size S   Define step size (<= window size) for each progression of the window across the genome sequence. Set = window size if non-overlapping windows.
  -ori ORI, --origin ORI
                        Genomic coordinate for origin of replication.
  -ter TER, --terminus TER
                        Genomic coordinate for terminus of replication.
  -f F, --frag_size F   Average fragment size of the sequencing reads.

Input .tab file from breseq bam2cov. To get the coverage file run the command below in your breseq directory which contains the 'data' and 'output' folders.
'```
breseq bam2cov -t[--table] --resolution 0 (0=single base resolution) --region <reference:START-END> --output <filename>
```'
```
