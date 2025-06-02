# breseq-ext-cnv
*breseq* copy number variation extension accepts .tab coverage output from [*breseq*](https://github.com/barricklab/breseq.git) BAM2COV and predicts copy number variations across the genome after correcting the biases in read counts introduced by variations in sequencing methods.

**Installation:**

Recommended: Create conda python environment.
```
conda create -n <env-name> python>=3.9
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

**Run examples:**
```
breseq-ext-cnv -i <input file>
```

```
# calculate coverage with a window size of 500 and a increment of 250 with average sequencing fragment length of 300bp
breseq-ext-cnv -i <input file> -w 500 -s 250 -f 300
```

```
# output copy number prediction and coverage plots of a specific genomic segment
breseq-ext-cnv -i <input file> --region 3497890-3955678 -w 1000 -s 500
```

```
#
```


```
$breseq-ext-cnv -h

usage: get_CNV.py [-h] -i I [-o O] [-w W] [-s S] [-ori ORI] [-ter TER] [-f F] [-e E]

The breseq-ext-cnv is python package extension to breseq that analyzes the sequencing coverage across the genome to determine specific regions that have undergone copy number variation (CNV)

options:
  -h , --help          show this help message and exit
  -i , --input        input .tab file address from breseq bam2cov.
  -o , --output       output file location preference. Defaults to the current folder.
  -w , --window       Define window length to parse through the genome and calculate coverage and GC statistics.
  -s , --step-size    Define step size (<= window size) for each progression of the window across the genome sequence. 
                      Set = window size if non-overlapping windows.
  --region            Set regions between which to display output coverage plots.
  -ori , --origin     Genomic coordinate for origin of replication.
  -ter , --terminus   Genomic coordinate for terminus of replication.
  -f , --frag_size    Average fragment size of the sequencing reads.
  -e , --error-rate   Error rate in sequencing read coverage, taken into account to accurately determine 0 copy coverage.

Input .tab file from breseq bam2cov. To get the coverage file run the command below in your breseq directory which contains the 'data' and 'output' folders.
```

```
breseq bam2cov -t[--table] --resolution 0 (0=single base resolution) --region <reference:START-END> --output <filename>
```

