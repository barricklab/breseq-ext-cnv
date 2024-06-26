# breseq-ext-cnv
*breseq* copy number variation extension accepts .tab coverage output from [*breseq*](https://github.com/barricklab/breseq.git) BAM2COV and predicts copy number variations across the genome after correcting the biases in read counts intordiced by variations in sequencing methods.

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
breseq-ext-cnv -i <input file> [-o <output folder location>]
```
```
breseq-ext-cnv --help
usage: breseq-ext-cnv [-h] -i I [-o O] [-ori ORI] [-ter TER]

Input .tab file from breseq bam2cov

breseq bam2cov -t[--table] --resolution 0 (0=single base resolution) --region <reference:START-END> --output <filename>

options:
  -h, --help            show this help message and exit
  -i I, --input I       input .tab file address from breseq bam2cov
  -o O, --output O      output file location preference. Defaults to the current
                        folder
  -ori ORI, --origin ORI
                        Genomic coordinate for origin of replication
  -ter TER, --terminus TER
                        Genomic coordinate for terminus of replication
```
