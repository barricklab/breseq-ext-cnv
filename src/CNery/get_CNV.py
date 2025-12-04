#!/usr/bin/env python
# coding: utf-8
import argparse
from pathlib import Path

from .core import (
    bam2cov_to_df,
    preprocess,
    gc_correction,
    gc_cor_plots,
    otr_correction,
    plot_otr_corr,
    run_HMM,
    plot_copy
)

def main():
    from argparse import RawTextHelpFormatter
    import textwrap

    parser = argparse.ArgumentParser(description = "CNery is python package extension to breseq that analyzes the sequencing coverage across the genome to predict copy number variation (CNV)",
                                        epilog=textwrap.dedent('''\
                                        Run this script in the breseq output folder that contains 'data' and 'output' folders. 
                                        '''), 
                                        formatter_class = RawTextHelpFormatter)

    # Define the command line arguments
    parser.add_argument(
        "-i",
        "--input",
        action="store",
        dest="i",
        required=False,
        default="./data/reference.bam",
        type=str,
        help="input .bam file from breseq output",
    )
    
    parser.add_argument(
        "-ref",
        action="store",
        dest="ref",
        required=False,
        default="./data/reference.fasta",
        type=str,
        help="select the reference file used for breseq",
    )

    parser.add_argument(
        "-reg",
        action="store",
        dest="reg",
        required=False,
        type=str,
        help="select the region of the genome to evaluate",
    )

    parser.add_argument(
        "-o",
        "--output",
        action="store",
        dest="o",
        required=False,
        default='CNV_out',
        type=str,
        help="output file prefix. Defaults to the CNV_out folder.",
    )

    parser.add_argument(
        "-w",
        "--window",
        action="store",
        dest="w",
        required=False,
        default = 200,
        type=int,
        help="Define window length to parse through the genome and calculate coverage and GC statistics.",
    )

    parser.add_argument(
        "-s",
        "--step-size",
        action="store",
        dest="s",
        required=False,
        default = 100,
        type=int,
        help="Define step size (<= window size) for each progression of the window across the genome sequence. Set step-size=window-size if non-overlapping windows.",
    )

    parser.add_argument(
        "-ori",
        "--origin",
        action="store",
        dest="ori",
        #default=3886082,
        required=False,
        type=int,
        help="Genomic coordinate for origin of replication.",
    )
    parser.add_argument(
        "-ter",
        "--terminus",
        action="store",
        dest="ter",
        #default=1567362,
        required=False,
        type=int,
        help="Genomic coordinate for terminus of replication.",
    )
    parser.add_argument(
        "-f",
        "--frag_size",
        action="store",
        dest="f",
        default=500,
        required=False,
        type=int,
        help="Average fragment size of the sequencing reads.",
    )
    parser.add_argument(
        "-e",
        "--error-rate",
        action="store",
        dest="e",
        default=0.05,
        required=False,
        type=float,
        help="Approximate error rate in sequencing read coverage/refrence alignment.",
    )
    parser.add_argument(
        "--bias", 
        choices=['all', 'none', 'gc', 'otr'],
        default='all',
        required=False,
        help="Select specific bias correction (only OTR or only GC) to run before CN prediction."
        )
    # Parse the command line arguments
    options = parser.parse_args()

    out_dir="./output/"+options.o
    
    out_subdirs = ['/CNV_plt' , '/CNV_csv', '/GC_bias', '/OTR_corr']
    for i in range(len(out_subdirs)):
        Path(out_dir+out_subdirs[i]).mkdir(parents=True, exist_ok=True)

    region = options.reg

    if region is not None:
        parts = region.split('-')
        if len(parts) == 2 :
            if parts[0] == '':
                pltend = int(parts[1])
                pltstart = 0
            elif parts[1] == '':
                pltstart = int(parts[0])
                pltend = 0
            else:
                pltstart, pltend = int(parts[0]), int(parts[1])
        else:
            return "Invalid region. Ensure the region is specified by int values of 2 genomic coordinates separated by a '-'."
    else:
        pltstart, pltend = 0, 0
    
    # Select the method to determine origin and terminus of the genome
    enforce=False

    if options.ori and options.ter is not None:
        print("Ori has been set (value is %s)" % options.ori)
        print("Ter has been set (value is %s)" % options.ter)
        ori=options.ori,
        ter=options.ter,
        enforce=True
    else:
        options.ori=None,
        options.ter=None,
        print("Ori has not been set (default value is %s)" % options.ori)
        print("Ter has not been set (default value is %s)" % options.ter)
        ori=options.ori,
        ter=options.ter,
        # enforce=False

    print("Calculating coverage pileup at each nucleotide base across the reference genome")
    df_tab = bam2cov_to_df(options.i, options.ref, options.o)
    print('Calculating coverage and GC% across sliding window over the genome.')
    df_b2c = preprocess(df_tab, options.w, options.s, options.f)

    smpl = out_dir.strip().split('/')[-1]
    
    if options.bias == "gc":
        df_gc = gc_correction(df_b2c)
        gc_cor_plots(df_gc, out_dir)
        print(f'{smpl}: GC bias vs coverage plots saved.')
        df_gc["otr_gc_corr_norm_cov"] = df_gc["gc_corr_norm_cov"]
        df_cnv=run_HMM(df_gc, out_dir)
        plot_copy(df_cnv, pltstart, pltend, output=out_dir)
        print(f'{smpl}: CNV prediction plots saved.')
    
    elif options.bias == "otr":
        df_b2c["gc_corr_norm_cov"] = df_b2c["norm_raw_cov"]
        df_otr, ori_win, ter_win = otr_correction(df_b2c, out_dir,
                                                   ori, ter, enforce)
        print('Corrected origin/terminus of replication(OTR) bias in coverage.')
        plot_otr_corr(df_otr, output=out_dir, ori=ori_win, ter=ter_win)
        print(f'{smpl}: OTR bias vs coverage plots saved.')
        df_cnv=run_HMM(df_otr, out_dir)
        plot_copy(df_cnv, pltstart, pltend, output=out_dir)
        print(f'{smpl}: CNV prediction plots saved.')

    elif options.bias == "none":
        df_b2c["otr_gc_corr_norm_cov"] = df_b2c["norm_raw_cov"]
        df_cnv=run_HMM(df_b2c, out_dir)
        plot_copy(df_cnv, pltstart, pltend, output=out_dir)
        print(f'{smpl}: CNV prediction plots saved.')

    elif options.bias == "all":
        df_gc = gc_correction(df_b2c)
        gc_cor_plots(df_gc, out_dir)
        print(f'{smpl}: GC bias vs coverage plots saved.')
        df_otr, ori_win, ter_win = otr_correction(df_gc, out_dir,
                                                   ori, ter, enforce)
        print('Corrected origin/terminus of replication(OTR) bias in coverage.')
        plot_otr_corr(df_otr, output=out_dir, ori=ori_win, ter=ter_win)
        print(f'{smpl}: OTR bias vs coverage plots saved.')
        df_cnv=run_HMM(df_otr, out_dir)
        plot_copy(df_cnv, pltstart, pltend, output=out_dir)
        print(f'{smpl}: CNV prediction plots saved.')

if __name__ == "__main__":
    main()