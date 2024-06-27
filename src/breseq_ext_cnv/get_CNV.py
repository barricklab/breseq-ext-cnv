#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
from scipy.special import  gammaln
from scipy import stats
from scipy.stats import geom
import seaborn as sns
from pathlib import Path
from scipy.optimize import minimize
from itertools import cycle, islice

def preprocess(filepath:str):
    df = pd.read_csv(filepath, delimiter = '\t',header = 0, index_col = 0, low_memory=False)
    #Get rid of redundant counts of coverage from the input file
    df["unique_cov"] = df["unique_top_cov"]+df["unique_bot_cov"]
    df["redundant"] = df['redundant_top_cov']+df['redundant_bot_cov']
    df = df[df['redundant'] == 0]
    genome = df["ref_base"].to_numpy()
    genome_len = len(df)
    #Total GC content of the genome
    n_g = np.count_nonzero(genome==('G'))
    n_c = np.count_nonzero(genome==('C'))
    t_gcp = ((n_g+n_c)/genome_len)*100
    winseq = []
    seq = []
    gcp_s = []
    window = []
    window_med_cov = []
    gc_df = pd.DataFrame(columns = ("window","sequence"))
    win_len = 1000
    win_shift = 500
    i = 0
    #inspect the genome contents through a sliding window of 1000bp
    while(i <= (genome_len - win_len)):
        winseq = genome[i:(i + win_len)]
        seq.insert(i,str(''.join(str(element) for element in winseq)))
        gcc = np.count_nonzero(winseq == 'G') + np.count_nonzero(winseq == 'C')
        gccp = (gcc/win_len)*100
        gcp_s.insert(i,gccp)
        window_med_cov.insert(i,float(df.iloc[i:(i + win_len),9].median()))
        window.insert(i,i + win_len)
        i = i + win_shift

    #Save the window median and 
    gc_df["gc_percent"] = gcp_s
    gc_df["sequence"] = seq
    gc_df["read_count_cov"] = window_med_cov
    gc_df["window"] = window

    gc_df["norm_raw_cov"] = gc_df["read_count_cov"]/gc_df["read_count_cov"].median()

    return gc_df


# In[10]:


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def gc_normalization(df):

    # df, ori, ter = preprocess(filepath)
    windos = df["window"]
    
    cov = df["norm_raw_cov"]
    gc = df["gc_percent"]
    
    lowess1 = sm.nonparametric.lowess
    gc_out = lowess1(cov, gc,frac=0.8, it=3, delta=0.0, is_sorted=False, missing='none', return_sorted=False)    

    df["gc_corr_norm_cov"] = cov/gc_out
    
    # gc_cor_med_fil = ndimage.median_filter(df["gc_cor"],size=200,mode="reflect")
    
    med = df["read_count_cov"].median()
    cor_rc = []
    
    for i in range(len(df)):
        cor_rc.insert(i, int (df["gc_corr_norm_cov"].iloc[i] * med))
    df["gc_corr_fact"] = gc_out
    
    return df


def fit_func(params, x, y, x3):
    x1, x2, y1, y2, = params
    
    y_circ = list(islice(cycle(y), int(x2), int(x2)+x3))
    
    x1 = x1 - x2
    x2 -= x2
    
    m1 = (y1-y2) / (x1-x2)
    m2 = (y2-y1) / (x3-x1)
    c1 = y1 - m1 * (x1)
    c2 = y1 - m2 * (x1)
    
    error = 0
    for i in range(int(x1)):
        y1_pred = m1*x[i] + c1
        error += (y_circ[i]-y1_pred) ** 2
    for i in range(int(x1),len(y)):
        y2_pred = m2*x[i] + c2
        error += (y_circ[i]-y2_pred) ** 2
    return error


def otr_fit(df, ter_idx, ori_idx):
    
    df_filt = df[np.abs(stats.zscore(df["gc_corr_norm_cov"])) < 3 ]
    
    x_init = df.index
    y_init = df["gc_corr_norm_cov"]
    x = df_filt.index
    y = df_filt["gc_corr_norm_cov"]
    
    x3_const = len(y_init)
    
    xori_guess = df_filt["gc_corr_norm_cov"].iloc[int(ori_idx*0.7):int(ori_idx*1.3)].idxmax()
    xter_guess = df_filt["gc_corr_norm_cov"].iloc[int(ter_idx*0.7):int(ter_idx*1.3)].idxmin()    
    yori_guess = np.max(y)
    yter_guess = np.min(y)  

    initial_guess = [xori_guess, xter_guess, yori_guess, yter_guess]
    
    result = minimize(fit_func, initial_guess, args = (x, y, x3_const))

    xori_opt, xter_opt, yori_opt, yter_opt = result.x

    m1_opt = (yori_opt-yter_opt) / (xori_opt-xter_opt)
    m2_opt = (yter_opt-yori_opt) / (len(x_init)-(xori_opt-xter_opt))
    
    c1_opt = yori_opt - m1_opt * (xori_opt-xter_opt)
    c2_opt = yori_opt - m2_opt * (xori_opt-xter_opt)
    
    y1_fit=[]
    y2_fit=[]

    y1_fit = [m1_opt * x_init + c1_opt for x_init in range(0,int(xori_opt)-int(xter_opt))]
    y2_fit = [m2_opt * x_init + c2_opt for x_init in range(int(xori_opt)-int(xter_opt),len(x_init))]
    
    y_fit = y1_fit + y2_fit
    y_fit = np.array(list(islice(cycle(y_fit), len(y_fit)-int(xter_opt), 2*len(y_fit)-int(xter_opt))))
    
    y_corr = y_init / y_fit
    
    # print(f'xori_opt: {xori_opt}, xter_opt: {xter_opt}, yori_opt: {yori_opt}, yter_opt: {yter_opt}')
    # print(f'm1_opt: {m1_opt}, m2_opt: {m2_opt}, c1_opt: {c1_opt}, c2_opt: {c2_opt}')
    
    return y_corr, y_fit

    
def otr_correction(filepath, ori, ter):

    df = gc_normalization(filepath)
    windows = df["window"]
    x1, x2 = find_nearest(windows,ter) , find_nearest(windows,ori)
    
    corr = []
    
    h1, f1 = otr_fit(df, x1, x2)
    
    df["otr_gc_corr_norm_cov"] = h1
    df["otr_gc_corr_fact"] = f1   
    
    return df




def solve_pr(mean, variance):
    r = (mean * mean)/(variance - mean)
    p = 1 - (mean/variance)
    return p, r

def calculate_prob(p, r, obs):
    probs = np.exp(gammaln(r + obs) - gammaln(obs + 1) - gammaln(r) + obs * np.log(p) + r * np.log(1 - p))
    return probs

def setup_emission_matrix(n_states, mean, variance, absmax, include_zero_state=True, error_rate=0):
    emission = np.full((n_states, absmax + 1), np.nan)
    
    for state in range(n_states):
        pr = solve_pr(mean * (state + 1), variance * (state + 1))
        p, r = pr[0], pr[1]
        
        for obs in range(absmax + 1):
            # Ensure obs is converted to a float or NumPy array before applying np.log
            emission[state, obs] = calculate_prob(p, r, obs)
    
    if include_zero_state:
        zero_row = np.array([geom.pmf(i, 1 - error_rate, loc =-1) for i in range(absmax + 1)])
        emission = np.vstack((zero_row, emission))
    # np.savetxt("emission.csv", emission, delimiter=",")  
    return emission


# In[21]:


def setup_transition_matrix(n_states, remain_prob, include_zero_state=True):
    if include_zero_state:
        n_states += 1
    
    change_prob = 1 - remain_prob
    per_state_prob = change_prob / (n_states - 1)
    
    transition = np.full((n_states, n_states), per_state_prob)
    
    for i in range(n_states):
        transition[i, i] = remain_prob
    # np.savetxt("transition.csv", transition, delimiter=",") 
    return transition


# In[93]:


def make_viterbi_mat(obs, transition_matrix, emission_matrix, include_zero_state):
    num_states = transition_matrix.shape[0]
    
    # Create a mask for the zero values
    mask = (emission_matrix == 0)
    # Take the logarithm of the non-zero values
    logemi = np.zeros_like(emission_matrix, dtype=float)
    logemi[~mask] = np.log(emission_matrix[~mask])

    # Handle the zero values separately, set to -inf
    logemi[mask] = -np.inf 

    logv = np.full((len(obs), num_states), np.nan)
    logtrans = np.log(transition_matrix)
    # logemi = np.log(emission_matrix)
    
    logv[0,:] = np.log(1.0e-10)
    
    if include_zero_state:
        logv[0, 1] = np.log(1)
    else:
        logv[0, 0] = np.log(1)
    
    for i in range(1, len(obs)):
        for l in range(num_states):
            statelprobcounti = logemi[l, obs[i]]
            maxstate = max(logv[i - 1, :] + logtrans[l, :])
            logv[i, l] = statelprobcounti + maxstate
    # np.savetxt("viterbi.csv", logv, delimiter = ',')
    return logv


# In[88]:


def HMM_copy_number(obs, transition_matrix, emission_matrix, include_zero_state, window_length, chr_length):
    states = np.arange(0, emission_matrix.shape[0] + 1)  # Assuming state indices start from 1
    
    # print("Attempting to create Viterbi matrix")
    
    v = make_viterbi_mat(obs, transition_matrix, emission_matrix, include_zero_state)
            
    # Go through each of the rows of the matrix v and find out which column has the maximum value for that row
    most_probable_state_path = np.argmax(v, axis=1)
    # np.savetxt('MPSS.csv', most_probable_state_path, delimiter = ',')
    results = pd.DataFrame(columns=['Startpos', 'Endpos', 'State'])
    
    prev_obs = obs[0]
    prev_most_probable_state = most_probable_state_path[0]
    prev_most_probable_state_name = states[prev_most_probable_state]  # Adjust for 0-based indexing
    start_pos = 1
    
    for i in range(1, len(obs)):
        observation = obs[i]
        most_probable_state = most_probable_state_path[i]
        most_probable_state_name = states[most_probable_state]  # Adjust for 0-based indexing
        
        if most_probable_state_name != prev_most_probable_state_name:
            # print(f"Positions {start_pos} - {i * window_length}: Most probable state = {prev_most_probable_state_name}")
            results = results._append({'Startpos': start_pos, 'Endpos': i * window_length, 
                                      'State': prev_most_probable_state_name}, ignore_index=True)
            start_pos = i * window_length + 1
        
        prev_obs = observation
        prev_most_probable_state_name = most_probable_state_name
    
    # print(f"Positions {start_pos} - {chr_length}: Most probable state = {prev_most_probable_state_name}")
    results = results._append({'Startpos': start_pos, 'Endpos': chr_length, 
                              'State': prev_most_probable_state_name}, ignore_index=True)
    
    return results



def run_HMM(sample, output, ori, ter,n_states=5, changeprob=0.0001, include_zero_state=True, error_rate=0.0000001):
    
    filepath = sample
    sample = sample.strip().split('/')[-1]
    samplename = sample.strip().split('.')[0]
    
    # print("Running HMM...")
    
    df = bias_correction(filepath,ori,ter)
    
    med = df["read_count_cov"].median()
    cor_rc = []
    for i in range(len(df)):
        cor_rc.insert(i, int (df["otr_gc_corr_norm_cov"].iloc[i] * med))
    
    read_counts = cor_rc

    mean = np.mean(df["read_count_cov"])
    var = np.var(df["read_count_cov"])
    
    med = df["read_count_cov"].median()
    cor_rc = []
    for i in range(len(df)):
        cor_rc.insert(i, int (df["otr_gc_corr_norm_cov"].iloc[i] * med))
    
    df["otr_gc_corr_rdcnt_cov"] = cor_rc
    
    new_exp = df.copy()
    
    rc_max = np.max(read_counts)
    
    this_emission = setup_emission_matrix(n_states=n_states, mean=mean, variance=var, absmax=rc_max, 
                                          include_zero_state=include_zero_state , error_rate=error_rate)
    this_transition = setup_transition_matrix(n_states, remain_prob=(1 - changeprob), include_zero_state=include_zero_state)
    
    # print("Finished setting up transition and emission matrices. Starting Viterbi algorithm...")
    copy_numbers = HMM_copy_number(read_counts, this_transition, this_emission, 
                                   include_zero_state, 500, df['window'].max())
    

    # print("Finished running Viterbi algorithm. Assigning most probable states to individual segments...")
    
    CN_HMM = []
    
    for cnrow in range(len(copy_numbers)):
        state = int(copy_numbers['State'][cnrow])
        hmmstart = int(copy_numbers['Startpos'][cnrow])
        hmmend = int(copy_numbers['Endpos'][cnrow])
        
        CN_HMM_row = []
        idx = 0
        for win in df['window']:
            if (win >= hmmstart) and (win <= hmmend):
                CN_HMM_row.append(state)
                idx+=1
        
        CN_HMM.extend(CN_HMM_row)
    
    new_exp['prob_copy_number'] = CN_HMM
    
    saveplt = str('./'+output+"/CNV_plt/")
    saveloc = str('./'+output+"/CNV_csv/")

    csv_full_path = os.path.join(saveloc,'%s_CNV.csv' % samplename)
    brk_full_path = os.path.join(saveloc,'%s_break_pts.csv' % samplename)

    new_exp.reset_index(drop = True)
    new_exp.to_csv(csv_full_path)
    copy_numbers.to_csv(brk_full_path)

    print(f"{sample}: Copy number prediction complete. .csv files saved.")
    return new_exp, sample


# In[14]:


def bias_correction(filepath,ori,ter):
    sample = filepath.strip().split('/')[-1]
    print(f'{sample}: Calculating coverage and GC% across sliding window over the genome.')
    df = preprocess(filepath)
    gc_corr = gc_normalization(df)
    print(f'{sample}: Corrected GC bias in coverage.')
    otr_corr = otr_correction(gc_corr, ori, ter)
    print(f'{sample}: Corrected origin/terminus of replication(OTR) bias in coverage.')
    return otr_corr


def gc_cor_plots(df, sample, output):
    
    sample = sample.strip().split('/')[-1]
    samplename = sample.strip().split('.')[0]
    
    saveplt = str('./'+output+"/GC_bias/")
    
    plt.figure(figsize=(10, 8))
    
    
    plt.scatter(df['gc_percent'], df['norm_raw_cov'], color='lightblue', label='Raw normalized reads vs GC', s=5)
    plt.scatter(df["gc_percent"], df["gc_corr_norm_cov"], color="darkblue", label='Corrected normalized reads', s=10, alpha = 0.7)
    
    # Adding labels and title
    plt.ylabel('Normalized read coverage')
    plt.xlabel('GC% per window')
    
    plt.title(f'{samplename}_GCvsNormalizedReads')
    plt.legend()

    plt_full_path =os.path.join(saveplt,'%s_GC_vs_NormRds.png' % samplename.replace(' ', '_'))
    plt.savefig(plt_full_path, format = 'png', bbox_inches = 'tight')
    
    plt.close()


# In[25]:


# def plot_gc_cor(df, samplename):
    
#     # df, x1, x2 = gc_normalization(filepath)
    
#     # dir_split = filepath.strip().split('/')[:-1]
#     saveplt = str('./'+"/CNV_out/GC_plt/")
    
#     ter=df["window"].iloc[x1]
#     ori=df["window"].iloc[x2]

        
#     plt.figure(figsize=(10, 8))
    
    
#     # plt.scatter(df['window'], df['norm_gc'], color='brown', label='GC%_normalized_reads', s=3)
#     plt.scatter(df["window"], df["norm_raw_cov"], color="gray", label='Raw reads', s=10, alpha =0.7)
#     plt.scatter(df['window'], df['gc_corr_norm_cov'], color='black', label='GC-corrected', s=5)
    
    
#     # plt.plot(df['window'], df['gc_cor_med_fil'], color='gray')
#     plt.axvline(x=ter, color='r', linestyle=':', label=f'Terminus: {ter}')
#     plt.axvline(x=ori, color='r', linestyle=':', label=f'Origin: {ori}')
#     # Adding labels and title
#     plt.xlabel('Window (Genomic Position)')
#     plt.ylabel('Normalized reads / GC%')
    
#     plt.title(f'{samplename}_GC bias correction')
#     plt.legend()

#     plt_full_path =os.path.join(saveplt,'%s_GC_corr.png' % samplename.replace(' ', '_'))
#     plt.savefig(plt_full_path, format = 'png', bbox_inches = 'tight')
#     plt.close()


def plot_otr_corr(df, sample, output, ori, ter):

    sample = sample.strip().split('/')[-1]
    samplename = sample.strip().split('.')[0]
    saveplt = str('./'+output+"/OTR_corr/")
  

    plt.figure(figsize=(10, 8))
    plt.scatter(df["window"],df["norm_raw_cov"], color="gray", label="Raw reads",s=8)
    plt.scatter(df["window"],df["gc_corr_norm_cov"], color="brown", label="GC corrected", s=5, alpha = 0.6)
    plt.scatter(df["window"],df["otr_gc_corr_norm_cov"], color = 'black', label="Ori/Ter bias corrected", s = 10)
    plt.plot(df["window"], df["otr_gc_corr_fact"], color = "white", label = "OTR-bias-fit-line")
    
    plt.axvline(x=ter, color='r', linestyle=':', label=f'Terminus: {ter}')
    plt.axvline(x=ori, color='r', linestyle=':', label=f'Origin: {ori}')
    plt.xlabel("Window (Genomic position)")
    plt.ylabel("Normalized read coverage")
    plt.title(f'{samplename}_Ori/Ter bias correction')
    plt.legend()

    plt_full_path = os.path.join(saveplt,'%s_OTR_corr.png' % samplename.replace(' ', '_'))
    
    plt.savefig(plt_full_path, format = 'png', bbox_inches = 'tight')
    df.reset_index(drop = True)
    
    plt.close()


def plot_copy(df_cnv, sample, output):
    
    sample = sample.strip().split('/')[-1]
    samplename = sample.strip().split('.')[0]
    saveplt = str('./'+output+"/CNV_plt/")
    
    plt.figure(figsize=(10, 8))

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()

    ax1.scatter(df_cnv["window"],df_cnv["read_count_cov"], color="gray", label="Raw reads",s=10)
    ax1.scatter(df_cnv["window"],df_cnv["otr_gc_corr_rdcnt_cov"], color="pink", label="Corrected reads",s=5)
    ax2.scatter(df_cnv["window"],df_cnv["prob_copy_number"], color="red", label="Predicted Copy Number", s=5)
    
    delta = (df_cnv['read_count_cov'].median()*0.4)
    
    
    ax1.set_ylim(df_cnv['read_count_cov'].min() - delta, df_cnv['read_count_cov'].max() + delta)
    ax2.set_ylim(df_cnv['norm_raw_cov'].min() - 0.4, df_cnv['norm_raw_cov'].max() + 0.4)
    
    ax1.set_xlabel("Window (Genomic position)")
    ax2.yaxis.label.set_color('red')
    ax1.set_ylabel("Read Counts")
    ax2.set_ylabel("Copy Numbers")
    plt.title(f'{samplename}_Copy Number Prediction')
    
    ax1.legend()
    # ax2.legend()
    
    plt_full_path = os.path.join(saveplt,'%s_copy_numbers.png' % samplename)
    plt.savefig(plt_full_path, format = 'png', bbox_inches = 'tight')
    
    plt.close()


# In[28]:


def main():
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(description = "Input .tab file from breseq bam2cov: \n \n```\nbreseq bam2cov -t[--table] --resolution 0 (0=single base resolution) --region <reference:START-END> --output <filename> \n```" , formatter_class = RawTextHelpFormatter)

    # Define the command line arguments
    parser.add_argument(
        "-i",
        "--input",
        action="store",
        dest="i",
        required=True,
        type=str,
        help="input .tab file address from breseq bam2cov",
    )
    
    parser.add_argument(
        "-o",
        "--output",
        action="store",
        dest="o",
        required=False,
        default='CNV_out',
        type=str,
        help="output file location preference. Defaults to the current folder",
    )
    parser.add_argument(
        "-ori",
        "--origin",
        action="store",
        dest="ori",
        default=3886082,
        required=False,
        type=int,
        help="Genomic coordinate for origin of replication",
    )
    parser.add_argument(
        "-ter",
        "--terminus",
        action="store",
        dest="ter",
        default=1567362,
        required=False,
        type=int,
        help="Genomic coordinate for terminus of replication",
    )
    
    # Parse the command line arguments
    options = parser.parse_args()
    
    out_dir = options.o
    
    out_subdirs = ['/CNV_plt' , '/CNV_csv', '/GC_bias', '/OTR_corr']
    for i in range(len(out_subdirs)):
        Path(out_dir+out_subdirs[i]).mkdir(parents=True, exist_ok=True)

    # Call the copy number (HMM) function with the provided options
    cnv,smpl = run_HMM(
        sample=options.i,
        output = options.o,
        ori=options.ori,
        ter=options.ter,
    )
    
    #Call the plotting functions to visualize bias correction and copy number predictions
    gc_cor_plots(cnv, sample=options.i, output=options.o)
    print(f'{smpl}: GC bias vs coverage plots saved.')
    plot_otr_corr(cnv, sample=options.i, output=options.o, ori=options.ori, ter=options.ter)
    print(f'{smpl}: OTR bias vs coverage plots saved.')
    plot_copy(cnv, sample=options.i, output=options.o)
    print(f'{smpl}: CNV prediction plots saved.')


if __name__ == "__main__":
    main()
