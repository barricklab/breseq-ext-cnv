#!/usr/bin/env python
# coding: utf-8

import os
import json
import pandas as pd
import numpy as np
import matplotlib as mplt 
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

def preprocess(filepath:str, win=200, step=100, frag=350):

    if (step > win) :
        return print(f'window size: {win} is smaller than step size: {step}. Excluding segments of the genome for analysis.')

    df_b2c = pd.read_csv(filepath ,delimiter = '\t',engine='python', header = 0, index_col = 0, skipfooter = 4 )
    df_b2c["unique_cov"] = df_b2c["unique_top_cov"]+df_b2c["unique_bot_cov"]
    df_b2c["redundant"] = df_b2c['redundant_top_cov']+df_b2c['redundant_bot_cov']
    
    med_gen_cov = df_b2c["unique_cov"].median()
    cov = df_b2c["unique_cov"]
    genome = df_b2c['ref_base']
    genome_len = len(genome)
    genome_cyc = list(islice(cycle(genome), int(genome_len*0.75), genome_len+int(genome_len*1.25)))
    g_num = np.count_nonzero(genome == 'G')
    c_num = np.count_nonzero(genome == 'C')
    gen_gcp = (g_num + c_num)*100/genome_len
    fragseq = []
    fragment = []
    winseq = []
    seq = []
    gcp_s = []
    window = []
    win_end = []
    window_med_cov = []
    win_cov_type = []
    
    df_b2c["cov_type"] = df_b2c["redundant"].apply(lambda x: 'R' if x > 0 else 'U')
    
    df_gc = pd.DataFrame(columns=["window_num"])
    
    cov_dict = {}
    i=0
    lst_win = 0

    #sliding window = win and increment size = step summarizes GC% and median coverage
    while (i <= (genome_len-1)) and (lst_win < genome_len):
        
        win_full_cov = df_b2c["unique_cov"].iloc[i : (i+win)].to_numpy()
        cov_type = df_b2c["cov_type"].iloc[i : (i+win)].to_numpy()
        win_cov = []

        # Filter the windows overlapping redundant coverage regions. Ignores any coverage in and adjacent to repititive/transposable changes
        
        winu = 0
        for j in range(len(cov_type)):
            if (cov_type[j] == 'U'):
                win_cov.insert(j, float(win_full_cov[j]))
                winu+=1
            else:
                break

        if (winu < win):

            i = i + step

        else:
            window_med_cov.insert(i,float(np.nanmedian(win_cov)))
            winseq = genome[i:i+winu]
            seq.insert(i,str(''.join(str(element) for element in winseq)))
            window.insert(i, i)
            win_end.insert(i,i+winu)
            lst_win = win_end[(len(win_end)-1)]
            i_off = i+int(genome_len*0.25)
            
            if (frag>win):
                diff = int((frag-win)/2)
                fragseq = genome_cyc[(i_off-diff):((i_off + win)+diff)]
                fragment.insert(i,str(''.join(str(element) for element in fragseq)))
                gcc = ''.join([nucleotide for nucleotide in fragseq if nucleotide in ['C', 'G']])
                gccp = (len(gcc)/len(fragseq))
                gcp_s.insert(i,gccp)
            else:
                diff = int((win-frag)/2)
                fragseq = list(genome_cyc[i_off-diff:(i_off + win)+diff])
                fragment.insert(i,str(''.join(str(element) for element in fragseq)))
                gcc = ''.join([nucleotide for nucleotide in fragseq if nucleotide in ['C', 'G']])
                gccp = (len(gcc)/len(fragseq))
                gcp_s.insert(i,gccp)

            i = i + step

    #Save the window median and GC% per fragment overlapping a window to the dataframe

    df_gc["win_st"] = window
    df_gc["win_end"] = win_end
    df_gc["win_len"] = df_gc["win_end"] - df_gc["win_st"]
    df_gc["gc_percent"] = gcp_s
    df_gc["read_count_cov"] = window_med_cov
    df_gc["window_num"] = np.arange(0,len(window_med_cov),1)
    df_gc["norm_raw_cov"] = df_gc["read_count_cov"]/df_gc["read_count_cov"].median()
    
    return df_gc


def gc_normalization(df):
    # Corrects trends between GC% and coverage in windows using locally weighted regression model

    cov = df["norm_raw_cov"]
    gc = df["gc_percent"]
    med = df["norm_raw_cov"].median()

    loess = sm.nonparametric.lowess
    gc_out = loess(cov, gc, frac=0.05, it=1, delta=0.0, is_sorted=False, missing='none', return_sorted=False)    

    gc_corr = cov/gc_out

    df["gc_corr_norm_cov"] = gc_corr
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
    for i in range(int(x1)-1):
        y1_pred = m1*x[i] + c1
        error += (y_circ[i]-y1_pred) ** 2
    for i in range(int(x1),len(y)):
        y2_pred = m2*x[i] + c2
        error += (y_circ[i]-y2_pred) ** 2
    return error


def otr_fit(df):
    
    df_filt = df[np.abs(stats.zscore(df["gc_corr_norm_cov"])) < 3 ]
    
    x_init = df.index
    y_init = df["gc_corr_norm_cov"]
    x = df_filt.index
    y = df_filt["gc_corr_norm_cov"]
    
    x3_const = len(y_init)     
        
    xori_guess = df_filt["gc_corr_norm_cov"].iloc[int(x3_const*0.55):int(x3_const)].idxmax()
    xter_guess = df_filt["gc_corr_norm_cov"].iloc[int(x3_const*0):int(x3_const*0.50)].idxmin()
    if (x3_const*0.45 <= (xori_guess-xter_guess) <= x3_const*0.55):
        pass
    else:
        xori_guess = x3_const * 0.999
        xter_guess = x3_const * 0.001

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
    
    return y_corr, y_fit, int(xori_opt), int(xter_opt)

def set_func(params, x, y, x1, x2, x3):
    y1, y2, = params
    
    y_circ = list(islice(cycle(y), int(x2), int(x2)+x3))
    
    x1 = x1 - x2
    x2 -= x2
    
    m1 = (y1-y2) / (x1-x2)
    m2 = (y2-y1) / (x3-x1)
    c1 = y1 - m1 * (x1)
    c2 = y1 - m2 * (x1)
    
    error = 0
    for i in range(int(x1)-1):
        y1_pred = m1*x[i] + c1
        error += (y_circ[i]-y1_pred) ** 2
    for i in range(int(x1),len(y)):
        y2_pred = m2*x[i] + c2
        error += (y_circ[i]-y2_pred) ** 2
    
    return error


def otr_set(df, ter_idx, ori_idx):
    
    df_filt = df[np.abs(stats.zscore(df["gc_corr_norm_cov"])) < 3 ]
    
    x_init = df.index
    y_init = df["gc_corr_norm_cov"]
    x = df_filt.index
    y = df_filt["gc_corr_norm_cov"]
    
    # ter_idx, ori_idx  = find_nearest(windows,ter) , find_nearest(windows,ori)
    x3_const = len(y_init)

    xori_guess = ori_idx
    xter_guess = ter_idx

    yori_guess = df_filt["gc_corr_norm_cov"].iloc[ori_idx]
    yter_guess = df_filt["gc_corr_norm_cov"].iloc[ter_idx]    

    initial_guess = [yori_guess, yter_guess]
    
    result = minimize(set_func, initial_guess, args = (x, y, xori_guess, xter_guess, x3_const))
    yori_opt, yter_opt = result.x

    m1_opt = (yori_opt-yter_opt) / (xori_guess-xter_guess)
    m2_opt = (yter_opt-yori_opt) / (len(x_init)-(xori_guess-xter_guess))
    
    c1_opt = yori_opt - m1_opt * (xori_guess-xter_guess)
    c2_opt = yori_opt - m2_opt * (xori_guess-xter_guess)
    
    y1_fit=[]
    y2_fit=[]

    y1_fit = [m1_opt * x_init + c1_opt for x_init in range(0,int(xori_guess)-int(xter_guess))]
    y2_fit = [m2_opt * x_init + c2_opt for x_init in range(int(xori_guess)-int(xter_guess),len(x_init))]
    y_fit = y1_fit + y2_fit

    y_fit = np.array(list(islice(cycle(y_fit), len(y_fit)-int(xter_guess), 2*len(y_fit)-int(xter_guess))))
    y_corr = y_init / y_fit

    return y_corr, y_fit

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def otr_correction(filepath, ori, ter, enforce):

    df = gc_normalization(filepath)
    windows = df["win_end"]
    
    corr = []
    # enforces user set genomic co-ordinates of ori/ter to check and fit the bias curve.
    if (enforce == True):
        x1, x2 = find_nearest(windows,ter) , find_nearest(windows,ori)
        h1, f1 = otr_set(df, x1, x2)
        yori = f1[x2]
        yter = f1[x1]
        OTR = yori / yter
        results = {"Origin location":ori, "Origin coverage (normalized)":yori,"Terminus window":ter, "Terminus coverage (normalized)":yter, "Origin-to-Termius/Bias Ratio":OTR, "Correction type" : "Ori-ter defined by user" }
        df["otr_gc_corr_norm_cov"] = h1
        df["otr_gc_corr_fact"] = f1
        
        return df, results, ori, ter 
    
    else:
    # fits the bias curve to the most probable location of ori/ter based on coverage peak and troughs respectively
        h1, f1 , ori_idx, ter_idx = otr_fit(df)
        
        xori = df["win_st"].iloc[ori_idx]
        xter = df["win_end"].iloc[ter_idx]
        yori = f1[ori_idx]
        yter = f1[ter_idx]
        OTR = yori / yter
        results = {"Origin window":int(xori), "Origin coverage (normalized)":yori, "Terminus window":int(xter), "Terminus coverage (normalized)":yter, "Origin-to-Termius/Bias Ratio":OTR, "Correction type" : "Ori-ter coordinates fit by coverage" }
        df["otr_gc_corr_norm_cov"] = h1
        df["otr_gc_corr_fact"] = f1 
        
        return df, results, xori, xter




def solve_pr(mean, variance):
    r = (mean * mean)/(variance - mean)
    p = 1 - (mean/variance)
    return p, r

def calculate_prob(p, r, obs):
    # probabilities calculated by assuming negative binomial distribution (Poisson family), to account for 
    # a wide dispersion of coverage data points given noisy data and high copy number possibilities (amplifications)
    # gammaln function allows for calculation of log probabilities without computational over-flow. 
    
    probs = np.exp(gammaln(r + obs) - gammaln(obs + 1) - gammaln(r) + obs * np.log(p) + r * np.log(1 - p))
    return probs

def setup_emission_matrix(n_states, mean, variance, absmax, error_rate):
    emission = np.full((n_states, absmax + 1), np.nan)
    
    for state in range(n_states):
        pr = solve_pr(mean * (state + 1), variance * (state + 1))
        p, r = pr[0], pr[1]
        
        for obs in range(absmax + 1):
            emission[state, obs] = calculate_prob(p, r, obs)
    
    # probability mass function determines the lower prbability of predicting zero as the readcounts approach absmax.
    # error rate offsets the probability threshold of predicting zero at higher readcounts accounting for the erronous read alignments
    zero_row = np.array([geom.pmf(i - 1, 1 - error_rate) for i in range(absmax + 1)])
    emission = np.vstack((zero_row, emission))
    # np.savetxt("emission.csv", emission, delimiter=",")  
    return emission


def setup_transition_matrix(n_states, remain_prob):
    #include zero state:
    n_states += 1
    
    change_prob = 1 - remain_prob
    per_state_prob = change_prob / (n_states - 1)
    
    transition = np.full((n_states, n_states), per_state_prob)
    
    for i in range(n_states):
        transition[i, i] = remain_prob
    # np.savetxt("transition.csv", transition, delimiter=",") 
    return transition


# In[93]:


def make_viterbi_mat(obs, transition_matrix, emission_matrix):
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
    
    logv[0,:] = -np.inf
    
    #start prob of state =1 when including zero state

    logv[0, 1] = np.log(1e-100)
    
    for i in range(1, len(obs)):
        for l in range(num_states):
            statelprobcounti = logemi[l, obs[i]]
            maxstate = max(logv[i - 1, :] + logtrans[l, :])
            logv[i, l] = statelprobcounti + maxstate
    # np.savetxt("viterbi.csv", logv, delimiter = ',')
    return logv


def HMM_copy_number(obs, transition_matrix, emission_matrix, win_st, win_end, chr_length):
    states = np.arange(0, emission_matrix.shape[0] + 1)  # Assuming state indices start from 1
    
    v = make_viterbi_mat(obs, transition_matrix, emission_matrix)
            
    # Go through each of the rows of the matrix v and find out which column has the maximum value for that row
    most_probable_state_path = np.argmax(v, axis=1)
    results = pd.DataFrame(columns=['Startpos', 'Endpos', 'State'])
    
    prev_obs = obs[0]
    prev_most_probable_state = most_probable_state_path[0]
    prev_most_probable_state_name = states[prev_most_probable_state]  # Adjust for 0-based indexing
    start_pos = 0

    
    for i in range(0, len(obs)-1):

        observation = obs[i]
        most_probable_state = most_probable_state_path[i]
        most_probable_state_name = states[most_probable_state]  # Adjust for 0-based indexing
        
        if most_probable_state_name != prev_most_probable_state_name:
            results = results._append({'Startpos': start_pos, 'Endpos': win_end[i-1], 
                                      'State': prev_most_probable_state_name}, ignore_index=True)
            start_pos = win_st[i]
        
        prev_obs = observation
        prev_most_probable_state_name = most_probable_state_name

    results = results._append({'Startpos': start_pos, 'Endpos': chr_length, 
                              'State': prev_most_probable_state_name}, ignore_index=True)
    
    return results



def run_HMM(sample, output, ori, ter, enforce, win, step, frag, error_rate=0.15, n_states=5, changeprob=(1e-10)):
    
    filepath = sample
    sample = sample.strip().split('/')[-1]
    samplename = sample.strip().split('.')[0]
    
    df, ori_loc, ter_loc = bias_correction(filepath, output, ori, ter, enforce, win, step, frag)
    
    cor_cov = df["otr_gc_corr_norm_cov"]

    med = df["read_count_cov"].median()
    cor_rc = []
    for i in range(len(df)):
        cor_rc.insert(i, int (df["otr_gc_corr_norm_cov"].iloc[i] * med))
    
    df['win_len'] = df["win_end"] - df["win_st"]
    
    mean = np.median(cor_rc)
    var = np.var(cor_rc)
    
    
    df["otr_gc_corr_rdcnt_cov"] = cor_rc
    
    # determines the number of expected states based on the normalized coverage range. Minimum number of expected states = 5
    n_states = int(df["otr_gc_corr_norm_cov"].max()) if int(df["otr_gc_corr_norm_cov"].max()) > 5 else 5

    new_exp = df.copy()
    
    rc_max = np.max(cor_rc)
    
    this_emission = setup_emission_matrix(n_states=n_states, mean=mean, variance=var, absmax=rc_max, error_rate=error_rate)
    this_transition = setup_transition_matrix(n_states, remain_prob=(1 - changeprob))
    
    # outputs consequtive segments of the genome where copy number changes based on HMM : Break points
    copy_numbers = HMM_copy_number(cor_rc, this_transition, this_emission, df["win_st"], df["win_end"], df['win_end'].max())
    
    CN_HMM = []
    # output copy number prediction for each instance of genomic sliding window
    for cnrow in range(len(copy_numbers)):

        state = int(copy_numbers['State'][cnrow])

        hmmstart = int(copy_numbers['Startpos'][cnrow])
        hmmend = int(copy_numbers['Endpos'][cnrow])
        
        CN_HMM_row = []
        idx_list = df[df["win_st"] == hmmstart].index

        if len(idx_list) == 0:
            continue  # skip if no matching start position found
    
        idx = idx_list[0]
        
        for idx in range(len(new_exp)):

            if ((df["win_st"].iloc[idx] >= hmmstart) and (df["win_end"].iloc[idx] <= hmmend)):
                CN_HMM_row.append(state)
                idx+=1

            else:
                continue
        
        CN_HMM.extend(CN_HMM_row)
    
    new_exp['prob_copy_number'] = CN_HMM

    
    saveplt = str(output+"/CNV_plt/")
    saveloc = str(output+"/CNV_csv/")

    csv_full_path = os.path.join(saveloc,'%s_CNV.csv' % samplename)
    brk_full_path = os.path.join(saveloc,'%s_break_pts.csv' % samplename)

    new_exp.reset_index(drop = True)
    new_exp.to_csv(csv_full_path, index = False)
    cn_brk = pd.DataFrame(copy_numbers,columns=['Startpos','Endpos', 'State'])
    cn_brk['Segment_Size'] = cn_brk['Endpos'] - cn_brk['Startpos']
    cn_brk.drop(columns='Endpos', inplace = True)
    cn_brk.to_csv(brk_full_path, index = False)

    print(f"{sample}: Copy number prediction complete. .csv files saved.")
    return new_exp, sample, ori_loc, ter_loc


def bias_correction(filepath, output, ori, ter, enforce, win, step, frag):

    # Function handles the bias correction functions and outputs to be fed into the HMM
    sample = filepath.strip().split('/')[-1]
    samplename = sample.strip().split('.')[0]
    saveplt = str(output+"/OTR_corr/")
    print(f'{sample}: Calculating coverage and GC% across sliding window over the genome.')
    df = preprocess(filepath, win, step, frag)
    gc_corr = gc_normalization(df)
    print(f'{sample}: Corrected GC bias in coverage.')
    otr_corr, otr_res, ori_win, ter_win = otr_correction(gc_corr, ori, ter, enforce)
    with open(saveplt+str(samplename)+'_otr_results.json', 'w') as f:
        json.dump(otr_res, f, indent = 4)
    print(f'{sample}: Corrected origin/terminus of replication(OTR) bias in coverage.')
    return otr_corr, ori_win, ter_win


def gc_cor_plots(df, sample, output):
    
    sample = sample.strip().split('/')[-1]
    samplename = sample.strip().split('.')[0]
    
    saveplt = str(output+"/GC_bias/")
    
    plt.figure(figsize=(10, 8))
    
    gc_fit = np.poly1d(np.polyfit(df['gc_percent'].unique(), df['gc_corr_fact'].unique(), 2))

    plt.scatter(df['gc_percent'], df['norm_raw_cov'], color='brown', label='Raw normalized reads vs GC', s=5)
    plt.scatter(df['gc_percent'], df['gc_corr_norm_cov'], color="green", label='Corrected normalized reads', s=10, alpha = 0.3)
    plt.plot(np.sort(df['gc_percent'].unique()), gc_fit(np.sort(df['gc_percent'].unique())), color = 'black', linewidth = 3, label = 'LOWESS fit')

    # Adding labels and title
    plt.ylabel('Normalized read coverage')
    plt.xlabel('GC% per window')
    
    plt.title(f'{samplename}_GCvsNormalizedReads')
    plt.legend(loc = 'upper right')

    plt_full_path =os.path.join(saveplt,'%s_GC_vs_NormRds.pdf' % samplename.replace(' ', '_'))
    plt.savefig(plt_full_path, format = 'pdf', bbox_inches = 'tight')
    
    plt.close()


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
    saveplt = str(output+"/OTR_corr/")
  

    plt.figure(figsize=(10, 8))
    plt.scatter(df["win_st"],df["norm_raw_cov"], color="gray", label="Raw reads",s=8, alpha = 0.3)
    plt.scatter(df["win_st"],df["gc_corr_norm_cov"], color="brown", label="GC corrected", marker = '*', s=15, alpha = 0.5)
    plt.scatter(df["win_st"],df["otr_gc_corr_norm_cov"], color = 'black', label="Ori/Ter bias corrected", s = 20, alpha = 0.1, 
                marker = mplt.markers.MarkerStyle(marker = 'o', fillstyle = 'none'))
    plt.plot(df["win_st"], df["otr_gc_corr_fact"], color = "white", label = "OTR-bias-fit-line")
    
    plt.axvline(x=ter, color='r', linestyle=':', label=f'Terminus: {ter}')
    plt.axvline(x=ori, color='r', linestyle=':', label=f'Origin: {ori}')
    plt.xlabel("Window (Genomic position)")
    plt.ylabel("Normalized read coverage")
    plt.title(f'{samplename}_Ori/Ter bias correction')
    plt.legend(loc = 'upper right')

    plt_full_path = os.path.join(saveplt,'%s_OTR_corr.pdf' % samplename.replace(' ', '_'))
    
    plt.savefig(plt_full_path, format = 'pdf', bbox_inches = 'tight')
    df.reset_index(drop = True)
    
    plt.close()


def plot_copy(df_cnv, sample, output):
    
    sample = sample.strip().split('/')[-1]
    samplename = sample.strip().split('.')[0]
    saveplt = str(output+"/CNV_plt/")
    
    plt.figure(figsize=(10, 8))

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()

    ax1.scatter(df_cnv["win_st"],df_cnv["read_count_cov"], color="gray", label="Raw reads",s=10)
    ax1.scatter(df_cnv["win_st"],df_cnv["otr_gc_corr_rdcnt_cov"], color="orange", label="Corrected reads",s=5, alpha = 0.5,
                marker = mplt.markers.MarkerStyle(marker = 'o', fillstyle = 'none'))
    ax2.scatter(df_cnv["win_st"],df_cnv["prob_copy_number"], color="red", label="Predicted Copy Number", s=3)
    
    delta = (df_cnv['read_count_cov'].median()*0.5)
    
    
    ax1.set_ylim(df_cnv['read_count_cov'].min() - delta, df_cnv['read_count_cov'].max() + delta)
    ax2.set_ylim(df_cnv['otr_gc_corr_norm_cov'].min() - 0.5, df_cnv['otr_gc_corr_norm_cov'].max() + 0.5)
    
    ax1.set_xlabel("Window (Genomic position)")
    ax2.yaxis.label.set_color('red')
    ax1.set_ylabel("Read Counts")
    ax2.set_ylabel("Copy Numbers")
    plt.title(f'{samplename}_Copy Number Prediction')
    
    ax1.legend(loc = 'upper right')
    # ax2.legend()
    
    plt_full_path = os.path.join(saveplt,'%s_copy_numbers.pdf' % samplename)
    plt.savefig(plt_full_path, format = 'pdf', bbox_inches = 'tight')
    
    plt.close()


def main():
    from argparse import RawTextHelpFormatter
    import textwrap

    parser = argparse.ArgumentParser(description = "The breseq-ext-cnv is python package extension to breseq that analyzes the sequencing coverage across the genome to determine specific regions that have undergone copy number variation (CNV)",
                                        epilog=textwrap.dedent('''\
                                        Input .tab file from breseq bam2cov. To get the coverage file run the command below in your breseq directory which contains the 'data' and 'output' folders. 
                                        ```
                                        breseq bam2cov -t[--table] --resolution 0 (0=single base resolution) --region <reference:START-END> --output <filename> 
                                        ```'''), 
                                        formatter_class = RawTextHelpFormatter)

    # Define the command line arguments
    parser.add_argument(
        "-i",
        "--input",
        action="store",
        dest="i",
        required=True,
        type=str,
        help="input .tab file address from breseq bam2cov.",
    )
    
    parser.add_argument(
        "-o",
        "--output",
        action="store",
        dest="o",
        required=False,
        default='CNV_out',
        type=str,
        help="output file location preference. Defaults to the current folder.",
    )

    parser.add_argument(
        "-w",
        "--window",
        action="store",
        dest="w",
        required=False,
        # default = 1000,
        type=int,
        help="Define window length to parse through the genome and calculate coverage and GC statistics.",
    )

    parser.add_argument(
        "-s",
        "--step-size",
        action="store",
        dest="s",
        required=False,
        # default = 500,
        type=int,
        help="Define step size (<= window size) for each progression of the window across the genome sequence. Set = window size if non-overlapping windows.",
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
        default=0.20,
        required=False,
        type=float,
        help="Error rate in sequencing read coverage, taken into account to accurately determine 0 copy coverage.",
    )
    
    # Parse the command line arguments
    options = parser.parse_args()
    
    out_dir = options.o
    
    out_subdirs = ['/CNV_plt' , '/CNV_csv', '/GC_bias', '/OTR_corr']
    for i in range(len(out_subdirs)):
        Path(out_dir+out_subdirs[i]).mkdir(parents=True, exist_ok=True)


    # Call the copy number (HMM) function with the provided options
    if options.ori and options.ter is not None:
        print("Ori has been set (value is %s)" % options.ori)
        print("Ter has been set (value is %s)" % options.ter)
        cnv,smpl, ori_win, ter_win = run_HMM(
            sample=options.i,
            output = options.o,
            ori=options.ori,
            ter=options.ter,
            enforce=True,
            win=options.w,
            step=options.s,
            frag=options.f,
            error_rate=options.e
        )
    else:
        options.ori=None,
        options.ter=None,
        print("Ori has not been set (default value is %s)" % options.ori)
        print("Ter has not been set (default value is %s)" % options.ter)
        cnv,smpl, ori_win, ter_win = run_HMM(
            sample=options.i,
            output = options.o,
            ori=options.ori,
            ter=options.ter,
            enforce=False,
            win=options.w,
            step=options.s,
            frag=options.f,
            error_rate=options.e
        )
    
    #Call the plotting functions to visualize bias correction and copy number predictions
    gc_cor_plots(cnv, sample=options.i, output=options.o)
    print(f'{smpl}: GC bias vs coverage plots saved.')
    plot_otr_corr(cnv, sample=options.i, output=options.o, ori=ori_win, ter=ter_win)
    print(f'{smpl}: OTR bias vs coverage plots saved.')
    plot_copy(cnv, sample=options.i, output=options.o)
    print(f'{smpl}: CNV prediction plots saved.')


if __name__ == "__main__":
    main()
