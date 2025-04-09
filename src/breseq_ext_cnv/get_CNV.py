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
from scipy import ndimage
from scipy.stats import geom
import seaborn as sns
from pathlib import Path
from scipy.optimize import minimize
from itertools import cycle, islice

def preprocess(filepath:str, start, end, win=200, step=100, frag=350):

    if (step > win) :
        return print(f'window size: {win} is smaller than step size: {step}. Excluding segments of the genome for analysis.')

    # Check if the variable is NoneType
    if start is None:
        start = 0
        if end is None:
            df_b2c = pd.read_csv(filepath ,delimiter = '\t',engine='python', header = 0, index_col = 0, skipfooter = 4 )        

    # Check if the variable is an int
    else: 
        df_b2c = pd.read_csv(filepath ,delimiter = '\t',engine='python', header = 0, index_col = 0, skipfooter = 4 )
        df_b2c = df_b2c.iloc[start:end]

    # df_b2c = pd.read_csv(filepath ,delimiter = '\t',engine='python', header = 0, index_col = 0, skipfooter = 4 )
    df_b2c["unique_cov"] = df_b2c["unique_top_cov"]+df_b2c["unique_bot_cov"]
    df_b2c["redundant"] = df_b2c['redundant_top_cov']+df_b2c['redundant_bot_cov']
    
    #start_coord = df_b2c["position"].iloc[0]
    start_coord = int(df_b2c.index[0])
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
            
            if (frag > win):
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
    # window += start_coord
    # win_end += start_coord
    df_gc["win_st"] = [x + start_coord for x in window]
    df_gc["win_end"] = [x + start_coord for x in win_end]
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
    df["gc_cor_med_fil"] = ndimage.median_filter(df["gc_corr_norm_cov"], size = int(len(df)/10), mode = "reflect")
    
    return df

def fit_func(params, x, y):
    x1, x2, y1, y2, = params
    
    m = (y1-y2) / (x1-x2)
    c = y1 - m * (x1)
    error = 0
    for i in range(len(x)):
        y_pred = m*x[i] + c
        error += (y[i]-y_pred) ** 2
    return error

def otr_fit(df):
    
    cyc = False
    bias = False
    pt = "trough"
    x = df.index
    y = df["gc_corr_norm_cov"]
    y_med_fil = df["gc_cor_med_fil"]
    
    len_init = len(x)
    
    x_cyc = list(islice(cycle(x), 0, len_init*3))
    y_cyc = list(islice(cycle(y), 0, len_init*3))
    
    xori_guess = y_med_fil.argmax()
    xter_guess = y_med_fil.argmin()
    yori_guess = y[y_med_fil.argmax()]
    yter_guess = y[y_med_fil.argmin()]
    
    if (abs(xori_guess-xter_guess) > len_init * 0.3 ):
        if (xori_guess < len_init*0.1) or (xori_guess > len_init * 0.9):
            xori_guess = 0
            y1 = y_cyc[:xter_guess]
            y2 = y_cyc[xter_guess:len_init]
            x1 = x_cyc[:xter_guess]
            x2 = x_cyc[xter_guess:len_init]
            initial_guess1 = [xori_guess, xter_guess, yori_guess, yter_guess]
            initial_guess2 = [xter_guess, xori_guess, yter_guess, yori_guess]
        elif (xter_guess < len_init*0.1) or (xter_guess > len_init * 0.9):
            xter_guess = 0
            y1 = y_cyc[:xori_guess]
            y2 = y_cyc[xori_guess:len_init]
            x1 = x_cyc[:xori_guess]
            x2 = x_cyc[xori_guess:len_init]
            initial_guess1 = [xori_guess, xter_guess, yori_guess, yter_guess]
            initial_guess2 = [len_init, xori_guess, yter_guess, yori_guess]
            pt = "peak"
        else:
            xi = xori_guess if (xori_guess < xter_guess) else xter_guess
            xj = xori_guess if (xori_guess > xter_guess) else xter_guess
            y1 = y_cyc[xi:xj]
            y2 = y_cyc[xj:(xi+len_init)]
            x1 = x_cyc[xi:xj]
            x2 = x_cyc[xj:(xi+len_init)]
            initial_guess1 = [xori_guess, xter_guess, yori_guess, yter_guess]
            initial_guess2 = [xter_guess, len_init, yter_guess, yori_guess]
            cyc = True
        bias = True
    else:
        y_corr = y
        y_fit = np.repeat(np.mean(y), len_init)
        print("OTR bias not detected")
        return y_corr, y_fit, xori_guess, xter_guess, bias
    
    result1 = minimize(fit_func, initial_guess1, args = (x1, y1))
    result2 = minimize(fit_func, initial_guess2, args = (x2, y2))
    
    xori_opt1, xter_opt1, yori_opt1, yter_opt1 = result1.x
    xter_opt2, xori_opt2, yter_opt2, yori_opt2 = result2.x
    
    xori_opt = np.mean([xori_opt1, xori_opt2])
    xter_opt = np.mean([xter_opt1, xter_opt2])
    yori_opt = np.mean([yori_opt1, yori_opt2])
    yter_opt = np.mean([yter_opt1, yter_opt2])
    
    y1_fit=[]
    y2_fit=[]

    if (yori_opt - yter_opt) >= 2*np.mean(y_med_fil):
        bias = True
    else:
        bias = False
    
    if bias and cyc:
        
        if (xori_opt > xter_opt):
            m_opt = (yori_opt - yter_opt) / (xori_opt - xter_opt)
            c_opt = yori_opt - m_opt * (xori_opt - xter_opt)
            m_opt1 = (yori_opt-yter_opt) / (xori_guess-xter_guess)
            m_opt2 = (yter_opt-yori_opt) / (len(x)-(xori_guess-xter_guess))
            
            c_opt1 = yori_opt - m_opt * (xori_opt-xter_opt)
            c_opt2 = yter_opt - m_opt * (xter_opt-xori_opt)
            
            y1_fit = [m_opt1 * x + c_opt1 for x in range(len(x1))]
            y2_fit = [m_opt2 * x + c_opt2 for x in range(len(x2))]
            y_fit = y1_fit + y2_fit
            y_fit = np.array(list(islice(cycle(y_fit), len(y_fit)-int(xter_guess), (2*len(y_fit) - int(xter_guess)))))
        else:
            m_opt = (yori_opt - yter_opt) / (xori_opt - xter_opt)
            c_opt = yori_opt - m_opt * (xori_opt - xter_opt)
            m_opt1 = (yori_opt-yter_opt) / (xori_guess-xter_guess)
            m_opt2 = (yter_opt-yori_opt) / (len(x)-(xori_guess-xter_guess))
            
            c_opt1 = yori_opt - m_opt * (xori_opt-xter_opt)
            c_opt2 = yter_opt - m_opt * (xter_opt-xori_opt)
            
            y1_fit = [m_opt1 * x + c_opt1 for x in range(len(x1))]
            y2_fit = [m_opt2 * x + c_opt2 for x in range(len(x2))]
            y_fit = y1_fit + y2_fit
            y_fit = np.array(list(islice(cycle(y_fit), len(y_fit)-int(xori_guess), (2*len(y_fit) - int(xori_guess)))))
        y_corr = y / y_fit
        
    elif bias and not cyc:
        if pt == "peak":
            xter_opt = 0
            
            m_opt1 = (yori_opt1 - yter_opt1) / (xori_opt1 - xter_opt1)
            m_opt2 = (yter_opt2 - yori_opt2) / (xter_opt2 - xori_opt2)

            c_opt1 = yori_opt2 - m_opt1 * (int(xori_opt1))
            c_opt2 = yori_opt2 - m_opt2 * (int(xori_opt2))
            
            y1_fit = [m_opt1 * x + c_opt1 for x in x1]
            y2_fit = [m_opt2 * x + c_opt2 for x in x2]
            y_fit = y1_fit + y2_fit
            
        else:
            xori_opt = 0
            
            m_opt1 = (yori_opt1 - yter_opt1) / (xori_opt1 - xter_opt1)
            m_opt2 = (yter_opt2 - yori_opt2) / (xter_opt2 - xori_opt2)
            
            c_opt1 = yter_opt1 - m_opt1 * (int(xter_opt1))
            c_opt2 = yter_opt1 - m_opt2 * (int(xter_opt2))
            
            y1_fit = [m_opt1 * x + c_opt1 for x in x1]
            y2_fit = [m_opt2 * x + c_opt2 for x in x2]
            
            y_fit = y1_fit + y2_fit
            
        y_corr = y / y_fit

    else:
        # y_corr = np.repeat(np.mean(y), len_init)
        y_corr = y
        y_fit = np.repeat(np.mean(y), len_init)
        print("OTR bias not detected")
        return y_corr, y_fit, xori_guess, xter_guess, bias

    
    return y_corr, y_fit, int(xori_opt), int(xter_opt), bias

def otr_set(df, ter_idx, ori_idx):
    
    cyc = False
    bias = False
    pt = "trough"
    x = df.index
    y = df["gc_corr_norm_cov"]
    y_med_fil = df["gc_cor_med_fil"]
    
    len_init = len(x)
    
    x_cyc = list(islice(cycle(x), 0, len_init*3))
    y_cyc = list(islice(cycle(y), 0, len_init*3))
    
    xori_guess = ori_idx
    xter_guess = ter_idx
    yori_guess = y[y_med_fil.argmax()]
    yter_guess = y[y_med_fil.argmin()] 
    
    if (abs(xori_guess-xter_guess) > len_init * 0.3 ):
        if (xori_guess < len_init*0.1) or (xori_guess > len_init * 0.9):
            xori_guess = 0
            y1 = y_cyc[:xter_guess]
            y2 = y_cyc[xter_guess:len_init]
            x1 = x_cyc[:xter_guess]
            x2 = x_cyc[xter_guess:len_init]
            initial_guess1 = [xori_guess, xter_guess, yori_guess, yter_guess]
            initial_guess2 = [xter_guess, xori_guess, yter_guess, yori_guess]
        elif (xter_guess < len_init*0.1) or (xter_guess > len_init * 0.9):
            xter_guess = 0
            y1 = y_cyc[:xori_guess]
            y2 = y_cyc[xori_guess:len_init]
            x1 = x_cyc[:xori_guess]
            x2 = x_cyc[xori_guess:len_init]
            initial_guess1 = [xori_guess, xter_guess, yori_guess, yter_guess]
            initial_guess2 = [len_init, xori_guess, yter_guess, yori_guess]
            pt = "peak"
        else:
            xi = xori_guess if (xori_guess < xter_guess) else xter_guess
            xj = xori_guess if (xori_guess > xter_guess) else xter_guess
            y1 = y_cyc[xi:xj]
            y2 = y_cyc[xj:(xi+len_init)]
            x1 = x_cyc[xi:xj]
            x2 = x_cyc[xj:(xi+len_init)]
            initial_guess1 = [xori_guess, xter_guess, yori_guess, yter_guess]
            initial_guess2 = [xter_guess, len_init, yter_guess, yori_guess]
            cyc = True
        bias = True
    else:
        y_corr = y
        y_fit = np.repeat(np.mean(y), len_init)
        print("OTR bias not detected")
        return y_corr, y_fit, bias
    
    result1 = minimize(fit_func, initial_guess1, args = (x1, y1))
    result2 = minimize(fit_func, initial_guess2, args = (x2, y2))
    
    xori_opt1, xter_opt1, yori_opt1, yter_opt1 = result1.x
    xter_opt2, xori_opt2, yter_opt2, yori_opt2 = result2.x
    
    xori_opt = np.mean([xori_opt1, xori_opt2])
    xter_opt = np.mean([xter_opt1, xter_opt2])
    yori_opt = np.mean([yori_opt1, yori_opt2])
    yter_opt = np.mean([yter_opt1, yter_opt2])
    
    y1_fit=[]
    y2_fit=[]
    
    if (yori_opt - yter_opt) >= 2*np.mean(y_med_fil):
        bias = True
    else:
        bias = False

    if bias and cyc:
        
        if (xori_opt > xter_opt):
            m_opt = (yori_opt - yter_opt) / (xori_opt - xter_opt)
            c_opt = yori_opt - m_opt * (xori_opt - xter_opt)
            m_opt1 = (yori_opt-yter_opt) / (xori_guess-xter_guess)
            m_opt2 = (yter_opt-yori_opt) / (len(x)-(xori_guess-xter_guess))
            
            c_opt1 = yori_opt - m_opt * (xori_opt-xter_opt)
            c_opt2 = yter_opt - m_opt * (xter_opt-xori_opt)
            
            y1_fit = [m_opt1 * x + c_opt1 for x in range(len(x1))]
            y2_fit = [m_opt2 * x + c_opt2 for x in range(len(x2))]
            y_fit = y1_fit + y2_fit
            y_fit = np.array(list(islice(cycle(y_fit), len(y_fit)-int(xter_guess), (2*len(y_fit) - int(xter_guess)))))
        else:
            m_opt = (yori_opt - yter_opt) / (xori_opt - xter_opt)
            c_opt = yori_opt - m_opt * (xori_opt - xter_opt)
            m_opt1 = (yori_opt-yter_opt) / (xori_guess-xter_guess)
            m_opt2 = (yter_opt-yori_opt) / (len(x)-(xori_guess-xter_guess))
            
            c_opt1 = yori_opt - m_opt * (xori_opt-xter_opt)
            c_opt2 = yter_opt - m_opt * (xter_opt-xori_opt)
            
            y1_fit = [m_opt1 * x + c_opt1 for x in range(len(x1))]
            y2_fit = [m_opt2 * x + c_opt2 for x in range(len(x2))]
            y_fit = y1_fit + y2_fit
            y_fit = np.array(list(islice(cycle(y_fit), len(y_fit)-int(xori_guess), (2*len(y_fit) - int(xori_guess)))))
        y_corr = y / y_fit
        
    elif bias and not cyc:
        if pt == "peak":
            xter_opt = 0
            
            m_opt1 = (yori_opt1 - yter_opt1) / (xori_opt1 - xter_opt1)
            m_opt2 = (yter_opt2 - yori_opt2) / (xter_opt2 - xori_opt2)

            c_opt1 = yori_opt2 - m_opt1 * (int(xori_opt1))
            c_opt2 = yori_opt2 - m_opt2 * (int(xori_opt2))
            
            y1_fit = [m_opt1 * x + c_opt1 for x in x1]
            y2_fit = [m_opt2 * x + c_opt2 for x in x2]
            y_fit = y1_fit + y2_fit
            
        else:
            xori_opt = 0
            
            m_opt1 = (yori_opt1 - yter_opt1) / (xori_opt1 - xter_opt1)
            m_opt2 = (yter_opt2 - yori_opt2) / (xter_opt2 - xori_opt2)
            
            c_opt1 = yter_opt1 - m_opt1 * (int(xter_opt1))
            c_opt2 = yter_opt1 - m_opt2 * (int(xter_opt2))
            
            y1_fit = [m_opt1 * x + c_opt1 for x in x1]
            y2_fit = [m_opt2 * x + c_opt2 for x in x2]
            
            y_fit = y1_fit + y2_fit
            
        y_corr = y / y_fit
    else:
        # y_corr = np.repeat(np.mean(y), len_init)
        y_corr = y
        y_fit = np.repeat(np.mean(y), len_init)
        print("OTR bias not detected")
        return y_corr, y_fit, xori_guess, xter_guess, bias

    return y_corr, y_fit, bias

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
        h1, f1, bias = otr_set(df, x1, x2)
        if bias:
            yori = f1[x2]
            yter = f1[x1]
            OTR = yori / (yter + 0.001)
        else:
            yori = np.nan
            yter = np.nan
            OTR = "Not detected"

        results = {"Origin location":ori, "Origin coverage (normalized)":yori,"Terminus window":ter, 
        "Terminus coverage (normalized)":yter, "Origin-to-Termius/Bias Ratio":OTR, "Correction type" : "Ori-ter defined by user" }
        df["otr_gc_corr_norm_cov"] = h1
        df["otr_gc_corr_fact"] = f1
        
        return df, results, ori, ter 
    
    else:
    # fits the bias curve to the most probable location of ori/ter based on coverage peak and troughs respectively
        h1, f1 , ori_idx, ter_idx, bias = otr_fit(df)
        if bias:
            xori = df["win_st"].iloc[ori_idx]
            xter = df["win_end"].iloc[ter_idx]
            yori = f1[ori_idx]
            yter = f1[ter_idx]
            OTR = yori / (yter + 0.001)
        else:
            xori = df["win_st"].iloc[0]
            xter = df["win_end"].iloc[len(df)-1]
            yori = np.nan
            yter = np.nan
            OTR = "Not detected"

        results = {"Origin window":int(xori), "Origin coverage (normalized)":yori, "Terminus window":int(xter), 
        "Terminus coverage (normalized)":yter, "Origin-to-Termius/Bias Ratio":OTR, "Correction type" : "Ori-ter coordinates fit by coverage" }
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
            results = results._append({'Startpos': start_pos,'Endpos': win_end[i-1], 
                                      'State': prev_most_probable_state_name}, ignore_index=True)
            start_pos = win_st[i]
        
        prev_obs = observation
        prev_most_probable_state_name = most_probable_state_name

    results = results._append({'Startpos': start_pos, 'Endpos': chr_length, 
                              'State': prev_most_probable_state_name}, ignore_index=True)
    
    return results



def run_HMM(sample, region, output, ori, ter, enforce, win, step, frag, error_rate=0.15, n_states=5, changeprob=(1e-10)):
    
    filepath = sample
    saveplt = str(output+"/CNV_plt/")
    saveloc = str(output+"/CNV_csv/")
    sample = sample.strip().split('/')[-1]
    samplename = sample.strip().split('.')[0]
    
    if region is not None:
        start, end = map(int, region.split('-'))
    else:
        start, end = None, None

    df, ori_loc, ter_loc = bias_correction(filepath, start, end, output, ori, ter, enforce, win, step, frag)
    
    df["otr_gc_corr_norm_cov"] = np.nan_to_num(df["otr_gc_corr_norm_cov"])
    cor_cov = df["otr_gc_corr_norm_cov"]

    med = df["read_count_cov"].median()
    cor_rc = []
    
    for i in range(len(df)):
        # cor_rc.insert(i, int(np.nan_to_num(df["otr_gc_corr_norm_cov"].iloc[i]) * med))
        cor_rc.insert(i, int(df["otr_gc_corr_norm_cov"].iloc[i] * med))
    
    # df['win_len'] = df["win_end"] - df["win_st"]
    
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
    
    brk_full_path = os.path.join(saveloc,'%s_break_pts.csv' % samplename)
    cn_brk = pd.DataFrame(copy_numbers,columns=['Startpos','Endpos', 'State'])
    cn_brk['Segment_Size'] = cn_brk['Endpos'] - cn_brk['Startpos']
    cn_brk.drop(columns='Endpos', inplace = True)
    cn_brk.to_csv(brk_full_path, index = False)
    
    CN_HMM = []
    # output copy number prediction for each instance of genomic sliding window
    for cnrow in range(len(copy_numbers)):

        state = int(copy_numbers['State'][cnrow])

        hmmstart = int(copy_numbers['Startpos'][cnrow])
        hmmend = int(copy_numbers['Endpos'][cnrow])
        
        CN_HMM_row = []
        # Index lists of windows between the breakpoints
        idx_list = df[(hmmstart <= df["win_st"]) & (df["win_st"] <= hmmend)].index


        if len(idx_list) == 0:
            continue  # skip if no matching start position found
    
        #Iterate over the select windows and append the copy number (state) to the window
        for idx in (idx_list):

            if ((df["win_st"].iloc[idx] >= hmmstart) and (df["win_end"].iloc[idx] <= hmmend)):
                CN_HMM_row.append(state)
            else:
                continue
        
        CN_HMM.extend(CN_HMM_row)
    
    new_exp['prob_copy_number'] = CN_HMM

    csv_full_path = os.path.join(saveloc,'%s_CNV.csv' % samplename)

    new_exp.reset_index(drop = True)
    new_exp.to_csv(csv_full_path, index = False)

    print(f"{sample}: Copy number prediction complete. .csv files saved.")
    return new_exp, sample, ori_loc, ter_loc


def bias_correction(filepath, start, end, output, ori, ter, enforce, win, step, frag):

    # Function handles the bias correction functions and outputs to be fed into the HMM
    sample = filepath.strip().split('/')[-1]
    samplename = sample.strip().split('.')[0]
    saveplt = str(output+"/OTR_corr/")
    print(f'{sample}: Calculating coverage and GC% across sliding window over the genome.')
    df = preprocess(filepath, start, end, win, step, frag)
    gc_corr = gc_normalization(df)
    print(f'{sample}: Corrected GC bias in coverage.')
    otr_corr, otr_res, ori_win, ter_win = otr_correction(gc_corr, ori, ter, enforce)
    with open(saveplt+str(samplename)+'_otr_results.json', 'w') as f:
        json.dump(otr_res, f, indent = 4)
    print(f'{sample}: Corrected origin/terminus of replication(OTR) bias in coverage.')
    # otr_corr.to_csv("../../MGD17-LB-E06_ADP1.csv")
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

    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax2 = ax1.twinx()
    ax1.patch.set_visible(False)

    ax1.set_zorder(2)  # Higher than ax2
    ax2.set_zorder(1)  # Lower than ax1

    
    ax2.scatter(df_cnv["win_st"],df_cnv["read_count_cov"], color="gray", label="Raw reads",s=10, alpha = 0.2)
    ax2.scatter(df_cnv["win_st"],df_cnv["otr_gc_corr_rdcnt_cov"], color="orange", label="Corrected reads",s=5, alpha = 0.5,
                marker = mplt.markers.MarkerStyle(marker = 'o', fillstyle = 'none'))
    ax1.scatter(df_cnv["win_st"],df_cnv["prob_copy_number"], color="red", label="Predicted Copy Number", marker="_", s = 30)

    delta = (df_cnv['read_count_cov'].median()*0.5)
    
    
    ax2.set_ylim(df_cnv['read_count_cov'].min() - delta, df_cnv['read_count_cov'].max() + delta)
    ax1.set_ylim(df_cnv['otr_gc_corr_norm_cov'].min() - 0.5, df_cnv['otr_gc_corr_norm_cov'].max() + 0.5)
    
    ax1.set_xlabel("Window (Genomic position)")
    ax1.yaxis.label.set_color('red')
    ax2.set_ylabel("Read Counts")
    ax1.set_ylabel("Copy Numbers")
    plt.title(f'{samplename}_Copy Number Prediction')
    
    handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()
    handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()

    # Combine handles and labels
    handles = handles_ax1 + handles_ax2
    labels = labels_ax1 + labels_ax2

    # Create a combined legend on one of the axes (or the figure)
    ax1.legend(handles, labels, loc='best')  # Place on ax1
    # Alternatively, place it on ax2:
    # ax2.legend(handles, labels, loc='best')
    
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
        "--region",
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
        help="output file location preference. Defaults to the current folder.",
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
            region=options.reg,
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
            region=options.reg,
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
