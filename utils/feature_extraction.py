# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np 
import os

import pickle
import argparse

from scipy.stats import iqr, skew
from scipy.fftpack import fft
from statsmodels.tsa.ar_model import AutoReg


def apply_mav_filter(ppg_signal, window_size):
    filtered_signal = np.zeros_like(ppg_signal)
    for n in range(len(ppg_signal)):
        start_index = max(0, n - window_size + 1)
        mav_value = np.mean(ppg_signal[start_index:n + 1])
        filtered_signal[n] = mav_value
    return filtered_signal


def apply_mav_filter_2d(data, window_size):
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i, :] = apply_mav_filter(data[i, :], window_size)
    return filtered_data


def cal_ar_features(ppg_data, p=3, window_size=128):
    print('---------------- Extract AR_feature ')
    all_arppg, all_arkte, all_arloge = [], [], []
    for sample_data in ppg_data:
        sample_length = len(sample_data)
        # KTEw - per 1 sample 
        KTEw = np.zeros(sample_length)
        for k in range(1, sample_length-1):
            KTEw[k] = sample_data[k]**2 - sample_data[k-1] * sample_data[k+1]

        # LogE - per 1 window
        LogE = np.zeros(sample_length // window_size)
        for i in range(0, sample_length, window_size):
            window = sample_data[i:i+window_size]
            energy = np.sum(window**2)
            LogE[i // window_size] = np.log(energy)
        
        model_Sw = AutoReg(sample_data, lags=p)
        results_Sw = model_Sw.fit()
        ARPPG = results_Sw.params[1:]

        model_KTEw = AutoReg(KTEw[1:-1], lags=p)
        results_KTEw = model_KTEw.fit()
        ARKTE = results_KTEw.params[1:]

        model_LogE = AutoReg(LogE, lags=p)
        results_LogE = model_LogE.fit()
        ARLogE = results_LogE.params[1:]

        all_arppg.append(ARPPG)
        all_arkte.append(ARKTE)
        all_arloge.append(ARLogE)

    return np.array(all_arppg), np.array(all_arkte), np.array(all_arloge)   
    
    
    
def cal_log_features(ppg_data , window_size = 128):
    print('---------------- Extract LogE_feature ')
    log_e_mean =[]
    log_e_variance =[]
    log_e_iqr =[]
    log_e_skew =[]    
    for sample_data in ppg_data:
        log_e_sequence = np.log(np.sum(sample_data.reshape(-1, window_size)**2, axis=1))
        log_e_mean.append(np.mean(log_e_sequence))
        log_e_variance.append(np.var(log_e_sequence))
        log_e_iqr.append(iqr(log_e_sequence))
        log_e_skew.append(skew(log_e_sequence))   

    return log_e_mean, log_e_variance, log_e_iqr, log_e_skew    



def cal_kte_operator(sequence):
    return sequence**2 - np.roll(sequence, 1) * np.roll(sequence, -1)

def cal_kte_features(ppg_data, window_size=128):
    print('---------------- Extract KTE_feature ')
    num_samples, num_data_points = ppg_data.shape
    windows_per_sample = num_data_points // window_size
    kte_means = []
    kte_variances = []
    kte_iqrs = []
    kte_skews = []
    for sample_data in ppg_data:
        windows = sample_data.reshape(windows_per_sample, window_size)
        sample_kte_means = []
        sample_kte_variances = []
        sample_kte_iqrs = []
        sample_kte_skews = []

        for window in windows:
            kte_sequence = cal_kte_operator(window)
            kte_mean = np.mean(kte_sequence)
            kte_variance = np.var(kte_sequence)
            kte_iqr = iqr(kte_sequence)
            kte_skew = skew(kte_sequence)
            sample_kte_means.append(kte_mean)
            sample_kte_variances.append(kte_variance)
            sample_kte_iqrs.append(kte_iqr)
            sample_kte_skews.append(kte_skew)

        kte_means.append(sample_kte_means)
        kte_variances.append(sample_kte_variances)
        kte_iqrs.append(sample_kte_iqrs)
        kte_skews.append(sample_kte_skews)

    return np.array(kte_means), np.array(kte_variances), np.array(kte_iqrs), np.array(kte_skews)



def cal_se_features(ppg_data, window_size=128):
    print('---------------- Extract SE_feature ')
    num_samples, num_data_points = ppg_data.shape
    windows_per_sample = num_data_points // window_size

    se_means = []
    se_variances = []
    se_iqrs = []
    se_skews = []
    for sample_data in ppg_data:
        windows = sample_data.reshape(windows_per_sample, window_size)
        sample_se_means = []
       
        for window in windows:
            fft_result = fft(window)
            squared_abs_fft = np.abs(fft_result)**2
            normalized_squared_abs_fft = squared_abs_fft / np.sum(squared_abs_fft)
            se = -np.sum(normalized_squared_abs_fft * np.log2(normalized_squared_abs_fft))
            sample_se_means.append(se)

        sample_se_means = np.array(sample_se_means)
        se_means.append(np.mean(sample_se_means))
        se_variances.append(np.var(sample_se_means))
        se_iqrs.append(iqr(sample_se_means))
        se_skews.append(skew(sample_se_means))

    return np.array(se_means), np.array(se_variances), np.array(se_iqrs), np.array(se_skews)   

def cal_hrv_feature(ppg_data):
    print('---------------- Extract HRV_feature ')
    heart_rates = []
    sdnns = []
    rmssds = []
    vlf_powers = [] 
    lf_powers = []
    hf_powers = []
    for i in range(ppg_data.shape[0]):
        ppg_data_sample = ppg_data[i]
        peak_indices, _ = signal.find_peaks(ppg_data_sample, distance=40)  
        peak_times = t[peak_indices]
        hrv = np.diff(peak_times)  # HRV
        if len(hrv) > 1:

            sdnn = np.std(hrv, ddof=1)  # SDNN
            rmssd = np.sqrt(np.mean(np.square(np.diff(hrv))))  # RMSSD
            fft_vals = np.fft.rfft(hrv)
            fft_freq = np.fft.rfftfreq(len(hrv), d=(peak_times[1] - peak_times[0]))
            vlf_band = (0.0033, 0.04)
            lf_band = (0.04, 0.15)
            hf_band = (0.15, 0.4)

            vlf_power = np.trapz(abs(fft_vals[(fft_freq >= vlf_band[0]) & (fft_freq < vlf_band[1])])**2, dx=fft_freq[1])
            lf_power = np.trapz(abs(fft_vals[(fft_freq >= lf_band[0]) & (fft_freq < lf_band[1])])**2, dx=fft_freq[1])
            hf_power = np.trapz(abs(fft_vals[(fft_freq >= hf_band[0]) & (fft_freq < hf_band[1])])**2, dx=fft_freq[1])

            heart_rates.append(60 / hrv.mean())
            sdnns.append(sdnn)
            rmssds.append(rmssd)
            vlf_powers.append(vlf_power)
            lf_powers.append(lf_power)
            hf_powers.append(hf_power)
        else:

            heart_rates.append(np.nan)
            sdnns.append(np.nan)
            rmssds.append(np.nan)
            vlf_powers.append(np.nan)
            lf_powers.append(np.nan)
            hf_powers.append(np.nan)

    return heart_rates, sdnns, rmssds, vlf_powers, lf_powers, hf_powers




def feature_extraction(ppg_data):
    print('---------------- Apply Filtering ')
    filtered_ppg = apply_mav_filter_2d(ppg_data , 10)

    print('---------------- Extract Feature ')
    log_e_means, log_e_variances, log_e_iqrs, log_e_skews = cal_log_features(filtered_ppg )    

    kte_means, kte_variances, kte_iqrs, kte_skews = cal_kte_features(filtered_ppg)   
    kte_means = np.mean(kte_means,axis = 1)
    kte_variances = np.mean(kte_variances,axis = 1)
    kte_iqrs = np.mean(kte_iqrs,axis = 1)
    kte_skews = np.mean(kte_skews,axis = 1)

    se_means, se_variances, se_iqrs, se_skews = cal_se_features(filtered_ppg)    
    se_means =np.nan_to_num(se_means)
    se_variances =np.nan_to_num(se_variances)
    se_iqrs =np.nan_to_num(se_iqrs)
    se_skews =np.nan_to_num(se_skews)
    
    # arppg, arkte, arloge = cal_ar_features(filtered_ppg)
    # result_fea = np.column_stack((se_means, se_variances, se_iqrs, se_skews, kte_means, kte_variances, kte_iqrs, kte_skews, log_e_means, log_e_variances, log_e_iqrs, log_e_skews, arppg,arkte, arloge))    
    
    result_fea = np.column_stack((se_means, se_variances, se_iqrs, se_skews, kte_means, kte_variances, kte_iqrs, kte_skews, log_e_means, log_e_variances, log_e_iqrs, log_e_skews))
    return result_fea
    
