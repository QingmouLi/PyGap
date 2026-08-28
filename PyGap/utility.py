# -*- coding: utf-8 -*-
"""
PyGap.utility

Utility used to demonstrate how to use PyGap

Created on Wed Aug 19 00:28:34 2024
@author: Qingmou Li & Jon W. Liu, Geological Survey of Canada, NRCan, Canada

"""
from scipy import stats as STATS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy import interpolate
from scipy import signal     
from scipy.optimize import curve_fit

import seaborn as sns
from scipy.stats import gaussian_kde, skew, kurtosis

def stats(lst, period, raw, filled):
     """statistics of residuals

    Args:
        lst (List): list of detected frequencies
        period (_type_): _description_
        raw (_type_): _description_
        filled (_type_): _description_

    Returns:
        numpy.ndarray: residuals 
     """

     rawsel = np.zeros((period*len(lst),),dtype=float)     
     fillsel= np.zeros_like(rawsel)
     c= 0
     for l in lst:
          rawsel[c*period:(c+1)*period] = raw[period*l:period*(l+1)]
          fillsel[c*period:(c+1)*period] = filled[period*l:period*(l+1)]
          c=c+1
     return rawsel-fillsel

def AmpFre(f, specDen, prtStats=False):
    """_summary_

    Args:
        f (_type_): _description_
        specDen (_type_): _description_
        prtStats (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
 
    fmin = f[0]; fmax=f[-1]
    flog = np.log(f)    
    speclog = np.log(specDen)
    slope, intercept, r, p, se = STATS.linregress(flog, speclog)
    if prtStats:
        print(f'slope={slope:.3f}, r={r:.3f}, p={p:.3f}, se={se:.3f}')
    SpecMin= np.exp(intercept)*fmin**slope
    SpecMax = np.exp(intercept)*fmax**slope
    return fmin,fmax,SpecMin,SpecMax

def plotSpectrum(df, U, args, prtStats=False):
    '''
    Calculate and plot power spectrum
    
    Parameters
    ----------
    df:  Pandas.dataframe
         input dataframe
    U:   numpy.ndarray
         input parameter
    args: arguparse namespace
          input parameter
    prtStats: bool
          input parameter to control if print statistics

    Return:
          None
    '''    
    fig = plt.figure(layout="constrained", figsize=(9,6)) 
 
    fs1 = 1.0 
    f1,p1 =  signal.periodogram(df[args.chName].values, fs1, scaling='spectrum', window='hamming', detrend='constant')
    fs0= 1.0/24.0  
    f0, p0 = signal.periodogram(U[0,:], fs0, scaling='spectrum', window='hamming', detrend='constant') #'linear')
    if prtStats:
        print('slope in time domain')
    fmin1, fmax1, specdenmin1,specdanmax1 = AmpFre(f1[1:], p1[1:], prtStats=False)
    if prtStats:
        print('slope in eigenspace')
    fmin0, fmax0, specdenmin0,specdanmax0 = AmpFre(f0[1:], p0[1:], prtStats=False)                                              

    gs = GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[1,0:2])
    ax3 = fig.add_subplot(gs[:,2])
    ax1.plot(df.index, df[args.chName],'-k', lw=0.75)
    ax2.plot(pd.date_range(start=args.startDay, end=args.endDay) , U[:,0],'-or', markersize=3)
    ax3.plot(f1[1:],p1[1:],'ok',markersize=1)
    ax3.plot([fmin1,fmax1],[specdenmin1,specdanmax1],'-k',lw=1)
    ax3.plot(f0[1:],p0[1:],'dr',markersize=2)
    ax3.plot([fmin0,fmax0],[specdenmin0,specdanmax0],'-r',lw=1.5)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax1.xaxis.set_ticklabels([])
    ax1.grid(True);ax2.grid(True);ax3.grid(True)
    ax3.set_xlabel('Frequency (1/day)')
    ax3.set_ylabel('Spectrum')
    ax1.set_ylabel('nt')
    ax2.set_ylabel('U')
    ax1.set_xlim([df.index[0],df.index[-1]])
    ax2.set_xlim([df.index[0],df.index[-1]])
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=25, ha='right')
    ax3.axvline(fmax0, color='b',linestyle='--', lw=2)
    plt.tight_layout()
    plt.show()


def compare_residuals_all(
    res1,
    res2,
    name1="Model 1",
    name2="Model 2"
):
    '''
    Compare residuals of two vector dump and chart statistics 

    Parameters
    ----------
    res1: numpy.ndarray
          Input residuals.
    res2: numpy.ndarray
          Input residuals.
    name1: string 
           Name of the res1
    name2: string 
           Name of the res2
           
    Return
    ------
    None
    '''
    res1 = np.asarray(res1)
    res2 = np.asarray(res2)

    if res1.shape != res2.shape:
        raise ValueError("res1 and res2 must have the same shape")

    # =========================
    # Summary statistics
    # =========================
    diff = res1 - res2

    corr = np.corrcoef(res1, res2)[0, 1]

    skew1, skew2 = skew(res1), skew(res2)
    kurt1, kurt2 = kurtosis(res1), kurtosis(res2)

    print("===== Residual Comparison Summary =====")
    print(f"Correlation coefficient          : {corr:.4f}")
    print(f"Mean({name1})                    : {res1.mean():.4f}")
    print(f"Mean({name2})                    : {res2.mean():.4f}")
    print(f"Std({name1})                     : {res1.std(ddof=1):.4f}")
    print(f"Std({name2})                     : {res2.std(ddof=1):.4f}")
    print(f"Mean difference ({name1}-{name2}) : {diff.mean():.4f}")
    print(f"RMSE between residuals           : {np.sqrt(np.mean(diff**2)):.4f}")
    print(f"{name1}: skew = {skew1:.3f}, kurtosis = {kurt1:.3f}")
    print(f"{name2}: skew = {skew2:.3f}, kurtosis = {kurt2:.3f}")

    # =========================
    # Plot styling
    # =========================
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.2)
    
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15, 4),  
        gridspec_kw=dict(width_ratios=[1.3, 1.1, 1])
    )

    # =========================
    # Panel 1: Scatter with density (FIRST)
    # =========================
    xy = np.vstack([res1, res2])
    density = gaussian_kde(xy)(xy)

    ax = axes[0]
    sc = ax.scatter(
        res1, res2,
        c=density,
        s=12,
        cmap="viridis"
    )

    lims = [
        min(res1.min(), res2.min()),
        max(res1.max(), res2.max())
    ]
    ax.plot(lims, lims, "k--", linewidth=1)

    ax.set_xlabel(f"Residuals ({name1})")
    ax.set_ylabel(f"Residuals ({name2})")
    ax.set_title("Residual Scatter (Density Enhanced)")
    ax.axis("equal")
    ax.grid(True)

    plt.colorbar(sc, ax=ax, label="Point density")

    # =========================
    # Panel 2: KDE distributions
    # =========================
    ax = axes[1]

    sns.kdeplot(res1, ax=ax, label=name1, fill=True)
    sns.kdeplot(res2, ax=ax, label=name2, fill=True, alpha=0.6)

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution")
    ax.legend(frameon=False)
    ax.grid(True, linestyle=":", linewidth=0.6)    
    
    # =========================
    # Panel 3: Violin plot
    # =========================
    ax = axes[2]

    sns.violinplot(
        data=[res1, res2],
        ax=ax,
        inner="quartile",
        cut=0
    )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticklabels([name1, name2])
    ax.set_ylabel("Residual")
    ax.set_title("Residual Shape & Spread")
    ax.grid(True, linestyle=":", linewidth=0.6)
    
    # =========================
    # Final polish
    # =========================
    fig.suptitle(
        "Residual Diagnostics: Agreement, Distribution, and Tails",
        y=1.05
    )
    fig.tight_layout()
    plt.show()


def compare_residuals_scatter_his(
    res1,
    res2,
    name1="Model 1",
    name2="Model 2"
):
    '''
    Compare residuals of two vector dump and chart statistics 

    Parameters
    ----------
    res1: numpy.ndarray
          Input residuals.
    res2: numpy.ndarray
          Input residuals.
    name1: string 
           Name of the res1
    name2: string 
           Name of the res2
           
    Return
    ------
    None
    '''
    res1 = np.asarray(res1)
    res2 = np.asarray(res2)

    if res1.shape != res2.shape:
        raise ValueError("res1 and res2 must have the same shape")

    # =========================
    # Summary statistics
    # =========================
    diff = res1 - res2

    corr = np.corrcoef(res1, res2)[0, 1]

    skew1, skew2 = skew(res1), skew(res2)
    kurt1, kurt2 = kurtosis(res1), kurtosis(res2)

    print("===== Residual Comparison Summary =====")
    print(f"Correlation coefficient          : {corr:.4f}")
    print(f"Mean({name1})                    : {res1.mean():.4f}")
    print(f"Mean({name2})                    : {res2.mean():.4f}")
    print(f"Std({name1})                     : {res1.std(ddof=1):.4f}")
    print(f"Std({name2})                     : {res2.std(ddof=1):.4f}")
    print(f"Mean difference ({name1}-{name2}) : {diff.mean():.4f}")
    print(f"RMSE between residuals           : {np.sqrt(np.mean(diff**2)):.4f}")
    print(f"{name1}: skew = {skew1:.3f}, kurtosis = {kurt1:.3f}")
    print(f"{name2}: skew = {skew2:.3f}, kurtosis = {kurt2:.3f}")

    # =========================
    # Plot styling
    # =========================
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.2)
    
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4),  #(15,4)
        gridspec_kw=dict(width_ratios=[1.3, 1.1])  #[1.3, 1.1, 1])
    )

    # =========================
    # Panel 1: Scatter with density (FIRST)
    # =========================
    xy = np.vstack([res1, res2])
    density = gaussian_kde(xy)(xy)

    ax = axes[0]
    sc = ax.scatter(
        res1, res2,
        c=density,
        s=12,
        cmap="viridis"
    )

    lims = [
        min(res1.min(), res2.min()),
        max(res1.max(), res2.max())
    ]
    ax.plot(lims, lims, "k--", linewidth=1)

    ax.set_xlabel(f"Residuals ({name1})")
    ax.set_ylabel(f"Residuals ({name2})")
    ax.set_title("Residual Scatter (Density Enhanced)")
    ax.axis("equal")
    ax.grid(True)

    plt.colorbar(sc, ax=ax, label="Point density")

    # =========================
    # Panel 2: KDE distributions
    # =========================
    ax = axes[1]

    sns.kdeplot(res1, ax=ax, label=name1, fill=True)
    sns.kdeplot(res2, ax=ax, label=name2, fill=True, alpha=0.6)

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution")
    ax.legend(frameon=False)
    ax.grid(True, linestyle=":", linewidth=0.6)

    # =========================
    # Final polish
    # =========================
    fig.suptitle(
        "Residual Diagnostics: Agreement, Distribution, and Tails",
        y=1.05
    )
    fig.tight_layout()
    plt.show()
    