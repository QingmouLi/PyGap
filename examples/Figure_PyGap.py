# -*- coding: utf-8 -*-
"""
Prepared for JOSS

@author: Qingmou Li & Jon W. Liu, Geological Survey of Canada
"""
from pathlib import Path
import sys
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

import numpy as np
import argparse
import pandas as pd
from PyGap import EigenSig 
from PyGap import dataSamples
from PyGap import LSHE
from PyGap import utility as util
import matplotlib.pyplot as plt
import time
from matplotlib.gridspec import GridSpec

import seaborn as sns    
from scipy.stats import skew, kurtosis,gaussian_kde, probplot
"""
        
    """
def residual_scatter_with_density(r1, r2, name1="Model 1", name2="Model 2"):
    """Scatter plot of residuals with density coloring and summary statistics.

    Args:
        r1 (_type_): _description_
        r2 (_type_): _description_
        name1 (str, optional): _description_. Defaults to "Model 1".
        name2 (str, optional): _description_. Defaults to "Model 2".

    Raises:
        ValueError: _description_
    """

    
    r1 = np.asarray(r1)
    r2 = np.asarray(r2)

    if r1.shape != r2.shape:
        raise ValueError("r1 and r2 must have the same shape")

    # ---- Density estimation ----
    xy = np.vstack([r1, r2])
    density = gaussian_kde(xy)(xy)

    # ---- Scatter plot ----
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(r1, r2, c=density, s=12, cmap="viridis")

    lims = [
        min(r1.min(), r2.min()),
        max(r1.max(), r2.max())
    ]
    ax.plot(lims, lims, "k--", linewidth=1)

    ax.set_xlabel(f"Residuals ({name1})")
    ax.set_ylabel(f"Residuals ({name2})")
    ax.set_title("Residual Comparison (Density Enhanced)")
    ax.axis("equal")
    ax.grid(True)

    plt.colorbar(sc, ax=ax, label="Point density")
    plt.show()

    # ---- Summary statistics ----
    diff = r1 - r2

    print("===== Residual Comparison Summary =====")
    print(f"Correlation coefficient        : {np.corrcoef(r1, r2)[0, 1]:.4f}")
    print(f"Mean({name1})                  : {r1.mean():.4f}")
    print(f"Mean({name2})                  : {r2.mean():.4f}")
    print(f"Std({name1})                   : {r1.std(ddof=1):.4f}")
    print(f"Std({name2})                   : {r2.std(ddof=1):.4f}")
    print(f"Mean difference ({name1}-{name2}): {diff.mean():.4f}")
    print(f"RMSE between residuals         : {np.sqrt(np.mean(diff**2)):.4f}")

def bland_altman_plot(r1, r2, name1="Model 1", name2="Model 2"):
    """comparing two residual sets

    Args:
        r1 (numpy.ndarray): one d numpy array holding residuals (observed-model1)
        r2 (numpy.ndarray): one d numpy array holding residuals (observed-model2)
        name1 (str, optional): Residual Name of model 1. Defaults to "Model 1".
        name2 (str, optional): Residual Name of model 2. Defaults to "Model 2".
    """    
    
    r1 = np.asarray(r1)
    r2 = np.asarray(r2)

    mean_res = 0.5 * (r1 + r2)
    diff_res = r1 - r2

    mean_diff = diff_res.mean()
    std_diff = diff_res.std(ddof=1)

    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean_res, diff_res, alpha=0.6, s=12)

    ax.axhline(mean_diff, color="red", linestyle="--", label="Mean difference")
    ax.axhline(loa_upper, color="gray", linestyle="--", label="±1.96 SD")
    ax.axhline(loa_lower, color="gray", linestyle="--")

    ax.set_xlabel(f"Mean residual ({name1}, {name2})")
    ax.set_ylabel(f"Residual difference ({name1} − {name2})")
    ax.set_title("Bland–Altman Plot")
    ax.legend()
    ax.grid(True)

    plt.show()

def qq_plot_residuals(r1, r2, name1="Model 1", name2="Model 2"):
    """
    QQ plot comparing the distributions of two residual sets to anwser:
     “Is one model better throughout the error distribution, or only at the center/tails?” This matters 
     because mean error alone hides tail behavior, which is usually what hurts models in practice.    
    Interpretation tips:
        Straight line → same distribution
        Curvature → skewness difference
        Tail deviations → variance / outlier differences
    Args:
        r1 (numpy.ndarray): one d numpy array holding residuals (observed-model1)
        r2 (numpy.ndarray): one d numpy array holding residuals (observed-model2)
        name1 (str, optional): Residual Name of model 1. Defaults to "Model 1".
        name2 (str, optional): Residual Name of model 2. Defaults to "Model 2".
    """    
    r1 = np.asarray(r1)
    r2 = np.asarray(r2)

    q = np.linspace(0.01, 0.99, 200)

    q1 = np.quantile(r1, q)
    q2 = np.quantile(r2, q)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(q1, q2, s=15)

    lims = [
        min(q1.min(), q2.min()),
        max(q1.max(), q2.max())
    ]
    ax.plot(lims, lims, "k--", linewidth=1)

    ax.set_xlabel(f"Quantiles ({name1})")
    ax.set_ylabel(f"Quantiles ({name2})")
    ax.set_title("QQ Plot: Residual Distributions")
    ax.grid(True)

    plt.show()

def pltS(ax, S):
    """
    plot singular values (sv) and accumulated sv for singular spectrum analysis

    Args:
        ax (axes of matplotlib): _description_
        S (numpy-ndarray): 1D numpy array holding singular values
    """

    ax.plot(S, '-rd', lw=1)
    ax.set_yscale('log')
    ax.set_xlabel('# of EigenValue')
    ax.set_ylabel('EigenValue')
    ax.tick_params(axis='y', labelrotation=90)
    """

    """
def compareResiduals(ax1, ax2, res1, res2, name1="Model 1", name2="Model 2"):
    """
    statistics of residuals 
        ✅ comparison of residuals from two models.
    Residual distribution of histogram look for better model:
        Peak closer to zero
        Narrower spread
        More symmetric shape
    
    ✅violin chart Look for better residuals:
        Median near zero
        Narrow violin body
        Short tails
    
    Diagnostics emphasized:
        - Centered at zero
        - Symmetry (skewness)
        - Tail behavior (kurtosis)
        - Distribution shape and spread

    Left panel: KDE (distribution around zero)
    Right panel: Violin plot (shape, tails, median)

    Args:
        ax1 (matplotlib.axes): axes object of matplotlib
        ax2 (matplotlib.axes): axes object of matplotlib
        res1 (numpy,ndarray): numpy array of residual1 
        res2 (numpy,ndarray): numpy array of residual2 
        name1 (str, optional): names of res1. Defaults to "Model 1".
        name2 (str, optional): names of res2. Defaults to "Model 2".
    """
    
    # --- statistics ---
    skew1, skew2 = skew(res1), skew(res2)
    kurt1, kurt2 = kurtosis(res1), kurtosis(res2)  # excess kurtosis

    print(f"{name1}: skew = {skew1:.3f}, kurtosis = {kurt1:.3f}")
    print(f"{name2}: skew = {skew2:.3f}, kurtosis = {kurt2:.3f}")

    # --- plotting style ---
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.2)

    # --- Left: KDE distribution ---
    sns.kdeplot(res1, ax=ax1, label=name1, fill=True)
    sns.kdeplot(res2, ax=ax1, label=name2, fill=True, alpha=0.6)

    ax1.axvline(0, color="black", linestyle="--", linewidth=1)
    ax1.set_xlabel("Residual")
    ax1.set_ylabel("Density")
    #ax1.set_title("Residual Distribution")
    ax1.legend(frameon=False)
    ax1.tick_params(axis='y', labelrotation=90)
    # --- Right: Violin plot ---
    sns.violinplot(
        data=[res1, res2],
        ax=ax2,
        inner="quartile",
        cut=0
    )
    ax2.tick_params(axis='y', labelrotation=90)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_xticklabels([name1, name2])
    ax2.set_ylabel("Residual")
    #ax2.set_title("Residual Shape & Spread")

    ax1.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax2.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    #fig.suptitle("Residual Diagnostics: Centering, Symmetry, and Tails", y=1.05)
    #fig.tight_layout()
    #plt.show()

def divideaxe():
    """complex arears for ploting

    Returns:
        axes array: axes read to plot
    """
    fig = plt.figure(figsize=(12, 10))
    axes=[]
    # Outer grid: 2 blocks ONLY
    # This hspace controls the gap between row 2 and row 3
    gs_outer = GridSpec(
        2, 1,
        figure=fig,
        hspace=0.2,        # ✅ gap between top block and bottom block
        height_ratios=[2, 2]
    )

    # ── Top block: rows 1 & 2 (no gap)
    gs_top = gs_outer[0].subgridspec(
        2, 1,
        hspace=0           # ✅ no gap between row 1 and 2
    )

    ax1 = fig.add_subplot(gs_top[0])
    axes.append(ax1)
    ax2 = fig.add_subplot(gs_top[1], sharex=ax1)
    axes.append(ax2)
    # ── Bottom block: rows 3 & 4 split into 3 columns
    gs_bottom = gs_outer[1].subgridspec(
        2, 3,
        hspace=0.4,          # ✅ no gap between row 3 and 4
        wspace=0.2
    )

    ax3 = [fig.add_subplot(gs_bottom[0, i]) for i in range(3)]
    axes.extend(ax3)
    ax4 = [fig.add_subplot(gs_bottom[1, i]) for i in range(3)]  #ax4 = [fig.add_subplot(gs_bottom[1, i], sharex=ax3[i]) for i in range(3)]
    axes.extend(ax4)
    
    return axes    

def plotUV(ax3,ax4, gf):
    """plot  U and V
    Args:
        ax3 (matplotlib.axes): axe to plot V
        ax4 (matplotlib.axes): axe to plot U
        gf (pandas.dataframe): dataframe holding U and V           
    """

    ax4.plot([x for x in range(gf.V.shape[1])], gf.V[0,:],  '-db', label='First EigenVector')
    ax4.set_xlabel('hour of the day')
    ax4.tick_params(axis='y', labelrotation=90)
    ax4.set_ylabel('nt')

    ax3.plot(gf.UU[:,0], '--dk', label='First Trace')
    ax3.plot(gf.dis, gf.U[:,0], 'or', label='First Trace')
    ax3.set_xlim([gf.dis[0], gf.dis[-1]])
    ax3.set_xlabel('# of Epoch')
    ax3.tick_params(axis='y', labelrotation=90)
    ax3.set_ylabel('load coef.')

def figure6(gs, T_del, TS, T, delList,args, df, TS4plt, lsheFreq,coef,prtStats=False):
    """_summary_

    Args:
        gs (_type_): _description_
        T_del (_type_): _description_
        TS (_type_): _description_
        T (_type_): _description_
        delList (_type_): _description_
        args (_type_): _description_
        df (_type_): _description_
        TS4plt (_type_): _description_
        lsheFreq (_type_): _description_
        coef (_type_): _description_
        prtStats (bool, optional): _description_. Defaults to False.
    """     
    EigenRes    = util.stats(delList, args.period, df[args.chName].values, df['GapFilled'].values)
    LSHERes     = util.stats(delList, args.period, df[args.chName].values, df['LSHEFilled'].values)

     #cuRes    = stats(delList, args.period, df[args.chName].values, cubic)    
    if prtStats:
         print ('gap Fill stats')
         print ('min, max, mean, std')
         print(f'EigenSig: {np.min(EigenRes):.3f},{np.max(EigenRes):.3f}, {np.mean(EigenRes):.3f}, {np.std(EigenRes):.3f}')
         print(f'LSHE:     {np.min(LSHERes):.3f},{np.max(LSHERes):.3f}, {np.mean(LSHERes):.3f}, {np.std(LSHERes):.3f}')
         
         #print(f'{np.min(cuRes):.3f},{np.max(cuRes):.3f}, {np.mean(cuRes):.3f}, {np.std(cuRes):.3f}')         

    ax = divideaxe()
    lw=0.75
    #first row display Y, LSHE, EigenSig
    ax[0].plot(df.index, TS4plt.flatten(),'-k', markersize=3,mec='b', label='Y', lw=0.5)   #the Y with gap as nan
    ax[0].plot(df.index, df[args.chName], '-.k', lw=lw)  #plot in the gap
    ax[0].plot(df.index, df['GapFilled'], '--r', lw=lw,label='EigenSig')
    ax[0].plot(df.index, df['LSHEFilled'], '--b', lw=lw, label='LSHE')
    ax[0].set_ylabel('nt')
     #residuals
    ax[1].plot(df.index, df["Residuals"],'--r', label='EigenSig', lw=lw)
    ax[1].plot(df.index, df['LSHEResiduals'],'-b', label='LS-HE', lw=lw)
    ax[1].set_ylabel('nt')
    ax[0].set_xlim(df.index[0], df.index[-1])

    c=0
    for x in ax:
         x.grid(True)
         x.legend(loc='upper left',fontsize=6)         
         x.set_title(f'({c+1})')
         c+=1   
    for x in range(2):
        ymin,ymax= ax[x].get_ylim()
        #[3,7,8,11,13, 14,15,16,17,20,21,22,23,24,25,26,27,29]
        ax[x].fill_between(['2018-05-04','2018-05-05'],[ymin,ymin],[ymax,ymax], color='b',alpha=0.1)
        ax[x].fill_between(['2018-05-12','2018-05-13'],[ymin,ymin],[ymax,ymax], color='b',alpha=0.1)
        ax[x].fill_between(['2018-05-08','2018-05-10'],[ymin,ymin],[ymax,ymax], color='b',alpha=0.1)
        ax[x].fill_between(['2018-05-14','2018-05-19'],[ymin,ymin],[ymax,ymax], color='b',alpha=0.1)
        ax[x].fill_between(['2018-05-21','2018-05-29'],[ymin,ymin],[ymax,ymax], color='b',alpha=0.1)
        ax[x].fill_between(['2018-05-30','2018-05-31'],[ymin,ymin],[ymax,ymax], color='b',alpha=0.1)
     
    compareResiduals(ax[6], ax[7], df["Residuals"].values, df['LSHEResiduals'].values, name1="EigenSig", name2="LSHE")
    pltS(ax[2], gf.S)
    plotUV(ax[3],ax[4], gf)
     
     #n = coef.shape[0]/2 
     #print(f'#f:{n}')
    P = np.zeros(len(lsheFreq), dtype=float) #,coef
    freq = np.array(lsheFreq)
    for i in range(P.shape[0]):
         P[i] = coef[2*(i+1)]**2+coef[2*(i+1)+1]**2
    idx = np.argsort(freq)
    F_sorted = freq[idx]
    P_sorted = P[idx]

    ax[5].plot(F_sorted,P_sorted, 'or', markersize=3 )
    ax[5].fill_between(F_sorted,0.0, P_sorted)
    ax[5].set_xlabel('frequency(1/hour)')
    ax[5].set_ylabel('Coef.')
    ax[5].set_yscale('log')
    ax[5].tick_params(axis='y', labelrotation=90)

     #plt.tight_layout()
    plt.show()  

def figure6_addFocus(T_del, TS, T, delList,args, df, TS4plt, prtStats=False):
     """_summary_

    Args:
        T_del (_type_): _description_
        TS (_type_): _description_
        T (_type_): _description_
        delList (_type_): _description_
        args (_type_): _description_
        df (_type_): _description_
        TS4plt (_type_): _description_
        prtStats (bool, optional): _description_. Defaults to False.
     """
     
     EigenRes    = util.stats(delList, args.period, df[args.chName].values, df['GapFilled'].values)
     LSHERes     = util.stats(delList, args.period, df[args.chName].values, df['LSHEFilled'].values)

     #cuRes    = stats(delList, args.period, df[args.chName].values, cubic)    
     if prtStats:
         print ('gap Fill stats')
         print ('min, max, mean, std')
         print(f'EigenSig: {np.min(EigenRes):.3f},{np.max(EigenRes):.3f}, {np.mean(EigenRes):.3f}, {np.std(EigenRes):.3f}')
         print(f'LSHE:     {np.min(LSHERes):.3f},{np.max(LSHERes):.3f}, {np.mean(LSHERes):.3f}, {np.std(LSHERes):.3f}')
         
         #print(f'{np.min(cuRes):.3f},{np.max(cuRes):.3f}, {np.mean(cuRes):.3f}, {np.std(cuRes):.3f}')
    
     #fig, ax = plt.subplots(4, figsize=(12,9) ,sharex=True)    
     # 1. Create the figure and axes
     # We define 4 rows, 1 column. 
     # height_ratios=[1, 1, 1, 2] means the first 3 rows have 
     # height 1, and the last has height 2 (double).
     fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(12, 10), 
                        gridspec_kw={'height_ratios': [1, 1, 1, 3]})
     linew = 0.85
     ax[0].plot(df.index, TS4plt.flatten(),'-k', markersize=3,mec='b', lw=linew, label='data with gap')   #thegap data
     
     ax[1].plot(df.index, df[args.chName], '-k',lw=linew, label='Y')
     ax[1].plot(df.index, df['GapFilled'], '--r',lw=linew,label='EigenSig')
     ax[1].plot(df.index, df['LSHEFilled'], '--b',lw=linew,label='LSHE')

     ax[2].plot(df.index, df["Residuals"],    '-r',  lw=linew,  label='EigenSig Residual')
     ax[2].plot(df.index, df['LSHEResiduals'],'--b', lw=linew,  label='LSHE Residual')     

    #focused second row
     ax[3].plot(df.index, df[args.chName], '-k',  lw=linew, label='Y')
     ax[3].plot(df.index, df['GapFilled'], '--r', lw=linew,label='EigenSig')
     ax[3].plot(df.index, df['LSHEFilled'], '--b',lw=linew,label='LSHE')
      
     ax[0].set_xlim(df.index[0], df.index[-1])
     ax[1].set_xlim(df.index[0], df.index[-1])
     ax[2].set_xlim(df.index[0], df.index[-1])
     ax[0].set_xticklabels([])
     ax[1].set_xticklabels([])
     
     c=0
     for x in ax:
         x.grid(True)
         x.legend(loc='upper left',fontsize=6) 
         x.set_ylabel('nt')
         x.set_title(f'({c+1})')
         c+=1   
     for x in range(4):
        ymin,ymax= ax[x].get_ylim()
        #[3,7,8,11,13, 14,15,16,17,20,21,22,23,24,25,26,27,29]
        ax[x].fill_between(['2016-05-04','2016-05-05'],[ymin,ymin],[ymax,ymax], color='b',alpha=0.1)
        ax[x].fill_between(['2016-05-12','2016-05-13'],[ymin,ymin],[ymax,ymax], color='b',alpha=0.1)
        ax[x].fill_between(['2016-05-08','2016-05-10'],[ymin,ymin],[ymax,ymax], color='b',alpha=0.1)
        ax[x].fill_between(['2016-05-14','2016-05-19'],[ymin,ymin],[ymax,ymax], color='b',alpha=0.1)
        ax[x].fill_between(['2016-05-21','2016-05-29'],[ymin,ymin],[ymax,ymax], color='b',alpha=0.1)
        ax[x].fill_between(['2016-05-30','2016-05-31'],[ymin,ymin],[ymax,ymax], color='b',alpha=0.1)
     
     ax[3].set_xlim(pd.to_datetime('2016-05-20'), pd.to_datetime('2016-06-01'))
     plt.tight_layout()
     plt.show()  

if __name__ == '__main__':
    """demonstrate gap filling using 1-hour interval data

    """
 
    parser=argparse.ArgumentParser(add_help=True,  prog ='gapfilling.py')    
 
    parser.add_argument('-p','--periods',action='store',dest= \
       'period',help='how many samples in a interpolation cycles, such as one day for \
       hourly, minutes, seconds data are 24, 60, and 3600, respectively.default is 24 for hourly data', \
        type=int, default= 24)    
    parser.add_argument('-l','--LanmudaSel',action='store',dest= \
       'lanmudaSel',type=str, default='0:',  help='select lanmuda in the construction, default is use all\
        , it is just like numpy slicing')
    parser.add_argument('-c','--chanellName',action='store',dest= \
       'chName',type=str, default='Y',  help='the chanell name to process, default is Y') 
    parser.add_argument('-i','--interpoEigenMode',action='store',dest= \
       'interpoEigen',type=str, default='slinear',  help='the interpolation method used in loading \
        weighting, default is spline, default is spline, possible is "linear","zero","slinear","quadratic", \
        "cubic"')     
    parser.add_argument('-s', '--startDay', action='store', dest= \
        'startDay', help='the start day of processing, default is 2018-05-01', default='2018-05-01')
    parser.add_argument('-e', '--endDay', action='store', dest= \
           'endDay', help='the end day of processing, default is 2018-06-01 ', default='2018-06-01')    
    parser.add_argument('-f','--numOfFrequency',action='store',dest= \
       'max_freqs',type=int, default=20,  help='maximum number of frequencies, default is 20')
    parser.add_argument('-m','--numOfExageration',action='store',dest= \
       'ofac',type=int, default=5,  help='times of multifier for frequencies, default is 5')
    parser.add_argument('-a','--alpha',action='store',dest= \
       'alpha',type=float, default=0.01,  help='confidence used in frequency component selection, default is 0.01')
    #max_freqs=20, ofac=5,alpha=0.01
   
    args= parser.parse_args()    
    df_data = dataSamples.load_Ott_Hour()      
    chanell = args.chName  #'Y'   
    df = df_data.loc[args.startDay:args.endDay] 
    print(f'rows={df[chanell].values.shape[0]}')
    rows = df[chanell].values.shape[0] // args.period    
    df   = df.iloc[:rows*args.period].copy()   #extract exactly the series of full periods
    
    #create the data for gap filling
    allDis=np.array([x for x in range(rows)], dtype=float)
    allTS =df[chanell].values
    T = np.array([x for x in range(allTS.shape[0])])
    #crate gaps
    delList=[3,7,8,11,13, 14,15,16,17,20,21,22,23,24,25,26,27,29]  
    TS4plt = allTS.copy().reshape(rows,-1)     
    
    for dl in delList:
          TS4plt[dl,:] = np.nan
    dis = np.delete(allDis,delList, axis=0)

    TS  = np.delete(allTS.reshape(rows,-1),delList, axis=0).flatten()  
    print(f'Known points:{TS.shape[0]}') 
    T_del = np.delete(T.reshape(rows,-1),delList, axis=0).flatten()  
    
    starttime  = time.perf_counter() #process_time()
    gf = EigenSig.GapFilling()
    gf.fit(dis,TS)
    filled = gf.predict(allDis)     
    endtime  =  time.perf_counter()   #.process_time()
    elapsed = endtime - starttime     
    print(f"EigenSig used time: {elapsed:.4e} seconds")
    
    df['GapFilled'] = filled
    df['Residuals'] = df[args.chName] - df['GapFilled']   

    #LS-HE 
    starttime  = time.perf_counter()  #process_time()  # cpu time only, not wall time
    
    lshe = LSHE.LSHE()
    lshe.fit(T_del, TS)
    lsHE = lshe.predict(T)
    lsHEres = lshe.frequencies_  #frequencies?
    coef= lshe.weights_
    
    print(f'Frenquency #:{len(lsHEres)}')
    print(f'cofs:{coef.shape}')
          
    #auto_lshe_amiri_plus
    print(f'{lsHEres}' )
    fsel = np.array(lsHEres, dtype=float)
    endtime  =  time.perf_counter()   #process_time()
    elapsed = endtime - starttime 
    print(f"LSHE used time: {elapsed:.4e} seconds")
    np.set_printoptions(precision=4, suppress=False)
    print('Selected frequences:', fsel)

    df['LSHEFilled']    = lsHE
    df['LSHEResiduals'] = lsHE - df[chanell].values  
 
    figure6(gf, T_del, TS, T, delList,args, df, TS4plt,lsHEres,coef, prtStats=True) #it works but 4 row with nonclear

    #figure6_addFocus(T_del, TS, T, delList,args, df, TS4plt, prtStats=True)   
    #compareResiduals(df['Residuals'].values, df['LSHEResiduals'].values, name1='EigenSig',name2='LS-HE')    
    #util.compare_residuals_all(df['Residuals'].values,df['LSHEResiduals'].values, "EigenSig", "LS-HE")
    #residual_scatter_with_density(df['Residuals'].values,df['LSHEResiduals'].values, "EigenSig", "LS-HE")
    #bland_altman_plot(df['Residuals'].values,df['LSHEResiduals'].values, "EigenSig", "LS-HE")
    #qq_plot_residuals(df['Residuals'].values,df['LSHEResiduals'].values, "EigenSig", "LS-HE")

    print('done')    