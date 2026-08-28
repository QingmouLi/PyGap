# Samples using PyGap

## 1. Quick samples

```Python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

#step 1: import the PyGap
from PyGap import dataSamples as ds
from PyGap import EigenSig
from PyGap import LSHE
from PyGap import charts

#step 2 load sequential data
#obs = np.loadtxt('eigenData_example.csv', delimiter=',',skiprows=1)

obs= np.array([
#epochSequence, y0,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,y19,y20,y21,y22,y23
[0.0,-4311.6,-4311.9,-4312.2,-4312.1,-4311.1,-4311.2,-4311.5,-4311.3,-4306.8,-4302.6,-4299.6,-4295.6,-4293.5,-4299.6,-4310.6,-4319.6,-4328.5,-4337.9,-4340.8,-4334.2,-4325.2,-4322.7,-4321.2,-4317.9],
[1.0,-4317.3,-4315.8,-4312.4,-4311.3,-4305.5,-4311.0,-4307.7,-4305.7,-4308.4,-4315.5,-4297.4,-4289.3,-4288.7,-4288.1,-4296.3,-4307.8,-4320.1,-4330.4,-4339.1,-4340.2,-4332.2,-4321.9,-4314.4,-4310.3],
[2.0,-4311.9,-4312.1,-4312.2,-4311.1,-4311.8,-4311.1,-4310.6,-4309.6,-4308.0,-4304.8,-4298.2,-4291.6,-4285.9,-4289.6,-4302.4,-4317.2,-4332.7,-4342.0,-4344.5,-4339.8,-4328.7,-4317.9,-4311.1,-4308.4],
[4.0,-4309.4,-4311.1,-4310.5,-4310.4,-4312.7,-4310.1,-4309.9,-4308.4,-4307.5,-4306.1,-4300.6,-4278.7,-4270.8,-4274.8,-4277.9,-4322.2,-4345.5,-4371.4,-4391.3,-4368.5,-4353.1,-4346.0,-4313.3,-4288.2],
[5.0,-4259.8,-4267.4,-4289.6,-4307.2,-4287.5,-4290.9,-4307.4,-4328.5,-4309.6,-4319.3,-4325.5,-4299.6,-4293.7,-4287.2,-4299.2,-4314.4,-4319.5,-4328.8,-4334.0,-4336.8,-4322.2,-4325.9,-4302.9,-4319.6],
[6.0,-4308.9,-4289.7,-4270.6,-4290.4,-4315.3,-4330.9,-4289.8,-4310.0,-4314.0,-4303.5,-4295.3,-4288.9,-4289.6,-4290.8,-4308.6,-4330.9,-4339.1,-4345.1,-4347.1,-4334.4,-4330.2,-4322.9,-4309.0,-4314.0],
[9.0,-4297.9,-4297.3,-4302.1,-4292.2,-4306.0,-4309.8,-4312.1,-4341.4,-4306.7,-4296.3,-4301.4,-4290.2,-4294.7,-4291.3,-4297.0,-4307.8,-4325.8,-4337.1,-4331.7,-4329.4,-4326.7,-4309.2,-4308.6,-4307.1],
[10.0,-4306.7,-4307.4,-4311.1,-4308.6,-4334.1,-4290.7,-4280.0,-4313.1,-4281.6,-4273.9,-4272.8,-4268.5,-4281.7,-4293.4,-4327.8,-4350.9,-4353.6,-4357.5,-4357.8,-4356.0,-4343.2,-4332.6,-4328.8,-4314.2],
[12.0,-4314.7,-4316.5,-4318.2,-4316.0,-4313.0,-4317.3,-4302.8,-4299.8,-4282.5,-4284.9,-4290.4,-4273.5,-4276.2,-4290.1,-4304.9,-4326.3,-4345.0,-4348.4,-4347.8,-4339.2,-4328.7,-4319.9,-4310.8,-4305.9],
[18.0,-4310.6,-4312.3,-4313.1,-4313.1,-4313.3,-4313.4,-4311.9,-4310.1,-4309.2,-4304.6,-4299.9,-4292.0,-4281.2,-4279.6,-4285.6,-4305.6,-4327.6,-4343.1,-4348.4,-4344.6,-4335.1,-4324.1,-4317.1,-4313.0],
[19.0,-4313.9,-4315.8,-4315.4,-4310.9,-4312.7,-4311.9,-4311.2,-4313.1,-4308.7,-4302.6,-4298.1,-4289.1,-4287.6,-4293.0,-4292.8,-4308.3,-4328.9,-4338.7,-4347.2,-4341.7,-4332.5,-4323.7,-4315.4,-4310.4],
[28.0,-4314.1,-4315.3,-4315.8,-4313.6,-4304.8,-4309.0,-4307.9,-4308.1,-4303.3,-4298.4,-4287.8,-4279.0,-4276.9,-4285.6,-4298.8,-4319.3,-4338.6,-4342.6,-4341.7,-4338.0,-4329.0,-4320.3,-4312.9,-4306.7],
[30.0,-4310.1,-4311.5,-4312.4,-4308.5,-4310.7,-4311.8,-4312.0,-4308.8,-4304.4,-4299.5,-4286.1,-4275.4,-4269.5,-4272.9,-4283.3,-4306.1,-4328.9,-4359.6,-4371.4,-4370.7,-4349.3,-4334.9,-4328.5,-4308.2],
[31.0,-4314.7,-4294.1,-4294.7,-4307.1,-4321.4,-4334.1,-4313.2,-4266.8,-4335.0,-4309.9,-4307.6,-4278.8,-4270.4,-4299.9,-4312.4,-4315.1,-4339.7,-4339.6,-4351.1,-4343.5,-4331.6,-4312.3,-4310.4,-4308.6]
])
t = obs[:,0]
y= obs[:,1:].flatten()  #stack epochs as one vector
T= np.array([x for x in range(int(t[0]),int(t[-1])+1)])

#step 3: construct gap filling object, fit model using data with gaps. then filling (predict) gaps in the data. 
gf = EigenSig.GapFilling()      
gf.fit(t,y)

Y = gf.predict(T, inter1D='slinear', filtStr='0:')  

#Y_m has the same structure as the input eigenData_example.csv but have every epochs filled,
# it can be save to disk.
#Y_m = np.column_stack((T, Y.reshape(-1,24)))
#np.savetxt('output.txt', Y_m, fmt='%12.4f', delimiter=',')

#plot data
plt.figure(figsize=(12, 3.5))
plt.plot((T[:, np.newaxis] * 24 + np.arange(24)).flatten(),Y,'-b',lw=0.75,label='EigenSig filled')
plt.plot((t[:, np.newaxis] * 24 + np.arange(24)).flatten(),y,'or',markersize=2,label='Observed')
tt = (t[:, np.newaxis] * 24 + np.arange(24)).flatten()
plt.xlim(tt[0],tt[-1]); plt.ylabel("Y(nt)")
plt.xlabel("hours from start time")
plt.legend()
plt.savefig('sequential.png', dpi=120, bbox_inches='tight')
```


## 2. EigenSig

EigenSig: Eigenspace-Based Gap Filling and Applications
PyGap implements the EigenSig methodology, an eigenspace-based approach for filling data gaps, separating signals, and analyzing structural patterns in spatio-temporal datasets. Unlike harmonic approaches that rely on predefined Fourier components, EigenSig is entirely data-driven, extracting dominant structures directly from the observations. This makes it particularly suitable for geophysical, environmental, and climate datasets that contain irregular sampling, long data gaps, transient events, and non-stationary behavior.

Purpose
The primary goal of EigenSig is to reconstruct missing observations while preserving the intrinsic structures of the original signal. In addition to gap filling, it supports signal separation, anomaly detection, noise reduction, and energy-structure analysis. By exploiting auto-correlations in the data, EigenSig can recover both global trends and localized variations without imposing a stationarity assumption.

Method and Mathematical Foundation
The method begins by partitioning a time series into epochs and reshaping it into an embedded matrix Y. Principal Component Analysis (PCA) or Singular Value Decomposition (SVD) is then applied:


where U and V are orthogonal eigenvector matrices and Σ contains singular values. The decomposition separates the signal into a set of ranked eigenspace components, revealing how energy is distributed among dominant structures. Because most signal energy is typically concentrated in a few leading modes, the dataset can be represented in a low-dimensional subspace while preserving its essential characteristics.

For gap filling, EigenSig reconstructs the signal by interpolating trajectories in the reduced eigenspace rather than directly fitting the original observations. Missing values are estimated through the reconstructed eigenspace coordinates: The decomposition can also be expressed as:


This approach exploits intrinsic structure within the data and avoids explicit modeling of high-frequency behavior.

A key advantage of EigenSig is its ability to preserve local variability, sharp gradients, and transient features, which are often smoothed or distorted by conventional interpolation methods. It also removes the requirement for a global stationarity assumption and is computationally efficient because it relies on low-rank matrix representations rather than repeated global optimizations.

In summary, EigenPy provides a robust framework for reconstruction, anomaly separation, and structural analysis of incomplete spatio-temporal datasets. Its eigenspace formulation enables accurate gap filling while maintaining both global trends and local signal characteristics, making it especially suitable for non-stationary geophysical observations the original signal domain.

Key Features
Data-driven eigenspace representation.
No global stationarity assumption.
Preserves local variability and sharp transitions.
Handles wide and continuous data gaps.
Efficient low-rank reconstruction using SVD.
Supports anomaly separation and noise filtering.
Provides energy-structure analysis through singular spectrum analysis (SSA)
5 steps (indicated in the demo codes)
import pyGap package
prepare .csv data, import it into Pandas's dataframe, the DateTime column follow standard Python format,see examples with
construct GapFilling object
fill gaps using gp.predict() with
dump, chart results and analysis
Discussion
Unlike traditional interpolation techniques such as splines or kriging, or advanced least square harmonic analysis (LSHE), EigenSig does not rely on fixed functional forms or stationary covariance assumptions. Instead, it learns dominant structures directly from the data, allowing it to capture both long-term behavior and local transient features. Because reconstruction occurs within a low-dimensional eigenspace, the method is computationally efficient while maintaining high fidelity to the original signal. Comparative studies show that EigenSig generally preserves localized features better than globally fitted models and performs particularly well for non-stationary geophysical observations.

Conclusion
EigenPy provides an effective eigenspace framework for gap filling, signal reconstruction, anomaly separation, and structural analysis. By combining matrix embedding, PCA/SVD decomposition, and eigenspace interpolation, it reconstructs missing observations while preserving the natural variability of the data. The method is especially valuable for complex non-stationary spatio-temporal datasets where maintaining local signal characteristics is critical.

Referrence
Li, Qingmou and Liu, Wenjun, 2026. Gap Filling and Anomaly Separation Using Inverse Principal Components Analysis, Pure Appl. Geophys., https://doi.org/10.1007/s00024-026-04053-5

```Python
import numpy as np
import time

from pathlib import Path
import sys
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

#step 1: import the PyGap
from PyGap import dataSamples as ds
from PyGap import EigenSig
from PyGap import LSHE
from PyGap import charts

#step 2: load asc data (.csv) into Pandas' DataFrame, display the first 5 rows 
df_data = ds.load_Ott_Hour()      
chanell = 'Y'          # select the column 
df = df_data.loc['2018-05-01':'2018-06-01'] 

rows = df[chanell].values.shape[0] // 24  #24 is the epoch: 1 day records have 24 samples for 1-hour data
df   = df.iloc[:rows*24].copy()           #extract exactly the series of full periods
print(df.head(5))

#step 3: prepare synthetic data gaps 
allDis=np.array([x for x in range(rows)], dtype=float) #allDis the number of rows in the continous (full) time series without data gaps
allTS =df[chanell].values
T = np.array([x for x in range(allTS.shape[0])])
#synthetic gaps
delList=[3,7,8,11,13, 14,15,16,17,20,21,22,23,24,25,26,27,29]  
TS4plt = allTS.copy().reshape(rows,-1)     
    
for dl in delList:
    TS4plt[dl,:] = np.nan
df['SynthticGaps'] = TS4plt.flatten()
dis = np.delete(allDis,delList, axis=0)     #sequence of rows in the folder time series with data gaps  

TS  = np.delete(allTS.reshape(rows,-1),delList, axis=0).flatten() #TS is 1D time series with gaps removed
T_del = np.delete(T.reshape(rows,-1),delList, axis=0).flatten()  
"""
#dump to show favorite data for EigenSig as shown in eigenData_example.csv 

with open('eigenData_example.csv', 'w') as hfile:
    hfile.write('epochSequence, y0,')
    for r in range(1,23):
        hfile.write(f'y{r},')
    hfile.write('y23\n')
    
    for ep in range(dis.shape[0]):        #dump epoch by epoch
        hfile.write(f'{dis[ep]},')
        for m in range(23):   #df['SynthticGaps'].values.shape[0]/24):  #how many epochs
            hfile.write(f'{TS[ep*24+m]:.1f},')
        hfile.write(f'{TS[ep*24+23]:.1f}\n')    #the last one       
"""

#step 4: construct gap filling object, fit model using data with gaps. then filling (predict) gaps in the data. 
gf = EigenSig.GapFilling()
      
gf.fit(dis,TS)

filled = gf.predict(allDis, inter1D='slinear', filtStr='0:')  
'''
inter1D can be one of following (supported from SciPy):
            'slinear', 
            'nearest'
            'nearest-up'
            'zero'            
            'quadratic'
            'cubic'
            'previous'
            'next' 
filtStr:
     ='0:'
'''
df['EigenSigFilled'] = filled                                 # save gap filled results into the dataframe
df['EigenSigResiduals'] = df[chanell] - df['EigenSigFilled']  #save residuals into the dataframe   

# step 5: display gap filled results and plot SSA, ...,...
print(df.head(5))
charts.plotEigenSigFilled(df)  #plot curves
charts.chartEigenSig(gf)       #plot energy structure, loadings and basis of eigenspace

```

## 3. LSHE

LSHE: Least Squares Harmonic Estimation (LSHE) for Gap Filling
LSHE (Least Squares Harmonic Estimation) is a spectral-analysis and signal-reconstruction method implemented in PyGap for filling missing observations, detecting significant frequencies (Amiri-Simkooei, 2007). Originally developed for irregularly sampled geodetic time series, LSHE models an observed signal as a combination of trend (first order polynomial), harmonic components (Fourier series), and stochastic noise (white or color noises). Because it operates directly on irregular observations, it is well suited to geophysical and environmental datasets where continuous data gaps are common.

Purpose
The primary objective of LSHE is to identify dominant periodic signals and reconstruct missing observations through a statistically rigorous least-squares framework. In addition to gap filling, the method can be used for spectral analysis, denoising, uncertainty estimation, and signal separation. Unlike conventional interpolation methods, LSHE explicitly models the physical frequency content of the observations.

Mathematical Foundation
LSHE represents the observed time series 
 as


where:

 is the constant offset (mean level) of the signal.
 is the linear trend coefficient describing long-term growth or decay.
 denotes the angular frequency of the 
-th harmonic component.
 and 
 are the cosine and sine coefficients that determine the amplitude and phase of each harmonic.
 is the total number of significant frequencies included in the model.
 represents measurement noise and unexplained residual variations.
In matrix form,


where:

 is the observation vector,
 is the design matrix containing trend and harmonic basis functions,
 is the vector of unknown model parameters,
 is the residual noise vector.
The LSHE model parameters are estimated using Generalized Least Squares (GLS):


where:

 is the noise covariance matrix,
 for white noise,
 may be a full non-diagonal covariance matrix for colored noise,
 denotes the transpose of the design matrix.
Once the parameters have been estimated, the deterministic signal can be evaluated at missing epochs to reconstruct observations, fill data gaps, and reduce noise.

Key Features
Harmonic analysis of irregularly sampled data.
Automatic detection of significant frequencies using Lomb-Scargle spectral analysis.
Statistical significance testing with false-alarm probability and F-tests.
Reconstruction of missing observations through harmonic modeling.
Support for white and colored noise models.
Formal uncertainty estimation through covariance propagation.
Residual analysis for anomaly identification and data quality assessment.
Processign steps
load pygap package
load asc data into Panda's dataframe
create LSHE object and fit with previously loaded data
reconstruct series using the LSHE model
dump detected frequencies and plot filled data gaps
Example
from pygap import LSHE

lshe = LSHE()

# Fit the model and detect significant frequencies
lshe.fit(time, signal_with_gaps)

# Reconstruct missing observations
reconstructed = lshe.fill()

# Retrieve detected frequencies
frequencies = lshe.frequencies_

print("Detected frequencies:", frequencies)
Discussion
LSHE provides smooth and globally consistent reconstructions because all observations contribute to a single optimization process. The method is particularly effective for stationary signals dominated by periodic behavior such as tides, seasonal cycles, and geophysical oscillations. A major advantage is its rigorous statistical formulation, which quantifies uncertainty in both parameter estimates and reconstructed values. However, because LSHE assumes global stationarity and performs a global fit, it may smooth localized variations and transient events. Computational cost also increases with dataset size due to repeated matrix inversions in the least-squares solution.

Conclusion
LSHE is a powerful Fourier-based reconstruction method that combines spectral analysis with least-squares estimation. It excels at recovering smooth, globally coherent signals, quantifying uncertainty, and identifying dominant frequencies within incomplete time series. Within PyGap, LSHE complements EigenSig by providing a statistically robust harmonic framework for gap filling and signal analysis, particularly when the underlying process is approximately stationary and periodic.

Reference
Amiri-Simkooei, A.R. 2007. Least-squares variance component estimation: theory and GPS applications. PhD thesis, Delft University of Technology, Publication on Geodesy, 64, Netherlands Geodetic Commission, Delft

```Python
# Step 1: load the PyGap Package
from pathlib import Path
import sys
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))
from PyGap import dataSamples as ds
from PyGap import EigenSig
from PyGap import LSHE
from PyGap import charts
import numpy as np
import time
import matplotlib.pyplot as plt

# Step 1: load the PyGap Package
from pathlib import Path
import sys
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))
from PyGap import dataSamples as ds
from PyGap import EigenSig
from PyGap import LSHE
from PyGap import charts
import numpy as np
import time
import matplotlib.pyplot as plt
```


## 4. Figure_PyGap

```Python
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
```