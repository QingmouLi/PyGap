# -*- coding: utf-8 -*-
"""
functions and classes of scikit-EigenSig

Created on Wed Aug 19 00:28:34 2024
@author: Qingmou Li & Jon W. Liu, Geological Survey of Canada
"""
import numpy as np
import pandas as pd
from scipy import interpolate
import statsmodels.api as sm
from . import LSHE

def CreateFilter(S, str='0:'):
    """create filtering str based on singular values

    Args:
         S(str): numpy type array slicing 
    Notes:    
         the str is numpy style numpy slicer, for example, 
         str= "0:"  uses all lanmubda
         str= ":7"  uses [0,1,2,3,4,5,6] th lamubda
         str= "2:4" uses [2,3] th lanmubda
         S: a vector of singular values from SVD

    Returns:
        numpy.dnarray: numpy 1D array
    """

    ic,jc = str.split(':')
    i = 0          if ic=='' else int(ic)
    j = S.shape[0] if jc=='' else int(jc)
    filter = np.zeros_like(S)
    filter[i:j] = 1.0     
    Sfil = S*filter    
    return Sfil

def _D1SARIMA(dis,y, Dis, order=(4,1,2), alpha=0.05):
    '''
    coordinates tracing using tranditional and ARMA modeling
    it gives the uncertainty for SARIAV method
    dis[:]
    y[n]
    Dis[:] :all distance but Dis[:] should be Dis[0]>=dis[0], Dis[-1]<=dis[-1], this emans interpolate only
    Y[:]:
    'linear', 'zero', 'slinear', 'quadratic', 'cubic'      
     zero, slinear, quadratic and cubic refer to a spline interpolation of 
     zeroth, first, second or third order; 
    '''      
    
    dta = pd.DataFrame(data=y, index = dis,columns = ['Activity'])    
    model = sm.tsa.ARIMA(dta, order=order, trend='c').fit()     
    fpre = model.get_prediction(start=int(Dis[0]), end= int(Dis[-1]), dynamic = 1).summary_frame() # infromation_set = "filtered"
    UU = fpre['mean'].values
    UU_lower = fpre['mean_ci_lower'].values
    UU_upper = fpre['mean_ci_upper'].values    

    return UU, UU_lower, UU_upper 

def EigenAnomaly(dis, TS, inter1D, filtStr='0:', prtStats=False):
    """anomaly separation using iPCA to calculate the diference between raw and
            reconstructed from other than current period.

    Args:
        dis (numpy.ndarray): sampling series
        TS (numpy.ndarray): sampled data
        inter1D (str): _description_
        filtStr (str, optional): _description_. Defaults to '0:'.
        prtStats (bool, optional): _description_. Defaults to False.

    Returns:
        (numpy.ndarray): Separated anomaly
    """

    arr = TS.reshape(dis.shape[0], -1)         
    S1 =0      
    Ano = arr.copy()
    if arr.shape[1] > arr.shape[0]:
        UU = np.zeros((1,arr.shape[0]-1), dtype=float)
    else:
        UU = np.zeros((1,arr.shape[1]), dtype=float)                     
    for r in range(1, arr.shape[0]-1):             
             disTemp = np.delete(dis, r, axis=0)
             arrTemp = np.delete(arr, r, axis=0)             
             U,S,V = np.linalg.svd(arrTemp,full_matrices=False)
             S1 =S
             Sel = CreateFilter(S, filtStr)             
             for c in range(U.shape[1]):
                    f = interpolate.interp1d(disTemp, U[:,c], kind=inter1D, assume_sorted=True)
                    UU[0,c] = f(r)                    
                    ArrRow = np.dot(UU, np.dot(np.diag(Sel),V))
                    Ano[r,:] = ArrRow[:]
    per = 100.0*S**2/np.sum(S**2)
    if prtStats:
            print(f'Energy distribution(%):{per[0]:.5f} {per[1]:.5f} {per[2]:.5f}')
    return Ano.flatten()

class GapFilling():
    """
    class for gap filling using eigenspace
    """    
    def __init__(self):
        """initialize the class
        """
        pass
        '''
        dis[] is the distance from the first point, dis[0]=0.0 and dis[-1] is the maximum distance we can do
        gap filling
        TS[:] is the time series 
        '''    
    def fit(self, dis,TS):
        """training models with given daat

        Args:
            dis (numpy.ndarray): epoch number of observations
            TS (numpy.ndarray): observed data 
        Note:
            dis.shape[0]*epoch = TS.shape[0]
        """

        self.dis=dis
        arr = TS.reshape(dis.shape[0],-1)
        self.U,self.S,self.V=np.linalg.svd(arr,full_matrices=False)

    def predict(self, fullDis, inter1D='slinear', filtStr='0:'):
        """gap filling 
        fulldis is the distance vector to fill every point(known or unknown)

        Args:           
            inter1D (str): coordinate tracing in the eigenspace. It can be one of:
                'slinear', 
                'nearest'
                'nearest-up'
                'zero'            
                'quadratic'
                'cubic'
                'previous'
                'next'          
         filtStr: a numpy style eigenvalues selection for filtering while interpolation
            fullDis (_type_): the distance vector to fill every point(known or unknown)
            inter1D (str, optional): coordinate tracing in the eigenspace. Defaults to 'slinear'.
                    it could be one of:
             'slinear', 
             'nearest'
             'nearest-up'
             'zero'            
             'quadratic'
             'cubic'
             'previous'
             'next'                     
            filtStr (str, optional): numpy style filtering selection. Defaults to '0:'.

        Returns:
            numpy.ndarray: the gap filled time series at fulldis points
        """

        
        SFil = CreateFilter(self.S, filtStr)
        fullU = np.zeros((fullDis.shape[0],self.U.shape[1]),dtype=float)
        for c in range(self.U.shape[1]):  #loop every column
            f  = interpolate.interp1d(self.dis, self.U[:,c], kind=inter1D, assume_sorted=True)    # _D1Interpolate(self.dis, self.U[:,c], fullDis, inter1D) #interpolate U            
            newColU = f(fullDis)
            fullU[:,c] = newColU[:]
        fullDataArr = np.dot(fullU[:,:],np.dot(np.diag(SFil[:]),self.V[:,:]))
        self.UU = fullU
        return fullDataArr.flatten()       

    def predictUseLSHE(self, fullDis, args, inter1D='slinear', filtStr='0:'):
        '''
        
        use LS-HE to trace coordinates in the eigenspace

        gap filling 
        fulldis is the distance vector to fill every point(known or unknown)
        return the gap filled time series at fulldis points
        filtStr: a numpy style eigenvalues selection for filtering while interpolation
        inter1D: 
        return a new time series at each point of fullDis
        '''
        SFil = CreateFilter(self.S, filtStr)
        fullU = np.zeros((fullDis.shape[0],self.U.shape[1]),dtype=float)
        # only the first N (1) eigenvector are used for interpolate

        newColU, _, fsel    = LSHE.auto_lshe_amiri(self.dis, self.U[:,0], fullDis, max_freqs=args.max_freqs, ofac=args.ofac, alpha=args.alpha)
        fullU[:,0] = newColU[:] 
        for c in range(1, self.U.shape[1]):  #fast loop other column
            f  = interpolate.interp1d(self.dis, self.U[:,c], kind=inter1D, assume_sorted=True)    # _D1Interpolate(self.dis, self.U[:,c], fullDis, inter1D) #interpolate U            
            newColU = f(fullDis)
            fullU[:,c] = newColU[:]
        fullDataArr = np.dot(fullU[:,:],np.dot(np.diag(SFil[:]),self.V[:,:]))
        self.UU = fullU
        return fullDataArr.flatten()       

    def predictUseUncertinty(self, fullDis, args=None, inter1D='slinear', filtStr='0:'):
        '''
        
        use LS-HE to trace coordinates in the eigenspace and calculate uncertainty

        gap filling 
        fulldis is the distance vector to fill every point(known or unknown)
        return the gap filled time series at fulldis points
        filtStr: a numpy style eigenvalues selection for filtering while interpolation

        return a new time series at each point of fullDis
        '''
        SFil = CreateFilter(self.S, filtStr)
        fullU = np.zeros((fullDis.shape[0],self.U.shape[1]),dtype=float)
        lowU = np.zeros((fullDis.shape[0],self.U.shape[1]),dtype=float)
        highU = np.zeros((fullDis.shape[0],self.U.shape[1]),dtype=float)
        # only the first N (1) eigenvector are used for interpolate

        #newColU, _, fsel    = LSHE.auto_lshe_amiri(self.dis, self.U[:,0], fullDis, max_freqs=args.max_freqs, ofac=args.ofac, alpha=args.alpha)
        newColU, low_b, high_b, fsel = LSHE.auto_lshe_amiri_confidence(self.dis, self.U[:,0], fullDis, max_freqs=2, ofac= 5, alpha= 0.05) #max_freqs=args.max_freqs, ofac=args.ofac, alpha=args.alpha)

        #the first eigen is the major and represent global variations
        fullU[:,0] = newColU[:] 
        lowU[:,0] =  low_b[:] 
        highU[:,0] = high_b[:] 
        for c in range(1, self.U.shape[1]):  #fast loop other coordinates (columns)
            f  = interpolate.interp1d(self.dis, self.U[:,c], kind=inter1D, assume_sorted=True)    # _D1Interpolate(self.dis, self.U[:,c], fullDis, inter1D) #interpolate U            
            newColU = f(fullDis)
            fullU[:,c] = newColU[:]
            lowU[:,c]  = newColU[:]
            highU[:,c] = newColU[:]

        fullDataArr = np.dot(fullU[:,:],np.dot(np.diag(SFil[:]),self.V[:,:]))
        lowBound    = np.dot(lowU[:,:],np.dot(np.diag(SFil[:]),self.V[:,:]))
        highBound   = np.dot(highU[:,:],np.dot(np.diag(SFil[:]),self.V[:,:]))
        self.UU = fullU
        self.lowU = lowU
        self.highU = highU

        return fullDataArr.flatten(), lowBound.flatten(),highBound.flatten()       

class GapFillingUncertainty():
    '''
    Uncertainty esitmation for gap filling
    '''
    def __init__(self):
        pass
    def fit(self, dis,TS):
        '''
        dis[] is the distance from the first point, dis[0]=0.0 and dis[-1] is the maximum distance we can do
        gap filling
        TS[:] is the time series 
        '''
        self.dis=dis
        arr = TS.reshape(dis.shape[0],-1)
        self.U,self.S,self.V=np.linalg.svd(arr,full_matrices=False)        
        
    def predict(self, fullDis, order=(3,0,0), filtStr='0:', prtStats=False):
        '''
        gap filling 
        fulldis is the distance vector to fill every point(known or unknown)
        return the gap filled time series at fulldis points
        filtStr: a numpy style eigenvalues selection for filtering while interpolation
        '''
        SFil = CreateFilter(self.S, filtStr)
        fullU = np.zeros((fullDis.shape[0],self.U.shape[1]),dtype=float)
        Ulower = np.zeros((fullDis.shape[0],self.U.shape[1]),dtype=float)
        Uupper = np.zeros((fullDis.shape[0],self.U.shape[1]),dtype=float)
        for c in range(self.U.shape[1]):  #loop every column
            #f  = interpolate.interp1d(self.dis, self.U[:,c], kind=inter1D, assume_sorted=True)    # _D1Interpolate(self.dis, self.U[:,c], fullDis, inter1D) #interpolate U
            UU, UUlower, UUupper =  _D1SARIMA(self.dis,  self.U[:,c], fullDis, order=(3,0,0), alpha=0.05)              
            #newColU = f(fullDis)
            fullU[:,c]   = UU[:]
            Ulower[:,c]  = UUlower[:]
            Uupper[:,c]  = UUupper
        #print('before construct')
        fullDataArr  = np.dot(fullU[:,:],np.dot(np.diag(SFil[:]),self.V[:,:]))
        dataArrlower = np.dot(Ulower[:,:],np.dot(np.diag(SFil[:]),self.V[:,:]))
        dataArrupper = np.dot(Uupper[:,:],np.dot(np.diag(SFil[:]),self.V[:,:]))
        self.UU = fullU
        if prtStats:
            per = 100.0*self.S**2/np.sum(self.S**2)
            print(f'energy dist. {per[0]:.5f}  {per[1]:.5f} {per[2]:.5f} ')
            print(f'fill  stats: min={np.min(fullDataArr.flatten()):.1f}  max={np.max(fullDataArr.flatten()):.1f}  mean={np.mean(fullDataArr.flatten()):.1f}  std={np.std(fullDataArr.flatten()):.2f}')
            print(f'upper stats: min={np.min(dataArrupper.flatten()):.1f}  max={np.max(dataArrupper.flatten()):.1f}  mean={np.mean(dataArrupper.flatten()):.1f}  std={np.std(dataArrupper.flatten()):.2f}')        
            print(f'lower stats: min={np.min(dataArrlower.flatten()):.1f}  max={np.max(dataArrlower.flatten()):.1f}  mean={np.mean(dataArrlower.flatten()):.1f}  std={np.std(dataArrlower.flatten()):.2f}')        
        
        return fullDataArr.flatten(), dataArrlower.flatten(), dataArrupper.flatten()      
