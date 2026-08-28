import numpy as np
import time

from pathlib import Path
import sys
project_root = Path(__file__).resolve().parents[1]  #Path.cwd().parent
sys.path.insert(0, str(project_root))

#step 1: import the PyGap
from PyGap import dataSamples as ds
from PyGap import EigenSig
from PyGap import LSHE
from PyGap import charts

def test_EigenSig():
    df_data = ds.load_Ott_Hour()      
    chanell = 'Y'          # select the column 
    df = df_data.loc['2018-05-01':'2018-06-01'] 

    rows = df[chanell].values.shape[0] // 24  #24 is the epoch: 1 day records have 24 samples for 1-hour data
    df   = df.iloc[:rows*24].copy()           #extract exactly the series of full periods
    #print(df.head(5))

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

    #step 4: construct gap filling object, fit model using data with gaps. then filling (predict) gaps in the data. 
    gf = EigenSig.GapFilling()      
    gf.fit(dis,TS)
    filled = gf.predict(allDis, inter1D='slinear', filtStr='0:')  
    
    assert filled is not None, 'EigenSig Objest is None '

if __name__ == "__main__":
  test_EigenSig()
  print("data example test passed")