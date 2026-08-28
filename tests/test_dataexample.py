from pathlib import Path
import sys
project_root = Path(__file__).resolve().parents[1]  #Path.cwd().parent
sys.path.insert(0, str(project_root))

import numpy as np

import pandas as pd
from PyGap import dataSamples as ds
from PyGap import EigenSig 
from PyGap import dataSamples
from PyGap import LSHE
from PyGap import utility as util

import seaborn as sns    
from scipy.stats import skew, kurtosis,gaussian_kde, probplot

   
def test_dataexample():
   df_data = ds.load_Ott_Hour()      
   chanell = 'Y'   # select the column
   df = df_data.loc['2018-05-01':'2018-06-01'] 

   rows = df[chanell].values.shape[0] // 24  #args.period    
   df   = df.iloc[:rows*24].copy()           #extract exactly the series of full periods 
   assert df is not None, 'load datamodule returns None'

if __name__ == "__main__":
  test_dataexample()
  print("data example test passed")
