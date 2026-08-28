# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 00:28:34 2020

@author: qli
"""

import numpy as np
import pandas as pd
import io
import pkgutil

def load_Ott_Hour():
      '''
      Geomagnetic recording in Ottawa, Canada, sampled in 1-hour interval
      during[2016-05-01 00:00:00.000, 2018-06-10 23:00:00.000],
      data source: Natural Resources Canada 
      '''
      data = pkgutil.get_data(__package__, r"./data/ottawa_mag_hour.csv")    
      strs = data.decode("utf-8") 
      df = pd.read_csv(io.StringIO(strs),  sep=',', comment="#")            
      clNames = list(df.columns)
      if clNames[0] =='DateTime':
            df['DateTime'] = pd.to_datetime(df['DateTime']) 
            df['DateTime'] = df['DateTime']    
            df.index = df['DateTime']
            df.drop(columns=['DateTime'], inplace=True)      
      return df

def load_Ott_Minute():
      '''
      load the geomagnetic obs  at Ottawa, Canada
      return a DataFrame 
      Geomagnetic recording in Ottawa, Canada, sampled in 1-hour interval
      during [2020-01-01 00:00:00.000, 2020-01-28 23:59:00.000]
      data source: from Natural Resources Canada 

      '''  
      data = pkgutil.get_data(__package__, r"./data/ott_min.csv")    
      strs = data.decode("utf-8") 
      df = pd.read_csv(io.StringIO(strs),  sep=',', comment="#")  #need test          
      clNames = list(df.columns)
      if clNames[0] =='DateTime':
            df['DateTime'] = pd.to_datetime(df['DateTime']) 
            df['DateTime'] = df['DateTime']    
            df.index = df['DateTime']
            df.drop(columns=['DateTime'], inplace=True)      
      return df

def load_ap():
      '''
      Geomagnetic indices (ISGI, http://isgi.unistra.fr)
      date time range: [2018-03-14 12:00:00.000, 2018-06-19 00:00:00.000]
      '''
      data = pkgutil.get_data(__package__, r"./data/ap.csv")    
      strs = data.decode("utf-8") 
      df = pd.read_csv(io.StringIO(strs),  sep=',', comment="#")  
      df['DateTime'] =pd.to_datetime(df['DateTime']) 
      df.index = df['DateTime']
      df.drop(columns=['DateTime','DOY','Ap','Cp','C9','BSRN','NdB'], inplace=True)

      return df        

def load_Bouguer_profile():
      '''
      load bouguer profile
      data source: Peter, J.M., and Czarnota, K., 2023, National-scale geophysical, 
      geologic, and mineral resource data and grids for the United States, Canada, 
      and Australia: Data in support of the tri-national Critical Minerals Mapping 
      Initiative: U.S.  Geological Survey data release, https://doi.org/10.5066/P970GDD5.
      '''
      data = pkgutil.get_data(__package__, r"./data/bougProfile.csv")    
      strs = data.decode("utf-8") 
      df = pd.read_csv(io.StringIO(strs),  sep=',', comment="#")            
          
      return df

def load_climate():
      '''
      Resources: Open Canada data, at 
      https://climate-change.canada.ca/climate-data/#/daily-climate-data 
      date time range: [1897-01-01 00:00:00,12/31/1906 0:00]
      '''
      data = pkgutil.get_data(__package__, r"./data/temp_precipation.csv")    
      strs = data.decode("utf-8")
      df = pd.read_csv(io.StringIO(strs),  sep=',', comment="#")   
      df['DateTime'] = pd.to_datetime(df['DateTime'], format='mixed') #lambda d : pd.to_datetime(d).strftime('%Y-%m-%d') if not pd.isnull(d) else ''
      df['DateTime'] = df['DateTime']    #-pd.Timedelta('5H')  #
      df.index = df['DateTime']
      df.drop(columns=['DateTime'], inplace=True)
      df = df.sort_index()     
      df = df.resample('ME').mean()    
      return df

def load_CBB():
      '''
      Geomagnetic recording in CBB, Canada, sampled in 1-hour interval
      during [1992-05-01 00:00:00.000, 1992-06-06 15:00:00.000]
      data source: from Natural Resources Canada 
      '''
      data = pkgutil.get_data(__package__, r"./data/cbb_mag_hour.csv")    
      strs = data.decode("utf-8")
      df = pd.read_csv(io.StringIO(strs),  sep=',', comment="#")         
      df['DateTime'] = pd.to_datetime(df['DateTime'])       
      df.index = df['DateTime']
      df.drop(columns=['DateTime'], inplace=True)    
      df.replace(to_replace= 99999.0, value= np.nan,inplace=True )  #99999.0 is badvalue 
      df.replace(to_replace= 88888.0, value= np.nan,inplace=True )  #88888.0 is badvalue 

      return df

def simulateData(n_points=200,true_frequencies = [0.5, 1.2, 2.0]):
     ''' 
     Synthesize verification timeline benchmarks
     '''

     np.random.seed(0)     
     x_data = np.sort(np.random.rand(n_points) * 10)
     true_frequencies = [0.5, 1.2, 2.0]
    
     y_clean = np.zeros_like(x_data)
     for freq in true_frequencies:
        y_clean += np.random.uniform(1, 2) * np.cos(2 * np.pi * freq * x_data + np.random.uniform(0, 2 * np.pi))
     y_data = y_clean + 0.5 * np.random.randn(n_points)

     return x_data, y_data, true_frequencies