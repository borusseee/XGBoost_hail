#!/usr/bin/env python
# coding: utf-8

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dateutil import rrule
import datetime
from datetime import timedelta
import glob
from netCDF4 import Dataset
import sys, traceback
import dateutil.parser as dparser
import string
import numpy as np
import numpy.ma as ma
import os
# from mpl_toolkits import basemap
# import ESMF
import pickle
import subprocess
import pandas as pd
from scipy import stats
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import pylab as plt
import random
import scipy.ndimage as ndimage
import matplotlib.gridspec as gridspec
# from mpl_toolkits.basemap import Basemap, cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from pylab import *
import string
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
# import shapefile
# import shapely.geometry
import math
from scipy.stats import gaussian_kde
from math import radians, cos, sin, asin, sqrt
# from shapely.geometry import Polygon, Point
from scipy.interpolate import interp1d
import csv
import os.path
import matplotlib.gridspec as gridspec
import scipy
import matplotlib.path as mplPath
import calendar 
from calendar import monthrange
from calendar import isleap
    
from numpy import linspace, meshgrid
from scipy.interpolate import griddata

import os
import matplotlib.pyplot as plt
from netCDF4 import Dataset as netcdf_dataset
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from cartopy import config
import cartopy.crs as ccrs
import geopandas as gpd
import cartopy.feature as cfeature

from matplotlib.cm import ScalarMappable

import random
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as mcolors
import matplotlib.patches as mpatch
from matplotlib.ticker import LogLocator
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import ListedColormap
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import xarray as xr 


# #######################################
#                            Settings

dStartDay=datetime.datetime(2000, 1, 1,0)
dStopDay=datetime.datetime(2021, 12, 31,23)
#generate time vectors
rgdTimeDD = pd.date_range(dStartDay, end=dStopDay, freq='d')
rgdTime1H = pd.date_range(dStartDay, end=dStopDay, freq='1h')

rgdFullTime=pd.date_range(datetime.datetime(1959, 1, 1, 0),
                          end=datetime.datetime(2022, 12, 31, 23), freq='1h')
rgdFullTime_day=pd.date_range(datetime.datetime(1959, 1, 1, 0),
                          end=datetime.datetime(2022, 12, 31, 23), freq='d')

iMonths=np.unique(rgdTimeDD.month)
iYears=np.unique(rgdTimeDD.year)

PlotDir='/glade/u/home/bblanc/HailModel/plots/'
sSaveDataDir='/glade/derecho/scratch/bblanc/ERA5_hail_model/'

sERAconstantFields='/glade/derecho/scratch/bblanc/197901/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc'

rgsModelVars_4i=['CAPEmax',
              'SRH03',
              'VS03',
              'FLH']

rgsModelVars_7n=['CINmax',
              'SRH06',
              'VS06',
              'DewT',
              'TotalTotals',
              'RH850',
              'RH500']

rgsModelVars=['CAPEmax',
              'SRH03',
              'VS03',
              'FLH',
              'CINmax',
              'SRH06',
              'VS06',
              'DewT',
              'TotalTotals',
              'RH850',
              'RH500']

# Initial 4 variables
rgsVarsAct_4i=(np.array([rgsModelVars_4i[ll].split('-') for ll in range(len(rgsModelVars_4i))]))
rgsERAVariables_4i = np.array([item for sublist in rgsVarsAct_4i for item in sublist])

# Additional 7 variables
rgsVarsAct_7n =(np.array([rgsModelVars_7n[ll].split('-') for ll in range(len(rgsModelVars_7n))]))
rgsERAVariables_7n = np.array([item for sublist in rgsVarsAct_7n for item in sublist])

# All variables 
rgsVarsAct=(np.array([rgsModelVars[ll].split('-') for ll in range(len(rgsModelVars))]))
rgsERAVariables = np.array([item for sublist in rgsVarsAct for item in sublist])

# -----------------------------------------------------------------------
# Read the observations from NOAA, ESSL, and BoM 
# -----------------------------------------------------------------------

# start with NOAA
sSaveFolder="/glade/work/bblanc/HailObs/SPC_data/ncdf_files/"
sDataSet = 'NOAA'

hailsize = ['Hail', 'HailSize']
rgrNOAAObs=np.zeros((2, len(rgdTimeDD), 130, 300))

for ii in range(len(hailsize)):
    dd=0
    for yy in range(len(iYears)):
        #print('Loadind NOAA data for year '+str(iYears[yy]))
        sFileName=sSaveFolder+'SPC-Hail-StormReports_gridded-75km_'+str(iYears[yy])+'.nc'

        # read in the variables
        ncid=Dataset(sFileName, mode='r')

        rgrLatGrid=np.squeeze(ncid.variables['lat'][:,0])
        rgrLonGrid=np.squeeze(ncid.variables['lon'][0,:])
        rgrLonNOAA=np.asarray(([rgrLonGrid,]*rgrLatGrid.shape[0]))
        rgrLatNOAA=np.asarray(([rgrLatGrid,]*rgrLonGrid.shape[0])).transpose()
        yearlength=365 + calendar.isleap(iYears[yy])
        
        rgrNOAAObs[ii,dd:dd+yearlength,:,:]=np.squeeze(ncid.variables[hailsize[ii]][:])
        
        dd=dd+yearlength
        ncid.close()


# Same with ESSL
sSaveFolder="/glade/work/bblanc/HailObs/ESSL/"
sDataSet = 'ESSL'
hailsize = ['Hail', 'size']
rgrESSLObs=np.zeros((2, len(rgdTimeDD), 180, 270))

for ii in range(len(hailsize)):
    dd=0
    for yy in range(len(iYears)):
        #print('Loadind BoM data')
        sFileName=sSaveFolder+'ESSL-Hail-StormReports_gridded-ERA5_'+str(iYears[yy])+'.nc'
#         print(sFileName)
        # read in the variables
        ncid=Dataset(sFileName, mode='r')

        rgrLatGrid=np.squeeze(ncid.variables['lat'][:,0])
        rgrLonGrid=np.squeeze(ncid.variables['lon'][0,:])
        rgrLonESSL=np.asarray(([rgrLonGrid,]*rgrLatGrid.shape[0]))
        rgrLatESSL=np.asarray(([rgrLatGrid,]*rgrLonGrid.shape[0])).transpose()
        yearlength=365 + calendar.isleap(iYears[yy])
        
        rgrESSLObs[ii,dd:dd+yearlength,:,:]=np.squeeze(ncid.variables[hailsize[ii]][:])
        dd=dd+yearlength
        ncid.close()
        

# Same with BoM
sSaveFolder="/glade/work/bblanc/HailObs/BoM/HailReports-ERA5/ERA5-gridded/"
sDataSet = 'BoM'
hailsize = ['Hail', 'HailSize']
rgrBoMObs=np.zeros((2, len(rgdTimeDD), 160, 200))

for ii in range(len(hailsize)):
    dd=0
    for yy in range(len(iYears)):
        #print('Loadind BoM data')
        sFileName=sSaveFolder+'BoM-Hail-StormReports_gridded-75km_'+str(iYears[yy])+'.nc'

        # read in the variables
        ncid=Dataset(sFileName, mode='r')

        rgrLatGrid=np.squeeze(ncid.variables['lat'][:,0])
        rgrLonGrid=np.squeeze(ncid.variables['lon'][0,:])
        rgrLonBoM=np.asarray(([rgrLonGrid,]*rgrLatGrid.shape[0]))
        rgrLatBoM=np.asarray(([rgrLatGrid,]*rgrLonGrid.shape[0])).transpose()
        yearlength=365 + calendar.isleap(iYears[yy])
        
        rgrBoMObs[ii,dd:dd+yearlength,:,:]=np.squeeze(ncid.variables[hailsize[ii]][:])
        dd=dd+yearlength
        ncid.close()
        

# Save Latitude, Longitude and Height in arrays
rgsERAdata='/glade/derecho/scratch/bblanc/ERA5_hail_model/'
sERAconstantFields='/glade/derecho/scratch/bblanc/197901/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc'
ncid=Dataset(sERAconstantFields, mode='r')
rgrLat=np.squeeze(ncid.variables['latitude'][:])
rgrLon=np.squeeze(ncid.variables['longitude'][:])
rgrHeight=(np.squeeze(ncid.variables['Z'][:]))/9.81
ncid.close()

rgiSize=rgrHeight.shape

# all lon grids have to be -180 to 180 degree
rgrLonERA=np.asarray(([rgrLon,]*rgrLat.shape[0]))
rgrLatERA=np.asarray(([rgrLat,]*rgrLon.shape[0])).transpose()
rgi180=np.where(rgrLonERA > 180)
rgrLonERA[rgi180]=rgrLonERA[rgi180]-360.

# check if we only want to read a subregion
rgiDomain=None
if rgiDomain != None:
    iloW=rgiDomain[3]
    iloE=rgiDomain[1]
    ilaN=rgiDomain[0]
    ilaS=rgiDomain[2]
else:
    iloW=rgrLonERA.shape[1]
    iloE=0
    ilaN=0
    ilaS=rgrLonERA.shape[0]

    
# ------------------------------------------------
# read NCDF files containing the hail predictors (11 predictors in total) 
# ------------------------------------------------

rgrERAVarall=np.zeros((len(rgdTimeDD), len(rgsERAVariables), rgrLatERA.shape[0],rgrLatERA.shape[1]))
iyear=0
#loop over years 
for yy in range(len(iYears)):
    iday=0
    outfile = '/glade/scratch/bblanc/ERA5_hail_model/ERA5_dailymax/ERA5_new_predictors_' + str(iYears[yy])+ '.npz'
    yearlength=365 + isleap(iYears[yy])
    
    if os.path.isfile(outfile) == False:
        rgrERAVarsyy=np.zeros((yearlength, len(rgsERAVariables), rgrLatERA.shape[0],rgrLatERA.shape[1]))

        #Loop over months
        for mm in range(len(iMonths)):
            # __________________________________________________________________
            # start reading the ERA data
            dDate=rgdTimeDD[0]
            monthlength=int(monthrange(iYears[yy],mm+1)[1])
            
            # Add the initial 4 variables in the array
            rgrERAdataAll=np.zeros((monthlength*24, len(rgsERAVariables), rgrLatERA.shape[0],rgrLatERA.shape[1])); rgrERAdataAll[:]=np.nan
            print (str(iYears[yy])+str( mm)+'    Loading CAPE, VS03, SRH03, FLH')

            # loop over variables
            sFileName=rgsERAdata + 'ERA5-hailpredictors/' + str(iYears[yy])+str("%02d" % (mm+1))+'_ERA-5_HailPredictors_newSRH03.nc'
            ncid=Dataset(sFileName, mode='r')

            for va in range(len(rgsERAVariables_4i)):
                rgrDataTMP=np.squeeze(ncid.variables[rgsERAVariables_4i[va]][:])
                rgrERAdataAll[:,va,:,:]=rgrDataTMP
            ncid.close()  
            
            # Now add the 7 new predictors in the same array for simplicity
            # loop over variables
            sFileName=rgsERAdata + 'ERA5-hailpredictors/' + str(iYears[yy])+str("%02d" % (mm+1))+'_New_RH_TT.nc'
            ncid=Dataset(sFileName, mode='r')
            print (str(iYears[yy])+str( mm)+'    CINmax, SRH06, VS06, Dew_T, TT, RH_850, RH_500')
            for va in range(len(rgsERAVariables_7n)):
                rgrDataTMP=np.squeeze(ncid.variables[rgsERAVariables_7n[va]][:])
                rgrERAdataAll[:,va+4,:,:]=rgrDataTMP
            ncid.close()  

            #reshaping the monthly data into daily data to be able to extract the daily max CAPE
            dayrgrERAdataAll=rgrERAdataAll.reshape(int(monthlength),24,rgrERAdataAll.shape[1],rgrERAdataAll.shape[2],rgrERAdataAll.shape[3])
            timemax=np.argmax(dayrgrERAdataAll[:,:,0,:,:],axis=1) # take the time of maximum CAPE
            timemax2=np.zeros((dayrgrERAdataAll.shape[0],dayrgrERAdataAll.shape[1],dayrgrERAdataAll.shape[2],
                               dayrgrERAdataAll.shape[3],dayrgrERAdataAll.shape[4]))  
            timemax2[:,:,:,:,:]=timemax[:,None,None,:]
            timemax2=timemax2.astype(int)

            #Take the predictors at the time of max CAPE for each day
            rgrERAVars=np.take_along_axis(dayrgrERAdataAll,timemax2,axis=1)
            rgrERAVars=rgrERAVars[:,1,:,:,:]
            
            #save monthly data in a yearly array
            rgrERAVarsyy[iday:iday+monthlength,:,:,:]=rgrERAVars

            iday=monthlength+iday
        
        np.savez(outfile,
                rgrERAVarsyy = rgrERAVarsyy,
                rgrLat = rgrLat,
                rgrLon = rgrLon,
                rgrHeight = rgrHeight,
                rgdTimeDD = rgdTimeDD)
        #save the yearly data in an array contaning the whole time period
        rgrERAVarall[iyear:iyear+yearlength,:,:,:]=rgrERAVarsyy
        iyear=iyear+yearlength
    
    else:
        print ('The file already exists')
        data_tmp = np.load(outfile)
        rgrERAVarall[iyear:iyear+yearlength,:,:,:] = data_tmp['rgrERAVarsyy']
        rgrLat = data_tmp['rgrLat']
        rgrLon = data_tmp['rgrLon']
        rgrHeight = data_tmp['rgrHeight']
        #rgdTimeDD = pd.to_datetime(data_tmp['rgdTimeDD'])
        iyear=iyear+yearlength


rgrLonAct=rgrLonNOAA
rgrLatAct=rgrLatNOAA

rgiLon=np.where((rgrLonERA[0,:] >= rgrLonAct[0,0]) & (rgrLonERA[0,:] <= rgrLonAct[0,-1]))[0]
rgiLat=np.where((rgrLatERA[:,0] <= rgrLatAct[0,0]) & (rgrLatERA[:,0] >= rgrLatAct[-1,0]))[0]
rgiDomain=[rgiLat[0],rgiLon[0],rgiLat[-1]+1,rgiLon[-1]+1]

iloW=rgiDomain[3]
iloE=rgiDomain[1]
ilaN=rgiDomain[0]
ilaS=rgiDomain[2]

#get the grid cells covered with land only
invar_dir='/glade/campaign/collections/rda/data/ds633.0/e5.oper.invariant/197901/'
sFileName=invar_dir+'e5.oper.invariant.128_172_lsm.ll025sc.1979010100_1979010100.nc'
ncid=Dataset(sFileName, mode='r')
LSM=np.squeeze(ncid.variables['LSM'][:])
ncid.close()
rgrERAVarall_US = rgrERAVarall[:,:,164:260,iloE:iloW]
LSM_US = LSM[164:260,iloE:iloW]

rgrLonAct=rgrLonESSL
rgrLatAct=rgrLatESSL

rgiLon=np.where((rgrLonERA[0,:] >= rgrLonAct[0,0]) & (rgrLonERA[0,:] <= rgrLonAct[0,-1]))[0]
rgiLat=np.where((rgrLatERA[:,0] <= rgrLatAct[0,0]) & (rgrLatERA[:,0] >= rgrLatAct[-1,0]))[0]
rgiDomain=[rgiLat[0],rgiLon[0],rgiLat[-1]+1,rgiLon[-1]+1]

iloW=rgiDomain[3]
iloE=rgiDomain[1]
ilaN=rgiDomain[0]
ilaS=rgiDomain[2]
# define the input and output variables.
indices = np.concatenate((np.arange(1370,1440), np.arange(0,200)))
rgrERAVarall_EU = rgrERAVarall[:,:,ilaN:ilaS,indices]
LSM_EU = LSM[ilaN:ilaS,indices]

rgrLonAct=rgrLonBoM
rgrLatAct=rgrLatBoM

rgiLon=np.where((rgrLonERA[0,:] >= rgrLonAct[0,0]) & (rgrLonERA[0,:] <= rgrLonAct[0,-1]))[0]
rgiLat=np.where((rgrLatERA[:,0] <= rgrLatAct[0,0]) & (rgrLatERA[:,0] >= rgrLatAct[-1,0]))[0]
rgiDomain=[rgiLat[0],rgiLon[0],rgiLat[-1]+1,rgiLon[-1]+1]

iloW=rgiDomain[3]
iloE=rgiDomain[1]
ilaN=rgiDomain[0]
ilaS=rgiDomain[2]

rgrERAVarall_AU = rgrERAVarall[:,:,ilaN:ilaS,iloE:iloW]
LSM_AU = LSM[ilaN:ilaS,iloE:iloW]

indices_test = [np.arange(0,int(1*rgdTimeDD.shape[0]/11)),
                np.arange(int(1*rgdTimeDD.shape[0]/11),int(2*rgdTimeDD.shape[0]/11)),
                np.arange(int(2*rgdTimeDD.shape[0]/11),int(3*rgdTimeDD.shape[0]/11)),
                np.arange(int(3*rgdTimeDD.shape[0]/11),int(4*rgdTimeDD.shape[0]/11)),
                np.arange(int(4*rgdTimeDD.shape[0]/11),int(5*rgdTimeDD.shape[0]/11)),
                np.arange(int(5*rgdTimeDD.shape[0]/11),int(6*rgdTimeDD.shape[0]/11)),
                np.arange(int(6*rgdTimeDD.shape[0]/11),int(7*rgdTimeDD.shape[0]/11)),
                np.arange(int(7*rgdTimeDD.shape[0]/11),int(8*rgdTimeDD.shape[0]/11)),
                np.arange(int(8*rgdTimeDD.shape[0]/11),int(9*rgdTimeDD.shape[0]/11)),
                np.arange(int(9*rgdTimeDD.shape[0]/11),int(10*rgdTimeDD.shape[0]/11)),
                np.arange(int(10*rgdTimeDD.shape[0]/11),rgdTimeDD.shape[0])
               ]

indices_train = [np.arange(int(1*rgdTimeDD.shape[0]/11),rgdTimeDD.shape[0]),
                 np.concatenate((np.arange(0,int(1*rgdTimeDD.shape[0]/11)), np.arange(int(2*rgdTimeDD.shape[0]/11),rgdTimeDD.shape[0]))),
                 np.concatenate((np.arange(0,int(2*rgdTimeDD.shape[0]/11)), np.arange(int(3*rgdTimeDD.shape[0]/11),rgdTimeDD.shape[0]))),
                 np.concatenate((np.arange(0,int(3*rgdTimeDD.shape[0]/11)), np.arange(int(4*rgdTimeDD.shape[0]/11),rgdTimeDD.shape[0]))),
                 np.concatenate((np.arange(0,int(4*rgdTimeDD.shape[0]/11)), np.arange(int(5*rgdTimeDD.shape[0]/11),rgdTimeDD.shape[0]))),
                 np.concatenate((np.arange(0,int(5*rgdTimeDD.shape[0]/11)), np.arange(int(6*rgdTimeDD.shape[0]/11),rgdTimeDD.shape[0]))),
                 np.concatenate((np.arange(0,int(6*rgdTimeDD.shape[0]/11)), np.arange(int(7*rgdTimeDD.shape[0]/11),rgdTimeDD.shape[0]))),
                 np.concatenate((np.arange(0,int(7*rgdTimeDD.shape[0]/11)), np.arange(int(8*rgdTimeDD.shape[0]/11),rgdTimeDD.shape[0]))),
                 np.concatenate((np.arange(0,int(8*rgdTimeDD.shape[0]/11)), np.arange(int(9*rgdTimeDD.shape[0]/11),rgdTimeDD.shape[0]))),
                 np.concatenate((np.arange(0,int(9*rgdTimeDD.shape[0]/11)), np.arange(int(10*rgdTimeDD.shape[0]/11),rgdTimeDD.shape[0]))),
                 np.arange(0,int(10*rgdTimeDD.shape[0]/11))
                ]

for yy in range(len(indices_train)):
    print(f'Working on year {iYears[2*yy]}\n')
    ## US
    # Create y_train
    rgrNOAAObs_2 = rgrNOAAObs[0,:,14:110,:] # Here consider the US only
    y_land_train = np.copy(rgrNOAAObs_2[indices_train[yy],:,:])
    y_land_train[:,LSM_US<0.5]=np.nan
    valid_indices_train = ~np.isnan(y_land_train)
    notvalid_indices_train = np.isnan(y_land_train)
    y_train_2 = y_land_train[valid_indices_train] # Discard the nan from the training data
    index_train0 = np.argwhere(y_train_2==0).flatten()
    index_train1 = np.argwhere(y_train_2==1).flatten()
    selected_index_train0 = random.sample(index_train0.tolist(), int(len(index_train0)/33))
    output_train0 = y_train_2[selected_index_train0]
    output_train1 = y_train_2[index_train1]
    indexes01 = np.sort(np.concatenate((selected_index_train0, index_train1)))
    y_train_US = y_train_2[indexes01]
    # Create X_train 
    # Take 164 and 260 to exclude Mexico and Canada 
    X_land_train = np.copy(rgrERAVarall_US[indices_train[yy],:,:,:])
    X_land_train[:,:,LSM_US<0.5]=np.nan
    X_land_train =  np.moveaxis(X_land_train, 1, 0)
    input1 = X_land_train[:,valid_indices_train]
    input1[1,:] = np.abs(input1[1,:])
    input1[5,:] = np.abs(input1[5,:])
    X_train1 = input1[:,index_train1]
    X_train0 = input1[:, selected_index_train0]
    X_train_US = input1[:,indexes01]
    X_train_US = np.transpose(X_train_US)

    ## EUROPE
    # y_train
    y_land_train = rgrESSLObs[0,indices_train[yy],:,:] 
    y_land_train[:,LSM_EU<0.5]=np.nan
    valid_indices_train = ~np.isnan(y_land_train)
    notvalid_indices_train = np.isnan(y_land_train)
    y_train_2 = y_land_train[valid_indices_train] # Discard the nan from the training data
    index_train0 = np.argwhere(y_train_2==0).flatten()
    index_train1 = np.argwhere(y_train_2==1).flatten()
    selected_index_train0 = random.sample(index_train0.tolist(), int(len(index_train0)/80))
    output_train0 = y_train_2[selected_index_train0]
    output_train1 = y_train_2[index_train1]
    indexes01 = np.sort(np.concatenate((selected_index_train0, index_train1)))  
    y_train_EU = y_train_2[indexes01]
    # X_train
    X_land_train = rgrERAVarall_EU[indices_train[yy],:,:,:]
    X_land_train[:,:,LSM_EU<0.5]=np.nan
    X_land_train =  np.moveaxis(X_land_train, 1, 0)
    input1 = X_land_train[:,valid_indices_train]
    input1[1,:] = np.abs(input1[1,:])
    input1[5,:] = np.abs(input1[5,:])
    X_train1 = input1[:, index_train1]
    X_train0 = input1[:, selected_index_train0]
    X_train_EU = input1[:,indexes01]
    X_train_EU = np.transpose(X_train_EU)

    ## AUSTRALIA
    # y_train
    y_land_train = rgrBoMObs[0,indices_train[yy],:,:] # Select the 1st 3/4 of the time period for training
    y_land_train[:,LSM_AU<0.5]=np.nan
    valid_indices_train = ~np.isnan(y_land_train)
    notvalid_indices_train = np.isnan(y_land_train)
    y_train_2 = y_land_train[valid_indices_train] # Discard the nan from the training data
    index_train0 = np.argwhere(y_train_2==0).flatten()
    index_train1 = np.argwhere(y_train_2==1).flatten()
    selected_index_train0 = random.sample(index_train0.tolist(), int(len(index_train0)/300))
    output_train0 = y_train_2[selected_index_train0]
    output_train1 = y_train_2[index_train1]
    indexes01 = np.sort(np.concatenate((selected_index_train0, index_train1)))
    y_train_AU = y_train_2[indexes01]
    # X_train
    X_land_train = rgrERAVarall_AU[indices_train[yy],:,:,:]
    X_land_train[:,:,LSM_AU<0.5]=np.nan
    X_land_train =  np.moveaxis(X_land_train, 1, 0)
    input1 = X_land_train[:,valid_indices_train]
    input1[1,:] = np.abs(input1[1,:])
    input1[5,:] = np.abs(input1[5,:])
    X_train1 = input1[:, index_train1]
    X_train0 = input1[:, selected_index_train0]
    X_train_AU = input1[:,indexes01]
    X_train_AU = np.transpose(X_train_AU)

    ## Combine the 3 datasets
    X_train = np.concatenate((X_train_US, X_train_EU, X_train_AU), axis=0)
    X_train = np.nan_to_num(X_train, nan=0)
    # print(X_train.shape)
    y_train = np.concatenate((y_train_US, y_train_EU, y_train_AU))
    y_train = np.nan_to_num(y_train, nan=0)

    # print(y_train.shape)
    
    # Create test data 
    ## US
    y_land_test = np.copy(rgrNOAAObs_2[indices_test[yy],:,:])
    y_test_US = y_land_test.reshape(y_land_test.shape[0]*y_land_test.shape[1]*y_land_test.shape[2])
    
    X_land_test = np.copy(rgrERAVarall_US[indices_test[yy],:,:,:])
    input1 =  np.moveaxis(X_land_test, 1, 0)
    input1[1,:] = np.abs(input1[1,:])
    input1[5,:] = np.abs(input1[5,:])
    X_test_US = input1.reshape(input1.shape[0], input1.shape[1]*input1.shape[2]*input1.shape[3])
    X_test_US = np.transpose(X_test_US)
    ## EUROPE
    y_land_test = rgrESSLObs[0,indices_test[yy],:,:] 
    y_test_EU = y_land_test.reshape(y_land_test.shape[0]*y_land_test.shape[1]*y_land_test.shape[2])
    
    X_land_test = rgrERAVarall_EU[indices_test[yy],:,:,:]
    input1 =  np.moveaxis(X_land_test, 1, 0)
    input1[1,:] = np.abs(input1[1,:])
    input1[5,:] = np.abs(input1[5,:])
    X_test_EU = input1.reshape(input1.shape[0], input1.shape[1]*input1.shape[2]*input1.shape[3])
    X_test_EU = np.transpose(X_test_EU)
    ## AUSTRALIA
    y_land_test = rgrBoMObs[0,indices_test[yy],:,:] 
    y_test_AU = y_land_test.reshape(y_land_test.shape[0]*y_land_test.shape[1]*y_land_test.shape[2])

    X_land_test = rgrERAVarall_AU[indices_test[yy],:,:,:]
    input1 =  np.moveaxis(X_land_test, 1, 0)
    input1[1,:] = np.abs(input1[1,:])
    input1[5,:] = np.abs(input1[5,:])
    X_test_AU = input1.reshape(input1.shape[0], input1.shape[1]*input1.shape[2]*input1.shape[3])
    X_test_AU = np.transpose(X_test_AU)

    ## Combine the 3 datasets
    X_test = np.concatenate((X_test_US, X_test_EU, X_test_AU), axis=0)
    X_test = np.nan_to_num(X_test, nan=0)
    # print(X_test.shape)
    y_test = np.concatenate((y_test_US, y_test_EU, y_test_AU))
    y_test = np.nan_to_num(y_test, nan=0)

    # print(y_test.shape)

    ## Create data for predictions    
    input2 = rgrERAVarall[indices_test[yy],:,:,:]
    input2[:,1,:,:] = np.abs(input2[:,1,:,:])
    input2[:,5,:,:] = np.abs(input2[:,5,:,:])
    X_test1 =  np.moveaxis(input2, 1, 0)
    X_pred = X_test1.reshape(X_test1.shape[0],X_test1.shape[1]*X_test1.shape[2]*X_test1.shape[3])
    X_pred = np.transpose(X_pred)
    X_pred = np.nan_to_num(X_pred, nan=0)


    # Create regression matrices
    xg_train_1 = xgb.DMatrix(X_train, label=y_train)
    xg_pred_1 = xgb.DMatrix(X_pred)
    xg_test_1 = xgb.DMatrix(X_test, label=y_test)

    # ------------------------------------------------------
    # Train the XGBoost model and perform Bayesian opti
    # ------------------------------------------------------
    def objective(params):
        # Define the XGBoost classifier with the current set of hyperparameters
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            max_depth=int(params['max_depth']),
            learning_rate=params['learning_rate'],
            n_estimators=int(params['n_estimators']),
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            gamma=params['gamma'],
            Min_Child_weight=int(params['Min_Child_weight']),
            reg_alpha = params['reg_alpha'],
            reg_lambda = params['reg_lambda'],
        )


        # Train the XGBoost model
        model.fit(X_train, y_train, eval_metric="logloss", verbose=False, early_stopping_rounds=10, eval_set=[(X_test, y_test)])
    
        # Make predictions on the test set
        y_pred = model.predict_proba(X_test)[:, 1]
        
        # Calculate the negative f1 score (to be minimized)
        # f1 = -f1_score(y_test, (y_pred > 0.5).astype(int))
        
        # Calculate the negative AUC-ROC (to be minimized)
        auc_roc = -roc_auc_score(y_test, y_pred)
    
        return {'loss': auc_roc, 'status': STATUS_OK} # Here specify the loss function

  # Define the hyperparameter search space
    space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1.0)),
        'n_estimators': hp.quniform('n_estimators', 50, 200, 1),
        'max_depth': hp.quniform('max_depth', 4, 8, 1),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'gamma': hp.uniform('gamma', 0, 1.0),
        'Min_Child_weight': hp.quniform('Min_Child_weight', 1, 10, 1),
        'reg_alpha': hp.uniform('reg_alpha', 0, 1.0),
        'reg_lambda': hp.uniform('reg_lambda', 0, 1.0),
    }

    # Run the optimization
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=5, trials=trials, rstate=np.random.RandomState(42))
    best.update({"objective": "binary:logistic","eval_metric": "logloss", "tree_method": "hist"})

    best['Min_Child_weight'] = int(best['Min_Child_weight'])
    best['max_depth'] = int(best['max_depth'])
    best['n_estimators'] = int(best['n_estimators'])
    # Print the best hyperparameters
    print("Best Hyperparameters:", best)
    
    evals = [(xg_train_1, "train"), (xg_test_1, "validation")]

    # Train the XGBoost model
    model = xgb.train(
       params=best,
       dtrain=xg_train_1,
       evals=evals,
       verbose_eval=10,
       early_stopping_rounds=10
    )
    
    model.save_model(f'/glade/scratch/bblanc/ERA5_hail_model/xgboost_predictions/Global_XGBmodel_iteration{yy}.jason')
    
    # Make predictions on the test dataset
    preds_test = model.predict(xg_pred_1)
    preds_test2 = np.copy(preds_test)
    print('predictions made on Test data')

    np.save(f'/glade/scratch/bblanc/ERA5_hail_model/xgboost_predictions/Predictions_world_{iYears[yy]}_{iYears[yy+1]}.npy', preds_test2)
