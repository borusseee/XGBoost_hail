#!/usr/bin/env python
# coding: utf-8
'''
    File name: SyntheticHailModel.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 24.08.2017
    Date last modified: 24.08.2017

    ##############################################################
    Purpos:

    Evaluates the conditions under which hail forms.

    1) hail observations are imported

    2) ERA-Interim conditions at the day and location where hail occured
    are read

    3) conditions for non-hail days are calculated

    4) investigation if hail days are significantly different from non hail
    days in several U.S. subregions

    5) trends in days with hail conditions are calculated

'''

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
import shapefile
# import shapely.geometry
import shapefile
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
from pdb import set_trace as stop

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
from netCDF4 import Dataset
import h5py


def grid(x, y, z, resX=20, resY=20):
    "Convert 3 column data to matplotlib grid"
    xi = linspace(min(x), max(x), resX)
    yi = linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi, interp='linear')
    X, Y = meshgrid(xi, yi)
    return X, Y, Z


########################################
#                            Settings
YYYY = int(sys.argv[1])

dStartDay=datetime.datetime(YYYY, 1, 1,0,0)
dStopDay=datetime.datetime(YYYY, 12, 31,23,59)
#generate time vectors
rgdTimeDD = pd.date_range(dStartDay, end=dStopDay, freq='d')
rgdTime1H = pd.date_range(dStartDay, end=dStopDay, freq='1h')

rgdFullTime=pd.date_range(datetime.datetime(1959, 1, 1, 0),
                          end=datetime.datetime(2022, 12, 31, 23), freq='1h')
rgdFullTime_day=pd.date_range(datetime.datetime(1959, 1, 1, 0),
                          end=datetime.datetime(2022, 12, 31, 23), freq='d')

iMonths=np.unique(rgdTimeDD.month)
iYears=np.unique(rgdTimeDD.year)

rgsLableABC=string.ascii_lowercase+string.ascii_uppercase

sERAconstantFields='/glade/derecho/scratch/bblanc/ERA5_hail_model/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc'

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


rgsVarCombinations=['CAPEmax-FLH']
rgsFinVars=['CAPEmax-FLH','CAPEmax','SRH03','VS03']
# Initial 4 variables
rgsVarsAct_4i=(np.array([rgsModelVars_4i[ll].split('-') for ll in range(len(rgsModelVars_4i))]))
rgsERAVariables_4i = np.array([item for sublist in rgsVarsAct_4i for item in sublist])

# Additional 7 variables
rgsVarsAct_7n =(np.array([rgsModelVars_7n[ll].split('-') for ll in range(len(rgsModelVars_7n))]))
rgsERAVariables_7n = np.array([item for sublist in rgsVarsAct_7n for item in sublist])

# All variables 
rgsVarsAct=(np.array([rgsModelVars[ll].split('-') for ll in range(len(rgsModelVars))]))
rgsERAVariables = np.array([item for sublist in rgsVarsAct for item in sublist])

rgsVarMaxLim=['FLH']
rgsVarMinLim=['CAPEmax','SRH03','VS03','FLH']

rMinHailSize= 2.5 # this is the maximum diameter of a hail stone in cm

# # provide radio sounding data if desired
# sSoundDataFile='/glade/p_old/work/prein/observations/RadioSoundings_Uni-Wyoming/data/'
# rgsSoundStat=['72469']

# iCorrelPred=20 # 20 # how many 2D predictors should be used
# iPosComb=scipy.math.factorial(len(rgsERAVariables))/(scipy.math.factorial(2) * scipy.math.factorial(len(rgsERAVariables)-2)) # max combinations of variables

#-----------------------------------------------------------------------
# Read the observations from NOAA, BoM & ESSL 
#-----------------------------------------------------------------------

# start with NOAA
sSaveFolder="/glade/work/bblanc/HailObs/SPC_data/ncdf_files/"
sDataSet = 'NOAA'


#RawData = None
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

#Get monthly and yearly hail observations
daily_NOAAobs = np.zeros((len(iYears), 12, rgrLonNOAA.shape[0], rgrLonNOAA.shape[1]))
for yy in range(len(iYears)):
    for mm in range(12):
        daily_NOAAobs[yy,mm,:,:] = np.sum(rgrNOAAObs[0,(rgdTimeDD.year == iYears[yy]) & (rgdTimeDD.month == (mm+1)),:,:], axis=0)
        
monthly_NOAAobs=np.mean(daily_NOAAobs, axis=0)
yearly_NOAAobs=np.sum(daily_NOAAobs, axis=1)


#------------------------------------------------#
# Save Latitude, Longitude and Height in arrays
#------------------------------------------------#

rgsERAdata='/glade/derecho/scratch/bblanc/ERA5_hail_model/ERA5-hailpredictors/'

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
    

#------------------------------------------------
# read NCDF files containing the hail predictors (11 predictors in total) 
#------------------------------------------------

rgrERAVarall=np.zeros((len(rgdTimeDD), len(rgsERAVariables), rgrLatERA.shape[0],rgrLatERA.shape[1]))
iyear=0
#loop over years 
for yy in range(len(iYears)):
    iday=0
    outfile = '/glade/derecho/scratch/bblanc/ERA5_hail_model/ERA5_dailymax/ERA5_new_predictors_' + str(iYears[yy])+ '.npz'
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
            sFileName=rgsERAdata + str(iYears[yy])+str("%02d" % (mm+1))+'_ERA-5_HailPredictors_newSRH03.nc'
            ncid=Dataset(sFileName, mode='r')

            for va in range(len(rgsERAVariables_4i)):
                rgrDataTMP=np.squeeze(ncid.variables[rgsERAVariables_4i[va]][:])
                rgrERAdataAll[:,va,:,:]=rgrDataTMP
            ncid.close()  
            
            # Now add the 7 new predictors in the same array for simplicity
            # loop over variables
            sFileName=rgsERAdata + str(iYears[yy])+str("%02d" % (mm+1))+'_New_RH_TT.nc'
            ncid=Dataset(sFileName, mode='r')
            print (str(iYears[yy])+str( mm)+'    CINmax, SRH06, VS06, Dew_T, TT, RH_850, RH_500')
            for va in range(len(rgsERAVariables_7n)):
                rgrDataTMP=np.squeeze(ncid.variables[rgsERAVariables_7n[va]][:])
                rgrERAdataAll[:,va+4,:,:]=rgrDataTMP
            ncid.close()  

            #reshaping the monthly data into daily data to be able to extract the daily max CAPE
            dayrgrERAdataAll=rgrERAdataAll.reshape(int(monthlength),24,rgrERAdataAll.shape[1],rgrERAdataAll.shape[2],rgrERAdataAll.shape[3])
            timemax=np.argmax(dayrgrERAdataAll[:,:,0,:,:],axis=1) # take the time of maximum CAPE
            timemax2=np.zeros((dayrgrERAdataAll.shape[0],dayrgrERAdataAll.shape[1],dayrgrERAdataAll.shape[2],dayrgrERAdataAll.shape[3],dayrgrERAdataAll.shape[4]))
            timemax2[:,:,:,:,:]=timemax[:,None,None,:]
            timemax2=timemax2.astype(int)

            #Take the predictors at the time of max CAPE for each day
            rgrERAVars=np.take_along_axis(dayrgrERAdataAll,timemax2,axis=1)
            rgrERAVars=rgrERAVars[:,1,:,:,:]
            
            #save monthly data in a yearly array
            rgrERAVarsyy[iday:iday+monthlength,:,:,:]=rgrERAVars
            print(f'max TT: {np.max(rgrERAVarsyy[:,8,:,:])}')
            print(f'max RH 850: {np.max(rgrERAVarsyy[:,9,:,:])}')
            print(f'max RH 500: {np.max(rgrERAVarsyy[:,10,:,:])}')
            print(f'mean TT: {np.mean(rgrERAVarsyy[:,8,:,:])}')
            print(f'mean RH 850: {np.mean(rgrERAVarsyy[:,9,:,:])}')
            print(f'mean RH 500: {np.mean(rgrERAVarsyy[:,10,:,:])}')
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


