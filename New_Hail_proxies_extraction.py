#!/usr/bin/env python
# coding: utf-8

'''
    File name: MAIN-HailVar-extraction.py
    Author: Andreas Prein & Boris Blanc
    E-mail: prein@ucar.edu bblanc@student.ethz.ch
    Date created: 23.08.2017
    Date last modified: 10.02.2023

    ##############################################################
    Purpos:

    1) Reads in hourly ERA-5 data from RDA

    2) Calculates variables that are related to large hail
       development

    3) Saves the variables to yearly files in NCDF format
    
    ##############################################################
    Purpose of modification:
    1) Reads in hourly ERA-5 data from RDA

    2) Calculates variables that are related to large hail
       development

    3) Saves the variables to yearly files in NCDF format

'''


from dateutil import rrule
import datetime
import calendar
import glob
from netCDF4 import Dataset
import xarray as xr
import sys, traceback
import dateutil.parser as dparser
import string
## from ipdb import set_trace as stop
import numpy as np
import numpy.ma as ma
import os
## import ESMF
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
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from pylab import *
import string
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import shapely.geometry
## import descartes
import shapefile
import math
from scipy.stats.kde import gaussian_kde
from math import radians, cos, sin, asin, sqrt
from scipy import spatial
import scipy.ndimage
import matplotlib.path as mplPath
from scipy.interpolate import interp1d
import time
from math import atan2, degrees, pi
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
## import SkewT
import csv
import pygrib
from scipy import interpolate
from shutil import copyfile
import wrf
from wrf import getvar, interpz3d
import thermodynamics_p3
from thermodynamics_p3 import Density, ThetaE, MixR2VaporPress, relhum, TempToDewpTemp, DewPoint
from tqdm import tqdm
from pdb import set_trace as stop
from calendar import monthrange



########################################
#                            USER INPUT SECTION

## change here with my own directory bblanc
sSaveDataDir='/glade/derecho/scratch/bblanc/ERA5_hail_model/ERA5-hailpredictors/'
# sSaveDataDir='/glade/scratch/prein/ERA5_hail_model/'

MM = int(sys.argv[1])
YYYY = int(sys.argv[2])
DD = monthrange(YYYY, MM)[1]

dStartDay = datetime.datetime(YYYY, MM, 1,0)  
dStopDay = datetime.datetime(YYYY, MM, DD,23) 


rgdTime1H = pd.date_range(dStartDay, end=dStopDay, freq='1h')
rgdTimeDD = pd.date_range(dStartDay, end=dStopDay, freq='d')

rgiYY=np.unique(rgdTime1H.year)

rgiERA_hours=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])

rgdFullTime=pd.date_range(datetime.datetime(1959, 1, 1, 0),
                          end=datetime.datetime(2022, 12, 31, 23), freq='1h')


# ERA levels
rgrLevelsERA=np.array(range(1,38,1))

rgr3Dvariables=['T','Q','Z','W', 'R']
rgr3DERAvars=['T_GDS4_HYBL','Q_GDS4_HYBL','Z_GDS4_HYBL','W_GDS4_HYBL', 'R']

rgr3Dstaggerd=['U','V']
rgr3DERAStag=['U_GDS4_HYBL','V_GDS4_HYBL']

rgr2Dvariables=['SP', 'VAR_10U', 'VAR_10V', 'VAR_2D']
rgr2DERAvars=['LNSP_GDS4_HYBL']

## Create variable names for each useful variable
T = '130_t'
Z = '129_z'
Q = '133_q'
W = '135_w'
R = '157_r'
U = '131_u'
V = '132_v'
SP = '134_sp'
U10 = '165_10u'
V10 = '166_10v'
DT = '168_2d'
name3Dvariables=[T, Q, Z, W, R]
name3Dstaggerd=[U, V]
name2Dvariables= [SP, U10, V10, DT]

# ________________________________________________________________________
# ________________________________________________________________________
# Define constants
cpa=1006.      # (J/kg C)  ||   heat capacity of air at constant pressure
Rd=287.058  # J kg-1 K-1 ||  gas constant of dry air
Rv=461.5      # J/(kg K) ||  gas constant of water vapor
Lv= 2501000.  #  (J/kg) || latent heat of vaporization
cpw = 1840.  # (J/kg K) ||   heat capacity of water vapor at constant pressure
cw = 4190.  #   (J/kg K) || specific heat water
g = 9.81 # (m/s^2) || gravitatoinal acceleration

# Pressure levels

pressurelvls = np.zeros((37,721,1440))
plvl = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225,
        250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825,
        850, 875, 900, 925, 950, 975, 1000]

# These are the ERA-5 pressure levels (in hPa)
for ii in range(721):
    for jj in range(1440):
        pressurelvls[:,ii,jj] = plvl

#Convert from hPa to Pa
presPa=pressurelvls*100  

# ________________________________________________________________________
# ________________________________________________________________________
# # first read the coordinates

sERAconstantFields='/glade/campaign/collections/rda/data/ds633.0/e5.oper.invariant/197901/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc'
# read the ERA-5 elevation
ncid=Dataset(sERAconstantFields, mode='r')
rgrLat75=np.squeeze(ncid.variables['latitude'][:])
rgrLon75=np.squeeze(ncid.variables['longitude'][:])
rgrHeight=(np.squeeze(ncid.variables['Z'][:]))/9.81

ncid.close()
rgiSize=rgrHeight.shape
rgrLon=np.asarray(([rgrLon75,]*rgrLat75.shape[0]))
rgrLat=np.asarray(([rgrLat75,]*rgrLon75.shape[0])).transpose()


print('Working on', str("%04d" %rgiYY),'-', str("%02d" % np.unique(rgdTime1H.month)))

s3Dfolder='/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.pl/'    
s3Dextension='/e5.oper.an.pl.128_'
s2Dfolder='/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.sfc/'    
s2Dextension='/e5.oper.an.sfc.128_'

sFileFin=sSaveDataDir + str(rgdTimeDD[0].year) + str(rgdTimeDD[0].month).zfill(2) + '_New_RH_TT.nc'
sFile=sFileFin+'COPY'

## last day of month: 
monthlength= len(rgdTime1H) # 24*calendar.monthrange(rgiYY,rgdTimeDD[0].month)[1]
print('Working on '+ str(monthlength) +' hours')


sFolderYYMO=str(rgdTime1H[0].year)+str("%02d" % (rgdTime1H[0].month))      
if os.path.isfile(sFileFin) != 1:
    rgrData2DAct= np.zeros((len(rgr2Dvariables), monthlength, 721,1440))# ; rgrData2DAct[:]=np.nan
    rgdTime_mon = pd.date_range(datetime.datetime(YYYY, MM, 1,0), 
                              end=datetime.datetime(YYYY, MM, monthrange(YYYY, MM)[1],23) , freq='1h')
    hh_sel = np.isin(rgdTime_mon, rgdTime1H)
    print('Working on 2D variables')
    for v3 in tqdm(range(len(name2Dvariables))):
    # read in the 2D variables (surface pressure, u & v wind components)
        sFileName=s2Dfolder+sFolderYYMO+s2Dextension+ name2Dvariables[v3] +'.ll025sc.'+str(rgdTime1H[0].year)+str("%02d" % (rgdTime1H[0].month))+'0100'+'_'+str(rgdTime1H[0].year)+str("%02d" % (rgdTime1H[0].month))+str("%02d" % (len(rgdTime_mon)/24))+str(23)+'.nc'
#         iFileNC='/glade/scratch/bblanc/'+sFileName.split('/')[-1]
#         copyfile(sFileName, iFileNC)
#         print ('    Load '+iFileNC)
        # read in the variables
        ncid=Dataset(sFileName, mode='r')
        rgrData2DAct[v3,:,:,:]=np.squeeze(ncid.variables[rgr2Dvariables[v3]][hh_sel,:])
        ncid.close()
        # clean up
#         os.system("rm "+sFileName)

# CAPE3D=np.zeros((len(rgdTime1H),rgrLon.shape[0],rgrLon.shape[1]))
CIN3D=np.zeros((len(rgdTime1H),rgrLon.shape[0],rgrLon.shape[1]))
rgrVS06=np.zeros((len(rgdTime1H),rgrLon.shape[0],rgrLon.shape[1]))
rgrSRH06=np.zeros((len(rgdTime1H),rgrLon.shape[0],rgrLon.shape[1]))
RH_500=np.zeros((len(rgdTime1H),rgrLon.shape[0],rgrLon.shape[1]))
RH_850=np.zeros((len(rgdTime1H),rgrLon.shape[0],rgrLon.shape[1]))
Dew_T=np.zeros((len(rgdTime1H),rgrLon.shape[0],rgrLon.shape[1]))
TT = np.zeros((len(rgdTime1H),rgrLon.shape[0],rgrLon.shape[1]))
tempkalltime = np.zeros((len(rgdTime1H),37,rgrLon.shape[0],rgrLon.shape[1]))

if os.path.isfile(sFileFin) != 1:
    for tt in tqdm(range(len(rgdTimeDD))): #  loop over time (in days)
        dDate=rgdTimeDD[tt]
        sDate=str(dDate); sDate = sDate.replace(' ', '_')
    
        sFolderYYMO=str(dDate.year)+str("%02d" % (dDate.month))    
        
        # select the 2D data for this day
        day_sel = rgdTime1H.day == dDate.day
        data2D_day = rgrData2DAct[:,day_sel,:]
        
#         print('Working on 3D variables')
        rgrData3DAct= np.zeros((len(rgr3Dvariables),24, 37,721,1440)) #; rgrData3DAct[:]=np.nan
        rgrData3Dstag= np.zeros((len(rgr3Dstaggerd),24, 37,721,1440)) #; rgrData3Dstag[:]=np.nan
        
        # _______________________________________________________________________
        # read the unstagered p-level data
        for v3 in range(len(name3Dvariables)):
            ## loadng the data  
            sFileName=s3Dfolder+sFolderYYMO+s3Dextension+ name3Dvariables[v3] +'.ll025sc.'+str(dDate.year)+str("%02d" % (dDate.month))+str("%02d" % (dDate.day))+'00_'+str(dDate.year)+str("%02d" % (dDate.month))+str("%02d" % (dDate.day))+str(23)+'.nc'
            # convert data to netcdf for faster processing
#             iFileNC='/glade/scratch/bblanc/'+sFileName.split('/')[-1]

            # read in the variables
            ncid=Dataset(sFileName, mode='r')
            rgrData3DAct[v3,:,:,:,:]=np.squeeze(ncid.variables[rgr3Dvariables[v3]][:])                
        ncid.close()

        for v3 in range(len(rgr3Dstaggerd)):
            # read in the staggered wind variables
            sFileName=s3Dfolder+sFolderYYMO+s3Dextension+ name3Dstaggerd[v3] +'.ll025uv.'+str(dDate.year)+str("%02d" % (dDate.month))+str("%02d" % (dDate.day))+'00'+'_'+str(dDate.year)+str("%02d" % (dDate.month))+str("%02d" % (dDate.day))+str(23)+'.nc'

            ncid=Dataset(sFileName, mode='r')
            rgrData3Dstag[v3,:,:,:,:]=np.squeeze(ncid.variables[rgr3Dstaggerd[v3]][:])
        ncid.close()

        # for ii in range(rgrData3DAct.shape[1]):
        #3D unstaggered variables
        tempk=rgrData3DAct[0,:,:,:,:]   
        tempkalltime[tt*24:(tt+1)*24,:,:,:]=rgrData3DAct[0,:,:,:,:]
        mixr=rgrData3DAct[1,:,:,:,:]
        geoph= rgrData3DAct[2,:,:,:,:]

        #u & v wind components at all 37 levels
        uu = rgrData3Dstag[0,:,:,:,:]
        vv = rgrData3Dstag[1,:,:,:,:]
        #sp, u & v wind components at 10m 
        ps = data2D_day[0,:,:,:]
        uu10 = data2D_day[1,:,:,:]
        vv10 =data2D_day[2,:,:,:]

        #height above surface at each grid cell          
        h_ab_surf = geoph/9.81 - rgrHeight

        # # __________________________________________________
        # # CALCULATE THE HAIL RELEVANT VARIABLES
        
        # RH at 850 and 500 hPa 
        RH_500[tt*24:(tt+1)*24,:,:] = rgrData3DAct[4,:,21,:,:]
        RH_850[tt*24:(tt+1)*24,:,:] = rgrData3DAct[4,:,30,:,:]

        # CIN [J kg-1]  
        # CAPE_CIN_3D=np.float32(np.zeros((2,monthlength,37,721,1440)))
        for ii in range(24):                
            CAPE_CIN_3D = wrf.cape_3d(pressurelvls, tempk[ii,:,:,:], mixr[ii,:,:,:],
                                    geoph[ii,:,:,:]/9.81, rgrHeight, ps[ii,:,:],
                                    ter_follow=False, missing=-99999., meta=False)
            indexes = np.argmax(CAPE_CIN_3D[0,:,:,:], axis=0)
            # Take CIN at the vertical level of Max CAPE
            for jj in range(rgrLon.shape[0]):
                for kk in range(rgrLon.shape[1]):
                    CIN3D[tt*24+ii,jj,kk] = CAPE_CIN_3D[1,indexes[jj,kk],jj,kk]
        # CAPE3D[:,:,:]=np.nanmax(np.array(CAPE_CIN_3D[0,:,:,:,:]), axis=1)
#         print("CAPE calculated")    


        # Vector Shear
        for ii in range(24):
            u6k= np.float32(wrf.interplevel(uu[ii,:,:,:],h_ab_surf[ii,:,:,:],6000,meta=False))
            v6k= np.float32(wrf.interplevel(vv[ii,:,:,:],h_ab_surf[ii,:,:,:],6000,meta=False))
            rgrVS06[tt*24+ii,:,:]=((u6k-uu10[ii,:,:])**2+
                                (v6k-vv10[ii,:,:])**2)**0.5

        # updraft relative helicity
        north = rgrLat[:,0] >= 0
        south = rgrLat[:,0] < 0
        
        for ii in range(24):
            rgrSRH06[tt*24+ii, north,:]=np.float32(wrf.srhel(uu[ii][::-1,north],
                                     vv[ii][::-1,north],
                                     geoph[ii][::-1,north]/9.81,
                                     rgrHeight[north,:],
                                     top=6000.0,
                                     meta=False,
                                     lats = np.array(rgrLat[north,:])))
            rgrSRH06[tt*24+ii, south,:]=np.float32(wrf.srhel(uu[ii][::-1,south],
                                     vv[ii][::-1,south],
                                     geoph[ii][::-1,south]/9.81,
                                     rgrHeight[south,:],
                                     top=6000.0,
                                     meta=False,
                                     lats = np.array(rgrLat[south,:])))
        # print ('storm relative helicity calculated')
# dewpoint Temperature 2m 
Dew_T[:,:,:] = rgrData2DAct[3,:,:,:]

# Total Totals index 
VT = tempkalltime[:,30,:,:] - tempkalltime[:,21,:,:]
dew_850 = tempkalltime[:,30,:,:] - (100-RH_850)/5
CT = dew_850 - tempkalltime[:,21,:,:]
TT = VT + CT
        

# THIS PART WILL GO INTO THE IO-FUNCTION FILES
# ________________________________________________________________________
# write the netcdf
iTime=np.where(np.isin(rgdFullTime, rgdTime1H) == 1)[0]
print ('    ----------------------')
print ('    Save data to '+sFileFin)
root_grp = Dataset(sFileFin, 'w', format='NETCDF4')
# dimensions
root_grp.createDimension('time', None)
root_grp.createDimension('longitude', rgrLat.shape[1])
root_grp.createDimension('latitude', rgrLon.shape[0])
# variables
lat = root_grp.createVariable('latitude', 'f4', ('latitude',))
lon = root_grp.createVariable('longitude', 'f4', ('longitude',))
time = root_grp.createVariable('time', 'f8', ('time',))

CINmax = root_grp.createVariable('CINmax', 'f4', ('time','latitude','longitude',),fill_value=-99999)
SRH06 = root_grp.createVariable('SRH06', 'f4', ('time','latitude','longitude',),fill_value=-99999)
VS06 = root_grp.createVariable('VS06', 'f4', ('time','latitude','longitude',),fill_value=-99999)
DewT = root_grp.createVariable('DewT', 'f4', ('time','latitude','longitude',),fill_value=-99999)
TotalTotals = root_grp.createVariable('TotalTotals', 'f4',('time','latitude','longitude',),fill_value=-99999)
RH850 = root_grp.createVariable('RH850', 'f4', ('time','latitude','longitude',),fill_value=-99999)
RH500 = root_grp.createVariable('RH500', 'f4', ('time','latitude','longitude',),fill_value=-99999)

time.calendar = "gregorian"
time.units = "hours since 1959-1-1 00:00:00"
time.standard_name = "time"
time.long_name = "time"
time.axis = "T"

lon.standard_name = "longitude"
lon.long_name = "longitude"
lon.units = "degrees_east"

lat.standard_name = "latitude"
lat.long_name = "latitude"
lat.units = "degrees_north"


# write data to netcdf
lat[:]=rgrLat[:,0]
lon[:]=rgrLon[0,:]
CINmax[:]=CIN3D
SRH06[:]=rgrSRH06
VS06[:]=rgrVS06
DewT[:]=Dew_T
TotalTotals[:]=TT
RH850[:]=RH_850
RH500[:]=RH_500


time[:]=iTime
root_grp.close()

print ('FINISHED')

