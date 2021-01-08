#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:35:02 2021

@author: leguillou
"""

import numpy as np
import xarray as xr
import os,sys

def ini(config, *args, **kwargs):
    """
    NAME
       ini
    DESCRIPTION
        Main function calling subfunctions considering the kind of init the
    user set
        Args:
            config (module): configuration module
    """
    
    if config.name_init == 'Steady_State':
        return ini_steady_state(config)
    else:
        sys.exit(config.name_init + ' not implemented yet')

def ini_steady_state(config, *args, **kwargs):
    """
    NAME
        ini_steady_state

    DESCRIPTION
        Create state grid and save to init file
        Args:
            config (module): configuration module
    """
    
    # Compute grid of the initialization file 
    lon = np.arange(config.lon_min, config.lon_max + config.dx, config.dx) % 360
    lat = np.arange(config.lat_min, config.lat_max + config.dy, config.dy) 
    
    lon2d,lat2d = np.meshgrid(lon,lat)
    dictout = {config.name_mod_lon: (('y', 'x'), lon2d),
               config.name_mod_lat: (('y', 'x'), lat2d)}

    for i, var in enumerate(config.name_mod_var):                                   
        if config.name_model=='QG1L' and i==2:
            # model parameter is included in state variable: we set default value
            f = 4*np.pi/86164*np.sin(lat*np.pi/180)
            K = (f/config.c)**2
            dictout[var] = (('y', 'x',), K.reshape([lat.size,lon.size]))
        else:            
            dictout[var] = (('y', 'x',), np.zeros((lat.size,lon.size)))
    ds = xr.Dataset(dictout)
    
    if not os.path.exists(config.tmp_DA_path):
        os.makedirs(config.tmp_DA_path)
        
    ds.to_netcdf(config.tmp_DA_path + config.name_init_file,
                 format='NETCDF3_CLASSIC')
    ds.close()
    