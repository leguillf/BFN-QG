#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 20:15:09 2021

@author: leguillou
"""
import xarray as xr
import numpy as np

from sat import read_satellite_info

def obs(config, *args, **kwargs):
    """
    NAME
        obs

    DESCRIPTION
        Main function calling subfunctions considering the kind of satellite observations
        Args:
            config (module): configuration module

        Param:

        Returns:
            dict_obs (dictionary): the keys are the dates of observations, and the values are dictionaries gathering all information
            needed to assimilate these observations
    """
    
    # Read box boundaries
    grd = xr.open_dataset(config.tmp_DA_path+config.name_init_file)
    lon = grd[config.name_mod_lon].values
    lat = grd[config.name_mod_lat].values
    bbox = [lon.min(),lon.max(),lat.min(),lat.max()]
    
    # Compute output observation dictionnary
    dict_obs = {}
    
    assim_dates = []
    date = config.init_date
    while date<=config.final_date:
        assim_dates.append(date)
        date += config.assimilation_time_step
        
    for sat in config.satellite:
        print('* for sat',sat,':')
        # Read satellite info
        sat_info = read_satellite_info(config,sat)
        print(sat_info)
        if sat_info.kind=='swot_simulator':
            _obs_swot_simulator(assim_dates, dict_obs, sat_info, 
                                config.assimilation_time_step, 
                                config.tmp_DA_path,bbox)
            
    return dict_obs


def _obs_swot_simulator(dt_list, dict_obs, sat_info, dt_timestep, out_path,bbox):
    """
    NAME
        _obs_swot_simulator

    DESCRIPTION
        Subfunction handling observations generated from swotsimulator module
        
    """
    # Read obs
    ds = xr.open_mfdataset(sat_info.path + sat_info.name,combine='by_coords')
    time_obs = ds[sat_info.name_obs_time]
    ds = ds[sat_info.name_obs_grd + sat_info.name_obs_var]
    
    # Select sub area
    lon_obs = ds[sat_info.name_obs_lon] % 360
    lat_obs = ds[sat_info.name_obs_lat]
    ds = ds.where((bbox[0]<=lon_obs) & (bbox[1]>=lon_obs) & 
                  (bbox[2]<=lat_obs) & (bbox[3]>=lat_obs), drop=True)
                  
    # Time loop
    for dt_curr in dt_list:
        
        dt1 = np.datetime64(dt_curr-dt_timestep/2)
        dt2 = np.datetime64(dt_curr+dt_timestep/2)
        
        _ds = ds.where((dt1<=time_obs) & (time_obs<=dt2), drop=True)
        
        if _ds[sat_info.name_obs_time].size>0:
            # Save the selected dataset in a new nc file
            date = dt_curr.strftime('%Y%m%d_%Hh%M')
            path = out_path + 'obs_' + sat_info.satellite + '_' +\
                '_'.join(sat_info.name_obs_var) + '_' + date + '.nc'
            print(dt_curr,': '+path)
            _ds[sat_info.name_obs_time].encoding.pop("_FillValue", None)
            _ds.to_netcdf(path,engine='scipy')
            _ds.close()
            # Add the path of the new nc file in the dictionnary
            if dt_curr in dict_obs:
                    dict_obs[dt_curr]['satellite'].append(sat_info)
                    dict_obs[dt_curr]['obs_name'].append(path)
            else:
                dict_obs[dt_curr] = {}
                dict_obs[dt_curr]['satellite'] = [sat_info]
                dict_obs[dt_curr]['obs_name'] = [path]
    
        

