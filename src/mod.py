#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 22:36:20 2021

@author: leguillou
"""

from importlib.machinery import SourceFileLoader 
import sys
import xarray as xr
import numpy as np
import calendar

import switchvar, grid, tools

def mod(config,*args, **kwargs):
    """
    NAME
        mod

    DESCRIPTION
        Main function calling subfunctions for specific models
    """
    if config.name_model=='QG1L':
        return mod_qg1l(config,*args, **kwargs)
    else:
        sys.exit(config.name_analysis + ' not implemented yet')
        
def mod_qg1l(config,state_vector0_name=None, tint=None, present_date0=None,
             Nudging_term=None, Hbc=None, Wbc=None, state_vector_name=None,
             *args, **kwargs):
    """
    NAME
        mod_qg1l

    DESCRIPTION
        1.5 layer QG model: Call the quasi-geostrophic shallow water model (C. Ubelmann) 
        and perform nudging towards observations
   
    """

    # Model specific libraries
    SourceFileLoader("modgrid", "models/model_qgsw/modgrid.py").load_module() 

    SourceFileLoader("moddyn", "models/model_qgsw/moddyn.py").load_module() 
    
    SourceFileLoader("modelliptic", "models/model_qgsw/modelliptic.py").load_module() 
    
    qgsw = SourceFileLoader("qgsw", "models/model_qgsw/qgsw.py").load_module() 

    # Read state grid
    ds = xr.open_dataset(state_vector0_name)
    lon = ds[config.name_mod_lon].values % 360
    lat = ds[config.name_mod_lat].values
    if len(lon.shape)==1:
        lon2d,lat2d = np.meshgrid(lon,lat)
    else:
        lon2d = lon
        lat2d = lat
        
    ssh_0 = ds[config.name_mod_var[0]].values

    ny, nx = lon2d.shape

    # Get model parameter from model state or use default one
    flag_K = False
    f = 4*np.pi/86164*np.sin(lat2d*np.pi/180)
    if len(config.name_mod_var)>2 and config.name_mod_var[2] in ds:
        flag_K = True
        K = ds[config.name_mod_var[2]].values
        c = np.mean(f/np.sqrt(K))
    else:
        c = config.c
        K = (f/c)**2

    # Get potential vorticity from model state or compute it from SSH
    flag_pv = False
    if len(config.name_mod_var)>1 and config.name_mod_var[1] in ds:
        flag_pv = True
        pv_0 = ds[config.name_mod_var[1]].values
    else:
        pv_0 = switchvar.ssh2pv(ssh_0,lon2d,lat2d,c,name_grd=config.name_grd)

    deltat = np.abs(tint)

    ds.close()

    # Boundary conditions
    if config.flag_use_boundary_conditions:
        if Hbc is None or Wbc is None:
            # Compute boundary conditions online if not provided
            timestamp = calendar.timegm(present_date0.timetuple())
            Hbc, Wbc = grid.boundary_conditions(config.file_boundary_conditions,
                                                config.lenght_bc,
                                                config.name_var_bc,
                                                timestamp,
                                                lon2d,
                                                lat2d)
        Qbc = switchvar.ssh2pv(Hbc, lon2d, lat2d, c, name_grd=config.name_grd)
        ssh_0 = Wbc*Hbc + (1-Wbc)*ssh_0
        pv_0 = Wbc*Qbc + (1-Wbc)*pv_0
    else:
        Wbc = np.zeros_like(ssh_0)

    # Model propagation
    ssh_1, pv_1, trash = qgsw.qgsw(Hi=ssh_0, PVi=pv_0, c=c,
                                   lon=lon2d, lat=lat2d,
                                   tint=tint,
                                   dtout=deltat,
                                   dt=config.dtmodel,
                                   name_grd=config.name_grd,
                                   diff=config.only_diffusion,
                                   snu=config.cdiffus)

    # Nudging
    if Nudging_term is not None:
        # Nudging towards relative vorticity
        if np.any(np.isfinite(Nudging_term['rv'])):
            indNoNan = ~np.isnan(Nudging_term['rv'])
            pv_1[-1][indNoNan] += (1-Wbc[indNoNan]) *\
                Nudging_term['rv'][indNoNan]
        # Nudging towards ssh
        if np.any(np.isfinite(Nudging_term['ssh'])):
            g = 9.81
            indNoNan = ~np.isnan(Nudging_term['ssh'])
            pv_1[-1][indNoNan] -= (1-Wbc[indNoNan]) * (g/f[indNoNan]) *\
                K[indNoNan] * Nudging_term['ssh'][indNoNan]


            # Inversion pv -> ssh
            ssh_b = ssh_1[-1].copy()
            ssh_1[-1] = switchvar.pv2ssh(
                lon2d,lat2d,pv_1[-1],ssh_b,c,nitr=config.qgiter,name_grd=config.name_grd)
    
    if np.any(np.isnan(ssh_1)):
        sys.exit('Invalid value encountered in mod_qg1l')

    # Write output    
    propagated_state_vector = ssh_1[-1].ravel()
    if flag_pv:
        propagated_state_vector = np.concatenate((propagated_state_vector,pv_1[-1].ravel()))
    if flag_K:
         propagated_state_vector = np.concatenate((propagated_state_vector,K.ravel()))
         
    tools.vector_save(propagated_state_vector, lon, lat,
                      config.n_mod_var, config.name_mod_var,
                      config.name_mod_lon, config.name_mod_lat,
                      state_vector_name)

    return state_vector_name


