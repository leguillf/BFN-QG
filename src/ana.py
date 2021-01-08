#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:59:11 2021

@author: leguillou
"""
from datetime import timedelta
import xarray as xr
import sys
import numpy as np 
import calendar
import os
import matplotlib.pylab as plt


import mod,tools,grid



def ana(config, *args, **kwargs):
    """
    NAME
        ana

    DESCRIPTION
        Main function calling subfunctions for specific Data Assimilation algorithms
    """
    
    if config.name_analysis=='BFN':
        return ana_bfn(config)
    else:
        sys.exit(config.name_analysis + ' not implemented yet')
        
        
    
def ana_bfn(config,dict_obs=None, *args, **kwargs):
    """
    NAME
        ana_bfn

    DESCRIPTION
        Perform a BFN experiment on altimetric data and save the results on output directory
    
    """
    
    
    import bfn_functions as bfn

    # Flag initialization
    bfn_first_window = True
    bfn_last_window = False
    if dict_obs is None:
        call_obs_func = True
        import obs
    else:
        call_obs_func = False
    # BFN middle date initialization
    middle_bfn_date = config.init_date
    # In the case of Nudging (i.e. bfn_max_iteration=1), set the bfn window length as the entire period of the experience
    if config.bfn_max_iteration==1:
        print('bfn_max_iteration has been set to 1 --> '
              + 'Only one forth loop will be done on the entiere period of the experience')
        new_bfn_window_size = config.final_date - config.init_date
    else:
        new_bfn_window_size = config.bfn_window_size

    # Use temp_DA_path to save the projections
    if config.save_obs_proj:
        if config.path_save_proj is None:
            pathsaveproj = config.tmp_DA_path
        else:
            pathsaveproj = config.path_save_proj
    else:
        pathsaveproj = None

    # Main time loop
    while (middle_bfn_date <= config.final_date) and not bfn_last_window :
        print('\n*** BFN window ***')
        #############
        # 1. SET-UP #
        #############
        # BFN period
        init_bfn_date = max(config.init_date, middle_bfn_date - new_bfn_window_size/2)
        init_bfn_date += timedelta(seconds=(init_bfn_date - config.init_date).total_seconds()\
                         / config.bfn_propation_timestep.total_seconds()%1)
        middle_bfn_date = max(middle_bfn_date, config.init_date + new_bfn_window_size/2)
        if ((middle_bfn_date + new_bfn_window_size/2) >= config.final_date):
            bfn_last_window = True
            final_bfn_date = config.final_date
        else:
            final_bfn_date = init_bfn_date + new_bfn_window_size
        print('\nfrom ', init_bfn_date, ' to ', final_bfn_date)
        # propagation timestep
        one_time_step = config.bfn_propation_timestep
        # Initialize nc files
        if bfn_first_window:
            present_date_forward0 = init_bfn_date
            present_file_forward0 = config.tmp_DA_path + "/state_forward0.nc"
            present_file_backward0 = config.tmp_DA_path + "/state_backward0.nc"
            present_file_forward = config.tmp_DA_path + "/state_forward.nc"
            present_file_backward = config.tmp_DA_path + "/state_backward.nc"
        analyzed_vectors_names = config.tmp_DA_path + "/state_analyzed.nc"

        ########################
        # 2.READING STATE GRID #
        ########################
        print('\n* Reading state grid *')
         # Read box boundaries
        with xr.open_dataset(config.tmp_DA_path+config.name_init_file) as grd:
            lon = grd[config.name_mod_lon].values
            lat = grd[config.name_mod_lat].values
            if len(lon.shape)==1:
                ny,nx = lat.size,lon.size
            else:
                ny,nx = lon.shape

        ########################
        # 3. Create BFN object #
        ########################
        print('\n* Initialize BFN *')
        if config.name_model == 'QG1L':
            bfn_obj = bfn.bfn_qg1l(
                 init_bfn_date,
                 final_bfn_date,
                 config.assimilation_time_step,
                 one_time_step,
                 lon,
                 lat,
                 config.name_mod_var,
                 config.name_grd,
                 config.dist_scale,
                 pathsaveproj,
                 'projections_' + config.name_domain + '_' + '_'.join(config.satellite),
                 config.c,
                 config.flag_plot,
                 config.scalenudg)

        elif config.name_model == 'QGML':
            bfn_obj = bfn.bfn_qgml(
                 init_bfn_date,
                 final_bfn_date,
                 config.assimilation_time_step,
                 one_time_step,
                 lon,
                 lat,
                 config.name_mod_var,
                 config.name_grd,
                 config.dist_scale,
                 config.Rom,
                 config.Fr,
                 config.dh,
                 config.N,
                 config.L0,
                 pathsaveproj,
                 'projections_' + config.name_domain + '_' + '_'.join(config.satellite),
                 config.flag_plot,
                 config.scalenudg)

        else:
            print('Error: No BFN class implemented for', config.name_model, 'model')
            sys.exit()

        ######################################
        # 4. BOUNDARY AND INITIAL CONDITIONS #
        ######################################
        print("\n* Boundary and initial conditions *")
        # Boundary condition
        if config.flag_use_boundary_conditions:
            timestamps = np.arange(calendar.timegm(init_bfn_date.timetuple()),
                                   calendar.timegm(final_bfn_date.timetuple()),
                                   one_time_step.total_seconds())

            bc_field, bc_weight = grid.boundary_conditions(config.file_boundary_conditions,
                                                            config.lenght_bc,
                                                            config.name_var_bc,
                                                            timestamps,
                                                            lon,
                                                            lat,
                                                            config.flag_plot,
                                                            bfn_obj.sponge)

        else:
            bc_field = bc_weight = bc_field_t = None
        # Initial condition
        if bfn_first_window:
            # Use previous state as initialization
            init_file = config.path_save + config.name_exp_save\
                        + '_y' + str(init_bfn_date.year)\
                        + 'm' + str(init_bfn_date.month).zfill(2)\
                        + 'd' + str(init_bfn_date.day).zfill(2)\
                        + 'h' + str(init_bfn_date.hour).zfill(2)\
                        + str(init_bfn_date.minute).zfill(2) + '.nc'
            if not os.path.isfile(init_file) :
                restart = False
                print(init_file, " : Init file is not present for nudging."
                      + "We will use the current state vectors...")
                cmd = "cp "+ config.tmp_DA_path+config.name_init_file + " " +\
                    present_file_forward0
                os.system(cmd)
            else:
                restart = True
                print(init_file, " is used as initialization")
                cmd = "cp " + init_file + " " + present_file_forward0
                os.system(cmd)
        elif config.bfn_window_overlap:
            # Use last state from the last forward loop as initialization
            name_previous = config.name_exp_save\
                            + '_y' + str(init_bfn_date.year)\
                            + 'm' + str(init_bfn_date.month).zfill(2)\
                            + 'd' + str(init_bfn_date.day).zfill(2)\
                            + 'h' + str(init_bfn_date.hour).zfill(2)\
                            + str(init_bfn_date.minute).zfill(2) + '.nc'
            filename_forward = config.tmp_DA_path + '/BFN_forth_' + name_previous
            cmd = 'cp ' + filename_forward + ' ' + present_file_forward0
            os.system(cmd)

        ###################
        # 5. Observations #
        ###################
        # Selection
        print('\n* Select observations *')
        
        if call_obs_func:
            print('Calling obs_all_observationcheck function...')
            dict_obs_it = obs.obs(config)
            bfn_obj.select_obs(dict_obs_it)
            dict_obs_it.clear()
            del dict_obs_it
        else:
            bfn_obj.select_obs(dict_obs)

        # Projection
        print('\n* Project observations *')
        bfn_obj.do_projections()

        ###############
        # 6. BFN LOOP #
        ###############
        err_bfn0 = 0
        err_bfn1 = 0
        bfn_iter = 0
        Nold_t = None

        while bfn_iter==0 or\
             (bfn_iter < config.bfn_max_iteration
              and abs(err_bfn0-err_bfn1)/err_bfn1 > config.bfn_criterion):

            if bfn_iter>0:
                present_date_forward0 = init_bfn_date
                # Save last backward analysis as init forward file
                cmd0 = "cp " + present_file_backward0 + " " + present_file_forward0
                os.system(cmd0)

            err_bfn0 = err_bfn1
            bfn_iter += 1
            if bfn_iter == config.bfn_max_iteration:
                print('\nMaximum number of iterations achieved ('
                      + str(config.bfn_max_iteration)
                      + ') --> last Forth loop !!')

            ###################
            # 6.1. FORTH LOOP #
            ###################
            print("\n* Forward loop " + str(bfn_iter) + " *")
            while present_date_forward0 < final_bfn_date :

                #print('i = ' + str(bfn_iter) + ' forward : ' +str(present_date_forward0)  )
                # Retrieve corresponding time index for the forward loop
                iforward = int((present_date_forward0 - init_bfn_date)/one_time_step)

                # Get BC field
                if bc_field is not None:
                    bc_field_t = bc_field[iforward]

                # Propagate the state by nudging the model vorticity towards the 2D observations
                present_file_forward = mod.mod(config,
                                      state_vector0_name=present_file_forward0,
                                      present_date0=present_date_forward0,
                                      tint=+one_time_step.total_seconds(),
                                      Nudging_term=Nold_t,
                                      Hbc=bc_field_t,
                                      Wbc=bc_weight,
                                      state_vector_name=present_file_forward)


                # Time increment
                present_date_forward = present_date_forward0 + one_time_step

                # Get analysis
                analysis = tools.vectorize(present_file_forward,
                                         config.name_mod_var,
                                         )

                # Nudging term (next time step)
                N_t = bfn_obj.compute_nudging_term(
                        present_date_forward, analysis
                        )

                # Update model parameter
                if config.name_model == 'QG1L':
                    analysis_updated = bfn_obj.update_parameter(
                            analysis, Nold_t, N_t, bc_weight, way=1
                            )

                    if np.any(analysis_updated!=analysis):
                        tools.vector_save(
                                analysis_updated, lon, lat,
                                config.n_mod_var, config.name_mod_var,
                                config.name_mod_lon, config.name_mod_lat,
                                present_file_forward)

                # Save output every *saveoutput_time_step*
                if (((present_date_forward - config.init_date)/config.saveoutput_time_step)%1 == 0)\
                   & (present_date_forward>config.init_date) :
                        name_save = config.name_exp_save + '_' + str(iforward).zfill(5) + '.nc'
                        filename_forward = config.tmp_DA_path + '/BFN_forth_' + name_save
                        save_presentoutputs(present_file_forward,
                                            present_date_forward,
                                            filename=filename_forward)
                        if config.save_bfn_trajectory:
                            filename_traj = config.path_save + 'BFN_' + str(middle_bfn_date)[:10]\
                                       + '_forth_' + str(bfn_iter) + '/' + name_save

                            if not os.path.exists(os.path.dirname(filename_traj)):
                                os.makedirs(os.path.dirname(filename_traj))
                            save_presentoutputs(present_file_forward,
                                                present_date_forward,
                                                filename=filename_traj)

                # Save updated state vector as initial state vector
                cmd0 = "cp " + present_file_forward + " " + present_file_forward0
                os.system(cmd0)

                # Time update
                present_date_forward0 = present_date_forward
                Nold_t = N_t

            # Plot for debugging
            if config.flag_plot > 0:
                analysis = analysis.reshape((config.n_mod_var, ny, nx)) # SSH, PV
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=((10, 5)))
                p1 = ax1.pcolormesh(lon, lat, analysis[1],shading='auto')
                p2 = ax2.pcolormesh(lon, lat, analysis[0],shading='auto')
                plt.colorbar(p1, ax=ax1)
                plt.colorbar(p2, ax=ax2)
                ax1.set_title('Potential vorticity')
                ax2.set_title('SSH')
                plt.show()

            ##################
            # 6.2. BACK LOOP #
            ##################
            if  bfn_iter < config.bfn_max_iteration:
                print("\n* Backward loop " + str(bfn_iter) + " *")
                present_date_backward0 = final_bfn_date
                # Save last forward analysis as init backward file
                cmd0 = "cp " + present_file_forward0 + " " + present_file_backward0
                os.system(cmd0)

                while present_date_backward0 > init_bfn_date :
                    #print('i = ' + str(bfn_iter) + ' backward : ' +str(present_date_backward0)  )

                    # Retrieve corresponding time index for the backward loop
                    ibackward = int((present_date_backward0 - init_bfn_date)/one_time_step)

                    # Get BC field
                    if bc_field is not None:
                        bc_field_t = bc_field[ibackward-1]

                    # Propagate the state by nudging the model vorticity towards the 2D observations
                    present_file_backward = mod.mod(config,
                            state_vector0_name=present_file_backward0,
                            present_date0=present_date_backward0,
                            tint=-one_time_step.total_seconds(),
                            Nudging_term=Nold_t,
                            Hbc=bc_field_t,
                            Wbc=bc_weight,
                            state_vector_name=present_file_backward)
                    
                    
                    # Time increment
                    present_date_backward = present_date_backward0 - one_time_step

                    # Get analysis
                    analysis = tools.vectorize(present_file_backward,
                                             config.name_mod_var)

                    # Nudging term (next time step)
                    N_t = bfn_obj.compute_nudging_term(
                            present_date_backward,
                            analysis
                            )

                    # Update model parameter
                    if config.name_model == 'QG1L':
                        analysis_updated = bfn_obj.update_parameter(
                            analysis, Nold_t, N_t, bc_weight, way=-1
                            )

                        if np.any(analysis_updated!=analysis):
                            tools.vector_save(
                                    analysis_updated, lon, lat,
                                    config.n_mod_var, config.name_mod_var,
                                    config.name_mod_lon, config.name_mod_lat,
                                    present_file_backward)

                    # Save output every *saveoutput_time_step*
                    if (((present_date_backward - config.init_date)/config.saveoutput_time_step)%1 == 0)\
                       & (present_date_backward>=config.init_date) :
                            name_save = config.name_exp_save + '_' + str(ibackward).zfill(5) + '.nc'
                            filename_backward = config.tmp_DA_path + '/BFN_back_' + name_save
                            save_presentoutputs(present_file_backward,
                                                present_date_backward,
                                                filename=filename_backward)
                            if config.save_bfn_trajectory:
                                filename_traj = config.path_save + 'BFN_' + str(middle_bfn_date)[:10]\
                                           + '_back_' + str(bfn_iter) + '/' + name_save

                                if not os.path.exists(os.path.dirname(filename_traj)):
                                    os.makedirs(os.path.dirname(filename_traj))
                                save_presentoutputs(present_file_backward,
                                                    present_date_backward,
                                                    filename=filename_traj)

                    # Save updated state vector as initial state vector
                    cmd0 = "cp " + present_file_backward + " " + present_file_backward0
                    os.system(cmd0)

                    # Time update
                    present_date_backward0 = present_date_backward
                    Nold_t = N_t

                if config.flag_plot > 0:
                    analysis = analysis.reshape((config.n_mod_var, ny, nx))  # SSH, PV
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=((10, 5)))
                    p1 = ax1.pcolormesh(lon, lat, analysis[1],shading='auto')
                    p2 = ax2.pcolormesh(lon, lat, analysis[0],shading='auto')
                    plt.colorbar(p1, ax=ax1)
                    plt.colorbar(p2, ax=ax2)
                    ax1.set_title('Potential vorticity')
                    ax2.set_title('SSH')
                    plt.show()

            #########################
            # 6.3. CONVERGENCE TEST #
            #########################
            if bfn_iter < config.bfn_max_iteration:
                print('\n* Convergence test *')
                err_bfn1 = bfn_obj.convergence(
                                        path_forth=config.tmp_DA_path + '/BFN_forth_',
                                        path_back=config.tmp_DA_path + '/BFN_back_'
                                        )


        print("\n* End of the BFN loop after " + str(bfn_iter) + " iterations *")

        #####################
        # 7. SAVING OUTPUTS #
        #####################
        print('\n* Saving last forth loop as outputs for the following dates : *')
        # Set the saving temporal window
        if config.bfn_max_iteration==1:
            write_date_min = init_bfn_date
            write_date_max = final_bfn_date
        elif bfn_first_window:
            write_date_min = init_bfn_date
            write_date_max = init_bfn_date + new_bfn_window_size/2 + config.bfn_window_output/2
        elif bfn_last_window:
            write_date_min = middle_bfn_date - config.bfn_window_output/2
            write_date_max = final_bfn_date
        else:
            write_date_min = middle_bfn_date - config.bfn_window_output/2
            write_date_max = middle_bfn_date + config.bfn_window_output/2
        # Write outputs in the saving temporal window

        present_date = init_bfn_date
        while present_date < final_bfn_date :
            present_date += one_time_step
            if (present_date > write_date_min) & (present_date <= write_date_max) :
                # Save output every *saveoutput_time_step*
                if (((present_date - config.init_date).total_seconds()
                   /config.saveoutput_time_step.total_seconds())%1 == 0)\
                   & (present_date>config.init_date)\
                   & (present_date<=config.final_date) :
                    print(present_date, end=' / ')                
                    # Read current converged state
                    iforward = int((present_date - init_bfn_date)/one_time_step) - 1
                    name_save = config.name_exp_save + '_' + str(iforward).zfill(5) + '.nc'
                    current_file = config.tmp_DA_path + '/BFN_forth_' + name_save
                    vars_current = tools.vectorize(current_file,
                                                 config.name_mod_var,
                                                )
                    # Smooth with previous BFN window
                    if config.bfn_window_overlap and (not bfn_first_window or restart):
                        # Read previous output at this timestamp
                        previous_file = config.path_save + config.name_exp_save\
                                        + '_y'+str(present_date.year)\
                                        + 'm'+str(present_date.month).zfill(2)\
                                        + 'd'+str(present_date.day).zfill(2)\
                                        + 'h'+str(present_date.hour).zfill(2)\
                                        + str(present_date.minute).zfill(2) + \
                                            '.nc'
                        if os.path.isfile(previous_file):
                            vars_previous = tools.vectorize(previous_file,
                                                                   config.name_mod_var
                                                                )
                            # weight coefficients
                            W1 = max((middle_bfn_date - present_date)
                                     / (config.bfn_window_output/2), 0)
                            W2 = min((present_date - write_date_min)
                                     / (config.bfn_window_output/2), 1)
                            analysis = W1*vars_previous + W2*vars_current
                        else:
                            analysis = vars_current
                    else:
                        analysis = vars_current
                    tools.vector_save(analysis, lon, lat,
                                    config.n_mod_var,
                                    config.name_mod_var,
                                    config.name_mod_lon,
                                    config.name_mod_lat,
                                    analyzed_vectors_names,
                                    date=present_date)
                    # Save output
                    if config.saveoutputs:
                        save_presentoutputs(analyzed_vectors_names,present_date,
                                            name_exp=config.name_experiment,
                                            path=config.path_save)
        print()
        ########################
        # 8. PARAMETERS UPDATE #
        ########################
        if config.bfn_window_overlap:
            window_lag = config.bfn_window_output/2
        else:
            window_lag = config.bfn_window_output

        if bfn_first_window:
            middle_bfn_date = config.init_date + new_bfn_window_size/2 + window_lag
            bfn_first_window = False
        else:
            middle_bfn_date += window_lag

    ###############
    # 9. CLEANING #
    ###############
    print("Cleaning")
    # Remove state file
    cmd = "rm " + present_file_forward + " " + present_file_forward0\
          + " " + present_file_backward + " " + present_file_backward0\
          + " " + analyzed_vectors_names
    os.system(cmd)
    
   
    
   
    


def save_presentoutputs(state_vector_name,date0,name_exp=None,path=None,filename=None):

    
    if filename is None:
        if os.path.isdir(path)==False:
            os.makedirs(path)
    
        year0 = str(date0.year)
        month0 = str(date0.month).zfill(2)
        day0 = str(date0.day).zfill(2)
        hour0 = str(date0.hour).zfill(2)
        minute0 = str(date0.minute).zfill(2)
        filename = path + name_exp + '_y' + year0 + 'm' + month0\
                   + 'd' + day0 + 'h' + hour0 + minute0 + '.nc'

    cmd2='cp '+state_vector_name+' '+filename
    os.system(cmd2)
    

    return