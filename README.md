# BFN-QG
BFN-QG is a tool allowing to map SSH by assimilating altimetric data into a quasi-geostrophic (QG) model. The data assimilation scheme used here is the Back and Forth Nudging (BFN; ). The code is the one used in Le Guillou et al. 2020 (refer to this paper for insights on the scientific context, the method and performances of the mapping algorithm).

# Structure of the code
The source code is located in the `src/` directory.
- `ini.py`: handle the initialization of the state grid.
- `obs.py`: handle the selection of observations (Nadir and/or SWOT altimetry data). For now, the tool handle only simulated observations provided by the SWOT simulator.
- `ana.py`: handle the data assimilation procedure. For now, only the BFN is implemented. 
- `mod.py`: Call a specific model operator that should be located in the `models` directory. For now, only a 1.5 layer QG model is present.

Each script mentioned just above is composed of a main function (which as the same name as the scritp name) that calls one subfunction specific to one usage of the tool. It is then easy for the user to add additional functions (for instance a 2-layer QG model or a 4Dvar data assimilation scheme) to the tool. 

# Study case
An example is provided in addition to the code. This example is taken from a [SSH Mapping Data Challenge (2020a)](https://github.com/ocean-data-challenges/2020a_SSH_mapping_NATL60).
## Download the data
The data is hosted [here](https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/catalog/meomopendap/extract/ocean-data-challenges/dc_data1/catalog.html) with the following directory structure

```
. 
|-- dc_obs
|   |-- 2020a_SSH_mapping_NATL60_topex-poseidon_interleaved.nc
|   |-- 2020a_SSH_mapping_NATL60_nadir_swot.nc 
|   |-- 2020a_SSH_mapping_NATL60_karin_swot.nc
|   |-- 2020a_SSH_mapping_NATL60_jason1.nc
|   |-- 2020a_SSH_mapping_NATL60_geosat2.nc
|   |-- 2020a_SSH_mapping_NATL60_envisat.nc

|-- dc_ref
|   |-- NATL60-CJM165_GULFSTREAM_y****m**d**.1h_SSH.nc

```

To start out download the *observation* dataset (dc_obs, 285M) using : 
```shell
wget https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/dc_data1/dc_obs.tar.gz --no-check-certificate
```

and the *reference* dataset (dc_ref, 11G) using (*this step may take several minutes*) : 

```shell
wget https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/dc_data1/dc_ref.tar.gz 
```
and then uncompress the files using `tar -xvf <file>.tar.gz`. You may also use `ftp`, `rsync` or `curl`to donwload the data.  

## Launch a BFN experiment
In the `notebooks/` directory, launch the notebook called `example_run_bfn.ipynb`. This notebook explains and performs a BFN experiment based on the previously observed data. 

# Launch your own experiment 
To launch an other BFN experiment, you can set-up a new configuration file in the `configs/` directory. All the parameters needed are explained in the `config_example.py` file. Then, you can re-use the `example_run_bfn.ipynb` notebook by specifying the name of your configuration file.
