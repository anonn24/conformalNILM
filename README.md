## Distribution-Free Uncertainty Quantification and Calibration for Energy Disaggregation

Datasets:
1. REDD (Reference Energy Disaggregation Data Set)
2. Pecan Street Dataport Dataset
3. UK-DALE (Domestic Appliance-Level Electricity) Dataset
4. REFIT Dataset

This repository contains all the experiments conducted for ACM e-Energy 2025. The entire code is written using JAX and Flax.

### Contents:

1. Each folder named after a dataset from the list above contains code for experiments. 
2. README - This README explains the basic structure to the reader.
3. Hyperparameter.md - This has the details of the hyperparameters used for all experiments in all datasets.
4. reproducibility.md - This contains information on how to reproduce the experiments. 
5. error_bars.md - This showcases the error value mean and standard deviation for the Fridge and Dishwasher appliance from REDD dataset.
6. environment.yml - contains all the libraries and their versions to setup the environment for reproducibility. 

### How to access Dataset:
1. REDD: Login to http://redd.csail.mit.edu through login credentials. Perform pre-processing using NILMTK tools https://github.com/nilmtk/nilmtk. 
2. Pecan Street Dataport Dataset: Download the free 75 homes dataset from https://www.pecanstreet.org/dataport/. 
3. UK-DALE: Download ukdale.h5.zip from https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated. Convert .h5 file to pandas array and use metadata from https://data.ukedc.rl.ac.uk/cgi-bin/data_browser/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated/metadata?dataid= to get appliance level data and mains data of each house. Downsample to 1 min frequency if needed using NILMTK tools. https://github.com/nilmtk/nilmtk
4. REFIT: Download dataset from https://pureportal.strath.ac.uk/files/52873459/Processed_Data_CSV.7z. Extract it. Downsample it to 1 min frequency if needed using NILMTK tools. https://github.com/nilmtk/nilmtk 
