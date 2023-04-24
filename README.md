# Warming_levels

This repository contains code for calculating global warming levels from CMIP6 data. CMIP6 data is sourced programatically from Google Cloud in zarr format. This repo contains functions for calculating global mean surface temperature (GMST), multiple approaches for calculating warming years, and functions for aggregating warming years to warming levels. A comprehensive description of this work is available in the report at https://docs.google.com/document/d/1NgpzSMPKjkMQFcNMlJDy8rcISbtpEHkQRuPF887OaXQ/edit#.

## Content Description

1. warming_years.py contains several useful utility functions. 

   `calc_warming_years_rolling_mean()` calculates warming years using the commonly used approach of finding the first year when the rolling mean GMST centered on that year exceeds the target warming level. This function takes a model, member, scenario, target warming level, number of years and latest year to average over.

   `calc_warming_years_temperature_window()` calculates warming years by selecting any years that are within a temperature tolerance of the target warming level. It outputs all years that fit this criteria, allowing the user to average climate variables across these years or inspect difference between years at the target warming level. The function takes a model, member, scenario, target warming level, temperature tolerance, and last year to include.

   `get_cmip6_data()` returns an xarray dataset of global (spatially explicit) monthly data for the model, member, scenario, variable and time period specified. It can also save the data to a file if desired.

   `get_cmip6_data_at_warming_years()` returns an xarray dataset of global (spatially explicit) annual data for the model, member, scenario, and variable specified. Data is returned for the warming years specified, of if year_window is a positive integer (>0) then a buffer of years around each warming year are also returned. The dataset can also be save to a file if desired.

   `spatial_mean()` calculates the area weighted mean across lon and lat for arrays in an xarray dataset.

2. make_table_of_CMIP6_zarr_models.ipynb makes a table (all_zarr_models.csv) of all CMIP6 model/member combinations that have monthly air temperature (tas) data available on Google Cloud. A snapshot of this table is shown below.

![image](https://user-images.githubusercontent.com/30153868/209983992-b840c4c5-b640-4cba-a9d0-9f8d703c2661.png)


3. create_GMST_table_from_zarr_data.ipynb creates a table (CMIP6_GMST_table_all.csv) of GMST for each CMIP6 model, member, scenario, and year. The table also provides pre-industrial (1850-1900 inclusive) mean GMST for each model, member, and scenario to enable quick calculation of warming. A snapshot of this table is shown below.

![image](https://user-images.githubusercontent.com/30153868/209886751-b7b9e6bd-384b-44bb-a392-b65a195c4f1e.png)


4. make_table_of_zarr_warming_years_21yr_window.ipynb makes a table (warming_years_zarr_21yr_window.csv) of warming years for each model, member, and scenario and for warming levels of 1, 1.5, 2, 2.5, 3, 3.5, 4, and 4.5°C using a 21 year moving window approach. A snapshot of this table is shown below.

![image](https://user-images.githubusercontent.com/30153868/209886602-d2ca2d0c-7e83-45bc-ab81-bbb964e8197d.png)



5. make_table_of_CMIP6_first_and_last_years.ipynb creates a table (all_zarr_models_first_last_year.csv) that indicates the first and last year of data available for each model, member, scenario combination and for both tas and pr. This can be used to exclude from warming year calculations cases where data does not cover the full time period relevant to the warming level.
 
6. output_warming_years_datasets.ipynb creates datasets of warming amount (relative to 1850-1900) for every possible model, member, scenario combination available on Google Cloud, for each warming year identified via a warming level and temperature tolerance approach, and for each lat/lon point on a common grid. This results in one dataset per model/member/scenario combination. From these datasets, the code salso create aggregated datasets that are the average across years and members (resulting in one file for each model and scenario).

7. output_warming_years_annual_pr_and_tas_datasets.ipynb creates datasets of mean annual temperature or annual precipitation for every possible model, member, scenario combination available on Google Cloud, for each warming year identified via a warming level and temperature tolerance approach, and for each lat/lon point on a common grid. This results in one dataset per model/member/scenario combination. From these datasets, the code also creates aggregated datasets that are the average across years and members (resulting in one file for each model and scenario).

8. make_warming_year_table_for_21yr_best_subset.ipynb creates a table of warming years using the 21 year moving window approach and the best subset of models, members, and scenarios as described in the report. Can also be used for generating warming year tables for other moving windows (other than 21 years).

9. make_warming_year_table_for_temperature_window_best_subset.ipynb creates a table of warming years using the temperature window approach and the best subset of models, members, and scenarios as described in the report. Can be used for generating warming year tables for other temperature windows and warming levels.

10. output_aggregated_warming_level_datasets.ipynb ingests all the warming years in the 'best subset' warming year tables and averages across all models, members, scenarios, and years at once for each warming level to create two netCDF files: one for mean annual temperature and one for annual precipitation. Both contain data for each warming level on a 0.5° global grid.

11. file_control_template.py allows the user to indicate file and directory locations. Once this file is edited to add these locations, the file should be saved as file_control.py. file_control.py is ingested by the other scripts in order to create paths to data inputs and outputs.


## Requirements
Warming_levels codes are written in Python 3. The Python environment used to run the code is available in the warming_levels.yml file.


## Examples
Please see Examples.ipynb.


## Citation Info
