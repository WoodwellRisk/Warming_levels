# functions for calculating warming years

# load packages
import numpy as np
import pandas as pd
import fsspec
import xarray as xr



# examples of calling functions:
#wltable = calc_warming_years_rolling_mean(mms, 1.5, 21)
#wltable = calc_warming_years_temperature_window(mms, 1, .1)
#out = get_cmip6_data('ACCESS-CM2','r1i1p1f1','SSP585','tas','2020-01','2040-12','mydata.nc')
#out = get_cmip6_data_at_warming_years(model='ACCESS-CM2', member='r1i1p1f1', scenario='ssp585', cmip6_variable='tas', warming_years=np.array([2010,2021,2024]), year_window=5, outfilename='mydata.nc')



def calc_warming_years_rolling_mean(mms_table, warming_level, window_size):
    # mms_table is a pandas dataframe containing one row for each model, member, scenario combination that you wish to calculate warming years for
    # warming_level is the global warming level in degrees C to calculate warming years for
    # window_size is the number of years (integer) to include in the rolling mean window. 
    #   The table the risk team often uses employs a window_size of 21
    #   if the window_size is an odd number, the rolling mean is assigned to the middle year (e.g. for window_size=21, the rolling mean is assigned
    #     to the 11th year)
    #   if the window_size is an even number, the rolling mean is assigned to the year (window_size/2)+1 (e.g. for window_size=20, the rolling mean 
    #     is assigned to the 11th year; same approach as https://github.com/mathause/cmip_warming_levels)
    # returns a pandas dataframe that is the mms_table with one column added for the warming year
    #   model/member/scenario combinations that do not reach the warming level are assigned a value of NaN for the warming year
    
    # setup the output dataframe
    outdf = mms_table
    outdf['warming_year'] = np.nan
    
    # import the gmst data
    gmst_table = pd.read_csv('/home/abbylute/alute_bucket/warming_levels/data/gmst_tables/CMIP6_GMST_table_all.csv')

    # exclude years after 2100
    gmst_table = gmst_table.loc[gmst_table['year'] <= 2100]

    # calculate warming for each year relative to 1850-1900 base period
    gmst_table['warming'] = gmst_table['GMST'] - gmst_table['1850-1900']

    # find each model/member/scenario combination from the mms_table in the gmst_table and calculate warming years
    for i in range(mms_table.shape[0]):
        mod = mms_table.iloc[i,0]
        mem = mms_table.iloc[i,1]
        scen = mms_table.iloc[i,2]
    
        tab1 = gmst_table.loc[(gmst_table['model']==mod) & (gmst_table['member']==mem) & ((gmst_table['scenario']==scen) | (gmst_table['scenario']=='historical')), :].copy()
        
        tab1['warming_mean'] = tab1['warming'].rolling(window_size, center=True).mean()

        tab1 = tab1.loc[tab1['warming_mean'] > warming_level]
        if tab1.shape[0]>0:
            outdf.iloc[i,3] = tab1.iloc[0,3]
        else:
            outdf.iloc[i,3] = np.nan

    return outdf


def calc_warming_years_temperature_window(mms_table, warming_level, temp_tolerance):
    # mms_table is a pandas dataframe containing one row for each model, member, scenario combination that you wish to calculate warming years for
    # warming_level is the global warming level in degrees C to calculate warming years for
    # temp_tolerance is the temperature tolerance in degrees C around the warming level that is used to identify warming years.
    #   For example, if warming_level=2 and temp_tolerance=0.25, then years with warming >1.75 and <2.25 will be selected

    # returns a pandas dataframe that has the same columns as mms_table plus one column for the warming_year. 
    #   There is a row for each model/member/scenario/warming_year combination.
    #   model/member/scenario combinations that do not reach the warming level are given one row and assigned a value of NaN for the warming year

    # import the gmst data
    gmst_table = pd.read_csv('/home/abbylute/alute_bucket/warming_levels/data/gmst_tables/CMIP6_GMST_table_all.csv')

    # exclude years after 2100
    #gmst_table = gmst_table.loc[gmst_table['year'] <= 2100]

    # calculate warming for each year relative to 1850-1900 base period
    gmst_table['warming'] = gmst_table['GMST'] - gmst_table['1850-1900']
    
    # create a new copy of mms_table so that the original is not modified
    mms = mms_table.copy()
    
    # add historical scenario to mms_table so that historical warming years get incorporated as well
    mms.loc[len(mms.index)] = [mms['model'].iloc[0], mms['member'].iloc[0], 'historical']

    # select only model/member/scenario combinations that are in mms_table    
    gmst_table = pd.merge(gmst_table, mms, how='right', on=['model','member','scenario'])

    # filter to only years within temp_tolerance
    mn = warming_level - temp_tolerance
    mx = warming_level + temp_tolerance
    outdf = gmst_table.loc[(gmst_table['warming'] > mn) & (gmst_table['warming'] < mx)]
    
    # add a row with warming_year=NaN for any model/member/scenario combination that did not have any warming years
    outdf = pd.merge(outdf, mms, how='outer', on = ['model','member','scenario'])
    
    # clean up the output dataframe
    outdf = outdf.drop(labels=['GMST','1850-1900','warming'], axis=1)
    outdf = outdf.rename(columns={'year':'warming_year'})
    
    return outdf
    

def get_cmip6_data(model, member, scenario, cmip6_variable, start_yr_mo, end_yr_mo, outfilename):
    
    # model:           name of the cmip6 model to import data for
    # member:          ensemble member (aka realization) to import data for (e.g. r1i1p1f1)
    # scenario:        CMIP6 scenario to import data for (e.g. historical, ssp585)
    # cmip6_variable:  name of the cmip6 variable to import data for (e.g. tas, pr)
    # start_yr_mo:     the first year and month to grab data for. Format should be YYYY-MM. e.g. '2012-01' for January 2012
    # end_yr_mo:       the last year and month to grab data for. Format should be YYYY-MM.
    # outfilename:     file name to save data to. Should end in '.nc'. If outfilename=None, data will be returned but not saved to file
    
    # returns an xarray dataset (and can save a netcdf file) of global (spatially explicit) monthly cmip6 data for the variable of interest for the specified model, member, and scenario
    
    # check arguments
    member = member.lower()
    scenario = scenario.lower()
    
    if (scenario == 'historical') & ((int(start_yr_mo[0:4]) > 2014) | (int(end_yr_mo[0:4]) > 2014)):
        raise ValueError('When scenario="historical", start_yr_mo and end_yr_mo must reference dates before 2015')
    
    if (outfilename != None):
        if(outfilename[-3:] != '.nc'):
            raise ValueError('If you wish to save the file, please specify an outfilename ending in ".nc" (i.e. a netcdf file)')           
    
    # Open the CMIP6 zarr data catatalog 
    df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')

    # import cmip6 data
    if (scenario != 'historical') & (int(start_yr_mo[0:4]) < 2015) & (int(end_yr_mo[0:4]) < 2015):
        print('Warning: due to the time window selected, all data will be imported from historical runs, not the specified scenario')
        df_zarr = df.query("table_id == 'Amon' & variable_id == '" + cmip6_variable + "' & experiment_id == 'historical' & source_id == '" + model + "' & member_id == '" + member + "'")
        zstore = df_zarr.zstore.values[0]
        mapper = fsspec.get_mapper(zstore)            
        zz = xr.open_zarr(mapper, consolidated=True)
    elif (scenario != 'historical') & (int(start_yr_mo[0:4]) < 2015):
        print('Warning: due to the time window selected, some data will be imported from historical runs, not the specified scenario')
        df_zarr = df.query("table_id == 'Amon' & variable_id == '" + cmip6_variable + "' & experiment_id == 'historical' & source_id == '" + model + "' & member_id == '" + member + "'")
        zstore = df_zarr.zstore.values[0]
        mapper = fsspec.get_mapper(zstore)            
        zh = xr.open_zarr(mapper, consolidated=True)
        
        df_zarr = df.query("table_id == 'Amon' & variable_id == '" + cmip6_variable + "' & experiment_id == '" + scenario + "' & source_id == '" + model + "' & member_id == '" + member + "'")
        zstore = df_zarr.zstore.values[0]
        mapper = fsspec.get_mapper(zstore)            
        zz = xr.open_zarr(mapper, consolidated=True)

        zz = xr.concat([zh,zz], dim='time')
        zz['time'] = zz.time.astype("datetime64[ns]")
    else:
        df_zarr = df.query("table_id == 'Amon' & variable_id == '" + cmip6_variable + "' & experiment_id == '" + scenario + "' & source_id == '" + model + "' & member_id == '" + member + "'")
        zstore = df_zarr.zstore.values[0]
        mapper = fsspec.get_mapper(zstore)            
        zz = xr.open_zarr(mapper, consolidated=True)
    
    # prepare time dimension
    # make sure time is sorted
    zz = zz.sortby('time')
    # need to trim the timeframe down here to avoid errors later
    yrs = np.arange(int(start_yr_mo[0:4]), int(end_yr_mo[0:4])+1, 1)
    zz = zz.where(zz.time.dt.year.isin(yrs), drop=True)
    # then trim time to the requested time period
    if zz.time.dtype != '<M8[ns]':
        zz['time'] = zz.indexes['time'].to_datetimeindex()
    zz = zz.sel(time=slice(start_yr_mo, end_yr_mo))

    # make sure that x and y are lon and lat for consistency
    try:
        zz = zz.rename({'latitude':'lat','longitude':'lon'})
    except:
        1
    # similar for lat and lon bounds
    try:
        zz = zz.rename({'lat_bounds':'lat_bnds','lon_bounds':'lon_bnds'})
    except:
        1
            
    # shift longitude from 0 to 360 to -180 to 180
    zz.coords['lon'] = (zz.coords['lon'] + 180) % 360 - 180
    zz = zz.sortby(zz.lon)
            
    # clean up 
    zz = zz.drop(['height','time_bounds','time_bnds','lat_bounds','lon_bounds'], errors='ignore')
    zz = zz.assign_coords(model=model).expand_dims('model')
    zz = zz.assign_coords(member=member).expand_dims('member')
    zz = zz.assign_coords(scenario=scenario).expand_dims('scenario')

    # option to save the file as a netcdf
    if outfilename != None:
        print('writing file to ' + outfilename)
        zz.to_netcdf(outfilename)
    
    return zz



def get_cmip6_data_at_warming_years(model, member, scenario, cmip6_variable, warming_years, year_window=0, outfilename=None):
    
    # model:           name of the cmip6 model to import data for
    # member:          ensemble member (aka realization) to import data for (e.g. r1i1p1f1)
    # scenario:        CMIP6 scenario to import data for (e.g. historical, ssp585)
    # cmip6_variable:  name of the cmip6 variable to import data for (e.g. tas, pr)
    # warming_years:   a numpy array of years to extract data for. They do not need to be consecutive. (e.g. [2010,2014,2020,2021])
    # year_window:     an integer specifying additional years to include either side of each specified warming_year. 
    #                  For example, if warming_years = [2010,2017] and year_window=0, then only data for years 2010 and 2017 will be included.
    #                  If warming_years = [2010,2017] and year_window=2, then data within +/- 2 years of the warming_years will be included (i.e. 
    #                  [2008,2009,2010,2011,2012,2015,2016,2017,2018,2019])
    # outfilename:     file name to save data to. If outfilename=None, data will be returned but not saved to file. Otherwise it should end in '.nc'. 
    
    # returns an xarray dataset (and/or saves a netcdf file) of the warming years (annual) timeseries of the cmip6 variable with lat/lon

    # check arguments
    member = member.lower()
    scenario = scenario.lower()
    
    # fill out warming_years list based on year_window
    if year_window > 0:
        for y in range(1,year_window + 1):
            if y==1:
                wynew = warming_years
            wynew = np.append(wynew, warming_years - y)
            wynew = np.append(wynew, warming_years + y)
        warming_years = np.sort(wynew)
    
    # check that outfilename is none or a netcdf file
    if (outfilename != None):
        if(outfilename[-3:] != '.nc'):
            raise ValueError('If you wish to save the file, please specify an outfilename ending in ".nc" (i.e. a netcdf file)')           
    
    # Open the CMIP6 zarr data catatalog 
    df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')

    # split historical and future warming_years
    yrsh = warming_years[warming_years < 2015]
    yrsf = warming_years[warming_years > 2014]
    
    # import cmip6 data
    if len(yrsh)>0: # if there are any historical years
        df_zarr = df.query("table_id == 'Amon' & variable_id == '" + cmip6_variable + "' & experiment_id == 'historical' & source_id == '" + model + "' & member_id == '" + member + "'")
        
        if df_zarr.shape[0]==0:
            raise ValueError(cmip6_variable + ' data is not available from Google Cloud for ' + model + ', ' + member + ', for the historical period.')
            
        zstore = df_zarr.zstore.values[0]
        mapper = fsspec.get_mapper(zstore) 
        zh = xr.open_zarr(mapper, consolidated=True)
        zh = zh.where(zh.time.dt.year.isin(yrsh), drop=True)
    if len(yrsf)>0: # if there are any future years
        df_zarr = df.query("table_id == 'Amon' & variable_id == '" + cmip6_variable + "' & experiment_id == '" + scenario + "' & source_id == '" + model + "' & member_id == '" + member + "'")
        
        if df_zarr.shape[0]==0:
            raise ValueError(cmip6_variable + ' data is not available from Google Cloud for ' + model + ', ' + member + ', ' + scenario + '.')

        zstore = df_zarr.zstore.values[0]
        mapper = fsspec.get_mapper(zstore)            
        zf = xr.open_zarr(mapper, consolidated=True)
        zf = zf.where(zf.time.dt.year.isin(yrsf), drop=True)
        
    if len(yrsh) == len(warming_years): # if it's all historical years
        zz = zh
    elif len(yrsf) == len(warming_years): # if it's all future years
        zz = zf
    else: # if it's a mix of historical and future years
        zz = xr.concat([zh, zf], dim='time')
        zz['time'] = zz.time.astype("datetime64[ns]")
            
    # prepare time dimension
    # make sure time is sorted
    zz = zz.sortby('time')
    if zz.time.dtype != '<M8[ns]':
        zz['time'] = zz.indexes['time'].to_datetimeindex()

    # aggregate monthly data to annual timeseries
    if zz[cmip6_variable].units == 'K':
        # weight months by days in month
        month_length = zz.time.dt.days_in_month
        weights = (month_length.groupby("time.year") / month_length.groupby("time.year").sum())
        # Calculate the time-weighted average for each year
        zz[cmip6_variable] = (zz[cmip6_variable] * weights).groupby("time.year").sum(dim="time")
        zz[cmip6_variable].attrs = {'units':'K'}
    elif zz[cmip6_variable].units == 'kg m-2 s-1':
        # convert pr from kg m-2 s-1 to mm month-1
        sec_in_month = zz.time.dt.days_in_month * 24 * 60 * 60
        zz[cmip6_variable] = zz[cmip6_variable] * sec_in_month
        # sum precipitation across months to get annual precipitation
        zz[cmip6_variable] = zz[cmip6_variable].groupby("time.year").sum(dim="time")
        zz[cmip6_variable].attrs = {'units':'mm'}
    else:
        raise ValueError('This function does not know how to aggregate monthly data to annual for variables with units of ' + zz[cmip6_variable].units + '. Currently only units of K or kg m-2 s-1 are supported.')
    
    # make sure that x and y are lon and lat for consistency
    try:
        zz = zz.rename({'latitude':'lat','longitude':'lon'})
    except:
        1
    # similar for lat and lon bounds
    try:
        zz = zz.rename({'lat_bounds':'lat_bnds','lon_bounds':'lon_bnds'})
    except:
        1
         
    # shift longitude from 0 to 360 to -180 to 180
    zz.coords['lon'] = (zz.coords['lon'] + 180) % 360 - 180
    zz = zz.sortby(zz.lon)
    
    # clean up 
    zz = zz.drop(['height','time','time_bounds','time_bnds','lat_bounds','lon_bounds'], errors='ignore')
    zz = zz.assign_coords(model=model).expand_dims('model')
    zz = zz.assign_coords(member=member).expand_dims('member')
    zz = zz.assign_coords(scenario=scenario).expand_dims('scenario')

    # option to save the file as a netcdf
    if outfilename != None:
        print('writing file to ' + outfilename)
        zz.to_netcdf(outfilename)
    
    return zz
    