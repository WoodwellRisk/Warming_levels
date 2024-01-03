# functions for calculating warming years

# load packages
import numpy as np
import pandas as pd
import fsspec
import xarray as xr
from file_control import gmst_table_dir
import dask


def calc_warming_years_rolling_mean(mms_table, warming_level, window_size, max_year):
    """Calculate warming years using a rolling window approach.
    
    This function calculates warming years for each model/member/scenario 
    combination in mms_table and for the specified warming level as the middle 
    year of the first period of years (of length window_size) with mean annual 
    temperature greater than the specified warming level.
    
    Args:
        mms_table (dataframe) : pandas dataframe containing one row for each 
        model, member, scenario combination that you wish to calculate warming 
        years for.
    
        warming_level (float) : global warming level in degrees C to calculate 
        warming years for.
        
        window_size (int) : number of years to include in the rolling mean 
        window. The table the risk team often uses employs a window_size of 
        21. If the window_size is an odd number, the rolling mean is assigned 
        to the middle year (e.g. for window_size=21, the rolling mean is 
        assigned to the 11th year). If the window_size is an even number, the 
        rolling mean is assigned to the year (window_size/2)+1 (e.g. for 
        window_size=20, the rolling mean is assigned to the 11th year; same 
        approach as https://github.com/mathause/cmip_warming_levels)
        
        max_year (int) : last year to include in calculations.
    
    Returns: 
        A pandas dataframe that is the mms_table with one column added for the 
        warming year. Model/member/scenario combinations that do not reach the 
        warming level are assigned a value of NaN for the warming year
    
    Example:
        mms = pd.DataFrame({'model': ['ACCESS-CM2']*3,
                            'member': ['r1i1p1f1']*3,
                            'scenario': ['ssp126','ssp245','ssp585']})
        warming_year_table = calc_warming_years_rolling_mean(mms, 1.5, 21,
        2100)
    
    """
    
    # setup the output dataframe
    outdf = mms_table
    outdf['warming_year'] = np.nan
    
    # import the gmst data
    gmst_table = pd.read_csv(gmst_table_dir + 'CMIP6_GMST_table_all.csv')

    # exclude years after max_year
    gmst_table = gmst_table.loc[gmst_table['year'] <= max_year]

    # calculate warming for each year relative to 1850-1900 base period
    gmst_table['warming'] = gmst_table['GMST'] - gmst_table['1850-1900']

    # find each model/member/scenario combination from the mms_table in the gmst_table and calculate warming years
    for i in range(mms_table.shape[0]):
        mod = mms_table.iloc[i,0]
        mem = mms_table.iloc[i,1]
        scen = mms_table.iloc[i,2]

        tabf = gmst_table.loc[(gmst_table['model']==mod) & 
                              (gmst_table['member']==mem) & 
                              (gmst_table['scenario']==scen), :].copy()
        tabh = gmst_table.loc[(gmst_table['model']==mod) & 
                              (gmst_table['member']==mem) & 
                              (gmst_table['scenario']=='historical'), 
                              :].copy()
        if tabf.shape[0] == 0:
            print('tas data for ' + mod + ' ' + mem + ' ' + scen + 
                  ' is not available on Google Cloud')
        if tabh.shape[0] == 0:
            print('tas data for ' + mod + ' ' + mem + 
                  ' historical is not available on Google Cloud')                                                                                          
        
        tab1 = gmst_table.loc[(gmst_table['model']==mod) & 
                              (gmst_table['member']==mem) & 
                              ((gmst_table['scenario']==scen) | 
                              (gmst_table['scenario']=='historical')), 
                              :].copy()
        
        tab1['warming_mean'] = tab1['warming'].rolling(window_size, 
                                                       center=True).mean()

        tab1 = tab1.loc[tab1['warming_mean'] > warming_level]
        
        if tab1.shape[0]>0:
            outdf.iloc[i,3] = tab1.iloc[0,3]
        elif (tab1.shape[0] == 0) & (tabf.shape[0]>0) & (tabh.shape[0]>0):
            print(mod + ' ' + mem + ' ' + scen + ' does not reach the specified warming level before \n   "max_year" or the end of the available data, whichever comes first.')
            print('   Consider increasing the value of "max_year" to see if there is additional data available.\n')
        else: # if future or historical data was simply unavailable
            outdf.iloc[i,3] = np.nan

    return outdf


def calc_warming_years_temperature_window(mms_table, warming_level, temp_tolerance, max_year):
    """Calculate warming years using a temperature window approach.
    
    This function calculates warming years for each model/member/scenario 
    combination in mms_table and the specified warming level as any year with 
    mean annual warming in the range of the warming level +/- the temperature 
    tolerance.

    Args: 
        mms_table (dataframe) : pandas dataframe containing one row for each 
        model, member, scenario combination that you wish to calculate warming 
        years for.
        
        warming_level (float) : global warming level in degrees C to calculate 
        warming years for.
        
        temp_tolerance (float) : temperature tolerance in degrees C around the 
        warming level that is used to identify warming years. For example, if 
        warming_level=2 and temp_tolerance=0.25, then years with warming 
        >1.75C and <2.25C will be selected
        
        max_year (int) : last year to include in calculations.

    Returns:
        A pandas dataframe that has the same columns as mms_table plus one 
        column for the warming_year. There is a row for each 
        model/member/scenario/warming_year combination. Model/member/scenario 
        combinations that do not reach the specified warming level are given 
        one row and assigned a value of NaN for the warming year.
        
    Example:
        mms = pd.DataFrame({'model': ['ACCESS-CM2']*3,
                            'member': ['r1i1p1f1']*3,
                            'scenario': ['ssp126','ssp245','ssp585']})
        warming_year_table = calc_warming_years_temperature_window(mms, 1, .1,
        2100)
 
    """
    scen = mms_table['scenario'].drop_duplicates()

    # import the gmst data
    gmst_table = pd.read_csv(gmst_table_dir + 'CMIP6_GMST_table_all.csv')

    # exclude years after max_year
    gmst_table = gmst_table.loc[gmst_table['year'] <= max_year]

    # calculate warming for each year relative to 1850-1900 base period
    gmst_table['warming'] = gmst_table['GMST'] - gmst_table['1850-1900']
    
    # create a new copy of mms_table so that the original is not modified
    mms = mms_table.copy()
    
    # add historical scenario to mms_table so that historical warming years get incorporated as well
    mmsh= mms.copy()
    mmsh['scenario'] = 'historical'
    mms= pd.concat([mms, mmsh])

    # select only model/member/scenario combinations that are in mms_table    
    gmst_table = pd.merge(gmst_table, mms, how='right', on=['model','member','scenario'])

    # identify mms combos that are not available 
    missing = gmst_table[gmst_table['year'].isnull()].drop_duplicates()
    for r in range(missing.shape[0]):
        print('tas data for ' + missing['model'].iloc[r] + ' ' + missing['member'].iloc[r] + ' ' + missing['scenario'].iloc[r] + ' is not available on Google Cloud')

    # filter to only years within temp_tolerance
    mn = warming_level - temp_tolerance
    mx = warming_level + temp_tolerance
    outdf = gmst_table.loc[(gmst_table['warming'] > mn) & (gmst_table['warming'] < mx)]
    
    # add a row with warming_year=NaN for any model/member/scenario combination that did not have any warming years (because unavailable or not reached)
    outdf = pd.merge(outdf, mms, how='outer', on = ['model','member','scenario'])
    
    # identify cases where warming level was not reached
    # first find the missing instances in the output table
    missout = outdf[(outdf['year'].isnull()) & 
                    (outdf['scenario'].isin(scen))].drop_duplicates()
    # then subtract the instances that didn't have data available to get the instances that didn't reach the warming level
    misswarm = pd.merge(missout,missing,how='outer', indicator=True)
    misswarm = misswarm[misswarm['_merge']=='left_only']
    for r in range(misswarm.shape[0]):
        print(misswarm['model'].iloc[r] + ' ' + misswarm['member'].iloc[r] + 
              ' ' + misswarm['scenario'].iloc[r] + 
              ' does not reach the specified warming level before \n   "max_year" or the end of the available data, whichever comes first.')
        print('   Consider increasing the value of "max_year" to see if there is additional data available.\n')

    # clean up the output dataframe
    outdf = outdf.drop(labels=['GMST','1850-1900','warming'], axis=1)
    outdf = outdf.rename(columns={'year':'warming_year'})
    
    return outdf
    

def get_cmip6_data(model, member, scenario, cmip6_variable, start_yr_mo, end_yr_mo, outfilename=None):
    """Download CMIP6 data from Google Cloud.
    
    This function downloads monthly CMIP6 data from Google Cloud for the 
    specified model, member, scenario, and variable for the period starting in 
    the year and month specified by start_yr_mo and ending in the year and 
    month specified by end_yr_mo. Longitude is shifted from [0 360] to [-180 
    180]. Includes the option of saving the data to file.
    
    Args:
        model (string) : name of the CMIP6 model to download data for (e.g. 
        'ACCESS-CM2')
        
        member (string) : ensemble member (aka realization) to download data 
        for (e.g. 'r1i1p1f1')
    
        scenario (string) : CMIP6 scenario to download data for (e.g. 
        'historical' or 'ssp585')
    
        cmip6_variable (string) : name of the CMIP6 variable to download data 
        for (e.g. 'tas' or 'pr')
        
        start_yr_mo (string) : the first year and month to grab data for. 
        Format should be 'YYYY-MM', e.g. '2012-01' for January 2012
        
        end_yr_mo (string) : the last year and month to grab data for. Format 
        should be 'YYYY-MM'.
        
        outfilename (string) : file name to save data to. Should end in '.nc'. 
        If outfilename=None, data will be returned but not saved to file
        
    Returns:
        An xarray dataset (and can save a netcdf file) of global (spatially 
        explicit) monthly CMIP6 data for the variable of interest for the 
        specified model, member, and scenario.
        
    Raises:
        ValueError: If scenario='historical' but start_yr_mo or end_yr_mo 
        refer to dates after 2014.
        
        ValueError: If an output file name is specified but it does not end in 
        '.nc'.
    
    Example:
        cmip6_data = get_cmip6_data('ACCESS-CM2', 'r1i1p1f1', 'SSP585', 'tas', 
        '2020-01', '2040-12', 'mydata.nc')

    """
    
    # check arguments
    member = member.lower()
    scenario = scenario.lower()
    
    if (scenario == 'historical') and ((int(start_yr_mo[0:4]) > 2014) or (int(end_yr_mo[0:4]) > 2014)):
        raise ValueError('When scenario="historical", start_yr_mo and end_yr_mo must reference dates before 2015')
    
    if (outfilename is not None) and (outfilename[-3:] != '.nc'):
        raise ValueError('If you wish to save the file, please specify an outfilename ending in ".nc" (i.e. a netcdf file)')           
    
    # Open the CMIP6 zarr data catatalog 
    df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')

    # import cmip6 data
    if (scenario != 'historical') and (int(start_yr_mo[0:4]) < 2015) and (int(end_yr_mo[0:4]) < 2015):
        print('Warning: due to the time window selected, all data will be imported from historical runs, not the specified scenario')
        df_zarr = df.query("table_id == 'Amon' and variable_id == '" + cmip6_variable + "' and experiment_id == 'historical' and source_id == '" + model + "' and member_id == '" + member + "'")
        zstore = df_zarr.zstore.values[0]
        mapper = fsspec.get_mapper(zstore)            
        zz = xr.open_zarr(mapper, consolidated=True)
    elif (scenario != 'historical') and (int(start_yr_mo[0:4]) < 2015):
        print('Warning: due to the time window selected, some data will be imported from historical runs, not the specified scenario')
        df_zarr = df.query("table_id == 'Amon' and variable_id == '" + cmip6_variable + "' and experiment_id == 'historical' and source_id == '" + model + "' and member_id == '" + member + "'")
        zstore = df_zarr.zstore.values[0]
        mapper = fsspec.get_mapper(zstore)            
        zh = xr.open_zarr(mapper, consolidated=True)
        
        df_zarr = df.query("table_id == 'Amon' and variable_id == '" + cmip6_variable + "' and experiment_id == '" + scenario + "' and source_id == '" + model + "' and member_id == '" + member + "'")
        zstore = df_zarr.zstore.values[0]
        mapper = fsspec.get_mapper(zstore)            
        zz = xr.open_zarr(mapper, consolidated=True)

        zz = xr.concat([zh,zz], dim='time')
        zz['time'] = zz.time.astype("datetime64[ns]")
    else:
        df_zarr = df.query("table_id == 'Amon' and variable_id == '" + cmip6_variable + "' and experiment_id == '" + scenario + "' and source_id == '" + model + "' and member_id == '" + member + "'")
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
    if outfilename is not None:
        print('writing file to ' + outfilename)
        zz.to_netcdf(outfilename)
    
    return zz



def get_cmip6_data_at_warming_years(model, member, scenario, cmip6_variable, warming_years, year_window=0, out_temporal_res='annual', outfilename=None):
    """Download CMIP6 data from Google Cloud for specific years.
    
    This function downloads monthly CMIP6 data from Google Cloud for the 
    specified model, member, scenario, variable, and years. The years 
    downloaded are specified by warming_years and can include a window of 
    years either side of the warming years. Data is aggregated to annual 
    values. Longitude is shifted from [0 360] to [-180 180]. Includes the 
    option of saving the data to file. This function only supports variables 
    with units of K (e.g. temperature variables) or kg m-2 s-1 (e.g. 
    precipitation) at this time.

    Args:
        model (string) : name of the CMIP6 model to download data for (e.g. 
        'ACCESS-CM2')
    
        member (string) : ensemble member (aka realization) to downlaod data 
        for (e.g. 'r1i1p1f1')
        
        scenario (string) : CMIP6 scenario to download data for (e.g. 
        'historical' or 'ssp585')
    
        cmip6_variable (string) : name of the CMIP6 variable to download data 
        for (e.g. 'tas' or 'pr')
        
        warming_years (array) : numpy array of years to download data for. 
        Years do not need to be consecutive. (e.g. [2010,2014,2020,2021])
        
        year_window (int) : integer specifying additional years to include 
        either side of each specified warming years. For example, if 
        warming_years = [2010,2017] and year_window=0, then only data for 
        years 2010 and 2017 will be included. If warming_years = [2010,2017] 
        and year_window=2, then data within +/- 2 years of the warming_years 
        will be included (i.e 
        [2008,2009,2010,2011,2012,2015,2016,2017,2018,2019])
        
        outfilename (string) : file name to save data to. If outfilename=None, 
        data will be returned but not saved to file. Otherwise it should end 
        in '.nc'. 
        
    Returns:
        An xarray dataset (and can save a netcdf file) of global (spatially 
        explicit) warming years (annual) timeseries of the CMIP6 variable for 
        the specified model, member, and scenario.
        
    Raises:
        ValueError: If an output file name is specified but it does not end in 
        '.nc'.
        
        ValueError: If data for the specified 
        model/member/scenario/variable/years combination is not available from 
        Google Cloud.
        
        ValueError: If the requested variable does not have units of K or kg 
        m-2 s-1.

    Example:
        warming_year_data = get_cmip6_data_at_warming_years(model=
        'ACCESS-CM2', member='r1i1p1f1', scenario='ssp585', 
        cmip6_variable='tas', warming_years=np.array([2010,2021,2024]),  
        year_window=5, outfilename='mydata.nc')

    """

    # check arguments
    member = member.lower()
    scenario = scenario.lower()
    
    if out_temporal_res not in (['monthly','annual']):
        raise ValueError('out_temporal_res must be "monthly" or "annual"')
    
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
        df_zarr = df.query("table_id == 'Amon' and variable_id == '" + cmip6_variable + "' and experiment_id == 'historical' and source_id == '" + model + "' and member_id == '" + member + "'")
        
        if df_zarr.shape[0]==0:
            raise ValueError(cmip6_variable + ' data is not available from Google Cloud for ' + model + ', ' + member + ', for the historical period.')
            
        zstore = df_zarr.zstore.values[0]
        mapper = fsspec.get_mapper(zstore) 
        zh = xr.open_zarr(mapper, consolidated=True)
        zh = zh.where(zh.time.dt.year.isin(yrsh), drop=True)
    if len(yrsf)>0: # if there are any future years
        df_zarr = df.query("table_id == 'Amon' and variable_id == '" + cmip6_variable + "' and experiment_id == '" + scenario + "' and source_id == '" + model + "' and member_id == '" + member + "'")
        
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
    # resample to month start to avoid issues with 360 day calendars
    zz = zz.resample(time="MS").mean()
    if zz.time.dtype != '<M8[ns]':
        zz['time'] = zz.indexes['time'].to_datetimeindex(unsafe = True)

    if cmip6_variable == 'pr':
        # convert pr from kg m-2 s-1 to mm month-1
        sec_in_month = zz.time.dt.days_in_month * 24 * 60 * 60
        zz[cmip6_variable] = zz[cmip6_variable] * sec_in_month
        zz[cmip6_variable].attrs = {'units':'mm'}

    # aggregate monthly data to annual timeseries
    if out_temporal_res == 'annual':
        if cmip6_variable in ['tas','tasmax','tasmin']:
            # weight months by days in month
            month_length = zz.time.dt.days_in_month
            weights = (month_length.groupby("time.year") / month_length.groupby("time.year").sum())
            # Calculate the time-weighted average for each year
            zz[cmip6_variable] = (zz[cmip6_variable] * weights).groupby("time.year").sum(dim="time",min_count=1)
            # include min_count above so that years where all months are nan become nan, not 0
            zz[cmip6_variable].attrs = {'units':'K'}
        elif cmip6_variable == 'pr':
            # sum precipitation across months to get annual precipitation
            zz[cmip6_variable] = zz[cmip6_variable].groupby("time.year").sum(dim="time",min_count=1)
        else:
            raise ValueError('This function does not know how to aggregate monthly data to annual for variables with units of ' + zz[cmip6_variable].units + '. Currently only units of K or kg m-2 s-1 are supported.')
        zz = zz.drop('time',errors='ignore')

    
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
    if outfilename is not None:
        print('writing file to ' + outfilename)
        zz.to_netcdf(outfilename)
    
    return zz
    
    

def spatial_mean(data_on_grid):
    """Calculate spatially weighted mean.
    
    This function calculates the area weighted spatial mean and is adapted
    from 
    https://docs.xarray.dev/en/stable/examples/area_weighted_temperature.html. 
    The input dataset should contain lon and lat dimensions.

    Args:
        data_on_grid (xarray dataset) : xarray dataset containing data on a 
        lon/lat grid to be averaged.
        
    Returns:
        An xarray dataset containing the same data arrays as the input 
        dataset, but spatially averaged.

    Example:
        global_mean_data = spatial_mean(my_xarray_dataset)

    """
    
    weights = np.cos(np.deg2rad(data_on_grid.lat))
    weights.name = "weights"
    data_on_grid_weighted = data_on_grid.weighted(weights)
    weighted_mean = data_on_grid_weighted.mean(("lon", "lat"))
    
    return weighted_mean


def get_cmip6_at_warming_level(mms, cmip6_variable, warming_level, outfn, out_temporal_res, max_year=2100, warming_year_type='moving_window', year_window=21, temp_window=0.5):
    """ Create a zarr file of CMIP6 data at a warming level.
    
    This function grabs CMIP6 data from Google Cloud for a set of 
    model/member/scenario combinations for a given warming level, interpolates 
    it to a common spatial grid, and combines it in a single xarray dataset or 
    set of zarr files with dimensions of time, model, member, scenario, lat, 
    and lon.
    
    Args:
        mms (pandas dataframe) : a pandas dataframe with three columns: 
        'model', 'member', and 'scenario' filled with CMIP6 model names, CMIP6 
        member names (also known as ensemble members or variant labels), and 
        CMIP6 scenarios.
    
        cmip6_variable (str) : name of a CMIP6 variable to grab data for (must 
        be 'tas', 'tasmin', 'tasmax', or 'pr').
    
        warming_level (float) : A warming level in °C, calculated as the 
        change in temperature since the pre-industrial period (1850-1900). 
        (e.g. 1.5)
    
        outfn (str) : full path of the directory where outputs should be 
        saved. If 'None', the outputs are returned but not saved.
    
        out_temporal_res (str) : Temporal resolution of the output data. 
        Current options are 'annual' or 'monthly'
    
        max_year (int) : Latest year to include in the output dataset (e.g. 
        2100)
    
        warming_year_type (str) : String indicating whether warming years 
        should be calculated using a temporal window corresponding to a number 
        of years (i.e. 'moving_window') or a temperature window that 
        identifies years within a specified temperature tolerance of the 
        warming level (i.e. 'temperature_window').

        year_window (int) : Integer indicating the span of years that should  
        be used with the 'moving_window' approach to identify warming levels 
        (e.g. 21)
    
        temp_window (float) : The number of degrees C that should be used as a 
        tolerance around the specified warming level to select years. (e.g. 
        for a warming level of 2 and a temp_window of 0.5, all years with 
        warming between 1.5 and 2.5 will be included)
        
    Returns:
        An xarray dataset containing CMIP6 data for the variable and warming 
        level of interest with dimensions time, model, member, scenario, lat, 
        and lon.
    
    Example:
        mms = pd.DataFrame({'model':['GFDL-CM4','FGOALS-g3'],
                           'member':['r1i1p1f1','r1i1p1f1'],
                           'scenario':['ssp245','ssp245']})
        cmip6_variable = 'tas' #'pr'
        warming_level = 2
        outfn = '/path/to/output/'
        out_temporal_res = 'monthly'
        max_year = 2100
        warming_year_type = 'moving_window'# 'temperature_window' # 
        year_window = 21
        temp_window = 0.5

        get_cmip6_at_warming_level(mms, cmip6_variable, warming_level, outfn, 
                                   out_temporal_res, max_year, 
                                   warming_year_type, year_window, 
                                   temp_window)
    """
            
   
    # create new common grid to interpolate to
    newlats = np.arange(-90, 90.01, .5)
    newlons = np.arange(-180, 180, .5)

    
    # to remove warnings about chunk sizes
    dask.config.set(**{'array.slicing.split_large_chunks': False})
    
            
    # create warming year table
    if warming_year_type == 'moving_window':
        tab = calc_warming_years_rolling_mean(mms, warming_level, year_window, max_year).dropna()
        # expand this so that it shows all years in the window
        df = pd.DataFrame()
        for r in range(tab.shape[0]):
            row = tab.iloc[[r]]
            row = row.reindex(row.index.repeat(year_window))
            centeryear = row.warming_year.iloc[0]
            if year_window % 2 == 0: # even window
                styear = centeryear-year_window/2
                enyear = centeryear+year_window/2 
            else: # odd window
                styear = centeryear-((year_window-1))/2
                enyear = centeryear+((year_window-1)/2)+1
            row.warming_year = np.arange(styear, enyear)
            df = pd.concat([df,row])
        tab = df.reset_index(drop=True)
    elif warming_year_type == 'temperature_window':
        tab = calc_warming_years_temperature_window(mms, warming_level, temp_window, max_year).dropna()


    if cmip6_variable == 'pr':
        # list of mms available for tas but not for pr
        nopr = pd.DataFrame({'model':['NorESM2-LM','ACCESS-ESM1-5'],
                             'member':['r1i1p1f1','r31i1p1f1'],
                             'scenario':['ssp585','ssp585']})

        #identify rows in mms that are in nopr
        checkdf = pd.merge(tab,nopr,how='left',indicator=True)
        badpr = checkdf[checkdf['_merge']=='both']
        for r in range(badpr.shape[0]):
            mod = badpr['model'].iloc[r]
            mem = badpr['member'].iloc[r]
            scen = badpr['scenario'].iloc[r]
            print('pr data for ' + mod + ' ' + mem + ' ' + scen + ' is not available on Google Cloud')
        # remove rows that aren't available
        tab = checkdf[checkdf['_merge']=='left_only'].drop('_merge',axis=1)
    
    
    if tab.shape[0] == 0:
        raise ValueError('None of the requested model/member/scenario combinations were available from Google Cloud or reached the specified warming level within the timespan of available data. Consider increasing the value of "max_year".')


    # make table to loop through
    iter_tab = tab[['model','member','scenario']].loc[tab['scenario']!='historical'].drop_duplicates()
        
    print('grabbing data...')
    i = 0
    for r in range(iter_tab.shape[0]):
        i = i+1
        mod = iter_tab['model'].iloc[r]
        mem = iter_tab['member'].iloc[r]
        scen = iter_tab['scenario'].iloc[r]
        tab0 = tab.loc[(tab['model']==mod) & 
                       (tab['member']==mem) & 
                       (tab['scenario'].isin([scen,'historical']))]
        warming_years = np.unique(tab0.warming_year.values)

        dat = get_cmip6_data_at_warming_years(tab0.model.iloc[0], 
                                              tab0.member.iloc[0], 
                                              tab0.scenario.iloc[0], 
                                              cmip6_variable, 
                                              warming_years, 
                                              year_window=0, 
                                        out_temporal_res=out_temporal_res,
                                        outfilename=None)

        # interpolate to common spatial grid
        dat = dat.interp(lon=newlons, lat=newlats, method="linear")

        # condense model and member into one dimension to save space
        #dat = dat.stack(model_member = ['model','member'])

        # remove lat and lon bounds since not all datasets have these 
        # variables, and we changed the lat and lon anyway
        dat = dat.drop(['lat_bnds','lon_bnds','bnds'],errors='ignore')
        if i==1:
            xout = dat
        else:
            xout = xr.merge([xout,dat])
            #xout = xr.concat([xout, dat], dim='model_member')
            #xout = xr.concat([xout, dat], dim='new').drop_dims('new')

            
    if out_temporal_res == 'annual':
        # convert years (int64) to dates (datetime64)
        xout['year'] = pd.to_datetime(xout['year'],format="%Y")
        xout = xout.rename({'year':'time'})

    if outfn != None:
        # save file
        print('saving output to: ' + outfn)

        #xout = encode_multiindex(xout,'model_member').chunk(chunks='auto')
        xout = xout.chunk(chunks='auto')
        
        
        import zarr
        compressor = zarr.Blosc(cname="zlib")

        xout.to_zarr(outfn, 
                     group=cmip6_variable,
                     encoding={cmip6_variable: {"compressor": compressor}})
#                     encoding = {cmip6_variable:{"zlib":True}})
        #xout.to_netcdf(outfn, encoding = {cmip6_variable:{"zlib":True}})
  
    # add a hint about how to work with the output dataset
    print("\nTo make working with this dataset easier, create a multiindex from model and member and then drop the combos that are not in the dataset:\n ds.stack(model_member = ['model','member']).dropna('model_member',how='all')")
    
    print("\nTo import the saved zarr file:\n import xarray as xr\n ds = xr.open_zarr(outfn, group=cmip6_variable)")
    
    print("\nDone")
    
    return xout


#def encode_multiindex(ds, idxname):
#    encoded = ds.reset_index(idxname)
#    coords = dict(zip(ds.indexes[idxname].names, ds.indexes[idxname].levels))
#    for coord in coords:
#        encoded[coord] = coords[coord].values
#    shape = [encoded.sizes[coord] for coord in coords]
#    encoded[idxname] = np.ravel_multi_index(ds.indexes[idxname].codes, shape)
#    encoded[idxname].attrs["compress"] = " ".join(ds.indexes[idxname].names)
#    return encoded