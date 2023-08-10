# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:48:41 2022

@author: bonanl
"""

import os 
import pandas as pd 
import numpy as np
import copy 
import matplotlib.pyplot as plt
import seaborn as sns
# from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr
from functools import reduce

    
def GetTimeSeriesPlot(pdSeries, siteName, timewindow, saveFigure = True):
    """Generate time series plots for isotope and micromet datasets.

    Parameters
    ----------
    pdSeries : pandas.DataFrame
        The dataframe contains micromet data.
    siteName : string
        The 4-letter abbreviation of the NEON site i.e., ABBY for Abby road, WA.
    timewindow : string
        Isotope data product time window.
    saveFigure : bool, optional
        If the rendered time series figure will be save. 
        The default is True.

    Raises
    ------
    ValueError
        Error will be raise if the provided window is not correct.
        Currently, this function does support night time data. 
        However, can be easily modifed and adapted for night time.

    Returns
    -------
    None.

    """
    data = copy.deepcopy(pdSeries)
    plt.figure()
    data.set_index('date', inplace = True)
    figureSize = (50, 50)
    data.plot(subplots = True, figsize = figureSize, fontsize = 50)
    plt.gcf().suptitle(siteName,x = .5, y =.9, fontsize=50)
    plt.gca().set_xlabel("date", fontsize = 50)
    [ax.legend(loc=2,prop={'size': 40}) for ax in plt.gcf().axes] 
    if saveFigure:
        if timewindow == "daytime": ## save the daytime iso figures to "processedDataFigures" folder 
            # plt.savefig(f"../Figures/processedDataFigures/{siteName}{timewindow}-time-series.pdf")
            # plt.savefig(f"../daytime_figures/{siteName}-{timewindow}-time-series.pdf")
            pass
        elif timewindow == "alltime": ## save the alltime iso figures to "processedDataFigures24hiso" folder 
            # plt.savefig(f"../Figures/processedDataFigures24iso/{siteName}{timewindow}-time-series.pdf")
            # plt.savefig(f"../alltime_figures/{siteName}-{timewindow}-time-series.pdf")
            plt.savefig(f"../alltime_time_correct_figures/{siteName}-{timewindow}-time-series-time-corrected.pdf")

        else:
            raise ValueError(f"Provided {timewindow} string is wrong.")

            

def GetPairedPlots(df, name, timewindow, saveFigure = True):
    """Generate paired plots plots for isotope and micromet datasets.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    name : TYPE
        DESCRIPTION.
    timewindow : TYPE
        DESCRIPTION.
    saveFigure : TYPE, optional
        DESCRIPTION. The default is True.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    sns.set(font_scale=2)
    dataCols = list(df.columns)[1:]
    g = sns.PairGrid(df[dataCols])
    g.map_diag(sns.kdeplot)
    g.map_lower(sns.regplot)
    g.map_lower(reg_coef)
    g.fig.suptitle(name, x = 0.5, y = 1, fontsize = 30)
    if saveFigure:
        if timewindow == "daytime": ## save the daytime iso figures to "processedDataFigures" folder 
            # g.fig.savefig(f"../Figures/processedDataFigures/{name}{timewindow}-paire-plots.pdf")
            # g.fig.savefig(f"../daytime_figures/{name}-{timewindow}-paire-plots.pdf")
            pass
        elif timewindow == "alltime": ## save the alltime iso figures to "processedDataFigures24hiso" folder 
            # g.fig.savefig(f"../Figures/processedDataFigures24iso/{name}{timewindow}-paire-plots.pdf")
            g.fig.savefig(f"../alltime_time_correct_figures/{name}-{timewindow}-paire-plot-time-corrected.pdf")

        else:
            raise ValueError(f"Provided {timewindow} string is wrong.")
            
    
def reg_coef(x,y, label = None, color = None, **kwargs):
    """Generate cross correlation between two variables.

    Parameters
    ----------
    x : pandas.Series
        data x.
    y : pandas.Series
        data y.
    label : TYPE, optional
        DESCRIPTION. The default is None.
    color : TYPE, optional
        DESCRIPTION. The default is None.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    interimDf = pd.concat([copy.deepcopy(x), copy.deepcopy(y)], axis=1)
    interimDf.columns = ['x', 'y']
    interimDf = interimDf.dropna()
    if interimDf.empty or interimDf.shape[0] <= 2:
        r = np.nan
    else:
        r,p = pearsonr(interimDf['x'], interimDf['y'])
    
    ax = plt.gca()
    ax.annotate(r'$\rho$ = {:.2f}'.format(r), xy=(0.75,0.9), xycoords='axes fraction', ha='center',
                fontsize = 20)
    
def InterQuantileFilter(df, listOfVars):
    """Filter the data that is stored in a dataframe using the inter-quantile filter.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe that needs to be filtered.
    listOfVars : list
        List of the column names of df to be filtered.
    showFilterFigure: boolean
        whether or not to show the time series before and after inter-quantile filter.

    Returns
    -------
    cpy_data : pandas.DataFrame
        The dataframe after applying the inter-quantile filter.
    """ 
    ### This operation only works on cpy_data to avoid unexpected changes on the original data 
    cpy_data = copy.deepcopy(df) 
    for k in listOfVars:
        ###find the interquantiles
        p75, p25 = np.nanpercentile(cpy_data[k], [75, 25], axis=0) 
        IQR = p75 - p25
        ###only keep the data within
        ###Q1 [25th percentile] - 1.5IQR <= values <= Q3 [75th percentile] + 1.5*IQR
        ### this is a view of cpy_data  
        cpy_data.loc[(cpy_data[k] > p75 + IQR*1.5) |
                     (cpy_data[k] < p25 - IQR*1.5),k] = np.nan 
        print(f"IQR filter on {k} is done.")
    return cpy_data


def FilterIsotopesWithQualityFlags(isotope, whatDataSet):  
    """Read in specific datasets and is then filtered with its quality flag.

    Parameters
    ----------
    isotope : str
        A string indicates isotope types d13C, d2H d18O.
    whatDataSet : str
        The isotope dataset name that the user wants to use.

    Raises
    ------
    ValueError
        If the user provided isotope name is not find, then ValueError will be raised.
        If the user provided time window name is not find, then ValueError will be raised.

    Returns
    -------
    flaggedData : pandas.DataFrame
        The isotope data .
    """
  
    allIsotopePath = "../Isotope data"
   ### make some assurance 
    if isotope in ["d13C"]:
        flux = "nee"
    elif isotope in ["d2H", "d18O"]:
        flux = "et"
    else:
        raise ValueError(f"Provided {isotope} string is wrong.")
            
    if whatDataSet not in ["daytime", "alltime"]:
        raise ValueError("whatDateSet argument only accept 'daytime' or 'alltime'.")
        
    ##'daily_et_flux_xxx_yyy.csv', where 'xxx' is the stable isotope and 'yyy' is the time-window
    ##'daily_et_flux_xxx_yyy_metadata.csv', where 'xxx' is the stable isotope and 'yyy' is the time-window
    isotopePath = os.path.join(allIsotopePath, f"daily_{flux}_flux_{isotope}_{whatDataSet}.csv")
    isotopeFlagPath = os.path.join(allIsotopePath, f"daily_{flux}_flux_{isotope}_{whatDataSet}_metadata.csv")
    

    isotopeDataHub = pd.read_csv(isotopePath) 
    isotopeFlagDataHub = pd.read_csv(isotopeFlagPath)
        
    isotopeDataHub.set_index("date", inplace = True)
    isotopeFlagDataHub.set_index("date", inplace = True)
    
    ##only use the data that has recommended data quality  i.e., flag = 0 
    ##rsq >=0.9 & npts >= 5 & within p25 - 1.5*IQR and p75 + 1.5*IQR
    ##refer to https://www.hydroshare.org/resource/e74edc35d45441579d51286ea01b519f/
    
    flaggedData = isotopeDataHub[isotopeFlagDataHub == 0] 
    flaggedData.reset_index(inplace = True)
    ##converted to datetime to string format
    flaggedData['date'] = pd.to_datetime(flaggedData['date']).dt.strftime('%Y-%m-%d')

    return flaggedData


def FilterOtherVariables(fileName, timewindow, timeSeriesPlot = False, pairedPlots = False, saveFigure = False, saveData = False):
    """Merge filtered micromet dataset with quality checked isotope datasets.
    
    Parameters
    ----------
    fileName : string
        A string that is the file name of the .txt file for the micromet data files.
    timewindow : string
        Isotope dataset calibration time window i.e., alltime | daytime | nighttime.
    timeSeriesPlot : bool, optional
        If the paired plots will be generated refer to the "GetTimeSeriesPlot" function.
        The default is False.
    pairedPlots : bool, optional
        If the scatter plots will be generated refer to "GetPairedPlots" function.
        The default is False.
    saveFigure : bool, optional
        If the generated figure will be saved somewhere.
        The default is False.
    saveData : bool, optional
        If the filtered data will be saved somewhere.
        The default is False.

    Raises
    ------
    ValueError
        If false timewindow is provided.Current does not support night-time data.

    Returns
    -------
    fullData : pandas.DataFrame
        DataFrame contains filter micromet and isotope datasets at daily scale.
    """
    # Flux and other variable dataset is at local, need this to correct to UTC
    timeCorrector= pd.read_csv("E:/NEON_iso_MI/Flux and other data verifty/NEON_time_correction.csv")
    
    siteName = fileName.split("-")[1]
    
    timeOffValue = timeCorrector.loc[timeCorrector['Site'] == siteName, "Time_off_UTC"].values[0]
    
    colList = ['Date Time', 'NEE_U50_f', 'LE_U50_f', 'VPD_f', 'Tair_f', 'wspd', 'Rg_f']
    data = pd.read_csv(f"../Flux and other data/{fileName}", 
                 delimiter = '\t', low_memory = False, usecols=colList, skiprows=[1])
    filterDataDefault = data[data!= -9999]
    filterDataDefault['Date Time'] = pd.to_datetime(filterDataDefault['Date Time'], format = '%Y-%m-%d %H:%M:%S')
    # adjust the time to UTC as the time in the .txt files are appears to be local 
    # UTC label = time lag + timelable of txt files
    filterDataDefault['Date Time'] = filterDataDefault['Date Time'] + pd.Timedelta(timeOffValue, unit = "hours") # make sure specify "unit" instead of "units" 
    ##generate a string to aggregate half-hourly scale data to hourly to daily scale
    filterDataDefault['date'] = filterDataDefault['Date Time'].dt.strftime('%Y-%m-%d')
    filterDataDefault.drop(columns= ['Date Time'], inplace = True)
    ##now take the average of all the variables we need 
    print(f"working on site {siteName}...")
    dailyMeanFluxes = filterDataDefault.groupby('date', as_index=False).mean()
    processedOtherVar = InterQuantileFilter(dailyMeanFluxes, colList[1:])
    ##Read in the isotope datasets 
    d18O = copy.deepcopy(FilterIsotopesWithQualityFlags('d18O', timewindow)[['date', siteName]])
    d2H =  copy.deepcopy(FilterIsotopesWithQualityFlags('d2H', timewindow)[['date', siteName]])
    d13C =  copy.deepcopy(FilterIsotopesWithQualityFlags('d13C',timewindow)[['date', siteName]])
    ##Rename the isotope columns    
    d18O.rename(columns = {siteName: 'd18O'}, inplace = True)
    d2H.rename(columns = {siteName: 'd2H'}, inplace = True)
    d13C.rename(columns = {siteName: 'd13C'}, inplace = True)
    
    
    ###This is merging with strings
    fullData =  reduce(lambda x, y: pd.merge(x, y, on = 'date', how = 'outer'), 
                                [processedOtherVar, d2H, d18O, d13C]) 
    # fullData.sort_values(by = 'date', inplace = True, ignore_index = True)
    fullData['date'] = pd.to_datetime(fullData['date'], format = '%Y-%m-%d')
    fullData.sort_values(by = 'date', inplace = True, ignore_index = True)

    if timeSeriesPlot:
        GetTimeSeriesPlot(fullData, siteName, timewindow,saveFigure = saveFigure)     
    if pairedPlots:
        GetPairedPlots(fullData, siteName, timewindow,saveFigure = saveFigure)
    if saveData:
        if timewindow == "daytime":
            ##changed to a new test folder
            # fullData.to_csv(f'../Processed data/{siteName}_data_{timewindow}.csv', index = False)
            # fullData.to_csv(f'../daytime_data/{siteName}_data_{timewindow}.csv', index = False)
            pass
        elif timewindow == "alltime":
            ##same as comment above
            # fullData.to_csv(f'../Processed data 24hiso/{siteName}_data_{timewindow}.csv', index = False)
            fullData.to_csv(f'../alltime_time_correct_data/{siteName}_data_{timewindow}_correct_time.csv', index = False)
        else:
            raise ValueError(f"Provided {timewindow} string is wrong.")
    print(f"finshed working on site {siteName}...\n")

    return fullData

if __name__ == '__main__':
    fileList = os.listdir('../Flux and other data')
    for i in fileList:
        # dd1 = FilterOtherVariables(i, "daytime", timeSeriesPlot = True, 
        #                            pairedPlots = True, saveFigure= True, saveData=True)
        # dd2 = FilterOtherVariables(i, "alltime", timeSeriesPlot = True, 
                                   # pairedPlots = True, saveFigure= True, saveData=True)
        # dd1 = FilterOtherVariables(i, "daytime", timeSeriesPlot = True, 
        #                             pairedPlots = True, saveFigure= True, saveData=True)
        dd2 = FilterOtherVariables(i, "alltime", timeSeriesPlot = True, 
                                    pairedPlots = True, saveFigure= True, saveData=True)
        
        
        
        
        
        