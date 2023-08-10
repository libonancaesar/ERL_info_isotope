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
            plt.savefig(f"../Figures/processedDataFigures/{siteName}{timewindow}-time-series.pdf")
        elif timewindow == "alltime": ## save the alltime iso figures to "processedDataFigures24hiso" folder 
            plt.savefig(f"../Figures/processedDataFigures24iso/{siteName}{timewindow}-time-series.pdf")
        else:
            raise ValueError(f"Provided {timewindow} string is wrong.")

            

def GetPairedPlots(df, name, timewindow, saveFigure = True):
    sns.set(font_scale=2)
    dataCols = list(df.columns)[1:]
    g = sns.PairGrid(df[dataCols])
    g.map_diag(sns.kdeplot)
    g.map_lower(sns.regplot)
    g.map_lower(reg_coef)
    g.fig.suptitle(name, x = 0.5, y = 1, fontsize = 30)
    if saveFigure:
        if timewindow == "daytime": ## save the daytime iso figures to "processedDataFigures" folder 
            g.fig.savefig(f"../Figures/processedDataFigures/{name}{timewindow}-paire-plots.pdf")
        elif timewindow == "alltime": ## save the alltime iso figures to "processedDataFigures24hiso" folder 
            g.fig.savefig(f"../Figures/processedDataFigures24iso/{name}{timewindow}-paire-plots.pdf")
        else:
            raise ValueError(f"Provided {timewindow} string is wrong.")
            
    
def reg_coef(x,y,label=None,color=None,**kwargs):
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
    '''        df: dataframe contains variables
       listOfVars: list of variable to process in df
    '''
    ### This operation only works on cpy_data the original inputs were preserved 
    cpy_data = copy.deepcopy(df) 
    for k in listOfVars:
        ###find the interquantiles
        p75, p25 = np.nanpercentile(cpy_data[k], [75, 25], axis=0) 
        IQR = p75 - p25
        ###only keep the data within
        ###Q1 - 1.5IQR <= values <= Q3 + 1.5*IQR
        
        ### this is a view of cpy_data  
        cpy_data.loc[(cpy_data[k] > p75 + IQR*1.5) |
                     (cpy_data[k] < p25 - IQR*1.5),k] = np.nan   
        print("{} filter done ....!!!".format(k))
    return cpy_data


def FilterIsotopesWithQualityFlags(isotope, whatDataSet):     
    allIsotopePath = "../Isotope data"
   ### make some assurance 
    if isotope in ["d13C"]:
        flux = "nee"
    elif isotope in ["d2H", "d18O"]:
        flux = "et"
    else:
        raise ValueError(f"Provided {isotope} string is wrong.")
            
    if whatDataSet not in ["daytime", "alltime"]:
        raise ValueError("whatDateSet argument only accept 'daytime' or 'alltime'. ")
        
    
    isotopePath = os.path.join(allIsotopePath, f"daily_{flux}_flux_{isotope}_{whatDataSet}.csv")
    isotopeFlagPath = os.path.join(allIsotopePath, f"daily_{flux}_flux_{isotope}_{whatDataSet}_metadata.csv")

    isotopeDataHub = pd.read_csv(isotopePath) 
    isotopeFlagDataHub = pd.read_csv(isotopeFlagPath)

    isotopeDataHub.set_index("date", inplace = True)
    isotopeFlagDataHub.set_index("date", inplace = True)
    
    flaggedData = isotopeDataHub[isotopeFlagDataHub == 0] ##only use the data that has recommended data quality
    flaggedData.reset_index(inplace = True)
    flaggedData['date'] = pd.to_datetime(flaggedData['date']).dt.strftime('%Y-%m-%d')
    return flaggedData


def FilterOtherVariables(fileName, timewindow, timeSeriesPlot = False, pairedPlots = False, saveFigure = False, saveData = False): 
    siteName = fileName.split("-")[1]
    colList = ['Date Time', 'NEE_U50_f', 'LE_U50_f', 'VPD_f', 'Tair_f', 'wspd', 'Rg_f']
    data = pd.read_csv(f"../Flux and other data/{fileName}", 
                 delimiter = '\t', low_memory = False, usecols=colList, skiprows=[1])
    filterDataDefault = data[data!= -9999]
    filterDataDefault['Date Time'] = pd.to_datetime(filterDataDefault['Date Time'], format = '%Y-%m-%d %H:%M:%S')
    filterDataDefault['date'] = filterDataDefault['Date Time'].dt.strftime('%Y-%m-%d')
    filterDataDefault.drop(columns= ['Date Time'], inplace = True)
    ##Now take the average of all the variables we need 
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

    fullData =  reduce(lambda x, y: pd.merge(x, y, on = 'date', how = 'outer'), 
                                [processedOtherVar, d2H, d18O, d13C]) 
    fullData.sort_values(by = 'date', inplace = True, ignore_index = True)
    if timeSeriesPlot:
        GetTimeSeriesPlot(fullData, siteName, timewindow,saveFigure = saveFigure)     
    if pairedPlots:
        GetPairedPlots(fullData, siteName, timewindow,saveFigure = saveFigure)
    if saveData:
        if timewindow == "daytime":
            fullData.to_csv(f'../Processed data/{siteName}_data_{timewindow}.csv', index = False)
        elif timewindow == "alltime":
            fullData.to_csv(f'../Processed data 24hiso/{siteName}_data_{timewindow}.csv', index = False)
        else:
            raise ValueError(f"Provided {timewindow} string is wrong.")
    return fullData

if __name__ == '__main__':
    fileList = os.listdir('../Flux and other data')
    for i in fileList:
        dd1 = FilterOtherVariables(i, "daytime", timeSeriesPlot = True, 
                                   pairedPlots = True, saveFigure= True, saveData=True)
        dd2 = FilterOtherVariables(i, "alltime", timeSeriesPlot = True, 
                                   pairedPlots = True, saveFigure= True, saveData=True)
    