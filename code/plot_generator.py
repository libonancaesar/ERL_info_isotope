# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 13:15:38 2021

@author: libon
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.api as sm
import copy
from functools import reduce
from scipy import stats
from tabulate import tabulate
import matplotlib as mb
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.preprocessing import StandardScaler, MinMaxScaler



def getFigPath(timeWindow): 
    if timeWindow == "alltime":
        resultsFigurePath = "../Figures/paperFigures24hiso/"
    elif timeWindow == "daytime":
        resultsFigurePath = "../Figures/paperFigures/"
    else:
        raise ValueError("only 'daytime' and 'alltime' options are avaiable for timewindow arg.")
    return resultsFigurePath 

def shuffleTest(dt, timewindow):
    ##dt: this should be the dataframe myData
    import os
    # shufflePath = 'C:/Users/libon/Box/neon_extrac_data/results/shuffled_unnorm_2021_10_13/'
    shufflePath = f'../ShuffleMIs/{timewindow}/'

    dt = copy.deepcopy(dt)
    allunshf = pd.concat([dt for i in np.arange(50)])
    allshuffle = []
    for i in os.listdir(shufflePath):
        tmef =  pd.read_csv(shufflePath + i)
        assert all(dt.columns == tmef.columns)
        assert all(dt['site'] == tmef['site'])
        allshuffle.append(tmef)
    allshuffle =  pd.concat(allshuffle) 

    assert all(allshuffle.columns == allunshf.columns)
    assert all(allshuffle['site'] == allunshf['site'])
    testArray = pd.DataFrame([], columns = allshuffle.columns[1:], index = ['pvalues', 'tstats'])
    for i in allshuffle.columns[1:]:
        dtt = pd.merge(allunshf[['site',i]],allshuffle[['site',i]], on = 'site').dropna()
        t_test = stats.ttest_rel(dtt[i + '_x'],dtt[i + '_y'], alternative = 'greater')
        testArray.loc['pvalues', i] = t_test[1] 
        testArray.loc['tstats', i] = t_test[0]
    return testArray, dt.drop(columns= 'site')

def getFigure1(ds, timewindow, saveFig = False):
    plt.style.use('seaborn')
    fig1, ax1 = plt.subplots(2, 1, figsize = (25,16))
    sig, data = shuffleTest(ds, timewindow)
    clss = ['gold', 'darkcyan', 'olive', 'hotpink', 'fuchsia',
            'gold', 'darkcyan', 'olive', 'hotpink', 'royalblue',
            'gold', 'darkcyan', 'olive', 'hotpink', 'dimgray']

  
    labelNEE = [
                '$I$($NEE$;$VPD$)', '$I$($NEE$;$T$)',
                '$I$($NEE$;$R_{g}$)','$I$($NEE$;$u$)',
                '$I$($NEE$;$\delta ^{13}C)$',
              
                '$I$($NEE$;$VPD$)', '$I$($NEE$;$T$)',
                '$I$($NEE$;$R_{g}$)', '$I$($NEE$;$u$)',
                '$I$($NEE$;$\delta ^{2}H$)',
              
                '$I$($NEE$;$VPD$)', '$I$($NEE$;$T$)',
                '$I$($NEE$;$R_{g}$)','$I$($NEE$;$u$)',
                '$I$($NEE$;$d$)'
                ]
    
    labelLH = [
                '$I$($LH$;$VPD$)', '$I$($LH$;$T$)',
                '$I$($LH$;$R_{g}$)','$I$($LH$;$u$)',
                '$I$($LH$;$\delta ^{13}C$)',
              
                '$I$($LH$;$VPD$)', '$I$($LH$;$T$)',
                '$I$($LH$;$R_{g}$)','$I$($LH$;$u$)',
                '$I$($LH$;$\delta ^{2}H$)',
              
                '$I$($LH$;$VPD$)', '$I$($LH$;$T$)',
                '$I$($LH$;$R_{g}$)', '$I$($LH$;$u$)',
                '$I$($LH$;$d$)'
                ]
     
    fieldNEE = ['I(NEE;VPD)13', 'I(NEE;T)13', 
                'I(NEE;R)13', 'I(NEE;u)13','I(NEE;C13)',
                
                'I(NEE;VPD)2', 'I(NEE;T)2', 
                'I(NEE;R)2', 'I(NEE;u)2','I(NEE;H2)',
                
                'I(NEE;VPD)dx', 'I(NEE;T)dx', 
                'I(NEE;R)dx', 'I(NEE;u)dx','I(NEE;dEx)']
                        
    fieldLH = ['I(LH;VPD)13', 'I(LH;T)13', 
                'I(LH;R)13', 'I(LH;u)13','I(LH;C13)',
                
                'I(LH;VPD)2', 'I(LH;T)2', 
                'I(LH;R)2', 'I(LH;u)2','I(LH;H2)',
                
                'I(LH;VPD)dx', 'I(LH;T)dx', 
                'I(LH;R)dx', 'I(LH;u)dx','I(LH;dEx)']   
    
    NEEdf = data[fieldNEE]
    LHdf = data[fieldLH]
    for s in np.arange(len(fieldNEE)):
        commonDirc = {'boxprops': dict(linewidth = 5, color = clss[s]),
                       'whiskerprops':dict(linewidth =5,linestyle = 'solid'),
                       'medianprops':dict(linewidth = 5, zorder = 9, color= 'white'),
                       'capprops': dict(linewidth =5),
                       'meanprops':dict(marker = '^',
                                   markersize = 15, zorder = 10, 
                                   markeredgecolor='black',
                                   markerfacecolor='black'),
                       'flierprops':dict(marker='o',
                                         markerfacecolor=clss[s],
                                         markersize=12,
                                         linestyle='none',
                                         markeredgecolor=clss[s])
                       }
        
        
        b1 = ax1[0].boxplot(NEEdf.loc[:,fieldNEE[s]].dropna(), widths = 0.45,
                   notch = False, labels = [labelNEE[s]],
                   showmeans = True, positions = [s],patch_artist=True,
                   showfliers = True,**commonDirc)
        b1['boxes'][0].set(facecolor = clss[s])
        b1['caps'][0].set(color=clss[s])   
        b1['caps'][1].set(color=clss[s])     
        b1['whiskers'][0].set(color=clss[s])
        b1['whiskers'][1].set(color=clss[s])
        [ax1[1].axvline(4.5 + i*5,color='white', linestyle = 'dashed',
                        linewidth = 3) for i in range(2)]
        p_value_nee = sig.loc['pvalues',fieldNEE[s]]
        if p_value_nee <= 0.01:
            ax1[0].annotate('**', xy=(s-0.1,1.2), fontsize=25)
        elif 0.01 < p_value_nee <= 0.05:
            ax1[0].annotate('*', xy=(s-0.1,1.2), fontsize=25)


        b2 = ax1[1].boxplot(LHdf.loc[:,fieldLH[s]].dropna(), widths = 0.45,
                    notch = False, labels = [labelLH[s]],patch_artist=True,
                    showmeans = True, positions = [s],
                    showfliers = True,**commonDirc)
        b2['boxes'][0].set(facecolor = clss[s])
        
        b2['boxes'][0].set(facecolor = clss[s])
        b2['caps'][0].set(color=clss[s])   
        b2['caps'][1].set(color=clss[s])     
        b2['whiskers'][0].set(color=clss[s])
        b2['whiskers'][1].set(color=clss[s])
        [ax1[0].axvline(4.5 + i*5, color = 'white',linestyle = 'dashed',
                        linewidth =3) for i in range(2)]
        p_value_lh = sig.loc['pvalues', fieldLH[s]]
        if p_value_lh <= 0.01:
            ax1[1].annotate('**', xy=(s-0.1,1.2), fontsize=25)
        elif 0.01 < p_value_lh <= 0.05:
            ax1[1].annotate('*', xy=(s-0.1,1.2), fontsize=25)


    ax1[0].set_ylim([-0.05,1.3]) ##change back to 1.1 if unNorm
    ax1[1].set_ylim([-0.05,1.3]) ##change back to 1.1 if unNorm
    
    
    ax1[0].set_ylabel('Mutual information (bits)',fontsize =  22)
    ax1[1].set_ylabel('Mutual information (bits)',fontsize =  22)
 
    ax1[0].annotate('A', xy=(0.95,0.8), 
                    xycoords='axes fraction', 
                    fontsize=30,
                    horizontalalignment='right',
                    verticalalignment='bottom') 
    
    ax1[1].annotate('B', xy=(0.95,0.8), 
                    xycoords='axes fraction', 
                    fontsize=30,
                    horizontalalignment='right',
                    verticalalignment='bottom') 

    ax1[0].tick_params(axis= 'both', labelsize= 16)
    ax1[1].tick_params(axis= 'both', labelsize= 16)
    plt.subplots_adjust(wspace=0.12,hspace=0.08)
    if saveFig:
        figPath = getFigPath(timewindow)
        fig1.savefig(figPath + f"/Fg1_PNAS_{timewindow}.png",
                           bbox_inches = 'tight', pad_inches = 0.1)




def getFigure2(ds2, timewindow, saveFig = False):
    plt.style.use('seaborn')

    fig2, ax2 = plt.subplots(2, 1, figsize = (25,16))
    sig, data = shuffleTest(ds2, timewindow)
    pidColors = ['r', 'darkcyan', 'olive']*3
    
    pidNEE = ['U(NEE;C13)', 'S(NEE;C13)', 'R(NEE;C13)',
              'U(NEE;H2)', 'S(NEE;H2)', 'R(NEE;H2)',
              'U(NEE;dEx)', 'S(NEE;dEx)', 'R(NEE;dEx)']    
    pidNEELabel = ['$U$($NEE$;$\delta ^{13}C$)', 
                   '$S$($NEE$;$\delta ^{13}C$)',
                   '$R$($NEE$;$\delta ^{13}C$)',
                   '$U$($NEE$;$\delta ^{2}H$)', 
                   '$S$($NEE$;$\delta ^{2}H$)',
                   '$R$($NEE$;$\delta ^{2}H$)',
                   '$U$($NEE$;$d$)', 
                   '$S$($NEE$;$d$)',
                   '$R$($NEE$;$d$)']  
    pidLH = ['U(LH;C13)', 'S(LH;C13)', 'R(LH;C13)',
             'U(LH;H2)', 'S(LH;H2)', 'R(LH;H2)',
             'U(LH;dEx)', 'S(LH;dEx)', 'R(LH;dEx)']
    pidLHLabel = ['$U$($LH$;$\delta ^{13}C$)', 
                  '$S$($LH$;$\delta ^{13}C$)',
                  '$R$($LH$;$\delta ^{13}C$)',
                  '$U$($LH$;$\delta ^{2}H$)', 
                  '$S$($LH$;$\delta ^{2}H$)',
                  '$R$($LH$;$\delta ^{2}H$)',
                  '$U$($LH$;$d$)', 
                  '$S$($LH$;$d$)',
                  '$R$($LH$;$d$)']


    NEEdfpid = data[pidNEE]
    LHdfpid = data[pidLH]       
    for s in np.arange(len(pidNEE)):
        commonDirc = {'boxprops': dict(linewidth = 5, color = pidColors[s]),
                       'whiskerprops':dict(linewidth =5,linestyle = 'solid'),
                       'medianprops':dict(linewidth = 5, zorder = 9, color= 'white'),
                       'capprops': dict(linewidth =5),
                       'meanprops':dict(marker = '^',
                                   markersize = 15, zorder = 10, 
                                   markeredgecolor='black',
                                   markerfacecolor='black'),
                       'flierprops':dict(marker='o',
                                         markerfacecolor=pidColors[s],
                                         markersize=12,
                                         linestyle='none',
                                         markeredgecolor=pidColors[s])
                       }
                  
        b1 = ax2[0].boxplot(NEEdfpid.loc[:,pidNEE[s]].dropna(), widths = 0.35,
                   notch = False, labels = [pidNEELabel[s]],
                   showmeans = True, positions = [s],patch_artist=True,
                   showfliers = True,**commonDirc)
        b1['boxes'][0].set(facecolor = pidColors[s])
        b1['caps'][0].set(color=pidColors[s])   
        b1['caps'][1].set(color=pidColors[s])     
        b1['whiskers'][0].set(color=pidColors[s])
        b1['whiskers'][1].set(color=pidColors[s])
        [ax2[1].axvline(2.5 + i*3,color='white', linestyle = 'dashed',
                        linewidth = 3) for i in range(2)]
        


        b2 = ax2[1].boxplot(LHdfpid.loc[:,pidLH[s]].dropna(), widths = 0.35,
                    notch = False, labels = [pidLHLabel[s]],patch_artist=True,
                    showmeans = True, positions = [s],
                    showfliers = True,**commonDirc)
        b2['boxes'][0].set(facecolor = pidColors[s])
        
        b2['boxes'][0].set(facecolor =  pidColors[s])
        b2['caps'][0].set(color= pidColors[s])   
        b2['caps'][1].set(color= pidColors[s])     
        b2['whiskers'][0].set(color= pidColors[s])
        b2['whiskers'][1].set(color= pidColors[s])
        [ax2[0].axvline(2.5 + i*3, color = 'white',linestyle = 'dashed',
                        linewidth =3) for i in range(2)]
       

        ax2[0].set_ylim([-0.01,0.19]) ## change back if unNorm      
        ax2[1].set_ylim([-0.01,0.19]) ## change back if unNorm 
 
        ax2[0].set_ylabel('Decomposed information (bits)',fontsize =  27)
        ax2[1].set_ylabel('Decomposed information (bits)',fontsize =  27)
        if (sig.loc['pvalues',pidNEE[s]] < 0.01):
            ax2[1].annotate('**', xy=(s-0.06,0.17), fontsize=30)
        elif 0.01 < sig.loc['pvalues',pidNEE[s]] <= 0.05:
            ax2[1].annotate('*', xy=(s-0.06,0.17), fontsize=30)

            
        if (sig.loc['pvalues',pidLH[s]] < 0.01):
            ax2[0].annotate('**', xy=(s-0.06,0.17), fontsize=30) 
        elif 0.01 < sig.loc['pvalues',pidLH[s]] <= 0.05:
            ax2[1].annotate('*', xy=(s-0.06,0.17), fontsize=30)

        ax2[0].annotate('A', xy=(0.95,0.8), 
                    xycoords='axes fraction', 
                    fontsize=30,
                    horizontalalignment='right',
                    verticalalignment='bottom') 
    
        ax2[1].annotate('B', xy=(0.95,0.8), 
                    xycoords='axes fraction', 
                    fontsize=30,
                    horizontalalignment='right',
                    verticalalignment='bottom') 

        ax2[0].tick_params(axis= 'both', labelsize= 23)
        ax2[1].tick_params(axis= 'both', labelsize= 23)

    plt.subplots_adjust(wspace=0.12,hspace=0.13)
    if saveFig:
        figPath = getFigPath(timewindow)
        fig2.savefig(figPath + f'/Fg2_PNAS_{timewindow}.png',
                            bbox_inches = 'tight', pad_inches = 0.1) 


def getIndividualLR(dataset, cmpInfo, predictorID):
    cy = copy.deepcopy(dataset)
    
    cy = cy[[cmpInfo, predictorID]].dropna()
    # print(cy.corr())
    X = sm.add_constant(cy[predictorID]) 

    model = sm.OLS(cy[cmpInfo], X)
    
    fittedModel = model.fit()
    # print(fittedModel.summary())
    return fittedModel


def getFigure3(inputData, timewindow, saveFig = False):
    plt.style.use('seaborn')
    fig3, ax3 = plt.subplots(2,3, figsize=(25,16))
    envVars = ["aridity", "MAT", "MAP", "Elev"]
    colors = ["orange", "red", "blue", "green"]
    labels = ["Aridity (PET/P)","Temperature", 
              "Precipitation",
              "Elevation"]
    markerStyle = [">", "<", "s", "o"]
    lw = 4
    scatterSize = 130
    for i0 in np.arange(2):
        flux = 'NEE' if i0 == 0 else "LH"
        pannels = ["A", "B", "C"] if i0 == 0 else ["D", "E", "F"]
       
        for j1 in np.arange(3):
            if j1 == 0:
                iso = "C13"
                isoLabel = "Scaled site factors ($\delta ^{13}C$)"
            elif j1 == 1:
                iso = "H2"
                isoLabel = "Scaled site factors ($\delta ^{2}H)$"
            else:
                iso = "dEx"
                isoLabel = "Scaled site factors ($d$)"
            plotName = "U+S("+ flux + ";" + iso + ")"
            subDataFrame = inputData[[plotName] + envVars].dropna()
        
            stdData = MinMaxScaler().fit_transform(subDataFrame[envVars]) 
            new_frame = subDataFrame[[plotName]].to_numpy()
            appeneded = np.append(new_frame, stdData, axis=1)
            appeneded = pd.DataFrame(appeneded, columns = [plotName] + envVars)
            for kk in range(4):
                transparent = 1
                siteSLR = getIndividualLR(appeneded, plotName, envVars[kk])
                pValues = siteSLR.pvalues[envVars[kk]]
                const, slope = siteSLR.params
                r2 = siteSLR.rsquared
                facecolor = colors[kk]
                minY, maxY = 0*slope + const, 1*slope + const
                print(f"For {envVars[kk]} {plotName}:slope = {slope:.5f}, p-val = {pValues}, R2 = {r2:.5f}")
                if pValues <= 0.05:
                    # alpha = 1
                    ax3[i0,j1].plot([0, 1],[minY, maxY],
                                    c = colors[kk],
                                    alpha = transparent,
                                    linewidth = lw)
                    
                elif pValues > 0.05 and pValues <= 0.1:
                    # alpha = 1
                    ax3[i0,j1].plot([0, 1],[minY, maxY],
                                    linestyle = "--", 
                                    c = colors[kk],
                                    alpha = transparent,
                                    linewidth = lw)
                else:
                    transparent = transparent * 0.2
                ax3[i0,j1].scatter(appeneded[envVars[kk]],
                                    appeneded[plotName], 
                                    label = labels[kk],
                                    alpha = transparent,
                                    s = scatterSize,
                                    marker = markerStyle[kk],
                                    facecolors = facecolor, edgecolors=colors[kk])
                                   
                ax3[i0,j1].set_xlabel(isoLabel,fontsize = 25)
                ax3[i0,j1].tick_params(axis= 'both', labelsize= 20)
            print()
            ax3[i0,j1].annotate(pannels[j1], xy=(0.9,0.85), 
                                xycoords='axes fraction', 
                                fontsize=25,
                                horizontalalignment='right',
                                verticalalignment='bottom') 
            ax3[i0,j1].set_yticks([0.02, 0.06, 0.10, 0.14, 0.18, 
                                   0.22,0.26, 0.30, 0.34])
        
        ###now create mannual legends 
        import matplotlib.lines as mlines
        
        envVars = ["aridity", "MAT", "MAP", "Elev"]
        colors = ["orange", "red", "blue", "green"]
        labels = ["Aridity (PET/P)","Temperature", 
                  "Precipitation",
                  "Elevation"]
        markerStyle = [">", "<", "s", "o"]

        aridity = mlines.Line2D([], [], markeredgecolor = "orange", marker='>', linestyle='none', markerfacecolor='none',
                          markersize=15, label="Aridity (PET/P)",markeredgewidth = 3)
        temp = mlines.Line2D([], [], markeredgecolor = "red", marker='<', linestyle='none',markerfacecolor='none',
                          markersize=15, label="Temperature",markeredgewidth = 3)
        precip = mlines.Line2D([], [], markeredgecolor="blue", marker='s', linestyle='none',markerfacecolor= 'none',
                          markersize=15, label='Precipitation',markeredgewidth= 3)
        elv = mlines.Line2D([], [], markeredgecolor = "green", marker='o', linestyle='none',markerfacecolor='none',
                          markersize=15, label="Elevation",markeredgewidth = 3)
        
        
        ax3[1,2].legend(handles = [aridity, temp, precip, elv],
                        loc = 2,labelspacing = 0.8, fontsize = 18)
        ax3[i0,0].set_ylabel("Added information for {} (bits)".format(flux),
                              fontsize = 25)
    if saveFig:
        figPath = getFigPath(timewindow)
        fig3.savefig(figPath + f'/Fg3_PNAS_{timewindow}.png',
                            bbox_inches = 'tight', pad_inches = 0.1) 


def SpatialMaps(adataFrame, part, timewindow, saveFig = False, cbarColor = "plasma"): 
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    title = [
             '($NEE$;$\delta ^{13}C$) (bits)', '($NEE$;$\delta ^{2}H$) (bits)','($NEE$;$d$) (bits)',
             '($LH$;$\delta ^{13}C$) (bits)', '($LH$;$\delta ^{2}H$) (bits)', '($LH$;$d$) (bits)'
             ]
         
    fieldTitle = ['{}(NEE;C13)', '{}(NEE;H2)', '{}(NEE;dEx)',
                  '{}(LH;C13)','{}(LH;H2)', '{}(LH;dEx)'
                  ]
    
    figl = ['A','B', 'C', 'D','E', 'F']
    dfStats = adataFrame.describe()
   
    dataLabel = [k.format(part) for k in fieldTitle]
    figureTitle = ['$'+ part + '$'+ i for i in title]
    if part == 'U+S':
        partMax = 0.21
        partMin = 0
        print(f'maximum of {part} is {dfStats.loc["max",dataLabel].max()}')

    else:
        partMax = dfStats.loc['max',dataLabel].max()
        partMin = dfStats.loc['min',dataLabel].min() 
  
    from mpl_toolkits.axes_grid1 import AxesGrid
    from cartopy.mpl.geoaxes import GeoAxes
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection = projection))   
    
    fig = plt.figure(1,figsize = (15,12))
    axgr = AxesGrid(fig,111, axes_class = axes_class, 
                        nrows_ncols=(2,3),
                        axes_pad=0.3,
                        cbar_location = 'right',
                        cbar_mode = 'single',
                        cbar_pad = 0.15,
                        cbar_size = '3%',
                        label_mode = '')
    for i in np.arange(len(axgr)):
        
        axgr[i].set_extent([-170, -55, 10, 70], ccrs.Geodetic())
        gl = axgr[i].gridlines(linestyle=':', draw_labels=False, zorder = 0)
        gl.xlabels_top = False
        gl.ylabels_right = False
        axgr[i].coastlines()

        im = axgr[i].scatter(adataFrame['Lon'], adataFrame['Lat'], 
                        alpha=1, s=80, c = adataFrame[dataLabel[i]],
                        cmap = plt.get_cmap(cbarColor), 
                        transform = ccrs.PlateCarree(), 
                        vmin = partMin, vmax = partMax)
        print(i, dataLabel[i])
        axgr.cbar_axes[i].colorbar(im)
        axgr[i].set_title(figureTitle[i], fontsize= 12)
        axgr[i].annotate(figl[i], xy=(0.1,0.48), 
                        xycoords='axes fraction', 
                        fontsize=15,zorder = 100,
                        horizontalalignment='right',
                        verticalalignment='bottom')
        
    plt.subplots_adjust(wspace=0.12,hspace=0.12)
    plt.show()
    if saveFig:
        figPath = getFigPath(timewindow)
        fig.savefig(figPath + f'/Fg{part}_PNAS_{timewindow}.png',
                    bbox_inches = 'tight', pad_inches = 0.1)


####This the figure4 of the paper
def getCoeff(data, component):
    '''This function prints the MLR information 
        between different U+S and site specific factors
    '''
    cy = copy.deepcopy(data)
    predictors = ['aridity', 'MAT','MAP','Elev']
    cy = cy[predictors + [component]].dropna()
    
    X = sm.add_constant(cy[predictors]) 

    model = sm.OLS(cy[component], X)
    
    fittedModel = model.fit()
    print(fittedModel.summary())
    return fittedModel



def getTable(df):
    headers = ["  ", "aridity", "MAT", "MAP", "Elev", "r-squred"]
    isoCompos = ["U+S(NEE;C13)","U+S(LH;C13)","U+S(NEE;H2)",
                  "U+S(LH;H2)","U+S(NEE;dEx)","U+S(LH;dEx)"]
    empty = []
    for i in isoCompos:
        mlrS = getCoeff(df, i)
        l1 = [i] + mlrS.params[1:].tolist() + [mlrS.rsquared]
        l2 = [i + "pval"] + mlrS.pvalues[1:].tolist() + [mlrS.rsquared]
        empty.append(l1)
        empty.append(l2)
    print(tabulate(empty, headers=headers))



def gatherFigInputs(timewindow):
    assert timewindow in ["daytime", "alltime"], "Only takes daytime or alltime."
    resultPath = '../MI_results/'
    siteInfo = pd.read_csv('../190115-field-sites.csv') ## whole lot of inforamtion here and we only need part of it
    siteInfo = siteInfo[(siteInfo['Site Type'] == 'Core Terrestrial')|  
                           (siteInfo['Site Type'] == 'Relocatable Terrestrial')]
    site_Mete = siteInfo[['Site ID', 'Mean Annual Temperature',
                             'Mean Annual Precipitation','Elevation']]
    Names = [i for i in site_Mete['Site ID']]
    Temp =  [float(i.split('C')[0]) for i in site_Mete['Mean Annual Temperature']]
    Precip = [float(i.split('mm')[0]) for i in site_Mete['Mean Annual Precipitation']]
    Elev = [float(i.split('m')[0]) for i in site_Mete['Elevation']]
    siteLat = [float(i.split(',')[0]) for i in siteInfo['Lat./Long.']]
    siteLon =  [float(i.split(',')[1]) for i in siteInfo['Lat./Long.']]
    Names2 = [i for i in siteInfo['Site ID']]
    spatialplots =  pd.DataFrame({'site': Names2, 'Lat': siteLat, 
                              'Lon': siteLon})
    site_dict = pd.DataFrame({'site': Names, 'MAT': Temp, 'MAP': Precip, 'Elev':Elev})
    site_dict.sort_values(by = ['site'], inplace=True)

    aridityCanopy = pd.read_csv("../NEONaridityandcanopy.csv",
                            index_col= 0)
    aridityCanopy['aridity'] = 1/aridityCanopy['aridity'] ##convert aridity to PET/P
    Env_vars = reduce(lambda x, y: pd.merge(x, y, on = 'site'),
                   [aridityCanopy, site_dict])
    # myData = pd.read_csv(resultPath + f'MI_and_PID_NEON_{timewindow}_iso.csv') ##pay attention to here
    myData = pd.read_csv(resultPath + f'MI_and_PID_NEON_{timewindow}_iso_02032023.csv') ##pay attention to here

    myData.sort_values(by = 'site', inplace=True)
    forSpatial = reduce(lambda x, y: pd.merge(x, y, on = 'site'),
                   [myData[['site','U(NEE;C13)', 'S(NEE;C13)', 'R(NEE;C13)','U+S(NEE;C13)',
                            'U(LH;C13)', 'S(LH;C13)', 'R(LH;C13)','U+S(LH;C13)',
                            'U(NEE;H2)', 'S(NEE;H2)', 'R(NEE;H2)','U+S(NEE;H2)',
                            'U(LH;H2)', 'S(LH;H2)', 'R(LH;H2)','U+S(LH;H2)',
                            'U(NEE;dEx)', 'S(NEE;dEx)', 'R(NEE;dEx)','U+S(NEE;dEx)',
                            'U(LH;dEx)', 'S(LH;dEx)', 'R(LH;dEx)','U+S(LH;dEx)']], spatialplots ])
    finalData = pd.merge(myData, Env_vars, on = 'site')
    return myData, finalData, forSpatial

def getTableStatistics(data):    
    desMI = data.describe()
    print(desMI[['I(NEE;C13)', 'I(NEE;H2)', 'I(NEE;dEx)']])
    print(desMI[['I(LH;C13)', 'I(LH;H2)', 'I(LH;dEx)']])
    print('-----------------------------------------------')
###PIDs
    print(desMI[['U(NEE;C13)', 'S(NEE;C13)', 'R(NEE;C13)', 'U+S(NEE;C13)']])
    print(desMI[['U(NEE;H2)', 'S(NEE;H2)', 'R(NEE;H2)', 'U+S(NEE;H2)']])
    print(desMI[['U(NEE;dEx)', 'S(NEE;dEx)', 'R(NEE;dEx)', 'U+S(NEE;dEx)']])
    print('-----------------------------------------------')
    print(desMI[['U(LH;C13)', 'S(LH;C13)', 'R(LH;C13)', 'U+S(LH;C13)']])
    print(desMI[['U(LH;H2)', 'S(LH;H2)', 'R(LH;H2)', 'U+S(LH;H2)']])
    print(desMI[['U(LH;dEx)', 'S(LH;dEx)', 'R(LH;dEx)', 'U+S(LH;dEx)']])
###
    addC13NEE = data['U+S(NEE;C13)']/(data['U(NEE;C13)'] + 
                                      data['S(NEE;C13)'] + 
                                      data['R(NEE;C13)'])
    addC13LH = data['U+S(LH;C13)']/(data['U(LH;C13)'] + 
                                    data['S(LH;C13)'] + 
                                    data['R(LH;C13)'])

    addH2NEE = data['U+S(NEE;H2)']/(data['U(NEE;H2)'] + 
                                    data['S(NEE;H2)'] + 
                                    data['R(NEE;H2)'])
    addH2LH = data['U+S(LH;H2)']/(data['U(LH;H2)'] + 
                                  data['S(LH;H2)'] + 
                                  data['R(LH;H2)'])
    
    adddExNEE = data['U+S(NEE;dEx)']/(data['U(NEE;dEx)'] + 
                                      data['S(NEE;dEx)'] + 
                                      data['R(NEE;dEx)'])
    adddExLH = data['U+S(LH;dEx)']/(data['U(LH;dEx)'] + 
                                                data['S(LH;dEx)'] + 
                                                data['R(LH;dEx)'])   
    
    H_headers = ['C13NEE', 'C13LH', 'H2NEE', 'H2LH', 'dEXNEE', 'dExLH']
    from tabulate import tabulate
    
    print(tabulate([['C13NEE',round(np.nanmean(addC13NEE),2)], 
                    ['C13LH',round(np.nanmean(addC13LH),2)], 
                    ['H2NEE',round(np.nanmean(addH2NEE),2)],
                    ['H2LH',round(np.nanmean(addH2LH),2)],
                    ['dExNEE',round(np.nanmean(adddExNEE),2)],
                    ['dExLH',round(np.nanmean(adddExLH),2)]],
                    tablefmt='grid'))

def CompareAllTimeAndDayTime(allTime, dayTime,saveFig = False):
    import dataframe_image as dfi

    allTime = copy.deepcopy(allTime)
    dayTime = copy.deepcopy(dayTime)
    testArray = pd.DataFrame([], columns = allTime.columns[1:], index = ['pvalues', 'tstats'])
    for i in allTime.columns[1:]:
        f, a = plt.subplots()
        max1 = max(allTime[i].max(),dayTime[i].max())
        min1 = min(allTime[i].min(),dayTime[i].min())
        a.scatter(dayTime[i],allTime[i])
        a.plot([min1, max1],[min1, max1])
        a.set_xlabel(f"{i}-dayTime")
        a.set_ylabel(f"{i}-allTime")
        newNameat = i  + '-alltime'
        newNamedt = i  + '-daytime'
        allTime.rename(columns={i:newNameat}, inplace = True)
        dayTime.rename(columns={i:newNamedt}, inplace = True)
        dtt = pd.merge(allTime[['site',newNameat]],dayTime[['site',newNamedt]], on = 'site').dropna()
        a.scatter(dtt.describe()[newNamedt]['mean'], 
                  dtt.describe()[newNameat]['mean'], 
                  marker = '*', s = 75, 
                  zorder = 2, c= "red", label = "mean")
        a.legend()
        t_test = stats.ttest_rel(dtt[newNameat], dtt[newNamedt], alternative = 'two-sided')
        testArray.loc['pvalues', i] = t_test[1] 
        testArray.loc['tstats', i] = t_test[0]
        a.annotate(f'allTime != dayTime, p = {t_test[1]:.2f}', xy=(0.1, 0.8), xycoords=a.transAxes,
                   fontsize = 12)
        if saveFig:
            f.savefig(f'../Figures/analysisplots/{i}.pdf',
                        bbox_inches = 'tight', pad_inches = 0.1)
            testArray.T.to_csv(f'../Figures/analysisplots/ttestarray.csv')
            testArray.T[testArray.T.iloc[:,0] <= 0.05].to_csv(f'../Figures/analysisplots/testnotequalto.csv')
                                 
            
    return testArray.T

if __name__ == "__main__":
    allTimeFigData, allTimeRegData, allTimeSptlData = gatherFigInputs("alltime") 
    # dayTimeFigData, dayTimeRegData, dayTimeSptlData = gatherFigInputs("daytime")
    # ok = CompareAllTimeAndDayTime(allTimeFigData, dayTimeFigData, saveFig = True)
    ##Figures using all time isotope datasets
    getFigure1(allTimeFigData, "alltime", saveFig = False)
    # getFigure1(dayTimeFigData, "daytime", saveFig = True)
    
    getFigure2(allTimeFigData, "alltime", saveFig = False)
    # getFigure2(dayTimeFigData, "daytime", saveFig = True)

    getFigure3(allTimeRegData, "alltime", saveFig = False) 
    # getFigure3(dayTimeRegData, "daytime", saveFig = True)
    ## run the following lines after
    #for i in ['U', 'S', 'R', 'U+S']:
    #    SpatialMaps(allTimeSptlData, i, "alltime", saveFig = True, cbarColor='CMRmap')
    #     SpatialMaps(dayTimeSptlData, i, "daytime", saveFig = True, cbarColor='CMRmap')
   




        
        

# getTable(finalData)




  
        






    
# getFigure3(finalData, savefig3=True)             
    
# envVars = ["aridity", "MAT", "MAP", "Elev"]
# colors = ["orange", "red", "blue", "cyan"]
# labels = ["Aridity (PET/P)","Temperature", 
#           "Precipitation",
#           "Elevation"]
# fig,ax = plt.subplots()
# plotName  = "U+S(LH;C13)"
# for kk in range(4):
#     subDataFrame = finalData[[plotName] + envVars].dropna()
#     '''This is a bug here fix it
#     '''
#     stdData = MinMaxScaler().fit_transform(subDataFrame[envVars]) 
#     new_frame = subDataFrame[[plotName]].to_numpy()
#     appeneded = np.append(new_frame, stdData, axis=1)
#     appeneded = pd.DataFrame(appeneded, columns = [plotName] + envVars)
#     ax.scatter(appeneded[envVars[kk]],
#                 appeneded[plotName], label = labels[kk],
#                 c = colors[kk])






# getFigure1(myData, saveFigure1 = True)




# getFigure2(myData, saveFigure2 = True)





        
# 





# def getFigure4(data, ifsaveFig = False):
#     data = copy.deepcopy(data)
#     data['Ele label'] = data['Elev'].apply(lambda value: 'o' 
#                                            if value <= 1000 else 's' 
#                                            if value <= 2000 else '^')
#     plotData = data.loc[~np.isnan(finalData['U+S(LH;H2)'])]
#     plotData.reset_index(drop=True, inplace=True)
#     ########Here I'd like to plot the data/model and inset with model vs observed
#     aridityMesh = [0.2, 0.4, 0.8, 1, 2, 4, 5, 8]
#     fig, ax = plt.subplots(figsize=(10,10))
#     cp = mb.cm.RdYlBu_r
#     cnorm = mb.colors.LogNorm(vmin=0.1, vmax=10)
#     smap = mb.cm.ScalarMappable(norm=cnorm, cmap=cp)
#     ####
#     lgdSize = 12
#     xylabel = 17
#     ####
#     ####plot the points
#     for i in aridityMesh:
#         t1, us1, _, _ = getMLR(plotData, 'U+S(LH;H2)', i, 500)
#         t2, us2, _, _ = getMLR(plotData, 'U+S(LH;H2)', i, 1500)
#         t3, us3, _, _ = getMLR(plotData, 'U+S(LH;H2)', i, 2500)
    
#         ax.plot(t1, us1, c = smap.to_rgba(i), linestyle = 'solid')
#         ax.plot(t2, us2, c = smap.to_rgba(i), linestyle = 'dashed')
#         ax.plot(t3, us3, c = smap.to_rgba(i), linestyle = 'dotted')
    
#     for i in np.arange(plotData.shape[0]):
#         ax.scatter(plotData.loc[i,'MAT'], plotData.loc[i,'U+S(LH;H2)'] , 
#                 c = np.array(smap.to_rgba(plotData.loc[i,'aridity'])).reshape(1,-1), 
#                 marker = plotData.loc[i,'Ele label'], s= 225,
#                 edgecolors = 'black')
#     ax.set_xlabel("Mean annual temperature (\u00B0C)", fontsize = xylabel)
#     ax.set_ylabel("Added information from $\delta ^2$H, U + S (bits)", fontsize = xylabel)
#     ax.set_xlim([-5,25])
#     ax.set_ylim([0,0.25])
#     ax.tick_params(axis= 'both', direction = 'in', labelsize = 15)
#     ax.scatter(None,None, marker = 'o', s = 120, label = 'Elevation $\leq$ 1000 m', 
#                facecolors = 'none', edgecolors='black')
#     ax.scatter(None, None, marker  = 's', s= 120, label = '1000 m < Elevation $\leq$ 2000 m',
#                facecolors = 'none', edgecolors='black')
#     ax.scatter(None, None, marker = '^', s = 120, label = '2000 m $\leq$ Elevation',
#                facecolors = 'none', edgecolors='black')
#     ax.plot([],[], linestyle = 'solid',label = 'Elevation = 500 m', c= 'black')
#     ax.plot([],[], linestyle = 'dashed', label = 'Elevation = 1500 m',c= 'black')
#     ax.plot([],[], linestyle = 'dotted', label = 'Elevation = 2500 m',c= 'black')
#     ax.legend(loc = 2, fontsize=lgdSize,framealpha = 0)
    
            
#     ####inset figure
#     inset_ax = ax.inset_axes([14.2, 0.16, 10.25, 0.085],
#                         transform=ax.transData)
#     ft, og, r2 = getMLR(plotData, 'U+S(LH;H2)', 0, 0, fit = True)
#     inset_ax.scatter(og, ft, c = 'black', s= 60, marker = 'D')
#     inset_ax.set_xlim(0.04, 0.15)
#     inset_ax.set_ylim(0.04, 0.15)
#     inset_ax.set_xlabel("Observed U + S (bits)", fontsize = 13)
#     inset_ax.set_ylabel("Modeled U + S (bits)", fontsize = 13)
#     inset_ax.plot([0.04, 0.15],[0.04, 0.15], linestyle = 'dashed', c = 'grey')
#     inset_ax.text(0.1, 0.9, '$R^2$ = {:.2f}'.format(r2),transform=inset_ax.transAxes,
#                   fontsize = 12)
    
#     inset_ax.tick_params(axis="both",direction="in")
#     ###position the colorbar inside the figure
#     cbaxes=fig.add_axes([0.15,0.16,0.3,0.02]) ##add axis to colorbar
    
#     cbar = fig.colorbar(smap, cax = cbaxes, orientation = 'horizontal')
                        
#     cbar.ax.xaxis.set_ticks_position('bottom')
#     cbar.ax.set_title('Annual aridity, PET/P',fontsize = 13)
#     cbar.ax.tick_params(axis="both", labelsize = 12)
#     if ifsaveFig:
#         fig.savefig('C:/Users/libon/Box/neon_extrac_data/results/PNASfigs/Figure4_v2.png',
#                    bbox_inches = 'tight', pad_inches = 0.05)
   
# getFigure4(finalData, ifsaveFig=False)

###lets get some statistics 
###results paragraph 2 

               

# def getFigure4(data, component, saveFigure = True):
#     olsPara, stats, ogUS, ftUS,r2s = getCoeff(finalData, 'U+S(LH;H2)')
#     ard = np.arange(0.1,10.1,0.01)
#     temp = np.arange(-5, 36, 1)
#     Xx, Yy= np.meshgrid(temp, ard)
#     ###Multivariate Linear regression 
#     Zz = Yy*olsPara['aridity'] + Xx*olsPara['MAT'] + \
#         stats['MAP']*olsPara['MAP'] + stats['Elev']*olsPara['Elev'] + \
#             olsPara['const']
#     Vmin = 0.03
#     Vmax = 0.20
#     cp = mb.cm.plasma
#     cnorm = mb.colors.Normalize(vmin=Vmin, vmax=Vmax)
#     smap = mb.cm.ScalarMappable(norm=cnorm, cmap=cp)
   
#     fig, ax = plt.subplots(figsize=(9,8))
#     CS = ax.contour(Xx, Yy, Zz, [0.05,0.07,
#                                  0.09,0.11,0.13,
#                                  0.15,0.17,0.19],
#                     linewidths=2.5, 
#                     cmap = 'plasma',
#                     norm = cnorm,
#                     vmin=Vmin, vmax=Vmax, zorder = 0) 
                   
#     for points in np.arange(len(ogUS)):
#         ax.scatter(ogUS.iloc[points,1], ogUS.iloc[points,0], s = 140,
#               c =  np.array(smap.to_rgba(ogUS.iloc[points,-1])).reshape(1,-1),
#               edgecolors = 'black', zorder = 10)
#     ax.set_ylabel('Mean annual aridity ($PET$/$P$)', fontsize= 16)
    
#     ax.set_xlabel('Mean annual temperature (\u00B0C)', fontsize= 16)
#     ax.set_yscale('log')
#     ax.clabel(CS, inline=True, fontsize=12)
#     ax.tick_params(axis="both",which = 'both',direction="in", labelsize = 14)
#     inset_ax = ax.inset_axes([19, 0.15, 15.5, 0.85],
#                         transform=ax.transData)
#     # inset_ax.set_aspect('equal')

#     inset_ax.scatter(ogUS['U+S(LH;H2)'], ftUS, c = 'black', s= 50, marker = 'D')
#     inset_ax.plot([])
#     inset_ax.set_xlim(0.04, 0.15)
#     inset_ax.set_ylim(0.04, 0.15)
#     inset_ax.plot([0.04,0.15],[0.04,0.15], linewidth = 2,
#                   linestyle = 'dashed', color = 'grey', zorder =0)
#     inset_ax.set_xlabel("Observed $U$ + $S$ (bits)", fontsize = 12)
#     inset_ax.set_ylabel("Modeled $U$ + $S$ (bits)", fontsize = 12)
#     inset_ax.tick_params(axis="both",direction="in",
#                          labelsize = 12)
#     inset_ax.locator_params(axis='y', nbins=6)
#     inset_ax.locator_params(axis='x', nbins=6)
#     inset_ax.text(0.1, 0.9, '$R^2$ = {:.2f}'.format(r2s),transform=inset_ax.transAxes,
#                   fontsize = 12)
    
#     cbaxes=fig.add_axes([0.63,0.54,0.25,0.03]) ##add axis to colorbar
#     cbar = fig.colorbar(smap, cax = cbaxes, orientation = 'horizontal')
    
#     cbar.ax.set_title('Added information from $\delta ^2$H \n $U$ + $S$ (bits)',
#                       fontsize = 11)
#     cbar.ax.tick_params(axis="both", labelsize = 12, direction='in')
#     if  saveFigure:
#         fig.savefig('C:/Users/libon/Box/neon_extrac_data/results/PNASfigs/Figure{}_v2.png'.format(component),
#                     bbox_inches = 'tight', pad_inches = 0.1)
# getFigure4(finalData, 'U+S(LH;H2)')

















