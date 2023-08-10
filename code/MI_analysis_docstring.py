# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:55:45 2021

@author: libon
"""
import numpy as np
import os 
import pandas as pd
import copy
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gaussian_kde


# NeonPath = "../Processed data 24hiso/"
# NeonPath = "../alltime_data/"

# cleanUpData = os.listdir(NeonPath)


def get_pdfs(dataset, *args):
    """
    Get probablity density function based on gaussian kernel and silverman binningh method.   

    Parameters
    ----------
    dataset : pandas.DataFrame
        Dataframe must include all variables in *args.
    *args : list
        list of column names in dataset.

    Raises
    ------
    
        ValueError will be raised while asked to computed more than 4-D pdf.

    Returns
    -------
    joint_pdf : pandas.DataFrame
        This dataframe contains evaluted meshgrid points and joint pdf.

    """
    ## set constraints on what dataset this function accepts.
    assert isinstance(dataset, pd.DataFrame), "Input must be pd.DataFrame"
    listOfVars = list(args)
    ## copy that list of vars and transform all variables between 0 and 1 [inclusive]
    dtcpy = copy.deepcopy(dataset[listOfVars]) 
    ## (npoints, ndim)
    data_trans = MinMaxScaler().fit_transform(dtcpy) 
    ndim = dtcpy.shape[1]
    ## evaluate the estimated pdf at a meshgrid of 0.05, i.e.,[0, 0.05, 0.1 ... 1]
    if ndim == 2:
        xx, yy =  np.mgrid[data_trans[:,0].min():data_trans[:,0].max():21j, 
                           data_trans[:,1].min():data_trans[:,1].max():21j]
        ## shape of (npoints, ndim)
        meshGrid = np.vstack([xx.ravel(), yy.ravel()]).T 
    elif ndim == 3:
        xx, yy, zz =  np.mgrid[data_trans[:,0].min():data_trans[:,0].max():21j, 
                               data_trans[:,1].min():data_trans[:,1].max():21j,
                               data_trans[:,2].min():data_trans[:,2].max():21j]  
        ## shape of (npoints, ndim)
        meshGrid = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T       
    else:
        raise  ValueError("This function is not applicable for ndim > 4 pdfs")
    ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html 
    ## gaussian_kde takes array of (#ndims, #npoint)
    ## Therefore, we need to transpose the data_trans 
    multiD_kde = gaussian_kde(data_trans.T, bw_method = 'silverman')
    ## multiD_kde takes array of (ndim, npoints)
    ## Threfore, transpose the meshGrid (npoints, ndim) --> meshGrid.T (ndim, npoints)
    pdf = multiD_kde(meshGrid.T)    
    ## Probability of each data point is computed as 
    ## P_i = pdf_i * gridx *  gridy / sum(pdf_i * gridx *  gridy) 
    ## P_i = Pdf_i / sum(pdf_i)
    n_pdf = pdf/np.sum(pdf)
    ## concatenate all information together
    joint_pdf = pd.DataFrame(np.column_stack((meshGrid, n_pdf)), 
                             columns= listOfVars + ['pdf'])
    return joint_pdf
 
def shuffleData(a_dataframe, *args):
    """
    shuffle a dataframe.

    Parameters
    ----------
    a_dataframe : pandas.DataFrame
        dataframe to be shuffled.
    *args : list
        list of columns to be shuffled.

    Returns
    -------
    pandas.DataFrame
        The dataframe/sub-dataframe after shuffling.

    """

    varsList = list(args)
    shuffleParts = copy.deepcopy(a_dataframe[varsList])
    toNumpyShuffle = shuffleParts.to_numpy(copy = True)
    for i in np.arange(len(varsList)):
        np.random.shuffle(toNumpyShuffle[:,i])
        
    return pd.DataFrame(toNumpyShuffle, columns = varsList)


def getEntropy(pdf):
    '''
    Compute Shannon's entropy (base of 2) for a given pdf.

    Parameters
    ----------
    pdf : pandas.DataFrame
        The dataframe must contain a column named 'pdf'.

    Returns
    -------
    float
        The entropy/joint entropy for a given pdf in unit of bit.

    '''
    return -np.sum(pdf['pdf']*np.log2(pdf['pdf']))


def get2DMI(dt, var1, var2, ifNorm = False):
    """
    Computes mutual information between var1 and var2.

    Parameters
    ----------
    dt : pandas.DataFrame
        The dataset that the agorithm will run on.
    var1 : str
        One of the column name of dt. i.e., 'VPD_f' OR 'Tair_f' OR 'wdsp' OR 'Rg_f'
    var2 : str
        One of the column name of dt. i.e., 'NEE_U50_f' OR 'LE_U50_f'.
    ifNorm : bool, optional
        If the mutual information is normalized by flux entropy. The default is False.

    Returns
    -------
    float
        The normalized or unnormalized mutual information.

    """
    assert var2 in ['NEE_U50_f', 'LE_U50_f'], "second variable must be CO2 or H2O flux."
    ## joint pdf
    pdf_2d = get_pdfs(dt, var1, var2)
    ## compute marginal pdfs of each variable.
    ## i.e., p_var1_i = sum over j of p_var1_i_var2_j,
    ## translated as ---> groupby var1_i sum over 
    x_pdf = pdf_2d.groupby([var1]).agg({'pdf':'sum'})  
    y_pdf = pdf_2d.groupby([var2]).agg({'pdf':'sum'})  
    ## Hx --> entropy of micrometerol0gical variable.
    ## Hy --> entropy of flux variable.
    ## Hxy --> joint entropy of micrometerological variables and flux varaible.
    Hxy = getEntropy(pdf_2d)
    Hx = getEntropy(x_pdf)
    Hy = getEntropy(y_pdf)
    
    if ifNorm:
        return (Hx + Hy - Hxy)/Hy
    else:
        return Hx + Hy - Hxy
    

        


def getPIDs(pid, source1, isotope, target, norm = False):
    '''
    Compute the partial information decomposition components using method 
    from Goodwell & Kumar(2017) https://doi.org/10.1002/2016WR020216
    
    Parameters
    ----------
    pid : pandas.DataFrame
        Shape of (npoints, ndim).
    source1 : str
        Source name while the PID will be using.
        This is the MICROMETEROLOGICAL variable name.
    isotope : str
        Isotope name while the PID will be using.
    target : str
        The target name PID will be using.
        This is flux variable name.
    norm : bool, optional
        If we wants to normalize components by the entropy of the flux variable.
        The default is False.

    Returns
    -------
    list
        The elements are:
        unique inforamtion of isotope(Us): information contribution uniquely from isotope dataset.
        synergistic information(S)       : information contribution synergistically from isotope and 
                                           micrometerological variable.
        redundanty information(R)        : information that is redundantly share between istope, micrometerological vairable 
                                           to the target variable/flux.
        total added inforamtion(Us+ R)   : conditional mutual information --> I(isotope,target|source1) 
                                           i.e., I(isotope, source1;target) - I(source1;target)

    '''
    assert source1 in ['VPD_f', 'Tair_f', 'wspd', 'Rg_f'],"source1 must be one of the mircometerological variables."
    assert isotope in ['d2H', 'd13C', 'dExcess'],"isotope must be one of these isotope names."
    assert target in ['NEE_U50_f', 'LE_U50_f'],"target variable must be CO2 or H2O flux."
    data2 = copy.deepcopy(pid)
    ## create dataframe with columns as [source1, isotope, taget]
    pdfs = get_pdfs(data2, source1, isotope, target)
    ## compute marginal pdfs of each variable.
    s1_pdf = pdfs.groupby([source1]).agg({'pdf':'sum'}) 
    iso_pdf = pdfs.groupby([isotope]).agg({'pdf':'sum'})
    flux_pdf = pdfs.groupby([target]).agg({'pdf':'sum'}) 
    ## compute 2-D joint pdfs
    s1_flux_pdf =  pdfs.groupby([source1, target]).agg({'pdf':'sum'})  
    iso_flux_pdf =  pdfs.groupby([isotope, target]).agg({'pdf':'sum'})
    s1_iso_pdf =  pdfs.groupby([source1, isotope]).agg({'pdf':'sum'}) 
    ## compute entropy
    Hs1 = getEntropy(s1_pdf)     #entropy of source1
    Hiso = getEntropy(iso_pdf)   #entropy of isotope
    Hflux = getEntropy(flux_pdf) #entropy of flux
    ## compute joint entropy of two variables
    Hs1_flux = getEntropy(s1_flux_pdf) 
    Hiso_flux = getEntropy(iso_flux_pdf)
    Hs1_iso = getEntropy(s1_iso_pdf)
    ## compute the joint entropy of all variables
    H_all = getEntropy(pdfs)
    ## compute mutual inforamtion 
    ## I(X;Y) = H(X) + H(Y) - H(X;Y)
    Is1_flux = Hs1 + Hflux - Hs1_flux
    Iiso_flux = Hiso + Hflux - Hiso_flux
    Is1_iso = Hiso + Hs1 - Hs1_iso
    ## compute multi-variate mutual information
    ## I(X,Y;Z) = H(X,Y) + H(Z) - H(X,Y,Z)
    ## Z --> fluxe; X,Y --> source1
    I_all = Hflux + Hs1_iso - H_all

    R_MMI = min(Iiso_flux, Is1_flux)    #RMMI = min[I(source1;target), I(isotope;target)], eq. (7) in Allison 2017 
    #interaction information 
    II = I_all - Iiso_flux - Is1_flux  #total information minus the sum of individuals
    
    R_min = max(0, -II)                #Rmin = max(0 -II) eq.(9) in Allison 2017
    Is = Is1_iso/min(Hiso, Hs1)        #Is = I(source1;isotope)/min[H(source1),H(isotope)], source dependency, eq.10 in Allison 2017
    Rs = R_min + Is*(R_MMI -  R_min)   #Rs = Rmin + Is*(RMMI - Rmin),rescaled redundancy, eq.(8) in Allison 2017
    u_iso = Iiso_flux - Rs             #Uiso = I(isotope;flux) - Rs eq.(4) or eq.(5) in Allison 2017
    S = II + Rs                        #II = S - R --> S = II + R
    u_S = [u_iso, S, Rs, u_iso + S]
   
    if norm:
        return [u_iso/Hflux, S/Hflux, Rs/Hflux, (u_iso + S)/Hflux]
    else:
        return u_S

# NeonPath = "../Processed data 24hiso/"
# NeonPath = "../alltime_data/"
NeonPath = "../alltime_time_correct_data/"

cleanUpData = os.listdir(NeonPath)


OverallDf = pd.DataFrame(np.full((47, 55), np.nan))

for s in np.arange(len(cleanUpData)):
    # startTime =  time.time()
    siteName = cleanUpData[s].split("_")[0]
    OverallDf.iloc[s,0] = siteName
    print("start working on {}".format(siteName.split("_")[0]))
    dfs = pd.read_csv(NeonPath + cleanUpData[s], index_col=0) 
    dfs['dExcess'] = dfs['d2H'] - 8*dfs['d18O'] # d-excess = dD(d2H) - 8*dO18 page 1475 Catherine 2021
    data = dfs.drop(columns=['d18O']).copy() ## just drop the O18 
    cols = list(data.columns)[:6]
    #==================only work on unshuffled dataset
    C13Iso = data[cols + ['d13C']].dropna().copy()
    dExIso = data[cols + ['dExcess']].dropna().copy()  
    H2Iso = data[cols + ['d2H']].dropna().copy()
    #==================only work on unshuffled dataset
    ###number of data points for each datasets 
    nptsC13 = C13Iso.shape[0]
    nptsH2 = H2Iso.shape[0]
    nptsdEx = dExIso.shape[0]
    print("Working on {}".format(siteName))
    nonFluxVar = ['VPD_f', 'Tair_f', 'wspd', 'Rg_f'] # this order is important to keep 
    
    ## we only do information analysis when we have certain number of datapoints simutaneously.
    if nptsC13 >= 100: #change back to 40 to recover 2021 pattern 
        c13sig = np.full((500, 18),np.nan)
       
        for rsp in np.arange(500):
            #originally as 40; # 100 samples and disallow sampling row more than once.
            sampleC13 = C13Iso.sample(n = 100, replace = False) 
            ## mutual information between NEE and meterological variable 
            ## mutual information between NEE and carbon isotope dataset. 
            rsC13NEE = [get2DMI(sampleC13, i, 'NEE_U50_f') for i in nonFluxVar + ['d13C']]
            ## mutual information between Water flux and meterological variable 
            ## mutual information between Water flux and carbon isotope dataset. 
            rsC13LH  = [get2DMI(sampleC13, i, 'LE_U50_f') for i in nonFluxVar + ['d13C']]
            
            pidC13NEE = np.array([getPIDs(sampleC13, i, 'd13C', 'NEE_U50_f') for i in nonFluxVar]) ## stacking as rows
            pidC13LH = np.array([getPIDs(sampleC13, i, 'd13C', 'LE_U50_f') for i in nonFluxVar])  ## stacking as rows
           
            c13sig[rsp,:] = rsC13NEE + rsC13LH + list(np.nanmean(pidC13NEE, axis = 0)) + list(np.nanmean(pidC13LH, axis = 0))

        OverallDf.iloc[s,1:19] = np.nanmean(c13sig, axis = 0)
        ## code check here done 
    if nptsH2 >= 100:#change back to 40
        H2sig = np.full((500, 18),np.nan)
       
        for ll in np.arange(500):

            sampleH2 = H2Iso.sample(n = 100, replace = False)#change back to 40 if necessary
                             
            rsH2NEE = [get2DMI(sampleH2, i, 'NEE_U50_f') for i in nonFluxVar + ['d2H']]
            rsH2LH  = [get2DMI(sampleH2, i, 'LE_U50_f') for i in nonFluxVar + ['d2H']]
            
            pidH2NEE = np.array([getPIDs(sampleH2, i, 'd2H', 'NEE_U50_f') for i in nonFluxVar])
            pidH2LH = np.array([getPIDs(sampleH2, i, 'd2H', 'LE_U50_f') for i in nonFluxVar])
            
            H2sig[ll,:] = rsH2NEE + rsH2LH + list(np.nanmean(pidH2NEE, axis = 0)) + list(np.nanmean(pidH2LH, axis = 0))
        
        OverallDf.iloc[s, 19:37] = np.nanmean(H2sig, axis = 0) ## we have 18 quantities for each isotope .iloc[] is a non-inclusive method
        ## code check here done 

    if nptsdEx >= 100:#change back to 40
        dExsig = np.full((500, 18),np.nan)
       
        for kk in np.arange(500):
            sampledEx = dExIso.sample(n = 100, replace = False)#change back to 40 if necessary
                             
            rsdExNEE = [get2DMI(sampledEx, i, 'NEE_U50_f') for i in nonFluxVar + ['dExcess']]
            rsdExLH  = [get2DMI(sampledEx, i, 'LE_U50_f') for i in nonFluxVar + ['dExcess']]
            
            piddExNEE = np.array([getPIDs(sampledEx, i, 'dExcess', 'NEE_U50_f') for i in nonFluxVar])
            piddExLH = np.array([getPIDs(sampledEx, i, 'dExcess', 'LE_U50_f') for i in nonFluxVar])
            
            dExsig[kk,:] = rsdExNEE + rsdExLH + list(np.nanmean(piddExNEE, axis = 0)) + list(np.nanmean(piddExLH, axis = 0))
     
        OverallDf.iloc[s,37:] = np.nanmean(dExsig, axis = 0)
        ## code check here done
    
OverallDf.set_axis(['site','I(NEE;VPD)13', 'I(NEE;T)13', 'I(NEE;u)13','I(NEE;R)13','I(NEE;C13)',#5
                    'I(LH;VPD)13', 'I(LH;T)13', 'I(LH;u)13','I(LH;R)13','I(LH;C13)',#10
                    'U(NEE;C13)','S(NEE;C13)','R(NEE;C13)', 'U+S(NEE;C13)',#14
                    'U(LH;C13)','S(LH;C13)','R(LH;C13)', 'U+S(LH;C13)',#18
                    
                    'I(NEE;VPD)2', 'I(NEE;T)2', 'I(NEE;u)2','I(NEE;R)2','I(NEE;H2)',
                    'I(LH;VPD)2', 'I(LH;T)2', 'I(LH;u)2','I(LH;R)2','I(LH;H2)',
                    'U(NEE;H2)','S(NEE;H2)','R(NEE;H2)', 'U+S(NEE;H2)',
                    'U(LH;H2)','S(LH;H2)','R(LH;H2)', 'U+S(LH;H2)', #36
                               
                    'I(NEE;VPD)dx', 'I(NEE;T)dx', 'I(NEE;u)dx','I(NEE;R)dx','I(NEE;dEx)',
                    'I(LH;VPD)dx', 'I(LH;T)dx', 'I(LH;u)dx','I(LH;R)dx','I(LH;dEx)',
                    'U(NEE;dEx)','S(NEE;dEx)','R(NEE;dEx)','U+S(NEE;dEx)',
                    'U(LH;dEx)','S(LH;dEx)','R(LH;dEx)', 'U+S(LH;dEx)'], 
            axis='columns', inplace=True)
# OverallDf.to_csv('../MI_results/MI_and_PID_NEON_alltime_iso.csv', index = False)
# OverallDf.to_csv('../MI_results/MI_and_PID_NEON_alltime_iso_02032023.csv', index = False)
from datetime import date
OverallDf.to_csv(f'../MI_results/MI_and_PID_NEON_alltime_iso_time_correctd_{date.today()}.csv', index = False)

                                







