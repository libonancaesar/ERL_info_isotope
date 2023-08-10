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
import glob 

    
def get_ws_data(path):
    list_of_path = list(glob.glob(os.path.join(path,'**/*.csv'),recursive=True))
    find_csvs = list_of_path
    file_list = []
    df_names = ['NEON', 'DOM', 'SITE', 'DPL', 'PRNUM', 'REV', 'HOR',
                'VER', 'TMI', 'DESC', 'YYYY-MM', 'PKGTYPE', 'GENTYPE', 'FILETYPE', 'PATH']
    for i in find_csvs:
        ## last string literal after "\\" should be a .csv file
        split_string = i.split("\\")[-1]
        split_list = split_string.split(".")
        ## refer to https://data.neonscience.org/file-naming-conventions for the abbreviation. 
        if len(split_list) == 14:
            file_list.append(split_list + [i])
    file_list = pd.DataFrame(file_list, columns=df_names)
    return file_list
    





fileList = os.listdir("E:/NEON_iso_MI/Flux and other data")
ws_path = "E:/NEON_iso_MI/Flux and other data verifty/NEON_wind-2d"
ws_site = get_ws_data(ws_path)
ws_site["VER"] = ws_site["VER"].apply(lambda x: int(x))
timeoff= pd.read_csv("E:/NEON_iso_MI/Flux and other data verifty/NEON_time_correction.csv")
# We used the windspeed dataset to examine if the time label is correct or not
# start_datetime = pd.to_datetime("2017-07-01 00:00:00")
# end_datetime = pd.to_datetime("2017-07-31 23:30:00")
for i in fileList:
    # site name 
    
    site_name = i.split("-")[1]
    print(timeoff.loc[timeoff['Site'] == site_name, "Time_off_UTC"])
    tf = timeoff.loc[timeoff['Site'] == site_name, "Time_off_UTC"].values[0]
    site_i = ws_site[(ws_site['SITE']== site_name)&(ws_site['DESC'] == "2DWSD_30min")].copy()
    tower_top_id = site_i["VER"].idxmax()
    link = site_i.loc[tower_top_id, "PATH"]
    ws_data = pd.read_csv(link,usecols= ['startDateTime', 'windSpeedMean', "windSpeedFinalQF"])
    ws_data["startDateTime"] = pd.to_datetime(ws_data["startDateTime"], format= "%Y-%m-%dT%H:%M:%SZ")
    ws_data = ws_data.rename(columns={"startDateTime": "Date Time",
                                      'windSpeedMean': "wspd", 
                                      "windSpeedFinalQF": "wspd_flag"})
    ws_data.loc[ws_data["wspd_flag"] != 0, "wspd"] = np.nan
    
    ws_start = ws_data["Date Time"].min() # start of windspeed obs.
    ws_end = ws_data["Date Time"].max()   # end of windspeed obs.
    # now read in Rich's dataset 
    
    data = pd.read_csv("E:/NEON_iso_MI/Flux and other data/" + i, 
                        delimiter = '\t',
                        usecols= ['Date Time', 'wspd'],
                        low_memory = False,  skiprows=[1])
    data["Date Time"] = pd.to_datetime(data["Date Time"])
    data.loc[data["wspd"] == -9999, "wspd"] = np.nan
   # let's now shift the labels, how do we shift the labels in
   # order to match the time from neon ?
   
    # shift the datalabels first
    before_adjust_data = data[data["Date Time"].isin(ws_data["Date Time"])].copy()
    # correct the time 
    data["Date Time"] = data["Date Time"] + pd.Timedelta(tf, unit= "hours")
    time_adjust_data = data[data["Date Time"].isin(ws_data["Date Time"])].copy()
    fig, ax = plt.subplots(figsize = (14, 3))
    
    ax.plot(ws_data["Date Time"], ws_data["wspd"],
            label = "NEON WS data", ls = "solid", c= "black")
    ax.plot(before_adjust_data["Date Time"], 
            before_adjust_data["wspd"],
            label = "Rich WS data", ls = "dashed", c= "green")
    ax.plot(time_adjust_data["Date Time"], 
            time_adjust_data["wspd"], 
            label = "Rich correct T WS data", ls = "dashed", c= "red")
    ax.set_title(site_name + " UTC - " + str(tf))
    ax.legend(loc = "upper right")  
    fig.savefig(f"../Figures/WS_time_correction_plots/{site_name}_time_correction.png")
    bool_after_adjustment = np.array_equal(ws_data["wspd"], 
                                           time_adjust_data["wspd"],
                                           equal_nan=True)
    assert bool_after_adjustment, "Time adjustment/correct break doube check!"















