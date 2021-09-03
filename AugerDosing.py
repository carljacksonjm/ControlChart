#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import math

from PlottingFunctions import *

from ControlChartsModule import *

jm_colors_dict = {
    'JM Blue':'#1e22aa',
    'JM Purple':'#e50075',
    'JM Cyan':'#00abe9','JM Green':'#9dd3cb',
    'JM Magenta':'#e3e3e3',
    'JM Light Grey':'#575756',
    'JM Dark Grey':'#6e358b'
}
jm_colors_list = list(jm_colors_dict.values())
marker_list = ["o","v","D","x","X","<","8","s","^","p","P","*","h","H","+","d","|","_"]


#%% Data Import, Exploration and Cleaning
batch_path = "C:/Users/JacksC01/OneDrive - Johnson Matthey/Documents/PPI/Data"
df_raw = read_data(batch_path,
         "AugerMassDosing.xlsx",
                   tab_name = "Auger Mass Dosing", skip_r = 0)

fig_titles = "AugerDosing"

print(df_raw.shape)
# print(df_raw.info())
# print(df_raw.describe())
print(f"Columns:\n{df_raw.columns.to_list()}")

df = df_raw.loc[df_raw["Delete"] == False]

df["Date"] = pd.to_datetime(df["Date"], dayfirst = True).dt.date
df = df.sort_values(by=['Date','Sample Number'], ascending=True) # "Nest No.",
df.dropna(subset = ["Powder mass"], inplace = True)

df.rename(columns = {'Rig':"Nest",
                    'Powder mass':"Shot Mass (g)"}, inplace = True)

df[["Nest","Auger","Funnel"]] = df[["Nest","Auger","Funnel"]].astype(str)
for col in ["Nest","Auger","Funnel"]:
    df[col] = df[col].apply(lambda x: x[-1])
    
df["Nest/Aug/Fun"] = df["Nest"] +"/"+ df["Auger"] + "/" + df["Funnel"]


# In[ ]:

# for setup in df["Rig/Aug/Fun"].unique():
setup = "1/2/2" # df["Nest/Aug/Fun"].unique()
df_all = df.loc[df["Nest/Aug/Fun"]==setup]
setup_stddev = np.std(df_all["Shot Mass (g)"], ddof = 0) # divisor = n-1
control_tests = [1,2,3]

# print(f"""
# ----------------------
# Setup (Nest/Aug/Fun): {setup}
# {df_all["Shot Mass (g)"].describe()}""")

trials = df_all["Trial Name"].unique()
for trial in trials[:]:

    df_sample = df_all.loc[df_all["Trial Name"] == trial]
    max_date = df_sample.Date.max()

    nest = df_sample.Nest.mode().values[0]
    auger = df_sample.Auger.mode().values[0]
    funnel = df_sample.Funnel.mode().values[0]

    # sample dataset
    sample_data = np.array(df_sample["Shot Mass (g)"])
    
    # plotting
    title = "Trial {0} - Setup N{1}A{2}F{3}".format(trial.strip(), nest, auger, funnel)
    figure_name = "ControlChart_{0}_N{1}A{2}F{3}".format(trial.replace(" ",""), nest, auger, funnel)
    
    ControlChart.mr_graph(sample_data, x_string = 'Auger Shot Sample', y_string = 'Shot Mass - Moving Range (g)',
                               title_string = title, fig_name = figure_name)
    
    ControlChart.control_graph(sample_data, setup_stddev, x_string = 'Auger Shot Sample', y_string = 'Shot Mass (g)',
                               p_control_tests = control_tests,
                               title_string = title, fig_name = figure_name, y_lim = None)
    


# D3 - D4 lower - uppder control limit constants
# absolute MR
# mr chart first

# .1 sigma shift would take 10000s

# x bar t chart (central limit theorem)
# I-MR chart is single points
# UCL is equidistant from xbar as LCL

# print("Assuming normal distribution, hence 99.73% of observations are with +-3 std dev of the mean...\n")
# Q-Q plot|Freq distribution|Normality test
# (shapiro-wilk test): small p -> accept null hypothesis that it is normal

#%% swarm plot of shot mass as func of nest, funnel, auger
# def sns_swarm(dataFrame, x_label, y_label, color_var, 
#         hue_label = None, col_label = None,  orient_var = "v",
#         fig_name_var = None, legend_var = False, axis_lim = None, title_var = None):
    
#     ax = sns.catplot(x = x_label, y = y_label, data = dataFrame, hue = hue_label, col = col_label,
#                          dodge = False, kind = "violin",
#                          palette = color_var, orient = orient_var,
#                          legend = False) # linewidth = 1., edgecolor = "black",
    
#     ax.fig.set_size_inches(10,8)
    
    
#     if axis_lim and orient_var == "v":
#         plt.ylim(axis_lim[0], axis_lim[1])
    
#     elif axis_lim:
#         plt.xlim(axis_lim[0], axis_lim[1])
        
#     if legend_var == True:
#         plt.legend(title = hue_label, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#     else:
#         plt.legend([],[], frameon=False)
    
#     if title_var:
#         plt.title(title_var)
    
#     plt.tight_layout()
    
#     if fig_name_var:
#         plt.savefig(fig_name_var+".png", facecolor='w')

#     plt.show()

# sns_swarm(df, "Shot Mass (g)", "Nest", jm_colors_list[:], 
#         hue_label = "Auger", col_label = "Funnel",  orient_var = "h",
#         fig_name_var = "AugerShotTest_Violin", legend_var = True, axis_lim = None, title_var = None)

#%% correlation plot
# correlation check of a/n/f
# corrMatrix_all = df.corr()
# features = ["Final BP (mBar)",'Dosing Time (s)']
# corr_limit_val = 0.4
# corrMatrix = corrMatrix_all[features]
# filteredDf = corrMatrix[(corrMatrix.abs() >= corr_limit_val) & (corrMatrix.abs() !=1.000)]
# filteredDf.dropna(how = "all", inplace = True)
# plt.figure(figsize=(10,10))
# sns.heatmap(filteredDf, annot=True, cmap="Reds")
# plt.show()