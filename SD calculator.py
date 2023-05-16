# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:21:23 2022

@author: Randy
"""
#mean / standard deviation calculator for cut off time per group.
import math
import pandas as pd

total_diff = 0
#insert file with all runtime data for a specific group.
df = pd.read_excel ('C:\\LabProject\\LadderProject\\Data\\excel\\Parameters_3.0WBSCTR28.xlsx')
df = df[df['time run'] <= 15000].reset_index()

Total_runtime = sum(df["time run"])
mean_runtime = Total_runtime / len(df["time run"])
for i in range (len(df["time run"])):
    summation = (df["time run"][i] - mean_runtime)**2
    total_diff = total_diff + summation    
average_diff = total_diff/len(df["time run"])
sq_av_diff = math.sqrt(average_diff)
mean_cut_off_time = mean_runtime + (2*sq_av_diff)
print(mean_cut_off_time)