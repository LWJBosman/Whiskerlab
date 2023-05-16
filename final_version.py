from Python_Lucas import database
from Python_Lucas import logger

# import os.path
# import sys
from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import gridplot
from bokeh.models import Range1d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# connect with the database (sdf file)
class Main(object):
    def __init__(self, path1, passw, path2=None):
        self.logger = logger.Logger("logs")
        self.db1 = database.Database(path1, passw, self.logger)
        if path2:
            self.db2 = database.Database(path2, passw, self.logger)

def find_nearest (array,value):
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return array[idx]
def remove_duplicates(lst):
  # Create a new list with only the unique values from the original list
  return list(set(lst))
# path to database file (sdf)
db_filepath = "C:\LabProject\LadderProject\Data\Data_Videos\ErasmusLadder2.sdf"
db_passw = "mypassword"

main = Main(db_filepath, db_passw)
db = main.db1

# load file with sessions for the different groups and make list for each group with session ids


session_group = pd.read_excel("C:\\LabProject\\LadderProject\\Data\\sessions_per_group.xlsx", header = 0, index_col=None)


# VA = session_group["VA"].dropna()
SCA1MUT = session_group["SCA1MUT"].dropna()
SCA1CTR = session_group["SCA1CTR"].dropna()
ercc1_mut = session_group["Ercc1-mut"].dropna()
ercc1_ctr = session_group["Ercc1-ctr"].dropna()
WBSCTR2 = session_group["WBSCTR2"].dropna()
WBSMUT2 = session_group["WBSMUT2"].dropna()
WBSCTR28 = session_group["WBSCTR28"].dropna()
WBSMUT28 = session_group["WBSMUT28"].dropna()

# dataframe to save general statistics about filtering
column_filter = ['runs', 'runs state sequence', 'runs limb detection']
# dataframe to save variables for further analysis
column_analysis = ['session id', 'state sequence' ,'run id', 'session nr', 'subject id','direction', 'timerun', 'number of steps', 'even steps', 'odd steps', 'step 2', 'step 4', 'step 6', 'swing phase lf', 'swing phase lh', 'swing phase rf', 'swing phase rh', 'stance phase lf', 'stance phase lh', 'stance phase rf', 'stance phase rh', 'lag lf/rh', 'lag rf/lh', 'support diagonal', 'support girdle', 'support lateral', 'support 0 paws', 'support 1 paw', 'support 3 paws', 'support 4 paws', 'high rungs front', 'high rungs hind', 'low rungs front', 'low rungs hind','mean touchtime']
df_analysis = pd.DataFrame(columns = column_analysis)
column_front = ['session id', 'subject id','run id', 'session nr', 'direction', 'total steps', 'h-h 2', 'h-h 4', 'h-h other', 'h-l', 'l-h', 'l-l']
df_front = pd.DataFrame(columns = column_front)
skip_run = []
skip_sequence = []
# touches undetermined and not classified
column_touches = ['session id', 'run id', 'session nr', 'total touches', 'lf', 'lh', 'rf', 'rh', 'undetermined', 'not classified']
df_stats_touches = pd.DataFrame(columns = column_touches)

# list to save lag and support for every run
all_lag = []
all_overlap = []
all_support = []
alltouches=[]
allrungs=[]
types=[]
allsessions=[]
allsubjects=[]
allsides=[]
alldirections=[]
allpaws=[]
allbegintouches=[]
allendtouches=[]
allrunid=[]
#%% cut off runtime

group = 'ercc1_mut'
group1 = ercc1_mut
 

# Cut_off_runtime for each group (experimentally determined)
if group == 'ercc1_mut':
   cut_off_runtime = 2804
elif group == 'ercc1_ctr':
     cut_off_runtime = 2298
elif group== 'SCA1CTR':
    cut_off_runtime = 6150
elif group== 'SCA1MUT':
    cut_off_runtime = 6555   
elif group== 'WBSCTR2':
    cut_off_runtime = 4757   
elif group== 'WBSMUT2':
    cut_off_runtime = 5163
elif group== 'WBSCTR28':
    cut_off_runtime = 4757   
elif group== 'WBSMUT28':
    cut_off_runtime = 5163        
#%% start 

for ss in range(len(group1)):
    "Filter runs based on runtime (find cutoff)"
    min_runtime = 600        # minimum runtime
    max_runtime = 15000   # maximum runtime
    session = group1[ss]     # session id of experiment
    query = "select trial.id, subject_id, statesequence, trialtype, session_id, state, state_duration, sessionnr from trial join state on trial.id = state.trial_id join session on trial.session_id = session.id where state = 6 and state_duration > %s and state_duration < %s and session_id = %s;" %(min_runtime, max_runtime, session)
    data = db.query(query)
    column = ["run id", "subject id", "state sequence", "runtype", "session id", "state", "state time", "session nr"]
    df_runs = pd.DataFrame(np.column_stack(data), columns = column)

    #use all runs also the escape runs
    string = df_runs["state sequence"]
    for ii in range(len(string)):
        if "1-3-6" in string[ii]:
            df_sequence1 = df_runs[df_runs["state sequence"]!= "0"]
            list_runs = list(df_sequence1["run id"]) 
        elif "1-3-4-6" in string[ii]:
            df_sequence2 = df_runs[df_runs["state sequence"]!= "0"]
            list_runs = list(df_sequence2["run id"]) 
    skip_sequence.append(df_runs[~df_runs["run id"].isin(list_runs)])
    list_runs = remove_duplicates(list_runs)
    for a in range(len(list_runs)):
        # Get touch data for approved runs
        "Collect data: already filter on total time of runs, only collect data from touches in state 6 and >0"
        run = list_runs[a]
        query = "select  subject_id, side, rung, touch_begin, touch_end, trial.id as trial_id, trialnr, touch.id as touch_id, state, state_duration, directiontoend, session_id, statesequence, sessionnr from trial join touch on trial.id = touch.trial_id join state on trial.id = state.trial_id join session on trial.session_id = session.id where state = 6  and trial.id = %s;" %run
        data = db.query(query)
        column = ["subject id","side", "rung", "begin touch", "end touch", "run id", "run number", "touch id", "state", "state duration", "direction", "session id", "statesequence", "session nr"]
        df = pd.DataFrame(np.column_stack(data), columns=column)
        # all values are strings -> change all columns except statesequence and session nr to numeric values
        for i in range(0,len(df.columns)-2):
            df.iloc[:, i] = pd.to_numeric(df.iloc[:, i])
        if len(df)<5:
            continue
        #calculate the touch time for each touch & only use the touches between and including rung 7 and 30 
        df["touch"] = df["end touch"] - df["begin touch"]
        df = df[(df["rung"] >= 7) & (df["rung"] <= 30)]
        
        #%% graph all touches
        # fig1 = figure(title = "Time vs rung | all touches | run %s" %run)
        # for k in range(len(df)):
        #     fig1.line([df["begin touch"].iloc[k], df["end touch"].iloc[k]], df["rung"].iloc[k], line_color = 'black')
        # fig1.xaxis.axis_label = 'Time [ms]'
        # fig1.yaxis.axis_label = 'Rung'
        # fig1.xaxis.axis_label_text_font_size = '20px'
        # fig1.xaxis.major_label_text_font_size = '16px'
        # fig1.yaxis.axis_label_text_font_size = '20px'
        # fig1.yaxis.major_label_text_font_size = '16px'
        # fig1.title.text_font_size = '16px'
        
        # Rewrite rungs so every run is from 1 -> 37
        if (df["direction"] == 0).all(): # 37 -> 1
            df["rung"] = (df["rung"] - 38)*-1
            # change side so that info left paw is interpreted as left
            for s in range(len(df)):
                if df["side"].iloc[s] == 0:
                    df["side"].iloc[s] = 1
                elif df["side"].iloc[s] == 1:
                    df["side"].iloc[s] = 0
        #%% filter touches
        "Filter the collected data: minimum and maximum duration of a touch"
        "We put the minimum at 32 because of the trough in the touch histogram, the maximum is a bit more arbitrary and depends on what you want to include and exclude in your research"
        touch_min = 34         # minimum touch time [ms]
        touch_max = 800        # maximum touch time [ms]
        min_swing = 15         # minimum time between swings of a paw [ms]
        max_step = 8            # maximum size of step
        split_time = 1.5        # value to multiply with mean time to determine if touch needs to be split
        df_touch_min = df[df["touch"] >= touch_min].copy()
        df_touches = df_touch_min[df_touch_min["touch"] <= touch_max].copy()
        mean = np.mean(df_touches["touch"])
        
        #preventing escape data touches to enter the analysis
        if len(df_touches)<5:
            continue

        time_end = max(df_touches["end touch"])
        time_begin = find_nearest(df_touches["begin touch"], (time_end-cut_off_runtime))
        df_touches = df_touches.sort_values(by=["begin touch"])
        index_number = list(df_touches["begin touch"]).index(time_begin)
        
        df_touches = df_touches[index_number:]
        state_dur = max(df_touches["state duration"])
        df_touches = df_touches[df_touches["state duration"] == state_dur]
        if len(df_touches)>150:
            continue
        gait_cycle = pd.DataFrame(0, index=['lf', 'rh', 'rf', 'lh'], columns= range(time_begin, time_end + 1))
        if (df_touches["touch"] > touch_max).any():
            skip_run.append([run, 'long touch', session])
            continue
        if len(df)<5:
            continue
        # current will represent the current/last touches of each paw
        current = pd.DataFrame(0, columns = df_touches.columns, index=('lf','lh','rf','rh'))
        touches = len(df_touches)
        # empty lists that will be filled with the touch ids of touches assigned to that paw and keep track of touches that are split
        lf = []
        lh = []
        rf = []
        rh = []
        undetermined = []
        split = []
        
        #solves 1-3-4-6-4-6-4-6 runs to clutter up dataset
        # state_dur = max(df_touches["state duration"])
        # df_touches = df_touches[df_touches["state duration"] == state_dur]
        # if len(df_touches)>150:
        #     continue
        left = df_touches[df_touches["side"] == 0].copy()
        right = df_touches[df_touches["side"] == 1].copy()
        # remove all touches except 2 largest on rung with more touches than 2
        # left side
        df_multi = left["rung"].value_counts() >= 3
        rung_list = df_multi.index[df_multi == True].tolist()
        for ru in range(len(rung_list)):
            df_multitouch = df_touches[df_touches["rung"] == rung_list[ru]].copy()
            n = len(df_multitouch)
            df_remove = df_multitouch.nsmallest(n-2, "touch")
            df_touches = df_touches[~df_touches["touch id"].isin(df_remove["touch id"])]
        # right side
        df_multi = right["rung"].value_counts() >= 3
        rung_list = df_multi.index[df_multi == True].tolist()
        for ru in range(len(rung_list)):
            df_multitouch = df_touches[df_touches["rung"] == rung_list[ru]].copy()
            n = len(df_multitouch)
            df_remove = df_multitouch.nsmallest(n-2, "touch")
            df_touches = df_touches[~df_touches["touch id"].isin(df_remove["touch id"])]
        #%% limb detection
        # give 2 touches on same rung and same side priority
        # assign first touch to front paw, second touch to hind paw

        left = df_touches[df_touches["side"] == 0].copy()
        right = df_touches[df_touches["side"] == 1].copy()
        df_double = left["rung"].value_counts() == 2
        double_rungs_left = df_double.index[df_double == True].tolist()
        for d in range(len(double_rungs_left)):
            df_doubletouch = left[left["rung"] == double_rungs_left[d]].copy()
            lf.append(min(df_doubletouch["touch id"]))
            lh.append(max(df_doubletouch["touch id"]))
        df_double = right["rung"].value_counts() == 2
        double_rungs_right = df_double.index[df_double == True].tolist()
        for d in range(len(double_rungs_right)):
            df_doubletouch = right[right["rung"] == double_rungs_right[d]].copy()
            rf.append(min(df_doubletouch["touch id"]))
            rh.append(max(df_doubletouch["touch id"]))
        list_touches = lf + lh + rf + rh
        # dataframe for each paw with the determined double touches
        df_lf = df_touches[df_touches["touch id"].isin(lf)].sort_values("touch id").copy()
        df_lh = df_touches[df_touches["touch id"].isin(lh)].sort_values("touch id").copy()
        df_rf = df_touches[df_touches["touch id"].isin(rf)].sort_values("touch id").copy()
        df_rh = df_touches[df_touches["touch id"].isin(rh)].sort_values("touch id").copy()
        
        
        # split touches based on mean touch time for each paw
        mean_lf = np.mean(df_lf["touch"])
        mean_lh = np.mean(df_lh["touch"])
        mean_rf = np.mean(df_rf["touch"])
        mean_rh = np.mean(df_rh["touch"])

        
        if pd.isna(mean_lf) == True or pd.isna(mean_lh) == True:
            ratio_left = 0.5
        else:
            ratio_left = mean_lf/(mean_lf + mean_lh)
        if pd.isna(mean_rf) == True or pd.isna(mean_rh) == True:
            ratio_right = 0.5
        else:
            ratio_right = mean_rf/(mean_rf + mean_rh)
        
        # single touches > 1.9*mean will be split, but if there is overlapping on another rung on the same side -> no split -> front and hind touch
        # check here if single touches have overlapping touches on other rung -> list of touches to not split
        no_split = []
        for ns in range(1,len(left)):
            if left["rung"].iloc[ns-1] not in double_rungs_left and left["rung"].iloc[ns] not in double_rungs_left:
                # both single touches (otherwise it will be dealt with further up in the code)        
                # if left["begin touch"].iloc[ns] < left["end touch"].iloc[ns-1]:
                    # touch overlaps with previous touch
                    if left["touch"].iloc[ns-1] > split_time*mean:
                        # previous touch could be split
                        if left["rung"].iloc[ns] < left["rung"].iloc[ns-1]:
                            # touch lower rung than previous touch
                            # consider both touches as single touches -> no split previous touch
                            no_split.append(left["touch id"].iloc[ns-1])
                    elif left["touch"].iloc[ns] > split_time*mean:
                        # current touch could be split
                        if left["rung"].iloc[ns] < left["rung"].iloc[ns-1]:
                            # touch lower rung than previous touch
                            # consider both touches as single touches -> not split
                            no_split.append(left["touch id"].iloc[ns])
        for ns in range(1,len(right)):
            if right["rung"].iloc[ns-1] not in double_rungs_right and right["rung"].iloc[ns] not in double_rungs_right:
                # both single touches (otherwise it will be dealt with further up in the code)        
                # if right["begin touch"].iloc[ns] < right["end touch"].iloc[ns-1]:
                    # touch overlaps with previous touch
                    if right["touch"].iloc[ns-1] > split_time*mean:
                        # previous touch could be split
                        if right["rung"].iloc[ns] < right["rung"].iloc[ns-1]:
                            # touch lower rung than previous touch
                            # consider both touches as single touches -> no split previous touch
                            no_split.append(right["touch id"].iloc[ns-1])
                    elif right["touch"].iloc[ns] > split_time*mean:
                        # current touch could be split
                        if right["rung"].iloc[ns] < right["rung"].iloc[ns-1]:
                            # touch lower rung than previous touch
                            # consider both touches as single touches -> not split
                            no_split.append(right["touch id"].iloc[ns])
        
        for i in range(len(df_touches)):
            touch = df_touches.iloc[i].copy()          # touch that needs to be identified
            # check if first left touch to be identified
            if current.loc["lf", "rung"] == 0 and current.loc["lh", "rung"] == 0 and touch["side"] == 0:
                if touch["touch id"] in lf:
                    # touch is identified double touch
                    current.loc["lf"] = touch
                    current.loc["lh", "rung"] = 1
                else:
                    # single touch -> check if needs to be split
                    if touch["touch"] >= split_time*mean and touch["touch id"] not in no_split:
                        touch_lf = df_touches.iloc[i].copy()
                        touch_lh = df_touches.iloc[i].copy()
                        time_lf = round(touch.loc["touch"]*ratio_left)
                        time_lh = touch.loc["touch"] - time_lf
                        ind = touch.name
                        # adjust touch front paw
                        touch_lf.loc["end touch"] = touch.loc["begin touch"] + time_lf
                        touch_lf.loc["touch id"] = touch.loc["touch id"] + 0.1
                        touch_lf.loc["touch"] = time_lf
                        # adjust touch hind paw
                        touch_lh.loc["begin touch"] = touch.loc["end touch"] - time_lh
                        touch_lh.loc["touch id"] = touch.loc["touch id"] + 0.2
                        touch_lh.loc["touch"] = time_lh
                        # put new touches in dataframe touches and current situation
                        current.loc["lf"] = touch_lf
                        current.loc["lh"] = touch_lh
                        df_touches.loc[ind] = touch_lf
                        df_touches = df_touches.append(touch_lh)
                        lf.append(touch_lf.loc["touch id"])
                        lh.append(touch_lh.loc["touch id"])
                        split.extend((touch_lf.loc["touch id"], touch_lh.loc["touch id"]))
                    else:
                        lf.append(touch["touch id"])
                        current.loc["lf"] = touch
                        current.loc["lh", "rung"] = 1
                        
            elif current.loc["rf", "rung"] == 0 and current.loc["rh", "rung"] == 0 and touch["side"] == 1:
                # first touch on right side
                if touch["touch id"] in rf:
                    # touch already identified as double touch
                    current.loc["rf"] = touch
                    current.loc["rh", "rung"] = 1
                else:
                    # single touch -> check if needs to be split
                    if touch["touch"] >= split_time*mean and touch["touch id"] not in no_split:
                        touch_rf = df_touches.iloc[i].copy()
                        touch_rh = df_touches.iloc[i].copy()
                        time_rf = round(touch.loc["touch"]*ratio_right)
                        time_rh = touch.loc["touch"] - time_rf
                        ind = touch.name
                        # adjust touch front paw
                        touch_rf.loc["end touch"] = touch.loc["begin touch"] + time_rf
                        touch_rf.loc["touch id"] = touch.loc["touch id"] + 0.1
                        touch_rf.loc["touch"] = time_rf
                        # adjust touch hind paw
                        touch_rh.loc["begin touch"] = touch.loc["end touch"] - time_rh
                        touch_rh.loc["touch id"] = touch.loc["touch id"] + 0.2
                        touch_rh.loc["touch"] = time_rh
                        # put new touches in dataframe touches and current situation
                        current.loc["rf"] = touch_rf
                        current.loc["rh"] = touch_rh
                        df_touches.loc[ind] = touch_rf
                        df_touches = df_touches.append(touch_rh)
                        rf.append(touch_rf.loc["touch id"])
                        rh.append(touch_rh.loc["touch id"])
                        split.extend((touch_rf.loc["touch id"], touch_rh.loc["touch id"]))
                    else:
                        rf.append(touch["touch id"])
                        current.loc["rf"] = touch
                        current.loc["rh", "rung"] = 1
            elif touch["touch id"] in list_touches:
                # not first touch -> check if touch already identified as one of the double touches
                if touch["touch id"] in lf:
                    if touch["begin touch"] - current.loc["lf", "end touch"] > min_swing:
                        current.loc["lf"] = touch
                    elif current.loc["lf", "touch id"] not in list_touches:
                        # if the current touch not one of the double touches
                        # check if the single touch overlaps with the front double touch -> if so move current touch to hind paw
                        if current.loc["lf", "begin touch"] > current.loc["lh", "end touch"]:
                            current.loc["lh"] = current.loc["lf"]
                            current.loc["lf"] = touch
                            lh.append(current.loc["lh", "touch id"])
                            lf.remove(current.loc["lh", "touch id"])
                        else:
                            undetermined.append(current.loc["lf", "touch id"])
                            lf.remove(current.loc["lf", "touch id"])
                            current.loc["lf"] = touch
                if touch["touch id"] in lh:
                    if touch["begin touch"] - current.loc["lh", "end touch"] > min_swing:
                        if touch["rung"] > current.loc["lh", "rung"]:
                            current.loc["lh"] = touch
                        else:
                            undetermined.append(touch["touch id"])
                            lh.remove(touch["touch id"])
                    elif current.loc["lh", "touch id"] not in list_touches:
                        # if the current touch not one of the double touches
                        # check if the single touch overlaps with the hind double touch -> if so undetermined touch
                        undetermined.append(current.loc["lh", "touch id"])
                        lh.remove(current.loc["lh", "touch id"])
                        current.loc["lh"] = touch
                if touch["touch id"] in rf:
                    if touch["begin touch"] - current.loc["rf", "end touch"] > min_swing:
                        current.loc["rf"] = touch
                    elif current.loc["rf", "touch id"] not in list_touches:
                        # if the current touch not one of the double touches
                        # check if the single touch overlaps with the front double touch -> if so move current touch to hind paw
                        if current.loc["rf", "begin touch"] > current.loc["rh", "end touch"]:
                            current.loc["rh"] = current.loc["rf"]
                            current.loc["rf"] = touch
                            rh.append(current.loc["rh", "touch id"])
                            rf.remove(current.loc["rh", "touch id"])
                        else:
                            undetermined.append(current.loc["rf", "touch id"])
                            rf.remove(current.loc["rf", "touch id"])
                            current.loc["rf"] = touch
                if touch["touch id"] in rh:
                    if touch["begin touch"] - current.loc["rh", "end touch"] > min_swing:
                        if touch["rung"] > current.loc["rh", "rung"]:
                            current.loc["rh"] = touch
                        else:
                            undetermined.append(touch["touch id"])
                            rh.remove(touch["touch id"])
                    elif current.loc["rh", "touch id"] not in list_touches:
                        # if the current touch not one of the double touches
                        # check if the single touch overlaps with the hind double touch -> if so undetermined touch
                        undetermined.append(current.loc["rh", "touch id"])
                        rh.remove(current.loc["rh", "touch id"])
                        current.loc["rh"] = touch
            else:
                # looked at first touches and double touches -> only single touch remains
                # single touch longer than 1.5* mean -> split touch
                if touch["touch"] >= split_time*mean and touch["touch id"] not in no_split:
                    if touch["side"] == 0:
                        touch_lf = df_touches.iloc[i].copy()
                        touch_lh = df_touches.iloc[i].copy()
                        time_lf = round(touch.loc["touch"]*ratio_left)
                        time_lh = touch.loc["touch"] - time_lf
                        ind = touch.name
                        # adjust touch front paw
                        touch_lf.loc["end touch"] = touch.loc["begin touch"] + time_lf
                        touch_lf.loc["touch id"] = touch.loc["touch id"] + 0.1
                        touch_lf.loc["touch"] = time_lf
                        # adjust touch hind paw
                        touch_lh.loc["begin touch"] = touch.loc["end touch"] - time_lh
                        touch_lh.loc["touch id"] = touch.loc["touch id"] + 0.2
                        touch_lh.loc["touch"] = time_lh
                        # check if paws are free
                        if touch_lf.loc["begin touch"] - current.loc["lf", "end touch"] > min_swing:
                            # front paw free
                            if touch_lh.loc["begin touch"] - current.loc["lh", "end touch"] > min_swing:
                                # hind paw free
                                current.loc["lf"] = touch_lf
                                df_touches.loc[ind] = touch_lf
                                lf.append(touch_lf.loc["touch id"])
                                current.loc["lh"] = touch_lh
                                df_touches = df_touches.append(touch_lh)
                                lh.append(touch_lh.loc["touch id"])
                                split.extend((touch_lf.loc["touch id"], touch_lh.loc["touch id"]))
                            else:
                                # hind paw not free
                                if current.loc["lh", "touch id"] not in list_touches:
                                    # previous touch not double touch -> make it undetermined and split this one
                                    current.loc["lf"] = touch_lf
                                    df_touches.loc[ind] = touch_lf
                                    lf.append(touch_lf.loc["touch id"])
                                    undetermined.append(current.loc["lh", "touch id"])
                                    lh.remove(current.loc["lh", "touch id"])
                                    current.loc["lh"] = touch_lh
                                    df_touches = df_touches.append(touch_lh)
                                    lh.append(touch_lh.loc["touch id"])
                                    split.extend((touch_lf.loc["touch id"], touch_lh.loc["touch id"]))
                                else:
                                    undetermined.append(touch["touch id"])
                        else:
                            # front paw not free
                            if current.loc["lf", "touch id"] not in list_touches:
                                # previous touch not double touch -> make it undetermined and split this one
                                if touch_lh.loc["begin touch"] - current.loc["lh", "end touch"] > min_swing:
                                    # hind paw free
                                    undetermined.append(current.loc["lf", "touch id"])
                                    lf.remove(current.loc["lf", "touch id"])    
                                    current.loc["lf"] = touch_lf
                                    df_touches.loc[ind] = touch_lf
                                    lf.append(touch_lf.loc["touch id"])
                                    current.loc["lh"] = touch_lh
                                    df_touches = df_touches.append(touch_lh)
                                    lh.append(touch_lh.loc["touch id"])
                                    split.extend((touch_lf.loc["touch id"], touch_lh.loc["touch id"]))
                                else:
                                    # hind paw not free
                                    if current.loc["lh", "touch id"] not in list_touches:
                                        # previous touch not double touch -> make it undetermined and split this one
                                        undetermined.append(current.loc["lf", "touch id"])
                                        lf.remove(current.loc["lf", "touch id"])
                                        current.loc["lf"] = touch_lf
                                        df_touches.loc[ind] = touch_lf
                                        lf.append(touch_lf.loc["touch id"])
                                        undetermined.append(current.loc["lh", "touch id"])
                                        lh.remove(current.loc["lh", "touch id"])
                                        current.loc["lh"] = touch_lh
                                        df_touches = df_touches.append(touch_lh)
                                        lh.append(touch_lh.loc["touch id"])                                            
                                        split.extend((touch_lf.loc["touch id"], touch_lh.loc["touch id"]))
                                    else:
                                        undetermined.append(touch["touch id"])
                            else:
                                undetermined.append(touch["touch id"])
                    elif touch["side"] == 1:
                        touch_rf = df_touches.iloc[i].copy()
                        touch_rh = df_touches.iloc[i].copy()
                        time_rf = round(touch.loc["touch"]*ratio_right)
                        time_rh = touch.loc["touch"] - time_rf
                        ind = touch.name
                        # adjust touch front paw
                        touch_rf.loc["end touch"] = touch.loc["begin touch"] + time_rf
                        touch_rf.loc["touch id"] = touch.loc["touch id"] + 0.1
                        touch_rf.loc["touch"] = time_rf
                        # adjust touch hind paw
                        touch_rh.loc["begin touch"] = touch.loc["end touch"] - time_rh
                        touch_rh.loc["touch id"] = touch.loc["touch id"] + 0.2
                        touch_rh.loc["touch"] = time_rh
                        # check if paws are free
                        if touch_rf.loc["begin touch"] - current.loc["rf", "end touch"] > min_swing:
                            # front paw free
                            if touch_rh.loc["begin touch"] - current.loc["rh", "end touch"] > min_swing:
                                # hind paw free
                                current.loc["rf"] = touch_rf
                                df_touches.loc[ind] = touch_rf
                                rf.append(touch_rf.loc["touch id"])
                                current.loc["rh"] = touch_rh
                                df_touches = df_touches.append(touch_rh)
                                rh.append(touch_rh.loc["touch id"])
                                split.extend((touch_rf.loc["touch id"], touch_rh.loc["touch id"]))
                            else:
                                # hind paw not free
                                if current.loc["rh", "touch id"] not in list_touches:
                                    # previous touch not double touch -> make it undetermined and split this one
                                    current.loc["rf"] = touch_rf
                                    df_touches.loc[ind] = touch_rf
                                    rf.append(touch_rf.loc["touch id"])
                                    undetermined.append(current.loc["rh", "touch id"])
                                    rh.remove(current.loc["rh", "touch id"])
                                    current.loc["rh"] = touch_rh
                                    df_touches = df_touches.append(touch_rh)
                                    rh.append(touch_rh.loc["touch id"])
                                    split.extend((touch_rf.loc["touch id"], touch_rh.loc["touch id"]))
                                else:
                                    undetermined.append(touch["touch id"])
                        else:
                            # front paw not free
                            if current.loc["rf", "touch id"] not in list_touches:
                                # previous touch not double touch -> make it undetermined and split this one
                                if touch_rh.loc["begin touch"] - current.loc["rh", "end touch"] > min_swing:
                                    # hind paw free
                                    undetermined.append(current.loc["rf", "touch id"])
                                    rf.remove(current.loc["rf", "touch id"])
                                    current.loc["rf"] = touch_rf
                                    df_touches.loc[ind] = touch_rf
                                    rf.append(touch_rf.loc["touch id"])
                                    current.loc["rh"] = touch_rh
                                    df_touches = df_touches.append(touch_rh)
                                    rh.append(touch_rh.loc["touch id"])
                                    split.extend((touch_rf.loc["touch id"], touch_rh.loc["touch id"]))
                                else:
                                    # hind paw not free
                                    if current.loc["rh", "touch id"] not in list_touches:
                                        # previous touch not double touch -> make it undetermined and split this one
                                        undetermined.append(current.loc["rf", "touch id"])
                                        rf.remove(current.loc["rf", "touch id"])
                                        current.loc["rf"] = touch_rf
                                        df_touches.loc[ind] = touch_rf
                                        rf.append(touch_rf.loc["touch id"])
                                        undetermined.append(current.loc["rh", "touch id"])
                                        rh.remove(current.loc["rh", "touch id"])
                                        current.loc["rh"] = touch_rh
                                        df_touches = df_touches.append(touch_rh)
                                        rh.append(touch_rh.loc["touch id"])
                                        split.extend((touch_rf.loc["touch id"], touch_rh.loc["touch id"]))
                                    else:
                                        undetermined.append(touch["touch id"])
                            else:
                                undetermined.append(touch["touch id"])
                else:
                    if touch["side"] == 0:
                        if touch["begin touch"] - current.loc["lf", "end touch"] > min_swing:
                            # front paw free
                            if touch["begin touch"] - current.loc["lh", "end touch"] > min_swing:
                                # hind paw free
                                if touch["rung"] > current.loc["lf", "rung"]:
                                    # front paw because hind paw not before front paw
                                    lf.append(touch["touch id"])
                                    current.loc["lf"] = touch
                                elif touch["rung"] < current.loc["lf", "rung"] and touch["rung"] > current.loc["lh", "rung"]:
                                    lh.append(touch["touch id"])
                                    current.loc["lh"] = touch
                            else:
                                # hind paw not free
                                if touch["rung"] > current.loc["lf", "rung"]:
                                    # forwards step
                                    lf.append(touch["touch id"])
                                    current.loc["lf"] = touch
                                else:
                                    # backwards
                                    undetermined.append(touch["touch id"])
                        elif touch["begin touch"] - current.loc["lf", "end touch"] <= min_swing:
                            # front paw not free
                            if touch["begin touch"] - current.loc["lh", "end touch"] > min_swing:
                                # hind paw free
                                if touch["begin touch"] < current.loc["lf", "end touch"] and touch["rung"] > current.loc["lf", "rung"]:
                                    # touch is further along than current front paw -> can not be hind paw
                                    if current.loc["lf", "begin touch"] - current.loc["lh", "end touch"] > min_swing:
                                        # current touch front paw coulb be hind paw -> switch touch to hind paw
                                        lf.remove(current.loc["lf", "touch id"])
                                        current.loc["lh"] = current.loc["lf"]
                                        lh.append(current.loc["lh", "touch id"])
                                        lf.append(touch["touch id"])
                                        current.loc["lf"] = touch
                                    else:
                                        # hind paw not possible -> make current front touch undetermined and touch front paw
                                        lf.remove(current.loc["lf", "touch id"])
                                        undetermined.append(current.loc["lf", "touch id"])
                                        lf.append(touch["touch id"])
                                        current.loc["lf"] = touch
                                elif touch["rung"] > current.loc["lh", "rung"]:
                                    # not backwards
                                    lh.append(touch["touch id"])
                                    current.loc["lh"] = touch
                                else:
                                    # otherwise backwards step
                                    undetermined.append(touch["touch id"])
                            else:
                                undetermined.append(touch["touch id"])
                    if touch["side"] == 1:
                        if touch["begin touch"] - current.loc["rf", "end touch"] > min_swing:
                            # front paw free
                            if touch["begin touch"] - current.loc["rh", "end touch"] > min_swing:
                                # hind paw free
                                if touch["rung"] > current.loc["rf", "rung"]:
                                    # front paw because hind paw not before front paw
                                    rf.append(touch["touch id"])
                                    current.loc["rf"] = touch
                                elif touch["rung"] < current.loc["rf", "rung"] and touch["rung"] > current.loc["rh", "rung"]:
                                    rh.append(touch["touch id"])
                                    current.loc["rh"] = touch
                            else:
                                # hind paw not free
                                if touch["rung"] > current.loc["rf", "rung"]:
                                    # forwards step
                                    rf.append(touch["touch id"])
                                    current.loc["rf"] = touch
                                else:
                                    # backwards
                                    undetermined.append(touch["touch id"])
                        elif touch["begin touch"] - current.loc["rf", "end touch"] <= min_swing:
                            # front paw not free
                            if touch["begin touch"] - current.loc["rh", "end touch"] > min_swing:
                                # hind paw free
                                if touch["begin touch"] < current.loc["rf", "end touch"] and touch["rung"] > current.loc["rf", "rung"]:
                                    # touch is further along than current front paw -> can not be hind paw
                                    if current.loc["rf", "begin touch"] - current.loc["rh", "end touch"] > min_swing:
                                        # current touch front paw coulb be hind paw -> switch touch to hind paw
                                        rf.remove(current.loc["rf", "touch id"])
                                        current.loc["rh"] = current.loc["rf"]
                                        rh.append(current.loc["rh", "touch id"])
                                        rf.append(touch["touch id"])
                                        current.loc["rf"] = touch
                                    else:
                                        # hind paw not possible -> make current front touch undetermined and touch front paw
                                        rf.remove(current.loc["rf", "touch id"])
                                        undetermined.append(current.loc["rf", "touch id"])
                                        rf.append(touch["touch id"])
                                        current.loc["rf"] = touch
                                elif touch["rung"] > current.loc["rh", "rung"]:
                                    # not backwards
                                    rh.append(touch["touch id"])
                                    current.loc["rh"] = touch
                                else:
                                    # otherwise backwards step
                                    undetermined.append(touch["touch id"])
                            else:
                                undetermined.append(touch["touch id"])
        
        # dataframes for each paw
        df_lf = df_touches[df_touches["touch id"].isin(lf)].sort_values("touch id")
        df_lh = df_touches[df_touches["touch id"].isin(lh)].sort_values("touch id")
        df_rf = df_touches[df_touches["touch id"].isin(rf)].sort_values("touch id")
        df_rh = df_touches[df_touches["touch id"].isin(rh)].sort_values("touch id")
        #%% calculate steps
        # steps for each paw: step size, swing time, rung begin, rung end, touch id begin, touch id end
        step_lf = []
        step_lh = []
        step_rf = []
        step_rh = []
        
        for st in range(1,len(df_lf)):
            size = df_lf["rung"].iloc[st] - df_lf["rung"].iloc[st-1]
            time = df_lf["begin touch"].iloc[st] - df_lf["end touch"].iloc[st-1]
            step_lf.append([size, time, df_lf["rung"].iloc[st-1], df_lf["rung"].iloc[st], df_lf["touch id"].iloc[st-1], df_lf["touch id"].iloc[st]])
        for st in range(1,len(df_lh)):
            size = df_lh["rung"].iloc[st] - df_lh["rung"].iloc[st-1]
            time = df_lh["begin touch"].iloc[st] - df_lh["end touch"].iloc[st-1]
            step_lh.append([size, time, df_lh["rung"].iloc[st-1], df_lh["rung"].iloc[st],df_lh["touch id"].iloc[st-1], df_lh["touch id"].iloc[st]])
        for st in range(1,len(df_rf)):
            size = df_rf["rung"].iloc[st] - df_rf["rung"].iloc[st-1]
            time = df_rf["begin touch"].iloc[st] - df_rf["end touch"].iloc[st-1]
            step_rf.append([size, time, df_rf["rung"].iloc[st-1], df_rf["rung"].iloc[st], df_rf["touch id"].iloc[st-1], df_rf["touch id"].iloc[st]])
        for st in range(1,len(df_rh)):
            size = df_rh["rung"].iloc[st] - df_rh["rung"].iloc[st-1]
            time = df_rh["begin touch"].iloc[st] - df_rh["end touch"].iloc[st-1]
            step_rh.append([size, time, df_rh["rung"].iloc[st-1], df_rh["rung"].iloc[st], df_rh["touch id"].iloc[st-1], df_rh["touch id"].iloc[st]])          
        #%% check detected limbs and steps
        # check the steps to see if there are steps that are not possible: negative swing time, below minimum swing time
        # steps with negative swing time (because of overlapping double touches touches) -> skip run
        # steps with low swing time -> first we try switching paws for a touch involved in the step, if this does not work we try to split the touch
        # if both options don't work -> we will skip the run
        skip = False
        # left front
        if True in [item[1] < min_swing for item in step_lf]: # check if any steps have shorter swing time (item[1]) than minimum
            adjust = True
            # copy the lists with touch ids for the left paws to see if adjustments work without changing the original limb detection
            lf_check = lf.copy()
            lh_check = lh.copy()

            while adjust == True:
                lf_check = lf.copy()
                lh_check = lh.copy()
                for z in range(len(step_lf)):
                    if step_lf[z][1] < min_swing:
                        if step_lf[z][3] in double_rungs_left: # if the end touch of the step is a touch on a double rung -> change paw for begin touch
                            if step_lf[z][2] in double_rungs_left:  # both touches from the step are on a double rung
                                # skip run
                                skip = True
                                adjust = False
                                reason = "overlapping double touches"
                            else:
                                # switch paw for first touch
                                if step_lf[z][4] in lf_check:  # check if value is in the list
                                    lf_check.remove(step_lf[z][4])
                                    lh_check.append(step_lf[z][4])
                                else:
                                    continue
                        elif step_lf[z][2] in double_rungs_left: # first touch on a double rung
                            # switch paw for second touch
                            lf_check.remove(step_lf[z][5])
                            lh_check.append(step_lf[z][5])
                # update touches paws (new dataframe)
                df_lf_check = df_touches[df_touches["touch id"].isin(lf_check)].sort_values("touch id")
                df_lh_check = df_touches[df_touches["touch id"].isin(lh_check)].sort_values("touch id")
                # check the new steps after switching touches between paws
                if skip == True:
                    break
                step_lf_check = []
                step_lh_check = []
                for st in range(1,len(df_lf_check)):
                    size = df_lf_check["rung"].iloc[st] - df_lf_check["rung"].iloc[st-1]
                    time = df_lf_check["begin touch"].iloc[st] - df_lf_check["end touch"].iloc[st-1]
                    step_lf_check.append([size, time, df_lf_check["rung"].iloc[st-1], df_lf_check["rung"].iloc[st], df_lf_check["touch id"].iloc[st-1], df_lf_check["touch id"].iloc[st]])
                for st in range(1,len(df_lh_check)):
                    size = df_lh_check["rung"].iloc[st] - df_lh_check["rung"].iloc[st-1]
                    time = df_lh_check["begin touch"].iloc[st] - df_lh_check["end touch"].iloc[st-1]
                    step_lh_check.append([size, time, df_lh_check["rung"].iloc[st-1], df_lh_check["rung"].iloc[st], df_lh_check["touch id"].iloc[st-1], df_lh_check["touch id"].iloc[st]])
                if False not in [item[1] >= min_swing for item in step_lf_check] and False not in [item[1] >= min_swing for item in step_lh_check]:
                    # no swing time is lower than minimum -> switch worked
                    # update the dataframes with the check data and quit the while loop
                    lf = lf_check
                    lh = lh_check
                    df_lf = df_lf_check
                    df_lh = df_lh_check
                    step_lf = step_lf_check
                    step_lh = step_lh_check
                    adjust = False
                else:
                    # changing paws did not work -> try splitting touch in step
                    lf_check = lf.copy()
                    lh_check = lh.copy()
                    df_touches_check = df_touches.copy()
                    for z in range(len(step_lf)):
                        if step_lf[z][1] < min_swing:
                            if step_lf[z][3] in double_rungs_left:
                                # if second touch in double rungs (no need to check both for double rungs, run would have been skipped earlier)
                                # split first touch
                                # check if touch already is split -> not splitting again
                                if step_lf[z][4] not in split:
                                    touch_check = df_touches_check[df_touches_check["touch id"] == step_lf[z][4]].squeeze()
                                    touch_lf = touch_check.copy()
                                    touch_lh = touch_check.copy()
                                    time_lf = round(touch_check["touch"]*ratio_left)
                                    time_lh = touch_check["touch"] - time_lf
                                    ind = touch_check.name
                                    # adjust touch front paw
                                    touch_lf["end touch"] = touch_check["begin touch"] + time_lf
                                    touch_lf["touch id"] = touch_check["touch id"] + 0.1
                                    touch_lf["touch"] = time_lf
                                    # adjust touch hind paw
                                    touch_lh["begin touch"] = touch_check["end touch"] - time_lh
                                    touch_lh["touch id"] = touch_check["touch id"] + 0.2
                                    touch_lh["touch"] = time_lh
                                    # insert new touches, remove old touch and update dataframe touches
                                    df_touches_check.loc[ind] = touch_lf
                                    df_touches_check = df_touches_check.append(touch_lh)
                                    lf_check.remove(touch_check["touch id"])
                                    lf_check.append(touch_lf["touch id"])
                                    lh_check.append(touch_lh["touch id"])
                                else:
                                    # already split -> skip run
                                    skip = True
                                    reason = "switch did not work + already split"
                                    adjust = False
                            elif step_lf[z][2] in double_rungs_left:
                                # if first touch in double rungs (no need to check both for double rungs, run would have been skipped earlier)
                                # split second touch
                                # check if touch already is split -> not splitting again
                                if step_lf[z][5] not in split:
                                    touch_check = df_touches_check[df_touches_check["touch id"] == step_lf[z][5]].squeeze()
                                    touch_lf = touch_check.copy()
                                    touch_lh = touch_check.copy()
                                    time_lf = round(touch_check["touch"]*ratio_left)
                                    time_lh = touch_check["touch"] - time_lf
                                    ind = touch_check.name
                                    # adjust touch front paw
                                    touch_lf["end touch"] = touch_check["begin touch"] + time_lf
                                    touch_lf["touch id"] = touch_check["touch id"] + 0.1
                                    touch_lf["touch"] = time_lf
                                    # adjust touch hind paw
                                    touch_lh["begin touch"] = touch_check["end touch"] - time_lh
                                    touch_lh["touch id"] = touch_check["touch id"] + 0.2
                                    touch_lh["touch"] = time_lh
                                    # insert new touches, remove old touch and update dataframe touches
                                    df_touches_check.loc[ind] = touch_lf
                                    df_touches_check = df_touches_check.append(touch_lh)
                                    lf_check.remove(touch_check["touch id"])
                                    lf_check.append(touch_lf["touch id"])
                                    lh_check.append(touch_lh["touch id"])
                                else:
                                    # already split -> skip run
                                    skip = True
                                    reason = "switch did not work + already split"
                                    adjust = False
                    # update touches paws (new dataframe)
                    df_lf_check = df_touches_check[df_touches_check["touch id"].isin(lf_check)].sort_values("touch id")
                    df_lh_check = df_touches_check[df_touches_check["touch id"].isin(lh_check)].sort_values("touch id")
                    # check the new steps after switching touches between paws
                    step_lf_check = []
                    step_lh_check = []
                    for st in range(1,len(df_lf_check)):
                        size = df_lf_check["rung"].iloc[st] - df_lf_check["rung"].iloc[st-1]
                        time = df_lf_check["begin touch"].iloc[st] - df_lf_check["end touch"].iloc[st-1]
                        step_lf_check.append([size, time, df_lf_check["rung"].iloc[st-1], df_lf_check["rung"].iloc[st], df_lf_check["touch id"].iloc[st-1], df_lf_check["touch id"].iloc[st]])
                    for st in range(1,len(df_lh_check)):
                        size = df_lh_check["rung"].iloc[st] - df_lh_check["rung"].iloc[st-1]
                        time = df_lh_check["begin touch"].iloc[st] - df_lh_check["end touch"].iloc[st-1]
                        step_lh_check.append([size, time, df_lh_check["rung"].iloc[st-1], df_lh_check["rung"].iloc[st], df_lh_check["touch id"].iloc[st-1], df_lh_check["touch id"].iloc[st]])
                    if False not in [item[1] >= min_swing for item in step_lf_check] and False not in [item[1] >= min_swing for item in step_lh_check]:
                        # no swing time is lower than minimum -> split worked
                        # update the dataframes with the check data and quit the while loop
                        lf = lf_check
                        lh = lh_check
                        df_lf = df_lf_check
                        df_lh = df_lh_check
                        df_touches = df_touches_check
                        step_lf = step_lf_check
                        step_lh = step_lh_check
                        adjust = False
                    else:
                        # switch and split did not work -> skip run
                        skip = True
                        reason = "switch + split did not work"
                        adjust = False
        if skip == True:
            skip_run.append([run, reason, session])
            continue
        # left hind
        if True in [item[1] < min_swing for item in step_lh]: # check if any steps have shorter swing time (item[1]) than minimum
            adjust = True
            # copy the lists with touch ids for the left paws to see if adjustments work without changing the original limb detection
            lf_check = lf.copy()
            lh_check = lh.copy()
            while adjust == True:
                lf_check = lf.copy()
                lh_check = lh.copy()
                for z in range(len(step_lh)):
                    if step_lh[z][1] < min_swing:
                        if step_lh[z][3] in double_rungs_left: # if the end touch of the step is a touch on a double rung -> change paw for begin touch
                            if step_lh[z][2] in double_rungs_left:  # both touches from the step are on a double rung
                                # skip run
                                skip = True
                                adjust = False
                                reason = "overlapping double touches"
                                
                            else:
                                # switch paw for first touch
                                lh_check.remove(step_lh[z][4])
                                lf_check.append(step_lh[z][4])
                        elif step_lh[z][2] in double_rungs_left: # first touch on a double rung
                            # switch paw for second touch
                            lh_check.remove(step_lh[z][5])
                            lf_check.append(step_lh[z][5])
                # update touches paws (new dataframe)
                df_lf_check = df_touches[df_touches["touch id"].isin(lf_check)].sort_values("touch id")
                df_lh_check = df_touches[df_touches["touch id"].isin(lh_check)].sort_values("touch id")
                # check the new steps after switching touches between paws
                if skip == True:
                    break
                step_lf_check = []
                step_lh_check = []
                for st in range(1,len(df_lf_check)):
                    size = df_lf_check["rung"].iloc[st] - df_lf_check["rung"].iloc[st-1]
                    time = df_lf_check["begin touch"].iloc[st] - df_lf_check["end touch"].iloc[st-1]
                    step_lf_check.append([size, time, df_lf_check["rung"].iloc[st-1], df_lf_check["rung"].iloc[st], df_lf_check["touch id"].iloc[st-1], df_lf_check["touch id"].iloc[st]])
                for st in range(1,len(df_lh_check)):
                    size = df_lh_check["rung"].iloc[st] - df_lh_check["rung"].iloc[st-1]
                    time = df_lh_check["begin touch"].iloc[st] - df_lh_check["end touch"].iloc[st-1]
                    step_lh_check.append([size, time, df_lh_check["rung"].iloc[st-1], df_lh_check["rung"].iloc[st], df_lh_check["touch id"].iloc[st-1], df_lh_check["touch id"].iloc[st]])
                if False not in [item[1] >= min_swing for item in step_lh_check] and False not in [item[1] >= min_swing for item in step_lf_check]:
                    # no swing time is lower than minimum -> switch worked
                    # update the dataframes with the check data and quit the while loop
                    lf = lf_check
                    lh = lh_check
                    df_lf = df_lf_check
                    df_lh = df_lh_check
                    step_lf = step_lf_check
                    step_lh = step_lh_check
                    adjust = False
                else:
                    # changing paws did not work -> try splitting touch in step
                    lf_check = lf.copy()
                    lh_check = lh.copy()
                    df_touches_check = df_touches.copy()
                    for z in range(len(step_lh)):
                        if step_lh[z][1] < min_swing:
                            if step_lh[z][3] in double_rungs_left:
                                # if second touch in double rungs (no need to check both for double rungs, run would have been skipped earlier)
                                # split first touch
                                # check if touch already is split -> not splitting again
                                if step_lh[z][4] not in split:
                                    touch_check = df_touches_check[df_touches_check["touch id"] == step_lh[z][4]].squeeze()
                                    touch_lf = touch_check.copy()
                                    touch_lh = touch_check.copy()
                                    time_lf = round(touch_check["touch"]*ratio_left)
                                    time_lh = touch_check["touch"] - time_lf
                                    ind = touch_check.name
                                    # adjust touch front paw
                                    touch_lf["end touch"] = touch_check["begin touch"] + time_lf
                                    touch_lf["touch id"] = touch_check["touch id"] + 0.1
                                    touch_lf["touch"] = time_lf
                                    # adjust touch hind paw
                                    touch_lh["begin touch"] = touch_check["end touch"] - time_lh
                                    touch_lh["touch id"] = touch_check["touch id"] + 0.2
                                    touch_lh["touch"] = time_lh
                                    # insert new touches, remove old touch and update dataframe touches
                                    df_touches_check.loc[ind] = touch_lf
                                    df_touches_check = df_touches_check.append(touch_lh)
                                    lh_check.remove(touch_check["touch id"])
                                    lh_check.append(touch_lh["touch id"])
                                    lf_check.append(touch_lf["touch id"])
                                else:
                                    # already split -> skip run
                                    skip = True
                                    reason = "switch did not work + already split"
                                    adjust = False
                            elif step_lh[z][2] in double_rungs_left:
                                # if first touch in double rungs (no need to check both for double rungs, run would have been skipped earlier)
                                # split second touch
                                # check if touch already is split -> not splitting again
                                if step_lh[z][5] not in split:
                                    touch_check = df_touches_check[df_touches_check["touch id"] == step_lh[z][5]].squeeze()
                                    touch_lf = touch_check.copy()
                                    touch_lh = touch_check.copy()
                                    time_lf = round(touch_check["touch"]*ratio_left)
                                    time_lh = touch_check["touch"] - time_lf
                                    ind = touch_check.name
                                    # adjust touch front paw
                                    touch_lf["end touch"] = touch_check["begin touch"] + time_lf
                                    touch_lf["touch id"] = touch_check["touch id"] + 0.1
                                    touch_lf["touch"] = time_lf
                                    # adjust touch hind paw
                                    touch_lh["begin touch"] = touch_check["end touch"] - time_lh
                                    touch_lh["touch id"] = touch_check["touch id"] + 0.2
                                    touch_lh["touch"] = time_lh
                                    # insert new touches, remove old touch and update dataframe touches
                                    df_touches_check.loc[ind] = touch_lf
                                    df_touches_check = df_touches_check.append(touch_lh)
                                    lh_check.remove(touch_check["touch id"])
                                    lh_check.append(touch_lh["touch id"])
                                    lf_check.append(touch_lf["touch id"])
                                else:
                                    # already split -> skip run
                                    skip = True
                                    reason = "switch did not work + already split"
                                    adjust = False
                    # update touches paws (new dataframe)
                    df_lf_check = df_touches_check[df_touches_check["touch id"].isin(lf_check)].sort_values("touch id")
                    df_lh_check = df_touches_check[df_touches_check["touch id"].isin(lh_check)].sort_values("touch id")
                    # check the new steps after switching touches between paws
                    step_lf_check = []
                    step_lh_check = []
                    for st in range(1,len(df_lf_check)):
                        size = df_lf_check["rung"].iloc[st] - df_lf_check["rung"].iloc[st-1]
                        time = df_lf_check["begin touch"].iloc[st] - df_lf_check["end touch"].iloc[st-1]
                        step_lf_check.append([size, time, df_lf_check["rung"].iloc[st-1], df_lf_check["rung"].iloc[st], df_lf_check["touch id"].iloc[st-1], df_lf_check["touch id"].iloc[st]])
                    for st in range(1,len(df_lh_check)):
                        size = df_lh_check["rung"].iloc[st] - df_lh_check["rung"].iloc[st-1]
                        time = df_lh_check["begin touch"].iloc[st] - df_lh_check["end touch"].iloc[st-1]
                        step_lh_check.append([size, time, df_lh_check["rung"].iloc[st-1], df_lh_check["rung"].iloc[st], df_lh_check["touch id"].iloc[st-1], df_lh_check["touch id"].iloc[st]])
                    if False not in [item[1] >= min_swing for item in step_lh_check] and False not in [item[1] >= min_swing for item in step_lf_check]:
                        # no swing time is lower than minimum -> split worked
                        # update the dataframes with the check data and quit the while loop
                        lf = lf_check
                        lh = lh_check
                        df_lf = df_lf_check
                        df_lh = df_lh_check
                        df_touches = df_touches_check
                        step_lf = step_lf_check
                        step_lh = step_lh_check
                        adjust = False
                    else:
                        # switch and split did not work -> skip run
                        skip = True
                        reason = "switch + split did not work"
                        adjust = True
        if skip == True:
            skip_run.append([run, reason, session])
            continue
        # right front
        if True in [item[1] < min_swing for item in step_rf]: # check if any steps have shorter swing time (item[1]) than minimum
            adjust = True
            # copy the lists with touch ids for the left paws to see if adjustments work without changing the original limb detection
            rf_check = rf.copy()
            rh_check = rh.copy()
            while adjust == True:
                rf_check = rf.copy()
                rh_check = rh.copy()
                for z in range(len(step_rf)):
                    if step_rf[z][1] < min_swing:
                        if step_rf[z][3] in double_rungs_right: # if the end touch of the step is a touch on a double rung -> change paw for begin touch
                            if step_rf[z][2] in double_rungs_right:  # both touches from the step are on a double rung
                                # skip run
                                skip = True
                                adjust = False
                                reason = 'overlapping double touches'
                            else:
                                # switch paw for first touch
                                if step_rf[z][4] in rf_check:
                                    rf_check.remove(step_rf[z][4])
                                    rh_check.append(step_rf[z][4])
                                # rf_check.remove(step_rf[z][4])
                                # rh_check.append(step_rf[z][4])
                        elif step_rf[z][2] in double_rungs_right: # first touch on a double rung
                            # switch paw for second touch
                            rf_check.remove(step_rf[z][5])
                            rh_check.append(step_rf[z][5])
                # update touches paws (new dataframe)
                df_rf_check = df_touches[df_touches["touch id"].isin(rf_check)].sort_values("touch id")
                df_rh_check = df_touches[df_touches["touch id"].isin(rh_check)].sort_values("touch id")
                # check the new steps after switching touches between paws
                if skip == True:
                    break
                step_rf_check = []
                step_rh_check = []
                for st in range(1,len(df_rf_check)):
                    size = df_rf_check["rung"].iloc[st] - df_rf_check["rung"].iloc[st-1]
                    time = df_rf_check["begin touch"].iloc[st] - df_rf_check["end touch"].iloc[st-1]
                    step_rf_check.append([size, time, df_rf_check["rung"].iloc[st-1], df_rf_check["rung"].iloc[st], df_rf_check["touch id"].iloc[st-1], df_rf_check["touch id"].iloc[st]])
                for st in range(1,len(df_rh_check)):
                    size = df_rh_check["rung"].iloc[st] - df_rh_check["rung"].iloc[st-1]
                    time = df_rh_check["begin touch"].iloc[st] - df_rh_check["end touch"].iloc[st-1]
                    step_rh_check.append([size, time, df_rh_check["rung"].iloc[st-1], df_rh_check["rung"].iloc[st], df_rh_check["touch id"].iloc[st-1], df_rh_check["touch id"].iloc[st]])
                if False not in [item[1] >= min_swing for item in step_rf_check] and False not in [item[1] >= min_swing for item in step_rh_check]:
                    # no swing time is lower than minimum -> switch worked
                    # update the dataframes with the check data and quit the while loop
                    rf = rf_check
                    rh = rh_check
                    df_rf = df_rf_check
                    df_rh = df_rh_check
                    step_rf = step_rf_check
                    step_rh = step_rh_check
                    adjust = False
                else:
                    # changing paws did not work -> try splitting touch in step
                    rf_check = rf.copy()
                    rh_check = rh.copy()
                    df_touches_check = df_touches.copy()
                    for z in range(len(step_rf)):
                        if step_rf[z][1] < min_swing:
                            if step_rf[z][3] in double_rungs_right:
                                # if second touch in double rungs (no need to check both for double rungs, run would have been skipped earlier)
                                # split first touch
                                # check if touch already is split -> not splitting again
                                if step_rf[z][4] not in split:
                                    touch_check = df_touches_check[df_touches_check["touch id"] == step_rf[z][4]].squeeze()
                                    touch_rf = touch_check.copy()
                                    touch_rh = touch_check.copy()
                                    time_rf = round(touch_check["touch"]*ratio_right)
                                    time_rh = touch_check["touch"] - time_rf
                                    ind = touch_check.name
                                    # adjust touch front paw
                                    touch_rf["end touch"] = touch_check["begin touch"] + time_rf
                                    touch_rf["touch id"] = touch_check["touch id"] + 0.1
                                    touch_rf["touch"] = time_rf
                                    # adjust touch hind paw
                                    touch_rh["begin touch"] = touch_check["end touch"] - time_rh
                                    touch_rh["touch id"] = touch_check["touch id"] + 0.2
                                    touch_rh["touch"] = time_rh
                                    # insert new touches, remove old touch and update dataframe touches
                                    df_touches_check.loc[ind] = touch_rf
                                    df_touches_check = df_touches_check.append(touch_rh)
                                    rf_check.remove(touch_check["touch id"])
                                    rf_check.append(touch_rf["touch id"])
                                    rh_check.append(touch_rh["touch id"])
                                else:
                                    # already split -> skip run
                                    skip = True
                                    reason = "switch did not work + already split"
                                    adjust = False
                            elif step_rf[z][2] in double_rungs_right:
                                # if first touch in double rungs (no need to check both for double rungs, run would have been skipped earlier)
                                # split second touch
                                # check if touch already is split -> not splitting again
                                if step_rf[z][5] not in split:
                                    touch_check = df_touches_check[df_touches_check["touch id"] == step_rf[z][5]].squeeze()
                                    touch_rf = touch_check.copy()
                                    touch_rh = touch_check.copy()
                                    time_rf = round(touch_check["touch"]*ratio_right)
                                    time_rh = touch_check["touch"] - time_rf
                                    ind = touch_check.name
                                    # adjust touch front paw
                                    touch_rf["end touch"] = touch_check["begin touch"] + time_rf
                                    touch_rf["touch id"] = touch_check["touch id"] + 0.1
                                    touch_rf["touch"] = time_rf
                                    # adjust touch hind paw
                                    touch_rh["begin touch"] = touch_check["end touch"] - time_rh
                                    touch_rh["touch id"] = touch_check["touch id"] + 0.2
                                    touch_rh["touch"] = time_rh
                                    # insert new touches, remove old touch and update dataframe touches
                                    df_touches_check.loc[ind] = touch_rf
                                    df_touches_check = df_touches_check.append(touch_rh)
                                    rf_check.remove(touch_check["touch id"])
                                    rf_check.append(touch_rf["touch id"])
                                    rh_check.append(touch_rh["touch id"])
                                else:
                                    # already split -> skip run
                                    skip = True
                                    reason = "switch did not work + already split"
                                    adjust = False
                    # update touches paws (new dataframe)
                    df_rf_check = df_touches_check[df_touches_check["touch id"].isin(rf_check)].sort_values("touch id")
                    df_rh_check = df_touches_check[df_touches_check["touch id"].isin(rh_check)].sort_values("touch id")
                    # check the new steps after switching touches between paws
                    step_rf_check = []
                    step_rh_check = []
                    for st in range(1,len(df_rf_check)):
                        size = df_rf_check["rung"].iloc[st] - df_rf_check["rung"].iloc[st-1]
                        time = df_rf_check["begin touch"].iloc[st] - df_rf_check["end touch"].iloc[st-1]
                        step_rf_check.append([size, time, df_rf_check["rung"].iloc[st-1], df_rf_check["rung"].iloc[st], df_rf_check["touch id"].iloc[st-1], df_rf_check["touch id"].iloc[st]])
                    for st in range(1,len(df_rh_check)):
                        size = df_rh_check["rung"].iloc[st] - df_rh_check["rung"].iloc[st-1]
                        time = df_rh_check["begin touch"].iloc[st] - df_rh_check["end touch"].iloc[st-1]
                        step_rh_check.append([size, time, df_rh_check["rung"].iloc[st-1], df_rh_check["rung"].iloc[st], df_rh_check["touch id"].iloc[st-1], df_rh_check["touch id"].iloc[st]])
                    if False not in [item[1] >= min_swing for item in step_rf_check] and False not in [item[1] >= min_swing for item in step_rh_check]:
                        # no swing time is lower than minimum -> split worked
                        # update the dataframes with the check data and quit the while loop
                        rf = rf_check
                        rh = rh_check
                        df_rf = df_rf_check
                        df_rh = df_rh_check
                        df_touches = df_touches_check
                        step_rf = step_rf_check
                        step_rh = step_rh_check
                        adjust = False
                    else:
                        # switch and split did not work -> skip run
                        skip = True
                        reason = 'switch + split did not work'
                        adjust = False
        if skip == True:
            skip_run.append([run, reason, session])
            continue
        # right hind
        if True in [item[1] < min_swing for item in step_rh]: # check if any steps have shorter swing time (item[1]) than minimum
            adjust = True
            # copy the lists with touch ids for the left paws to see if adjustments work without changing the original limb detection
            rf_check = rf.copy()
            rh_check = rh.copy()
            while adjust == True:
                rf_check = rf.copy()
                rh_check = rh.copy()
                for z in range(len(step_rh)):
                    if step_rh[z][1] < min_swing:
                        if step_rh[z][3] in double_rungs_right: # if the end touch of the step is a touch on a double rung -> change paw for begin touch
                            if step_rh[z][2] in double_rungs_right:  # both touches from the step are on a double rung
                                # skip run
                                skip = True
                                adjust = False
                                reason = "overlapping double touches"
                            else:
                                # switch paw for first touch
                                rh_check.remove(step_rh[z][4])
                                rf_check.append(step_rh[z][4])
                        elif step_rh[z][2] in double_rungs_right: # first touch on a double rung
                            # switch paw for second touch
                            rh_check.remove(step_rh[z][5])
                            rf_check.append(step_rh[z][5])
                # update touches paws (new dataframe)
                df_rf_check = df_touches[df_touches["touch id"].isin(rf_check)].sort_values("touch id")
                df_rh_check = df_touches[df_touches["touch id"].isin(rh_check)].sort_values("touch id")
                # check the new steps after switching touches between paws
                if skip == True:
                    break
                step_rf_check = []
                step_rh_check = []
                for st in range(1,len(df_rf_check)):
                    size = df_rf_check["rung"].iloc[st] - df_rf_check["rung"].iloc[st-1]
                    time = df_rf_check["begin touch"].iloc[st] - df_rf_check["end touch"].iloc[st-1]
                    step_rf_check.append([size, time, df_rf_check["rung"].iloc[st-1], df_rf_check["rung"].iloc[st], df_rf_check["touch id"].iloc[st-1], df_rf_check["touch id"].iloc[st]])
                for st in range(1,len(df_rh_check)):
                    size = df_rh_check["rung"].iloc[st] - df_rh_check["rung"].iloc[st-1]
                    time = df_rh_check["begin touch"].iloc[st] - df_rh_check["end touch"].iloc[st-1]
                    step_rh_check.append([size, time, df_rh_check["rung"].iloc[st-1], df_rh_check["rung"].iloc[st], df_rh_check["touch id"].iloc[st-1], df_rh_check["touch id"].iloc[st]])
                if False not in [item[1] >= min_swing for item in step_rh_check] and False not in [item[1] >= min_swing for item in step_rf_check]:
                    # no swing time is lower than minimum -> switch worked
                    # update the dataframes with the check data and quit the while loop
                    rf = rf_check
                    rh = rh_check
                    df_rf = df_rf_check
                    df_rh = df_rh_check
                    step_rf = step_rf_check
                    step_rh = step_rh_check
                    adjust = False
                else:
                    # changing paws did not work -> try splitting touch in step
                    rf_check = rf.copy()
                    rh_check = rh.copy()
                    df_touches_check = df_touches.copy()
                    for z in range(len(step_rh)):
                        if step_rh[z][1] < min_swing:
                            if step_rh[z][3] in double_rungs_right:
                                # if second touch in double rungs (no need to check both for double rungs, run would have been skipped earlier)
                                # split first touch
                                # check if touch already is split -> not splitting again
                                if step_rh[z][4] not in split:
                                    touch_check = df_touches_check[df_touches_check["touch id"] == step_rh[z][4]].squeeze()
                                    touch_rf = touch_check.copy()
                                    touch_rh = touch_check.copy()
                                    time_rf = round(touch_check["touch"]*ratio_right)
                                    time_rh = touch_check["touch"] - time_rf
                                    ind = touch_check.name
                                    # adjust touch front paw
                                    touch_rf["end touch"] = touch_check["begin touch"] + time_rf
                                    touch_rf["touch id"] = touch_check["touch id"] + 0.1
                                    touch_rf["touch"] = time_rf
                                    # adjust touch hind paw
                                    touch_rh["begin touch"] = touch_check["end touch"] - time_rh
                                    touch_rh["touch id"] = touch_check["touch id"] + 0.2
                                    touch_rh["touch"] = time_rh
                                    # insert new touches, remove old touch and update dataframe touches
                                    df_touches_check.loc[ind] = touch_rf
                                    df_touches_check = df_touches_check.append(touch_rh)
                                    rh_check.remove(touch_check["touch id"])
                                    rh_check.append(touch_rh["touch id"])
                                    rf_check.append(touch_rf["touch id"])
                                else:
                                    # already split -> skip run
                                    skip = True
                                    reason = "switch did not work + already split"
                                    adjust = False
                            elif step_rh[z][2] in double_rungs_right:
                                # if first touch in double rungs (no need to check both for double rungs, run would have been skipped earlier)
                                # split second touch
                                # check if touch already is split -> not splitting again
                                if step_rh[z][5] not in split:
                                    touch_check = df_touches_check[df_touches_check["touch id"] == step_rh[z][5]].squeeze()
                                    touch_rf = touch_check.copy()
                                    touch_rh = touch_check.copy()
                                    time_rf = round(touch_check["touch"]*ratio_right)
                                    time_rh = touch_check["touch"] - time_rf
                                    ind = touch_check.name
                                    # adjust touch front paw
                                    touch_rf["end touch"] = touch_check["begin touch"] + time_rf
                                    touch_rf["touch id"] = touch_check["touch id"] + 0.1
                                    touch_rf["touch"] = time_rf
                                    # adjust touch hind paw
                                    touch_rh["begin touch"] = touch_check["end touch"] - time_rh
                                    touch_rh["touch id"] = touch_check["touch id"] + 0.2
                                    touch_rh["touch"] = time_rh
                                    # insert new touches, remove old touch and update dataframe touches
                                    df_touches_check.loc[ind] = touch_rf
                                    df_touches_check = df_touches_check.append(touch_rh)
                                    rh_check.remove(touch_check["touch id"])
                                    rh_check.append(touch_rh["touch id"])
                                    rf_check.append(touch_rf["touch id"])
                                else:
                                    # already split -> skip run
                                    skip = True
                                    reason = "switch did not work + already split"
                                    adjust = False
                    # update touches paws (new dataframe)
                    df_rf_check = df_touches_check[df_touches_check["touch id"].isin(rf_check)].sort_values("touch id")
                    df_rh_check = df_touches_check[df_touches_check["touch id"].isin(rh_check)].sort_values("touch id")
                    # check the new steps after switching touches between paws
                    step_rf_check = []
                    step_rh_check = []
                    for st in range(1,len(df_rf_check)):
                        size = df_rf_check["rung"].iloc[st] - df_rf_check["rung"].iloc[st-1]
                        time = df_rf_check["begin touch"].iloc[st] - df_rf_check["end touch"].iloc[st-1]
                        step_rf_check.append([size, time, df_rf_check["rung"].iloc[st-1], df_rf_check["rung"].iloc[st], df_rf_check["touch id"].iloc[st-1], df_rf_check["touch id"].iloc[st]])
                    for st in range(1,len(df_rh_check)):
                        size = df_rh_check["rung"].iloc[st] - df_rh_check["rung"].iloc[st-1]
                        time = df_rh_check["begin touch"].iloc[st] - df_rh_check["end touch"].iloc[st-1]
                        step_rh_check.append([size, time, df_rh_check["rung"].iloc[st-1], df_rh_check["rung"].iloc[st], df_rh_check["touch id"].iloc[st-1], df_rh_check["touch id"].iloc[st]])
                    if False not in [item[1] >= min_swing for item in step_rh_check] and False not in [item[1] >= min_swing for item in step_rf_check]:
                        # no swing time is lower than minimum -> split worked
                        # update the dataframes with the check data and quit the while loop
                        rf = rf_check
                        rh = rh_check
                        df_rf = df_rf_check
                        df_rh = df_rh_check
                        df_touches = df_touches_check
                        step_rf = step_rf_check
                        step_rh = step_rh_check
                        adjust = False
                    else:
                        # switch and split did not work -> skip run
                        skip = True
                        reason = "switch + split did not work"
                        adjust = True
        if skip == True:
            skip_run.append([run, reason, session])
            continue
        step_size = [item[0] for item in step_lf] + [item[0] for item in step_lh] + [item[0] for item in step_rf] + [item[0] for item in step_rh]
        if True in [item > max_step for item in step_size]:
            skip_run.append([run, 'contains large step', session])
            continue
        if len(step_lf) == 0 or len(step_lh) == 0 or len(step_rf) == 0 or len(step_rh) == 0:
            skip_run.append([run, 'no touches for paw(s)', session])
            continue
        # change the rungs back
        if (df_touches["direction"] == 0).all(): # 37 -> 1
            df_touches["rung"] = (df_touches["rung"] - 38)*-1
            # change side so that info left paw is interpreted as left
            for s in range(len(df_touches)):
                if df_touches["side"].iloc[s] == 0:
                    df_touches["side"].iloc[s] = 1
                elif df_touches["side"].iloc[s] == 1:
                    df_touches["side"].iloc[s] = 0
        # update dataframe for each paw          
        df_lf = df_touches[df_touches["touch id"].isin(lf)].sort_values("touch id")
        df_lh = df_touches[df_touches["touch id"].isin(lh)].sort_values("touch id")
        df_rf = df_touches[df_touches["touch id"].isin(rf)].sort_values("touch id")
        df_rh = df_touches[df_touches["touch id"].isin(rh)].sort_values("touch id")
        if len(df_lh)<3:
            skip_run.append([run, 'Not Enough touches lh', session])
            continue
        if len(df_lf)<3:
            skip_run.append([run, 'Not Enough touches lf', session])
            continue
        if len(df_rh)<3:
            skip_run.append([run, 'Not Enough touches rh', session])
            continue
        if len(df_rf)<3:
            skip_run.append([run, 'Not Enough touches rf', session])
            continue
        # recalculate steps after changing rungs back (only necesarry if rungs changed)
        if (df_touches["direction"] == 0).all():
            step_lf = []
            step_lh = []
            step_rf = []
            step_rh = []
        
            for st in range(1,len(df_lf)):
                size = df_lf["rung"].iloc[st-1] - df_lf["rung"].iloc[st]
                time = df_lf["begin touch"].iloc[st] - df_lf["end touch"].iloc[st-1]
                step_lf.append([size, time, df_lf["rung"].iloc[st-1], df_lf["rung"].iloc[st], df_lf["touch id"].iloc[st-1], df_lf["touch id"].iloc[st]])
            for st in range(1,len(df_lh)):
                size = df_lh["rung"].iloc[st-1] - df_lh["rung"].iloc[st]
                time = df_lh["begin touch"].iloc[st] - df_lh["end touch"].iloc[st-1]
                step_lh.append([size, time, df_lh["rung"].iloc[st-1], df_lh["rung"].iloc[st],df_lh["touch id"].iloc[st-1], df_lh["touch id"].iloc[st]])
            for st in range(1,len(df_rf)):
                size = df_rf["rung"].iloc[st-1] - df_rf["rung"].iloc[st]
                time = df_rf["begin touch"].iloc[st] - df_rf["end touch"].iloc[st-1]
                step_rf.append([size, time, df_rf["rung"].iloc[st-1], df_rf["rung"].iloc[st], df_rf["touch id"].iloc[st-1], df_rf["touch id"].iloc[st]])
            for st in range(1,len(df_rh)):
                size = df_rh["rung"].iloc[st-1] - df_rh["rung"].iloc[st]
                time = df_rh["begin touch"].iloc[st] - df_rh["end touch"].iloc[st-1]
                step_rh.append([size, time, df_rh["rung"].iloc[st-1], df_rh["rung"].iloc[st], df_rh["touch id"].iloc[st-1], df_rh["touch id"].iloc[st]])

        #%% gait cycle
        # also add to touches dataframe column which paw the touch belongs to
        df_touches = df_touches.sort_values("touch id")
        df_touches["paw"] = 0
        for xx in range(len(df_touches)):
            if df_touches["touch id"].iloc[xx] in lf:
                gait_cycle.loc["lf", range(int(df_touches["begin touch"].iloc[xx]), int(df_touches["end touch"].iloc[xx]) + 1)] = 1
                df_touches["paw"].iloc[xx] = 'lf'
            elif df_touches["touch id"].iloc[xx] in lh:
                gait_cycle.loc["lh", range(int(df_touches["begin touch"].iloc[xx]), int(df_touches["end touch"].iloc[xx]) + 1)] = 1
                df_touches["paw"].iloc[xx] = 'lh'
            elif df_touches["touch id"].iloc[xx] in rf:
                gait_cycle.loc["rf", range(int(df_touches["begin touch"].iloc[xx]), int(df_touches["end touch"].iloc[xx]) + 1)] = 1
                df_touches["paw"].iloc[xx] = 'rf'
            elif df_touches["touch id"].iloc[xx] in rh:
                gait_cycle.loc["rh", range(int(df_touches["begin touch"].iloc[xx]), int(df_touches["end touch"].iloc[xx]) + 1)] = 1
                df_touches["paw"].iloc[xx] = 'rh'
        df_gait = df_touches[(df_touches['rung'] <= 30) & (df_touches["rung"] >= 7)]
        if len(df_gait)==0:
            continue
        gaitbegin = min(df_gait["begin touch"])
        gaitend = max(df_gait["end touch"])
        
        allpaws.extend(df_touches["paw"])
        alltouches.extend(df_touches["touch"])
        allbegintouches.extend(df_touches["begin touch"])
        allendtouches.extend(df_touches["end touch"])
        allrunid.extend(df_touches["run id"])
        allrungs.extend(df_touches["rung"])
        allsessions.extend(df_touches["session nr"])
        allsubjects.extend(df_touches["subject id"])
        allsides.extend(df_touches["side"])
        alldirections.extend(df_touches["direction"])  
        
        #%% gather parameters for further analysis
        stance = [df_lf, df_lh, df_rf, df_rh]
        df_stance = pd.concat(stance)
        # # add empty row to dataframe for variables current run
        df_analysis.loc[df_analysis.shape[0]] = 0
        front_limbs = step_lf + step_rf
        hind_limbs = step_lh + step_rh
        # fill in variables
        df_analysis["session id"].iloc[-1] = session
        df_analysis["run id"].iloc[-1] = run
        df_analysis["subject id"].iloc[-1] = df["subject id"].iloc[0] 
        df_analysis["session nr"].iloc[-1] = df["session nr"].iloc[0]
        df_analysis["direction"].iloc[-1] = df["direction"].iloc[0]
        df_analysis["state sequence"].iloc[-1] = df["statesequence"].iloc[0]
        df_analysis["timerun"].iloc[-1] = max(df_touches["end touch"]) - min(df_touches["begin touch"])
        df_analysis["number of steps"].iloc[-1] = len(step_size)  
        df_analysis["mean touchtime"].iloc[-1]=df_touches["touch"].mean()
        for fr in range(len(front_limbs)):
            # count even and odd front steps
            if front_limbs[fr][0] % 2 == 0:
                df_analysis["even steps"].iloc[-1] += 1
                if front_limbs[fr][0] == 2:
                      df_analysis["step 2"].iloc[-1] += 1 
                elif front_limbs[fr][0] == 4:
                    df_analysis["step 4"].iloc[-1] += 1
                elif front_limbs[fr][0] == 6:
                    df_analysis["step 6"].iloc[-1] += 1
            elif front_limbs[fr][0] % 2 == 1:
                df_analysis["odd steps"].iloc[-1] += 1
        for fr in range(len(hind_limbs)):
            # count even and odd hind steps
            if hind_limbs[fr][0] % 2 == 0:
                df_analysis["even steps"].iloc[-1] += 1
                if hind_limbs[fr][0] == 2:
                      df_analysis["step 2"].iloc[-1] += 1 
                elif hind_limbs[fr][0] == 4:
                    df_analysis["step 4"].iloc[-1] += 1
                elif hind_limbs[fr][0] == 6:
                    df_analysis["step 6"].iloc[-1] += 1
            elif hind_limbs[fr][0] % 2 == 1:
                df_analysis["odd steps"].iloc[-1] += 1
        # calculate average swing and stance phase
        swing_phase_lf = []
        stance_phase_lf = []
        for ph in range(len(step_lf)):
            swing = step_lf[ph][1]      # time of swing
            stance = int(df_lf["touch"][df_lf["touch id"] == step_lf[ph][4]])        # touch (stance) time from touch before swing
            total = swing + stance
            swing_phase_lf.append((swing/total)*100)                    # swing phase %
            stance_phase_lf.append((stance/total)*100)                  # stance phase %
        df_analysis["swing phase lf"].iloc[-1] = sum(swing_phase_lf)/len(swing_phase_lf)
        df_analysis["stance phase lf"].iloc[-1] = sum(stance_phase_lf)/len(stance_phase_lf)
        
        swing_phase_lh = []
        stance_phase_lh = []
        for ph in range(len(step_lh)):
            swing = step_lh[ph][1]      # time of swing
            stance = int(df_lh["touch"][df_lh["touch id"] == step_lh[ph][4]])        # touch (stance) time from touch before swing
            total = swing + stance
            swing_phase_lh.append((swing/total)*100)                    # swing phase %
            stance_phase_lh.append((stance/total)*100)                  # stance phase %
        df_analysis["swing phase lh"].iloc[-1] = sum(swing_phase_lh)/len(swing_phase_lh)
        df_analysis["stance phase lh"].iloc[-1] = sum(stance_phase_lh)/len(stance_phase_lh)
        
        swing_phase_rf = []
        stance_phase_rf = []
        for ph in range(len(step_rf)):
            swing = step_rf[ph][1]      # time of swing
            stance = int(df_rf["touch"][df_rf["touch id"] == step_rf[ph][4]])        # touch (stance) time from touch before swing
            total = swing + stance
            swing_phase_rf.append((swing/total)*100)                    # swing phase %
            stance_phase_rf.append((stance/total)*100)                  # stance phase %
        df_analysis["swing phase rf"].iloc[-1] = sum(swing_phase_rf)/len(swing_phase_rf)
        df_analysis["stance phase rf"].iloc[-1] = sum(stance_phase_rf)/len(stance_phase_rf)
        
        swing_phase_rh = []
        stance_phase_rh = []
        for ph in range(len(step_rh)):
            swing = step_rh[ph][1]      # time of swing
            stance = int(df_rh["touch"][df_rh["touch id"] == step_rh[ph][4]])        # touch (stance) time from touch before swing
            total = swing + stance
            swing_phase_rh.append((swing/total)*100)                    # swing phase %
            stance_phase_rh.append((stance/total)*100)                  # stance phase %
        df_analysis["swing phase rh"].iloc[-1] = sum(swing_phase_rh)/len(swing_phase_rh)
        df_analysis["stance phase rh"].iloc[-1] = sum(stance_phase_rh)/len(stance_phase_rh)
        #%% inter limb coupling 
        #inter-limb coupling -> synchronisety diagonal paws
        # lag_lf_rh = []
        # for ii in range(len(df_rh)-1):
        #     ref_steps = []
        #     tar = range(int(df_rh["begin touch"].iloc[ii]), int(df_rh["begin touch"].iloc[ii+1]))
        #     for jj in range(len(df_lf)-1):
        #         ref = range(int(df_lf["begin touch"].iloc[jj]), int(df_lf["begin touch"].iloc[jj+1]))
        #         overlap = len(set(tar) & set(ref))
        #         ref_steps.append([overlap, df_lf["touch id"].iloc[jj], len(ref)])
        #     refs = pd.DataFrame(ref_steps, columns = ['overlap', 'touch id', 'step time'])
        #     in_max = refs["overlap"].idxmax()
        #     touch_ref = df_lf[df_lf["touch id"] == refs["touch id"].loc[in_max]]
        #     touch_tar = df_rh.iloc[ii]
        #     time_step = refs["step time"].loc[in_max]
        #     lag = ((touch_tar["begin touch"] - touch_ref["begin touch"])/time_step)*100     # phase lag
        #     lag_lf_rh.append([lag.iloc[0], refs["touch id"].loc[in_max]])
        # lf_rh = pd.DataFrame(lag_lf_rh, columns = ['lag', 'ref touch id'])
        # lf_rh = lf_rh.drop_duplicates(subset=['ref touch id'], keep='first')
        # for uu in range(len(lf_rh)):
        #     all_lag.append([session, df["subject id"].iloc[0], run, df["session nr"].iloc[0], lf_rh["lag"].iloc[uu], 'lf/rh'])
        # if len(lf_rh) == 0:
        #     df_analysis["lag lf/rh"].iloc[-1] = 0
        # else:
        #     df_analysis["lag lf/rh"].iloc[-1] = sum(map(abs, lf_rh["lag"]))/len(lf_rh)
            
        # lag_rf_lh = []
        # for ii in range(len(df_lh)-1):
        #     ref_steps = []
        #     tar = range(int(df_lh["begin touch"].iloc[ii]), int(df_lh["begin touch"].iloc[ii+1]))
        #     for jj in range(len(df_rf)-1):
        #         ref = range(int(df_rf["begin touch"].iloc[jj]), int(df_rf["begin touch"].iloc[jj+1]))
        #         overlap = len(set(tar) & set(ref))
        #         ref_steps.append([overlap, df_rf["touch id"].iloc[jj], len(ref)])
        #     refs = pd.DataFrame(ref_steps, columns = ['overlap', 'touch id', 'step time'])
        #     in_max = refs["overlap"].idxmax()
        #     touch_ref = df_rf[df_rf["touch id"] == refs["touch id"].loc[in_max]]
        #     touch_tar = df_lh.iloc[ii]
        #     time_step = refs["step time"].loc[in_max]
        #     lag = ((touch_tar["begin touch"] - touch_ref["begin touch"])/time_step)*100     # phase lag
        #     lag_rf_lh.append([lag.iloc[0], refs["touch id"].loc[in_max]])
        # rf_lh = pd.DataFrame(lag_rf_lh, columns = ['lag', 'ref touch id'])
        # rf_lh = rf_lh.drop_duplicates(subset=['ref touch id'], keep='first')
        # for uu in range(len(rf_lh)):
        #     all_lag.append([session, df["subject id"].iloc[0], run, df["session nr"].iloc[0], rf_lh["lag"].iloc[uu], 'rf/lh'])
        # if len(rf_lh) == 0:
        #     df_analysis["lag rf/lh"].iloc[-1] = 0
        # else:
        #     df_analysis["lag rf/lh"].iloc[-1] = sum(map(abs, rf_lh["lag"]))/len(rf_lh)
        
        phase_lf_rh = []                      # moment hind touch start with respect to step cycle front 0 = begin stance, pi = end stance/begin swing, 2pi = end swing/begin stance
        for ii in range(len(df_lf)-1):
            # stance -> 0-pi
            # swing -> pi-2pi
            zeropi = df_lf["begin touch"].iloc[ii]
            pi = df_lf["end touch"].iloc[ii]
            twopi = df_lf["begin touch"].iloc[ii+1]
            for jj in range(len(df_rh)):
                tar = df_rh["begin touch"].iloc[jj]
                if zeropi <= tar <= pi:
                    # stance phase ref
                    x = pi - zeropi
                    xtar = tar - zeropi
                    phase = xtar/x
                    phase_lf_rh.append(phase)
                    all_lag.append([session, df["subject id"].iloc[0], run, df["session nr"].iloc[0], phase, 'lf/rh'])
                elif pi < tar < twopi:
                    # swing phase ref
                    x = twopi - pi
                    xtar = tar - pi
                    phase = (xtar/x) + 1
                    phase_lf_rh.append(phase)
                    all_lag.append([session, df["subject id"].iloc[0], run, df["session nr"].iloc[0], phase, 'lf/rh'])
        if len(phase_lf_rh) == 0:
            df_analysis["lag lf/rh"].iloc[-1] = 0
        else:
            df_analysis["lag lf/rh"].iloc[-1] = sum(phase_lf_rh)/len(phase_lf_rh)
        # overlap stance lf/rh
        stance_lf = 0
        stance_lf_rh = 0
        for tt in range(len(gait_cycle.columns)):
            if gait_cycle.loc["lf"].iloc[tt] == 1:
                stance_lf += 1
                if gait_cycle.loc["rh"].iloc[tt] == 1:
                    stance_lf_rh += 1
        
        overlap = (stance_lf_rh/stance_lf)*100                  # percentage of stance time front hind paw also in stance
        all_overlap.append([session, run, df["session nr"].iloc[0], overlap, 'lf/rh'])
        
        phase_rf_lh = []                      # moment hind touch start with respect to step cycle front 0 = begin stance, pi = end stance/begin swing, 2pi = end swing/begin stance
        for ii in range(len(df_rf)-1):
            zeropi = df_rf["begin touch"].iloc[ii]
            pi = df_rf["end touch"].iloc[ii]
            twopi = df_rf["begin touch"].iloc[ii+1]
            for jj in range(len(df_lh)):
                tar = df_lh["begin touch"].iloc[jj]
                if zeropi <= tar <= pi:
                    x = pi - zeropi
                    xtar = tar - zeropi
                    phase = xtar/x
                    phase_rf_lh.append(phase)
                    all_lag.append([session, df["subject id"].iloc[0], run, df["session nr"].iloc[0], phase, 'rf/lh'])
                elif pi < tar < twopi:
                    x = twopi - pi
                    xtar = tar - pi
                    phase = (xtar/x) + 1
                    phase_rf_lh.append(phase)
                    all_lag.append([session, df["subject id"].iloc[0], run, df["session nr"].iloc[0], phase, 'rf/lh'])
        if len(phase_rf_lh) == 0:
            df_analysis["lag rf/lh"].iloc[-1] = 0
        else:
            df_analysis["lag rf/lh"].iloc[-1] = sum(phase_rf_lh)/len(phase_rf_lh)
        # overlap stance lf/rh
        # stance_rf = 0
        # stance_rf_lh = 0
        # for tt in range(len(gait_cycle.columns)):
        #     if gait_cycle.loc["rf"].iloc[tt] == 1:
        #         stance_rf += 1
        #         if gait_cycle.loc["lh"].iloc[tt] == 1:
        #             stance_rf_lh += 1
        # overlap = (stance_rf_lh/stance_rf)*100                  # percentage of stance time front hind paw also in stance
        # all_overlap.append([session, run, df["session nr"].iloc[0], overlap, 'rf/lh'])


        # calculate support pattern
        diagonal = 0        # support on 2 paws lf/rh or rf/lh
        girdle = 0          # support on 2 paws lf/rf or lh/rh
        lateral = 0         # support on 2 paws lf/lh or rf/rh
        zero = 0            # support on 0 paws
        single = 0          # support on 1 paw
        triple = 0          # support on 3 paws
        four = 0            # support on 4 paws 
        gait_cycle=gait_cycle.sort_index(axis=1)
        gait_cycle = gait_cycle.truncate(before=gaitbegin,after=gaitend,axis=1)
        sup_time = len(gait_cycle.columns)
        for tt in range(sup_time):
            sup = sum(gait_cycle.iloc[:,tt])
            if sup == 0:
                zero += 1
            elif sup == 1:
                single += 1
            elif sup == 3:
                triple += 1
            elif sup == 4:
                four += 1
            elif sup == 2:
                if (gait_cycle.loc["lf"].iloc[tt] == 1 and gait_cycle.loc["rh"].iloc[tt] == 1) or (gait_cycle.loc["rf"].iloc[tt] == 1 and gait_cycle.loc["lh"].iloc[tt] == 1):
                    diagonal += 1
                elif (gait_cycle.loc["lf"].iloc[tt] == 1 and gait_cycle.loc["rf"].iloc[tt] == 1) or (gait_cycle.loc["lh"].iloc[tt] == 1 and gait_cycle.loc["rh"].iloc[tt] == 1):
                    girdle += 1
                elif (gait_cycle.loc["lf"].iloc[tt] == 1 and gait_cycle.loc["lh"].iloc[tt] == 1) or (gait_cycle.loc["rf"].iloc[tt] == 1 and gait_cycle.loc["rh"].iloc[tt] == 1):
                    lateral += 1
        df_analysis["support diagonal"].iloc[-1] = (diagonal/sup_time)*100
        df_analysis["support girdle"].iloc[-1] = (girdle/sup_time)*100
        df_analysis["support lateral"].iloc[-1] = (lateral/sup_time)*100
        df_analysis["support 0 paws"].iloc[-1] = (zero/sup_time)*100
        df_analysis["support 1 paw"].iloc[-1] = (single/sup_time)*100
        df_analysis["support 3 paws"].iloc[-1] = (triple/sup_time)*100
        df_analysis["support 4 paws"].iloc[-1] = (four/sup_time)*100
        # count high and low rungs for front and hind limbs
        for ab in range(len(df_lf)):
            if df_lf["direction"].iloc[ab] == 1:
                # 1 -> 37 on left mouseside: even rungs -> low, odd rungs -> high
                if df_lf["rung"].iloc[ab] % 2 == 0:
                    df_analysis["low rungs front"].iloc[-1] += 1
                else:
                    df_analysis["high rungs front"].iloc[-1] += 1
            elif df_lf["direction"].iloc[ab] == 0:
                # 37 -> 1 on left mouseside: even rungs -> high, odd rungs -> low
                if df_lf["rung"].iloc[ab] % 2 == 0:
                    df_analysis["high rungs front"].iloc[-1] += 1
                else:
                    df_analysis["low rungs front"].iloc[-1] += 1
        for ab in range(len(df_lh)):
            if df_lh["direction"].iloc[ab] == 1:
                # 1 -> 37 on left mouseside: even rungs -> low, odd rungs -> high
                if df_lh["rung"].iloc[ab] % 2 == 0:
                    df_analysis["low rungs hind"].iloc[-1] += 1
                else:
                    df_analysis["high rungs hind"].iloc[-1] += 1
            elif df_lh["direction"].iloc[ab] == 0:
                # 37 -> 1 on left mouseside: even rungs -> high, odd rungs -> low
                if df_lh["rung"].iloc[ab] % 2 == 0:
                    df_analysis["high rungs hind"].iloc[-1] += 1
                else:
                    df_analysis["low rungs hind"].iloc[-1] += 1
        for ab in range(len(df_rf)):
            if df_rf["direction"].iloc[ab] == 1:
                # 1 -> 37 on right mouseside: even rungs -> high, odd rungs -> low
                if df_rf["rung"].iloc[ab] % 2 == 0:
                    df_analysis["high rungs front"].iloc[-1] += 1
                else:
                    df_analysis["low rungs front"].iloc[-1] += 1
            elif df_rf["direction"].iloc[ab] == 0:
                # 37 -> 1 on right mouseside: even rungs -> low, odd rungs -> high
                if df_rf["rung"].iloc[ab] % 2 == 0:
                    df_analysis["low rungs front"].iloc[-1] += 1
                else:
                    df_analysis["high rungs front"].iloc[-1] += 1
        for ab in range(len(df_rh)):
            if df_rh["direction"].iloc[ab] == 1:
                # 1 -> 37 on right mouseside: even rungs -> high, odd rungs -> low
                if df_rh["rung"].iloc[ab] % 2 == 0:
                    df_analysis["high rungs hind"].iloc[-1] += 1
                else:
                    df_analysis["low rungs hind"].iloc[-1] += 1
            elif df_rh["direction"].iloc[ab] == 0:
                # 37 -> 1 on right mouseside: even rungs -> low, odd rungs -> high
                if df_rh["rung"].iloc[ab] % 2 == 0:
                    df_analysis["low rungs hind"].iloc[-1] += 1
                else:
                    df_analysis["high rungs hind"].iloc[-1] += 1
        #%% steps front paws
        df_front.loc[df_front.shape[0]] = 0
        df_front["session id"].iloc[-1] = session
        df_front["run id"].iloc[-1] = run
        df_front["session nr"].iloc[-1] = df["session nr"].iloc[0]
        df_front["direction"].iloc[-1] = df["direction"].iloc[0]
        df_front["subject id"].iloc[-1] = df["subject id"].iloc[0]
        df_front["total steps"].iloc[-1] = len(step_lf) + len(step_rf)
        # columns in list step: 0 = stepsize, 1 = swingtime, 2 = rung begin, 3 = rung end, 4 = touch id begin, 5 = touch id end
        for lfp in range(len(step_lf)):
            if df_front["direction"].iloc[-1] == 1:
                # 1 -> 37 on left mouseside: even rungs -> low, odd rungs -> high
                if step_lf[lfp][2] % 2 == 0:
                    # first rung low
                    if step_lf[lfp][3] % 2 == 0:
                        # both low rungs -> l-l
                        df_front["l-l"].iloc[-1] += 1
                    else:
                        # second rung high -> l-h
                        df_front["l-h"].iloc[-1] += 1
                else:
                    # first rung high
                    if step_lf[lfp][3] % 2 == 0:
                        # second rung low rungs -> h-l
                        df_front["h-l"].iloc[-1] += 1
                    else:
                        # both high rungs
                        if step_lf[lfp][0] == 2:
                            df_front["h-h 2"].iloc[-1] += 1
                        elif step_lf[lfp][0] == 4:
                            df_front["h-h 4"].iloc[-1] += 1
                        else:
                            df_front["h-h other"].iloc[-1] += 1
            elif df_front["direction"].iloc[-1] == 0:
                # 37 -> 1 on left mouseside: even rungs -> high, odd rungs -> low
                if step_lf[lfp][2] % 2 == 0:
                    # first rung high
                    if step_lf[lfp][3] % 2 == 0:
                        # both high rungs h-h
                        if step_lf[lfp][0] == 2:
                            df_front["h-h 2"].iloc[-1] += 1
                        elif step_lf[lfp][0] == 4:
                            df_front["h-h 4"].iloc[-1] += 1
                        else:
                            df_front["h-h other"].iloc[-1] += 1
                    else:
                        # second rung low -> h-l
                        df_front["h-l"].iloc[-1] += 1
                else:
                    # first rung low
                    if step_lf[lfp][3] % 2 == 0:
                        # second rung high -> l-h
                        df_front["l-h"].iloc[-1] += 1
                    else:
                        # second rung low -> l-l
                        df_front["l-l"].iloc[-1] += 1
        for rfp in range(len(step_rf)):
            if df_front["direction"].iloc[-1] == 0:
                # 37 -> 1 on right mouseside: even rungs -> low, odd rungs -> high       
                if step_rf[rfp][2] % 2 == 0:
                    # first rung low
                    if step_rf[rfp][3] % 2 == 0:
                        # both low rungs -> l-l
                        df_front["l-l"].iloc[-1] += 1
                    else:
                        # second rung high -> l-h
                        df_front["l-h"].iloc[-1] += 1
                else:
                    # first rung high
                    if step_rf[rfp][3] % 2 == 0:
                        # second rung low rungs -> h-l
                        df_front["h-l"].iloc[-1] += 1
                    else:
                        # both high rungs
                        if step_rf[rfp][0] == 2:
                            df_front["h-h 2"].iloc[-1] += 1
                        elif step_rf[rfp][0] == 4:
                            df_front["h-h 4"].iloc[-1] += 1
                        else:
                            df_front["h-h other"].iloc[-1] += 1
            elif df_front["direction"].iloc[-1] == 1:
                # 1 -> 37 on right mouseside: even rungs -> high, odd rungs -> low
                if step_rf[rfp][2] % 2 == 0:
                    # first rung high
                    if step_rf[rfp][3] % 2 == 0:
                        # both high rungs h-h
                        if step_rf[rfp][0] == 2:
                            df_front["h-h 2"].iloc[-1] += 1
                        elif step_rf[rfp][0] == 4:
                            df_front["h-h 4"].iloc[-1] += 1
                        else:
                            df_front["h-h other"].iloc[-1] += 1
                    else:
                        # second rung low -> h-l
                        df_front["h-l"].iloc[-1] += 1
                else:
                    # first rung low
                    if step_rf[rfp][3] % 2 == 0:
                        # second rung high -> l-h
                        df_front["l-h"].iloc[-1] += 1
                    else:
                        # second rung low -> l-l
                        df_front["l-l"].iloc[-1] += 1  
                        
        # # %% save data per run
        # # dataframe phase lag
        # phase_lag = [phase_lf_rh, phase_rf_lh]
        # df_phase_lag = pd.DataFrame(phase_lag, index = ['lag lf/rh', 'lag rf/lh'])
        # # dataframe support pattern
        df_support = df_analysis[['session id', 'run id', 'session nr', 'support diagonal', 'support girdle', 'support lateral', 'support 0 paws', 'support 1 paw', 'support 3 paws', 'support 4 paws']].iloc[-1]
        all_support.append(df_support)
        # with pd.ExcelWriter("C:\\LabProject\\LadderProject\\Data\\excel\\SCA1\\"+group+"\\Data_run_"+str(run)+"_.xlsx", engine='xlsxwriter') as writer:
        #     df_touches.to_excel(writer, sheet_name = 'Touches')
        #     gait_cycle.to_excel(writer, sheet_name = 'Gait cycle')
        #     df_phase_lag.to_excel(writer, sheet_name = 'Phase lag')
        #     df_support.to_excel(writer, sheet_name = 'Support pattern')


        #%% undetermined and not classified touches
        und = 0
        unc = 0
        paws = lf + lh + rf + rh
        for k in range(len(df_touches)):
            if df_touches["touch id"].iloc[k] in undetermined:
                und += 1
            elif df_touches["touch id"].iloc[k] not in paws:
                unc += 1
        df_stats_touches.loc[df_stats_touches.shape[0]] = 0
        df_stats_touches["session id"].iloc[-1] = session
        df_stats_touches["run id"].iloc[-1] = run
        df_stats_touches["session nr"].iloc[-1] = df["session nr"].iloc[0]
        ['session id', 'run id', 'session nr', 'total touches', 'lf', 'lh', 'rf', 'rh', 'undetermined', 'not classified']
        df_stats_touches["total touches"].iloc[-1] = len(df_touches)
        df_stats_touches["lf"].iloc[-1] = len(df_lf)
        df_stats_touches["lh"].iloc[-1] = len(df_lh)
        df_stats_touches["rf"].iloc[-1] = len(df_rf)
        df_stats_touches["rh"].iloc[-1] = len(df_rh)
        df_stats_touches["undetermined"].iloc[-1] = und
        df_stats_touches["not classified"].iloc[-1] = unc
        # #%% graph detected touches
        # output_file(filename="C:\\LabProject\\LadderProject\\Data\\Figures\\"+group+"\\Runs\\Gait_pattern_run_"+str(run)+".html", title="Gait pattern run %s" %run)
        # fig = figure(title = "Time vs rung | 4 paws | run %s" %run)
        # # 1 -> 37   left mouseside: even rungs -> low, odd rungs -> high
        # #           right mouseside: even rungs -> high, odd rungs -> low
        # # 37 -> 1   left mouseside: even rungs -> high, odd rungs -> low
        # #           right mouseside: even rungs -> low, odd rungs -> high
        
        # if (df_touches["direction"] == 1).all():
        #     for k in range(len(df_touches)):
        #         if df_touches["touch id"].iloc[k] in lf:
        #             if df_touches["rung"].iloc[k] % 2 == 0:
        #                 # even rung, left mouseside -> low rung
        #                 fig.line([df_touches["begin touch"].iloc[k], df_touches["end touch"].iloc[k]], df_touches["rung"].iloc[k], line_color = 'firebrick', line_dash = 'dashed', legend_label = 'left front paw')
        #             else:
        #                 # odd rung, left mouseside -> high rung
        #                 fig.line([df_touches["begin touch"].iloc[k], df_touches["end touch"].iloc[k]], df_touches["rung"].iloc[k], line_color = 'firebrick', legend_label = 'left front paw')
        #         elif df_touches["touch id"].iloc[k] in lh:
        #             if df_touches["rung"].iloc[k] % 2 == 0:
        #                 # even rung, left mouseside -> low rung
        #                 fig.line([df_touches["begin touch"].iloc[k], df_touches["end touch"].iloc[k]], df_touches["rung"].iloc[k], line_color = 'darkorange', line_dash = 'dashed', legend_label = 'left hind paw')
        #             else:
        #                 # odd rung, left mouseside -> high rung
        #                 fig.line([df_touches["begin touch"].iloc[k], df_touches["end touch"].iloc[k]], df_touches["rung"].iloc[k], line_color = 'darkorange', legend_label = 'left hind paw')
        #         elif df_touches["touch id"].iloc[k] in rf:
        #             if df_touches["rung"].iloc[k] % 2 == 1: 
        #                 # odd rung, right mouseside -> low rung
        #                 fig.line([df_touches["begin touch"].iloc[k], df_touches["end touch"].iloc[k]], df_touches["rung"].iloc[k], line_color = 'navy', line_dash = 'dashed', legend_label = 'right front paw')
        #             else:
        #                 # even rung, right mouseside -> high rung
        #                 fig.line([df_touches["begin touch"].iloc[k], df_touches["end touch"].iloc[k]], df_touches["rung"].iloc[k], line_color = 'navy', legend_label = 'right front paw')
        #         elif df_touches["touch id"].iloc[k] in rh:
        #             if df_touches["rung"].iloc[k] % 2 == 1:
        #                 # odd rung, right mouseside -> low rung
        #                 fig.line([df_touches["begin touch"].iloc[k], df_touches["end touch"].iloc[k]], df_touches["rung"].iloc[k], line_color = 'deepskyblue', line_dash = 'dashed', legend_label = 'right hind paw')
        #             else:
        #                 # even rung, right mouseside -> high rung
        #                 fig.line([df_touches["begin touch"].iloc[k], df_touches["end touch"].iloc[k]], df_touches["rung"].iloc[k], line_color = 'deepskyblue', legend_label = 'right hind paw')
        #         elif df_touches["touch id"].iloc[k] in undetermined:
        #             fig.line([df_touches["begin touch"].iloc[k], df_touches["end touch"].iloc[k]], df_touches["rung"].iloc[k], line_color = 'grey', legend_label = 'undetermined')
        #         else:
        #             fig.line([df_touches["begin touch"].iloc[k], df_touches["end touch"].iloc[k]], df_touches["rung"].iloc[k], line_color = 'green', legend_label = 'not classified')
        #     # set y range to get same plots
        #     fig.y_range = Range1d(5, 35)
        #     fig1.y_range = Range1d(5, 35)
        # elif (df_touches["direction"] == 0).all(): # 37 -> 1
        #     for k in range(len(df_touches)):
        #         if df_touches["touch id"].iloc[k] in lf:
        #             if df_touches["rung"].iloc[k] % 2 == 1:
        #                 # odd rung, left mouseside -> low rung
        #                 fig.line([df_touches["begin touch"].iloc[k], df_touches["end touch"].iloc[k]], df_touches["rung"].iloc[k], line_color = 'firebrick', line_dash = 'dashed', legend_label = 'left front paw')
        #             else:
        #                 # even rung, left mouseside -> high rung
        #                 fig.line([df_touches["begin touch"].iloc[k], df_touches["end touch"].iloc[k]], df_touches["rung"].iloc[k], line_color = 'firebrick', legend_label = 'left front paw')
        #         elif df_touches["touch id"].iloc[k] in lh:
        #             if df_touches["rung"].iloc[k] % 2 == 1:
        #                 # odd rung, left mouseside -> low rung
        #                 fig.line([df_touches["begin touch"].iloc[k], df_touches["end touch"].iloc[k]], df_touches["rung"].iloc[k], line_color = 'darkorange', line_dash = 'dashed', legend_label = 'left hind paw')
        #             else:
        #                 # even rung, left mouseside -> high rung
        #                 fig.line([df_touches["begin touch"].iloc[k], df_touches["end touch"].iloc[k]], df_touches["rung"].iloc[k], line_color = 'darkorange', legend_label = 'left hind paw')
        #         elif df_touches["touch id"].iloc[k] in rf:
        #             if df_touches["rung"].iloc[k] % 2 == 0:
        #                 # even rung, right mouseside -> low rung
        #                 fig.line([df_touches["begin touch"].iloc[k], df_touches["end touch"].iloc[k]], df_touches["rung"].iloc[k], line_color = 'navy', line_dash = 'dashed', legend_label = 'right front paw')
        #             else:
        #                 # odd rung, right mouseside -> high rung
        #                 fig.line([df_touches["begin touch"].iloc[k], df_touches["end touch"].iloc[k]], df_touches["rung"].iloc[k], line_color = 'navy', legend_label = 'right front paw')
        #         elif df_touches["touch id"].iloc[k] in rh:
        #             if df_touches["rung"].iloc[k] % 2 == 0:
        #                 # even rung, right mouseside -> low rung
        #                 fig.line([df_touches["begin touch"].iloc[k], df_touches["end touch"].iloc[k]], df_touches["rung"].iloc[k], line_color = 'deepskyblue', line_dash = 'dashed', legend_label = 'right hind paw')
        #             else:
        #                 # odd rung, right mouseside -> high rung
        #                 fig.line([df_touches["begin touch"].iloc[k], df_touches["end touch"].iloc[k]], df_touches["rung"].iloc[k], line_color = 'deepskyblue', legend_label = 'right hind paw')
        #         elif df_touches["touch id"].iloc[k] in undetermined:
        #             fig.line([df_touches["begin touch"].iloc[k], df_touches["end touch"].iloc[k]], df_touches["rung"].iloc[k], line_color = 'grey', legend_label = 'undetermined')
        #         else:
        #             fig.line([df_touches["begin touch"].iloc[k], df_touches["end touch"].iloc[k]], df_touches["rung"].iloc[k], line_color = 'green', legend_label = 'not classified')
        #     # set and flip y range to get all plots to look the same
        #     fig.y_range.flipped = True
        #     fig.y_range = Range1d(35, 5)
        #     fig1.y_range.flipped = True
        #     fig1.y_range = Range1d(35, 5)
        
        # # plot lines between steps
        # for k in range(len(df_lf) - 1):
        #     fig.line([df_lf["end touch"].iloc[k], df_lf["begin touch"].iloc[k+1]], [df_lf["rung"].iloc[k], df_lf["rung"].iloc[k+1]], line_color = 'firebrick')
        # for k in range(len(df_lh) - 1):
        #     fig.line([df_lh["end touch"].iloc[k], df_lh["begin touch"].iloc[k+1]], [df_lh["rung"].iloc[k], df_lh["rung"].iloc[k+1]], line_color = 'darkorange')
        # for k in range(len(df_rf) - 1):
        #     fig.line([df_rf["end touch"].iloc[k], df_rf["begin touch"].iloc[k+1]], [df_rf["rung"].iloc[k], df_rf["rung"].iloc[k+1]], line_color = 'navy')
        # for k in range(len(df_rh) - 1):
        #     fig.line([df_rh["end touch"].iloc[k], df_rh["begin touch"].iloc[k+1]], [df_rh["rung"].iloc[k], df_rh["rung"].iloc[k+1]], line_color = 'deepskyblue')
        
        # # visuals for the graphs
        # fig.legend.location = "bottom_right"
        # fig.legend.label_text_font_size = '14px'
        # fig.xaxis.axis_label = 'Time [ms]'
        # fig.xaxis.axis_label_text_font_size = '20px'
        # fig.xaxis.major_label_text_font_size = '16px'
        # fig.yaxis.axis_label = 'Rung'
        # fig.yaxis.axis_label_text_font_size = '20px'
        # fig.yaxis.major_label_text_font_size = '16px'
        # fig.title.text_font_size = '16px'
        
        # fig2 =gridplot([fig1,fig], ncols=2, width=700, height=700, toolbar_location='right')
        # save(fig2)
        # # plot gait (touches relative to each other)
        # tick = int(round((time_end - time_begin)/10))
        # sns.heatmap(gait_cycle, cbar = False, xticklabels = tick, cmap = ['white', 'black'])
        # # sns.heatmap(gait_cycle, cbar = False, xticklabels = tick, cmap = ['white', 'firebrick', 'darkorange', 'navy', 'deepskyblue'])
        # plt.title("Gait phase")
        # plt.savefig("C:\\LabProject\\LadderProject\\Data\\Figures\\"+group+"\\Runs\\Gait_phase_run_"+str(run)+".png")
        # plt.clf()
        # # plot support pattern 
        # pie_chart = [diagonal, girdle, lateral, zero, single, triple, four]
        # plt.pie(pie_chart, labels = ["diagonal", "girdle", "lateral", "zero", "single", "triple", "four"], autopct='%1.1f%%')
        # plt.title("Support pattern")
        # plt.savefig("C:\\LabProject\\LadderProject\\Data\\Figures\\"+group+"\\Runs\\Support_pattern_run_"+str(run)+".png")
        # plt.clf()
        # # plot lag phase
        # r = np.full((len(lag_lf_rh),), 1)
        # r2 = np.full((len(lag_rf_lh),), 2)
        # betweenx = x/100
        # lf_rh = [betweenx*2*np.pi for x in lag_lf_rh]
        # rf_lh = [betweenx*2*np.pi for x in lag_rf_lh]
        # fig = plt.figure()
        # ax = fig.add_subplot(121, projection='polar')
        # ax.scatter(lf_rh, r, label = 'lf_rh', color = 'firebrick')
        # ax.scatter(rf_lh, r2, label = 'rf_lh', color = 'navy')
        # ax.set_yticks([1,2])
        # ax.set_yticklabels([])
        # ax.set_xticklabels(['0 / 100%', '12.5%','25%', '37.5%','50%', '62.5%','75%', '87.5%'])
        # ax.set_theta_zero_location('S')
        # fig.legend(loc = 'lower center', bbox_to_anchor=(0.25,0))
        # ax.set_title("Lag phase distribution", y = 1.2)
        # ax1 = fig.add_subplot(122)
        # indx = np.arange(1)
        # height = 0.2
        # ax1.bar(indx, df_analysis["lag lf/rh"].iloc[-1], height, label = 'lf_rh', color = 'firebrick')
        # ax1.bar(indx+height, df_analysis["lag rf/lh"].iloc[-1], height, label = 'rf_lh', color = 'navy')
        # ax1.set_xticks([0,0.2])
        # ax1.set_xticklabels(['lf_rh', 'rf_lh'])
        # ax1.yaxis.tick_right()
        # ax1.set_title("Lag phase mean |%|")
        # fig.savefig("C:\LabProject\LadderProject\\Data\\Figures\\"+group+"\\Runs\\Lag_phase_run_"+str(run)+".png", bbox_inches = 'tight')
        # fig.clf()
        # plt.close('all')
#%% plot histogram of all touches 
# bins = np.linspace(0,300,75)
# plt.hist(alltouches, bins=bins)
# plt.xlabel('Touch time (ms)')
# plt.ylabel('Amount of touches')
df_alltouches = pd.DataFrame(alltouches)
df_allrungs = pd.DataFrame(allrungs)
df_allsessions = pd.DataFrame(allsessions)
df_allsubjects = pd.DataFrame(allsubjects)
df_allsides = pd.DataFrame(allsides)
df_alldirections = pd.DataFrame(alldirections)
df_allpaws = pd.DataFrame(allpaws)
df_allrunid = pd.DataFrame(allrunid)
df_allbegintouches = pd.DataFrame(allbegintouches)
df_allendtouches = pd.DataFrame(allendtouches)
df_final = [df_alltouches, df_allrungs, df_allsessions, df_allsubjects,df_allsides,df_alldirections,df_allpaws,df_allrunid,df_allbegintouches,df_allendtouches]
test= pd.concat(df_final,axis=1)
test.columns = ["touch", "rung","sessionnr","subject id","sides","direction","paw","run id","begintouches","endtouches"]
with pd.ExcelWriter("C:\\LabProject\\LadderProject\\Data\\excel\\alltouches"+group+"_.xlsx", engine='xlsxwriter') as writer:
        test.to_excel(writer, sheet_name = 'Touches')
df_types = pd.DataFrame(types)
with pd.ExcelWriter("C:\\LabProject\\LadderProject\\Data\\excel\\runtypes"+group+".xlsx", engine='xlsxwriter') as writer:
      df_types.to_excel(writer, sheet_name = 'Touches')

#%% output files          
# # save file with parameters analysis
df_analysis.to_excel("C:\\Labproject\\LadderProject\\Data\\excel\\Parameters"+group+".xlsx", index = False)
# df_analysis.to_excel("C:\\LabProject\\LadderProject\\Data\\Figures\\analysis"+group+".xlsx", index = False)
# save list with runs skipped
skipped = pd.DataFrame(skip_run, columns = ['run id', 'reason', 'session id'])
skipped.to_excel("C:\\Labproject\\LadderProject\\Data\\excel\\skippedruns"+group+"_.xlsx", index = False)
w_sequence = pd.concat(skip_sequence)
w_sequence.to_excel("C:\\Labproject\\LadderProject\\Data\\excel\\Wrong_sequence_runs"+group+"_.xlsx", index = False)
# save front steps
df_front.to_excel("C:\\LabProject\\LadderProject\\Data\\excel\\Steps_front_"+group+"_.xlsx", index = False)
# save lag and support all runs
supports = pd.concat(all_support, axis = 1).T
supports.to_excel("C:\\LabProject\\LadderProject\\Data\\excel\\Support_all_runs"+group+".xlsx", index = False)
lags = pd.DataFrame(all_lag, columns = ['session id', 'subject id', 'run id', 'ses nr', 'lag', 'pair'])
lags.to_excel("C:\\LabProject\\LadderProject\\Data\\excel\\Lag"+group+".xlsx", index = False)
# save overlap
overlaps = pd.DataFrame(all_overlap, columns = ['session id', 'run id', 'ses nr', 'overlap', 'pair'])
overlaps.to_excel("C:\\LabProject\\LadderProject\\Data\\excel\\Overlap_1run"+group+".xlsx", index = False)
# save stats touches
df_stats_touches.to_excel("C:\\LabProject\\LadderProject\\Data\\excel\\Stats_touches"+group+"_.xlsx", index = False)