import pandas as pd 

#path of input files
base_path_mut = "C:\\LabProject\\LadderProject\\Data\\excel\\sorted\\mutant\\test\\"
#subject ids of which the sessions number has to be altered
values_b = [883]
amount_of_sessions_added = 7 

#%% mutants
df2 = pd.read_excel(base_path_mut+"steps_front.xlsx")

# add 1 to column A of all rows where column B is in values_b
new_df2=df2.copy()
new_df2.loc[new_df2["subject id"].isin(values_b), "session nr"] += amount_of_sessions_added
with pd.ExcelWriter(base_path_mut+ "steps_front.xlsx", engine='xlsxwriter') as writer:
      new_df2.to_excel(writer, sheet_name = 'MUT',index=False)
      
df2 = pd.read_excel(base_path_mut+ "lag.xlsx")
new_df2=df2.copy()
new_df2.loc[new_df2["subject id"].isin(values_b), "ses nr"] += amount_of_sessions_added

with pd.ExcelWriter(base_path_mut+"lag.xlsx", engine='xlsxwriter') as writer:
      new_df2.to_excel(writer, sheet_name = 'MUT',index=False)

df2 = pd.read_excel(base_path_mut+"parameters.xlsx")
new_df2=df2.copy()
new_df2.loc[new_df2["subject id"].isin(values_b), "session nr"] += amount_of_sessions_added

with pd.ExcelWriter(base_path_mut+"parameters.xlsx", engine='xlsxwriter') as writer:
      new_df2.to_excel(writer, sheet_name = 'MUT',index=False)
 
df2 = pd.read_excel(base_path_mut+"alltouches.xlsx")

new_df2=df2.copy()
new_df2.loc[new_df2["subject id"].isin(values_b), "sessionnr"] += amount_of_sessions_added
new_df2 = new_df2.iloc[: , 1:]


with pd.ExcelWriter(base_path_mut+"alltouches.xlsx", engine='xlsxwriter') as writer:
      new_df2.to_excel(writer, sheet_name = 'MUT')

#%% controls

# base_path_ctr = "C:\\LabProject\\LadderProject\\Data\\excel\\sorted\\control\\"
# values_b = [801, 802,803,804,805,806,807,808,809]
# amount_of_sessions_added = 7 
# df = pd.read_excel(base_path_ctr+ "steps_front.xlsx")
# new_df= df.copy()
# new_df.loc[new_df["subject id"].isin(values_b), "session nr"] += amount_of_sessions_added
# new_df = new_df.iloc[: , 1:]

# with pd.ExcelWriter(base_path_ctr + "steps_front.xlsx", engine='xlsxwriter') as writer:
#       new_df.to_excel(writer, sheet_name = 'CTR')


# df = pd.read_excel(base_path_ctr+ "control\\lag.xlsx")
# new_df= df.copy()
# new_df.loc[new_df["subject id"].isin(values_b), "ses nr"] += amount_of_sessions_added
# new_df = new_df.iloc[: , 1:]

# with pd.ExcelWriter(base_path_ctr+"lag.xlsx", engine='xlsxwriter') as writer:
#       new_df.to_excel(writer, sheet_name = 'CTR')

# df = pd.read_excel(base_path_ctr+"parameters.xlsx")
# new_df= df.copy()
# new_df.loc[new_df["subject id"].isin(values_b), "session nr"] += amount_of_sessions_added
# new_df = new_df.iloc[: , 1:]

# with pd.ExcelWriter(base_path_ctr+"parameters.xlsx", engine='xlsxwriter') as writer:
#       new_df.to_excel(writer, sheet_name = 'CTR')

# df = pd.read_excel(base_path_ctr+"alltouches.xlsx")
# new_df= df.copy()
# new_df.loc[new_df["subject id"].isin(values_b), "sessionnr"] += amount_of_sessions_added
# new_df = new_df.iloc[: , 1:]

# with pd.ExcelWriter(base_path_ctr+"alltocuhes.xlsx", engine='xlsxwriter') as writer:
#       new_df.to_excel(writer, sheet_name = 'CTR')