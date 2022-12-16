#!/usr/bin/env python3
import numpy as np
import pandas as pd
example_list = [1,2,3,4,5,6,7,8,9]
df = pd.DataFrame() #Create empty dataframe
N = 3 #Number of columns
split_list = np.array_split(example_list,N) #split the list in N frames

for i in range(N):
    df[f'Column_{i}'] = split_list[i] #Create a columns for each frame

print(df)
df.to_excel("./output.xlsx") #write excel output


import pandas as pd
import openpyxl

workbook = openpyxl.load_workbook("test.xlsx")
writer = pd.ExcelWriter('test.xlsx', engine='openpyxl')
writer.book = workbook
writer.sheets = dict((ws.title, ws) for ws in workbook.worksheets)
data_df.to_excel(writer, 'Existing_sheetname')
writer.save()
writer.close()