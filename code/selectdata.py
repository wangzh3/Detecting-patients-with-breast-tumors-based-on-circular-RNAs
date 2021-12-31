import numpy as np
import pandas as pd
import os
import csv
labelfile="/Users/jfv472/Documents/mywork/cRNA/known_cRNA_breast.csv"
selectfile="/Users/jfv472/Documents/mywork/cRNA/select_avg.csv"
label = pd.read_csv(labelfile).values[:, 1]
chr_know=pd.read_csv(labelfile).values[:, 0]
chr_rna1=[]
for i in range(len(label)):
    item=chr_know[i]+"_"+str(label[i])
    chr_rna1.append(item)

select = pd.read_csv(selectfile).values[:, 1]
chr_select = pd.read_csv(selectfile).values[:, 0]
chr_rna2=[]
for i in range(len(select)):
    item=chr_select[i]+"_"+str(select[i])
    chr_rna2.append(item)

allneed=np.union1d(chr_rna1, chr_rna2)
print(len(allneed)) #6478

#select data
csvfile="/Users/jfv472/Documents/mywork/cRNA/patient_transpose.csv"
savefile="/Users/jfv472/Documents/mywork/cRNA/cmpt830/selected_data.csv"
df=pd.read_csv(csvfile)
print("load")
name=pd.read_csv(csvfile).values[:,0]
type=pd.read_csv(csvfile).values[:,1]
head=pd.read_csv(csvfile).columns.values
cleanhead=np.intersect1d(allneed,head)
select_col=[head[0],head[1]]+cleanhead.tolist()
select_df=pd.DataFrame(df,columns=select_col)
print(select_df)
select_df.to_csv(savefile,index=False)
