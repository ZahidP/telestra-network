import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import datetime as dtime
import sys

def countsfn(bins,items,df,extra,bar):
    colsf = df[bins].ravel()
    unique = pd.Series(colsf).unique()
    u_counts = []
    counts_bucket = []
    for ii in range(0,len(unique)):
        subset = df.loc[df[bins] == unique[ii]]
        item_list = subset[items].ravel().tolist()
        u_item_list = pd.Series(item_list).unique()
        u_counts.append(len(u_item_list))
        if bar:
            counts_bucket.append((unique[ii],len(u_item_list)))
    avg = np.average(u_counts)
    std = np.std(u_counts)
    # counts_bucket.sort(key=lambda x: x[1],reverse=True)
    # top_tf = [x[1]>15 for x in counts_bucket]
    # # or
    # top_bucket = counts_bucket[0:10]
    # top_bucket = [a for (a, truth) in zip(counts_bucket, top_tf) if truth]
    counts_bucket.sort(key=lambda x: x[1],reverse=True)
    if bar:
        retval = counts_bucket
    else:
        retval = avg, std
    return retval

def severity_summary(results, severity):
    fsev = results[results.fault_severity==severity]
    uids,cuids = ucountsfn(fsev)
    ext = 'Severity: ' + str(severity)
    avglogfeats, stdlogfeats = countsfn('id','log_feature',fsev, ext)
    avgevtype, stdeventtype = countsfn('id','event_type',fsev, ext)
    u_counts,u_colcounts = ucountsfn(fsev)

# this can get total number of counts per item
# example: ids: [11,14,18] --> [(11,25),(14,3),(18,90)]
# it cannot use different bins:
## ids: [11,14,18] count of resources [(11,??),(14,??),(18,??)]
def ucountsfn(fsev):
    colsf = fsev['id'].ravel()
    ucolsf = pd.Series(colsf).unique()
    colsf = colsf.tolist()
    u_colcounts = [ (i, colsf.count(i)) for i in set(colsf) ]
    u_counts = [ (colsf.count(i)) for i in set(ucolsf) ]
    avg_events = np.average(u_counts)
    print('Average Events: ' + str(avg_events))
    return u_counts,u_colcounts
