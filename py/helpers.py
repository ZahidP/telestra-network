import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import datetime as dtime
import sys
import time

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
    # get unique ids
    ucolsf = pd.Series(colsf).unique()
    # all ids to list
    colsf = colsf.tolist()
    # unique column and count
    u_colcounts = [ (i, colsf.count(i)) for i in set(colsf) ]
    # just count
    u_counts = [ (colsf.count(i)) for i in set(ucolsf) ]
    avg_events = np.average(u_counts)
    print('Average Events: ' + str(avg_events))
    return u_counts,u_colcounts

def removecols(df):
    del df['resource_type']
    del df['event_type']
    del df['location']
    del df['log_feature']
    return df

def assigncounts(df):
    df['event_length'] = 0
    colsf = df['id'].ravel()
    # get unique ids
    ucolsf = pd.Series(colsf).unique()
    # all ids to list
    colsf = colsf.tolist()
    # unique column and count
    print('getting unique counts')
    start = time.time()
    u_colcounts = [ (i, colsf.count(i)) for i in set(colsf) ]
    end = time.time()
    print(end - start)
    for ii in range(0,len(ucolsf)):
        if (ii < 2):
            start = time.time()
        subset = df.loc[df['id'] == ucolsf[ii]]
        event_length = len(subset['id'])
        df = df.set_value(subset.index,'event_length',event_length)
        if (ii < 2):
            end = time.time()
            print('loop iter')
            print(end - start)
    return df

def mendgroups(results,test,col):
    res_loc =  countsfn(col,'id',results,'extra',True)
    test_loc =  countsfn(col,'id',test,'extra',True)
    tlo40 = []
    rlo40 = []
    [tlo40.append(x[0]) for x in test_loc]
    [rlo40.append(x[0]) for x in res_loc]
    tlo40 = tlo40[0:40]
    rlo40 = rlo40[0:40]
    difftr = list(set(tlo40) - set(rlo40))
    diffrt = list(set(rlo40) - set(tlo40))
    if (len(difftr) > 0):
        rs1 = test[col].isin(difftr)
        test.set_value(rs1,col,'other')
    if (len(diffrt) > 0):
        rs0 = results[col].isin(diffrt)
        results.set_value(rs0,col,'other')
    return results, test
