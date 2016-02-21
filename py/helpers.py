import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import datetime as dtime
import sys
import time


def slice_n_dice(df,df2,key,value):
    df[key].unique().tolist()

# if not train we pass in blist to map values
# otherwise if train, we create the map
def fsev_count(df,fsev,bin,train,blist,bidx):
    colname = 'fsev_' + str(fsev) + '_' + str(bin)
    if train:
        a = df[df['fault_severity']==fsev]
        b = a[bin].value_counts()[0:40]
        blist = b.tolist()
        bidx = b.index
        bdf = pd.DataFrame(b)
    df[colname] = 0
    #subset = df.loc[df.location.isin(a.index)]
    for i in range(0,len(blist)):
        percentile = blist[i]/np.sum(blist)
        locstr = str(bidx[i])
        subset = df.location == locstr
        df = df.set_value(df.location==locstr,colname,percentile)
    rval = df
    if train:
        rval = [df, blist,bidx]
    return rval


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


# binning data by a particular bin, we get ID counts for that bin
# and for each severity level
# this is no longer used
def slice_n_dice(results,bin):
    print('slice_n_dice')
    start = time.time()
    bins = str(bin)
    b_name0 = ''.join([bin,'_sev0_top20'])
    b_name1 = ''.join([bin,'_sev1_top20'])
    b_name2 = ''.join([bin,'_sev2_top20'])
    # variable creation
    sev2_loc =  helpers.countsfn(bins,'id',results.loc[results.fault_severity==2],'extra',True)
    sev1_loc =  helpers.countsfn(bins,'id',results.loc[results.fault_severity==1],'extra',True)
    sev0_loc =  helpers.countsfn(bins,'id',results.loc[results.fault_severity==0],'extra',True)
    sev2_loc = sev2_loc[0:30]
    sev1_loc = sev1_loc[0:30]
    sev0_loc = sev0_loc[0:30]
    # or we could do percentile?? make it non-categorical
    results[b_name0] = 0
    results[b_name1] = 0
    results[b_name2] = 0
    sev0_40 = []
    sev1_40 = []
    sev2_40 = []
    [sev0_40.append(x[0]) for x in sev0_loc]
    [sev1_40.append(x[0]) for x in sev1_loc]
    [sev2_40.append(x[0]) for x in sev2_loc]
    lists = [sev0_40, sev1_40, sev2_40]
    rs0 = results[bins].isin(sev0_40)
    rs1 = results[bins].isin(sev1_40)
    rs2 = results[bins].isin(sev2_40)
    res0 = results.loc[rs0]
    res1 = results.loc[rs1]
    res2 = results.loc[rs2]
    # results.set_value(rs0,b_name0,1)
    # results.set_value(rs1,b_name1,1)
    # results.set_value(rs2,b_name2,1)
    results = rowquantilefn(results,sev0_40,b_name0,bins)
    results = rowquantilefn(results,sev1_40,b_name1,bins)
    results = rowquantilefn(results,sev2_40,b_name2,bins)
    end = time.time()
    print(end - start)
    return results, lists
