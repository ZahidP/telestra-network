import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import datetime as dtime
import sys
import helpers

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_curve, auc

# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(rsx2, rsy2)

def extract():
    # read training set
    train = pd.read_csv('../data/train.csv', index_col='id')
    resource = pd.read_csv('../data/resource_type.csv')
    log_feature = pd.read_csv('../data/log_feature.csv')
    severity_type = pd.read_csv('../data/severity_type.csv')
    event = pd.read_csv('../data/event_type.csv')
    result = pd.merge(resource, event, how="left",on="id")
    results = pd.merge(result,log_feature, how="left")
    #results = results.set_index(['id'])
    results = pd.merge(train,results,left_index="true", how="left",right_on="id")
    # results = pd.merge(train,results,left_index="true", right_index="true", how="left")
    # results_2 = pd.merge(train,results,left_index="true", right_index="right" how="left", on="id")
    return results

def slice_n_dice(results,bin):
    bins = str(bin)
    b_name0 = ''.join([bin,'_sev0_top20'])
    b_name1 = ''.join([bin,'_sev1_top20'])
    b_name2 = ''.join([bin,'_sev2_top20'])
    # variable creation
    sev2_loc =  helpers.countsfn(bins,'id',results.loc[results.fault_severity==2],'extra',True)
    sev1_loc =  helpers.countsfn(bins,'id',results.loc[results.fault_severity==1],'extra',True)
    sev0_loc =  helpers.countsfn(bins,'id',results.loc[results.fault_severity==0],'extra',True)

    sev2_loc = sev2_loc[0:40]
    sev1_loc = sev1_loc[0:40]
    sev0_loc = sev0_loc[0:40]
    # or we could do percentile?? make it non-categorical
    results[b_name0] = 0
    results[b_name1] = 0
    results[b_name2] = 0
    sev0_t40list = []
    sev1_t40list = []
    sev2_t40list = []
    [sev0_t40list.append(x[0]) for x in sev0_loc]
    [sev1_t40list.append(x[0]) for x in sev1_loc]
    [sev2_t40list.append(x[0]) for x in sev2_loc]
    rs0 = results.location.isin(sev0_t40list)
    rs1 = results.location.isin(sev1_t40list)
    rs2 = results.location.isin(sev2_t40list)
    results.set_value(rs0,b_name0,1)
    results.set_value(rs1,b_name1,1)
    results.set_value(rs2,b_name2,1)
    return results


def summarize(results, binning):
    # get number of unique locations
    # number of unique features
    # histogram of # of events by location
    locations, event_types = results.location.ravel(), results.event_type.ravel()
    ids, resources = results.id.ravel(), results.resource_type.ravel()
    fault_severity = results.fault_severity.ravel()
    log_features = results.log_feature.ravel()
    unique_loc = pd.Series(locations).unique()
    ulog_features = pd.Series(log_features).unique()
    unique_ev_type = pd.Series(event_types).unique()
    unique_id = pd.Series(ids).unique()
    unique_resources = pd.Series(resources).unique()
    print('All rows')
    print(len(results))
    print('Unique Log Features')
    print(len(ulog_features))
    print('Unique Locations')
    print(len(unique_loc))
    print('Unique Ids')
    print(len(unique_id))
    print('Unique Resources')
    print(len(unique_resources))
    print('Unique Event Types')
    print(len(unique_ev_type))
    # unique events per location
    uniques_loc_id = []
    # unique event types per event id
    uniques_id_eventtype = []
    if binning:
        # for each unique location
        for ii in range(0,len(unique_loc)):
            subset = results.loc[results.location==unique_loc[ii]]
            ids = subset.id.ravel().tolist()
            unique_locs = pd.Series(ids).unique()
            uniques_loc_id.append(len(unique_locs))
            # unique_idcounts = [ (i,idsf2.count(i)) for i in set(uidsf2) ]

        ###### etype_counts = [ (i,event_types.count(i)) for i in set(unique_ev_type) ]
        # for each unique id
        for jj in range(0,len(unique_id)):
            # subset based on id
            subset2 = results.loc[results.id==unique_id[jj]]
            # get all event types for that id
            event_types = subset2.event_type.ravel().tolist()
            # distill to unique events
            unique_ev_type = pd.Series(event_types).unique()
            # get unique event count
            uniques_id_eventtype.append(len(unique_ev_type))
            # unique_counts = [ (i,unique_id.count(i)) for i in set(unique_id) ]
    return uniques_loc_id


def everything():
    # use prediction study design
    # training set
    # test set
    # validation set
    return 1

# pass in array of x per y, where x is the count per bucket
def histogram(x,title,use_np,norm):
    title = str(title)
    title = "More Histograms"
    if use_np:
        x = np.bincount(x)
    mu = np.average(x)
    #sigma = np.std(uniques)
    td, dnow = dtime.date.today(), dtime.date.isoformat(td)
    bin_count = int(np.max(x)/2.25)
    n, bins, patches = plt.hist(x, 8, normed=norm, facecolor='green', alpha=0.75)
    plt.xlabel('')
    plt.ylabel('')
    def split_join(word):
        word1 = list(word); word1.append('\ ')
        word2 = ''.join(word1)
        return word2
    def shorten(word):
        word = word[0:4]
        return word
    title = title.split(); title2 = map(split_join , title)
    title_hist = ''.join(list(title2))
    title2 = map(shorten, title)
    title_file = '-'.join(list(title2))
    plt.title(r'$\mathrm{Histogram\ of:\ ' + title_hist + '}\ $')
    if norm:
        max_y = 1
    else:
        max_y = len(x) *.8
    plt.axis([0, np.max(x), 0, max_y])
    plt.grid(True)
    #limit file name length
    title_file = title_file[0:14]
    fname = '../visualizations/' + str(title_file) + '-' + str(dnow) + '.png'
    plt.savefig(fname)
    plt.show()
