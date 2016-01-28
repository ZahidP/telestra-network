import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import datetime as dtime
import sys

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_curve, auc


def extract():
    # read training set
    train = pd.read_csv('../data/train.csv', index_col='id')
    resource = pd.read_csv('../data/resource_type.csv')
    log_feature = pd.read_csv('../data/log_feature.csv')
    event = pd.read_csv('../data/event_type.csv')
    result = pd.merge(resource, event, how="left",on="id")
    results = pd.merge(result,log_feature, how="left")
    #results = results.set_index(['id'])
    results = pd.merge(train,results,left_index="true", how="left",right_on="id")
    # results = pd.merge(train,results,left_index="true", right_index="true", how="left")


    # results_2 = pd.merge(train,results,left_index="true", right_index="right" how="left", on="id")
    return results

def summarize():
    # get number of unique locations
    # number of unique features
    # histogram of # of events by location
    locations, event_types = results.location.ravel(), results.event_type.ravel()
    ids, resources = results.id.ravel(), results.resource_type.ravel()
    unique_loc = pd.Series(locations).unique()
    unique_ev_type = pd.Series(event_types).unique()
    unique_id = pd.Series(ids).unique()
    unique_resources = pd.Series(resources).unique()
    # unique events per location
    uniques_loc_id = []
    # unique event types per event id
    uniques_id_eventtype = []
    # for each unique location
    for ii in range(0,len(unique_loc)):
        subset = results.loc[results.location==unique_loc[ii]]
        ids = subset.id.ravel().tolist()
        unique_locs = pd.Series(ids).unique()
        uniques_loc_id.append(len(unique_locs))
        # unique_counts = [ (i,unique_id.count(i)) for i in set(unique_id) ]
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


def everything():
    # use prediction study design
    # training set
    # test set
    # validation set

def histogram(x,title):
    # mu, sigma = 100, 15
    # x = mu + sigma*np.random.randn(10000)
    # the histogram of the data
    mu = np.average(x)
    sigma = np.std(uniques)
    td = dtime.date.today()
    dnow = dtime.date.isoformat(td)
    bin_count = int(np.max(x)/2.25)
    n, bins, patches = plt.hist(x, 8, normed=1, facecolor='green', alpha=0.75)
    # add a 'best fit' line
    # y = mlab.normpdf(bins, mu, sigma)
    # l = plt.plot(bins, y, 'r--', linewidth=1)
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
    plt.axis([0, np.max(x), 0, 0.70])
    plt.grid(True)
    #limit file name length
    title_file = title_file[0:14]
    fname = '../visualizations/' + str(title_file) + '-' + str(dnow) + '.png'
    plt.savefig(fname)
