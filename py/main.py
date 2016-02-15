import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import datetime as dtime
import sys
import helpers
import time
import math

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_curve, auc

# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(rsx2, rsy2)

def extract(test):
    # read training set
    resource = pd.read_csv('../data/resource_type.csv')
    log_feature = pd.read_csv('../data/log_feature.csv')
    severity_type = pd.read_csv('../data/severity_type.csv')
    event = pd.read_csv('../data/event_type.csv')
    if test:
        train = pd.read_csv('../data/test.csv', index_col='id')
    else:
        train = pd.read_csv('../data/train.csv', index_col='id')
    result = pd.merge(resource, event, how="left",on="id")
    results = pd.merge(result,log_feature, how="left")
    #results = results.set_index(['id'])
    results = pd.merge(train,results,left_index="true", how="left",right_on="id")
    # results = pd.merge(train,results,left_index="true", right_index="true", how="left")
    # results_2 = pd.merge(train,results,left_index="true", right_index="right" how="left", on="id")
    return results

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

def map_to(test, maps, bin):
    bins = str(bin)
    b_name0 = ''.join([bin,'_sev0_top20'])
    b_name1 = ''.join([bin,'_sev1_top20'])
    b_name2 = ''.join([bin,'_sev2_top20'])
    names = [b_name0, b_name1, b_name2]
    test[b_name0] = 0
    test[b_name1] = 0
    test[b_name2] = 0
    for j in range(0,len(maps)):
        cmap = maps[j]
        for i in range(0,len(cmap)):
            cbin = cmap[i]
            a = test[bin].isin([cbin[0]])
            test = test.set_value(a, names[j], cbin[1])
    return test


def do_all(results,test, predict):
    start = time.time()
    results, test = helpers.mendgroups(results,test,'resource_type')
    results, test = helpers.mendgroups(results,test,'event_type')
    print(len(results.columns))
    print(len(test.columns))
    results, qmaps1 = slice_n_dice(results, 'location')
    results, qmaps2 = slice_n_dice(results, 'log_feature')
    print('assigncounts')
    start2 = time.time()
    results = helpers.assigncounts(results)
    test = helpers.assigncounts(test)
    end2 = time.time()
    print(end2 - start2)
    print('-----')
    test = map_to(test, qmaps1, 'location')
    testx = map_to(test, qmaps2, 'log_feature')
    summarize(results,False, True)
    summarize(testx,False, False)
    resultsx = results.ix[:,results.columns != 'fault_severity']
    resultsy = results.ix[:,'fault_severity']
    resultsx = pd.merge(resultsx,pd.get_dummies(resultsx.event_type),left_index=True,right_index=True,how="left")
    resultsx = pd.merge(resultsx,pd.get_dummies(resultsx.resource_type),left_index=True,right_index=True,how="left")
    resultsx = helpers.removecols(resultsx)
    testx = pd.merge(testx,pd.get_dummies(testx.event_type),left_index=True,right_index=True,how="left")
    testx = pd.merge(testx,pd.get_dummies(testx.resource_type),left_index=True,right_index=True,how="left")
    end = time.time()
    print(end - start)
    return resultsx, resultsy, testx


def fits(resultsx,resultsy,tid,testx,predict):
    print('Predictions')
    print('---------')
    start = time.time()
    #tid = testx['id']
    # del resultsx['id']
    # del testx['id']
    # testx = helpers.removecols(testx)
    train_len = math.floor(len(resultsx)*(3/4))
    train = resultsx.iloc[0:train_len]
    trainy = resultsy.iloc[0:train_len]
    holdout = resultsx.iloc[train_len:len(resultsx)]
    holdouty = resultsy.iloc[train_len:len(resultsx)]
    n_est, mdep, lrate = 700, 5, .015
    n_est2, mdep2, lrate2 = 500, 5, .02
    print('GB1')
    print('n_estimators:' + str(n_est) + ' -- mdep: ' + str(mdep) + ' -- lrate: ' + str(lrate))
    startgb1 = time.time()
    clf1 = GradientBoostingClassifier(n_estimators=n_est, max_depth=mdep, learning_rate=lrate, max_features='auto', random_state=0).fit(train, trainy)
    endgb1 = time.time()
    print(endgb1 - startgb1)
    print('GB2')
    print('n_estimators:' + str(n_est2) + ' -- mdep: ' + str(mdep2) + ' -- lrate: ' + str(lrate2))
    startgb1 = time.time()
    clf2 = GradientBoostingClassifier(n_estimators=n_est, max_depth=mdep2, learning_rate=lrate2,random_state=1).fit(train, trainy)
    endgb1 = time.time()
    print(endgb1 - startgb1)
    startgb1 = time.time()
    clf3 = RandomForestClassifier(n_estimators=800, max_features='auto').fit(train, trainy)
    #nnet = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1).fit(train, trainy)
    endgb1 = time.time()
    print(endgb1 - startgb1)
    testx = testx.ix[:,resultsx.columns]
    print('Scores: GBM1, GBM2, RF')
    print(clf1.score(holdout, holdouty))
    print(clf2.score(holdout, holdouty))
    print(clf3.score(holdout, holdouty))
    #print(nnet.score(holdout, holdouty))
    print('Score on whole train set:')
    print(clf1.score(resultsx, resultsy))
    print(clf2.score(resultsx, resultsy))
    print(clf3.score(resultsx, resultsy))
    y_pred = clf2.predict(holdout)
    print('Number of columns in train/test:')
    print(len(resultsx.columns))
    print(len(testx.columns))
    hout = [holdout, holdouty, y_pred]
    return clf1, clf2, clf3, tid, testx, hout


def plot_predict(clf1, clf2, clf3, tid, testx, hout, predict):
    predict = True
    start = time.time()
    [holdout, holdouty, y_pred] = hout
    testx = testx.fillna(value=0)
    test_score = np.zeros((500,), dtype=np.float64)
    # for i, y_pred in enumerate(clf2.staged_predict(holdout)):
    #     test_score[i] = clf2.loss_(y_pred,holdouty)
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.title('Deviance')
    # plt.plot(np.arange(params['n_estimators']) + 1, clf2.train_score_, 'b-',
    #          label='Training Set Deviance')
    # plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
    #          label='Test Set Deviance')
    # plt.legend(loc='upper right')
    # plt.xlabel('Boosting Iterations')
    # plt.ylabel('Deviance')
    if predict:
        pred1 = clf1.predict_proba(testx)
        pred2 = clf2.predict_proba(testx)
        pred3 = clf3.predict_proba(testx)
        df1 = pred_to_df(pred1,testx,tid)
        df2 = pred_to_df(pred2,testx,tid)
        df3 = pred_to_df(pred3,testx,tid)
        rval = [df1, df2, df3, pred1, pred2, pred3]
    else:
        rval = clf1
    end = time.time()
    print(end - start)
    return rval

def pred_to_df(pred1, testx, tid):
    df_sub = pd.DataFrame(pred1)
    df_sub = df_sub.set_index(testx.index)
    df_sub['id'] = tid
    df3 = df_sub
    df3 = df3.drop_duplicates('id')
    return df3

def rowquantilefn(df,list1,bname,bins):
    l_len = len(list1)
    for i in range(0,len(list1)):
        item = str(list1[i])
        rows = df.loc[df[bins] == item]
        percentile = (l_len - i)/(l_len)
        df = df.set_value(rows.index,bname,percentile)
    return df

def summarize(results, binning, train):
    # get number of unique locations
    # number of unique features
    # histogram of # of events by location
    locations, event_types = results.location.ravel(), results.event_type.ravel()
    ids, resources = results.id.ravel(), results.resource_type.ravel()
    if train:
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
    print('--------------------')
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
