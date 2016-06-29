import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import datetime as dtime
import sys
import helpers
import time
import math

from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report


# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(rsx2, rsy2)


def extract():
    # read training set
    resource = pd.read_csv('../data/resource_type.csv')
    log_feature = pd.read_csv('../data/log_feature.csv')
    severity_type = pd.read_csv('../data/severity_type.csv')
    event = pd.read_csv('../data/event_type.csv')
    print('Fixing event type')
    event, resource = event.iloc[0:(round(.5*len(event)))], resource.iloc[0:(round(.5*len(resource)))]
    severity_type = severity_type.iloc[0:(round(.5*len(severity_type)))]
    log_feature = log_feature.iloc[0:(round(.5*len(log_feature)))]  
    event = helpers.fix_event_type(event)
    #if dotest:
    test = pd.read_csv('../data/test.csv', index_col='id')
    #if dotrain:
    train = pd.read_csv('../data/train.csv', index_col='id')
    print('Merge resource and event')
    result = pd.merge(resource, event, how="left",on="id")
    print('Merge resource-event and severity type')
    result = pd.merge(result, severity_type, how="left",on="id")
    results = pd.merge(result, log_feature, how="left")
    # results = results.set_index(['id'])
    # retvals = []
    print('Merge all')
    train = pd.merge(train, results, left_index="true", how="left", right_on="id")
    test = pd.merge(test, results, left_index="true", how="left", right_on="id")
    return [train, test]


# maps = list
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


def do_all(results: pd.DataFrame, test: pd.DataFrame, predict: bool):
    start = time.time()
    results, test = helpers.mendgroups(results, test, 'resource_type')
    # results, test = helpers.mendgroups(results,test,'event_type')
    print(len(results.columns))
    print(len(test.columns))
    # results, qmaps1 = slice_n_dice(results, 'location')

    for severity in range(0, 3):
        # location percentile
        dfl = helpers.fsev_count(results, severity, 'location', True, [], [])
        results = dfl[0]
        test = helpers.fsev_count(test, severity, 'location', False, dfl[1], dfl[2])
        # log feature percentile
        dflf = helpers.fsev_count(results, severity, 'log_feature', True, [], [])
        results = dflf[0]
        test = helpers.fsev_count(test, severity, 'log_feature', False, dflf[1], dflf[2])
    print('assigncounts')
    start2 = time.time()
    results = helpers.assigncounts(results)
    testx = helpers.assigncounts(test)
    results = collapse_ids(results)
    testx = collapse_ids(testx)
    end2 = time.time()
    print(end2 - start2)
    print('-----')

    # summarize(results,False, True)
    # summarize(testx,False, False)
    resultsx = results.ix[:, results.columns != 'fault_severity']
    resultsy = results.ix[:, 'fault_severity']
    # resultsx = pd.merge(resultsx,pd.get_dummies(resultsx.event_type),left_index=True,right_index=True,how="left")
    resultsx = pd.merge(resultsx, pd.get_dummies(resultsx.resource_type), left_index=True,right_index=True,how="left")
    resultsx = pd.merge(resultsx, pd.get_dummies(resultsx.severity_type), left_index=True,right_index=True,how="left")
    resultsx = helpers.removecols(resultsx)
    # testx = pd.merge(testx,pd.get_dummies(testx.event_type),left_index=True,right_index=True,how="left")
    testx = pd.merge(testx, pd.get_dummies(testx.resource_type), left_index=True,right_index=True,how="left")
    testx = pd.merge(testx, pd.get_dummies(testx.severity_type), left_index=True,right_index=True,how="left")
    end = time.time()
    print(end - start)
    return resultsx, resultsy, testx


def fits(resultsx, resultsy, tid, testx, predict):
    print('Predictions')
    print('---------')
    start = time.time()
    if 'id' in resultsx.columns:
        tid = testx['id']
        del resultsx['id']
        del testx['id']
        testx = helpers.removecols(testx)
    train_len = math.floor(len(resultsx)*(5/6))
    train = resultsx.iloc[0:train_len]
    trainy = resultsy.iloc[0:train_len]
    holdout = resultsx.iloc[train_len:len(resultsx)]
    holdouty = resultsy.iloc[train_len:len(resultsx)]
    n_est, mdep, lrate = 300, 7, .04
    n_est2, mdep2, lrate2 = 400, 7, .035
    print('GB1')
    print('n_estimators:' + str(n_est) + ' -- mdep: ' + str(mdep) + ' -- lrate: ' + str(lrate))
    startgb1 = time.time()
    tuned_parameters = [
      {'n_estimators': [400], 'max_depth': [3,4], 'learning_rate': [0.1, 0.08]},
      {'n_estimators': [500], 'max_depth': [3,4], 'learning_rate': [0.08, 0.06]},
      {'n_estimators': [700], 'max_depth': [3,4], 'learning_rate': [0.04, 0.02]},
      {'n_estimators': [800], 'max_depth': [3,4], 'learning_rate': [0.03, 0.015]},
    ]
    scores = ['precision', 'recall']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf = GridSearchCV(GradientBoostingClassifier(), tuned_parameters, cv=5,
                           scoring='%s_weighted' % score).fit(train, trainy)
        print("Best parameters set found on development set:")
        print(clf.best_params_)
        print("Grid scores on development set:")
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = holdouty, clf.predict(holdout)
        print(classification_report(y_true, y_pred))
        print()

    clf1 = clf
    endgb1 = time.time()
    print(clf1.score(holdout, holdouty))
    print(endgb1 - startgb1)
    startgb1 = time.time()
    clf3 = RandomForestClassifier(n_estimators=200, max_features='auto').fit(train, trainy)
    # nnet = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1).fit(train, trainy)
    endgb1 = time.time()
    print(endgb1 - startgb1)
    clfsvm = SVC().fit(PolynomialFeatures(2).fit_transform(train), trainy)
    # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    # decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    # max_iter=-1, probability=False, random_state=None, shrinking=True,
    # tol=0.001, verbose=False)
    testx = testx.ix[:,resultsx.columns]
    print('Scores: GBM1, GBM2, RF')
    print(clf3.score(holdout, holdouty))
    print(clfsvm.score(holdout, holdouty))
    #print(nnet.score(holdout, holdouty))
    print('Score on whole train set:')
    print(clf1.score(resultsx, resultsy))
    print(clf3.score(resultsx, resultsy))
    print('Number of columns in train/test:')
    print(len(resultsx.columns))
    print(len(testx.columns))
    hout = [holdout, holdouty, y_pred]
    return clf1, clf3, clfsvm, tid, testx, hout


def feature_imp(clf2,hout,feats):
    holdout = hout[0]
    holdouty = hout[1]
    y_pred = clf2.predict(holdout)
    feature_importance = clf2.feature_importances_[0:feats]
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, holdout.columns[sorted_idx][0:feats])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()


def plot_predict(clf1, clf3, tid, testx, hout, predict):
    param_grid = [
      {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
      {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
     ]
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
        pred3 = clf3.predict_proba(testx)
        df1 = pred_to_df(pred1, testx, tid)
        df3 = pred_to_df(pred3, testx, tid)
        rval = [df1, df3, pred1, pred3]
    else:
        rval = clf1
    end = time.time()
    print(end - start)
    return rval


def pred_to_df(pred1, testx, tid):
    df_sub = pd.DataFrame(pred1)
    df_sub = df_sub.set_index(testx.index)
    print(len(df_sub))
    df_sub['id'] = tid
    df3 = df_sub
    print(len(df3))
    df3 = df3.drop_duplicates('id')
    print(len(df3))
    return df3

def collapse_ids(df):
    df['fsev_log_feature'] = 0
    df['fsev_location'] = 0
    colsf = df['id'].ravel()
    # get unique ids
    unique = pd.Series(colsf).unique()
    # all ids to list
    colsf = colsf.tolist()
    for ii in range(0,len(unique)):
        subset = df.loc[df['id'] == unique[ii]]
        sum_loc = subset['fsev_0_location'].sum()
        sum_loc += subset['fsev_1_location'].sum()
        sum_loc += subset['fsev_2_location'].sum()
        sum_logfeat = subset['fsev_0_log_feature'].sum()
        sum_logfeat += subset['fsev_1_log_feature'].sum()
        sum_logfeat += subset['fsev_2_log_feature'].sum()
        df = df.set_value(subset.index,'fsev_log_feature',(sum_logfeat/len(subset)))
        df = df.set_value(subset.index,'fsev_location',(sum_loc/len(subset)))
    df = df.drop_duplicates('id')
    return df


def collapse_after(df):
    df['fsev_log_feature'] = 0
    df['fsev_location'] = 0
    colsf = df['id'].ravel()
    # get unique ids
    unique = pd.Series(colsf).unique()
    # all ids to list
    colsf = colsf.tolist()
    current_prob = 0
    for ii in range(0,len(unique)):
        subset = df.loc[df['id'] == unique[ii]]
    return 0


# whole DataFrame
# list of top 30
def rowquantilefn(df,list1,bname,bins):
    l_len = len(list1)
    for i in range(0,len(list1)):
        item = str(list1[i])
        rows = df.loc[df[bins] == item]
        percentile = (l_len - i)/(l_len)
        df = df.set_value(rows.index, bname, percentile)
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
def histogram(x: list, title: str, use_np: bool, norm):
    title = str(title)
    title = "More Histograms"
    if use_np:
        x = np.bincount(x)
    mu = np.average(x)
    #sigma = np.std(uniques)
    td, dnow = dtime.date.today(), dtime.date.isoformat()
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
