from email.mime import base
from tkinter import Y
import numpy as np
import math
import json
from sklearn import tree
from scipy.stats import sem
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel
import logging
import os
import shutil
import pickle
from pathlib import Path
from decimal import *

# Local imports
import randomForest as rf
import read
import utils
import plot
import report


logging.getLogger('matplotlib.font_manager').disabled = True
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.getLogger('PIL').setLevel(logging.WARNING)

#logging.basicConfig(filename='preprocessing.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.DEBUG)

OUTPUT_PATH_ROOT = 'output'
OUTPUT_GRAPH_DIR = 'graphs'
OUTPUT_DATA_DIR = 'data'


# this method is based on the algorithm proposed in:
# Robin Genuer, Jean-Michel Poggi, and Christine Tuleau-Malot. 2010.
# Variable selection using random forests. Pattern Recogn. Lett. 31, 14 (October 2010), 2225-2236.
# DOI=http://dx.doi.org/10.1016/j.patrec.2010.03.014

def feature_selection_threshold(X, y, ntrees, replace, mtry, max_depth, missing_branch, balance,
                                cutoff, class_name, ntimes=25, missing_rate=False, vitype='err', vimissing=True,
                                backwards=False, save_models=False, random_subspace=False, output_path='.'):

    if save_models:
        models_path = os.path.join(OUTPUT_PATH_ROOT,'models', output_path)
        Path(models_path).mkdir(parents=True, exist_ok=True)     
        output_data_path = os.path.join(OUTPUT_PATH_ROOT,OUTPUT_DATA_DIR,output_path,class_name)
        Path(output_data_path).mkdir(parents=True, exist_ok=True)

    # get average feature importances for each feature
    vis,vimean = average_varimp(X, y, ntrees, replace, mtry, max_depth, missing_branch, balance=balance,
                         missing_rate=missing_rate, ntimes=ntimes, select=False, mean=False, class_name=class_name, vitype=vitype,
                         vimissing=vimissing, printvi=True, plotvi=save_models, random_subspace=random_subspace, output_path=output_data_path)

    # if backwards is True, then the feature selection will start the process with all features,
    # and eliminating the least important ones in each step
    if(backwards is True):
        reverse = False
        def comp_threshold(x, y): return x <= y
        def get_slice(x, index): return x[index:]
        stop_index = -1
        chosen_model = -1   # last model
    # if it's False, then it starts with only the most important feature, and then adding
    # the most important ones in each step
    else:
        reverse = True
        def comp_threshold(x, y): return x > y
        def get_slice(x,index): return x[0:index]
        stop_index = 0
        chosen_model = 0   # first model

    sorted_vis = sorted(vis, key=lambda x: np.mean(x[1]), reverse=reverse)
    # Get the feature ordered according the feature mean importance
    ordered_features = [a[0] for a in sorted_vis]
    # Get a list of the mean importances of the features
    thresholds = [np.mean(vis[a][1]) for a in ordered_features]
    # Get a list of the rounded mean importances of the features without repetitions
    #threshold_values = sorted([math.round(a, 10) for a in set(thresholds)], reverse=reverse)
    threshold_values = sorted([Decimal(a).quantize(Decimal('.0000000001'), rounding=ROUND_UP) if a < 0
                                else Decimal(a).quantize(Decimal('.0000000001'), rounding=ROUND_DOWN)
                              for a in set(thresholds)], reverse=reverse)

    stop_indexes = []
    scores = []
    features_contributions = []
    nn = 0
    # for each threshold value (feature importance value), create a forest
    # only using: (a) features whose importance value is <= than the threshold
    # if backwards is True (starting from the least important), or
    # (b) features whose importance value is > than the threshold if backwards is False
    # (starting from the most important one)
    for threshold in threshold_values:
        s_index = stop_index+1
        while s_index < len(thresholds):
            if(comp_threshold(threshold, thresholds[s_index])):
                break
            else:
                s_index += 1
        stop_index = s_index
        stop_indexes.append(stop_index)

        seed = np.random.randint(0, 10000)
        clf = rf.RandomForest(ntrees=ntrees, oob_error=True, random_state=seed, mtry=mtry,
                              missing_branch=missing_branch, prob_answer=False, max_depth=max_depth, replace=replace, balance=balance,
                              cutoff=cutoff, control_class=1, random_subspace=random_subspace)

        selected_features = X.columns[get_slice(ordered_features, stop_index)]

        clf.fit(X[selected_features], y)
        clf.threshold = threshold

        scores.append(1-clf.oob_error_)
        features_contributions.append(clf.feature_contribution())

        # Save the model
        model_file_path = os.path.join(
            models_path, class_name + '_' + str(nn) + '.pickle')
        with open(model_file_path, 'wb') as handle:
            pickle.dump(clf, handle)
        nn += 1

    stdm = sem(scores)
    max_scores = max(scores)
    scores_aux = (np.abs(np.array([a for a in scores if a != max_scores])- (max_scores - stdm)))

    # The forest with the best score (closest to the max score subtracted from the standard error of scores) and
    # the biggest threshold value (by index -1) is chosen as the best model
    if len(scores_aux) == 0:
        best_model_index = chosen_model
    else:
        indexes = np.where(np.array(scores) == scores[(scores_aux.argmin())])[0]
        best_model_index = indexes[chosen_model]

    # Get the features used to train the best model
    selected_features = list(X.columns[get_slice(ordered_features, stop_indexes[best_model_index])])
    logging.info('features selected for the best model:' + str(selected_features))

    if(save_models is True):
        threshold_values = [float(t) for t in threshold_values]
        plot.save_feat_importance_vs_threshold_accuracy_data(
            output_data_path, threshold_values, scores, class_name=class_name, special=best_model_index)

        # Plot the importance of the selected features
        importance_values = [a[1] for a in get_slice(sorted_vis, stop_indexes[best_model_index])]
        plot.save_feature_importance(output_data_path,importance_values, selected_features, class_name,True)

        # Plot the contributions of the values of the selected features in the best model
        fcs = features_contributions[best_model_index]
        plot.save_all_feature_contributions(output_data_path,X[selected_features],y,fcs,1,class_name)

    return selected_features, vimean


def average_varimp(X, y, ntrees, replace, mtry, max_depth, missing_branch, balance, vitype='err', vimissing=True, ntimes=25,
                   select=True, printvi=False, plotvi=False, cutpoint=0.0, mean=False, class_name=None, missing_rate=False,
                   random_subspace=False, output_path='.'):

    vi = {a: [] for a in range(X.shape[1])}

    for i in range(ntimes):

            seed = np.random.randint(0, 10000)
            clf = rf.RandomForest(ntrees=ntrees, oob_error=True, random_state=seed, mtry=mtry,
                                  missing_branch=missing_branch, prob_answer=False, max_depth=max_depth,
                                  replace=replace, control_class=1, balance=balance, random_subspace=random_subspace)
            clf.fit(X, y)
            varimps = clf.variable_importance(vitype=vimissing, vimissing=True)
            for var in varimps.keys():
                if(missing_rate):
                    vi[var].append(
                        varimps[var] * utils.notNanProportion(X[X.columns[var]]))
                else:
                    vi[var].append(varimps[var])
        # else:
        #     break

    vimean = {a: [] for a in range(X.shape[1])}
    for var in vi.keys():
        vimean[var] = np.mean(vi[var])

    if(printvi):
        vis = sorted(vimean.items(), key=lambda x: x[1], reverse=True)
        for v, i in vis:
            logging.info('feature: %r importance: %r' % (X.columns[v], i))

    if(plotvi):
        #logging.info('cutpoint for feature importance: %r' %cutpoint)
        importance_values = []
        features = []
        vis = sorted(vi.items(), key=lambda x: x[0])
        for v, i in vis:
            # if(vimean[v] >= cutpoint):
            importance_values.append(i)
            features.append(X.columns[v])
        plot.save_feature_importance(output_path, importance_values, features, class_name, False)

    if(select):
        vis = sorted(vimean.items(), key=lambda x: x[1], reverse=True)
        return sorted([var[0] for var in vis if vimean[var[0]] >= cutpoint])
    if(mean):
        return sorted(vimean.items(), key=lambda x: x[1], reverse=True)
        # return [var[0] for var in vis]

    return sorted(vi.items(), key=lambda x: x[0]), vimean


def generateForests(data_path, classes, ntimes = 25, ntrees = 5001,  output_path='.'):

    missing_input = 'none'  # 'mean'
    transform = False
    scale = True
    use_text = False
    dummy = False
    use_feature_selection = False

    import random
    seed = random.randint(0, 10000)

    for class_name, is_included in classes.items():
        if is_included:
            logging.info('####################################################')
            logging.info('MODEL: %r' % (class_name))
            logging.info('####################################################')

            data_file_path = os.path.join(data_path, class_name+'_data.csv')

            X = read.readData(data_path=data_file_path, class_name=class_name, missing_input=missing_input, dummy=dummy,
                              transform_numeric=transform, use_text=use_text)

            y = X[class_name]
            del(X[class_name])
            del(X['participant_code'])
            # TODO avisar a Cris que eu tirei essas colunas do preenchimento pq elas só tem NINA e Y
            #X = X.drop(['yonPexPainTime1_v2','yonPexPainTime2_v2'], axis=1, errors='ignore')

            # Considera classes desbalanceadas se a proporção for menor que 1 : 2
            counts = y.value_counts()            
            balance = (counts.max() / counts.min() >= 2)
            logging.info('Balance: %r ' % balance)           

            mtry = math.sqrt
            max_depth = None
            missing_branch = True
            seed = 89444
            replace = False
            vitype = 'auc'
            cutoff = 0.5

            # TODO organizar esses parâmetros, garantir que são os mesmos nas funções, evitar passar separados
            clf = rf.RandomForest(ntrees=ntrees, oob_error=True, random_state=seed, mtry=mtry,
                                  missing_branch=missing_branch, prob_answer=False, max_depth=max_depth, replace=replace, balance=balance,
                                  cutoff=cutoff, control_class=1, random_subspace=True)

            selected_features, features_importance = feature_selection_threshold(X, y, ntrees, replace, mtry, max_depth, missing_branch, balance,
                                                            cutoff, class_name, ntimes=ntimes, missing_rate=True, vitype=vitype, vimissing=True,
                                                            backwards=True, save_models=True, random_subspace=True, output_path=output_path)

            # Train and evaluate the model using LOOCV and the selected features
            metrics = leaveOneOutCrossValidation(clf, X[selected_features], y)

            metrics['number of initial attributes'] = X.shape[1]
            attributes_names = list(X.columns)
            attributes_names.sort()
            metrics['initial attributes, % of missing'] = {}
            metrics['initial attributes, importance'] = {}
            missing = X.isna().mean() * 100

            for col in attributes_names:
                metrics['initial attributes, % of missing'][col] =  missing[col]
                i = X.columns.get_loc(col)
                metrics['initial attributes, importance'][col] = features_importance[i]

            models_data_path = os.path.join(OUTPUT_PATH_ROOT,'models',output_path)

            # Saves de final model
            model_file_path = os.path.join(
                models_data_path, 'final_prognostic_model_' + class_name + '.pickle')
            with open(model_file_path, 'wb') as handle:
                pickle.dump(clf, handle)
            
            model_file_path = os.path.join(
                models_data_path, 'final_prognostic_model_' + class_name + '_metrics.json')
            with open(model_file_path, 'w') as handle:
                json.dump(metrics, handle, ensure_ascii=False)

            # accuracy_threshold_path = os.path.join(OUTPUT_PATH_ROOT,'accuracy_threshold', output_path)
            # Path(accuracy_threshold_path).mkdir(parents=True, exist_ok=True)        
            # file_path = os.path.join(
            #     accuracy_threshold_path, class_name + '_field_relevance.json')
            output_data_path = os.path.join(OUTPUT_PATH_ROOT, OUTPUT_DATA_DIR, output_path)
            Path(output_data_path).mkdir(parents=True, exist_ok=True)        
            file_path = os.path.join(
                 output_data_path, 'field_relevance__'+class_name+'.json')
            with open(file_path, 'w') as handle:
                json.dump(metrics['initial attributes, importance'], handle, ensure_ascii=False)                


def leaveOneOutCrossValidation(clf, X, y):
    scores = 0
    s = []
    ivp, ifp, ifn, ivn, svp, sfp, sfn, svn = 0, 0, 0, 0, 0, 0, 0, 0

    # leave-one-out cross validation (LOOCV)
    for i in range(X.shape[0]):
        Xtrain = X.drop(i)  # np.concatenate([X[0:i],X[i+1:]])
        ytrain = y.drop(i)  # np.concatenate([y[0:i],y[i+1:]])
        clf.fit(Xtrain, ytrain)

        if(clf.predict(X.loc[i]) == 1):  # 'SUCCESS'
            if(y[i] == 1):   # 'SUCCESS'
                svp += 1
                ivn += 1
            else:
                sfp += 1
                ifn += 1
        else:
            if(y[i] == 1):   # 'SUCCESS'
                sfn += 1
                ifp += 1
            else:
                svn += 1
                ivp += 1

        scores += clf.score(X.loc[i], y.loc[i])
        s.append(clf.score(X.loc[i], y.loc[i]))

    metrics = {'number of trainning instances': X.shape[0]}
    metrics['number of selected attributes'] = X.shape[1]
    attributes_names = list(X.columns)
    attributes_names.sort()
    metrics['selected attributes'] =  attributes_names
    metrics['class attribute'] = y.name
    metrics['accuracy (%)'] = scores/X.shape[0] * 100
    metrics['accuracy standard deviation'] = np.std(s)
    logging.info('accuracy: %r%%' % metrics['accuracy (%)'])
    logging.info('deviation: %r' % metrics['accuracy standard deviation'])

    # precision for SATISFACTORY RECOVERY
    if svp+sfp == 0:
        p = 0
    else:
        p = svp/(svp+sfp)

    # recall for SATISFACTORY RECOVERY
    if (svp+sfn) == 0:
        c = 0
    else:
        c = svp/(svp+sfn)

    # specificity for SATISFACTORY RECOVERY
    if (svn + sfp ) == 0:
        s = 0
    else:
        s = svn/(svn + sfp)

    # f-measure for SATISFACTORY RECOVERY
    if(p + c == 0):
        f = 0
    else:
        f = (2*p*c)/(p+c)

    logging.info(
        'SATISFACTORY RECOVERY --- recall: %r precision: %r F-measure: %r ' % (c, p, f))

    metrics['true positive'] = svp
    metrics['false positive'] = sfp
    metrics['true negative'] = svn
    metrics['false negative'] = sfn
    metrics['satisfactory recovery - precision'] = p
    metrics['satisfactory recovery - recall'] = c
    metrics['satisfactory recovery - specificity'] = s
    metrics['satisfactory recovery - f-measure'] = f

    # precision for UNSATISFACTORY RECOVERY
    if (ivp+ifp) == 0:
        p = 0
    else:
        p = ivp/(ivp+ifp)

    # recall for UNSATISFACTORY RECOVERY
    if (ivp+ifn) == 0:
        c = 0
    else:
        c = ivp/(ivp+ifn)

    # specificity for UNSATISFACTORY RECOVERY
    if (ivn + ifp ) == 0:
        s = 0
    else:
        s = ivn/(ivn + ifp)

    # f-measure for UNSATISFACTORY RECOVERY
    if(p + c == 0):
        f = 0
    else:
        f = (2*p*c)/(p+c)

    metrics['unsatisfactory recovery - precision'] = p
    metrics['unsatisfactory recovery - recall'] = c
    metrics['unsatisfactory recovery - specificity'] = s
    metrics['unsatisfactory recovery - f-measure'] = f

    logging.info(
        'UNSATISFACTORY RECOVERY --- recall: %r precision: %r F-measure: %r ' % (c, p, f))

    return metrics


if __name__ == '__main__':
    # Complete Execution (It takes a few hours to conclude.)
    target_classes = {'yonPexPain': True, 'lstPexMuscstrength_ElbowFlex': True,
                       'lstPexMuscstrength_ShoulderAbduc': True, 'lstPexMuscstrength_ShoulderExrot': True}
    target_origins = {'indc':True, 'into':True,'all':True}
    groups = range(1,7)
    ntimes = 25
    ntrees = 5001
      
    # Test Execution
    # target_classes = {'yonPexPain': True, 'lstPexMuscstrength_ElbowFlex': True,
    #                   'lstPexMuscstrength_ShoulderAbduc': True, 'lstPexMuscstrength_ShoulderExrot': True}
    # target_origins = {'indc':True, 'into':False,'all':False,'nina':False}
    # groups = range(6,7)  
    # ntimes = 2
    # ntrees = 5
    
    # Remove output dir if it already exists
    shutil.rmtree(OUTPUT_PATH_ROOT,ignore_errors=True)

    # Train models
    for origin, value in target_origins.items():
        if value:
            for i in groups:
                logging.info('####################################################')
                logging.info('GROUP %d --- ORIGIN: %s' % (i, origin))
                input_data_path = os.path.join(
                    './training_data/group%d' % i, origin)
                group_path = os.path.join('group%d' % i, origin)

                generateForests(input_data_path, target_classes, ntimes, ntrees, group_path)

    # Generate models' plots and performance report
    out_data_path = os.path.join(OUTPUT_PATH_ROOT,OUTPUT_DATA_DIR)
    out_graph_path = os.path.join(OUTPUT_PATH_ROOT,OUTPUT_GRAPH_DIR)
    plot.create_model_plots(out_data_path,out_graph_path)

    report.generateModelReports()