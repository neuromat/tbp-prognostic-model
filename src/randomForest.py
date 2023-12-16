# This is an implementation of Random Forests as proposed by:
# Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from scipy.stats import mode
import utils
import decisionTree as dt
import logging
from joblib import Parallel, delayed

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
#logging.basicConfig(filename='randomForest.log', filemode='a',format='%(asctime)s - %(message)s', level=logging.DEBUG)


class RandomForest(object):

    def __init__(self, ntrees=300, mtry=np.sqrt, max_depth=None,
        min_samples_split=0.8, bootstrap=0.8, oob_error = True,replace = True,
        missing_branch=False, balance=False,prob_answer = False, cutoff=0.5,
        control_class=None, random_state=9,random_subspace = False):
        
        #TODO min_samples_split is not been used in the code

        # number of trees
        self.ntrees = ntrees
        #  a function that determines how many features will used to build the tree
        self.mtry = mtry
        # defines the depth to which each tree should be grown
        self.max_depth = max_depth
        # fraction of samples to used to grow each tree
        self.bootstrap = bootstrap
        # list of tree objects
        self.forest = []
        # if oob_error is True, then the out-of-bag error of the forest will be calculated
        self.oob_error = oob_error
        # if the random choice of samples should be made with or without replacement
        self.replace = replace
        # if missing_branch = False, then missing values will be handled according to the C4.5 algorithm approach.
        # if missing_branch = True, then a branch will descend from each node of the tree for the missing values
        self.missing_branch = missing_branch
        # if answer should be returned as a final class (prob_answer = False) or a class distribution
        self.prob = prob_answer
        # defines the seed that will be used to randomly select the features for each tree
        self.random_state = random_state
        # if balance = True, then Balanced Random Forests are used
        #   (From "Using Random Forest to Learn Imbalanced Data" (2004) by Chao Chen, Andy Liaw, Leo Breiman).
        # if balance = False, keep the same class proportion for each randomly selected subsample
        self.balance = balance
        # cutoff represents the proportion of votes to consider in case of imbalanced classes.
        # only makes sense for Balanced Random Forests (if balance = False)
        self.cutoff = cutoff
        # control class
        self.control_class = control_class
        self.random_subspace = random_subspace

    # fits a forest for the data X with classes y
    def fit(self, X, y):

        self.forest = []
        self.X = X
        self.attributes = X.columns
        if(isinstance(y,pd.Series)):
            self.y = y.values
        else:
            self.y = y


        n_samples = len(self.y)


        if(self.replace is True):
            n_sub_samples = n_samples
        else:
            n_sub_samples = round(n_samples*self.bootstrap)



        # list of out-of-bag sets
        self.oob = []
        index_oob_samples = np.array([])

        classes = []
        min_len = len(self.y)
        self.min_class = list(set(self.y))[0]
        min_class_index = 0
    # separate samples according to their classes
        for c in set(self.y):


            classes.append([j for j in range(len(y)) if self.y[j] == c])

            if(len(classes[-1]) < min_len):
                min_len = len(classes[-1])
                self.min_class = c
                min_class_index = len(classes)-1
        if(isinstance(self.X,pd.DataFrame)):
            Xfit = self.X.values
            yfit = self.y
        else:
            Xfit = self.X
            yfit = self.y

        self.forest = Parallel(n_jobs=-2,backend="multiprocessing")(delayed(self.create_trees)(n_samples, n_sub_samples, classes, min_class_index, t, Xfit, yfit) for t in range(self.ntrees))

        if self.oob_error is True:
            #logging.info('Calculating oob error...')
            # set of all intances that belong to at least one out-of-bag set
            oob_set = set([j for i in self.forest for j in i.oob])


            ypred = {i: {a:0 for a in set(yfit)} for i in range(self.X.shape[0])}
            for i in range(self.X.shape[0]):
                for t in self.forest:
                    if(i not in t.oob):
                        continue
                    # predict the class (or the class distribution) for instance X[i]
                    tmp = t.predict(Xfit[i].reshape(1,-1),self.prob)[0]
                    # in case of class prediction (not distribution)
                    if(self.prob is False):
                        ypred[i][tmp] += 1
                    # in case of class distribution
                    else:
                        for k in tmp.keys():
                            ypred[i][k] += tmp[k]

                        yp = max(ypred[i].keys(), key= (lambda k: ypred[k]))


            err = 0
            for i in ypred.keys():
                if(self.cutoff==0.5 and list(ypred[i].values())[0] == list(ypred[i].values())[1]):
                    yp = mode(yfit)[0][0]
                else:
                    if(self.balance is False or self.cutoff==0.5):
                        yp = max(ypred[i].keys(), key= (lambda k: ypred[i][k]))
                    else:
                        s = sum(ypred[i].values())
                        k = self.min_class
                        if(k in ypred[i].keys() and ypred[i][k] > self.cutoff*s):
                            yp = k
                        else:
                            yp = max(ypred[i].keys(), key= (lambda k: ypred[i][k]))
                if(yp != yfit[i]):
                    err += 1

            self.oob_error_ = err / len(set(oob_set))


    def create_trees(self, n_samples, n_sub_samples, classes, min_class_index,i,X,y):

        np.random.seed(self.random_state+i)

        # select same proportion of instances from each class
        if(self.balance is False):
            # select indexes of sub samples considering class balance
            index_sub_samples = sorted([k for l in [np.random.choice(a, round(n_sub_samples*(len(a)/n_samples)),
                replace=self.replace) for a in classes] for k in l])
            index_oob_samples = np.delete(np.array(range(n_samples)),index_sub_samples)
        # Balanced Random Forests
        else:
            index_sub_samples = sorted(np.random.choice(classes[min_class_index],len(classes[min_class_index]),replace=True))
            for c in range(len(classes)):
                if(c != min_class_index):
                    if(n_sub_samples-len(classes[min_class_index]) > len(classes[c])):
                        replace = True
                    else:
                        replace = self.replace
                    index_sub_samples = np.append(index_sub_samples,
                        sorted(np.random.choice(classes[c],n_sub_samples-len(classes[min_class_index]),replace=replace)))
            index_oob_samples = np.delete(np.array(range(n_samples)),index_sub_samples)

        X_subset = X[index_sub_samples]
        y_subset = y[index_sub_samples]
        tree = dt.DecisionTreeClassifier(max_depth=self.max_depth,mtry=self.mtry,
            missing_branch=self.missing_branch, random_state=self.random_state+i, random_subspace=self.random_subspace)
        tree.oob = index_oob_samples
        #tree.index = i
        tree.fit(X_subset,y_subset)
        return tree

    def predict(self, X,prob=None):

        if(prob is None):
            prob = self.prob
        if(len(np.array(X).shape) == 1 or np.array(X).shape[0] == 1):
            if(isinstance(X,pd.DataFrame) and len(X) != self.X.shape[1]):
                X = X[X.columns[[np.where(a == X.columns)[0][0] for a in self.X.columns if a in X.columns]]]
                for f in range(len(self.X.columns)):
                    if(self.X.columns[f] not in X.columns):
                        X.insert(f,self.X.columns[f],np.nan)
                X = X.values
            X = [X]
            n_samples = 1
        else:
            n_samples = np.array(X).shape[0]
            if(isinstance(X, pd.DataFrame) and X.shape[1] != self.X.shape[1]):
                X = X[X.columns[[np.where(a == X.columns)[0][0] for a in self.X.columns if a in X.columns]]]
                for f in range(len(self.X.columns)):
                    if(self.X.columns[f] not in X.columns):
                        X.insert(f,self.X.columns[f],[np.nan]*X.shape[0])
                X = X.values

        n_trees = len(self.forest)
        predictions = np.empty(n_samples,dtype=object)
        #y = [{}] * n_samples
        for i in range(n_samples):
            ypreds = []
            for j in range(n_trees):
                ypreds.append(self.forest[j].predict(X[i],prob=False))

            if(prob is False):
                if(self.balance is True and self.cutoff != 0.5 and
                    len([a[0] for a in ypreds if a[0] == self.min_class]) > self.cutoff*len(ypreds)):
                    predictions[i] = self.min_class
                else:
                    predictions[i] = mode(ypreds)[0][0][0]
            else:
                predictions[i] = {c:len([a[0] for a in ypreds if a[0] == c]) for c in set(self.y)}


        return predictions


    def score(self, X, y):

        y_predict = self.predict(X)

        # n_samples = len(y)
        # if(isinstance(y,str)):
        #     y = [y]
        #     n_samples = 1

        if(isinstance(y,int) or isinstance(y,float)):
            y = [y]
            n_samples = 1
        else:
            n_samples = len(y)

        correct = 0
        for i in range(n_samples):
            if y_predict[i] == y[i]:
                correct = correct + 1
        accuracy = correct/n_samples
        return accuracy

    # this function implements the method proposed in:
    # Palczewska, A., Palczewski, J., Robinson, R. M., & Neagu, D. (2013).
    # Interpreting random forest models using a feature contribution method.
    # In 2013 IEEE 14th International Conference on Information Reuse and Integration (pp. 112–119).
    # Retrieved from http://eprints.whiterose.ac.uk/79159/1/feature_contribution_camera_ready.pdf
    def feature_contribution(self,X=None):
        #logging.info('Calculating feature contribution...')
        #C = set(self.y)
        if(X is None):
            if(isinstance(self.X,pd.DataFrame)):
                X = self.X.values
            else:
                X = self.X
        else:
            if(isinstance(X, pd.DataFrame) and X.shape[1] != self.X.shape[1]):
                X = X[X.columns[[np.where(a == X.columns)[0][0] for a in self.X.columns if a in X.columns]]]
                for f in range(len(self.X.columns)):
                    if(self.X.columns[f] not in X.columns):
                        X.insert(f,self.X.columns[f],[np.nan]*X.shape[0])
            X = X.values
            
        if(self.control_class is None):
            control_class = list(set(self.y))[0]
            logging.info('Control class set as %r' % control_class)
        else:
            control_class = self.control_class


        fcs = []

        for i in range(X.shape[0]):

            FC = {}
            c = 0
            #for k in C:
            t_index = 0
            # if(i_index == 9):
            #     import pdb
            #     pdb.set_trace()

            for t in self.forest:
                if(i in self.forest[t_index].oob):
                    t_index+=1
                    continue

                t_index +=1
                child_list = [[1,t.root]]


                while len(child_list) > 0:
                    w, parent = child_list.pop(0)

                    while parent.is_class is False:
                        f = parent.feature_index

                        if(f not in FC.keys()):
                            FC[f] = 0
                        #    FC[f] =  {c:0 for c in C}

                        if(utils.isnan(X[i][f])):
                            if(parent.branch_nan is None):
                                sp = sum(parent.distr.values())
                                for c in parent.branches:
                                    child_list.append([round(w*(sum(c.distr.values()))/sp,2),c])
                                w,child = child_list.pop(0)
                            else:
                                child = parent.branch_nan
                        else:
                            if(len(parent.values) == 1):
                                if X[i][f] <= parent.values[0]:
                                    child = parent.branches[0]
                                else:
                                    child = parent.branches[1]
                            else:
                                if(str(X[i][f]) not in parent.values):
                                    if(parent.branch_nan is None):
                                        sp = sum(parent.distr.values())
                                        for c in parent.branches:
                                            child_list.append([round(w*(sum(c.distr.values()))/sp,2),c])
                                        w,child = child_list.pop(0)
                                    else:
                                        child = parent.branch_nan

                                else:

                                    child = parent.branches[parent.values.index(str(X[i][f]))]


                        sc = sum(child.distr.values())
                        if(sc == 0):
                            child.distr = t.root.distr
                            sc = sum(child.distr.values())
                        sp = sum(parent.distr.values())

                        FC[f] = FC[f] + w*(child.distr[control_class]/sc - parent.distr[control_class]/sp)

                        parent = child

            for element in FC:
                FC[element] = FC[element] / self.ntrees
                #for el in FC[element]:
                #    FC[element][el] = FC[element][el] / self.ntrees

            fcs.append(FC)
        return fcs

    # variable importance calculation for Random Forests.
    # --- when vitype='err' and vimissing=False, then calculation is made as proposed in:
    #       Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
    # --- when vitype='err' and vimissing=True, then calculation is made as proposed in:
    #       Hapfelmeier, A., & Ulm, K. (2014). Variable selection by Random Forests using data with missing values.
    #         Computational Statistics and Data Analysis, 80, 129–139. https://doi.org/10.1016/j.csda.2014.06.017
    # --- when vitype='auc' and vimissing=False, then calculation is made as proprosed in:
    #       Janitza, S., Strobl, C., & Boulesteix, A.-L. (2012).
    #       An AUC-based Permutation Variable Importance Measure for Random Forests.
    #           Retrieved from http://www.stat.uni-muenchen.de
    # --- when vitype='auc' and vimissing=True, then calculation is made by joining the two methods above
    def variable_importance(self,vitype='err',vimissing=True,y=None):

        if(y is None):
            y = self.y
        if(vitype == 'auc'):
            ntreesc = 0
        else:
            ntreesc = self.ntrees

        variable_importance = {attribute: 0 for attribute in range(self.X.shape[1])}
        for t in self.forest:
            for m in range(len(self.X.columns)):#t.feature_indices:
                if(m in t.feature_indices):
                    X_permuted = self.X.copy().values
                    if(vimissing is False):
                        np.random.shuffle(X_permuted[:,m])
                        sa = None
                    else:
                        sa = m

                    if(vitype == 'auc'):
                        if(len(set(y[t.oob])) > 1):
                            ntreesc += 1
                            auc_before = t.auc(self.X.values[t.oob],y.values[t.oob],shuffle_attribute=None,control_class=self.control_class)
                            auc = t.auc(X_permuted[t.oob],y.values[t.oob],shuffle_attribute=sa,control_class=self.control_class)
                            variable_importance[m] += auc_before - auc

                    else:
                        err = 1-t.score(self.X.values[t.oob],y[t.oob],shuffle_attribute=None)
                        err_permuted = 1 - t.score(X_permuted[t.oob], y[t.oob],shuffle_attribute=sa)
                        variable_importance[m] += (err_permuted - err)


        return {a:b/ntreesc for a,b in variable_importance.items()}

    # This method implements the d2 algorithm proposed in:
    # Banerjee, M., Ding, Y., Noone, A. (2012).
    # Identifying representative trees from ensembles
    def representative_trees(self,attributes,title):
        #logging.info('Calculating representative trees...')
        min_dif = 1
        rep_trees = {i: 0 for i in range(self.ntrees)}
        for t1 in range(self.ntrees):
            for t2 in range(t1+1,self.ntrees):
                dif = 0
                c = 0
                for i in range(self.X.shape[0]):
                    if(i in self.forest[t2].oob or i in self.forest[t1].oob):
                        continue
                    pred = self.forest[t1].predict(self.X[i],prob=True)[0]
                    y1 = pred[self.control_class]/sum(pred.values())
                    pred2 = self.forest[t2].predict(self.X[i],prob=True)[0]
                    y2 = pred2[self.control_class]/sum(pred.values())
                    dif += (y1-y2)**2
                    c += 1
                dif = dif/c
                rep_trees[t1] += dif
                rep_trees[t2] += dif


        reps = [a for a in rep_trees.keys() if rep_trees[a] in sorted(rep_trees.values())[0:5]]
        logging.info(reps)
        for r in reps:
            self.forest[r].to_pdf(out='representative_tree_'+str(r)+title+'.pdf',attributes=attributes)
        return rep_trees
