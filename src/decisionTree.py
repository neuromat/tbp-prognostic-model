import numpy as np
import pandas as pd
import utils
from collections import Counter, deque
from scipy.stats import mode
from sklearn.metrics import roc_curve as roc,auc
import math
import random


class Node(object):
    
    # A node object of a decision tree
    def __init__(self, feature_index, values,branches,branch_nan,sample_size,distr,is_class=False,final_class=None):
        
        # the index of the test feature at the node 
        self.feature_index = feature_index
        # list of values for each branch from the node to its children 
        self.values = values
        # list of children nodes (object) that descend from each branch
        self.branches = branches
        # the node descending from the branch corresponding to missing values
        self.branch_nan = branch_nan
        # number os intances at the node
        self.sample_size = sample_size
        # if is_class is True, then the node corresponds to a final node.
        self.is_class = is_class
        # final_class -- may be useless
        self.final_class = final_class
        # class distribution at the node
        self.distr = distr



class DecisionTreeClassifier(object):

    def __init__(self, max_depth=None,
                    min_samples_split=2,missing_branch = True, mtry=None,random_state=9,random_subspace=False):

        # a function that determines how many features will used to build the tree
        
        self.mtry = mtry

        # defines the depth to which the tree should be grown  
        self.max_depth = max_depth
        # if missing_branch = False, then missing values will be handled according to the C4.5 algorithm approach.
        # if missing_branch = True, then a branch will descend from each node of the tree for the missing values
        self.missing_branch = missing_branch
        # defines the seed that will be used to randomly select the features 
        self.random_state = random_state

        # list of lists of attributes in every tree 
        self.feature_indices = []

        self.random_subspace = random_subspace

    # fits a tree for the data X with classes y
    def fit(self, X, y):
        if(isinstance(X,pd.DataFrame)):
            X = X.values
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # if the mtry is None, then use all the features
        
        if(self.mtry is None or self.random_subspace is False):
            n_sub_features = n_features
            self.feature_indices = np.arange(n_features)
        else:
            n_sub_features = int(self.mtry(n_features))        
            random.seed(self.random_state)
            self.feature_indices = random.sample(range(n_features), n_sub_features)
              

        # builds tree from the root
        self.root = self.build_tree(X,y,self.feature_indices,0,dict([[a,1] for a in range(n_samples)]))


    # runs every instance of data X through the tree and returns the predicted class.
    # if prob=True, then return the distribution for each class at the final node. 
    # if prob=False, then return the most frequent class at the final node.
    # if shuffle_atribute is not None, then it predicts the classes of the instances 
    # with their values permuted for the shuffle_attribute
    def predict(self, X,prob=False,shuffle_attribute=None):
        
        # len(np.array(X).shape) = 1 when there's only one instance on data X
        if(len(np.array(X).shape)) == 1: 
            X = [X]
            num_samples = 1
        else:
            num_samples = np.array(X).shape[0]

        y = np.empty(num_samples,dtype=object)

        # for each sample
        for j in range(num_samples):

            # runs instance X[j] through the tree, starting from the root
            # and gets the class distribution (that is a dict of the form {class: frequency})
            # at the final node.
            p = self.predict_rec(X[j],self.root,shuffle_attribute)

            # if prob is False, then return the most frequent class at the final node
            if(prob is False):
                
                # len(set(p.values())) will be one when there's a tie.
                # in that case, the most probable class at the root will be assigned 
                # to the instance   
                if(len(list(p.values())) > 1 and len(set(p.values())) == 1):
                    y[j] = max(self.root.distr.keys(), key = lambda k: self.root.distr[k])
                # returns the most frequent class at the final node
                else:
                    y[j] = max(p.keys(), key=(lambda k: p[k]))
            # if prob is False, then return the class distribution
            else:
                y[j] = p 

        # returns the list of predictions for each instance of the data X
        return y

    # recursively predicts the class of instance X, starting from the root.
    # if the node is class node, then it returns the final class distribution at the node.
    # if it's not a class node, than it returns the corresponding child, according to the
    # value of X for the node feature. 
    def predict_rec(self,X,node,shuffle_attribute=None):

        # if the node is a class node, then it should return the class distribution
        if node.is_class:

            d = {}
            # sum of class distributions (absolute values)
            s = sum(node.distr.values())
            # for each class
            for k in node.distr.keys():
                # if s == 0, then there are no instances at the node - which means that 
                # it's a decision node coming from a nan branch (branch_nan)
                if(s == 0):
                    # adds 1 to the final_class (most probable one) in case of a possible 
                    # future classification of an instance that ends up at this final node
                    d[node.final_class] = 1
                    return d
                # returns the node distribution (relative values)
                else:
                    d[k] = node.distr[k] / s

            return d
        
        # if the value of the node feature should be permuted on the instance
        if(shuffle_attribute is not None and node.feature_index == shuffle_attribute):

            # list of probabilities to randomly assign the instance to the node branches
            probs = [] 
            # for each node that descend from the branches (except the last one)
            for j in range(len(node.branches)-1):
                # add to the list the probability that the instance would end up at the node
                # if it was randomly assigned to it - that is, the number of instances at the node
                # divided by the number of instances at its parent's node (round to 5 decimal digits).
                probs.append(round(sum(node.branches[j].distr.values())/ sum(node.distr.values()),5))

            # if there is a branch for missing values at the node 
            if(node.branch_nan is not None):
                # add to the list the probability that the instanece would end up at the node from the last branch
                probs.append(round(sum(node.branches[len(node.branches)-1].distr.values())/ sum(node.distr.values()),5))

            # if the sum of probabilities exceeded 1
            if(1-sum(probs) < 0):
                #change the last probability to be 1 - the sum of probabilities
                probs[-1] = round(1-sum(probs[:-1]),5)
                # the last branch (or the nan branch, if it exists) will be assigned with probability 0
                probs.append(0)
            # probability for the last branch (or the nan branch, if it exists) 
            else:
                probs.append(1-sum(probs))

            # randomly select the branch according the the probabilities calculated above
            i = np.random.choice(range(len(probs)),p=probs)

            # if the nan branch was selected, continue the prediction running the instance
            # through that branch
            if(i == len(node.branches) and node.branch_nan is not None):
                return self.predict_rec(X,node.branch_nan,shuffle_attribute)
            # continue the prediction running the instance through the randomly chosen branch
            else:
                return self.predict_rec(X,node.branches[i],shuffle_attribute)
              
             
        # if the value of instance X for the feature on the node is missing
        if(utils.isnan(X[node.feature_index])):
            # if there isn't a nan branch (C4.5 approach)
            if(node.branch_nan is None):
                # list of possible outcomes
                distr = []
                # list of relative distribution of possible outcomes
                prob_branch = []
                # add to the list of possible outcomes the prediction of the instance
                # through each one of the branches
                for n in node.branches:
                    distr.append(self.predict_rec(X,n,shuffle_attribute))
                    prob_branch.append(sum(n.distr.values()) / sum(node.distr.values()))

                d = {}
                # for each possible class at the node
                for k in node.distr.keys():
                    d[k] = 0
                    # for each possible outcome, add to the distribution the
                    # probability of that outcome
                    for i in range(len(distr)):
                        d[k] += prob_branch[i] * distr[i][k] 
                return d 
            # if there is a branch for the missing values
            else:
                # continue prediction through the nan branch
                y = self.predict_rec(X,node.branch_nan,shuffle_attribute)

        # if the value of instance X for the node feature is not missing and
        # it corresponds to a numeric feature (len(node.values) = 1)
        elif len(node.values) == 1:
            # if the value of instance X for the node feature is less than the 
            # value to compare, then continue the prediction through the left 
            # branch (node.branches[0]).
            if(X[node.feature_index] <= node.values[0]):
                y = self.predict_rec(X,node.branches[0],shuffle_attribute)
            # else continue through the right branch (node.branches[1])
            else:
                y = self.predict_rec(X,node.branches[1],shuffle_attribute)
        # if the node feature is categorical 
        else:
            # node.values.index(str(X[node.feature_index])) should return the 
            # index of the value of X for the node feature on the node.values list,
            # but if it can't find it, it means that this value hasn't been seen yet 
            # (none of the instances used to train the tree had that value). In that case,
            # it'll raise an ValueError.
            try:
                y = self.predict_rec(X,node.branches[node.values.index(str(X[node.feature_index]))],shuffle_attribute)
            except(ValueError):
                # if the value hasn't been seen at the training phase, then it'll be considered as a missing value.
                # if there is a nan branch, continue prediction through it
                if (node.branch_nan is not None):
                    y  = self.predict_rec(X,node.branch_nan,shuffle_attribute)
                # if there isn't a nan branch, then use C4.5 approach
                else:
                    distr = []
                    prob_branch = []
                    for n in node.branches:
                        distr.append(self.predict_rec(X,n,shuffle_attribute))
                        prob_branch.append(sum(n.distr.values()) / sum(node.distr.values()))
                    d = {}
                    for k in node.distr.keys():
                        d[k] = 0
                        for i in range(len(distr)):
                            d[k] += prob_branch[i] * distr[i][k]
                    return d

        return y



    # returns the score (accuracy rate) of the model when predicting the classes 
    # for data X and comparing it with y.  
    def score(self, X, y,shuffle_attribute=None):

        if(isinstance(y,pd.Series)):
            y = y.values
        y_predict = self.predict(X,prob=False,shuffle_attribute=shuffle_attribute)
        n_samples = len(y)


        correct = 0
        for i in range(n_samples):
            if y_predict[i] == y[i]:
                correct = correct + 1

        accuracy = correct/n_samples

        return accuracy

    # returns the area under the ROC curve of the model when predicting the classes
    # for data X and comparing it with y, considering control_class as the control class
    def auc(self,X,y,control_class,shuffle_attribute=None):
        if(control_class is None):
            pc = list(set(y))[0]
            # print('considering %r as control' % pc)
        else:
            pc = control_class
            
        y_predict = self.predict(X,prob=True,shuffle_attribute=shuffle_attribute)
        y_pred = []

        for instance_prediction in y_predict:
            if(pc in instance_prediction.keys()):
                y_pred.append(instance_prediction[pc])
            else:
                y_pred.append(0)

        fpr,tpr,thresholds = roc(y,y_pred,pos_label=pc)

        return auc(fpr,tpr)

    def to_pdf(self,attributes,out='out.pdf'):
        self.to_dot(attributes,out='tmp.dot')
        bash_command = 'dot -Tpdf tmp.dot -o' + out
        import subprocess
        subprocess.check_output(bash_command, shell=True)
        import os
        os.remove('tmp.dot')
    # creates a dot file for the tree model, with attributes defining the
    # feature names at each node of the tree.
    # the dot file name is defined by the out parameter.
    # after dot file is generated, use: $ dot -Tpdf out.dot -o out.pdf
    def to_dot(self,attributes,out='out.dot'):
        f = open(out,'w')
        f.write('digraph tree {\n')
        if(not self.root):
            print('No data found to be fit.')
            exit(-1)

        queue = deque([[0,self.root]])
        cont = 0
        while(len(queue) != 0):
            v,node = queue.popleft()            
            #not leaf
            #if(isinstance(node,Node)):
            if(node.is_class is False):
                f.write(str(v) + ' [label = "' + attributes[node.feature_index] + '\n' + str(node.sample_size) + '\n' + str(list(map(utils.prettyfloat,node.distr.values()))) +'"];\n')
                #numeric feature
                if(len(node.values) == 1):
                    #left child
                    cont+=1
                    f.write(str(v) + ' -> ' + str(cont))  
                    f.write(' [label = "<= ' + str(round(node.values[0],4)) + '"];\n')
                    queue.append([cont,node.branches[0]])

                    #right child
                    cont+=1
                    f.write(str(v) + ' -> ' + str(cont))  
                    f.write(' [label = "> ' + str(round(node.values[0],4)) + '"];\n')

                    queue.append([cont,node.branches[1]])
                #categoric feature
                else:
                    for i in range(len(node.values)):
                        cont+=1
                        f.write(str(v) + ' -> ' + str(cont))  
                        f.write(' [label = "= ' + str(node.values[i]) + '"];\n')

                        queue.append([cont,node.branches[i]])
                if(self.missing_branch and node.branch_nan):
                    cont+=1
                    f.write(str(v) + ' -> ' + str(cont))
                    f.write(' [label = " = NAN"];\n')
                    queue.append([cont,node.branch_nan]) 
            #leaf             
            else:
                f.write(str(v) + ' [label = "' + str(node.final_class) + '\n' + str(node.sample_size) + '\n' + str(list(map(utils.prettyfloat,node.distr.values()))) + '",shape=box];\n')
        f.write('}\n')
                 

    # recursively builds a decision tree with data Xc and its class yc.
    # feature_indices corresponds to the list of feature indexes that should be 
    # considered to build the tree.
    # weights corresponds to a dictionary with the instance indexes 
    # considered on the current node being built as the keys, and its
    # weights as the values. an instance will have weight 1 if it's "entirely" at
    # the node and (0 < w < 1) if it's partially there (C4.5 approach).
    # pweights are to the node's parent's class distribution
    def build_tree(self, Xc, yc, feature_indices, depth, weights,pdist=None):#,parent_fiv='root'):

        # only consider the instances that are at the node, partially or entirely

        rows_to_consider =  sorted(weights.keys())
        X = Xc[rows_to_consider,:]
        y = yc[rows_to_consider]  


        # calculate the class distribution at the node (absolute values)
        dist = {}
        for k in set(yc):
            dist[k] = 0 
        for k in weights.keys():
            dist[yc[k]] += weights[k]

        # if all "whole" instances at the node belong to the same class or if maximum tree depth was reached
        if (utils.entropy(y) == 0  or (len([k for k in dist.keys() if dist[k] < 1]) > 0) 
            or depth == self.max_depth):

            # in case of a tie of the class distributions, final class will be the most frequent
            # class at the parent node
            if(len(dist.keys()) > 1 and len(set(dist.values()))==1 and pdist is not None):
                #print('tie of class distributions. depth: %r distribution: %r' % (depth,dist))
                final_class = max(pdist.keys(),key=lambda k: pdist[k])

            # final class will be the most frequent class at the node
            else:
                final_class = max(dist.keys(),key=lambda k: dist[k])

            # return a decision node
            return Node(feature_index=None, values=None, branches=None, branch_nan=None,
                sample_size=sum([k for k in weights.values() if k == 1]), distr=dist, 
                is_class=True, final_class=final_class)#,config=parent_fiv+'->'+str(final_class))

        # get the feature and its split value(s) that maximize the information gain  
        if(self.random_subspace is False and self.mtry is not None):
            nfeature_indices = random.sample(list(feature_indices), int(self.mtry(len(feature_indices))))
        else:
            nfeature_indices = feature_indices
        feature_index, values = self.find_split(X,y, nfeature_indices, weights)

        #if the best split could not be found, returns a decision node 
        if(feature_index == -1):
            #print('best split could not be found.')
            if(len(dist.keys()) > 1 and len(set(dist.values()))==1 and pdist is not None):
                #print('tie of class distributions. depth: %r distribution: %r' % (depth,dist))
                final_class = max(pdist.keys(),key=lambda k: pdist[k])

            # final class will be the most frequent class at the node
            else:
                final_class = max(dist.keys(),key=lambda k: dist[k])

            return Node(feature_index=None, values=None, branches=None, branch_nan=None,
                sample_size=sum([k for k in weights.values() if k == 1]), distr=dist,
                is_class = True, final_class=final_class)

        # get rows where the values of X for the feature are not missing
        not_nan_rows = [a for a in range(X.shape[0]) if (not utils.isnan(X[:,feature_index][a]))]
        # get the rows where they are missing
        nan_rows = np.delete(list(range(X.shape[0])), not_nan_rows)
        
        Xnotnan = (X[not_nan_rows,:])
        ynotnan = y[not_nan_rows]
        ynan = y[nan_rows]    

        # get the sets (and its weights) that result when the not missing data are split
        # based on the feature and its value(s)    
        Xs,ys,dweights = utils.split(Xnotnan,ynotnan,feature_index,values)

        
        # if instances belong to only one subset, returns a decision node -- might be useless
        if(len(ys) < 2):
            #print('instances belong to only one subset.')

            if(len(dist.keys()) > 1 and len(set(dist.values()))==1 and pdist is not None):
                #print('tie of class distributions. depth: %r distribution: %r' % (depth,dist))
                final_class = max(pdist.keys(),key=lambda k: pdist[k])

            # final class will be the most frequent class at the node
            else:
                final_class = max(dist.keys(),key=lambda k: dist[k])

            # if(self.print):
            return Node(feature_index=None, values=None, branches=None, branch_nan=None,
                sample_size=sum([k for k in weights.values() if k == 1]), distr=dist,
                is_class=True, final_class=final_class)


        branch_nan = None    
        branches = []

        # translate the dweights indexes to the weights indexes
        for i in range(len(dweights)):
            dweights[i] = dict((rows_to_consider[not_nan_rows[j]],dweights[i][j]) for j in dweights[i])
            for j in dweights[i].keys():
                if j in weights.keys():
                    dweights[i][j] = weights[j]

        # sum of the weights of the instances in the node with known values 
        s = (sum([sum(x.values()) for  x in dweights]))

        # for each split set 
        for i in range(len(ys)):
            # if it's not empty       
            if len(ys[i]) != 0:
                # C.45 approach 
                if(self.missing_branch is False):
                    # calculate probability of outcome values[i], estimated as the sum of the weights
                    # of instances in the node known to have outcome values[i] divided by the sum of the
                    # weights of the cases in the node with known outcomes
                    prob_values_i = round(float(sum(dweights[i].values()) / s),5)
                    # for each instance with missing value, update its weight for the child node 
                    for j in nan_rows:
                        (dweights[i])[rows_to_consider[j]] = weights[rows_to_consider[j]] * prob_values_i


                branches.append(self.build_tree(Xc,yc,feature_indices,depth+1,dweights[i],dist))#,parent_fiv=str(feature_index)+'->'+v))

        # nan branch approach
        if(self.missing_branch):
            # if there are samples with known values
            if(ynan.shape[0] != 0):
                # continue building tree from the nan branch

                branch_nan = self.build_tree(Xc,yc,feature_indices,depth+1,
                    dict([[a,1] for a in np.array(rows_to_consider)[nan_rows]]),dist)#,str(feature_index)+'->NAN')
            # if there aren't, then assign to the nan branch a decision node with no instances 
            # (for future classification purposes).
            else:   
                if(len(dist.keys()) > 1 and len(set(dist.values()))==1 and pdist is not None):
                    #print('tie of class distributions. depth: %r distribution: %r' % (depth,dist))
                    final_class = max(pdist.keys(),key=lambda k: pdist[k])
                # final class will be the most frequent class at the node
                else:
                    final_class = max(dist.keys(),key=lambda k: dist[k])

                # assign to the nan branch a decision node
                branch_nan =  Node(feature_index=None, values=None, branches=None, branch_nan=None,
                    sample_size=0, distr={k:0 for k in set(y)}, is_class=True, final_class=final_class)#,config=parent_fiv+'->'+str(final_class))

        same_class = False
        fclass = branches[0].final_class
        import pdb
        if(fclass is not None):
            for child in range(1,len(branches)):
                if(branches[child].final_class != fclass):
                    same_class=False
                    break
                if(child == len(branches)-1):
                    if(branch_nan):
                        if(branch_nan.final_class != fclass):
                            same_class=False
                        else:
                            same_class=True
                    else:
                        same_class=True
        if(same_class is True):
            #print('class node - all children nodes belong to the same class')
            if(len(dist.keys()) > 1 and len(set(dist.values()))==1 and pdist is not None):
                #print('tie of class distributions. depth: %r distribution: %r' % (depth,dist))
                final_class = max(pdist.keys(),key=lambda k: pdist[k])

            else:
                final_class = max(dist.keys(),key=lambda k: dist[k])

            return Node(feature_index=None, values=None, branches=None, branch_nan=None,
                sample_size=sum([k for k in weights.values() if k == 1]), distr=dist,
                is_class=True, final_class=final_class)#,config=parent_fiv+'->'+str(final_class))


        # # returns a test node with its feature index and values and its branches.
        return Node(feature_index=feature_index, values=values, branches=branches, branch_nan=branch_nan,
            sample_size=sum([k for k in weights.values() if k == 1]), distr=dist)


    # returns the feature index from the feature_indices list that 
    # maximizes the information gain
    def find_split(self,X, y, feature_indices, weights):

        best_gain = -float('inf')
        best_feature_index = -1
        best_value = [0]

        # for each feature to be considered 
        for feature_index in sorted(feature_indices):
            # get rows of instances with known values for the feature  
            not_nan_rows = [a for a in range(X.shape[0]) if not utils.isnan(X[:,feature_index][a])]

            Xnotnan = (X[not_nan_rows,:])
            ynotnan = y[not_nan_rows]

            #if there aren't any instances with known values for the feature, go to the next one
            if(Xnotnan.shape[0] == 0):
                continue

            # get all possible values for the feature index
            values = sorted(set(Xnotnan[:, feature_index]))
            
            # if the values are numeric
            if(utils.isnum(Xnotnan[0,feature_index])):
                
                # split the data using each value
                for j in range(len(values) - 1):

                    #value = (float(values[j]) + float(values[j+1]))/2 -- original
                    value = values[j]
                    # split data using the feature and the value 
                    Xs,ys,d = utils.split_num(Xnotnan, ynotnan, feature_index, value)
                    # calculate gain considering the rate of missing values. 
                    # the bigger the rate, the smaller the gain
                    gain = (len(ynotnan)/len(y)) * utils.information_gain(ynotnan, ys)

                    if gain >= best_gain:
                        # if there's a tie on info gain, decide using gain ratio
                        # if(gain == best_gain and best_feature_index != -1):
                        #     print('tie of gain')
                        #     gr = utils.gain_ratio(ynotnan,ys,y)
                        #     not_nan_rows = [a for a in range(X.shape[0]) if not utils.isnan(X[:,best_feature_index][a])]
                        #     Xss,yss, ds = utils.split(X[not_nan_rows,:],y[not_nan_rows],best_feature_index,best_value)
                        #     # calculate gain ratio of previous best feature to compare 
                        #     gr_p = utils.gain_ratio(ynotnan,yss,y)
                        #     # if the current feature's gain ratio is not better than the previous one, then
                        #     # go to the next feature
                        #     if(gr < gr_p):
                        #         continue

                        best_gain = gain
                        best_feature_index = feature_index
                        best_value = [values[j]] #c4.5 choses the largest value in the trainig set that 
                                                 #does not exceed the midpoint (value). This ensures that all
                                                 #threshold values appearing in trees actually occur in the data
            # if the values are categorical
            else:
                # split the data using the values
                Xs,ys,d = utils.split_categ(Xnotnan,ynotnan,feature_index,values)

                
                gain = ((len(ynotnan)/len(y)) *utils.information_gain(ynotnan,ys))#utils.gain_ratio(ynotnan,ys,y)) 

                if gain >= best_gain:
                    # if(gain == best_gain and best_feature_index != -1):
                    #     print('tie of gain')
                    #     gr = utils.gain_ratio(ynotnan,ys,y)
                    #     not_nan_rows = [a for a in range(X.shape[0]) if not utils.isnan(X[:,best_feature_index][a])]
                    #     Xss,yss, ds = utils.split(X[not_nan_rows,:],y[not_nan_rows],best_feature_index,best_value)
                    #     gr_p = utils.gain_ratio(ynotnan,yss,y)
                    #     if(gr < gr_p):
                    #         continue

                    best_gain = gain
                    best_feature_index = feature_index
                    best_value = values

        return best_feature_index, best_value


