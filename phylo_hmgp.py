
"""
Devoloped using hmmlearn
"""

import numpy as np
from scipy.misc import logsumexp
from sklearn import cluster
from sklearn import mixture
from sklearn.mixture import (
    sample_gaussian,
    log_multivariate_normal_density,
    distribute_covar_matrix_to_match_covariance_type, _validate_covars)
from sklearn.utils import check_random_state

from base1 import _BaseHMM
from utils import iter_from_X_lengths, normalize

import pandas as pd
import numpy as np
import os
import sys
import math
import random
import scipy
import scipy.io
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode

import lasagne
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan

from lasagne.layers.base import MergeLayer, Layer
from lasagne.layers.input import InputLayer
from lasagne.layers.dense import DenseLayer
from lasagne.layers import helper

from lasagne.layers.recurrent import Gate
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.random import get_rng

from numpy.linalg import inv, det, norm

import hmmlearn
from hmmlearn import hmm
from scipy.optimize import minimize

import pickle

from optparse import OptionParser

import os.path

import warnings


__all__ = ["PhyloHMM", "GMMHMM", "GaussianHMM", "MultinomialHMM"]

COVARIANCE_TYPES = frozenset(("linear","spherical", "diag", "full", "tied"))

def weight_init(shape):
  init_variable = tf.truncated_normal(shape, mean = 1.0, stddev=0.5, dtype=tf.float32)
  return tf.Variable(init_variable)

# likelihood of O-U process
def ou_lik(params, cv, state_id, stats, T, n_samples):
        
        alpha, sigma, theta0, theta1 = params[0], params[1], params[2], params[3:]
        # T = self.leaf_time
        a1 = 2.0*alpha

        V = sigma**2/a1*np.exp(-a1*(T-cv))*(1-np.exp(-a1*cv))
        s1 = np.exp(-alpha*T)
        # print theta0, theta1, theta
        theta = theta0*s1+theta1*(1-s1)
        c = state_id
        obsmean = np.outer(stats['obs'][c], theta)

        Sn_w = (stats['obs*obs.T'][c]
                - obsmean - obsmean.T
                + np.outer(theta, theta)*stats['post'][c])

        # weights_sum = stats['post'][c]
        lik = stats['post'][c]*np.log(det(V))/n_samples+np.sum(inv(V)*Sn_w)/n_samples

        return lik

def ou_lik1(params, cv, state_id, stats, T, n_samples):
        
        alpha, sigma, theta0, theta1, lambda1 = params[0], params[1], params[2], params[3:-1], params[-1]

        # T = self.leaf_time
        a1 = 2.0*alpha
        
        # sigma1 = sigma**2
        V1 = 1.0/a1*np.exp(-a1*(T-cv))*(1-np.exp(-a1*cv))

        s1 = np.exp(-alpha*T)
        # print theta0, theta1, theta
        theta = theta0*s1+theta1*(1-s1)
        print theta

        c = state_id
        obsmean = np.outer(stats['obs'][c], theta)

        Sn_w = (stats['obs*obs.T'][c]
               - obsmean - obsmean.T
               + np.outer(theta, theta)*stats['post'][c])

        # denom = stats['post'][c]
        denom = stats['post'][:, np.newaxis]

        mean_value = (stats['obs']/denom)
        m1 = mean_value[c]
        print "m1", m1

        obsmean1 = np.outer(stats['obs'][c], m1)

        Sn_w1 = (stats['obs*obs.T'][c]
                - obsmean1 - obsmean1.T
                + np.outer(m1, m1)*stats['post'][c])

        d = m1.shape[0]
        q1 = np.sum(inv(V1)*Sn_w1)/n_samples
        w1 = stats['post'][c]/n_samples
        sigma1 = q1/(w1*d)
        sigma = np.sqrt(sigma1)
        V = sigma**2*V1
        lik = w1*d*np.log(sigma1)+w1*np.log(det(V1))+np.sum(inv(V)*Sn_w)/n_samples

        return lik

def brownian_lik(params, Sn_w, weighted_sum, base_vec, n_samples, n_features):
        
        cv = np.zeros((n_features,n_features))
        i = 0
        for branch_param in params:
            cv += branch_param*base_vec[i]
            i += 1
        
        lik = weighted_sum*np.log(det(cv))/n_samples+np.sum(inv(cv)*Sn_w)/n_samples

        return lik

def brownian_lik1(params, state_id, stats, mean_value, base_vec, n_samples, n_features):
        
        c = state_id
        obsmean = np.outer(stats['obs'][c], mean_value)

        Sn_w = (stats['obs*obs.T'][c]
                - obsmean - obsmean.T
                + np.outer(mean_value, mean_value)*stats['post'][c])
        
        cv = np.zeros((n_features,n_features))
        i = 0
        for branch_param in params:
            cv += branch_param*base_vec[i]
            i += 1
        
        # weights_sum = stats['post'][c]
        lik = stats['post'][c]*np.log(det(cv))/n_samples+np.sum(inv(cv)*Sn_w)/n_samples

        return lik

# equality constraints
def constraint1(params):
    return params

class phyloHMM1(_BaseHMM):
    
    def __init__(self, n_samples, n_features, edge_list, branch_list, cons_param, 
                 initial_mode, initial_weight, initial_weight1, initial_magnitude, observation,
                 n_components=1, run_id=0, covariance_type='full',
                 min_covar=1e-3,
                 startprob_prior=1.0, transmat_prior=1.0, 
                 means_prior=0, means_weight=0,
                 covars_prior=1e-2, covars_weight=1,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stmc", init_params="stmc", 
                 learning_rate = 0.001):
        _BaseHMM.__init__(self, n_components=n_components, run_id=run_id,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          tol=tol, params=params, verbose=verbose,
                          init_params=init_params)

        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.random_state = random_state
        self.lik = 0

        #self.state_id = 0
        self.observation = observation
        print "data loaded", self.observation.shape
        
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_components = n_components
        self.learning_rate = learning_rate

        self.tree_mtx, self.node_num = self._initilize_tree_mtx(edge_list)

        self.branch_params = branch_list
        self.branch_dim = self.node_num - 1   # number of branches
        
        self.n_params = self.node_num + self.branch_dim*2 + 1  # optimal values (n1), selection strength and variance (n2*2), variance of root node
        
        self.params_vec1 = np.random.rand(n_components, self.n_params)

        self.init_ou_params = self.params_vec1.copy()

        print "branch dim", self.branch_dim
        print "number of parameters", self.n_params

        self.branch_vec = [None]*self.node_num  # all the leaf nodes that can be reached from a node

        self.base_struct = [None]*self.node_num
        print "compute base struct"
        self.leaf_list = self._compute_base_struct()
        print self.leaf_list
        self.index_list = self._compute_covariance_index()
        print self.index_list
        self.base_vec = self._compute_base_mtx()
        self.leaf_time = branch_list[0]+branch_list[1]  # this needs to be updated
        self.leaf_vec = self._search_leaf()  # search for the leaves of the tree
        self.path_vec = self._search_ancestor()

        mtx = np.zeros((n_features,n_features))
        print "initilization, branch parameters", self.branch_params
        print "initilization, branch dim:", self.branch_dim
        for i in range(0,self.branch_dim):
            mtx += self.branch_params[i]*self.base_vec[i+1]

        print mtx
        self.cv_mtx = mtx
        print self.leaf_time
        print self.params_vec1

        #posteriors = np.random.rand(self.n_samples,n_components)
        posteriors = np.ones((self.n_samples,n_components))
        den1 = np.sum(posteriors,axis=1)
        
        # self.posteriors = posteriors/(np.reshape(den1,(self.n_samples,1))*np.ones((1,n_components)))
        self.posteriors = np.ones((self.n_samples,n_components))    # for testing
        self.mean = np.random.rand(n_components, self.n_features)   # for testing
        
        self.stats = dict()
        self.counter = 0

        self.A1, self.A2, self.pair_list, self.parent_list = self._matrix1()

        self.lambda_0 = cons_param  # ridge regression coefficient
        self.initial_mode, self.initial_w1, self.initial_w1a, self.initial_w2 = initial_mode, initial_weight, initial_weight1, initial_magnitude

        print "initial weights", self.initial_w1, self.initial_w1a, self.initial_w2

        print "lambda_0", cons_param, n_samples, self.lambda_0*1.0/np.sqrt(n_samples)

    def _get_covars(self):
        """Return covars as a full matrix."""
        if self.covariance_type == 'full':
            return self._covars_
        elif self.covariance_type == 'diag':
            return np.array([np.diag(cov) for cov in self._covars_])
        elif self.covariance_type == 'tied':
            return np.array([self._covars_] * self.n_components)
        elif self.covariance_type == 'spherical':
            return np.array(
                [np.eye(self.n_features) * cov for cov in self._covars_])
        elif self.covariance_type == 'linear':
            return self._covars_

    def _set_covars(self, covars):
        self._covars_ = np.asarray(covars).copy()

    covars_ = property(_get_covars, _set_covars)

    def _check(self):
        super(phyloHMM1, self)._check()

        self.means_ = np.asarray(self.means_)
        self.n_features = self.means_.shape[1]

        if self.covariance_type not in COVARIANCE_TYPES:
            raise ValueError('covariance_type must be one of {0}'
                             .format(COVARIANCE_TYPES))

        _validate_covars(self._covars_, self.covariance_type,
                         self.n_components)

    # initialize the parameters of the multiple OU models
    def _init_ou_param(self, X, init_label, mean_values):
        n_components = self.n_components 
        # init_label = self.init_label_.copy()
        init_ou_params = self.params_vec1.copy()

        for i in range(0,n_components):
            b = np.where(init_label==i)[0]
            num1 = b.shape[0]  # number of samples in the initialized cluster
            if num1==0:
                print "empty cluster!"
            else:
                x1 = X[b]
                print "number in the cluster", x1.shape[0]
                cur_param, lik = self._ou_optimize_init(x1, mean_values[i])
                init_ou_params[i,:] = cur_param.copy()

        print "initial ou paramters"
        print init_ou_params
        
        return init_ou_params

    def _init(self, X, lengths=None):
        super(phyloHMM1, self)._init(X, lengths=lengths)
        
        _, n_features = X.shape
        if hasattr(self, 'n_features') and self.n_features != n_features:
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (n_features, self.n_features))

        self.n_features = n_features
        if 'm' in self.init_params or not hasattr(self, "means_"):
            kmeans = cluster.KMeans(n_clusters=self.n_components,
                                    random_state=self.random_state,
                                    max_iter=300, n_jobs=-5, n_init=10)
            kmeans.fit(X)
            self.means_ = kmeans.cluster_centers_
            self.init_label = kmeans.labels_
            print "initialize parameters..."
            self.init_ou_params = self._init_ou_param(X, self.init_label.copy(), self.means_.copy())
            self.params_vec1 = self.init_ou_params.copy()

        if 'c' in self.init_params or not hasattr(self, "covars_"):
            cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
            if not cv.shape:
                cv.shape = (1, 1)
            self._covars_ = distribute_covar_matrix_to_match_covariance_type(
                cv, self.covariance_type, self.n_components).copy()

    def _compute_log_likelihood(self, X):
        return log_multivariate_normal_density(
            X, self.means_, self._covars_, self.covariance_type)

    def _generate_sample_from_state(self, state, random_state=None):
        if self.covariance_type == 'tied':
            cv = self._covars_
        else:
            cv = self._covars_[state]
        return sample_gaussian(self.means_[state], cv, self.covariance_type,
                               random_state=random_state)

    def _initialize_sufficient_statistics(self):
        stats = super(phyloHMM1, self)._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        stats['obs**2'] = np.zeros((self.n_components, self.n_features))
        stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features,
                                           self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super(phyloHMM1, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice)

        if 'm' in self.params or 'c' in self.params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs)

        if 'c' in self.params:
            stats['obs**2'] += np.dot(posteriors.T, obs ** 2)
            stats['obs*obs.T'] += np.einsum(
                    'ij,ik,il->jkl', posteriors, obs, obs)

    def params_Initialization(self, state_num, branch_dim):
        params = {
            'v_branch': weight_init([state_num, branch_dim])
        }

        print "parameters intialized"
        return params

    # initilize the connected graph of the tree given the edges
    def _initilize_tree_mtx(self, edge_list):
        node_num = np.max(np.max(edge_list))+1  # number of nodes; index starting from 0
        tree_mtx = np.zeros((node_num,node_num))
        for edge in edge_list:
            p1, p2 = np.min(edge), np.max(edge)
            tree_mtx[p1,p2] = 1

        print "tree matrix built"
        print tree_mtx

        return tree_mtx, node_num

    # find all the leaf nodes which can be reached from a given node
    def _sub_tree_leaf(self, index):
        tmp = self.tree_mtx[index] # find the neighbors
        idx = np.where(tmp==1)[0]
        print idx
        print "size of branch vec", len(self.branch_vec)

        node_vec = []
        if idx.shape[0]==0:
            node_vec = [index]  # the leaf node 
            print "leaf", node_vec
        else:
            for j in idx:
                node_vec1 = self._sub_tree_leaf(j)
                node_vec = node_vec + node_vec1
                print "interior", node_vec

        self.branch_vec[index] = node_vec  # all the leaf nodes that can be reached from this node
        
        print "branch_dim", index, node_vec

        return node_vec

    # find all the pairs of leaf nodes which has a given node as the nearest common ancestor
    def _compute_base_struct(self):  
        node_num = self.node_num
        node_vec = self._sub_tree_leaf(0)  # start from the root node
        cnt = 0
        leaf_list = dict()

        for i in range(0,node_num):
            list1 = self.branch_vec[i]  # all the leaf nodes that can be reached from this node
            num1 = len(list1)
            if num1 == 1:
                leaf_list[i] = cnt
                cnt +=1
            self.base_struct[i] = []
            for j in range(0,num1):
                for k in range(j,num1):
                    self.base_struct[i].append(np.array((list1[j],list1[k])))

        print "index built"
        if node_num>2:
            print self.branch_vec[1], self.branch_vec[2]
        return leaf_list

    # find the pair of nodes that share a node as common ancestor
    def _compute_covariance_index(self):
        index = []
        num1 = self.node_num    # the number of nodes
        for k in range(0,num1): # starting from index 1
            t_index = []
            leaf_vec = self.base_struct[k]   # the leaf nodes that share this ancestor
            num2 = len(leaf_vec)
            for i in range(0,num2):
                id1, id2 = self.leaf_list[leaf_vec[i][0]], self.leaf_list[leaf_vec[i][1]]
                t_index.append([id1,id2])
                if id1!=id2:
                    t_index.append([id2,id1])
            index.append(t_index)

        return index

    # compute base matrix
    def _compute_base_mtx(self):
        base_vec = dict()
        index, n_features = self.index_list, self.n_features
        num1 = len(index)
        print "index size", index
        base_vec[0] = np.ones((n_features,n_features)) # base matrix for the root node
        for i in range(1,num1):
            indices = index[i]
            cv = np.zeros((n_features,n_features))
            num2 = len(indices)
            for j in range(0,num2):
                id1 = indices[j]
                cv[id1[0],id1[1]] = 1
                cv[id1[1],id1[0]] = 1  # symmetric matrix
            base_vec[i] = cv

        for i in range(0,num1):
            filename = "base_mtx_%d"%(i)
            np.savetxt(filename, base_vec[i], fmt='%d', delimiter='\t')

        return base_vec

    # compute covariance matrix for a state
    def _compute_covariance_mtx_2(self, params):
        mtx = self.cv_mtx
        T = self.leaf_time
        alpha, sigma = params[0], params[1]
        a1 = 2.0*alpha
        sigma1 = sigma**2
        cv = sigma**2/a1*np.exp(-a1*(T-mtx))*(1-np.exp(-a1*mtx))

        return cv

    def _compute_log_likelihood_2(self, params, state_id):
        cv, n_samples = self._compute_covariance_mtx_2(params), self.n_samples
        inv_cv = inv(cv)
        weights_sum = self.stats['post'][state_id]

        T = cv[0,0]
        a1 = 2.0*alpha
        sigma1 = sigma**2
        V = sigma**2/a1*np.exp(-a1*(T-cv))*(1-np.exp(-a1*cv))
        s1 = np.exp(-alpha*T)
        print theta0
        print theta1
        theta = theta0*s1+theta1*(1-s1)
        print theta
        n,d = data1.shape[0], data1.shape[1]

        x1 = data1-theta
        lik = weights_sum*np.log(det(V))/n_samples+np.sum(inv(V)*np.matmul(x1.T,x1))/n
        
        return likelihood

    def _output_stats(self, number):
        filename = "log1/stats_iter_%d"%(number)
        np.savetxt(filename, self.stats['post'], fmt='%.4f', delimiter='\t')

    def _ou_lik(self, params, cv, state_id):
        
        alpha, sigma, theta0, theta1 = params[0], params[1], params[2], params[3:]
        T = self.leaf_time
        a1 = 2.0*alpha
        
        # sigma1 = sigma**2
        # V1 = 1.0/a1*np.exp(-a1*(T-cv))*(1-np.exp(-a1*cv))
        V = sigma**2/a1*np.exp(-a1*(T-cv))*(1-np.exp(-a1*cv))
        s1 = np.exp(-alpha*T)
        # print theta0, theta1, theta
        theta = theta0*s1+theta1*(1-s1)
        c = state_id
        obsmean = np.outer(self.stats['obs'][c], theta)

        Sn_w = (self.stats['obs*obs.T'][c]
                - obsmean - obsmean.T
                + np.outer(theta, theta)*self.stats['post'][c])

        n_samples = self.n_samples
        lik = self.stats['post'][c]*np.log(det(V))/n_samples+np.sum(inv(V)*Sn_w)/n_samples

        return lik

    # search all the ancestors of a leaf node   
    def _search_ancestor(self):
        path_vec = []        
        tree_mtx = self.tree_mtx
        # n = tree_mtx.shape[0]
        n = self.leaf_vec.shape[0]
        for i in range(0,n):
            leaf_idx = self.leaf_vec[i]
            b = np.where(tree_mtx[:,leaf_idx]>0)[0] # ancestor of the leaf node
            b1 = []
            while b.shape[0]>0:
                idx = b[0]  # parent index
                b1.insert(0,idx)
                b = np.where(tree_mtx[:,idx]>0)[0] # ancestor of the leaf node
            
            b1 = np.append(b1,leaf_idx)  # with the node as the end
            path_vec.append(b1)

        return path_vec

    def _search_leaf(self):
        tree_mtx = self.tree_mtx
        n1 = tree_mtx.shape[0] # number of nodes
        
        leaf_vec = []
        for i in range(0,n1):
            idx = np.where(tree_mtx[i,:]>0)[0]
            if idx.shape[0]==0:
                leaf_vec.append(i)

        return np.array(leaf_vec)

    def _matrix1(self): 
        
        tree_mtx = self.tree_mtx
        leaf_vec = self.leaf_vec

        print self.branch_dim
        n2, N2 = self.node_num, self.node_num   # assign a branch to the first node
        n1 = np.array(leaf_vec).shape[0]        # the number of leaf nodes 
        N1 = int(n1*(n1-1)/2)

        print N1, N2
        # common_ans = np.zeros((N1,1))
        pair_list, parent_list = [], [None]*n2

        A1 = np.zeros((n1,n2))  # leaf node number by branch dim
        print "path_vec", self.path_vec
        for i in range(0,n1):
            print leaf_vec[i], self.path_vec[i]

        for i in range(0,n2):
            b = np.where(tree_mtx[:,i]>0)[0]
            if b.shape[0]>0:
                parent_list[i] = b[0]
            else:
                parent_list[i] = []

        for i in range(0,n1):
            leaf_idx = leaf_vec[i]
            A1[i,parent_list[leaf_idx]] = 1

        A2 = np.zeros((N1,N2))
        cnt = 0
        for i in range(0,n1):
            # leaf_idx1 = leaf_vec[i]
            vec1 = self.path_vec[i]
            for j in range(i+1,n1):
                # leaf_idx2 = leaf_vec[j]
                vec2 = self.path_vec[j]
                t1 = np.intersect1d(vec1,vec2)  # common ancestors
                id1 = np.max(t1)  # the nearest common ancestor

                c1 = np.setdiff1d(vec1, t1)
                c2 = np.setdiff1d(vec2, t1)

                A2[cnt,c1], A2[cnt,c2] = 1, 1
                # common_ans[cnt] = id1
                pair_list.append([leaf_vec[i],leaf_vec[j],id1])

                cnt += 1

        print pair_list
        filename = "ou_A1.txt"
        np.savetxt(filename, A1, fmt='%d', delimiter='\t')
        filename = "ou_A2.txt"
        np.savetxt(filename, A2, fmt='%d', delimiter='\t')

        return A1, A2, pair_list, parent_list
    
    def _ou_lik_varied(self, params, state_id):

        n1, n2 = self.leaf_vec.shape[0], self.node_num  # number of leaf nodes, number of nodes
        
        c = state_id
        values = np.zeros((n2,2))  # expectation and variance
        covar_mtx = np.zeros((n1,n1))

        num1 = self.branch_dim  # number of branches; assign parameters to each of the branches
        params1 = params[1:]
        beta1, lambda1, theta1 = params1[0:num1], params1[num1:2*num1], params1[2*num1:3*num1+1]

        ratio1 = lambda1/(2*beta1)
        values[0,0] = theta1[0]  # mean value of the root node
        values[0,1] = params[0]
        beta1_exp = np.exp(-beta1)
        beta1_exp = np.insert(beta1_exp,0,0)

        # compute the transformation matrix
        A1, A2, pair_list, p_idx = self.A1, self.A2, np.array(self.pair_list), self.parent_list    

        # add a branch to the first node
        beta1, lambda1, ratio1 = np.insert(beta1,0,0), np.insert(lambda1,0,0), np.insert(ratio1,0,0)
        
        # print p_idx
        print beta1

        for i in range(1,n2):
            values[i,0] = values[p_idx[i],0]*beta1_exp[i] + theta1[i]*(1-beta1_exp[i])
            values[i,1] = ratio1[i]*(1-beta1_exp[i]**2) + values[p_idx[i],1]*(beta1_exp[i]**2)

        # print values
        s1 = np.matmul(A2, beta1)
        idx = pair_list[:,-1]   # index of common ancestor
        s2 = values[idx,1]*np.exp(-s1)
        
        num = pair_list.shape[0]
        leaf_list = self.leaf_list
        for k in range(0,num):
            id1,id2 = pair_list[k,0], pair_list[k,1]
            i,j = leaf_list[id1], leaf_list[id2]
            covar_mtx[i,j] = s2[k]
            covar_mtx[j,i] = covar_mtx[i,j]

        for i in range(0,n1):
            covar_mtx[i,i] = values[self.leaf_vec[i],1] # variance
        
        # sigma1 = sigma**2
        # V1 = 1.0/a1*np.exp(-a1*(T-cv))*(1-np.exp(-a1*cv))
        V = covar_mtx.copy()
        theta = theta1[self.leaf_vec]
        mean_values1 = values[self.leaf_vec,0]
        # obsmean = np.outer(self.stats['obs'][c], theta)
        obsmean = np.outer(self.stats['obs'][c], mean_values1)
        # print covar_mtx, theta

        Sn_w = (self.stats['obs*obs.T'][c]
                - obsmean - obsmean.T
                + np.outer(mean_values1, mean_values1)*self.stats['post'][c])

        n_samples = self.n_samples
        # weights_sum = stats['post'][c]
        lik = self.stats['post'][c]*np.log(det(V))/n_samples+np.sum(inv(V)*Sn_w)/n_samples

        self.values = values.copy()
        self.cv_mtx = covar_mtx.copy()

        print "likelihood", state_id, lik

        return lik

    def _ou_lik_varied_constraint(self, params, state_id):

        n1, n2 = self.leaf_vec.shape[0], self.node_num  # number of leaf nodes, number of nodes
        
        c = state_id
        values = np.zeros((n2,2))   # expectation and variance
        covar_mtx = np.zeros((n1,n1))

        num1 = self.branch_dim  # number of branches; assign parameters to each of the branches
        params1 = params[1:]
        beta1, lambda1, theta1 = params1[0:num1], params1[num1:2*num1], params1[2*num1:3*num1+1]

        ratio1 = lambda1/(2*beta1)
        values[0,0] = theta1[0]  # mean value of the root node
        values[0,1] = params[0]
        beta1_exp = np.exp(-beta1)
        beta1_exp = np.insert(beta1_exp,0,0)

        # compute the transformation matrix
        A1, A2, pair_list, p_idx = self.A1, self.A2, np.array(self.pair_list), self.parent_list    

        # add a branch to the first node
        beta1, lambda1, ratio1 = np.insert(beta1,0,0), np.insert(lambda1,0,0), np.insert(ratio1,0,0)
                
        for i in range(1,n2):
            values[i,0] = values[p_idx[i],0]*beta1_exp[i] + theta1[i]*(1-beta1_exp[i])
            values[i,1] = ratio1[i]*(1-beta1_exp[i]**2) + values[p_idx[i],1]*(beta1_exp[i]**2)

        # print values
        s1 = np.matmul(A2, beta1)
        idx = pair_list[:,-1]   # index of common ancestor
        s2 = values[idx,1]*np.exp(-s1)
        
        num = pair_list.shape[0]
        leaf_list = self.leaf_list
        for k in range(0,num):
            id1,id2 = pair_list[k,0], pair_list[k,1]
            i,j = leaf_list[id1], leaf_list[id2]
            covar_mtx[i,j] = s2[k]
            covar_mtx[j,i] = covar_mtx[i,j]

        for i in range(0,n1):
            covar_mtx[i,i] = values[self.leaf_vec[i],1] # variance
        
        V = covar_mtx.copy()
        theta = theta1[self.leaf_vec]
        mean_values1 = values[self.leaf_vec,0]
        obsmean = np.outer(self.stats['obs'][c], mean_values1)
        # print covar_mtx, theta

        Sn_w = (self.stats['obs*obs.T'][c]
                - obsmean - obsmean.T
                + np.outer(mean_values1, mean_values1)*self.stats['post'][c])

        n_samples = self.n_samples
        lambda_0 = self.lambda_0

        lambda_1 = 1.0/np.sqrt(n_samples)
        lik = (self.stats['post'][c]*np.log(det(V))/n_samples
                +np.sum(inv(V)*Sn_w)/n_samples
                +lambda_0*lambda_1*np.dot(params.T,params))

        self.values = values.copy()
        self.cv_mtx = covar_mtx.copy()

        print "likelihood", state_id, lik

        return lik

    # compute log likelihood for a single state
    def _ou_lik_varied_single(self, params, obs):

        n1, n2 = self.leaf_vec.shape[0], self.node_num  # number of leaf nodes, number of nodes
        
        values = np.zeros((n2,2))  # expectation and variance
        covar_mtx = np.zeros((n1,n1))

        num1 = self.branch_dim  # number of branches; assign parameters to each of the branches
        params1 = params[1:]
        beta1, lambda1, theta1 = params1[0:num1], params1[num1:2*num1], params1[2*num1:3*num1+1]

        ratio1 = lambda1/(2*beta1)
        values[0,0] = theta1[0]  # mean value of the root node
        values[0,1] = params[0]
        beta1_exp = np.exp(-beta1)
        beta1_exp = np.insert(beta1_exp,0,0)

        # compute the transformation matrix
        A1, A2, pair_list, p_idx = self.A1, self.A2, np.array(self.pair_list), self.parent_list    

        # add a branch to the first node
        beta1, lambda1, ratio1 = np.insert(beta1,0,0), np.insert(lambda1,0,0), np.insert(ratio1,0,0)

        for i in range(1,n2):
            values[i,0] = values[p_idx[i],0]*beta1_exp[i] + theta1[i]*(1-beta1_exp[i])
            values[i,1] = ratio1[i]*(1-beta1_exp[i]**2) + values[p_idx[i],1]*(beta1_exp[i]**2)

        # print values
        s1 = np.matmul(A2, beta1)
        idx = pair_list[:,-1]   # index of common ancestor
        s2 = values[idx,1]*np.exp(-s1)
        
        num = pair_list.shape[0]
        leaf_list = self.leaf_list
        for k in range(0,num):
            id1,id2 = pair_list[k,0], pair_list[k,1]
            i,j = leaf_list[id1], leaf_list[id2]
            covar_mtx[i,j] = s2[k]
            covar_mtx[j,i] = covar_mtx[i,j]

        for i in range(0,n1):
            covar_mtx[i,i] = values[self.leaf_vec[i],1] # variance
        
        # sigma1 = sigma**2
        # V1 = 1.0/a1*np.exp(-a1*(T-cv))*(1-np.exp(-a1*cv))
        V = covar_mtx.copy()
        mean_values1 = values[self.leaf_vec,0]
        n = obs.shape[0]

        obsmean = np.outer(np.mean(obs,axis=0),mean_values1)

        Sn_w = np.dot(obs.T,obs)/n - obsmean - obsmean.T + np.outer(mean_values1, mean_values1)

        lik = np.log(det(V))+np.sum(inv(V)*Sn_w)

        self.values = values.copy()
        self.cv_mtx = covar_mtx.copy()

        print "likelihood", lik

        return lik

    def _ou_lik1(self, params, cv, state_id):
        
        alpha, sigma, theta0, theta1 = params[0], params[1], params[2], params[3:]
        T = self.leaf_time
        a1 = 2.0*alpha
        
        V = sigma**2/a1*np.exp(-a1*(T-cv))*(1-np.exp(-a1*cv))
        s1 = np.exp(-alpha*T)
        # print theta0, theta1, theta
        theta = theta0*s1+theta1*(1-s1)
        c = state_id
        obsmean = np.outer(self.stats['obs'][c], theta)

        Sn_w = (self.stats['obs*obs.T'][c]
                - obsmean - obsmean.T
                + np.outer(theta, theta)*self.stats['post'][c])

        n_samples = self.n_samples
        # weights_sum = stats['post'][c]
        lik = self.stats['post'][c]*np.log(det(V))/n_samples+np.sum(inv(V)*Sn_w)/n_samples

        return lik

    def _ou_optimize(self, state_id):
        initial_guess = np.random.rand((1,self.n_params))
        # initial_guess = self.params_vec1[state_id].copy()

        method_vec = ['L-BFGS-B','BFGS','SLSQP','Nelder-Mead','Newton-CG']
        id1 = 0

        method_vec = ['L-BFGS-B','BFGS','SLSQP','COBYLA','Nelder-Mead','Newton-CG']
        id1 = 0
        con1 = {'type':'ineq', 'fun': lambda x: x-1e-07}

        res = minimize(ou_lik,initial_guess,args = (self.cv_mtx, state_id,
                        self.stats, self.leaf_time, self.n_samples),
                       constraints=con1, tol=1e-5, options={'disp': False})

        lik = self._ou_lik(res.x, self.cv_mtx, state_id)
        
        return res.x, lik

    def _ou_optimize1(self, state_id):
        
        a1, a2 = 0.35, 0.35
        n1 = self.node_num

        random1 = 2*np.random.rand(self.n_params)-1
        random1[0:-n1] = np.random.rand(self.n_params-n1)
        initial_guess = (a1*self.init_ou_params[state_id].copy() 
        				+ a2*self.params_vec1[state_id].copy()
        				+ 0.3*random1)
        print "initial guess", initial_guess
        
        method_vec = ['L-BFGS-B','BFGS','SLSQP','Nelder-Mead','Newton-CG']
        id1 = 0
        
        con1 = {'type': 'ineq', 'fun': lambda x: x[0:-n1]-1e-07}  # selection strength and variance are positive
        res = minimize(self._ou_lik_varied, initial_guess, args = (state_id),
                       constraints=con1, tol=1e-6, options={'disp': False})

        params1 = res.x
        lik = self._ou_lik_varied(params1, state_id)
        
        return params1, lik

    def _ou_optimize2(self, state_id):
        
        # initial_guess = 1*np.random.rand(self.n_params)
        a1 = self.initial_w1
        a2 = self.initial_w1a
        w2 = self.initial_w2
        n1 = self.node_num

        if self.initial_mode==1:
            random1 = 2*np.random.rand(self.n_params)-1
            random1[0:-n1] = np.random.rand(self.n_params-n1)
            random1 = w2*random1
        else:
            random1 = w2*np.random.rand(self.n_params)
        
        initial_guess = (a1*self.init_ou_params[state_id].copy() 
                        + a2*self.params_vec1[state_id].copy()
                        + (1-a1-a2)*random1)
        print "initial guess", initial_guess
        
        method_vec = ['L-BFGS-B','BFGS','SLSQP','Nelder-Mead','Newton-CG']
        id1 = 0
        
        con1 = {'type': 'ineq', 'fun': lambda x: x[0:-n1]-1e-07}  # selection strength and variance are positive

        res = minimize(self._ou_lik_varied_constraint, initial_guess, args = (state_id),
                       constraints=con1, tol=1e-6, options={'disp': False})

        params1 = res.x
        lik = self._ou_lik_varied_constraint(params1, state_id)
        
        return params1, lik

    def _ou_optimize_init(self, X, mean_values):
        
        initial_guess = self.initial_w2*np.random.rand(self.n_params)
        p_idx = self.parent_list
        leaf_vec = self.leaf_vec
        print "p_idx", p_idx
        print "leaf_vec", leaf_vec
        
        n2 = leaf_vec.shape[0]

        n1 = self.node_num
        mean_values1 = np.zeros(n1)

        flag = np.zeros(n1)
        mean_values1[leaf_vec] = mean_values.copy()
        flag[leaf_vec] = 2

        for j in range(n1-1,0,-1):
        	p_id1 = p_idx[j]
        	if flag[p_id1]==0:
        		mean_values1[p_id1] = mean_values1[j]
        		flag[p_id1] += 1
        	elif flag[p_id1]==1:
        		mean_values1[p_id1] = 0.5*mean_values1[p_id1]+0.5*mean_values1[j]
        		flag[p_id1] += 1

        initial_guess[self.n_params-n1:self.n_params] = mean_values1.copy() # initialize the mean values

        print "initial guess", initial_guess

        method_vec = ['L-BFGS-B','BFGS','SLSQP','Nelder-Mead','Newton-CG']
        id1 = 0

        con1 = {'type': 'ineq', 'fun': lambda x: x[0:-n1]-1e-07}  # selection strength and variance are positive

        res = minimize(self._ou_lik_varied_single, initial_guess, args = (X),
        				constraints=con1, tol=1e-6, options={'disp': False})

        params1 = res.x
        lik = self._ou_lik_varied_single(params1, X)
        
        return params1, lik

    def _do_mstep(self, stats):
        super(phyloHMM1, self)._do_mstep(stats)

        self.stats = stats.copy()
        means_prior = self.means_prior
        means_weight = self.means_weight

        print "M_step"
        # TODO: find a proper reference for estimates for different
        #       covariance models.
        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445
        denom = stats['post'][:, np.newaxis]

        print denom

        if 'c' in self.params:
            print "flag: true covariance"

            for c in range(self.n_components):
                print "state_id: %d"%(c)
                params, value = self._ou_optimize2(c)
                print params
                print value
                self.lik = value
                self.params_vec1[c] = params.copy()

                theta = self.values[self.leaf_vec,0]
                self.means_[c] = theta.copy()
                self._covars_[c] = self.cv_mtx.copy()+self.min_covar*np.eye(self.n_features) 

        print self.params_vec1

        for c in range(self.n_components):
            print("%.4f\t")%(det(self._covars_[c]))
        print("\n")

    def _compute_covariance_mtx_varied(self, params):
        n1, n2 = self.leaf_vec.shape[0], self.node_num  # number of leaf nodes, number of nodes
        values = np.zeros((n2,2))	# expectation and variance
        covar_mtx = np.zeros((n1,n1))
        print "leaf number, node number", n1, n2
        
        num1 = self.branch_dim  # number of branches; assign parameters to each of the branches
        # alpha, sigma, theta1 = params[0:num1], params[num1:2*num1], params1[2*num1:3*num1+1]
        params1 = params[1:]
        beta1, lambda1, theta1 = params1[0:num1], params1[num1:2*num1], params1[2*num1:3*num1+1]

        ratio1 = lambda1/(2*beta1)

        values[0,0] = theta1[0]  # mean value of the root node
        values[0,1] = params[0]
        beta1_exp = np.exp(-beta1)
        beta1_exp = np.insert(beta1_exp,0,0)

        # compute the transformation matrix
        A1, A2, pair_list, p_idx = self.A1, self.A2, np.array(self.pair_list), self.parent_list    

        # add a branch to the first node
        beta1, lambda1, ratio1 = np.insert(beta1,0,0), np.insert(lambda1,0,0), np.insert(ratio1,0,0)
        
        # print p_idx
        print beta1
        # print beta1_exp
        print lambda1
        print theta1
        # print ratio1
        
        for i in range(1,n2):
            values[i,0] = values[p_idx[i],0]*beta1_exp[i] + theta1[i]*(1-beta1_exp[i])
            values[i,1] = ratio1[i]*(1-beta1_exp[i]**2) + values[p_idx[i],1]*(beta1_exp[i]**2)

        # print values
        s1 = np.matmul(A2, beta1)
        idx = pair_list[:,-1]   # index of common ancestor
        s2 = values[idx,1]*np.exp(-s1)
        
        num = pair_list.shape[0]
        leaf_list = self.leaf_list
         
        for k in range(0,num):
            id1,id2 = pair_list[k,0], pair_list[k,1]
            i,j = leaf_list[id1], leaf_list[id2]
            covar_mtx[i,j] = s2[k]
            covar_mtx[j,i] = covar_mtx[i,j]

        for i in range(0,n1):
            covar_mtx[i,i] = values[self.leaf_vec[i],1] # variance

        mean_values1 = values[self.leaf_vec,0]
        return covar_mtx.copy(), mean_values1.copy()

    def _load_simu_parameters(self, filename1):

        simu_params = scipy.io.loadmat(filename1)

        start_prob, equili_prob, transmat, branch_param = simu_params['startprob'], simu_params['equiliprob'], simu_params['transmat'], simu_params['branch_param']
        print branch_param.shape
        # print start_prob, transmat

        n_components, n_components1 = np.array(start_prob).shape[0], np.array(branch_param).shape[0]    # the number of states
        
        if n_components!=n_components1:
            print "the number of components is error!"

        covar_mtx = np.zeros((n_components,self.n_features,self.n_features))
        mean_values = np.zeros((n_components,self.n_features))
        print covar_mtx.shape
        for i in range(0,n_components):
            covar_mtx[i], mean_values[i] = self._compute_covariance_mtx_varied(branch_param[i])
            #print temp1
            #print temp1.shape
            print covar_mtx[i]

        return n_components, start_prob, equili_prob, transmat, mean_values, covar_mtx

    # simulate sequence from the defined covariance matrix and mean values
    def _generate_sequence(self, n_samples, filename1):
        
        n_components2, start_prob, transmat, mean_values, covar_mtx = self._load_simu_parameters(filename1)
        np.random.seed(42)
        model = hmm.GaussianHMM(n_components=n_components2, covariance_type="full")
        model.startprob_ = np.array(start_prob)

        model.transmat_ = np.array(transmat)
        model.means_ = np.array(mean_values)
        model.covars_ = covar_mtx.copy()
        x, z = model.sample(n_samples)

        return x, z

    def _generate_sequence_distribution(self, n_samples, filename1, len_mean, len_std):
        
        n_components, startprob_, equiliprob_, transmat, mean_values, covar_mtx = self._load_simu_parameters(filename1)
        print equiliprob_

        n_features = mean_values.shape[1]
        
        x, y = np.zeros((n_samples,n_features)), np.zeros(n_samples)
        n_samples1 = 0
        
        if equiliprob_.shape[0]!=n_components:
            print "state number not equal! %d %d"%(equiliprob_.shape[0], n_components)

        if np.sum(equiliprob_)!=1:
            print "Error: probabilities not sum to 1"
            equiliprob_ = equiliprob_/np.sum(equiliprob_)
            print equiliprob_

        prob_vec = np.zeros(n_components+1)
        for i in range(0,n_components):
            prob_vec[i+1] = prob_vec[i] + equiliprob_[i] 
        
        len_vec = []
        while n_samples1 < n_samples:
            t1 = np.random.rand(1)[0]
            b = np.where(np.logical_and(t1<=prob_vec[1:],t1>prob_vec[0:-1]))[0]
            state_id = b[0]
            region_len = int(np.max((5,np.random.normal(len_mean, len_std))))
            region_len = int(np.min((200,region_len)))
            if region_len > n_samples-n_samples1:
                region_len = n_samples-n_samples1 
            x1 = sample_gaussian(mean_values[state_id], covar_mtx[state_id], covariance_type='full', n_samples=region_len)
            x[n_samples1:n_samples1+region_len,:] = x1.T
            y[n_samples1:n_samples1+region_len] = state_id
            n_samples1 = n_samples1+region_len
            len_vec.append(region_len)

        n_segment = len(len_vec)
        print "segment number: %d; average segment length: %.2f; max segment length: %.2f; min segment length: %.2f"%(n_segment, np.mean(len_vec), np.max(len_vec), np.min(len_vec))

        return x, y

    def _simu_paramters(n_samples, filename1):
        n_components2, equili_prob, transmat, mean_values, covar_mtx = self._load_simu_parameters(filename1)


    def _check_1(self):
        """Validates model parameters prior to fitting.

        Raises
        ------

        ValueError
            If any of the parameters are invalid, e.g. if :attr:`startprob_`
            don't sum to 1.
        """
        self.startprob_ = np.asarray(self.startprob_)
        if len(self.startprob_) != self.n_components:
            raise ValueError("startprob_ must have length n_components")
        if not np.allclose(self.startprob_.sum(), 1.0):
            raise ValueError("startprob_ must sum to 1.0 (got {0:.4f})"
                             .format(self.startprob_.sum()))

        self.transmat_ = np.asarray(self.transmat_)
        if self.transmat_.shape != (self.n_components, self.n_components):
            raise ValueError(
                "transmat_ must have shape (n_components, n_components)")
        if not np.allclose(self.transmat_.sum(axis=1), 1.0):
            #raise ValueError("rows of transmat_ must sum to 1.0 (got {0})"
            #                 .format(self.transmat_.sum(axis=1)))
            v1 = self.transmat_.sum(axis=1)
            print "rows of transmat_ must sum to 1.0", v1
            b = np.where(v1<1e-07)[0]
            for id1 in b:
                t1 = np.zeros(self.n_components)
                t1[id1] = 1
                self.transmat_[id1] = t1.copy()


class phyloHMM(_BaseHMM):
    
    def __init__(self, n_samples, n_features, edge_list, observation, initial_magnitude,
                 n_components=1, run_id=0, covariance_type='full',
                 min_covar=1e-3,
                 startprob_prior=1.0, transmat_prior=1.0, 
                 means_prior=0, means_weight=0,
                 covars_prior=1e-2, covars_weight=1,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stmc", init_params="stmc", 
                 learning_rate = 0.001):
        _BaseHMM.__init__(self, n_components=n_components, run_id=run_id,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          tol=tol, params=params, verbose=verbose,
                          init_params=init_params)

        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.random_state = random_state
        self.initial_w2 = initial_magnitude
        print "initial magnitude", self.initial_w2

        #self.state_id = 0
        self.observation = observation
        print "data loaded", self.observation.shape
        
        #self.n_samples, self.n_features = observation.shape
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_components = n_components
        self.learning_rate = learning_rate

        self.tree_mtx, self.node_num = self._initilize_tree_mtx(edge_list)
        self.branch_dim, self.n_params = self.node_num, self.node_num
        self.branch_params = 2*np.ones((n_components,self.branch_dim))+0*np.random.rand(n_components,self.branch_dim)

        self.branch_vec = [None]*self.branch_dim
        self.base_struct = [None]*self.branch_dim
        print "compute base struct"
        self.leaf_list = self._compute_base_struct()
        print self.leaf_list
        self.index_list = self._compute_covariance_index()
        print "index_list", self.index_list, len(self.index_list)
        self.base_vec = self._compute_base_mtx()

        #posteriors = np.random.rand(self.n_samples,n_components)
        posteriors = np.ones((self.n_samples,n_components))
        den1 = np.sum(posteriors,axis=1)
        # self.posteriors = posteriors/(np.reshape(den1,(self.n_samples,1))*np.ones((1,n_components)))
        self.posteriors = np.ones((self.n_samples,n_components))    # for testing

        self.mean = np.random.rand(n_components, self.n_features)   # for testing
        #self.mean = tf.placeholder(tf.float32, [n_components, self.n_features])
        self.Sn_w = np.zeros((self.n_components, self.n_features, self.n_features))
        self.stats = dict()
        self.counter = 0

    def _get_covars(self):
        """Return covars as a full matrix."""
        if self.covariance_type == 'full':
            return self._covars_
        elif self.covariance_type == 'diag':
            return np.array([np.diag(cov) for cov in self._covars_])
        elif self.covariance_type == 'tied':
            return np.array([self._covars_] * self.n_components)
        elif self.covariance_type == 'spherical':
            return np.array(
                [np.eye(self.n_features) * cov for cov in self._covars_])
        elif self.covariance_type == 'linear':
            return self._covars_

    def _set_covars(self, covars):
        self._covars_ = np.asarray(covars).copy()

    # TODO: set the mean values
    def _set_means(self, means):
        self.means_ = np.asarray(means).copy()

    covars_ = property(_get_covars, _set_covars)

    def _check(self):
        super(phyloHMM, self)._check()

        self.means_ = np.asarray(self.means_)
        self.n_features = self.means_.shape[1]

        if self.covariance_type not in COVARIANCE_TYPES:
            raise ValueError('covariance_type must be one of {0}'
                             .format(COVARIANCE_TYPES))

        _validate_covars(self._covars_, self.covariance_type,
                         self.n_components)

    def _init(self, X, lengths=None):
        super(phyloHMM, self)._init(X, lengths=lengths)
 
        _, n_features = X.shape
        if hasattr(self, 'n_features') and self.n_features != n_features:
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (n_features, self.n_features))

        self.n_features = n_features
        if 'm' in self.init_params or not hasattr(self, "means_"):
            kmeans = cluster.KMeans(n_clusters=self.n_components,
                                    random_state=self.random_state)
            kmeans.fit(X)
            self.means_ = kmeans.cluster_centers_
        if 'c' in self.init_params or not hasattr(self, "covars_"):
            cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
            if not cv.shape:
                cv.shape = (1, 1)
            self._covars_ = distribute_covar_matrix_to_match_covariance_type(
                cv, self.covariance_type, self.n_components).copy()

    # def _init_branch_length(self):
        
    def _compute_log_likelihood(self, X):
        return log_multivariate_normal_density(
            X, self.means_, self._covars_, self.covariance_type)

    def _generate_sample_from_state(self, state, random_state=None):
        if self.covariance_type == 'tied':
            cv = self._covars_
        else:
            cv = self._covars_[state]
        return sample_gaussian(self.means_[state], cv, self.covariance_type,
                                random_state=random_state)

    def _initialize_sufficient_statistics(self):
        stats = super(phyloHMM, self)._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        stats['obs**2'] = np.zeros((self.n_components, self.n_features))
        stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features,
                                           self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super(phyloHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice)

        if 'm' in self.params or 'c' in self.params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs)

        if 'c' in self.params:
            stats['obs**2'] += np.dot(posteriors.T, obs ** 2)
            stats['obs*obs.T'] += np.einsum(
                    'ij,ik,il->jkl', posteriors, obs, obs)

    def params_Initialization(self, state_num, branch_dim):
        params = {
            'v_branch': weight_init([state_num, branch_dim])
        }

        print "parameters intialized"
        return params

        # initilize the connected graph of the tree given the edges
    def _initilize_tree_mtx(self, edge_list):
        node_num = np.max(np.max(edge_list))+1  # number of nodes; index starting from 0
        tree_mtx = np.zeros((node_num,node_num))
        for edge in edge_list:
            p1, p2 = np.min(edge), np.max(edge)
            tree_mtx[p1,p2] = 1

        print "tree matrix built"
        print tree_mtx

        return tree_mtx, node_num

    # find all the leaf nodes which can be reached from a given node
    def _sub_tree_leaf(self, index):
        tree_mtx = self.tree_mtx   # the connection graph
        idx = np.where(tree_mtx[index,:]>0)[0]
        print idx
        node_vec = []
        if idx.shape[0]==0:
            node_vec = [index]  # the leaf node 
            print "leaf", node_vec
        else:
            for j in idx:
                node_vec1 = self._sub_tree_leaf(j)
                node_vec = node_vec + node_vec1
                print "interior", j, node_vec1

        self.branch_vec[index] = node_vec  # all the leaf nodes that can be reached from this node
        print index, node_vec

        return node_vec
  
    # find all the pairs of leaf nodes which has a given node as the nearest common ancestor
    def _compute_base_struct(self):  
        node_num = self.node_num
        node_vec = self._sub_tree_leaf(0)  # start from the root node
        cnt = 0
        leaf_list = dict()
        for i in range(0,node_num):
            list1 = self.branch_vec[i]  # all the leaf nodes that can be reached from this node
            num1 = len(list1)
            if num1 == 1:
                leaf_list[i] = cnt
                cnt +=1
            self.base_struct[i] = []
            for j in range(0,num1):
                for k in range(j,num1):
                    self.base_struct[i].append(np.array((list1[j],list1[k])))

        print "index built"
        if node_num>2:
            print self.branch_vec[1], self.branch_vec[2]
        return leaf_list

    def _compute_covariance_index(self):
        index = []
        # num1 = self.branch_dim
        num1 = self.node_num  # each node is assigned a branch
        for k in range(0,num1):
            leaf_vec = self.base_struct[k]
            print leaf_vec

        for k in range(0,num1): # starting from index 0
            t_index = []
            leaf_vec = self.base_struct[k]   # the leaf nodes that share this ancestor
            num2 = len(leaf_vec)
            for i in range(0,num2):
                id1, id2 = self.leaf_list[leaf_vec[i][0]], self.leaf_list[leaf_vec[i][1]]
                t_index.append([id1,id2])
                if id1!=id2:
                    t_index.append([id2,id1])
            index.append(t_index)

        return index

    # compute base matrix
    def _compute_base_mtx(self):
        base_vec = dict()
        index, n_features = self.index_list, self.n_features
        num1 = len(index)   # the number of nodes
        base_vec[0] = np.ones((n_features,n_features)) # base matrix for the root node
        # base_vec[0] = np.zeros((n_features,n_features)) # base matrix for the root node
        for i in range(1,num1):
            indices = index[i]
            cv = np.zeros((n_features,n_features))
            num2 = len(indices)
            for j in range(0,num2):
                id1 = indices[j]
                cv[id1[0],id1[1]] = 1
                cv[id1[1],id1[0]] = 1  # symmetric matrix
            base_vec[i] = cv

        for i in range(0,num1):
            filename = "base_mtx_%d.1"%(i)
            np.savetxt(filename, base_vec[i], fmt='%d', delimiter='\t')      

        return base_vec

    # compute covariance matrix for a state
    def _compute_covariance_mtx_2(self, params, state_id):
        #cv1 = self._covars_     # covariance matrix
        branch_dim, n_features = self.branch_dim, self.n_features
        cv = np.zeros((n_features,n_features))
        for i in range(1,branch_dim):
            cv += params[i]*self.base_vec[i]
        return cv

    # compute covariance matrix for a state
    def _compute_covariance_mtx_2alt(self, state_id):
        num = self.n_features
        branch_params, index = self.branch_params[state_id], self.index_list
        cv = np.zeros((num,num))
        num1 = len(index)
        for i in range(0,num1):
            indices = index[i]
            for id1 in indices:
                cv[id1[0],id1[1]] += branch_params[i]
                cv[id1[1],id1[0]] += branch_params[i]  # symmetric matrix

        return cv

    def _compute_log_likelihood_2(self, params, state_id):
        cv, n_samples = self._compute_covariance_mtx_2(params, state_id), self.n_samples
        inv_cv = inv(cv)
        weights_sum = self.stats['post'][state_id]
        Sn = self.Sn_w[state_id]/n_samples
        likelihood = weights_sum*np.log(det(cv))/n_samples+np.matrix.trace(np.matmul(Sn,inv_cv))
        
        return likelihood

    # search for a step size by line search  
    def _line_search(self, params, derivative, state_id):
        alpha_vec = np.arange(0.001,20,0.001)
        n1 = alpha_vec.shape[0]
        vec, flag = np.zeros(n1), np.zeros(n1)
        thresh = 1e-04   # threshold for the branch length
        n_params = params.shape[0]
        for k in range(0,n1):
            params1 = params-alpha_vec[k]*derivative
            b = np.where(params1[1:]<thresh)[0] # whether there is negative branch length
            if b.shape[0]==0:
                vec[k] = self._compute_log_likelihood_2(params1, state_id)
                flag[k] = 1
        b = np.where(flag==1)[0]
        if b.shape[0]>0:
            idx = np.argmin(vec[b])
            alpha = alpha_vec[b[idx]]
        else:
            temp1 = params[1:]/(derivative[1:]+1e-10)
            b2 = np.where(temp1>0)[0]
            idx2 = np.argmin(temp1[b2])
            alpha = 0.5*temp1[b2[idx2]]

        return alpha

    def _gradient_descent(self, state_id):
        
        thresh, distance, distance1, iteration, cnt, pre_derivative = 1e-05, 1, 1, 0, 0, 0
        params = self.branch_params[state_id].copy()
        pre_params, n_params = params.copy(), self.branch_dim
        flags = []

        derivative, pre_value, cv = self._directional_derivative(params,state_id)
        cnt, cnt1, cnt_limit1, cnt_limit2, cnt_limit3 = 0, 0, 3, 3, 1000
        print pre_value

        while cnt<cnt_limit1 and iteration<cnt_limit3:
            # derivative, value = self._directional_derivative(params,state_id, X)
            delta1 = params-pre_params
            pre_params = np.array(params)
            delta2 = derivative-pre_derivative
            pre_derivative = np.array(derivative)
            flag = 1
            step1 = np.dot(delta1,delta2)/(np.dot(delta2,delta2)+1e-10)
            print step1
            r1 = step1*derivative
            if norm(r1)<1e-05 or step1<-0.01:
                flag = 0
            if iteration < cnt_limit2:
                step = self._line_search(params, derivative, state_id)
                update = step*derivative
                if norm(update)>1e-05:
                    flag = 0
            if flag == 1:
                update = r1
                step = step1
            else:
                step = self._line_search(params, derivative, state_id)
                update = step*derivative
            
            flags.append(flag)
            params = params - update  # update the gradient
            d1 = params[1:]-pre_params[1:] # starting from children of the root node
            # value = self._compute_log_likelihood_2(params, state_id)
            derivative, value, cv = self._directional_derivative(params,state_id)
            distance, distance1 = pre_value-value, norm(d1)/(n_params-1)
            pre_value = value
            iteration += 1
            if iteration%10==0:
                print"%d %.6f %.6f %.6f %.6f %d"%(iteration, value, distance, step, step1, flag)
            if np.abs(distance)<thresh:
                cnt += 1
            else:
                cnt = 0
            if distance<0:
                cnt1 += 1
            else:
                cnt1 = 0

        return params, value, cv

    def _directional_derivative(self, params, state_id):
        
        n_params, n_samples, n_features, base_vec = self.branch_dim, self.n_samples, self.n_features, self.base_vec

        Sn = self.Sn_w[state_id]
        weights_sum = self.stats['post'][state_id]
        mtx_vec = [None]*n_params
        dv = np.zeros(n_params)
        thresh = 0.001
        
        cv = self._compute_covariance_mtx_2(params, state_id)
        inv_cv = inv(cv) # compute inverse matrix
        # print inv_cv
        mtx1 = np.matmul(Sn,inv_cv)
        # compute the likelihood
        likelihood = weights_sum*np.log(det(cv))/n_samples+np.matrix.trace(mtx1)/n_samples 
        for i in range(0,n_params):
            mtx2 = np.matmul(base_vec[i],inv_cv)
            dv[i] = weights_sum*np.matrix.trace(mtx2)/n_samples-np.matrix.trace(np.matmul(mtx1,mtx2))/n_samples

        return dv, likelihood, cv

    def _directional_derivative_1(self, state_id, X):
        
        params, param_dim, n_samples, n_features, base_vec = self.branch_params, self.branch_dim, self.n_samples, self.n_features, self.base_vec

        Sn = self.Sn_w[state_id]
        weights_sum = self.stats['post'][state_id]
        mtx_vec = [None]*n_params
        dv = np.zeros(n_params)
        thresh = 0.001
        
        cv = self._compute_covariance_mtx_2(params, state_id)
        inv_cv = inv(cv) # compute inverse matrix
        # print inv_cv
        mtx1 = np.matmul(S1,inv_cv)
        for i in range(0,param_dim):
            mtx_vec[i] = np.matmul(base_vec[i],inv_cv)
        for i in range(0,param_dim): 
            mtx2 = mtx_vec[i]
            dv[i] = weights_sum*np.matrix.trace(mtx2)-np.matrix.trace(np.matmul(mtx1,mtx2))
            for j in range(0,param_dim):
                temp1 = np.matmul(mtx_vec[j],mtx2)
                dv2[i,j] = weights_sum*np.matrix.trace(temp1)-2*np.matrix.trace(np.matmul(mtx1,temp1))

        return dv

    def _brownian_lik(self, params, state_id):
        
        c = state_id

        cv = np.zeros((self.n_features, self.n_features))

        i = 0
        for branch_param in params:
            cv += branch_param*self.base_vec[i]
            i += 1
        
        # weights_sum = stats['post'][c]
        lik = self.stats['post'][state_id]*np.log(det(cv))/self.n_samples+np.sum(inv(cv)*self.Sn_w[state_id])/self.n_samples
        # lik = stats['post'][c]*np.log(det(cv))/n_samples+np.matrix.trace(np.matmul(Sn_w,inv(cv)))/n_samples

        return lik

    def _brownian_optimize(self, state_id):
        
        initial_guess = self.initial_w2*np.random.rand(1,self.n_params)

        method_vec = ['L-BFGS-B','BFGS','SLSQP','COBYLA','Nelder-Mead','Newton-CG']
        id1 = 0
        # con1 = {'type': 'ineq', 'fun': constraint1}
        con1 = {'type':'ineq', 'fun': lambda x: x-1e-07}

        res = minimize(self._brownian_lik,initial_guess,args = (state_id),
                        constraints=con1, tol=1e-5, options={'disp': False})

        lik = self._brownian_lik(res.x, state_id)
        cv = np.zeros((self.n_features, self.n_features))
        i = 0
        for branch_param in res.x:
            cv += branch_param*self.base_vec[i]
            i += 1
        
        return res.x, lik, cv

    def _output_stats(self, number):
        filename = "log1/stats_iter_%d"%(number)
        np.savetxt(filename, self.stats['post'], fmt='%.4f', delimiter='\t')


    def _do_mstep(self, stats):
        super(phyloHMM, self)._do_mstep(stats)

        self.stats = stats
        means_prior = self.means_prior
        means_weight = self.means_weight

        print "M_step"

        # TODO: find a proper reference for estimates for different
        #       covariance models.
        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445
        denom = stats['post'][:, np.newaxis]

        print denom
        if 'm' in self.params:
            print "flag: true mean"
            self.means_ = ((means_weight * means_prior + stats['obs'])
                           / (means_weight + denom))

        if 'c' in self.params:
            print "flag: true covariance"
            covars_prior = self.covars_prior
            covars_weight = self.covars_weight
            meandiff = self.means_ - means_prior

            for c in range(self.n_components):
                print "state_id: %d"%(c)
                obsmean = np.outer(stats['obs'][c], self.means_[c])
                self.Sn_w[c] = (means_weight * np.outer(meandiff[c],meandiff[c]) 
                             + stats['obs*obs.T'][c]
                             - obsmean - obsmean.T
                             + np.outer(self.means_[c], self.means_[c])
                             * stats['post'][c])

                # params, value, cv = self._gradient_descent(c)
                params, value, cv = self._brownian_optimize(c)

                print params 
                print value
                self.branch_params[c] = params.copy()
                self._covars_[c] = cv.copy()+self.min_covar*np.eye(self.n_features)

        print self.branch_params
        print self.means_
        print self._covars_
        for c in range(self.n_components):
            print("%.4f\t")%(det(self._covars_[c]))
        print("\n")

        print self.startprob_
        print self.transmat_


class GaussianHMM(_BaseHMM):
    """Hidden Markov Model with Gaussian emissions.

    Parameters
    ----------
    n_components : int
        Number of states.

    covariance_type : string, optional
        String describing the type of covariance parameters to
        use.  Must be one of

        * "spherical" --- each state uses a single variance value that
          applies to all features.
        * "diag" --- each state uses a diagonal covariance matrix.
        * "full" --- each state uses a full (i.e. unrestricted)
          covariance matrix.
        * "tied" --- all states use **the same** full covariance matrix.

        Defaults to "diag".

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting. Defaults to 1e-3.

    startprob_prior : array, shape (n_components, ), optional
        Parameters of the Dirichlet prior distribution for
        :attr:`startprob_`.

    transmat_prior : array, shape (n_components, n_components), optional
        Parameters of the Dirichlet prior distribution for each row
        of the transition probabilities :attr:`transmat_`.

    means_prior, means_weight : array, shape (n_components, ), optional
        Mean and precision of the Normal prior distribtion for
        :attr:`means_`.

    covars_prior, covars_weight : array, shape (n_components, ), optional
        Parameters of the prior distribution for the covariance matrix
        :attr:`covars_`.

        If :attr:`covariance_type` is "spherical" or "diag" the prior is
        the inverse gamma distribution, otherwise --- the inverse Wishart
        distribution.

    algorithm : string, optional
        Decoder algorithm. Must be one of "viterbi" or`"map".
        Defaults to "viterbi".

    random_state: RandomState or an int seed, optional
        A random number generator instance.

    n_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.

    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'm' for means and 'c' for covars. Defaults
        to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'm' for means and 'c' for covars.
        Defaults to all parameters.

    Attributes
    ----------
    n_features : int
        Dimensionality of the Gaussian emissions.

    monitor\_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    transmat\_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    startprob\_ : array, shape (n_components, )
        Initial state occupation distribution.

    means\_ : array, shape (n_components, n_features)
        Mean parameters for each state.

    covars\_ : array
        Covariance parameters for each state.

        The shape depends on :attr:`covariance_type`::

            (n_components, )                        if "spherical",
            (n_features, n_features)                if "tied",
            (n_components, n_features)              if "diag",
            (n_components, n_features, n_features)  if "full"

    Examples
    --------
    >>> from hmmlearn.hmm import GaussianHMM
    >>> GaussianHMM(n_components=2)
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    GaussianHMM(algorithm='viterbi',...
    """
    def __init__(self, n_components=1, covariance_type='diag',
                 min_covar=1e-3,
                 startprob_prior=1.0, transmat_prior=1.0,
                 means_prior=0, means_weight=0,
                 covars_prior=1e-2, covars_weight=1,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stmc", init_params="stmc"):
        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          tol=tol, params=params, verbose=verbose,
                          init_params=init_params)

        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight

    def _get_covars(self):
        """Return covars as a full matrix."""
        if self.covariance_type == 'full':
            return self._covars_
        elif self.covariance_type == 'diag':
            return np.array([np.diag(cov) for cov in self._covars_])
        elif self.covariance_type == 'tied':
            return np.array([self._covars_] * self.n_components)
        elif self.covariance_type == 'spherical':
            return np.array(
                [np.eye(self.n_features) * cov for cov in self._covars_])

    def _set_covars(self, covars):
        self._covars_ = np.asarray(covars).copy()

    covars_ = property(_get_covars, _set_covars)

    def _check(self):
        super(GaussianHMM, self)._check()

        self.means_ = np.asarray(self.means_)
        self.n_features = self.means_.shape[1]

        if self.covariance_type not in COVARIANCE_TYPES:
            raise ValueError('covariance_type must be one of {0}'
                             .format(COVARIANCE_TYPES))

        _validate_covars(self._covars_, self.covariance_type,
                         self.n_components)

    def _init(self, X, lengths=None):
        super(GaussianHMM, self)._init(X, lengths=lengths)

        _, n_features = X.shape
        if hasattr(self, 'n_features') and self.n_features != n_features:
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (n_features, self.n_features))

        self.n_features = n_features
        if 'm' in self.init_params or not hasattr(self, "means_"):
            kmeans = cluster.KMeans(n_clusters=self.n_components,
                                    random_state=self.random_state)
            kmeans.fit(X)
            self.means_ = kmeans.cluster_centers_
        if 'c' in self.init_params or not hasattr(self, "covars_"):
            cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
            if not cv.shape:
                cv.shape = (1, 1)
            self._covars_ = distribute_covar_matrix_to_match_covariance_type(
                cv, self.covariance_type, self.n_components).copy()

    def _compute_log_likelihood(self, X):
        return log_multivariate_normal_density(
            X, self.means_, self._covars_, self.covariance_type)

    def _generate_sample_from_state(self, state, random_state=None):
        if self.covariance_type == 'tied':
            cv = self._covars_
        else:
            cv = self._covars_[state]
        return sample_gaussian(self.means_[state], cv, self.covariance_type,
                               random_state=random_state)

    def _initialize_sufficient_statistics(self):
        stats = super(GaussianHMM, self)._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        stats['obs**2'] = np.zeros((self.n_components, self.n_features))
        if self.covariance_type in ('tied', 'full'):
            stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features,
                                           self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super(GaussianHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice)

        if 'm' in self.params or 'c' in self.params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs)

        if 'c' in self.params:
            if self.covariance_type in ('spherical', 'diag'):
                stats['obs**2'] += np.dot(posteriors.T, obs ** 2)
            elif self.covariance_type in ('tied', 'full'):
                # posteriors: (nt, nc); obs: (nt, nf); obs: (nt, nf)
                # -> (nc, nf, nf)
                stats['obs*obs.T'] += np.einsum(
                    'ij,ik,il->jkl', posteriors, obs, obs)

    def _do_mstep(self, stats):
        super(GaussianHMM, self)._do_mstep(stats)

        means_prior = self.means_prior
        means_weight = self.means_weight

        # TODO: find a proper reference for estimates for different
        #       covariance models.
        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445
        denom = stats['post'][:, np.newaxis]
        if 'm' in self.params:
            self.means_ = ((means_weight * means_prior + stats['obs'])
                           / (means_weight + denom))

        if 'c' in self.params:
            covars_prior = self.covars_prior
            covars_weight = self.covars_weight
            meandiff = self.means_ - means_prior

            if self.covariance_type in ('spherical', 'diag'):
                cv_num = (means_weight * meandiff**2
                          + stats['obs**2']
                          - 2 * self.means_ * stats['obs']
                          + self.means_**2 * denom)
                cv_den = max(covars_weight - 1, 0) + denom
                self._covars_ = \
                    (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)
                if self.covariance_type == 'spherical':
                    self._covars_ = np.tile(
                        self._covars_.mean(1)[:, np.newaxis],
                        (1, self._covars_.shape[1]))
            elif self.covariance_type in ('tied', 'full'):
                cv_num = np.empty((self.n_components, self.n_features,
                                  self.n_features))
                for c in range(self.n_components):
                    obsmean = np.outer(stats['obs'][c], self.means_[c])

                    cv_num[c] = (means_weight * np.outer(meandiff[c],
                                                         meandiff[c])
                                 + stats['obs*obs.T'][c]
                                 - obsmean - obsmean.T
                                 + np.outer(self.means_[c], self.means_[c])
                                 * stats['post'][c])
                cvweight = max(covars_weight - self.n_features, 0)
                if self.covariance_type == 'tied':
                    self._covars_ = ((covars_prior + cv_num.sum(axis=0)) /
                                     (cvweight + stats['post'].sum()))
                elif self.covariance_type == 'full':
                    self._covars_ = ((covars_prior + cv_num) /
                                     (cvweight + stats['post'][:, None, None]))

def parse_args():
    parser = OptionParser(usage="Replication timing state estimation", add_help_option=False)
    parser.add_option("-h","--hmm", default="false", help="Perform HMM estimation (default: false")
    parser.add_option("-n", "--num_states", default="8", help="Set the number of states to estimate for HMM model")
    parser.add_option("-f","--filename", default="brownian_data4.mat", help="Filename of dataset")
    parser.add_option("-l","--length", default="one", help="Filename of length vectors")
    parser.add_option("-p","--root_path", default="/home/yy3/data1/replication_timing/hmmseg/vbak2/em", help="Root directory of the data files")
    parser.add_option("-m","--multiple", default="true", help="Use multivariate data (true, default) or single variate data (false) ")
    parser.add_option("-a","--species_name", default="human", help="Species to estimate states (used under single variate mode)")
    parser.add_option("-o","--sort_states", default="false", help="Whether to sort the states")
    parser.add_option("-r","--run_id", default="0", help="experiment id")
    parser.add_option("-c","--cons_param", default="1", help="constraint parameter")
    parser.add_option("-t","--method_mode", default="3", help="method_mode: 0: Gaussian-HMM; 1: Kmeans; 2: BM-HMM; 3: OU-HMM")
    parser.add_option("-d","--initial_mode", default="0", help="initial mode: 0: positive random vector; 1: positive random vector for branches")
    parser.add_option("-i","--initial_weight", default="0.2", help="initial weight 0 for initial parameters")
    parser.add_option("-k","--initial_weight1", default="0", help="initial weight 1 for initial parameters")
    parser.add_option("-j","--initial_magnitude", default="1", help="initial magnitude for initial parameters")
    parser.add_option("-s","--version", default="12", help="dataset version")


    (opts, args) = parser.parse_args()
    return opts

def run(hmm_estimate,num_states,filename,length_vec,root_path,multiple,species_name,
        sort_states,run_id1,cons_param,method_mode,initial_mode,initial_weight,initial_weight1,initial_magnitude, simu_version):
    
     # load the edge list
    filename2 = "input_example/edge.1.txt"    
    if(os.path.exists(filename2)==True):
        f = open(filename2, 'r')
        print("edge list loaded")
        edge_list = [map(int,line.split('\t')) for line in f]
        print edge_list

    # load branch length file if provided
    filename2 = "input_example/branch_length.1.txt"
    if(os.path.exists(filename2)==True):
        f = open(filename2, 'r')
        print("branch list loaded")
        branch_list = [map(float,line.split('\t')) for line in f]
        branch_list = branch_list[0]
        print branch_list

    learning_rate = 0.001
    run_id = int(run_id1)
    n_components1 = int(num_states)
    cons_param = float(cons_param)
    initial_weight = float(initial_weight)
    initial_magnitude = float(initial_magnitude)
    method_mode = int(method_mode)
    version = int(version)

    # load the features
    filename1 = "input_example/sig.feature.1.txt" # input
    if(os.path.exists(filename1)==False):
        print "there is no such file %s"%(filename1)
        return
    filename2 = "input_example/sig.lenVec.1.txt"  # input

    if(os.path.exists(filename2)==False):
        print "there is no such file %s"%(filename2)
        return

    x1 = np.loadtxt(filename1, dtype='float', delimiter='\t')

    x2 = np.zeros(x1.shape)
    base_num = x1.shape[1]
    for i in range(0,base_num):
        x2[:,i] = x1[:,base_num-1-i]

    # x = np.log(x)
    x = x2.copy()
    print x.shape
    print x[0]

    len_vec = np.loadtxt(filename2,dtype='int32',delimiter='\t')
    print sum(len_vec)

    path_1 = ""  # please define the output directory

    learning_rate=0.001
    run_id = int(run_id1)
    n_components1 = int(num_states)
    cons_param = float(cons_param)
    simu_version = int(simu_version)
    cons_param = float(cons_param)
    initial_mode = int(initial_mode)
    initial_weight = float(initial_weight)
    initial_weight1 = float(initial_weight1)
    initial_magnitude = float(initial_magnitude)
    method_mode = int(method_mode)

    annot = "phly-hmgp"
    
    if not os.path.exists(path_1):
        try:
            os.makedirs(path_1)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise    

    if method_mode==0:
        tree1 = phyloHMM(n_components=n_components1, run_id=run_id, n_samples = x.shape[0], n_features = x.shape[1], 
                     observation=x, edge_list=edge_list, initial_magnitude = initial_magnitude, 
                     learning_rate=learning_rate, n_iter=5000, tol=1e-6)
        tree1.fit(x, lengths=len_vec)
        print "predicting states..."
        state_est = tree1.predict(x,len_vec)

        # save the estimated states and parameters
        filename3 = "%s/estimate_bm_%d_%d_s%d.txt"%(path_1, run_id, n_components1, version)
        np.savetxt(filename3, state_est, fmt='%d', delimiter='\t')
        filename1 = "%s/branch_bm_%d_%d_s%d.txt"%(path_1, run_id, n_components1, version)
        np.savetxt(filename1, tree1.branch_params, fmt='%.6f', delimiter='\t')
        filename1 = "%s/means_bm_%d_%d_s%d.txt"%(path_1, run_id, n_components1, version)
        np.savetxt(filename1, tree1.means_, fmt='%.6f', delimiter='\t')
        filename1 = "%s/trans_bm_%d_%d_s%d.txt"%(path_1, run_id, n_components1, version)
        np.savetxt(filename1, tree1.transmat_, fmt='%.6f', delimiter='\t')
        filename1 = "%s/start_bm_%d_%d_s%d.txt"%(path_1, run_id, n_components1, version)
        np.savetxt(filename1, tree1.startprob_, fmt='%.6f', delimiter='\t')

    elif method_mode==1:
        tree1 = phyloHMM1(n_components=n_components1, run_id=run_id, n_samples = x.shape[0], n_features = x.shape[1], 
                     observation=x, edge_list=edge_list, branch_list=branch_list, cons_param=cons_param, initial_mode = initial_mode, 
                     initial_weight = initial_weight, initial_weight1 = initial_weight1, initial_magnitude = initial_magnitude, 
                     learning_rate=learning_rate, n_iter=5000, tol=1e-6)
        tree1.fit(x, lengths=len_vec)

        lambda_0 = cons_param

        print tree1.startprob_
        print tree1.transmat_

        tree1._check_1()

        print "predicting states..."
        state_est = tree1.predict(x,len_vec)

        # save the estimated states and parameters
        filename3 = "%s/estimate_ou_%d_%.2f_%d_s%d.txt"%(path_1, run_id, lambda_0, n_components1, version)
        state_est = tree1.predict(x,len_vec)
        np.savetxt(filename3, state_est, fmt='%d', delimiter='\t')
        filename1 = "%s/params_ou_%d_%.2f_%d_s%d.txt"%(path_1, run_id, lambda_0, n_components1, version)
        np.savetxt(filename1, tree1.params_vec1, fmt='%.6f', delimiter='\t')
        filename1 = "%s/trans_ou_%d_%.2f_%d_s%d.txt"%(path_1, run_id, lambda_0, n_components1, version)
        np.savetxt(filename1, tree1.transmat_, fmt='%.6f', delimiter='\t')
        filename1 = "%s/start_ou_%d_%.2f_%d_s%d.txt"%(path_1, run_id, lambda_0, n_components1, version)
        np.savetxt(filename1, tree1.startprob_, fmt='%.6f', delimiter='\t')

    else:
        pass
    
if __name__ == '__main__':

    opts = parse_args()
    run(opts.hmm,opts.num_states,opts.filename,opts.length,opts.root_path,opts.multiple,\
        opts.species_name,opts.sort_states,opts.run_id,opts.cons_param, opts.method_mode, \
        opts.initial_mode, opts.initial_weight, opts.initial_weight1, opts.initial_magnitude, opts.simu_version)


