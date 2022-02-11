import logging
from random import choice
from random import randint
from time import time
from typing import List

import numpy as np
import scipy
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.base import ClusterMixin
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score as ari


class CoClust(BaseEstimator, ClusterMixin, TransformerMixin):
    """
    Parameter-less co-clustering algorithm

    """

    def __init__(self, n_iterations=500, n_iter_per_mode = 1, initialization= 'extract_centroids', k = [20,20,20], verbose = False):
        """
        Create the model object and initialize the required parameters.

        :type n_iterations: int
        :param n_iterations: the max number of iterations to perform
        :type n_iter_per_mode: int
        :param n_iter_per_mode: the max number of sub-iterations for eac iteration and each mode
        :type initialization: string
        :param initialization: the initialization method, one of {'random', 'discrete', 'random_optimal', 'extract_centroids', 'customized'}
        :type k: array-like of integers
        :param k: number of clusters on each mode. 
        :type verbose: boolean
        :param verbose: if True, it prints details of the computations
        """

        self.n_iterations = n_iterations
        self.n_iter_per_mode = n_iter_per_mode
        self.initialization = initialization
        self.k = np.array(k)
        self.verbose = verbose

        np.seterr(all='ignore')

    def _init_all(self, V):
        """
        Initialize all variables needed by the model.

        :param V: the dataset
        :return:
        """
        # verify that all matrices are correctly represented
        # check_array is a sklearn utility method
        self._dataset = None

        self._dataset = check_array(V, accept_sparse='csr', ensure_2d = False, allow_nd = True, dtype=[np.int32, np.int8, np.float64, np.float32])

        self._csc_dataset = None
        if issparse(self._dataset):
            # transform also to csc
            self._csc_dataset = self._dataset.tocsc()
            

        # the number of modes and dimensions on each mode
        self._n = np.array(self._dataset.shape)
        self._n_modes = len(self._dataset.shape)


        # the number of row/ column clusters
        self._n_clusters = np.zeros(self._n_modes, dtype = 'int16')


        # a list of n_documents (n_features) elements
        # for each document (feature) d contains the row cluster index d is associated to
        self._assignment = [np.zeros(self._n[i], 'int16')for i in range(self._n_modes)]


        # computation time
        self.execution_time_ = 0

        self._tot = np.sum(self._dataset)
        self._dataset = self._dataset/self._tot
        self.tau = list()
        
        if (self.initialization == 'discrete'):
            self._discrete_initialization()
        elif self.initialization == 'extract_centroids':
            self._extract_centroids_initialization()
        else:
            raise ValueError("The only valid initialization methods are: discrete, extract_centroids")


    def fit(self, V, y=None):
        """
        Fit CoClust to the provided data.

        Parameters
        -----------

        V : array-like or sparse matrix;
            shape of the matrix = (n_documents, n_features)

        y : unused parameter

        Returns
        --------

        self

        """

        # Initialization phase
        self._init_all(V)

        self._T = self._init_contingency_tensor(0)[1]
        self.tau.append(self._compute_taus())

        start_time = time()

        # Execution phase
        self._actual_n_iterations = 0 #conta il numero totale di iterazioni
        actual_n_iterations = 0 #conta come una sola iterazione un intero giro, spostamenti delle x + spostamenti delle y
        
        while actual_n_iterations < self.n_iterations:
            actual_iteration = np.zeros(self._n_modes, dtype = int)
            for m in range(self._n_modes):
                #actual_iteration_x = 0    # all'interno di ogni iterazione vengono fatte più iterazioni consecutive su ogni modo
                cont = True
                while cont:
                    # each iterations performs a move on rows

                    iter_start_time = time()

                    # perform a move within the rows partition
                    cont = self._perform_move(m)
                    #print( '############################' )
                    #self._perform_col_move()
                    #print( '############################' )

                    actual_iteration[m] += 1
                    self._actual_n_iterations +=1 
                    iter_end_time = time()

                    if actual_iteration[m] > self.n_iter_per_mode:
                        cont = False
                    if self.verbose:
                        self._T = self._init_contingency_tensor(0)[1]
                        self.tau.append(self._compute_taus())

            

                
            if np.sum(actual_iteration) == self._n_modes:
                actual_n_iterations = self.n_iterations
            else:
                actual_n_iterations += 1
            

        end_time = time()
        if not self.verbose:
            self._T = self._init_contingency_tensor(0)[1]
            self.tau.append(self._compute_taus())

        execution_time = end_time - start_time

        logging.info('#####################################################')
        logging.info("[INFO] Execution time: {0}".format(execution_time))

        # clone cluster assignments and transform in lists
        self.execution_time_ = execution_time
        self.labels_ = [np.copy(self._assignment[i]) for i in range(self._n_modes)]

        return self



    def _discrete_initialization(self):

        for m in range(self._n_modes):
            # simple assign each row to a row cluster and each column of a view to a column cluster
            self._n_clusters[m] = self._n[m]

            # assign each row to a row cluster
            self._assignment[m] = np.arange(self._n[m])

            

    def _extract_centroids_initialization(self):
        if len(self.k) == 0 :
            raise ValueError("Parameter k is needed when initialization = 'extract_centroids'")


        if np.sum(self.k>self._n) >0:
            raise ValueError("The number of clusters must be <= the number of objects, on all dimensions")

        self._n_clusters = np.copy(self.k)
        dataset = np.copy(self._dataset)

        for d in range(self._n_modes):
        
            a = np.random.choice(self._n[d], self._n_clusters[d], replace = False)
            T = dataset[a]
            T = T/np.sum(T)
            for i in range(self._n[d]):
                all_tau = np.sum(np.nan_to_num(np.true_divide(dataset[i], np.sum(dataset,0))* T, nan = 0), (1,2)) - np.sum(dataset[i])*np.sum(T,(1,2))
                #all_tau = np.sum(np.nan_to_num(dataset[i] * np.true_divide(T, np.sum(T,0)), nan = 0), (1,2)) - np.sum(dataset[i])*np.sum(T,(1,2))

                max_tau = np.max(all_tau)

                #if max_tau >=0:
                e_max = np.where(max_tau == all_tau)[0][0]
                self._assignment[d][i] = e_max
                #else:
                #    self._assignment[d][i] = self._n_clusters[d]
                #    self._n_clusters[d] +=1
            dataset = dataset.transpose(tuple(np.arange(1,self._n_modes)) + tuple([0]))
            self._check_clustering(d)



    def _check_clustering(self, dimension):
        
        k = len(set(self._assignment[dimension]))
        h = [j for j in range(k) if j not in set(self._assignment[dimension])]
        p = 0
        for i in set(self._assignment[dimension]):
            if i >=k:
                self._assignment[dimension][self._assignment[dimension]==i] = h[p]
                p +=1
        self._n_clusters[dimension] = k


        
            
    def _init_contingency_tensor(self, dimension):  
        """
        Initialize the T contingency tensor
        :return:
            - the dataset with the other modes aggregated according to the current co-clustering
            - the contingency tensor
        """
        logging.debug("[INFO] Compute the contingency matrix...")

        dataset = self._update_dataset(dimension)

        t = tuple([self._n_clusters[i] for i in range(self._n_modes) if i != dimension])
        new_t = np.zeros(tuple([self._n_clusters[dimension]]) + t)


        for i in range(self._n_clusters[dimension]):
            new_t[i] = np.sum(dataset[self._assignment[dimension] == i], axis = 0)
        

        logging.debug("[INFO] End of contingency matrix computation...")
        
        return dataset, new_t

    def _update_dataset(self, dimension): 

        n = tuple([i for i in range(self._n_modes) if i != dimension])
        t = tuple([self._n_clusters[i] for i in range(self._n_modes) if i != dimension])
        d = tuple([self._n[i] for i in range(self._n_modes) if i != dimension])
        dataset = np.transpose(self._dataset, tuple([dimension]) + n)
        new_t = np.zeros(dataset.shape[:-1] + tuple([t[-1]]))

        for m in range(self._n_modes -1):
            #print(m)
            dataset = np.transpose(dataset, tuple([-1]) + tuple(np.arange(self._n_modes -1)))
            new_t = np.transpose(new_t, tuple([-1]) + tuple(np.arange(self._n_modes -1)))
            #print("dataset:", dataset.shape)
            #print("new_t:", new_t.shape)
            for i in range(t[-m -1]):
                new_t[i] = np.sum(dataset[self._assignment[n[-1-m]] == i], axis = 0)

            if m < self._n_modes -2:
                dataset = np.copy(new_t)
                new_t = np.zeros(dataset.shape[:-1] + tuple([t[-2-m]]))
                #print("dataset:", dataset.shape)
                #print("new_t:", new_t.shape)
        new_t = np.transpose(new_t, tuple([-1]) + tuple(np.arange(self._n_modes -1)))
        #print("new_t:", new_t.shape)

        return new_t


    def _perform_move(self, dimension):
        """
        Perform a single move to improve the partition on rows.

        :return:
        """
        #dataset = self._update_dataset(0)
        dataset, T = self._init_contingency_tensor(dimension)
        moves = 0

        for i in range(self._n[dimension]):
            all_tau = np.sum(np.nan_to_num(dataset[i] * np.true_divide(T, np.sum(T,0)), nan = 0), (1,2)) - np.sum(dataset[i])*np.sum(T,(1,2))
            max_tau = np.max(all_tau)

            equal_solutions = np.where(max_tau == all_tau)[0]
            e_min = equal_solutions[0]
            if e_min != self._assignment[dimension][i]:
                moves += 1

                    
            self._assignment[dimension][i] = e_min
            
        self._check_clustering(dimension)
        
        if self.verbose:
            print(f"iteration {self._actual_n_iterations}, moving mode {dimension}, n_clusters: {self._n_clusters}")
        if moves ==0:
            return False
        else:
            return True



    def _compute_taus(self):
        """
        Compute the value of tau_x, tau_y and tau_z

        :return: a tuple (tau_x, tau_y, tau_z)
        """
        
        tau = np.zeros(self._n_modes)
        for j in range(self._n_modes):
            d = tuple([i for i in range(self._n_modes) if i != j])
            a = np.sum(np.nan_to_num(np.true_divide(np.sum(np.power(self._T, 2), axis = d), np.sum(self._T, axis = d)))) # scalar
            b = np.sum(np.power(np.sum(self._T, axis = j), 2)) #scalar
            tau[j] = np.nan_to_num(np.true_divide(a - b, 1 - b))

        
        #logging.debug("[INFO] a_x, a_y, a_z, b_x, b_y, b_z: {0},{1}, {2}, {3}, {4}, {5}".format(a_x, a_y, a_z, b_x, b_y, b_z))

        return tau



