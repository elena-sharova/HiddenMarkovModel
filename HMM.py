# -*- coding: utf-8 -*-
"""
Created on Wed Sep 07 10:18:13 2016

@author: Elena Sharova
"""
import numpy as np
import pandas as pd
import json as js
from yahoo_finance import Share
import datetime as dt


class HMM(object):
    # Implements discrete 1-st order Hidden Markov Model 

    def __init__(self, tolerance = 1e-6, max_iterations=10000, scaling=True):
        self.tolerance=tolerance
        self.max_iter = max_iterations
        self.scaling = scaling

    def HMMfwd(self, a, b, o, pi):
        # Implements HMM Forward algorithm
    
        N = np.shape(b)[0]
        T = np.shape(o)[0]
    
        alpha = np.zeros((N,T))
        # initialise first column with observation values
        alpha[:,0] = pi*b[:,o[0]]
        c = np.ones((T))
        
        if self.scaling:
            
            c[0]=1.0/np.sum(alpha[:,0])
            alpha[:,0]=alpha[:,0]*c[0]
            
            for t in xrange(1,T):
                c[t]=0
                for i in xrange(N):
                    alpha[i,t] = b[i,o[t]] * np.sum(alpha[:,t-1] * a[:,i])
                c[t]=1.0/np.sum(alpha[:,t])
                alpha[:,t]=alpha[:,t]*c[t]

        else:
            for t in xrange(1,T):
                for i in xrange(N):
                    alpha[i,t] = b[i,o[t]] * np.sum(alpha[:,t-1] * a[:,i])
        
        return alpha, c

    def HMMbwd(self, a, b, o, c):
        # Implements HMM Backward algorithm
    
        N = np.shape(b)[0]
        T = np.shape(o)[0]
    
        beta = np.zeros((N,T))
        # initialise last row with scaling c
        beta[:,T-1] = c[T-1]
    
        for t in xrange(T-2,-1,-1):
            for i in xrange(N):
                beta[i,t] = np.sum(b[:,o[t+1]] * beta[:,t+1] * a[i,:])
            # scale beta by the same value as a
            beta[:,t]=beta[:,t]*c[t]

        return beta

    def HMMViterbi(self, a, b, o, pi):
        # Implements HMM Viterbi algorithm        
        
        N = np.shape(b)[0]
        T = np.shape(o)[0]
    
        path = np.zeros(T)
        delta = np.zeros((N,T))
        phi = np.zeros((N,T))
    
        delta[:,0] = pi * b[:,o[0]]
        phi[:,0] = 0
    
        for t in xrange(1,T):
            for i in xrange(N):
                delta[i,t] = np.max(delta[:,t-1]*a[:,i])*b[i,o[t]]
                phi[i,t] = np.argmax(delta[:,t-1]*a[:,i])
    
        path[T-1] = np.argmax(delta[:,T-1])
        for t in xrange(T-2,-1,-1):
            path[t] = phi[int(path[t+1]),t+1]
    
        return path,delta, phi

 
    def HMMBaumWelch(self, o, N, dirichlet=False, verbose=False, rand_seed=1):
        # Implements HMM Baum-Welch algorithm        
        
        T = np.shape(o)[0]

        M = int(max(o))+1 # now all hist time-series will contain all observation vals, but we have to provide for all

        digamma = np.zeros((N,N,T))

    
        # Initialise A, B and pi randomly, but so that they sum to one
        np.random.seed(rand_seed)
        
        # Initialisation can be done either using dirichlet distribution (all randoms sum to one) 
        # or using approximates uniforms from matrix sizes
        if dirichlet:
            pi = np.ndarray.flatten(np.random.dirichlet(np.ones(N),size=1))
            
            a = np.random.dirichlet(np.ones(N),size=N)
            
            b=np.random.dirichlet(np.ones(M),size=N)
        else:
            
            pi_randomizer = np.ndarray.flatten(np.random.dirichlet(np.ones(N),size=1))/100
            pi=1.0/N*np.ones(N)-pi_randomizer

            a_randomizer = np.random.dirichlet(np.ones(N),size=N)/100
            a=1.0/N*np.ones([N,N])-a_randomizer

            b_randomizer=np.random.dirichlet(np.ones(M),size=N)/100
            b = 1.0/M*np.ones([N,M])-b_randomizer

        
        error = self.tolerance+10
        itter = 0
        while ((error > self.tolerance) & (itter < self.max_iter)):   

            prev_a = a.copy()
            prev_b = b.copy()
    
            # Estimate model parameters
            alpha, c = self.HMMfwd(a, b, o, pi)
            beta = self.HMMbwd(a, b, o, c) 
    
            for t in xrange(T-1):
                for i in xrange(N):
                    for j in xrange(N):
                        digamma[i,j,t] = alpha[i,t]*a[i,j]*b[j,o[t+1]]*beta[j,t+1]
                digamma[:,:,t] /= np.sum(digamma[:,:,t])
    

            for i in xrange(N):
                for j in xrange(N):
                    digamma[i,j,T-1] = alpha[i,T-1]*a[i,j]
            digamma[:,:,T-1] /= np.sum(digamma[:,:,T-1])
    
            # Maximize parameter expectation
            for i in xrange(N):
                pi[i] = np.sum(digamma[i,:,0])
                for j in xrange(N):
                    a[i,j] = np.sum(digamma[i,j,:T-1])/np.sum(digamma[i,:,:T-1])
    	

                for k in xrange(M):
                    filter_vals = (o==k).nonzero()
                    b[i,k] = np.sum(digamma[i,:,filter_vals])/np.sum(digamma[i,:,:])
    
            error = (np.abs(a-prev_a)).max() + (np.abs(b-prev_b)).max() 
            itter += 1            
            
            if verbose:            
                print ("Iteration: ", itter, " error: ", error, "P(O|lambda): ", np.sum(alpha[:,T-1]))
    
        return a, b, pi, alpha
        
def parseStockPrices(from_date, to_date, symbol):
    # Download yahoo_finance package from https://pypi.python.org/pypi/yahoo-finance    
    
    yahoo = Share(symbol)
    hist_prices = yahoo.get_historical(from_date, to_date)
    np_hist_prices = np.empty(shape=[len(hist_prices),len(hist_prices[0])-1])   
    i=0
    for record in hist_prices:
        np_hist_prices[i,0]=float(record['Adj_Close'])
        np_hist_prices[i,1]=float(record['Close'])
        np_hist_prices[i,2]=(dt.datetime.strptime(record['Date'] , '%Y-%m-%d')).toordinal()
        np_hist_prices[i,3]=float(record['High'])
        np_hist_prices[i,4]=float(record['Low'])
        np_hist_prices[i,5]=float(record['Open'])
        #np_hist_prices[i,6]=record['Symbol']
        np_hist_prices[i,6]=long(record['Volume'])
        i+=1
    
    return np_hist_prices       
        
def calculateDailyMoves(hist_prices, holding_period):
    # calculate daily moves as absolute difference between close_(t+1) - close_(t)    

    assert holding_period > 0, "Holding period should be above zero"
    return (hist_prices[:-holding_period,1]-hist_prices[holding_period:,1])


if __name__ == '__main__':

    hmm = HMM()
    
    # parse Yahoo stock price and create time-series in {0,1,2} for down, up, unchanged on daily absolute move
    
    hist_prices = parseStockPrices('2016-08-17', '2016-08-30', 'YHOO')  # yyyy-mm-dd
    assert len(hist_prices)>0, "Houston, we've got a problem"
    hist_moves = calculateDailyMoves(hist_prices,1)
    hist_O=np.array(map(lambda x: 1 if x>0 else (0 if x<0 else 2), hist_moves))
    hist_O = hist_O[::-1] # need to flip to be least receint to most recent ordering
    assert len(hist_prices)>0, "Houston, we've definitely got a problem"
    
    (a, b, pi_est, alpha_est) = hmm.HMMBaumWelch(hist_O, 2, False, True)
    (path, delta, phi)=hmm.HMMViterbi(a, b, hist_O, pi_est)
