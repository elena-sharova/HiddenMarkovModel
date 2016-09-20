# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:43:25 2016

@author: Elena
"""

import HMM as discreteHMM
import numpy as np

hmm = discreteHMM.HMM()

test_start_date = '2007-06-01'
test_end_date = '2016-09-02'
all_data = hist_prices = discreteHMM.parseStockPrices(test_start_date, test_end_date, 'YHOO')
assert len(all_data)>0, "Houston, we've got a problem"


num_correct=0.0
test_window = 6
N=len(all_data)
num_tests=N/test_window
for n in xrange(1,N-test_window,test_window):
    train_data = all_data[-n:-n-test_window:-1,:]
    hist_moves = discreteHMM.calculateDailyMoves(train_data,1)
    hist_O=np.array(map(lambda x: 1 if x>0 else (0 if x<0 else 2), hist_moves))
    hist_O=hist_O[::-1]
    (a, b, pi_est, alpha_est) = hmm.HMMBaumWelch(hist_O, 2, False, False)
    (path, delta, phi)=hmm.HMMViterbi(a, b, hist_O, pi_est)
    prediction_state=np.argmax(a[path[-1],:])
    prediction = np.argmax(b[prediction_state,:])
    if ((all_data[-n-test_window-1,1]-all_data[-n-test_window,1])>0 and prediction==1) or ((all_data[-n-test_window-1,1]-all_data[-n-test_window,1])<0 and prediction==0) or ((all_data[-n-test_window-1,1]-all_data[-n-test_window,1])==0 and prediction==2):
        num_correct+=1.0
print num_correct/num_tests