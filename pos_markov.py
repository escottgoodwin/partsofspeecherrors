

import os
import sys
sys.path.append(os.path.abspath('..'))
from hmmd_scaled import HMM ## import hidden markov model
import numpy as np
from sklearn.utils import shuffle
from datetime import datetime
from predict_errors2 import predicterrors
from get_data import get_data_error


def accuracy(T,Y): ## compute accuracy of prediction
    n_correct = 0
    n_total = 0
    for t,y in zip(T,Y): ## iter over zipped predicted tags and true tags
        n_correct += np.sum(t==y) ## append sum for where pred tag matches true tags
        n_total += len(y) ## count all true tags
    return float(n_correct) / n_total ## % correct predictions

def predictions(T,Y): ## compute accuracy of prediction
    p = []
    for t,y in zip(T,Y): ## iter over zipped predicted tags and true tags
        p.append(t)
    p = np.concatenate(p)
    return p ## % correct predictions

def output_predictions():
    t0 = datetime.now()
    print(t0)
    train_file = "train.txt"
    test_file = "test.txt"
    smoothing=10e-2
    ## process train and test entries and output train and test sets of X(word) Y(tags)
    Xtrain, Ytrain, Xtrain_seq, Ytrain_seq, Xtest, Ytest,Xtest_seq, Ytest_seq, word2idx, tag2idx,idx2word,idx2tag,trainwordlist,traintaglist,testwordlist,testtaglist = get_data_error(train_file,test_file,split_sequences=True)
    V = len(word2idx) + 1 ## vocab (col) = num of words in index
    M = max(max(y) for y in Ytrain_seq) + 1 ## num of hidden states = num of tags
    A = np.ones((M,M))*smoothing ## matrix of 1's MxM - hidden states (y's)
    pi = np.zeros(M) ## pi
    for y in Ytrain_seq: ## iter over y train
        pi[y[0]] += 1 ## start state - add 1 to pi[index[0]] - pi[1[0]] = 1
        for i in range(len(y)-1): ## iter over y
            A[y[i], y[i+1]] += 1 ## transitions - add 1s to every A[y[index]] and y[index+1]

    A /= A.sum(axis=1,keepdims=True) ## normalize distribution - create % (add to 1)
    pi /= pi.sum() ## normalize pi - create %

    B = np.ones((M,V))*smoothing ## MxV matrix of 1's with smoothing
    for x,y in zip(Xtrain_seq,Ytrain_seq): ## iter over X/Y train set
        for xi,yi in zip(x,y): ## iter over zipped x/y pairs
            B[yi, xi] += 1 ## B[state, observation] - state also target
    B /= B.sum(axis=1,keepdims=True) ## normalize B - %

    hmm = HMM(M) ## initialize hidden markov instance
    hmm.pi = pi ## assign hmm obj attr to pi (zeros num of tags)
    hmm.A = A  ## assign A attr of hmm obj (MxM matrix)
    hmm.B = B ## assign B attr of hmm obj (MxV matrix)

    Ptrain = [] ## list to store  predictions for train set
    for x in Xtrain_seq:
        p = hmm.get_state_sequence(x) ## get state for word index
        Ptrain.append(p)

    Ptest = [] ## list to store predictions for test set
    for x in Xtest_seq:
        p = hmm.get_state_sequence(x) ## get state for word index
        Ptest.append(p)
    ## print accurracy - provide real tags and predicted tags to accuracy to see % matches
    ## print f1 score - provide real tags and predicted tags to f1 score to see ratio false pos/false neg
    #print("train f1:", total_f1_score(Ytrain_seq,Ptrain))

    markov_preds = np.concatenate(Ptest)
    fit_params = "Hidden: " + str(hmm.M) + " Smoothing: " + str(smoothing)
# convert to numpy arrays
    model_name = "Hidden Markov"

    train_time = str(datetime.now() - t0)

    return markov_preds,Xtest,Ytest,testwordlist,testtaglist,idx2tag,word2idx,model_name,fit_params,train_time


def total_f1_score(T,Y): ## calc F1 false neg/false pas
    T = np.concatenate(T)
    Y = np.concatenate(Y)
    return f1_score(T,Y,average=None).mean() ## calc mean of f1_score

def main(smoothing=10e-2): ## main function
    ## get data for train/test sets and word index
    t0 = datetime.now()
    print(t0)
    train_file = "train.txt"
    test_file = "test.txt"
    Xtrain, Ytrain, Xtrain_seq, Ytrain_seq, Xtest, Ytest,Xtest_seq, Ytest_seq, word2idx, tag2idx,idx2word,idx2tag,trainwordlist,traintaglist,testwordlist,testtaglist = get_data_error(train_file,test_file,split_sequences=True)

    V = len(word2idx) + 1 ## vocab (col) = num of words in index
    M = max(max(y) for y in Ytrain_seq) + 1 ## num of hidden states = num of tags
    A = np.ones((M,M))*smoothing ## matrix of 1's MxM - hidden states (y's)
    pi = np.zeros(M) ## pi
    for y in Ytrain_seq: ## iter over y train
        pi[y[0]] += 1 ## start state - add 1 to pi[index[0]] - pi[1[0]] = 1
        for i in range(len(y)-1): ## iter over y
            A[y[i], y[i+1]] += 1 ## transitions - add 1s to every A[y[index]] and y[index+1]

    A /= A.sum(axis=1,keepdims=True) ## normalize distribution - create % (add to 1)
    pi /= pi.sum() ## normalize pi - create %

    B = np.ones((M,V))*smoothing ## MxV matrix of 1's with smoothing
    for x,y in zip(Xtrain_seq,Ytrain_seq): ## iter over X/Y train set
        for xi,yi in zip(x,y): ## iter over zipped x/y pairs
            B[yi, xi] += 1 ## B[state, observation] - state also target
    B /= B.sum(axis=1,keepdims=True) ## normalize B - %

    hmm = HMM(M) ## initialize hidden markov instance
    hmm.pi = pi ## assign hmm obj attr to pi (zeros num of tags)
    hmm.A = A  ## assign A attr of hmm obj (MxM matrix)
    hmm.B = B ## assign B attr of hmm obj (MxV matrix)

    Ptrain = [] ## list to store  predictions for train set
    for x in Xtrain_seq:
        p = hmm.get_state_sequence(x) ## get state for word index
        Ptrain.append(p)

    Ptest = [] ## list to store predictions for test set
    for x in Xtest_seq:
        p = hmm.get_state_sequence(x) ## get state for word index
        Ptest.append(p)
    ## print accurracy - provide real tags and predicted tags to accuracy to see % matches
    ## print f1 score - provide real tags and predicted tags to f1 score to see ratio false pos/false neg
    heading = "Hidden Markov"
    #print("train accuracy:", accuracy(Ytrain_seq,Ptrain))
    test_acc = "test accuracy:", accuracy(Ytest_seq,Ptest)
    #print("train f1:", total_f1_score(Ytrain_seq,Ptrain))


    markov_preds = np.concatenate(Ptest)
    fit_params = "Hidden: " + str(hmm.M) + " Smoothing: " + str(smoothing)

    train_time = str(datetime.now() - t0)
    print(train_time)
    predicterrors(markov_preds,Xtest,Ytest,fit_params,testwordlist,testtaglist,idx2tag,word2idx,heading,train_time,browser=True)

if __name__ == '__main__':
    main()
