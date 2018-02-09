import numpy as np

def get_data_error(train_file,test_file,split_sequences=False): ## process train and test files
    ## holder variables
    word2idx = {} ## will hold dict of word and its index {'the':1,'and':2}
    tag2idx = {} ## will hold dict of tag and its index {'NN':1,'VB':2}
    word_idx = 0
    tag_idx = 0
    Xtrain = []
    Ytrain = []
    currentX = []
    currentY = []
    trainwordlist = [] ## list of words in train set ['the','and']
    traintaglist = [] ## list of tags in train set ['NN','VB']
    Xtrain_seq = [] ## for storing words in sequences for use in training HMMs and RNNs which use sequences to learn
    Ytrain_seq = [] ## for storing tags in sequences for use in training HMMs and RNNs which use sequences to learn

    for line in open(train_file): ## open train file
        line = line.strip() ## strip line of white space

        if line: ## iter thru each line of word and tag
            r = line.split()
            word, tag, _ = r ## splits line into word and tag

            trainwordlist.append(word) ## add word to train word list

            traintaglist.append(tag) ## add word to train tag list
            if word not in word2idx: ## adds new words to dict with incremented index
                word2idx[word] = word_idx
                word_idx += 1
            currentX.append(word2idx[word])

            if tag not in tag2idx: ## adds new tags to dict with incremented index
                tag2idx[tag] = tag_idx
                tag_idx += 1
            currentY.append(tag2idx[tag])
        elif split_sequences: ## if train set in sequences
            Xtrain_seq.append(currentX)
            Ytrain_seq.append(currentY)
            currentX = []
            currentY = []
    if not split_sequences: ## for train data not in squences
        Xtrain = currentX
        Ytrain = currentY

    ## process test data same as training set depending on whether data is in sequences
    Xtest = []
    Ytest = []
    Xtest_seq = []
    Ytest_seq = []
    currentX = []
    currentY = []
    testwordlist = []
    testtaglist = []

    for line in open(test_file):
        line = line.strip()
        if line:
            r = line.split()
            word, tag, _ = r

            testwordlist.append(word)

            testtaglist.append(tag)
            if word  in word2idx:
                currentX.append(word2idx[word])
            else:
                currentX.append(word_idx) ## unknown word case
            currentY.append(tag2idx[tag])
        elif split_sequences:
            Xtest_seq.append(currentX)
            Ytest_seq.append(currentY)
            currentX = []
            currentY = []
            Xtest = np.concatenate(Xtest_seq)
            Ytest = np.concatenate(Ytest_seq)
    if not split_sequences:
        Xtest = currentX
        Ytest= currentY


    idx2word = dict(zip(word2idx.values(),word2idx.keys())) ## dict for index and words {1:'and',2:'the'}
    idx2tag = dict(zip(tag2idx.values(),tag2idx.keys())) ## dict for index and tags {1:'NN',2:'VB'}

    # returns X - words, Y -tags in list and sequences -
    return Xtrain, Ytrain, Xtrain_seq, Ytrain_seq, Xtest, Ytest, Xtest_seq, Ytest_seq, word2idx, tag2idx,idx2word,idx2tag,trainwordlist,traintaglist,testwordlist,testtaglist
