import numpy as np
import theano
import theano.tensor as T
from sklearn.utils import shuffle
from predict_errors2 import predicterrors
from get_data import get_data_error
from datetime import datetime

class LogisticRegression: ## create Logistic Regression class for training
    def __init__(self):
        pass

    def fit(self, X, Y, V=None, K=None, lr=1e-1, mu=0.99, batch_sz=100, epochs=6):
        self.mu = mu
        self.lr = lr
        self.epochs = epochs
        if V is None:
            V = len(set(X)) ## set of words - vocabulary
        if K is None:
            K = len(set(Y)) ## set of tags
        N = len(X) ## num of entries of word-tags

        W = np.random.randn(V, K) / np.sqrt(V + K) ## generate random weights matrix of V (vocab) by K (unique tags) - scale by sqrt of V + K to get 'percentages'
        b = np.zeros(K) ## zeros for bias
        self.W = theano.shared(W) ## theano share for weights
        self.b = theano.shared(b) ## theano share for bias
        self.params = [self.W, self.b] ## put W and b in list for updating

        thX = T.ivector('X')
        thY = T.ivector('Y')

        py_x = T.nnet.softmax(self.W[thX] + self.b) ## inputs to hidden - X dotted with W weighs plus bias then softmax
        prediction = T.argmax(py_x, axis=1) ## predict tag by taking tag (Y) with highest value

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY])) ## calc costs by finding mean of how far py_x was from thY by taking log
        grads = T.grad(cost, self.params) ## calc gradients from costs
        dparams = [theano.shared(p.get_value()*0) for p in self.params] ## dparams for momentum
        self.cost_predict_op = theano.function( ## run on X,Y inputs to get costs and predictions
            inputs=[thX, thY],
            outputs=[cost, prediction],
            allow_input_downcast=True,
        )

        updates = [ ## update weights with gradients calculated from costs
            (p, p + mu*dp - lr*g) for p, dp, g in zip(self.params, dparams, grads)  # params(W & b) plus momentum minus gradients scaled by learning rate
        ] + [
            (dp, mu*dp - lr*g) for dp, g in zip(dparams, grads) ## new dparams for momentum
        ]
        train_op = theano.function( ## training - run with inputs to get costs, predictions and updates
            inputs=[thX, thY],
            outputs=[cost, prediction],
            updates=updates,
            allow_input_downcast=True
        )

        costs = [] ## store costs for each epoch
        n_batches = N / batch_sz ## get batch size by dividing num of entries by chosen size (100 default)
        for i in range(epochs): ## run training round for chosen number of epochs
            X, Y = shuffle(X, Y)
            print("epoch:", i)
            for j in range(int(n_batches)): ## iter through batches for training - default train on 100 size batches
                Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)] ## run first on X[:100], then X[100:200] etc
                Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz)]

                c, p = train_op(Xbatch, Ybatch) ## run tranining op on batch - get predictions and costs
                costs.append(c) ## track costs per batch
                if j % 200 == 0: ## every 200 batch iters print results
                    print("i:", i, "j:", j, "n_batches:", n_batches, "cost:", c, "error:", np.mean(p != Ybatch))

    def predictions(self, X, Y): ## accuracy score
        _, p = self.cost_predict_op(X,Y) ## generates tag predictions by taking argmax

        return p

    def score(self,p,Y): ## how many predicted tags match real tags
        return np.mean(p == Y)

def main():

    t0 = datetime.now()
    print(t0)
    train_file = "train.txt"
    test_file = "test.txt"
    ## process train and test entries and output train and test sets of X(word) Y(tags)
    Xtrain, Ytrain, Xtrain_seq, Ytrain_seq, Xtest, Ytest,Xtest_seq, Ytest_seq, word2idx, tag2idx,idx2word,idx2tag,trainwordlist,traintaglist,testwordlist,testtaglist = get_data_error(train_file,test_file)

    # convert to numpy arrays
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)

    # convert Xtrain to indicator matrix
    N = len(Xtrain)
    V = len(word2idx) + 1
    print("vocabulary size:", V)

    # train and score
    model = LogisticRegression()
    model.fit(Xtrain, Ytrain, V=V,epochs=5)

    print("mu: ", model.mu," learing rate: ", model.lr," epochs: ", model.epochs)

    train_time = str(datetime.now() - t0)
    print("training complete: " + train_time)

    predLR = model.predictions(Xtest, Ytest) ## generate model predictions from test set
    ## create string for model's parameters
    fit_params = "MU: " + str(model.mu) + " LEARNING RATE: " + str(model.lr) + " EPOCHS: " + str(model.epochs)

    heading = "LOGISTIC REGRESSION"

    ## create report of misclassifications of POS by the model
    predicterrors(predLR,Xtest,Ytest,fit_params,testwordlist,testtaglist,idx2tag,word2idx,heading,train_time,browser=True)

if __name__ == '__main__':
    main()
