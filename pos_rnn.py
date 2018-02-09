
import numpy as np
import theano
import theano.tensor as T
import os
import sys
sys.path.append(os.path.abspath('..'))
from gru import GRU
from sklearn.utils import shuffle

from datetime import datetime
from predict_errors1 import predicterrors
from get_data import get_data_error
from datetime import datetime

def init_weight(Mi, Mo):
    return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)

class RNN:
    def __init__(self, D, hidden_layer_sizes, V):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.D = D ## dimensions
        self.V = V ## vocab

    def fit(self, X, Y, learning_rate=1e-4, mu=0.99, epochs=10, show_fig=False, activation=T.nnet.relu, RecurrentUnit=GRU, normalize=False):
        D = self.D
        V = self.V
        N = len(X)

        We = init_weight(V, D) ## words as rows dimensions as cols
        self.hidden_layers = []
        Mi = D
        for Mo in self.hidden_layer_sizes:
            ru = RecurrentUnit(Mi, Mo, activation)
            self.hidden_layers.append(ru)
            Mi = Mo

        Wo = init_weight(Mi, V)
        bo = np.zeros(V)

        self.We = theano.shared(We)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.Wo, self.bo]
        for ru in self.hidden_layers:
            self.params += ru.params

        thX = T.ivector('X')
        thY = T.ivector('Y')

        Z = self.We[thX]
        for ru in self.hidden_layers:
            Z = ru.output(Z)
        py_x = T.nnet.softmax(Z.dot(self.Wo) + self.bo)

        prediction = T.argmax(py_x, axis=1)

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        dWe = theano.shared(self.We.get_value()*0)
        gWe = T.grad(cost, self.We)
        dWe_update = mu*dWe - learning_rate*gWe
        We_update = self.We + dWe_update
        if normalize:
            We_update /= We_update.norm(2)

        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ] + [
            (self.We, We_update), (dWe, dWe_update)
        ]

        self.cost_predict_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            allow_input_downcast=True,
        )

        self.train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            updates=updates
        )

        costs = []
        sequence_indexes = range(N)
        n_total = sum(len(y) for y in Y)
        for i in range(epochs):
            t0 = datetime.now()
            sequence_indexes = shuffle(sequence_indexes)
            n_correct = 0
            cost = 0
            it = 0
            for j in sequence_indexes:
                c, p = self.train_op(X[j], Y[j])
                cost += c
                n_correct += np.sum(p == Y[j])
                it += 1
                if it % 200 == 0:
                    sys.stdout.write("j/N: %d/%d correct rate so far: %f, cost so far: %f\r" % (it, N, float(n_correct)/n_total, cost))
                    sys.stdout.flush()
            print("Epoch:", i, "cost:", cost, "correct rate:", (float(n_correct)/n_total), "time for epoch:", (datetime.now() - t0))
            costs.append(cost)

        return learning_rate, mu, epochs, activation, RecurrentUnit


    def score(self, X, Y):
        n_total = sum(len(y) for y in Y)
        n_correct = 0
        for x, y in zip(X, Y):
            _, p = self.cost_predict_op(x, y)
            n_correct += np.sum(p == y)
        return float(n_correct) / n_total

    def predictions(self, X, Y):
        P = []
        for x, y in zip(X, Y):
            _, p = self.cost_predict_op(x, y)
            P.append(p)
        Y = np.concatenate(Y)
        P = np.concatenate(P)
        return P

def main():

    train_file = "train.txt"
    test_file = "test.txt"
    Xtrain, Ytrain, Xtrain_seq, Ytrain_seq, Xtest, Ytest,Xtest_seq, Ytest_seq, word2idx, tag2idx,idx2word,idx2tag,trainwordlist,traintaglist,testwordlist,testtaglist = get_data_error(train_file,test_file,split_sequences=True)

    t0 = datetime.now()
    print(t0)

    V = len(word2idx) + 1
    rnn = RNN(10, [10], V)
    learning_rate, mu, epochs, activation, RecurrentUnit = rnn.fit(Xtrain_seq, Ytrain_seq)
    rnn_preds = rnn.predictions(Xtest_seq, Ytest_seq)
    train_time = str(datetime.now() - t0)
    print("training complete: " + train_time)

    heading = "RNN"
    fit_params = "MU: " + str(mu) + " LEARNING RATE: " + str(learning_rate) + " EPOCHS: " + str(epochs) + " Recurrent Unit: " + str(RecurrentUnit)

    ## create report of misclassifications of POS by the model
    predicterrors(rnn_preds,Xtest,Ytest,fit_params,testwordlist,testtaglist,idx2tag,word2idx,heading,train_time,console_output=False,web_rpt=True)

if __name__ == '__main__':
    main()
