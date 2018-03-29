This script generates an error report for parts of speech tagging. 

You can run different classifiers (LR,RNN,HMM) and see how the errors differ. The report runs down the most common misclassifications (say mistaking verbs for nouns, nouns for adj etc) and provides the words the classifier failed to classify correctly. 

I've also broken it down by known errors (misclassifications of words in the train set) and unknown (misclassifications of 'unseen' words, words only in the test set). The unknown errors gives a clue to how well the classifiers are truly learning characteristics of the words pertaining to classification. 

The report also includes the parameters for the classifier (epochs, learning rate, etc). 

You can run each python classifier file, it should run and then it will produce an html file and immediately bring up a new tab/window with the report. You can shut off brining up the tab by setting browser to False in each classifier file and just find the report directory in the directory of the classification scripts. 

![alt text](https://raw.githubusercontent.com/escottgoodwin/partsofspeecherrors/branch/path/to/img.png)
