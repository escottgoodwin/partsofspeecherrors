import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from datetime import datetime
import re
import pandas as pd


def predicterrors(p,X, Y,fit_params,wordlist,taglist,idx2tag,word2idx,heading,train_time,console_output=True,text_save=False,web_rpt=False):

    predtags = []
    for x in p:
        tag = idx2tag[x]
        predtags.append(tag) ## np model tag predictions ids
         ## np word ids

    wordtags = pd.DataFrame(list(zip(X,wordlist,taglist,predtags)))
    wordtags.columns = ['wordid','word','correct','prediction']

    wrong_predictions = wordtags[wordtags['correct'] != wordtags['prediction']]
    known_errors = wrong_predictions[wrong_predictions['wordid']!= len(word2idx)]
    unknown_errors = wrong_predictions[wrong_predictions['wordid']== len(word2idx)]
    unknown = wordtags[wordtags['wordid']  == 19122]
    known = wordtags[wordtags['wordid']  != 19122]
    unknown_correct = unknown[unknown['correct'] == unknown['prediction']]
    unknown_incorrect = unknown[unknown['correct'] != unknown['prediction']]
    known_correct = known[known['correct'] == known['prediction']]
    known_incorrect = known[known['correct'] != known['prediction']]

    if console_output:
        print(heading + " - "+fit_params)
        print("PERCENT WRONG PREDICTIONS", len(wrong_predictions) / len(Y))
        print("PERCENT UNKNOWN WORDS PREDICTED CORRECTLY", len(unknown_correct) / len(unknown))
        print("PERCENT UNKNOWN WORDS PREDICTED INCORRECTLY", len(unknown_incorrect) / len(unknown))

        def errors(errorsdf,errgrp):
            errgrp = errgrp + ' ERRORS'
            grouped_df = errorsdf.groupby(['correct','prediction'])
            total_error = len(errorsdf) / len(Y)
            errors = []
            for name, group in grouped_df : ## create list of missclass errors and num of occurences ((NN,VB), 188)
                err = name,len(group)
                errors.append(err)
            errorsdf = pd.DataFrame(data=errors)
            errorsdf[2] = errorsdf[1] / errorsdf[1].sum()
            errorsdf[3] = errorsdf[1] / len(Y)
            errorsorted = errorsdf.sort_values(1,ascending=False) ##sort descending (NN/VB,188, VB/NN,120)
            errorsorted.columns = ['TYPE (CORRECT/PREDICTION)','OCCURENCES','% OF ERRORS','% OF ALL PREDICTIONS']
            #errorsorted.plot(x=0,y=1,kind='bar',figsize=(60,60),title=errgrp) ## bar chart of missclassifications by type
            #plt.show()

            print("PERCENT OF ERRORS - KNOWN WORDS:", len(known_errors) / len(errorsdf))
            print("PERCENT OF ERRORS - UNKNOWN WORDS:", len(unknown_errors) / len(errorsdf))
            print(errgrp)
            print(errorsorted)


            grouped_list = list(grouped_df)
            groups = []
            index = len(grouped_list)
            for x in range(index):
                misclass = grouped_list[x][0]
                num = len(grouped_list[x][1])
                percent = num / errorsdf[1].sum()
                percent_all = num / len(Y)
                percent_dec = format(percent_all, '.8f')
                word_set = set(list(grouped_list)[x][1]['word'])
                word_list = []
                for x in word_set:
                    word_list.append(x)
                type_word = misclass,num,percent,word_list,percent_dec
                groups.append(type_word)
            df = pd.DataFrame(data=groups)
            df_sorted = df.sort_values(by=1, ascending=False)

            for index,row in df_sorted.iterrows():
                print(heading, " - ",errgrp," - MISCLASS TYPE (CORRECT/PREDICTION):",row[0])
                print("OCCURENCES:",row[1],"PERCENT OF ERRORS:",row[2],"PERCENT OF ALL PREDICTIONS",row[4])
                print("WORDS IN TYPE:")
                for x in row[3]:
                    print(x)
                print("                                ")

        errors(wrong_predictions,'ALL WORD')
        errors(known_errors,'KNOWN WORD')
        errors(unknown_errors,'UNKNOWN WORD')

    if text_save:
        timedate = str(datetime.now())
        timedate = re.sub('[^0-9]','', timedate)
        write_text = heading.upper()+ " - "+fit_params +" ERROR REPORT - " +timedate+"\r\n"
        write_text += "PERCENT UNKNOWN WORDS CORRECT: " + str(len(unknown_correct) / len(unknown)) + "\r"
        write_text += "PERCENT UNKNOWN WORDS INCORRECT: " + str(len(unknown_incorrect) / len(unknown)) + "\r\n"
        write_text += "PERCENT WRONG PREDICTIONS: " + str(total_error) + "\r\n"
        write_text += "PERCENT KNOWN ERRORS: " + str(len(known_errors) / len(wrong_predictions)) + "\r\n"
        write_text += "PERCENT UNKNOWN ERRORS: " +  str(len(unknown_errors) / len(wrong_predictions)) + "\r\n\r\n"

        def errors(errorsdf,errgrp):
            errgrp = errgrp + ' ERRORS'
            grouped_df = errorsdf.groupby(['correct','prediction'])
            total_error = len(errorsdf) / len(Y)
            errors = []
            for name, group in grouped_df : ## create list of missclass errors and num of occurences ((NN,VB), 188)
                err = name,len(group)
                errors.append(err)
            errorsdf = pd.DataFrame(data=errors)
            errorsdf[2] = errorsdf[1] / errorsdf[1].sum()
            errorsdf[3] = errorsdf[1] / len(Y)
            errorsorted = errorsdf.sort_values(1,ascending=False) ##sort descending (NN/VB,188, VB/NN,120)
            errorsorted.columns = ['(CORRECT/PREDICTION)','OCCURENCES','% OF ERRORS','% OF ALL PREDICTIONS']
            #errorsorted.plot(x=0,y=1,kind='bar',figsize=(60,60),title=errgrp) ## bar chart of missclassifications by type
            #plt.show()

            grp_text = errgrp + "\r\n"
            grp_text += errorsorted.to_string(index=False) +  "\r\n\r\n"


            grouped_list = list(grouped_df)
            groups = []
            index = len(grouped_list)
            for x in range(index):
                misclass = grouped_list[x][0]
                num = len(grouped_list[x][1])
                percent = num / errorsdf[1].sum()
                percent_all = num / len(Y)
                percent_dec = format(percent_all, '.8f')
                word_set = set(list(grouped_list)[x][1]['word'])
                word_list = []
                for x in word_set:
                    word_list.append(x)
                type_word = misclass,num,percent,word_list,percent_dec
                groups.append(type_word)
            df = pd.DataFrame(data=groups)
            df_sorted = df.sort_values(by=1, ascending=False)

            for index,row in df_sorted.iterrows():
                grp_text += heading.upper() + " - " + errgrp + " - MISCLASS TYPE (CORRECT/PREDICTION): " + str(row[0]) + "\r\n"
                grp_text += "OCCURENCES: " + str(row[1]) + " PERCENT OF ERRORS: " + str(row[2]) + " PERCENT OF ALL PREDICTIONS " + str(row[4]) + "\r\n\r\n"
                grp_text +="WORDS IN TYPE: " + "\r\n"
                for x in row[3]:
                    grp_text += x + "\r\n"
                grp_text += "                                " + "\r\n"
            return grp_text

        divider = "\r\n" + "*" * 90 + "\r\n\r\n"
        all_word = errors(wrong_predictions,'ALL WORD')
        known = errors(known_errors,'KNOWN WORD')
        unknown = errors(unknown_errors,'UNKNOWN WORD')
        write_text +=  all_word + divider + known + divider + unknown

        f= open("prediction_errors-"+timedate+".txt","w+")
        f.write(write_text)
        f.close()


    if web_rpt:
        timedate = str(datetime.now())
        timedate = re.sub('[^0-9]','', timedate)



        def errors(errorsdf,errgrp):
            errgrp = errgrp + ' ERRORS'
            grouped_df = errorsdf.groupby(['correct','prediction'])

            errors = []
            for name, group in grouped_df : ## create list of missclass errors and num of occurences ((NN,VB), 188)
                err = name,len(group)
                errors.append(err)
            errorsdf = pd.DataFrame(data=errors)
            errorsdf[2] = errorsdf[1] / errorsdf[1].sum()
            errorsdf[3] = errorsdf[1] / len(Y)
            errorsorted = errorsdf.sort_values(1,ascending=False) ##sort descending (NN/VB,188, VB/NN,120)
            errorsorted.columns = ['(CORRECT/PREDICTION)','OCCURENCES','% OF ERRORS','% OF ALL PREDICTIONS']
            errorsorted.plot(x=0,y=1,kind='bar',figsize=(20,20),title=errgrp+" - " + timedate) ## bar chart of missclassifications by type
            figname = errgrp.upper()+timedate
            figname=figname.replace(" ", "")
            plt.savefig(figname+'.jpg')
            plt.close()

            grp_text = "<p>" + errgrp + "<br>"
            grp_text += errorsorted.to_html(index=False) + "</p>"


            grouped_list = list(grouped_df)
            groups = []
            index = len(grouped_list)
            for x in range(index):
                misclass = grouped_list[x][0]
                num = len(grouped_list[x][1])
                percent = num / errorsdf[1].sum()
                percent_all = num / len(Y)
                percent_dec = format(percent_all, '.8f')
                word_set = set(list(grouped_list)[x][1]['word'])
                word_list = []
                for x in word_set:
                    word_list.append(x)
                type_word = misclass,num,percent,word_list,percent_dec
                groups.append(type_word)
            df = pd.DataFrame(data=groups)
            df_sorted = df.sort_values(by=1, ascending=False)

            for index,row in df_sorted.iterrows():
                grp_text += "<p>" + heading.upper() + " - " + errgrp + " - MISCLASS TYPE (CORRECT/PREDICTION): " + str(row[0]) + "<br>"
                grp_text += "OCCURENCES: " + str(row[1]) + " PERCENT OF ERRORS: " + str(row[2]) + " PERCENT OF ALL PREDICTIONS " + str(row[4]) + "<br>"
                grp_text +="WORDS IN TYPE: " + "</p><p>"
                for x in row[3]:
                    grp_text += x + "&emsp;"
                grp_text += "</p>"
            return grp_text

        html_head = "<p>" + heading.upper()+" - "+fit_params +" ERROR REPORT - " +timedate + "<br>"
        html_head += "TRAIN TIME: " + train_time + "<br>"
        html_head += "PERCENT WRONG PREDICTIONS: " + str(len(wrong_predictions) / len(Y)) + "<br>"
        html_head += "PERCENT OF KNOWN WORDS PREDICTED CORRECTLY: " + str(len(known_correct) / len(known)) + "<br>"
        html_head += "PERCENT KNOWN WORDS PREDICTED INCORRECTLY: " + str(len(known_incorrect) / len(known)) + "<br>"
        html_head += "PERCENT OF UNKNOWN WORDS PREDICTED CORRECTLY: " + str(len(unknown_correct) / len(unknown)) + "<br>"
        html_head += "PERCENT UNKNOWN WORDS PREDICTED INCORRECTLY: " + str(len(unknown_incorrect) / len(unknown)) + "<br>"
        html_head += "PERCENT OF ERRORS - KNOWN WORDS: " + str(len(known_errors) / len(wrong_predictions)) + "<br>"
        html_head += "PERCENT OF ERRORS - UNKNOWN WORDS: " +  str(len(unknown_errors) / len(wrong_predictions)) + "</p>"
        all_word_grp = 'ALL WORD'
        known_word_grp = 'KNOWN WORD'
        unknown_word_grp = 'UNKNOWN WORD'
        all_word = errors(wrong_predictions,all_word_grp)
        known = errors(known_errors,known_word_grp)
        unknown = errors(unknown_errors,unknown_word_grp)
        tit = heading.upper() + " - " + fit_params + "ERRORS " + " - " + timedate
        figname1 = all_word_grp.upper() + "ERRORS " + timedate
        figname1=figname1.replace(" ", "")
        figname2 = known_word_grp.upper() + "ERRORS " + timedate
        figname2=figname2.replace(" ", "")
        figname3 = unknown_word_grp.upper() + "ERRORS " + timedate
        figname3=figname3.replace(" ", "")

        html = '''<!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>
        ''' + tit + '''</title>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.4/jquery.min.js"></script>
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
        <style type="text/css">
        .bs-example{
        margin: 20px;
        padding-top:50px;
        }
        .nav {
        background-color: #FFFFFF;
        }
        img {
        max-width: 100%;
        max-height: 100%;
        }

        </style>
        </head>
        <body data-spy="scroll" data-target=".nav" data-offset="50">

        <div class="bs-example">
        ''' + html_head + '''<ul class="nav nav-tabs navbar-fixed-top">
            <li class="active"><a data-toggle="tab" href="#all_errors">ALL ERRORS</a></li>
            <li><a data-toggle="tab" href="#known_errors">KNOWN ERRORS</a></li>
            <li><a data-toggle="tab" href="#unknown_errors">UNKNOWN ERRORS</a></li>

        </ul>


        <div class="tab-content">
            <div id="all_errors" class="tab-pane fade in active">
             <a data-toggle="tab" href="#chart_all_errors">Bar Graph</a><br>
        ''' + all_word + ''' </div>
            <div id="known_errors" class="tab-pane fade">
            <a data-toggle="tab" href="#chart_known_errors">Bar Graph</a><br>
        ''' + known + '''</div>
            <div id="unknown_errors" class="tab-pane fade">
            <a data-toggle="tab" href="#chart_unknown_errors">Bar Graph</a><br>
        ''' + unknown + '''</div>
            <div id="chart_unknown_errors" class="tab-pane fade">
            <a data-toggle="tab" href="#unknown_errors">UNKNOWN ERRORS</a><br>
            <img src="
        ''' + figname3 + '''.jpg" alt="Unknown Errors Bar Graph" >
            </div>
            <div id="chart_all_errors" class="tab-pane fade">
            <a data-toggle="tab" href="#all_errors">ALL ERRORS</a><br>
            <img src="
        ''' + figname1 + '''.jpg" alt="All Errors Bar Graph" >
            </div>
            <div id="chart_known_errors" class="tab-pane fade">
            <a data-toggle="tab" href="#known_errors">KNOWN ERRORS</a><br>
            <img src="
        ''' + figname2 + '''.jpg" alt="Known Errors Bar Graph" >
            </div>

        </div>
        </div>
        </body></html>
        '''


        import webbrowser
        f= open("prediction_errors-"+timedate+".html","w+")
        f.write(html)
        f.close()
        url = "prediction_errors-"+timedate+".html"
        chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
        brave_path = 'open -a Brave.app %s'
        webbrowser.get(chrome_path).open(url,new=1)


def get_data_error(train_file,test_file,split_sequences=False):
    ## holder variables
    word2idx = {}
    tag2idx = {}
    word_idx = 0
    tag_idx = 0
    Xtrain = []
    Ytrain = []
    currentX = []
    currentY = []
    trainwordlist = []
    traintaglist = []
    Xtrain_seq = []
    Ytrain_seq = []
    for line in open(train_file):
        line = line.strip()

        if line:
            r = line.split()
            word, tag, _ = r

            trainwordlist.append(word)

            traintaglist.append(tag)
            if word not in word2idx:
                word2idx[word] = word_idx
                word_idx += 1
            currentX.append(word2idx[word])

            if tag not in tag2idx:
                tag2idx[tag] = tag_idx
                tag_idx += 1
            currentY.append(tag2idx[tag])
        elif split_sequences:
            Xtrain_seq.append(currentX)
            Ytrain_seq.append(currentY)
            currentX = []
            currentY = []
    if not split_sequences:
        Xtrain = currentX
        Ytrain = currentY

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


    idx2word = dict(zip(word2idx.values(),word2idx.keys()))
    idx2tag = dict(zip(tag2idx.values(),tag2idx.keys()))


    return Xtrain, Ytrain, Xtrain_seq, Ytrain_seq, Xtest, Ytest,Xtest_seq, Ytest_seq, word2idx, tag2idx,idx2word,idx2tag,trainwordlist,traintaglist,testwordlist,testtaglist

def main():
    train_file = "train.txt"
    test_file = "test.txt"
    ## process train and test entries and output train and test sets of X(word) Y(tags)
    Xtrain, Ytrain, Xtest, Ytest, word2idx, tag2idx,idx2word,idx2tag,trainwordlist,traintaglist,testwordlist,testtaglist = get_data(train_file,test_file)



    #predLR = predictions(Xtest, Ytest)

    #predicterrors(predLR,Xtest, Ytest,testwordlist,testtaglist,idx2tag,word2idx)





if __name__ == '__main__':
    main()
