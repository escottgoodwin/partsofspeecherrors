import matplotlib
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re
import os
import pandas as pd

def predicterrors(p,X, Y,fit_params,wordlist,taglist,idx2tag,word2idx,heading,train_time,browser=False):

    ## np model tag predictions ids and np word ids
    predtags = [idx2tag[x] for x in p]

    wordtags = pd.DataFrame(list(zip(X,wordlist,taglist,predtags)))
    wordtags.columns = ['wordid','word','correct','prediction']
    alltags = wordtags.groupby(['correct'])
    alltagscounts =[]
    for name, group in alltags: ## create list of missclassification errors and num of occurences ((NN,VB), 188)
        tag = name,len(group)
        alltagscounts.append(tag)
    alltagsdf = pd.DataFrame(data=alltagscounts)

    wrong_predictions = wordtags[wordtags['correct'] != wordtags['prediction']]
    known_errors = wrong_predictions[wrong_predictions['wordid']!= len(word2idx)]
    unknown_errors = wrong_predictions[wrong_predictions['wordid']== len(word2idx)]
    unknown = wordtags[wordtags['wordid']  == 19122]
    known = wordtags[wordtags['wordid']  != 19122]
    unknown_correct = unknown[unknown['correct'] == unknown['prediction']]
    unknown_incorrect = unknown[unknown['correct'] != unknown['prediction']]
    known_correct = known[known['correct'] == known['prediction']]
    known_incorrect = known[known['correct'] != known['prediction']]

    wrongtags = wrong_predictions.groupby(['correct'])
    wrongtagcounts = []
    for name, group in wrongtags: ## create list of missclass errors and num of occurences ((NN,VB), 188)
        tag = name,len(group)
        wrongtagcounts.append(tag)

    wrongtagsdf = pd.DataFrame(data=wrongtagcounts)
    tagcountmerge = wrongtagsdf.merge(alltagsdf, left_on=[0], right_on=[0])
    tagcountmerge['percent'] = tagcountmerge['1_x'] / tagcountmerge['1_y']
    tagcountsort = tagcountmerge.sort_values(by=['percent'],ascending=False)
    tagcountsorttwocol = tagcountsort[[0,'1_x','1_y','percent']]
    tagcountsorttwocol.columns = ['word','incorrect pred','all','percent']
    tagcounthtml = tagcountsorttwocol.to_html(index=False)

    timedate = str(datetime.now())
    timedate = re.sub('[^0-9]','', timedate)
    dir_name = "prediction_errors-"+timedate
    os.makedirs(dir_name)

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
        plt.savefig(dir_name + "/" +figname+'.jpg')
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
            word_list = [x for x in word_set]
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
        <li><a data-toggle="tab" href="#all_types">ERROR PERCENT BY POS TYPE</a></li>

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
        <div id="all_types" class="tab-pane fade in"><h2>Percent Error by POS Type</h2>
    ''' + tagcounthtml + ''' </div>
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

    file_name = dir_name + "/" + "prediction_errors-"+timedate+".html"
    f= open(file_name,"w+")
    f.write(html)
    f.close()

    if browser:
        import webbrowser
        chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
        webbrowser.get(chrome_path).open(file_name,new=1)

def main():
    train_file = "train.txt"
    test_file = "test.txt"
    ## process train and test entries and output train and test sets of X(word) Y(tags)


if __name__ == '__main__':
    main()
