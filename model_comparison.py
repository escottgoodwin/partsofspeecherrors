import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from predict_errors import predicterrors,get_data_error
from datetime import datetime
import pandas as pd
import re
from pos_baseline import output_predictions as LR_output_predictions
from pos_markov import output_predictions as markov_output_predictions

def comparison(pred1,pred2,X,Y,wordlist,taglist,idx2tag,model_name1,fit_params1,model_name2,fit_params2,train_time1,train_time2):
    heading="Comparison"
    predtags1 = []
    for x in pred1:
        tag = idx2tag[x]
        predtags1.append(tag) ## np model tag predictions ids
         ## np word ids
    predtags2 = []
    for x in pred2:
        tag = idx2tag[x]
        predtags2.append(tag)

    df = pd.DataFrame(list(zip(X,wordlist,taglist,predtags1,predtags2)))
    df.columns = ['wordid','word','correct','prediction1','prediction2']
    errs1 = df[df['correct']!=df['prediction1']]
    errs2 = df[df['correct']!=df['prediction2']]
    errdif = errs2[errs2['prediction1']!=errs2['prediction2']]
    errsimp = errs1[errs1['correct']==errs1['prediction2']]
    errsworse = errs2[errs2['correct']==errs2['prediction1']]
    errsnoimp = errs1[errs1['correct']!=errs1['prediction2']]
    err1total = len(errs1)
    err2total = len(errs2)
    errsimptotal = len(errsimp)
    errsworsetotal = len(errsworse)
    errrate1 = len(errs1)/len(df)
    errrate2 = len(errs2)/len(df)

    dfknown = df[df['wordid'] != 19122]
    knownerrs1 = dfknown[dfknown['correct']!=dfknown['prediction1']]
    knownerrs2 = dfknown[dfknown['correct']!=dfknown['prediction2']]
    knownerrdif = knownerrs2[knownerrs2['prediction1']!=knownerrs2['prediction2']]
    knownerrsimp = knownerrs1[knownerrs1['correct']==knownerrs1['prediction2']]
    knownerrsworse = knownerrs2[knownerrs2['correct']==knownerrs2['prediction1']]
    knownerrsnoimp = knownerrs1[knownerrs1['correct']!=knownerrs1['prediction2']]
    knownerr1total = len(knownerrs1)
    knownerr2total = len(knownerrs2)
    knownerrsimptotal = len(knownerrsimp)
    knownerrsworsetotal = len(knownerrsworse)
    knownerrrate1 = len(knownerrs1)/len(dfknown )
    knownerrrate2 = len(knownerrs2)/len(dfknown )

    dfunknown = df[df['wordid'] == 19122]
    unknownerrs1 = dfunknown[dfunknown['correct']!=dfunknown['prediction1']]
    unknownerrs2 = dfunknown[dfunknown['correct']!=dfunknown['prediction2']]
    unknownerrdif = unknownerrs2[unknownerrs2['prediction1']!=unknownerrs2['prediction2']]
    unknownerrsimp = unknownerrs1[unknownerrs1['correct']==unknownerrs1['prediction2']]
    unknownerrsworse = unknownerrs2[unknownerrs2['correct']==unknownerrs2['prediction1']]
    unknownerrsnoimp = unknownerrs1[unknownerrs1['correct']!=unknownerrs1['prediction2']]
    unknownerr1total = len(unknownerrs1)
    unknownerr2total = len(unknownerrs2)
    unknownerrsimptotal = len(unknownerrsimp)
    unknownerrsworsetotal = len(unknownerrsworse)
    unknownerrrate1 = len(unknownerrs1)/len(dfunknown)
    unknownerrrate2 = len(unknownerrs2)/len(dfunknown)


    def errors(errorsdf,errgrp,exp,model_col):
        errgrp = errgrp
        grouped_df = errorsdf.groupby(['correct',model_col])

        errors = []
        for name, group in grouped_df : ## create list of missclass errors and num of occurences ((NN,VB), 188)
            err = name,len(group)
            errors.append(err)
        errorsdf = pd.DataFrame(data=errors)
        errorsdf[2] = errorsdf[1] / errorsdf[1].sum()
        errorsdf[3] = errorsdf[1] / len(X)
        errorsorted = errorsdf.sort_values(1,ascending=False) ##sort descending (NN/VB,188, VB/NN,120)
        errorsorted.columns = ['(CORRECT/PREDICTION)','OCCURENCES','% OF IMPROVEMENT','% OF ALL PREDICTIONS']
        errorsorted.plot(x=0,y=1,kind='bar',figsize=(20,20),title=errgrp+" - " + timedate) ## bar chart of missclassifications by type
        figname = errgrp.upper()+timedate
        figname=figname.replace(" ", "")
        plt.savefig(figname+'.jpg')
        plt.close()

        grp_text = "<p>" + errgrp + exp + "<br>"
        grp_text += errorsorted.to_html(index=False) + "</p>"

        grouped_list = list(grouped_df)
        groups = []
        index = len(grouped_list)
        for x in range(index):
            misclass = grouped_list[x][0]
            num = len(grouped_list[x][1])
            percent = num / errorsdf[1].sum()
            percent_all = num / len(X)
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

    timedate = str(datetime.now())
    timedate = re.sub('[^0-9]','', timedate)
    heading="Comparison"
    html_head = "<p>" + heading.upper()+" ERROR REPORT - " +timedate + "<br>"
    html_head += "MODEL 1: " + model_name1.upper()+" - "+ fit_params1 + " train time: " + train_time1 + "<br>"
    html_head += "MODEL 2: " + model_name2.upper()+" - "+ fit_params2 + " train time: " + train_time2 + "<br>"
    html_head += "MODEL 1 TOTAL ERRORS: " + str(err1total) + "<br>"
    html_head += "MODEL 2 TOTAL ERRORS: " + str(err2total) + "<br>"
    html_head += "PREDICTION IMPROVEMENT FROM MODEL 1 TO 2: " + str(errsimptotal)  + "<br>"
    html_head += "PREDICTION REDUCTION FROM MODEL 1 TO 2: " + str(errsworsetotal)  + "<br>"
    html_head += "NET PREDICTION IMPROVEMENT: " + str(err1total - err2total)  + "<br>"
    html_head += "MODEL 1 ERROR PERCENTAGE: " + str(errrate1)  + "<br>"
    html_head += "MODEL 2 ERROR PERCENTAGE: " + str(errrate2)  + "<br>"
    tit = "COMPARISON " + " - " + timedate
    imp_grp = 'IMPROVED'
    worse_grp = 'WORSENED'
    imp_exp = ' - ' + model_name1.upper() + ' ERRED, ' + model_name2.upper() + ' PREDICTED CORRECTLY (CORRECT/' + model_name2.upper() + ' PRED, ' + model_name1.upper() + ' ERROR)'
    worse_exp = ' - ' + model_name1.upper() + ' PREDICTED CORRECTLY, ' + model_name2.upper() + ' ERRED (CORRECT/' + model_name1.upper() + ' PRED, ' + model_name2.upper() + ' ERROR)'

    knownimp_grp = 'KNOWN IMPROVED'
    knownworse_grp = 'KNOWN WORSENED'
    knownimp_exp = ' - ' + model_name1.upper() + ' ERRED, ' + model_name2.upper() + ' PREDICTED CORRECTLY (CORRECT/' + model_name2.upper() + ' PRED, ' + model_name1.upper() + ' ERROR)'
    knownworse_exp = ' - ' + model_name1.upper() + ' PREDICTED CORRECTLY, ' + model_name2.upper() + ' ERRED (CORRECT/' + model_name1.upper() + ' PRED, ' + model_name2.upper() + ' ERROR)'

    unknownimp_grp = 'UNKNOWN IMPROVED'
    unknownworse_grp = 'UNKNOWN WORSENED'
    unknownimp_exp = ' - ' + model_name1.upper() + ' ERRED, ' + model_name2.upper() + ' PREDICTED CORRECTLY (CORRECT/' + model_name2.upper() + ' PRED, ' + model_name1.upper() + ' ERROR)'
    unknownworse_exp = ' - ' + model_name1.upper() + ' PREDICTED CORRECTLY, ' + model_name2.upper() + ' ERRED (CORRECT/' + model_name1.upper() + ' PRED, ' + model_name2.upper() + ' ERROR)'

    figname1 = imp_grp.upper() + timedate
    figname1=figname1.replace(" ", "")
    figname2 = worse_grp.upper() + timedate
    figname2=figname2.replace(" ", "")

    figname3 = knownimp_grp.upper() + timedate
    figname3= figname3.replace(" ", "")
    figname4 = knownworse_grp.upper() + timedate
    figname4=figname4.replace(" ", "")

    figname5 = unknownimp_grp.upper() + timedate
    figname5= figname5.replace(" ", "")
    figname6 = unknownworse_grp.upper() + timedate
    figname6=figname6.replace(" ", "")

    improvement = errors(errsimp,imp_grp,imp_exp,'prediction1')
    worse = errors(errsworse,worse_grp,worse_exp,'prediction2')

    knownimprovement = errors(knownerrsimp,knownimp_grp,knownimp_exp,'prediction1')
    knownworse = errors(knownerrsworse,knownworse_grp,knownworse_exp,'prediction2')

    unknownimprovement = errors(unknownerrsimp,unknownimp_grp,unknownimp_exp,'prediction1')
    unknownworse = errors(unknownerrsworse,unknownworse_grp,unknownworse_exp,'prediction2')


    #model1err = predicterrors(pred1,Xtest,Ytest,fit_params1,wordlist,taglist,idx2tag,word2idx,heading1,train_time1,console_output=False,web_rpt=True)

    #model2err = predicterrors(pred2,Xtest,Ytest,fit_params2,wordlist,taglist,idx2tag,word2idx,heading2,train_time2,console_output=False,web_rpt=True)


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
        <li class="active"><a data-toggle="tab" href="#improvement">IMPROVED</a></li>

        <li><a data-toggle="tab" href="#worse">WORSENED</a></li>

        <li><a data-toggle="tab" href="#knownimprovement">KNOWN IMPROVED</a></li>

        <li><a data-toggle="tab" href="#knownworse">KNOWN WORSENED</a></li>

        <li><a data-toggle="tab" href="#unknownimprovement">UNKNOWN IMPROVED</a></li>

        <li><a data-toggle="tab" href="#unknownworse">UNKNOWN WORSENED</a></li>


    </ul>


    <div class="tab-content">
        <div id="improvement" class="tab-pane fade in active">
         <a data-toggle="tab" href="#chart_improvement">Bar Graph</a><br>
    ''' + improvement + ''' </div>
        <div id="worse" class="tab-pane fade">
        <a data-toggle="tab" href="#chart_worse">Bar Graph</a><br>
    ''' + worse + '''</div>

        <div id="chart_improvement" class="tab-pane fade">
        <a data-toggle="tab" href="#improvement">IMPROVED</a><br>
        <img src="
    ''' + figname1 + '''.jpg" alt="All Errors Bar Graph" >
        </div>
        <div id="chart_worse" class="tab-pane fade">
        <a data-toggle="tab" href="#worse">WORSENED</a><br>
        <img src="
    ''' + figname2 + '''.jpg" alt=" Errors Bar Graph" >
        </div>

        <div id="knownimprovement" class="tab-pane fade">
         <a data-toggle="tab" href="#chart_knownimprovement">Bar Graph</a><br>
    ''' + knownimprovement + ''' </div>
        <div id="knownworse" class="tab-pane fade">
        <a data-toggle="tab" href="#chart_knownworse">Bar Graph</a><br>
    ''' + knownworse + '''</div>

        <div id="chart_knownimprovement" class="tab-pane fade">
        <a data-toggle="tab" href="#knownimprovement">KNOWN IMPROVED</a><br>
        <img src="
    ''' + figname3 + '''.jpg" alt="All Errors Bar Graph" >
        </div>
        <div id="chart_knownworse" class="tab-pane fade">
        <a data-toggle="tab" href="#knownworse">KNOWN WORSENED</a><br>
        <img src="
    ''' + figname4 + '''.jpg" alt="Known Errors Bar Graph" >
        </div>


        <div id="unknownimprovement" class="tab-pane fade">
         <a data-toggle="tab" href="#chart_unknownimprovement">Bar Graph</a><br>
    ''' + unknownimprovement + ''' </div>
        <div id="unknownworse" class="tab-pane fade">
        <a data-toggle="tab" href="#chart_unknownworse">Bar Graph</a><br>
    ''' + unknownworse + '''</div>

        <div id="chart_unknownimprovement" class="tab-pane fade">
        <a data-toggle="tab" href="#unknownimprovement">UNKNOWN IMPROVED</a><br>
        <img src="
    ''' + figname5 + '''.jpg" alt="All Errors Bar Graph" >
        </div>
        <div id="chart_unknownworse" class="tab-pane fade">
        <a data-toggle="tab" href="#unknownworse">UNKNOWN WORSENED</a><br>
        <img src="
    ''' + figname6 + '''.jpg" alt="Known Errors Bar Graph" >
        </div>

    </div>
    </div>
    </body></html>
    '''


    import webbrowser
    f= open("comparison-"+timedate+".html","w+")
    f.write(html)
    f.close()
    url = "comparison-"+timedate+".html"
    chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
    brave_path = 'open -a Brave.app %s'
    webbrowser.get(chrome_path).open(url,new=1)


def main():
    pred1,Xtest,Ytest,wordlist,taglist,idx2tag,word2idx,model_name1,fit_params1,train_time1 = LR_output_predictions(5)
    pred2,_,_,_,_,_,_,model_name2,fit_params2,train_time2 = markov_output_predictions()

    comparison(pred1,pred2,Xtest,Ytest,wordlist,taglist,idx2tag,model_name1,fit_params1,model_name2,fit_params2,train_time1,train_time2)
    predicterrors(pred1,Xtest,Ytest,fit_params1,wordlist,taglist,idx2tag,word2idx,model_name1,train_time1,console_output=False,web_rpt=True)
    predicterrors(pred2,Xtest,Ytest,fit_params2,wordlist,taglist,idx2tag,word2idx,model_name2,train_time2,console_output=False,web_rpt=True)


if __name__ == '__main__':
    main()
