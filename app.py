from flask import Flask,render_template,request 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import string
import re

app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")
    

def wordopt(text):
    text=text.lower()
    text=re.sub('\[.*?\]', '',text)
    text=re.sub("\\W"," ",text)
    text=re.sub('https?://\S+|www\.\S+', '',text)
    text=re.sub('<.*?>+', '',text)
    text=re.sub('[%s]' % re.escape(string.punctuation), '',text)
    text=re.sub('\n', '',text)
    text=re.sub('\w*\d\w*', '',text)
    return text


@app.route('/result',methods=['POST'])
def result():
    model=pickle.load(open('model.pkl','rb'))
    vect=pickle.load(open('tfidfvect2.pkl','rb'))
    news=request.form.get("article")
    testing_news={"text":[news]}
    new_def_test=pd.DataFrame(testing_news)
    new_def_test["text"]=new_def_test["text"].apply(wordopt)
    new_x_test=new_def_test["text"]
    new_xv_test=vect.transform(new_x_test)
    prediction=model.predict(new_xv_test)[0]
    if prediction==0:
        result='FAKE'
    else:
        result='TRUE'        
    return render_template('result.html',news=news,result=result)


if(__name__=='__main__'):
    app.run()





