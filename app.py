from flask import Flask,render_template,url_for,request
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import joblib

filename = 'model.pkl'
model = pickle.load(open(filename,'rb'))
cv = pickle.load(open('cv.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])


def predict(): 
    if request.method == 'POST':
        message = request.form['message']
        review=[message]
        corpus = cv.transform(review).toarray()
        pred=model.predict(corpus)
    return render_template('result.html', prediction = pred)
    
if __name__ == '__main__':
    app.run(debug=True)
