from __future__ import division
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
import re
import regex
from sklearn.cross_validation import train_test_split
ps = PorterStemmer()
from nltk import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

with open('train.txt') as f:
    train = f.readlines()
trainRaw = [x.strip() for x in train]

with open('test.txt') as f:
    test = f.readlines()
testRaw = [x.strip() for x in test]


#Paper suggested normalizing
def Normalize(data):
    labels = [i.split('\t', 1)[0] for i in data]
    traindata = [i.split('\t', 1)[1] for i in data]
    ps = PorterStemmer() 
    traindata = [" ".join([ps.stem(word.decode('utf-8')) for word in sentence.split(" ")]) for sentence in traindata]
    traindata = [re.sub(r"\$\d+", "$MONEY", i) for i in traindata]
    traindata = [re.sub(r"(<?)http:\S+", "$LINK", i) for i in traindata]
    traindata = [re.sub(r'[\w\.-]+@[\w\.-]+', "$EMAILID", i) for i in traindata]
    traindata = [i.lower() for i in traindata]
    traindata = [regex.sub(r"[^\P{P}$]+", " ", i) for i in traindata]
    traindata = [re.sub(r"[^0-9A-Za-z/$' ]", " ", i) for i in traindata]
    regString = r'monday|tuesday|wednesday|thursday|friday|saturday|sunday'
    traindata = [re.sub(regString, "$days", i) for i in traindata]
    regString = r'january|jan|february|feb|march|mar|april|june|july|august|aug|september|sept|october|oct|november|nov|december|dec'
    traindata = [re.sub(regString, "$month", i) for i in traindata]
    regString = r'after|before|during'
    traindata = [re.sub(regString, "$time", i) for i in traindata]
    traindata = [re.sub(r'\b\d+\b', "$number", i) for i in traindata]
    traindata = [re.sub(r'\b(me|her|him|us|them|you)\b', "$me", i) for i in traindata]
    traindata = [i.strip() for i in traindata]
    
    return traindata, labels
    


    
    

#cleaning and splitting data
trainData, trainLabels = Normalize(trainRaw)
testData, testLabels = Normalize(testRaw)
data = trainData + testData
labels = trainLabels + testLabels
vectorizer = TfidfVectorizer(min_df=2)#tried 1,2,3. 2 gave balanced results for precision and recall. 
X = vectorizer.fit_transform(data) 
Y = labels

x_train, x_test, y_train, y_test = train_test_split(
  X, Y, test_size=0.2, random_state=42
)
#using a SVM
svm = SVC(C=1900, kernel='rbf')#shouldve used grid search, trial and error
svm.fit(x_train, y_train)
pred = svm.predict(x_test)
#calculating precision and recall
cm = confusion_matrix(y_test, pred)
TP = cm[0][0]
FN = cm[1][0]
FP = cm[0][1]
TPFN = TP+FN
p = TP/TPFN
TPFP = TP+FP
r = TP/TPFP
print 'Precision:',float(p) 
print 'Recall:',float(r)
#Precision: 0.800356506239
#Recall: 0.826887661142




