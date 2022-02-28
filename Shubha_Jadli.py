#!/usr/bin/env python
# coding: utf-8

# # Assignment 2

# In[121]:


import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pandas import read_csv


# # German Credit data set
# Prediction using:
# 1. KNN
# 2. Binary Tree
# 3. Emsemble classifier ( Using Binary Tree )

# In[199]:


# INPUT VALUES WHICH HAVE TO BE TESTED
print("Enter all values as asked, do not enter decimal values (round off if necessary)")
c1=input('Enter Age')
c2=input('Enter 1 for male or 0 for female')
c3=input('Enter Job Value between \n 0 1 2')
c4=input('Enter Housing value \n free=0, rent=1,own=2')
c5=input('Enter Saving accounts value \n little=0, moderate=1, quite rich =2, rich= 3')
c6=input('Enter Checking account value \n little=0, moderate=1, rich=2')
c7=input('Enter Credit amount')
c8=input('Enter Duration in months')
c9=input('Enter purpose \n car= 0, radio/TV= 1,furniture/equipment= 2,business= 3,education= 4,repairs= 5,vacation/others= 6,domestic appliances =7')        
pred_input=[c1,c2,c3,c4,c5,c6,c7,c8,c9]


# In[240]:


data1=pd.read_csv(r'/Users/shubhamjadli/Desktop/Term 3/MLP/DataSets/german_credit_data.csv')
data1.iloc[:,5].value_counts() # Finding missing values
data1.iloc[:,5].fillna('little',inplace=True)# Replacing with most probable
data1=data1.dropna(axis=0) # dropping all NA values as they were too close
data1.duplicated().sum() #checking for duplicates
#Mapping numeric values to independent variables
data1.Sex=data1.Sex.map({'male':1,'female':0})
data1.Housing=data1.Housing.map({'free':0,'rent':1,'own':2})
data1.iloc[:,5]=data1.iloc[:,5].map({'little':0,'moderate':1,'quite rich':2,'rich':3})
data1.iloc[:,6]=data1.iloc[:,6].map({'little':0,'moderate':1,'rich':2})
data1.Purpose=data1.Purpose.map({'car':0,'radio/TV':1,'furniture/equipment':2,
                                'business':3,'education':4,'repairs':5,'vacation/others':6,
                               'domestic appliances':7})
data1.Risk=data1.Risk.map({'good':1,'bad':0})
#Segregating target and independent variables
x=data1.iloc[:,1:10]
y=data1.iloc[:,10]
#Splitting the data set into train test 
x_train,x_test,y_train,y_test= train_test_split(x,y,random_state=1)


#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
KNN=KNeighborsClassifier(n_neighbors=5, metric='euclidean')
KNN.fit(x_train,y_train)
y_pred_KNN=KNN.predict(x_test)
print('Accuracy of dataset with KNN = \n',accuracy_score(y_test, y_pred_KNN))
print('Precision score of dataset with KNN = \n',precision_score(y_test, y_pred_KNN))
print('Recall score of dataset with KNN = \n',recall_score(y_test, y_pred_KNN))
print('F1 score of dataset with KNN = \n',f1_score(y_test, y_pred_KNN))

#Binary Tree
from sklearn.tree import DecisionTreeClassifier as dtree
bt=dtree(criterion='entropy',max_depth=None)
bt.fit(x_train,y_train)
y_pred_BT=bt.predict(x_test)
print('Accuracy of dataset with Binary(Entropy) = \n',accuracy_score(y_test, y_pred_BT))
print('Feature importances of columns = \n',bt.feature_importances_)

#Ensembled (Random forest)
from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(n_estimators=90)
RF.fit(x_train,y_train)
y_pred_RF=RF.predict(x_test)
print('Accuracy of dataset with Random Forest= \n',accuracy_score(y_test, y_pred_RF))

#As the accuracy for Ensembled is the highest hence we use the same for prediction of risk 
#The Values have already been inputted in pred_input
pred_input=np.array(pred_input).astype('int32')
pred_input=pred_input.reshape(1,9)
pred_input=pd.DataFrame(pred_input)
pred_input.columns= (['Age', 'Sex', 'Job', 'Housing', 'Saving accounts','Checking account', 'Credit amount', 'Duration', 'Purpose'])
print('\n If risk present value =1 if no risk value =0')
print('\n Risk = \n',RF.predict(pred_input))


# # Income Classification data set
# Prediction using:
# 1. KNN
# 2. Binary Tree
# 3. Emsemble classifier ( Using Binary Tree )

# In[419]:


data2.columns
d1=input('Enter Age')
d2=input('Enter Workclass as following: \nPrivate :0 \nSelf-emp-not-inc :1 \nLocal-gov :2 \nNot known :3, \nState-gov :4 \nSelf-emp-inc :5\nFederal-gov :6 \nWithout-pay :7 \nNever-worked :8')
d3=input('Enter fnlwgt')
d4=input('Enter Education as following: \nHS-grad :0 \nSome-college :1 \nBachelors :2 \nMasters :3 \nAssoc-voc :4 \n11th :5 \nAssoc-acdm :6, \n10th :7 \n7th-8th :8 \nProf-school :9 \n9th:10  \n12th :11 \nDoctorate :12 \n5th-6th :13 \n1st-4th :14 \nPreschool:15')
d5=input('Enter Eduction-num')
d6=input('Enter marital status as following: \n Married-civ-spouse :0 \nNever-married :1 \nDivorced :2 \nSeparated :3 \nWidowed :4 \nMarried-spouse-absent :5 \nMarried-AF-spouse :6')
d7=input('Enter occupation as following: \n Unknown:0 \n Adm-clerical :1 \nArmed-Forces :2 \nCraft-repair :3 \nExec-managerial :4 \nFarming-fishing :5 \nHandlers-cleaners :6 \nMachine-op-inspct :7 \nOther-service :8 \nPriv-house-serv:9 \nProf-specialty :10 \nProtective-serv:11 \nSales :12 \nTech-support:13 \nTransport-moving :14')
d8=input('Enter relationship as following: \nHusband :0 \nNot-in-family :1 \nOther-relative :2 \nOwn-child :3 \nUnmarried :4 \nWife :5')
d9=input('Enter race as following: \nAmer-Indian-Eskimo :0 \nAsian-Pac-Islander :1 \nBlack :2, \nOther :3 \nWhite :4')
d10=input('Enter gender as follows Male :1 Female :0')
d11=input('Enter capital gain')
d12=input('Enter capital loss')
d13=input('Enter hours per week')
d14=input('Enter native country as follows: \nUnknown :0 \nCambodia :1 \nCanada :2 \nChina :3 \nColumbia :4 \nCuba :5 \nDominican-Republic :6 \nEcuador :7 \nEl-Salvador :8 \nEngland :9 \nFrance :10 \nGermany :11 \nGreece :12 \nGuatemala :13 \nHaiti :14 \nHoland-Netherlands :15 \nHonduras :16 \nHong :17 \nHungary :18 \nIndia :19 \n Iran :20 \nIreland :21 \nItaly :22 \nJamaica :23 \nJapan :24 \nLaos :25 \nMexico :26 \nNicaragua :27 \nOutlying-US(Guam-USVI-etc) :28 \nPeru :29 \nPhilippines :30 \nPoland :31 \nPortugal :32 \nPuerto-Rico :33 \nScotland :34 \nSouth :35 \nTaiwan :36 \nThailand :37 \nTrinadad&Tobago :38 \nUnited-States :39 \nVietnam :40 \nYugoslavia:41')
pred_input2=[d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14]


# In[420]:


pred_input2


# In[426]:


data2=pd.read_csv(r'/Users/shubhamjadli/Desktop/Term 3/MLP/DataSets/income_evaluation.csv')
data2.isna().sum()
data2.duplicated().sum()
data2.drop_duplicates(keep='first')
data2.columns=['age','workclass','fnlwgt','education','education-num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours-per-week','native_country','income']
data2.workclass=data2.workclass.map({' Private':0,' Self-emp-not-inc':1,' Local-gov':2,' ?':3,' State-gov':4,' Self-emp-inc':5,' Federal-gov':6,' Without-pay':7,' Never-worked':8})
data2.education=data2.education.map({' HS-grad':0,' Some-college':1,' Bachelors':2,' Masters':3,' Assoc-voc':4,' 11th':5,' Assoc-acdm':6,' 10th':7,' 7th-8th':8,' Prof-school':9,' 9th':10,' 12th':11,' Doctorate':12,' 5th-6th':13,' 1st-4th':14,' Preschool':15})
data2.marital_status=data2.marital_status.map({' Married-civ-spouse':0,' Never-married':1,' Divorced':2,' Separated':3,' Widowed':4,' Married-spouse-absent':5,' Married-AF-spouse':6})
data2.occupation=data2.occupation.map({' ?':0,' Adm-clerical':1,' Armed-Forces':2,' Craft-repair':3,
       ' Exec-managerial':4,' Farming-fishing':5,' Handlers-cleaners':6,
       ' Machine-op-inspct':7,' Other-service':8,' Priv-house-serv':9,
       ' Prof-specialty':10,' Protective-serv':11,' Sales':12,' Tech-support':13,
       ' Transport-moving':14})
data2.relationship=data2.relationship.map({' Husband':0,' Not-in-family':1,' Other-relative':2,' Own-child':3,
       ' Unmarried':4,' Wife':5})
data2.race=data2.race.map({' Amer-Indian-Eskimo':0,' Asian-Pac-Islander':1,' Black':2,' Other':3,
       ' White':4})
data2.sex=data2.sex.map({' Male':1,' Female':0})
data2.native_country=data2.native_country.map({' ?':0, ' Cambodia':1, ' Canada':2, ' China':3, ' Columbia':4, ' Cuba':5,
       ' Dominican-Republic':6, ' Ecuador':7, ' El-Salvador':8, ' England':9,
       ' France':10, ' Germany':11, ' Greece':12, ' Guatemala':13, ' Haiti':14,
       ' Holand-Netherlands':15, ' Honduras':16, ' Hong':17, ' Hungary':18, ' India':19,
       ' Iran':20, ' Ireland':21, ' Italy':22, ' Jamaica':23, ' Japan':24, ' Laos':25,
       ' Mexico':26, ' Nicaragua':27, ' Outlying-US(Guam-USVI-etc)':28, ' Peru':29,
       ' Philippines':30, ' Poland':31, ' Portugal':32, ' Puerto-Rico':33,
       ' Scotland':34, ' South':35, ' Taiwan':36, ' Thailand':37, ' Trinadad&Tobago':38,
       ' United-States':39, ' Vietnam':40, ' Yugoslavia':41})
data2.income=data2.income.map({' <=50K':0,' >50K':1})

#Segregating target and independent variables
x=data2.iloc[:,0:14]
y=data2.iloc[:,14]
#Splitting the data set into train test 
x_train,x_test,y_train,y_test= train_test_split(x,y,random_state=1)

#KNN
KNN=KNeighborsClassifier(n_neighbors=5, metric='euclidean')
KNN.fit(x_train,y_train)
y_pred_KNN=KNN.predict(x_test)
print('Accuracy of dataset with KNN = \n',accuracy_score(y_test, y_pred_KNN))
print('Precision score of dataset with KNN = \n',precision_score(y_test, y_pred_KNN))
print('Recall score of dataset with KNN = \n',recall_score(y_test, y_pred_KNN))
print('F1 score of dataset with KNN = \n',f1_score(y_test, y_pred_KNN))

#Binary Tree
bt=dtree(criterion='gini',max_depth=None)
bt.fit(x_train,y_train)
y_pred_BT=bt.predict(x_test)
print('Accuracy of dataset with Binary(Entropy) = \n',accuracy_score(y_test, y_pred_BT))
print('Feature importances of columns = \n',bt.feature_importances_)

#Ensembled (Random forest)
RF=RandomForestClassifier(n_estimators=100)
RF.fit(x_train,y_train)
y_pred_RF=RF.predict(x_test)
print('Accuracy of dataset with Random Forest= \n',accuracy_score(y_test, y_pred_RF))

#As the accuracy for Ensembled is the highest hence we use the same for prediction of income 
#The Values have already been inputted in pred_input2
pred_input2=np.array(pred_input2).astype('int64')
pred_input2=pred_input2.reshape(1,14)
pred_input2=pd.DataFrame(pred_input2)
pred_input2.columns=['age','workclass','fnlwgt','education','education-num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours-per-week','native_country']
print('Estimated income: \n Less than or equal to 50K = 0 \n Greater than 50K =1')
print('Final prediction = ' , RF.predict(pred_input2))


# # Real and Fake news data set
#  TfidfVectorizer Method

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import re
import string
data3=pd.read_csv(r'/Users/shubhamjadli/Desktop/Term 3/MLP/DataSets/Real_fake_news.csv')
data3.isna().sum()
data3.duplicated().sum()
data3=data3.iloc[:,[3,6]]
data3.reset_index(inplace = True)
data3.drop(["index"], axis = 1, inplace = True)

def cleaning(statement):
    statement = statement.lower()
    statement = re.sub('\[.*?\]', '', statement)
    statement = re.sub("\\W"," ",statement) 
    statement = re.sub('https?://\S+|www\.\S+', '', statement)
    statement = re.sub('<.*?>+', '', statement)
    statement = re.sub('[%s]' % re.escape(string.punctuation), '', statement)
    statement = re.sub('\n', '', statement)
    statement = re.sub('\w*\d\w*', '', statement)    
    return statement
data3["statement"] =data3["statement"].apply(cleaning)
data3.BinaryTarget=data3.BinaryTarget.map({'REAL':1,'FAKE':0})
x=data3["statement"]
y=data3["BinaryTarget"]
label=data3.BinaryTarget
x_train,x_test,y_train,y_test=train_test_split(data3['statement'], label, test_size=0.7)
vector=TfidfVectorizer(stop_words='english', max_df=0.8)
x_train_vector=vector.fit_transform(x_train)
x_test_vector=vector.transform(x_test)
vector=TfidfVectorizer(stop_words='english', max_df=0.8)

#KNN
KNN=KNeighborsClassifier(n_neighbors=5, metric='euclidean')
KNN.fit(x_train_vector, y_train)
y_pred_KNN=KNN.predict(x_test_vector)
print("Accuracy score for data set using KNN = \n",accuracy_score(y_test, y_pred_KNN))

# Decision Tree 
DT = dtree()
DT.fit(x_train_vector, y_train)
pred_dt = DT.predict(x_test_vector)
print("Accuracy score for data set using Decision Tree = \n",accuracy_score(y_test, pred_dt))

#Ensembled (Random Forest)
RFC = RandomForestClassifier(random_state=0)
RFC.fit(x_train_vector, y_train)
pred_rfc = RFC.predict(x_test_vector)
print("Accuracy score for data set using Random Forest = \n",accuracy_score(y_test, pred_rfc))

#As we are getting highest accuracy for random forest we 
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"statement":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["statement"] = new_def_test["statement"].apply(cleaning) 
    new_x_test = new_def_test["statement"]
    new_xv_test = vector.transform(new_x_test)
    pred_KNN = KNN.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)
    return print("Predictions \nKNN Prediction: {} \nDT Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_KNN[0]), 
                                                                                                              output_lable(pred_DT[0]), 
                                                                                                              output_lable(pred_RFC[0])))
print("Enter test news")
news= str(input())
manual_testing(news)


# In[ ]:




