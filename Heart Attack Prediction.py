#!/usr/bin/env python
# coding: utf-8

# In[78]:


import numpy as np
import pandas as pd


# In[79]:


df=pd.read_csv('Blood_Pressure_data.csv')


# In[80]:


df


# In[81]:


df['diag_2'].value_counts().head(30)


# In[82]:


df.drop_duplicates(subset='patient_no',keep=False,inplace=True)


# In[83]:


len(df)


# In[84]:


for col_name in df.columns:
    if df[col_name].dtypes == 'object':
        unique_cat = len(df[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat))


# In[85]:


print(df['medical_specialty'].value_counts().sort_values(ascending=False).head(20))


# In[86]:




df['cast'] = ['Caucasian' if x == 'Caucasian' else 'Other' for x in df['cast']]


# In[87]:



print(df['cast'].value_counts().sort_values(ascending=False))


# In[ ]:





# In[88]:


df.columns


# In[89]:


df['label'].value_counts()


# In[90]:


df['diag_1'].dtype


# In[ ]:





# In[ ]:





# In[91]:


df['change'].value_counts()


# In[92]:


df['number_diagnoses'].value_counts()


# In[93]:


#df['discharge_disposition_id'].value_counts()
#df['admission_typeid'] = [1 if x == 'Caucasian' else 'Other' for x in df['cast']]


# In[94]:


df['admission_typeid'].value_counts()


# In[95]:


df['discharge_disposition_id'].value_counts()
                                  


# In[96]:


df['discharge_disposition_id'] = [1 if x == 1 else 3 if x==3 else 6 if x==6 else 9  for x in df['discharge_disposition_id']]


# In[97]:


df['discharge_disposition_id'].value_counts()


# In[98]:


df['admission_typeid'] = [1 if x == 1 else 3 if x==3 else 2 if x==2 else 4  for x in df['admission_typeid']]


# In[99]:


df['admission_typeid'].value_counts()


# In[100]:


df['admission_source_id'].value_counts()


# In[101]:


df['admission_source_id'] = [7 if x == 7 else 1 if x==1 else 2 for x in df['admission_source_id']]


# In[102]:


df['admission_source_id'].value_counts()


# In[103]:


df['diag_2'].value_counts().head(20)


# In[104]:


df['Med'].value_counts()


# In[105]:


df['change'].value_counts()


# In[106]:


print(df['weight'].value_counts())
print('Shape of data frame :',df.shape)


# In[107]:


#df['age group'].value_counts()
print(df.groupby(['age group'])['weight'].value_counts())


# In[108]:


print('Admission Type ID...')
print(df['admission_typeid'].value_counts())

print()

print('discharge_disposition_id')
print(df['discharge_disposition_id'].value_counts())


# In[109]:


#checking missing Data in data frame


df.isnull().sum().sort_values(ascending=False)


# In[110]:


# checking Data Frame Values with '?' (missing values)

df[df=='?'].count()


# In[111]:


print(df[df=='No'].count())
print(df.shape)


# In[ ]:





# In[112]:


# dropping features with close to no contribtion to the models acccuracy Having more than 96% missing values

df=df.drop(['weight','id', 'patient_no','payer_code','metformin-pioglitazone','metformin-rosiglitazone','glimepiride-pioglitazone','glipizide-metformin','glyburide-metformin','citoglipton','examide','tolazamide','troglitazone','miglitol','tolbutamide','acarbose','repaglinide','nateglinide','chlorpropamide','acetohexamide','glimepiride','pioglitazone','rosiglitazone','medical_specialty'], axis = 1) 



# In[113]:


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
df['age group']=le.fit_transform(df['age group'])
   


# In[114]:


df.columns


# In[115]:



df['diag_1']=df['diag_1'].apply(lambda df: pd.to_numeric(df,errors='coerce'))


# In[116]:


df['diag_2']=df['diag_2'].apply(lambda df: pd.to_numeric(df,errors='coerce'))


# In[117]:


df['diag_3']=df['diag_3'].apply(lambda df: pd.to_numeric(df,errors='coerce'))


# In[118]:


df['diag_2'].dtype


# In[119]:


df.isnull().sum().sort_values(ascending=False)


# In[120]:


df['diag_1'].fillna(value=df['diag_1'].mode(), inplace=True)


# In[121]:


df['diag_2'].fillna(value=df['diag_2'].mode(), inplace=True)


# In[122]:


df['diag_3'].fillna(value=df['diag_3'].mode(), inplace=True)


# In[123]:


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
df['diag_1']=le.fit_transform(df['diag_1'])
df['diag_2']=le.fit_transform(df['diag_2'])
df['diag_3']=le.fit_transform(df['diag_3'])


# In[124]:


df.isnull().sum().sort_values(ascending=False)


# In[125]:


Dummies=pd.get_dummies(df,columns=['cast','gender','change','Med'])


# In[ ]:





# In[ ]:





# In[126]:


Dummies.drop([col for col in Dummies.columns if '?' in col],axis=1,inplace=True)


# In[127]:


Dummies


# In[128]:


print(Dummies.columns)


# In[129]:


Dummies.columns


# In[130]:


#print(df['medical_specialty'].value_counts().sort_values(ascending=False).head(20))


# In[131]:


Dummies['max_glu_serum'].value_counts()


# In[132]:


from sklearn.preprocessing import LabelEncoder


# In[133]:


order={"None":0,"Norm":1,">200":2,">300":3}
Dummies['max_glu_serum']=Dummies['max_glu_serum'].map(order)


# In[134]:


Dummies['max_glu_serum'].value_counts()


# In[135]:


Dummies['A1Cresult'].value_counts()


# In[136]:


order={"None":0,"Norm":1,">7":2,">8":3}
Dummies['A1Cresult']=Dummies['A1Cresult'].map(order)


# In[137]:


Dummies['A1Cresult'].value_counts()


# In[138]:


Dummies['metformin'].value_counts()


# In[139]:


order={"Down":0,"No":1,"Steady":2,"Up":3}



Dummies['metformin']=Dummies['metformin'].map(order)




Dummies['glipizide']=Dummies['glipizide'].map(order)
Dummies['glyburide']=Dummies['glyburide'].map(order)


Dummies['insulin']=Dummies['insulin'].map(order)




#Dummies=Dummies.drop('metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','insulin','glyburide-metformin', axis = 1)


# In[140]:


#from sklearn.preprocessing import LabelEncoder

#le=LabelEncoder()
#Dummies['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','insulin','glyburide-metformin']=le.fit_transform(Dummies['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide','glipizide','glyburide','pioglitazone','rosiglitazone','insulin'])
   


# In[141]:


Dummies


# In[142]:


order={"NO":0,">5":1,"<30":2}

Dummies['label']=Dummies['label'].map(order)


# In[143]:


Dummies['label'].value_counts()


# In[144]:


X = Dummies.drop("label", axis = 1)
y = Dummies["label"]


# In[145]:


print('Shape of X = ', X.shape)
print('Shape of y = ', y.shape)


# In[146]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=54)


# In[147]:


import sklearn.feature_selection

select = sklearn.feature_selection.SelectKBest(k=20)
selected_features = select.fit(X_train, y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [X.columns[i] for i in indices_selected]

X_train_selected = X_train[colnames_selected]
X_test_selected = X_test[colnames_selected]


# In[148]:


from sklearn.preprocessing import StandardScaler


# In[149]:


sc = StandardScaler()

sc.fit(X_train)

X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)


# In[150]:


X_test_sc.shape


# In[151]:


X_train.isnull().sum().sort_values(ascending=False)


# In[152]:


#from sklearn.decomposition import PCA

#pca = PCA(n_components=10)
#X_pca = pd.DataFrame(pca.fit_transform(Dummies))




from sklearn.decomposition import PCA

pca=PCA(n_components=10)
X_trainn=pca.fit_transform(X_train_sc)
X_testt=pca.transform(X_test_sc)


# In[153]:


from sklearn.neighbors import KNeighborsClassifier


# In[154]:


clf0=KNeighborsClassifier()
clf0.fit(X_train_sc,y_train)
pred0=clf0.predict(X_test_sc)
clf0.score(X_test_sc,y_test)


# In[155]:


from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier


# In[156]:


clf1=AdaBoostClassifier()
clf1.fit(X_train_sc,y_train)
pred1=clf1.predict(X_test_sc)
clf1.score(X_test_sc,y_test)


# In[157]:


clf1=AdaBoostClassifier()
clf1.fit(X_train_sc,y_train)
pred1=clf1.predict(X_test_sc)
clf1.score(X_test_sc,y_test)


# In[158]:



clf2=RandomForestClassifier()
clf2.fit(X_train_sc,y_train)
pred2=clf2.predict(X_test_sc)
clf2.score(X_test_sc,y_test)


# In[159]:


from sklearn import svm


# In[160]:


clf3=svm.SVC()
clf3.fit(X_train_sc,y_train)
pred3=clf3.predict(X_test_sc)
clf3.score(X_test_sc,y_test)


# In[ ]:





# In[161]:


#only the best from the 4 models will be used


# In[162]:


len(pred1)


# In[ ]:





# In[163]:


len(pred2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[164]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report 
print ("Accuracy : " , accuracy_score(y_test,pred3)*100)  
print("Report : \n", classification_report(y_test, pred3))


# In[ ]:




