#!/usr/bin/env python
# coding: utf-8

# # Preamble

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


spam = pd.read_csv('spam.csv', sep=',')


# In[3]:


spam.head()


# In[4]:


spam.info()


# # Data Normalisation

# In[5]:


spam.isnull().sum()


# In[6]:


#spam.drop('Unnamed: 2', axis=1, inplace=True)
#spam.drop('Unnamed: 3', axis=1, inplace=True)
#spam.drop('Unnamed: 4', axis=1, inplace=True)


# In[7]:


spam.isnull().sum()


# In[8]:


spam.head()


# In[9]:


label_quality = LabelEncoder()


# In[10]:


spam['class'] = label_quality.fit_transform(spam['class'])


# In[11]:


spam.head(10)


# In[12]:


spam['class'].value_counts()


# In[13]:


sns.countplot(spam['class'])


# In[14]:


nltk.download('stopwords')
stopword=set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
spam["message"] = spam["message"].apply(clean)
spam.head()


# In[15]:


x = np.array(spam["message"])
y = np.array(spam["class"])

cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# # Decision Tree Classifier

# In[16]:


dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
pred_dtc = dtc.predict(X_test)
print(dtc.score(X_test,y_test))


# In[17]:


print(classification_report(y_test, pred_dtc))
print(confusion_matrix(y_test, pred_dtc))


# # Random Forest Classifier

# In[18]:


rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)


# In[19]:


print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))


# # SVM Classifier

# In[20]:


svmc = svm.SVC()
svmc.fit(X_train, y_train)
pred_svmc = svmc.predict(X_test)


# In[21]:


print(classification_report(y_test, pred_svmc))
print(confusion_matrix(y_test, pred_svmc))


# # Neural Network

# In[22]:


mlpc = MLPClassifier(hidden_layer_sizes=(11,11,11), max_iter=10000) #500 originally but did not converge
mlpc.fit(X_train, y_train)
pred_mlpc = mlpc.predict(X_test)


# In[23]:


print(classification_report(y_test, pred_mlpc))
print(confusion_matrix(y_test, pred_mlpc))


# # Accuracy Scores

# In[24]:


from sklearn.metrics import accuracy_score
cm = accuracy_score(y_test, pred_dtc)
cm


# In[25]:


from sklearn.metrics import accuracy_score
cm = accuracy_score(y_test, pred_rfc)
cm


# In[26]:


from sklearn.metrics import accuracy_score
cm = accuracy_score(y_test, pred_svmc)
cm


# In[27]:


from sklearn.metrics import accuracy_score
cm = accuracy_score(y_test, pred_mlpc)
cm


# # Single sentence predictors

# In[40]:


user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = dtc.predict(data)
if output == 1:
    print('spam')
else: 
    print ('ham')


# In[33]:


user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = rfc.predict(data)
if output == 1:
    print('spam')
else: 
    print ('ham')


# In[31]:


user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = rfc.predict(data)
if output == 1:
    print('spam')
else: 
    print ('ham')


# In[35]:


user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = svmc.predict(data)
if output == 1:
    print('spam')
else: 
    print ('ham')


# In[39]:


user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = mlpc.predict(data)
if output == 1:
    print('spam')
else: 
    print ('ham')

