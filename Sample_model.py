#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
#os.chdir("Vinayak")


# In[170]:


#Patient_Data_df = pd.read_csv("Patient_Basic_Data.csv")
#Patient_Data_df = pd.read_csv("Our_Data.csv")
Patient_Data_df = pd.read_csv("PB_Data.csv")


# In[171]:


Patient_Data_df.head()


# In[172]:


Patient_Data_df.shape


# In[173]:


Patient_Data_df.info()


# In[174]:


Patient_Data_df.isnull().any()


# In[175]:


Patient_Data_df['ScheduledDay'] = pd.to_datetime(Patient_Data_df['ScheduledDay'])#Converting TimeStamp attribute to of type Datetime
Patient_Data_df['AppointmentDay'] = pd.to_datetime(Patient_Data_df['AppointmentDay'])
#Patient_Data_df['Date Of Birth'] = pd.to_datetime(Patient_Data_df['Date Of Birth'])


# In[176]:


#Validating whether there are any duplicated rows
Patient_Data_df[Patient_Data_df["PatientId"].duplicated()]
#There are no duplicated rows


# In[177]:


#Verifying if same patientId is given to the same patient or different patients
Patient_Data_df[Patient_Data_df.PatientId == 55639]


# It is evident that same patientId is given to multiple patients. Hence, we should also consider First Name and Last Name attributes for the model training

# In[178]:


#Deriving the Age of the patient from the Date of Birth
#Patient_Data_df["Age"] = round(((Patient_Data_df["AppointmentDay"]-Patient_Data_df["Date Of Birth"]).dt.days)/365).astype("int")
Patient_Data_df.head()


# In[179]:


#Converting Date columns to Numeric by splitting them into Year, Month and Day
Patient_Data_df['Sch Year'] = Patient_Data_df['ScheduledDay'].dt.year
Patient_Data_df['Sch Month'] = Patient_Data_df['ScheduledDay'].dt.month
Patient_Data_df['Sch DOM'] = Patient_Data_df['ScheduledDay'].dt.day
Patient_Data_df["Sch DOW"] = Patient_Data_df['ScheduledDay'].dt.dayofweek
Patient_Data_df['Sch Hour'] = Patient_Data_df['ScheduledDay'].dt.hour

Patient_Data_df['Apmnt Year'] = Patient_Data_df['AppointmentDay'].dt.year
Patient_Data_df['Apmnt Month'] = Patient_Data_df['AppointmentDay'].dt.month
Patient_Data_df['Apmnt DOM'] = Patient_Data_df['AppointmentDay'].dt.day
Patient_Data_df["Apmnt DOW"] = Patient_Data_df['AppointmentDay'].dt.dayofweek


# In[180]:


Patient_Data_df.head(10)


# In[181]:


#Combining the First Name and Last Name to make Full Name
Patient_Data_df["Full Name"] = Patient_Data_df["First Name"]+Patient_Data_df["Last Name"]


# In[182]:


#Ordinal Encoding the full namekmk
from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()
Patient_Data_df["Full Name Code"] = ord_enc.fit_transform(Patient_Data_df[["Full Name"]])
Patient_Data_df["Neighbourhood Code"] = ord_enc.fit_transform(Patient_Data_df[["Neighbourhood"]])
Patient_Data_df[["Full Name", "Full Name Code","Neighbourhood","Neighbourhood Code"]].head(11)


# In[183]:


Patient_Data_df.groupby(["Full Name"])["Full Name"].count().sort_values(ascending=False).head(5)


# In[184]:


#Verifying if the same Encoding value applied to the Patient having more than one appointment.
Patient_Data_df[Patient_Data_df["Full Name"]== "MattiePoquette"]


# In[185]:


#Converting Gender to Binary Encoding using dummies
pd.get_dummies(Patient_Data_df,columns=["Gender"])


# In[190]:


y = Patient_Data_df["No-show"]
X = Patient_Data_df.drop(["No-show","Address","First Name","Full Name","Last Name","Neighbourhood","Gender","AppointmentID","Date Of Birth","ScheduledDay","AppointmentDay","Zip Code"],axis =1)


# In[191]:


X.head()


# In[192]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[193]:


from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)


# In[194]:


# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[195]:


# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
#get_ipython().run_line_magic('matplotlib', 'inline')

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[196]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#print("Precision:",metrics.precision_score(y_test, y_pred))
#print("Recall:",metrics.recall_score(y_test, y_pred))


# # Naive Bayes Classifier

# In[197]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_nb_pred = gnb.predict(X_test)


# In[198]:


#Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_nb_pred))


# In[199]:


y_nb_pred
