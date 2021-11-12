#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import os
#os.chdir("/tmp")


# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[20]:


Patient_Data_df = pd.read_csv("/PB_Data.csv")


# In[21]:


Patient_Data_df.head()


# In[22]:


Patient_Data_df.shape


# There are 1,10,527 appointments with associated 19 variables

# In[23]:


Patient_Data_df.info()


# In[24]:


Patient_Data_df.isnull().any()


# In[25]:


# Converting "ScheduledDay" and "AppointmentDay" columns into date-time
Patient_Data_df['ScheduledDay'] = pd.to_datetime(Patient_Data_df['ScheduledDay'])#Converting TimeStamp attribute to of type Datetime
Patient_Data_df['AppointmentDay'] = pd.to_datetime(Patient_Data_df['AppointmentDay'])
Patient_Data_df['Date Of Birth'] = pd.to_datetime(Patient_Data_df['Date Of Birth'])
Patient_Data_df[['ScheduledDay','AppointmentDay','Date Of Birth']].dtypes


# In[26]:


#Deriving the Age of the patient from the Date of Birth
#Commenting Below Line
#Patient_Data_df["Age"] = round(((Patient_Data_df["AppointmentDay"]-Patient_Data_df["Date Of Birth"]).dt.days)/365).astype("int")
Patient_Data_df.head()


# In[27]:


# Get time columns from date-time columns
Patient_Data_df['ScheduledHour'] = pd.to_datetime(Patient_Data_df.ScheduledDay).dt.hour
Patient_Data_df['AppointmentTime'] = pd.to_datetime(Patient_Data_df.AppointmentDay).dt.time

# Convert time included date-time columns to only date columns
Patient_Data_df['ScheduledDay'] = Patient_Data_df['ScheduledDay'].dt.date
Patient_Data_df['AppointmentDay'] = Patient_Data_df['AppointmentDay'].dt.date

# Get month and week of day information from the Appointment date
Patient_Data_df['AppointmentWeekDay'] = pd.to_datetime(Patient_Data_df.AppointmentDay).dt.day_name()
Patient_Data_df['AppointmentMonth'] = pd.to_datetime(Patient_Data_df.AppointmentDay).dt.month_name()

# Calculating no. of waiting days before appointment
Patient_Data_df['WaitingDays'] = Patient_Data_df.AppointmentDay - Patient_Data_df.ScheduledDay

# Convert datatype to int
Patient_Data_df.WaitingDays = Patient_Data_df.WaitingDays.astype('str')
Patient_Data_df.WaitingDays = Patient_Data_df.WaitingDays.apply(lambda x: x.split()[0])
Patient_Data_df.WaitingDays = Patient_Data_df.WaitingDays.astype('int64')


# In[28]:


#Checking how the transformed dates look like
Patient_Data_df['AppointmentTime'].head()


# In[29]:


#Validating whether there are any duplicated rows
Patient_Data_df[Patient_Data_df["PatientId"].duplicated()]
#There are no duplicated rows


# In[30]:


#Verifying if same patientId is given to the same patient or different patients
Patient_Data_df[Patient_Data_df.PatientId == 55639]


# It is evident that same patientId is given to multiple patients. Hence, we should also consider First Name and Last Name attributes for the model training

# In[31]:


#Checking if there is any Age less than zero or zero
#Commenting below line
#len(Patient_Data_df[Patient_Data_df["Age"] == 0]),len(Patient_Data_df[Patient_Data_df["Age"] < 0])


# There are 999 observations where Age is equal to zero and 5161 observations where Age is less than zero. These observatins look weird
#

# In[32]:


#We could see from the above results that Appointment time is unique. Hence checking how many unique values the AppointmentTime
#variable has
Patient_Data_df.AppointmentTime.nunique()


# In[33]:


# droping `AppointmentTime` column as there is only one unique value for Appointment time.
Patient_Data_df.drop(columns='AppointmentTime',inplace=True)


# In[34]:


Patient_Data_df.head()


# In[35]:


#Checking if there is any observation with appointment day < scheduled day? if True WaitingDays will be negative and this is impossible
appointment_error = Patient_Data_df[Patient_Data_df.WaitingDays < 0 ]
appointment_error


# In[36]:


#We should remove the data that has waiting days less than zero.
Patient_Data_df = Patient_Data_df[~(Patient_Data_df.WaitingDays <0)]


# # Exploratory Data Analysis

# Analyzing how many appointments have no-show as "yes" and how many have "no"
#

# In[37]:


sns.countplot(Patient_Data_df['No-show'], palette='RdPu')
plt.title("No-Show Vs Show")
plt.show()


# Genderwise Analysis for No-Show
#
# SMS_Received Yes Vs No analysis for No-Show

# In[38]:


fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
sns.countplot(x='Gender', data=Patient_Data_df, hue='No-show', ax=ax1, palette='RdPu');
sns.countplot(x='SMS_received', data=Patient_Data_df, hue='No-show', ax=ax2, palette='RdPu');
fig.set_figwidth(12)
fig.set_figheight(4)


# Females have more appointments than Men.
#
# No-Show ratio is higher in case of SMS_received than that of SMS Not received group

# In[39]:


fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True)
sns.countplot(x='Hipertension', data=Patient_Data_df, hue='No-show', ax=ax1, palette='BuGn');
sns.countplot(x='Diabetes', data=Patient_Data_df, hue='No-show', ax=ax2, palette='RdYlGn_r');
sns.countplot(x='Alcoholism', data=Patient_Data_df, hue='No-show', ax=ax3, palette='spring_r');
fig.set_figwidth(12)
fig.set_figheight(4)


# Hypertension patients are more than Diabetes Patients
#
# From the not showing group, Diabetes patients do not show for the appointment much than Hypertension Patients
#

# identifying during which month there are more appointments

# In[40]:


sns.countplot(x='AppointmentMonth', data=Patient_Data_df, hue='No-show', palette='ocean_r')
fig.set_figwidth(15)
fig.set_figheight(5)


# During the May month there are more appointments

# Identifying which week of the day has more appointments

# In[41]:


sns.countplot(x='AppointmentWeekDay', data=Patient_Data_df, hue='No-show', palette='ocean_r')
fig.set_figwidth(15)
fig.set_figheight(5)


# There are more appointments on Tuesdays and Wednesdays

# Identifying which hour of the day there are more appointments scheduled.

# In[42]:


#Combining the First Name and Last Name to make Full Name
Patient_Data_df["Full Name"] = Patient_Data_df["First Name"]+Patient_Data_df["Last Name"]


# In[43]:


#Ordinal Encoding the full namekmk
from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()
Patient_Data_df["Full Name Code"] = ord_enc.fit_transform(Patient_Data_df[["Full Name"]])
Patient_Data_df["Neighbourhood Code"] = ord_enc.fit_transform(Patient_Data_df[["Neighbourhood"]])
Patient_Data_df[["Full Name", "Full Name Code","Neighbourhood","Neighbourhood Code"]].head(11)


# In[142]:


#Patient_Data_df.groupby(["Full Name"])["Full Name"].count().sort_values(ascending=False).head(5)


# In[44]:


#Verifying if the same Encoding value applied to the Patient having more than one appointment.
Patient_Data_df[Patient_Data_df["Full Name"]== "MattiePoquette"]


# In[45]:


y = Patient_Data_df["No-show"]
Patient_Data_df = pd.get_dummies(Patient_Data_df,columns=["Gender",'AppointmentWeekDay', 'AppointmentMonth'],prefix=['Gender', 'AppointmentWeekDay', 'AppointmentMonth'])
#X = Patient_Data_df.drop(["No-show","Address","First Name","Full Name","Last Name","Neighbourhood","Card Holder Id","AppointmentID","Date Of Birth","ScheduledDay","AppointmentDay","Zip Code"],axis =1)
X = Patient_Data_df.drop(["No-show","Address","First Name","Full Name","Last Name","Neighbourhood","AppointmentID","Date Of Birth","ScheduledDay","AppointmentDay","Zip Code"],axis =1)


# In[46]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
y


# In[47]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[48]:


from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)


# In[49]:


# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[50]:


# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

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


# In[51]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#print("Precision:",metrics.precision_score(y_test, y_pred))
#print("Recall:",metrics.recall_score(y_test, y_pred))


# # Naive Bayes Classifier

# In[52]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_nb_pred = gnb.predict(X_test)


# In[53]:


#Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_nb_pred))


# In[54]:


from sklearn.metrics import classification_report
print(classification_report(y_nb_pred,y_test))


# In[ ]:





