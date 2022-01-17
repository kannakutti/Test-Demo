import pandas as pd
import os
os.chdir("/tmp")
import matplotlib.pyplot as plt
import seaborn as sns

Patient_Data_df = pd.read_csv("PB_Data.csv")

Patient_Data_df.head()

Patient_Data_df.shape

Patient_Data_df.info()

Patient_Data_df.isnull().any()

# Converting "ScheduledDay" and "AppointmentDay" columns into date-time
Patient_Data_df['ScheduledDay'] = pd.to_datetime(Patient_Data_df['ScheduledDay'])#Converting TimeStamp attribute to of type Datetime
Patient_Data_df['AppointmentDay'] = pd.to_datetime(Patient_Data_df['AppointmentDay'])
Patient_Data_df['Date Of Birth'] = pd.to_datetime(Patient_Data_df['Date Of Birth'])
Patient_Data_df[['ScheduledDay','AppointmentDay','Date Of Birth']].dtypes


#Deriving the Age of the patient from the Date of Birth
#Commenting Below Line
#Patient_Data_df["Age"] = round(((Patient_Data_df["AppointmentDay"]-Patient_Data_df["Date Of Birth"]).dt.days)/365).astype("int")
Patient_Data_df.head()

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

#Checking how the transformed dates look like
Patient_Data_df['AppointmentTime'].head()

#Validating whether there are any duplicated rows
#Patient_Data_df[Patient_Data_df["PatientId"].duplicated()]
#There are no duplicated rows

#Verifying if same patientId is given to the same patient or different patients
#Patient_Data_df[Patient_Data_df.PatientId == 55639]
# It is evident that same patientId is given to multiple patients. Hence, we should also consider First Name and Last Name attributes for the model training

#We could see from the above results that Appointment time is unique. Hence checking how many unique values the AppointmentTime
#variable has
Patient_Data_df.AppointmentTime.nunique()

# droping `AppointmentTime` column as there is only one unique value for Appointment time.
Patient_Data_df.drop(columns='AppointmentTime',inplace=True)

Patient_Data_df.head()

#Checking if there is any observation with appointment day < scheduled day? if True WaitingDays will be negative and this is impossible
appointment_error = Patient_Data_df[Patient_Data_df.WaitingDays < 0 ]
appointment_error

#We should remove the data that has waiting days less than zero.
Patient_Data_df = Patient_Data_df[~(Patient_Data_df.WaitingDays <0)]

# # Exploratory Data Analysis
# Analyzing how many appointments have no-show as "yes" and how many have "no"
#

sns.countplot(Patient_Data_df['No-show'], palette='RdPu')
plt.title("No-Show Vs Show")
plt.show()


# Genderwise Analysis for No-Show
#
# SMS_Received Yes Vs No analysis for No-Show

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
sns.countplot(x='Gender', data=Patient_Data_df, hue='No-show', ax=ax1, palette='RdPu');
sns.countplot(x='SMS_received', data=Patient_Data_df, hue='No-show', ax=ax2, palette='RdPu');
fig.set_figwidth(12)
fig.set_figheight(4)


# Females have more appointments than Men.
# No-Show ratio is higher in case of SMS_received than that of SMS Not received group

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True)
sns.countplot(x='Hipertension', data=Patient_Data_df, hue='No-show', ax=ax1, palette='BuGn');
sns.countplot(x='Diabetes', data=Patient_Data_df, hue='No-show', ax=ax2, palette='RdYlGn_r');
sns.countplot(x='Alcoholism', data=Patient_Data_df, hue='No-show', ax=ax3, palette='spring_r');
fig.set_figwidth(12)
fig.set_figheight(4)


# Hypertension patients are more than Diabetes Patients
# From the not showing group, Diabetes patients do not show for the appointment much than Hypertension Patients
# identifying during which month there are more appointments

sns.countplot(x='AppointmentMonth', data=Patient_Data_df, hue='No-show', palette='ocean_r')
fig.set_figwidth(15)
fig.set_figheight(5)

# During the May month there are more appointments
# Identifying which week of the day has more appointments

sns.countplot(x='AppointmentWeekDay', data=Patient_Data_df, hue='No-show', palette='ocean_r')
fig.set_figwidth(15)
fig.set_figheight(5)


# There are more appointments on Tuesdays and Wednesdays
# Identifying which hour of the day there are more appointments scheduled.

#Combining the First Name and Last Name to make Full Name
Patient_Data_df["Full Name"] = Patient_Data_df["First Name"]+Patient_Data_df["Last Name"]

#Ordinal Encoding the full namekmk
from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()
Patient_Data_df["Full Name Code"] = ord_enc.fit_transform(Patient_Data_df[["Full Name"]])
Patient_Data_df["Neighbourhood Code"] = ord_enc.fit_transform(Patient_Data_df[["Neighbourhood"]])
Patient_Data_df[["Full Name", "Full Name Code","Neighbourhood","Neighbourhood Code"]].head(11)

#Verifying if the same Encoding value applied to the Patient having more than one appointment.
Patient_Data_df[Patient_Data_df["Full Name"]== "MattiePoquette"]

y = Patient_Data_df["No-show"]
Patient_Data_df = pd.get_dummies(Patient_Data_df,columns=["Gender",'AppointmentWeekDay', 'AppointmentMonth'],prefix=['Gender', 'AppointmentWeekDay', 'AppointmentMonth'])
X = Patient_Data_df.drop(["No-show","Address","First Name","Full Name","Last Name","Neighbourhood","AppointmentID","Date Of Birth","ScheduledDay","AppointmentDay","Zip Code"],axis =1)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)

my_df = pd.DataFrame()
my_df["Predicted Outcome"]=y_train
my_df.to_csv("Prediction.csv")

# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

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

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# # Naive Bayes Classifier

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_nb_pred = gnb.predict(X_test)

#Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_nb_pred))
#metrics.accuracy_score(y_test, y_nb_pred).to_csv("ok1")

from sklearn.metrics import classification_report
print(classification_report(y_nb_pred,y_test))
