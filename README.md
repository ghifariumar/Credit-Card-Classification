# Credit-Card-Classification

### Import Packages
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')
```

Dataset Source : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

```python
df = pd.read_csv('creditcard.csv')
df
```

<img width="578" alt="image" src="https://user-images.githubusercontent.com/99155979/197744916-51c787f6-0b1d-4739-8fe4-539f0031688c.png">
<img width="464" alt="image" src="https://user-images.githubusercontent.com/99155979/197745038-bf5842ed-9d6f-4d84-9b4d-95ef213962ef.png">

### Check Imbalance Dataset
```python
pd.crosstab(index=df['Class'], columns='count', normalize = True)*100
```
<img width="89" alt="image" src="https://user-images.githubusercontent.com/99155979/197747515-0e8aa978-c64b-437d-af89-ea6395417b83.png">

### Create Model to Detect Fraud
- 1 -> Fraud
- 0 -> Non-Fraud

```python
plt.figure(figsize=(13,8))
sns.heatmap(df.isna(),cmap = 'viridis', cbar = False, yticklabels=False)
plt.show()
```
![download](https://user-images.githubusercontent.com/99155979/197747995-043b6243-1ab0-4e74-b746-188f17547d21.png)

##### Splitting Data
```python
X = df.drop(columns='Class')
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X,y, stratify = y,train_size = .80, random_state = 42)

from sklearn.linear_model import LogisticRegression
```

##### Evaluation Matrix
```python
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score, classification_report, confusion_matrix
```

##### Machine Learning Modelling
```python
LR = LogisticRegression()
LR.fit(X_train, y_train)
y_predLR = LR.predict(X_test)
y_trainLR = LR.predict(X_train)
```
##### Train
```python
print(classification_report(y_train,y_trainLR))
```
<img width="308" alt="image" src="https://user-images.githubusercontent.com/99155979/197750460-ce294bd5-5145-4dcc-bde3-ea423eb179e1.png">

##### Test
```python
print(classification_report(y_test,y_predLR))
```
<img width="305" alt="image" src="https://user-images.githubusercontent.com/99155979/197750542-84ac1a3e-14bd-47cb-964e-551936636d27.png">

```python
accuracy_score(y_test,y_predLR)
```
0.9989291106351603

```python
cm_LR_ts = confusion_matrix(y_test, y_predLR, labels=[1,0])
df_LR_ts = pd.DataFrame(cm_LR_ts, index=['Akt 1','Akt 0'], columns = ['Pred 1','Pred 0'])
df_LR_ts
```
<img width="113" alt="image" src="https://user-images.githubusercontent.com/99155979/197751607-d5813748-ede5-464f-98e4-4a5a82a3c71d.png">

```python
sns.heatmap(df_LR_ts, annot=True, cbar=False)
plt.show()
```
![download](https://user-images.githubusercontent.com/99155979/197751688-71a057b4-57c3-49f5-94fc-18793c45cfb2.png)

```python
cm_LR_tr = confusion_matrix(y_train, y_trainLR, labels=[1,0])
df_LR_tr = pd.DataFrame(cm_LR_tr, index=['Akt 1','Akt 0'], columns = ['Pred 1','Pred 0'])
df_LR_tr
```
<img width="113" alt="image" src="https://user-images.githubusercontent.com/99155979/197912340-7f80d9c8-be99-4475-89ff-c320f3c5db46.png">

```python
sns.heatmap(df_LR_tr, annot=True, cbar=False)
plt.show()
```
![download](https://user-images.githubusercontent.com/99155979/197912381-05dd44d9-1b57-4472-af64-768c57ce42eb.png)

###### Handling Imbalance Data
- random sampling and smote only used for train data (X_train - y_train)

###### Random Sampling
```python
df_train  =pd.concat([X_train, y_train], axis=1)
df_train['Class'].value_counts()

0    227451
1       394
Name: Class, dtype: int64


non_fraud = df_train[df_train['Class'] == 0] ## kelas majority
fraud = df_train[df_train['Class']==1] ## kelas minority
```

##### Random Oversampling
- **Duplicating dara randomly** class-target minority (class 1) until it has the same amount with class-target majority (class 0).
- **Fraud** dataframe will be over sampling until it has the same amount with **Non Fraud** dataframe 

```python
from sklearn.utils import resample
fraud_oversample = resample(fraud, replace=True, n_samples=len(non_fraud), random_state = 42)
df_Oversample = pd.concat([non_fraud,fraud_oversample])
df_Oversample['Class'].value_counts()

0    227451
1    227451
Name: Class, dtype: int64

X_train_OS = df_Oversample.drop(columns='Class')
y_train_OS = df_Oversample['Class']
LR_OS = LogisticRegression()
LR_OS.fit(X_train_OS, y_train_OS)
y_predOS = LR_OS.predict(X_test)

print(classification_report(y_test,y_predOS))
              precision    recall  f1-score   support

           0       1.00      0.95      0.98     56864
           1       0.03      0.91      0.06        98

    accuracy                           0.95     56962
   macro avg       0.52      0.93      0.52     56962
weighted avg       1.00      0.95      0.97     56962

cm_OS = confusion_matrix(y_test, y_predOS, labels=[1,0])
df_OS = pd.DataFrame(cm_OS, index=['Akt 1', 'Akt 0'], columns = ['Pred 1', 'Pred 0'])
df_OS
```
<img width="107" alt="image" src="https://user-images.githubusercontent.com/99155979/197913134-38fce3a4-0411-4772-aeb4-7830bd7badc8.png">

```python
sns.heatmap(df_OS, annot=True, cbar=False)
plt.show()
```
![download](https://user-images.githubusercontent.com/99155979/197913283-5bbedd07-5da0-4c15-ac02-b62415cb7e06.png)

##### Random Under Sampling
- **Remove data randomly** in majority class(clas 0) until it has the same amount with (class 1)
- **Non Fraud** data frame will be undersampling until it has the same amount with **Fraud** dataframe
-  Under Sampling rarely to be used. It has a chance to lose some information 

```python
df_train['Class'].value_counts()
0    227451
1       394
Name: Class, dtype: int64

non_fraud = df_train[df_train['Class']==0] ## majority class
fraud = df_train[df_train['Class']==1] ## minority class

non_fraud_Undersample = resample(non_fraud, # majority class
                                 replace=False,
                                 n_samples=len(fraud), #minority class
                                 random_state = 42)
                                 
df_Undersample = pd.concat([non_fraud_Undersample, fraud])
df_Undersample['Class'].value_counts()
0    394
1    394
Name: Class, dtype: int64

X_train_US = df_Undersample.drop(columns='Class')
y_train_US = df_Undersample['Class']
LR_US = LogisticRegression()
LR_US.fit(X_train_US, y_train_US)
y_predUS = LR_US.predict(X_test)

print(classification_report(y_test,y_predUS))
              precision    recall  f1-score   support

           0       1.00      0.96      0.98     56864
           1       0.04      0.92      0.07        98

    accuracy                           0.96     56962
   macro avg       0.52      0.94      0.53     56962
weighted avg       1.00      0.96      0.98     56962

cm_US = confusion_matrix(y_test, y_predUS, labels=[1,0])
df_US = pd.DataFrame(cm_US, index=['Akt 1', 'Akt 0'], columns = ['Pred 1','Pred 0'])
df_US
```
<img width="112" alt="image" src="https://user-images.githubusercontent.com/99155979/197913702-5e93fd24-6820-4de4-8f64-4e20677e357a.png">

```python
sns.heatmap(df_US, annot=True, cbar=False)
plt.show()
```
![download](https://user-images.githubusercontent.com/99155979/197913734-d2a2a858-e4d5-4a55-a7af-807a7ff53372.png)

##### SMOTE - Synthetic Minority Oversampling Technique
- **Create Synthetic Randomly** from minoruty class (Class 1), until it has the same amount with majority class (Class 0)

```python
## Install Package Imblearn
import imblearn
conda install -c conda-forge imbalanced-learn==0.6
from imblearn.over_sampling import SMOTE
```

```python
df_train['Class'].value_counts()
0    227451
1       394
Name: Class, dtype: int64
```

```python
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)

## Optional for checking proportion
df_SMOTE = pd.concat([X_train_sm, y_train_sm], axis=1)
```

```python
df_SMOTE['Class'].value_counts()
0    227451
1    227451
Name: Class, dtype: int64
```

```python
LR_SMOTE = LogisticRegression()
LR_SMOTE.fit(X_train_sm, y_train_sm)
y_predSMOTE = LR_SMOTE.predict(X_test)

print(classification_report(y_test, y_predSMOTE))
              precision    recall  f1-score   support

           0       1.00      0.98      0.99     56864
           1       0.07      0.91      0.12        98

    accuracy                           0.98     56962
   macro avg       0.53      0.94      0.56     56962
weighted avg       1.00      0.98      0.99     56962

cm_SMOTE = confusion_matrix(y_test, y_predSMOTE, labels=[1,0])
df_SMOTE = pd.DataFrame(cm_SMOTE, index=['Akt 1', 'Akt 0'], columns=['Pred 1', 'Pred 0'])
df_SMOTE
```
<img width="107" alt="image" src="https://user-images.githubusercontent.com/99155979/197917670-8663bdc3-fb91-4a4c-81e9-e6f981389cba.png">

```python
sns.heatmap(df_SMOTE, annot=True, cbar=False)
plt.show()
```
![download](https://user-images.githubusercontent.com/99155979/197917802-6de32a7b-ea4f-41d4-926c-3944b4d223c5.png)

##### Class Weight
- With class weighting, no need to use SMOTE or random sampling
- Using the first X_train and  y_train
- Focusing on learning the model
- Total Class Weight = 1

```python
LR_CW = LogisticRegression(class_weight={0:0.3, 1:.97})
LR_CW.fit(X_train, y_train)
y_predCW = LR_CW.predict(X_test)
```

```python
print(classification_report(y_test, y_predCW))
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.66      0.83      0.73        98

    accuracy                           1.00     56962
   macro avg       0.83      0.91      0.87     56962
weighted avg       1.00      1.00      1.00     56962
```

```python
cm_CW = confusion_matrix(y_test, y_predCW, labels=[1,0])
df_CW = pd.DataFrame(cm_CW, index=['Akt 1', 'Akt 0'], columns = ['Pred 1','Pred 0'])
df_CW
```
<img width="113" alt="image" src="https://user-images.githubusercontent.com/99155979/197917997-7584207a-1d5c-496e-85cf-a21766356620.png">

```python
sns.heatmap(df_CW, annot=True, cbar=False)
plt.show()
```
![download](https://user-images.githubusercontent.com/99155979/197918021-dc13deb2-f8c2-4fdd-ab22-04b9b02f264f.png)

```python
LR_CW2 = LogisticRegression(class_weight={0:.10, 1:.90})
LR_CW2.fit(X_train, y_train)
y_predCW2 = LR_CW2.predict(X_test)
```

```python
print(classification_report(y_test,y_predCW2))
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.42      0.84      0.56        98

    accuracy                           1.00     56962
   macro avg       0.71      0.92      0.78     56962
weighted avg       1.00      1.00      1.00     56962
```

```python
cm_CW2 = confusion_matrix(y_test, y_predCW2, labels=[1,0])
df_CW2 = pd.DataFrame(cm_CW2, index=['Akt 1', 'Akt 0'], columns = ['Pred 1', 'Pred 0'])
df_CW2
```
<img width="108" alt="image" src="https://user-images.githubusercontent.com/99155979/197918205-ae0b2780-4c93-4710-afac-738827140eae.png">

##### Improvement
```python
from sklearn.svm import SVC
SVM_1 =SVC()
SVM_1.fit(X_train, y_train)
y_predSVM = SVM_1.predict(X_test)
```

```python
print(classification_report(y_test, y_predSVM))
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.00      0.00      0.00        98

    accuracy                           1.00     56962
   macro avg       0.50      0.50      0.50     56962
weighted avg       1.00      1.00      1.00     56962
```

```python
cm_svm = confusion_matrix(y_test, y_predSVM, labels=[1,0])
df_svm = pd.DataFrame(cm_svm, index=['Akt 1', 'Akt 0'], columns = ['Pred 1', 'Pred 0'])
df_svm
```
<img width="107" alt="image" src="https://user-images.githubusercontent.com/99155979/197918446-e767f9d1-cfa2-4100-b41c-9aa58ef4f078.png">

##### Optimize Model
```python
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
std.fit(X_train)
X_train_sc = std.transform(X_train)
X_test_sc = std.transform(X_test)
SVM_2 = SVC(max_iter=400)
SVM_2.fit(X_train_sc, y_train)
y_SVM = SVM_2.predict(X_test_sc)
```

```python
print(classification_report(y_test,y_SVM))
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.96      0.76      0.85        98

    accuracy                           1.00     56962
   macro avg       0.98      0.88      0.92     56962
weighted avg       1.00      1.00      1.00     56962
```

```python
cm_svm2 = confusion_matrix(y_test, y_SVM, labels=[1,0])
df_svm2 = pd.DataFrame(cm_svm2, index=['Akt 1', 'Akt 0'], columns = ['Pred 1', 'Pred 0'])
df_svm2
```
<img width="113" alt="image" src="https://user-images.githubusercontent.com/99155979/197919060-04f4ebbe-cd40-43aa-a4e9-24101d00695d.png">

```python
sns.heatmap(df_svm2, annot=True, cbar=False)
plt.show()
```
![download](https://user-images.githubusercontent.com/99155979/197919098-2bd8083c-c8ac-4152-8a21-f4121cb42f51.png)

