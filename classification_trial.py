import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier




df_train = pd.read_csv("train.csv")

df_train['Work_Experience'].fillna(df_train['Work_Experience'].mode()[0], inplace=True)
df_train['Var_1'].fillna(df_train['Var_1'].mode()[0], inplace=True)
df_train['Gender'].fillna(df_train['Gender'].mode()[0], inplace=True)
df_train['Ever_Married'].fillna(df_train['Ever_Married'].mode()[0], inplace=True)
df_train['Age'].fillna(df_train['Age'].mode()[0], inplace=True)
df_train['Family_Size'].fillna(df_train['Family_Size'].mode()[0], inplace=True)
df_train['Spending_Score'].fillna(df_train['Spending_Score'].mode()[0], inplace=True)
df_train['Profession'].fillna(df_train['Profession'].mode()[0], inplace=True)
df_train['Segmentation'].fillna(df_train['Segmentation'].mode()[0], inplace=True)
df_train['Graduated'].fillna(df_train['Graduated'].mode()[0], inplace=True)


newdf_train2 = df_train.drop_duplicates()

CounterA = 0
CounterB = 0
CounterC = 0
CounterD = 0

for i in range(df_train.shape[0]):
    if df_train.at[i, 'Segmentation'] == 'A':
        CounterA+=1
    elif df_train.at[i, 'Segmentation'] == 'B':
        CounterB+=1
    elif df_train.at[i, 'Segmentation'] == 'C':
        CounterC+=1
    elif df_train.at[i, 'Segmentation'] == 'D':
        CounterD+=1
CounterA = (CounterA / df_train.shape[0])*100
CounterB = (CounterB / df_train.shape[0])*100
CounterC = (CounterC / df_train.shape[0])*100
CounterD = (CounterD / df_train.shape[0])*100
print(CounterA)
print(CounterB)
print(CounterC)
print(CounterD)
LE = LabelEncoder()
newdf_train2['Spending_Score'] = LE.fit_transform(newdf_train2['Spending_Score'])
newdf_train2['Var_1'] = LE.fit_transform(newdf_train2['Var_1'])

output_LE = LabelEncoder()
newdf_train2['Segmentation'] = output_LE.fit_transform(newdf_train2['Segmentation'])

Gender_train = pd.get_dummies(newdf_train2['Gender'])
Married_train = pd.get_dummies(newdf_train2['Ever_Married'])
Graduated_train = pd.get_dummies(newdf_train2['Graduated'])
Profession_train = pd.get_dummies(newdf_train2['Profession'])
newdf_train2.drop(['Gender', 'Ever_Married', 'Graduated', 'Profession'], axis=1, inplace=True)
newdf_train3 = pd.concat([newdf_train2, Gender_train, Married_train, Graduated_train, Profession_train], axis=1)
newdf_train3.reset_index(inplace=True)
newdf_train3.drop(['index'], axis=1, inplace=True)
X = newdf_train3.drop(['ID', 'Segmentation'], axis=1)
y = newdf_train3['Segmentation']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=A, random_state=20, stratify=y)
stand = StandardScaler().fit(X_train)
X_train_stand = stand.transform(X_train)
X_test_stand = stand.transform(X_test)
norm = MinMaxScaler().fit(X_train)
X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train_norm, y_train)
y_pred_decision_tree = decision_tree.predict(X_test_norm)
logisticregression = LogisticRegression()
ovo = OneVsOneClassifier(logisticregression)
ovo.fit(X_train_stand, y_train)
y_pred_logistic_regression = ovo.predict(X_test_stand)
random_forest = RandomForestClassifier()
random_forest.fit(X_train_norm, y_train)
y_pred_random_forest = random_forest.predict(X_test_norm)
KNN = KNeighborsClassifier()
KNN.fit(X_train_norm, y_train)
y_pred_KNN = KNN.predict(X_test_norm)
SVM = SVC()
SVM.fit(X_train_stand, y_train)
y_pred_SVM = SVM.predict(X_test_stand)
Naive_Bayes = GaussianNB()
Naive_Bayes.fit(X_train_stand, y_train)
y_pred_naive_bayes = Naive_Bayes.predict(X_test_stand)
XGboost = XGBClassifier().fit(X_train_norm, y_train)
y_pred_XGboost = XGboost.predict(X_test_norm)
CatBoost = CatBoostClassifier().fit(X_train_norm, y_train)
y_pred_CatBoost = CatBoost.predict(X_test_norm)
LightGBM = LGBMClassifier().fit(X_train_norm, y_train)
y_pred_LightGBM = LightGBM.predict(X_test_norm)
HGB = HistGradientBoostingClassifier().fit(X_train_norm, y_train)
y_pred_HGB = HGB.predict(X_test_norm)
AdaBoost = AdaBoostClassifier().fit(X_train_norm, y_train)
y_pred_AdaBoost = AdaBoost.predict(X_test_norm)
Bagging = BaggingClassifier(base_estimator=KNN).fit(X_train_norm, y_train)
y_pred_Bagging = Bagging.predict(X_test_norm)
print("Bagging Accuracy is {}".format(accuracy_score(y_test, y_pred_Bagging)))
print("AdaBoost Accuracy is {}".format(accuracy_score(y_test, y_pred_AdaBoost)))
print("HGB Accuracy is {}".format(accuracy_score(y_test, y_pred_HGB)))
print("LGBM Accuracy is {}".format(accuracy_score(y_test, y_pred_LightGBM)))
print("CB Accuracy is {}".format(accuracy_score(y_test, y_pred_CatBoost)))
print("XGB Accuracy is {}".format(accuracy_score(y_test, y_pred_XGboost)))
print("DT Accuracy is {}".format(accuracy_score(y_test, y_pred_decision_tree)))
print("RF Accuracy is {}".format(accuracy_score(y_test, y_pred_random_forest)))
print("KNN Accuracy is {}".format(accuracy_score(y_test, y_pred_KNN)))
print("SVM Accuracy is {}".format(accuracy_score(y_test, y_pred_SVM)))
print("NB Accuracy is {}".format(accuracy_score(y_test, y_pred_naive_bayes)))
print("LR Accuracy is {}".format(accuracy_score(y_test, y_pred_logistic_regression)))
cm_DT = confusion_matrix(y_test, y_pred_decision_tree)
cm_RF = confusion_matrix(y_test, y_pred_random_forest)
cm_KNN = confusion_matrix(y_test, y_pred_KNN)
cm_SVM = confusion_matrix(y_test, y_pred_SVM)
cm_NB = confusion_matrix(y_test, y_pred_naive_bayes)
cm_LR = confusion_matrix(y_test, y_pred_logistic_regression)
cm_XGB = confusion_matrix(y_test, y_pred_XGboost)
cm_CB = confusion_matrix(y_test, y_pred_CatBoost)
cm_LGBM = confusion_matrix(y_test, y_pred_LightGBM)
cm_HGB = confusion_matrix(y_test, y_pred_HGB)
cm_AdaBoost = confusion_matrix(y_test, y_pred_AdaBoost)
cm_Bagging = confusion_matrix(y_test, y_pred_Bagging)
print('DT Confusion Matrix: \n', cm_DT)
print('RF Confusion Matrix: \n', cm_RF)
print('KNN Confusion Matrix: \n', cm_KNN)
print('SVM Confusion Matrix: \n', cm_SVM)
print('NB Confusion Matrix: \n', cm_NB)
print('LR Confusion Matrix: \n', cm_LR)
print('XGB Confusion Matrix: \n', cm_XGB)
print('CB Confusion Matrix: \n', cm_CB)
print('LGBM Confusion Matrix: \n', cm_LGBM)
print('HGB Confusion Matrix: \n', cm_HGB)
print('AdaBoost Confusion Matrix: \n', cm_AdaBoost)
print('Bagging Confusion Matrix: \n', cm_Bagging)




