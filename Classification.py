import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy.stats import probplot
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

df_train['Work_Experience'].fillna(df_train['Work_Experience'].mode()[0], inplace=True)
df_test['Work_Experience'].fillna(df_test['Work_Experience'].mode()[0], inplace=True)

df_train['Var_1'].fillna(df_train['Var_1'].mode()[0], inplace=True)
df_test['Var_1'].fillna(df_test['Var_1'].mode()[0], inplace=True)

df_train['Gender'].fillna(df_train['Gender'].mode()[0], inplace=True)
df_test['Gender'].fillna(df_test['Gender'].mode()[0], inplace=True)

df_train['Ever_Married'].fillna(df_train['Ever_Married'].mode()[0], inplace=True)
df_test['Ever_Married'].fillna(df_test['Ever_Married'].mode()[0], inplace=True)

df_train['Age'].fillna(df_train['Age'].mode()[0], inplace=True)
df_test['Age'].fillna(df_test['Age'].mode()[0], inplace=True)

df_train['Family_Size'].fillna(df_train['Family_Size'].mode()[0], inplace=True)
df_test['Family_Size'].fillna(df_test['Family_Size'].mode()[0], inplace=True)

df_train['Spending_Score'].fillna(df_train['Spending_Score'].mode()[0], inplace=True)
df_test['Spending_Score'].fillna(df_test['Spending_Score'].mode()[0], inplace=True)

df_train['Profession'].fillna(df_train['Profession'].mode()[0], inplace=True)
df_test['Profession'].fillna(df_test['Profession'].mode()[0], inplace=True)

df_train['Segmentation'].fillna(df_train['Segmentation'].mode()[0], inplace=True)

df_train['Graduated'].fillna(df_train['Graduated'].mode()[0], inplace=True)
df_test['Graduated'].fillna(df_test['Graduated'].mode()[0], inplace=True)

newdf_train2 = df_train.drop_duplicates()
newdf_test2 = df_test.drop_duplicates()

LE = LabelEncoder()
newdf_train2['Spending_Score'] = LE.fit_transform(newdf_train2['Spending_Score'])
newdf_test2['Spending_Score'] = LE.transform(newdf_test2['Spending_Score'])

newdf_train2['Var_1'] = LE.fit_transform(newdf_train2['Var_1'])
newdf_test2['Var_1'] = LE.transform(newdf_test2['Var_1'])

output_LE = LabelEncoder()
newdf_train2['Segmentation'] = output_LE.fit_transform(newdf_train2['Segmentation'])

Gender_train = pd.get_dummies(newdf_train2['Gender'])
Gender_test = pd.get_dummies(newdf_test2['Gender'])

Married_train = pd.get_dummies(newdf_train2['Ever_Married'])
Married_test = pd.get_dummies(newdf_test2['Ever_Married'])

Graduated_train = pd.get_dummies(newdf_train2['Graduated'])
Graduated_test = pd.get_dummies(newdf_test2['Graduated'])

Profession_train = pd.get_dummies(newdf_train2['Profession'])
Profession_test = pd.get_dummies(newdf_test2['Profession'])

newdf_train2.drop(['Gender', 'Ever_Married', 'Graduated', 'Profession'], axis=1, inplace=True)
newdf_test2.drop(['Gender', 'Ever_Married', 'Graduated', 'Profession'], axis=1, inplace=True)

newdf_train3 = pd.concat([newdf_train2, Gender_train, Married_train, Graduated_train, Profession_train], axis=1)
newdf_test3 = pd.concat([newdf_test2, Gender_test, Married_test, Graduated_test, Profession_test], axis=1)

newdf_train3.reset_index(inplace=True)
newdf_test3.reset_index(inplace=True)

newdf_train3.drop(['index'], axis=1, inplace=True)
newdf_test3.drop(['index'], axis=1, inplace=True)

train = newdf_train3.sample(frac=1, random_state=20).reset_index(drop=True)
test = newdf_test3.sample(frac=1, random_state=20).reset_index(drop=True)

# g = pd.DataFrame(train['Var_1']).copy()
# for i in g.columns:
#  probplot(x=g[i], dist='norm', plot=plt)
#  plt.title(i)
#  plt.show()
X_train = train.drop(['Segmentation', 'ID'], axis=1)
y_train = train['Segmentation']
X_test = test.drop(['ID'], axis=1)
y_test_desicion_tree = pd.DataFrame(test['ID']).copy()
y_test_random_forest = pd.DataFrame(test['ID']).copy()
y_test_KNN = pd.DataFrame(test['ID']).copy()
y_test_naive_bayes = pd.DataFrame(test['ID']).copy()
y_test_SVM = pd.DataFrame(test['ID']).copy()
y_test_logistic_regression = pd.DataFrame(test['ID']).copy()
y_test_AdaBoost = pd.DataFrame(test['ID']).copy()
stand = StandardScaler().fit(X_train)
X_train_stand = stand.transform(X_train)
X_test_stand = stand.transform(X_test)
norm = MinMaxScaler().fit(X_train)
X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train_norm, y_train)
y_pred_decision_tree = decision_tree.predict(X_test_norm)
KNN = KNeighborsClassifier()
KNN.fit(X_train_norm, y_train)
y_pred_KNN = KNN.predict(X_test_norm)
random_forest = RandomForestClassifier()
random_forest.fit(X_train_norm, y_train)
y_pred_random_forest = random_forest.predict(X_test_norm)
SVM = SVC()
SVM.fit(X_train_stand, y_train)
y_pred_SVM = SVM.predict(X_test_stand)
Naive_Bayes = GaussianNB()
Naive_Bayes.fit(X_train_stand, y_train)
y_pred_naive_bayes = Naive_Bayes.predict(X_test_stand)
logistic_regression = LogisticRegression()
ovo = OneVsOneClassifier(logistic_regression)
ovo.fit(X_train_stand, y_train)
y_pred_logistic_regression = ovo.predict(X_test_stand)
AdaBoost = AdaBoostClassifier().fit(X_train_norm, y_train)
y_pred_AdaBoost = AdaBoost.predict(X_test_norm)
y_pred_decision_tree = output_LE.inverse_transform(y_pred_decision_tree)
y_pred_random_forest = output_LE.inverse_transform(y_pred_random_forest)
y_pred_KNN = output_LE.inverse_transform(y_pred_KNN)
y_pred_SVM = output_LE.inverse_transform(y_pred_SVM)
y_pred_naive_bayes = output_LE.inverse_transform(y_pred_naive_bayes)
y_pred_logistic_regression = output_LE.inverse_transform(y_pred_logistic_regression)
y_pred_AdaBoost = output_LE.inverse_transform(y_pred_AdaBoost)

y_test_desicion_tree['Segmentation'] = y_pred_decision_tree
y_test_random_forest['Segmentation'] = y_pred_random_forest
y_test_KNN['Segmentation'] = y_pred_KNN
y_test_SVM['Segmentation'] = y_pred_SVM
y_test_naive_bayes['Segmentation'] = y_pred_naive_bayes
y_test_logistic_regression['Segmentation'] = y_pred_logistic_regression
y_test_AdaBoost['Segmentation'] = y_pred_AdaBoost

# y_test_desicion_tree.to_csv("DT_predictions.csv", index=False)
# y_test_random_forest.to_csv("RF_predictions.csv", index=False)
# y_test_KNN.to_csv("KNN_predictions.csv", index=False)
# y_test_SVM.to_csv("SVM_predictions.csv", index=False)
# y_test_naive_bayes.to_csv("NB_predictions.csv", index=False)
# y_test_logistic_regression.to_csv("Logistic_Regression_predictions.csv", index=False)
# y_test_AdaBoost.to_csv("Adaboost_predictions.csv", index=False)