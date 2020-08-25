import pandas as pd
import numpy as np
from  sklearn import linear_model as lm
import seaborn as sns
from matplotlib import pyplot as plt

dados = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
dados.describe().T
print(list(dados.columns))

dados.loc[488,"TotalCharges"] = dados.loc[488,"MonthlyCharges"]
dados.loc[753,"TotalCharges"] = dados.loc[753,"MonthlyCharges"]
dados.loc[936,"TotalCharges"] = dados.loc[936,"MonthlyCharges"]
dados.loc[2259,"TotalCharges"] = dados.loc[2259,"MonthlyCharges"]


dados["TotalCharges"] = pd.to_numeric(dados["TotalCharges"])



sns.barplot(x = "Churn", y = "MonthlyCharges", data = dados)
plt.show()
sns.boxplot(x = "Churn", y = "tenure", data = dados)
plt.show()
sns.boxplot(x = "Churn", y = "TotalCharges", data = dados)
plt.show()


print(pd.crosstab(dados.Contract, dados.Churn, margins = True, normalize = "index")) 
print(pd.crosstab(dados.PaperlessBilling, dados.Churn, margins = True, normalize = "index"))
print(pd.crosstab(dados.gender, dados.Churn, margins = True, normalize = "index")) #There is no difference between male and female
print(pd.crosstab(dados.Dependents, dados.Churn, margins = True, normalize = "index")) 
print(pd.crosstab(dados.SeniorCitizen, dados.Churn, margins = True, normalize = "index"))
print(pd.crosstab(dados.Partner, dados.Churn, margins = True, normalize = "index")) 
print(pd.crosstab(dados.PhoneService, dados.Churn, margins = True, normalize = "index")) #There is no difference here
print(pd.crosstab(dados.MultipleLines, dados.Churn, margins = True, normalize = "index")) #Multiple lines has a little trend to churn
print(pd.crosstab(dados.InternetService, dados.Churn, margins = True, normalize = "index"))
print(pd.crosstab(dados.OnlineSecurity, dados.Churn, margins = True, normalize = "index"))
print(pd.crosstab(dados.DeviceProtection, dados.Churn, margins = True, normalize = "index"))
print(pd.crosstab(dados.TechSupport, dados.Churn, margins = True, normalize = "index"))
print(pd.crosstab(dados.StreamingTV, dados.Churn, margins = True, normalize = "index"))
print(pd.crosstab(dados.StreamingMovies, dados.Churn, margins = True, normalize = "index"))
print(pd.crosstab(dados.PaperlessBilling, dados.Churn, margins = True, normalize = "index"))
print(pd.crosstab(dados.PaymentMethod, dados.Churn, margins = True, normalize = "index"))
print(pd.crosstab(dados.Dependents, dados.Churn, margins = True, normalize = "index"))



X = dados.iloc[:,1:18]
print(X)
Y = dados.iloc[:,20]

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



label_encoder = LabelEncoder()
categorical_cols = [col for col in X.select_dtypes(exclude=["number"]).columns]

X_categorical = pd.DataFrame()
for col in categorical_cols:
    X_categorical[col] = label_encoder.fit_transform(X[col])
print(X_categorical)
    
Y_categorical = pd.DataFrame()
Y_categorical = label_encoder.fit_transform(Y)

X_num = dados[["MonthlyCharges", "tenure"]]


X_categorical = pd.concat([X_categorical.reset_index(drop = True), X_num], axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(X_categorical, Y_categorical, test_size=0.2, random_state=42)

modelo = lm.Lasso(alpha = 0.2)
modelo.fit(X_train, Y_train)
ajuste = modelo.fit(X_train, Y_train)
ola = ajuste.predict(X_test) >= 0.5
print(pd.crosstab(ola, Y_test))

modelo = lm.RidgeClassifier(alpha = 0.4)
modelo.fit(X_train, Y_train)
ajuste = modelo.fit(X_train, Y_train)
ola = ajuste.predict(X_test)
print(pd.crosstab(ola, Y_test))


modelo = lm.LogisticRegression()
modelo.fit(X_train, Y_train)
ajuste = modelo.fit(X_train, Y_train)
ola = ajuste.predict(X_test)
print(pd.crosstab(ola, Y_test))



from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 
modelo2 = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.3))
ajuste2 = modelo2.fit(X_train, Y_train)
ola2 = ajuste2.predict(X_test) >= 0.5
print(pd.crosstab(ola2, Y_test))


from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth = 6, splitter = "random")
caminho = clf.cost_complexity_pruning_path(X_train, Y_train)
ccp_alphas, impurities = caminho.ccp_alphas, caminho.impurities
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.show()


ajuste3 = clf.fit(X_train, Y_train)
ola3 = ajuste3.predict(X_test)
print(pd.crosstab(ola3, Y_test))


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

clf = RandomForestClassifier(max_depth=4, random_state=3)
ajuste4 = clf.fit(X_train, Y_train)
ola4 = ajuste4.predict(X_test)
print(pd.crosstab(ola4, Y_test))


