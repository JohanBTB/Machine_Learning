import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================LIMPIEZA DE DATOS==================================

banco = pd.read_csv(r'E:\JOHAN\retomando_python\redes_neuronales\bank_marketing.csv')

#banco.isnull().sum(axis = 0)

#banco.describe(include='all')

#banco.dtypes


# visualizacion prematura
# import seaborn as sns

# correlation = banco.corr()

# sns.heatmap(correlation, cmap = "RdBu", vmin = -1, center = 0)
# =============================================================================
# 1 - age (numeric)
# 
# 2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur", "student","blue-collar","self-employed","retired","technician","services")
# 
# 3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
# 
# 4 - education (categorical: "unknown","secondary","primary","tertiary")
# 
# 5 - default: has credit in default? (binary: "yes","no")
# 
# 6 - balance: average yearly balance, in euros (numeric)
# 
# 7 - housing: has housing loan? (binary: "yes","no")
# 
# 8 - loan: has personal loan? (binary: "yes","no")
# 
# related with the last contact of the current campaign:
# 9 - contact: contact communication type (categorical: "unknown","telephone","cellular")
# 
# 10 - day: last contact day of the month (numeric)
# 
# 11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
# 
# 12 - duration: last contact duration, in seconds (numeric)
# 
# other attributes:
# 13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 
# 14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
# 
# 15 - previous: number of contacts performed before this campaign and for this client (numeric)
# 
# 16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")
# 
# output variable (desired target):
# =============================================================================


# banco['job'].unique()
banco['job'] = banco['job'].map(lambda x: 0 if x=='student' or x=='unknown' or x=='retired' else 1)
valores_job = {'unemployed':0, 'employed':1}
banco.replace(valores_job, inplace = True)

# banco['marital'].unique()
valores_marital = {'single':0, 'married':1, 'divorced':2}
banco.replace(valores_marital, inplace = True)

# banco['educational'].unique()
valores_education = {'unknown':0, 'primary':1, 'secondary':2,'tertiary':3}
banco.replace(valores_education, inplace = True)

# banco['default'].unique()
valores_yes_no = {'no':0, 'yes':1}
banco.replace(valores_yes_no, inplace = True)

# banco['contact'].unique()
banco['contact'] = banco['contact'].map(lambda x: 0 if x=='unknown' or x == 0 else 1 )

# banco['month'].unique()
dict_mes = {}
from datetime import datetime
for mes in banco['month'].unique():
    datetime_mes = datetime.strptime(mes, '%b')
    num_mes = datetime_mes.month
    dict_mes[mes] = num_mes
banco.replace(dict_mes, inplace = True)

banco['poutcome'] = banco['poutcome'].map(lambda x: 0 if x!= 'success' else 1)

# ======================= FIN LIMPIEZA DATOS ==================================

# ======================= CORRELACION DE DATOS ===============================

import seaborn as sns
correlation = banco.corr()
sns.heatmap(correlation, cmap = "RdBu", vmin = -1, center = 0)

# import matplotlib.pyplot as plt

# temp_banco = banco.groupby('age', as_index = False).mean()
# plt.scatter(data = temp_banco, x='age', y='job')
# plt.title('Edad vs Trabajo')
# plt.xlabel('edad')
# plt.ylabel('trabajo')


# plt.scatter(data = temp_banco, x='age', y='marital')
# plt.title('Edad vs Matrimonio')
# plt.xlabel('edad')
# plt.ylabel('matrimonio')



banco.pop('job')
banco.pop('marital')

banco.pop('day')
banco.pop('month')
banco.pop('campaign')
banco.pop('credit_card')
banco.pop('loan')
banco.pop('pdays')
# ======================FIN CORRELACION DE DATOS =============================

# ====================== LOGISTIC REGRESSION =================================

# from sklearn import preprocessing
# from sklearn.pipeline import make_pipeline
# from sklearn.linear_model import LogisticRegression

# x_train = banco.sample(frac = 0.7, random_state=3)
# x_test = banco.drop(x_train.index)

# y_train = x_train.pop('y')
# y_test = x_test.pop('y')

# pipe = make_pipeline(preprocessing.StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=20))
# pipe.fit(x_train,y_train)
# pipe.fit(x_test,y_test)
# print(pipe.score(x_test,y_test))

# ======================= FIN LOGISTIC REGRESSION =============================

# ========================== ARBOL DE DECISION ================================


# x_train = banco.sample(frac = 0.7, random_state=3)
# x_test = banco.drop(x_train.index)

# y_train = x_train.pop('y')
# y_test = x_test.pop('y')

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# dt = DecisionTreeClassifier()

# dt.fit(x_train, y_train)

# y_predict = dt.predict(x_test)

# dt_accuracy = accuracy_score(y_test, y_predict)

# ======================== FIN ARBOL DE DECISION ==============================

from sklearn.naive_bayes import GaussianNB


x_train = banco.sample(frac = 0.7, random_state=3)
x_test = banco.drop(x_train.index)

y_train = x_train.pop('y')
y_test = x_test.pop('y')


gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_predict = gnb.predict(x_test)
gnb_accuracy = gnb.score(x_train,y_train)
print("El score accuracy es: {}".format(round(gnb_accuracy, 10)))






