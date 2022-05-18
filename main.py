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
dict_month = {}
from datetime import datetime
for month in banco['month'].unique():
    datetime_month = datetime.strptime(month, '%b')
    num_month = datetime_month.month
    dict_month[month] = num_month
banco.replace(dict_month, inplace = True)

banco['poutcome'] = banco['poutcome'].map(lambda x: 0 if x!= 'success' else 1)

# import seaborn as sns
# correlation = banco.corr()
# sns.heatmap(correlation, cmap = "RdBu", vmin = -1, center = 0)






