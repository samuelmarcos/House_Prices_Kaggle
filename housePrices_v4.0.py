#!/usr/local/bin/python

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV
from scipy.stats import skew
from scipy import stats

class HousePrices():

    def __init__(self):
        # carrega os dados
        self.train = pd.read_csv('train.csv')
        self.test = pd.read_csv('test.csv')
        self.variaveis_numericas = self.train.select_dtypes(include=[np.number])
        self.variaveis_categoricas = self.train.select_dtypes(exclude=[np.number])
        self.x_train = self.train
        self.x_test = self.test

    # mostra a correlacao dos atributos com SalePrice
    def correlacao_variaveis(self):
     # as 10 mais positivas e 10 mais negativas
        correlacao = self.variaveis_numericas.corr()
        print("Correlacao")
        print (correlacao['SalePrice'].sort_values(ascending=True)[:20], '\n')
        print (correlacao['SalePrice'].sort_values(ascending=False)[-10:])

    def mostrar_valores_nulos(self):
        miss = self.x_test.isnull().sum()/len(self.x_test)
        miss = miss[miss > 0]
        miss.sort_values(ascending=False,inplace=True)
        print(miss)

    def selecionar_atributos(self):        
        correlacao = self.variaveis_numericas.corr()
        for i in correlacao:
            if ((correlacao['SalePrice'][i] < 0.05 and correlacao['SalePrice'][i] > -0.05)
            and i != 'SalePrice' and i != 'Id'):
                self.x_train = self.x_train.drop([i],axis=1)
                self.x_test = self.x_test.drop([i],axis=1)
        
        self.x_train = self.x_train.drop(['Id'],axis=1)
        self.x_test = self.x_test.drop(['Id'],axis=1)
        
        # atributos excluidos pois a maioria dos valores sao NAs
        self.x_train = self.x_train.drop('PoolQC',axis=1)
        self.x_test = self.x_test.drop('PoolQC',axis=1)
        self.x_train = self.x_train.drop('MiscFeature',axis=1)
        self.x_test = self.x_test.drop('MiscFeature',axis=1)
        self.x_train = self.x_train.drop('Alley',axis=1)
        self.x_test = self.x_test.drop('Alley',axis=1)
                
    def mostrar_outliers(self):
        variaveis_numericas = self.x_train.select_dtypes(include=[np.number])
        for i in variaveis_numericas:
            plt.figure()
            plt.clf()
            plt.boxplot(list(self.x_train[i]))
            plt.title(i)
            plt.show()

#            plt.scatter(x=self.x_train[i], y=self.x_train.SalePrice)
#            plt.ylabel('Sale Price')
#            plt.xlabel(i)
#            plt.show()

    def remover_outliers(self):
        
        # remove outliers do conjunto de treino
        self.x_train = self.x_train[self.x_train['GarageArea'] < 1200]
        self.x_train = self.x_train[self.x_train['EnclosedPorch'] < 500]
        self.x_train = self.x_train[self.x_train['GrLivArea'] < 4000 ]
        self.x_train = self.x_train[self.x_train['1stFlrSF'] < 4000 ]
        self.x_train = self.x_train[self.x_train['TotalBsmtSF'] < 3000]
        self.x_train = self.x_train[self.x_train['BsmtFinSF1'] < 5000]
        self.x_train = self.x_train[self.x_train['MasVnrArea'] < 1300]
        self.x_train = self.x_train[self.x_train['LotArea'] < 60000]
        self.x_train = self.x_train[self.x_train['LotFrontage'] < 200]
        

    def tratar_valores_ausentes(self):
        self.x_train.isnull().sum().sort_values(ascending=False)
        self.x_train = self.x_train.fillna(self.x_train.mean())
        self.x_test.isnull().sum().sort_values(ascending=False)
        self.x_test = self.x_test.fillna(self.x_test.mean())
        
        train_test =  pd.concat(objs=[self.x_train, self.x_test], axis=0).reset_index(drop=True)
        train_test["GarageType"] = train_test["GarageType"].fillna("None")
        train_test["MasVnrType"] = train_test["MasVnrType"].fillna("None")
        train_test["Electrical"] = train_test["Electrical"].fillna("None")
        self.x_train = train_test[:len(self.x_train)]
        self.x_test = train_test[len(self.x_train):]
        
    def numericasToOrdenada(self):
        train_test =  pd.concat(objs=[self.x_train, self.x_test], axis=0).reset_index(drop=True)
        train_test["BsmtCond"] = train_test["BsmtCond"].astype("category",categories=['No','Po','Fa','TA','Gd','Ex'],ordered=True).cat.codes
        train_test["BsmtExposure"] = train_test["BsmtExposure"].astype("category",categories=['No','Mn','Av','Gd'],ordered=True).cat.codes
        train_test["BsmtFinType1"] = train_test["BsmtFinType1"].astype("category",categories=['No','Unf','LwQ','Rec','BLQ','ALQ','GLQ'],ordered=True).cat.codes
        train_test["BsmtFinType2"] = train_test["BsmtFinType2"].astype("category",categories=['No','Unf','LwQ','Rec','BLQ','ALQ','GLQ'],ordered=True).cat.codes
        train_test["BsmtQual"] = train_test["BsmtQual"].astype("category",categories=['No','Po','Fa','TA','Gd','Ex'],ordered=True).cat.codes
        train_test["ExterCond"] = train_test["ExterCond"].astype("category",categories=['Po','Fa','TA','Gd','Ex'],ordered=True).cat.codes
        train_test["ExterQual"] = train_test["ExterQual"].astype("category",categories=['Po','Fa','TA','Gd','Ex'],ordered=True).cat.codes
        train_test["Fence"] = train_test["Fence"].astype("category",categories=['No','MnWw','GdWo','MnPrv','GdPrv'],ordered=True).cat.codes
        train_test["FireplaceQu"] = train_test["FireplaceQu"].astype("category",categories=['No','Po','Fa','TA','Gd','Ex'],ordered=True).cat.codes
        train_test["Functional"] = train_test["Functional"].astype("category",categories=['Sal','Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'],ordered=True).cat.codes
        train_test["GarageCond"] = train_test["GarageCond"].astype("category",categories=['No','Po','Fa','TA','Gd','Ex'],ordered=True).cat.codes
        train_test["GarageFinish"] = train_test["GarageFinish"].astype("category",categories=['No','Unf','RFn','Fin'],ordered=True).cat.codes
        train_test["GarageQual"] = train_test["GarageQual"].astype("category",categories=['No','Po','Fa','TA','Gd','Ex'],ordered=True).cat.codes
        train_test["HeatingQC"] = train_test["HeatingQC"].astype("category",categories=['Po','Fa','TA','Gd','Ex'],ordered=True).cat.codes
        train_test["KitchenQual"] = train_test["KitchenQual"].astype("category",categories=['Po','Fa','TA','Gd','Ex'],ordered=True).cat.codes
        train_test["PavedDrive"] = train_test["PavedDrive"].astype("category",categories=['N','P','Y'],ordered=True).cat.codes
        train_test["Utilities"] = train_test["Utilities"].astype("category",categories=['ELO','NoSeWa','NoSewr','AllPub'],ordered=True).cat.codes
        self.x_train = train_test[:len(self.x_train)]
        self.x_test = train_test[len(self.x_train):]
        self.x_test = self.x_test.drop(['SalePrice'], axis=1)
    
    def converter_categoricas_numericas(self):
       # Create dummy variables for the categorical features and handle the missing values
       train_test =  pd.concat(objs=[self.x_train, self.x_test], axis=0).reset_index(drop=True)
       train_test = pd.get_dummies(train_test)
       self.x_train = train_test[:len(self.x_train)]
       self.x_test = train_test[len(self.x_train):]
       self.x_test = self.x_test.drop(['SalePrice'], axis=1)
        
    def regressaoLinear(self):
        y_train = np.log(self.x_train.SalePrice)
        X_train = self.x_train.drop(['SalePrice'],axis=1)
        rl = linear_model.LinearRegression()
        modelo = rl.fit(X_train,y_train)
#        y_pred = modelo.predict(self.x_test)

        # Root mean squre with random forest
        rmse = np.sqrt(-cross_val_score(rl, X_train, y_train, scoring="neg_mean_squared_error", cv = 10)).mean()
        print ("RMSE Regressao Linear", rmse)
        self.submissao(modelo,"Regressao Linear")
    
    def randomForest2(self):
        y_train = np.log(self.x_train.SalePrice)
        X_train = self.x_train.drop(['SalePrice'],axis=1)
        rfr = RandomForestRegressor(10, n_jobs=-1, random_state=42)
        modelo = rfr.fit(X_train,y_train)
#        y_pred = modelo.predict(self.x_test)

        # Root mean squre with random forest
        rmse = np.sqrt(-cross_val_score(rfr, X_train, y_train, scoring="neg_mean_squared_error", cv = 10)).mean()
        print ("RMSE Random Forest", rmse)
        self.submissao(modelo,"Random Forest")
        
    def gradientBoosting(self):
        y_train = np.log(self.x_train.SalePrice)
        X_train = self.x_train.drop(['SalePrice'],axis=1)
        gbr = GradientBoostingRegressor(random_state=0)
        param_grid = {# 'n_estimators': [500],'max_features': [10,15],'max_depth': [6,8,10],'learning_rate': [0.05,0.1,0.15],'subsample': [0.8]
        }
        modelo = gbr.fit(X_train,y_train)
        y_pred = modelo.predict(self.x_test)
        rmse = np.sqrt(-cross_val_score(gbr, X_train, y_train, scoring="neg_mean_squared_error", cv = 10)).mean()
        print ("RMSE Gradient Boosting",rmse)
#        y_pred = model.predict(self.x_test)
        self.submissao(modelo,"Gradient Boosting")
        
    def submissao(self,model,name):
        submissao = pd.DataFrame()
        submissao['Id'] = self.test.Id
        features = self.x_test
        predictions = model.predict(features)
        submissao['SalePrice'] = np.exp(predictions)
        submissao.to_csv('submissao'+name+'.csv',index=False)
        
    def imprime_atributos_nulos(self):
        nulls = pd.DataFrame(self.x_train.isnull().sum().sort_values(ascending=False))
        nulls.columns = ['Null Count']
        nulls.index.name = 'Feature'
        print(nulls)


    def main(self):

#        print ("Train data shape:", self.train.shape)
#        print ("Test data shape:", self.test.shape)

        self.selecionar_atributos()
#        
        self.tratar_valores_ausentes()
#
        self.remover_outliers()
        
        self.numericasToOrdenada()
#        
        self.converter_categoricas_numericas()

        self.regressaoLinear()

        self.randomForest2()

        self.gradientBoosting()

        
        print ("x_train data shape:", self.x_train.shape)
        print ("x_Test data shape:", self.x_test.shape)

hp = HousePrices()
hp.main()
