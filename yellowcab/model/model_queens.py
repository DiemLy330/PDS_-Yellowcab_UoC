from ..import io
import os
from yellowcab.cabana import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import SGDRegressor, SGDClassifier
import sklearn.metrics as sm
from math import radians
from .transform_queens import *
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import SCORERS

class model_queens:
    def __init__(self,X=None):
        self.X = transform_queens()
        self.Xscaled = pre_process_queens(self.X)

    def transform(self,X):
        return transform_queens(X)

    def predict_distance_queens(self,X=None):
        Xscaled = self.Xscaled
        ya = pd.DataFrame(Xscaled['trip_distance'], index=Xscaled.index, columns=['trip_distance'])
        Xa = Xscaled.drop(
            ['trip_distance', 'pd', 'duration', 'passenger_count', 'payment_type', 'pay2', 'pay3', 'pay4', 'r2',
             'r3', 'r4', 'r5', 'r6',
             'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
             'improvement_surcharge', 'total_amount', 'congestion_surcharge',
             'DOweekend', 'DOhoursin', 'DOhourcos', 'DOdaysin', 'DOdaycos', 'DOmonthsin', 'DOmonthcos'], axis=1)
        Xtrain, Xtest, ytrain, ytest = train_test_split(Xa, ya, test_size=0.2, random_state=0)

        ytrainarr = np.array(ytrain).reshape(len(ytrain),1)
        ytestarr = np.array(ytest).reshape(len(ytest), 1)

        transformer_x = RobustScaler().fit(Xtrain)
        transformer_y = RobustScaler().fit(ytrainarr)
        X_train = pd.DataFrame(transformer_x.transform(Xtrain), index=Xtrain.index, columns=Xtrain.columns)
        y_train = transformer_y.transform(ytrainarr)
        X_test = pd.DataFrame(transformer_x.transform(Xtest), index=Xtest.index, columns=Xtest.columns)
        y_test = transformer_y.transform(ytestarr)

        modelhya = SGDRegressor(random_state=0)
        hyperparameters = {'loss': ['squared_loss', 'huber', 'epsilon_insensitive'], 'alpha': np.linspace(0, 0.001, 3)}
        # grid = GridSearchCV(modelhya, hyperparameters, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
        grid = GridSearchCV(modelhya, hyperparameters, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid.fit(Xtrain, ytrain)
        comparison = pd.DataFrame(np.abs(grid.cv_results_['mean_test_score']), columns=['Loss'],
                                  index=['squared0', 'huber0', 'ep0', 'squared0.0005', 'huber0.0005', 'ep.0005',
                                         'squared0.001', 'huber0.001', 'ep0.001'])
        modela = SGDRegressor(alpha=0, random_state=0)
        model = modela.fit(X_train, y_train)

        # Predict X
        if X == None:
            res = model.predict(X_test)
        else:
            Xtransformed = transform_queens(X)
            Xprocessed = pre_process_queens(Xtransformed)
            Xdropped = Xprocessed.drop(
            ['trip_distance', 'pd', 'duration', 'passenger_count', 'payment_type', 'pay2', 'pay3', 'pay4', 'r2',
             'r3', 'r4', 'r5', 'r6',
             'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
             'improvement_surcharge', 'total_amount', 'congestion_surcharge',
             'DOweekend', 'DOhoursin', 'DOhourcos', 'DOdaysin', 'DOdaycos', 'DOmonthsin', 'DOmonthcos'], axis=1)
            Xnew = pd.DataFrame(transformer_x.transform(Xdropped), index=Xdropped.index, columns=Xdropped.columns)
            res = model.predict(Xnew)
        predicted_distance = transformer_y.inverse_transform(np.reshape(res, (res.shape[0], 1))).flatten()
        filename = 'predict_distance_queens.pkl'
        io.save_model(model, filename)
        return predicted_distance


    def predict_fare_queens(self,X=None):
        Xscaled = self.Xscaled
        yb = pd.DataFrame(Xscaled['fare_amount'], index=Xscaled.index, columns=['fare_amount'])
        Xb = Xscaled.drop(['fare_amount', 'payment_type',
                           'extra',
                           'mta_tax', 'tip_amount',
                           'tolls_amount', 'improvement_surcharge',
                           'total_amount', 'congestion_surcharge', 'trip_distance', 'duration'
                           ], axis=1)
        Xtrain, Xtest, ytrain, ytest = train_test_split(Xb, yb, test_size=0.2, random_state=0)

        ytraindf = np.array(ytrain).reshape(len(ytrain), 1)
        ytestdf = np.array(ytest).reshape(len(ytest), 1)

        transformer_x = RobustScaler().fit(Xtrain)
        transformer_y = RobustScaler().fit(ytraindf)
        X_train = pd.DataFrame(transformer_x.transform(Xtrain), index=Xtrain.index, columns=Xtrain.columns)
        y_train = transformer_y.transform(ytraindf)
        X_test = pd.DataFrame(transformer_x.transform(Xtest), index=Xtest.index, columns=Xtest.columns)
        y_test = transformer_y.transform(ytestdf)

        # Find the best hyperparameter
        modelhyb = SGDRegressor(random_state=0)
        hyperparameters = {'loss': ['squared_loss', 'huber'], 'alpha': [0, 0.0001]}
        # grid = GridSearchCV(modelhyb, hyperparameters, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
        grid = GridSearchCV(modelhyb, hyperparameters, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        comparison = pd.DataFrame(np.abs(grid.cv_results_['mean_test_score']), columns=['Loss'],
                                  index=['squared0', 'huber0', 'squared0.0001', 'huber0.0001'])
        # doing cross validation to find out how many features to go for
        modelb = SGDRegressor(alpha=0, random_state=0)
        scores = cross_val_score(modelb, X_train, y_train, cv=5)
        # np.mean(scores)
        # Xscaled.corr()

        #6 strongest features: pd + 4 distance features + r2
        modelb7 = SGDRegressor(alpha=0, random_state=0)
        # scoresb7 = cross_val_score(modelb7, X_train[['pd','tojfk','tolgd', 'toql', 'tomar','r2' ]], y_train, cv=5)
        # np.mean(scoresb7)

        modelb1 = SGDRegressor(alpha=0, random_state=0)
        # scoresb1 = cross_val_score(modelb1, pd.DataFrame(X_train['pd'], index=Xtrain.index, columns=['pd']), y_train,cv=5)
        # np.mean(scoresb1)

        # => conclusion: 1 feature should be best both in terms of the performance of the model and the number of parameters. Let's see how the 1 feature perform on the test se
        finalmodelb = SGDRegressor(alpha=0, random_state=0)
        col = pd.DataFrame(X_test.pd, index=X_test.index, columns=['pd'])
        col_train = pd.DataFrame(X_train.pd, index=X_train.index, columns=['pd'])
        finalmodelb.fit(col_train, y_train)
        filename = 'predict_fare_queens.pkl'

        # Predict X
        if X == None:
            res = finalmodelb.predict(col)
        else:
            Xtransformed = transform_queens(X)
            Xprocessed = pre_process_queens(Xtransformed)
            Xdropped = Xprocessed.drop(['fare_amount', 'payment_type',
                                    'extra',
                                    'mta_tax', 'tip_amount',
                                    'tolls_amount', 'improvement_surcharge',
                                    'total_amount', 'congestion_surcharge', 'trip_distance', 'duration'
                                    ], axis=1)
            Xnew = pd.DataFrame(transformer_x.transform(Xdropped), index=Xdropped.index, columns=Xdropped.columns)
            res = finalmodelb.predict(Xnew)
        predicted_fare = transformer_y.inverse_transform(np.reshape(res,(res.shape[0],1))).flatten()
        io.save_model(finalmodelb, filename)
        return predicted_fare


    def predict_payment_type_queens(self, X=None):
        Xscaled = self.Xscaled
        yc = pd.DataFrame(Xscaled['payment_type'], index=Xscaled.index, columns=['payment_type'])
        Xc = Xscaled.drop(['payment_type', 'pay2', 'pay3', 'pay4'], axis=1)
        Xtrain, Xtest, ytrain, ytest = train_test_split(Xc, yc, test_size=0.2, random_state=0)

        ytraindf = np.array(ytrain).reshape(len(ytrain), 1)
        ytestdf = np.array(ytest).reshape(len(ytest), 1)

        transformer_x = RobustScaler().fit(Xtrain)
        transformer_y = RobustScaler().fit(ytraindf)
        X_train = pd.DataFrame(transformer_x.transform(Xtrain), index=Xtrain.index, columns=Xtrain.columns)
        y_train = transformer_y.transform(ytraindf)
        X_test = pd.DataFrame(transformer_x.transform(Xtest), index=Xtest.index, columns=Xtest.columns)
        y_test = transformer_y.transform(ytestdf)

        modelclog = SGDClassifier(loss='log', random_state=0)
        scoreclog = cross_val_score(modelclog, X_train, y_train, cv=5)
        # np.mean(scoreclog)
        modelcsvc = SGDClassifier(loss='hinge', random_state=0)
        scorecsvc = cross_val_score(modelcsvc, X_train, y_train, cv=5)
        # np.mean(scorecsvc)
        modelc = SGDClassifier(loss='hinge', random_state=0)
        model = modelc.fit(X_train, y_train)
        filename = 'predict_payment_type_queens.pkl'

        # Predict X
        if X == None:
            res = model.predict(X_test)
        else:
            Xtransformed = transform_queens(X)
            Xprocessed = pre_process_queens(Xtransformed)
            Xdropped = Xprocessed.drop(['payment_type', 'pay2', 'pay3', 'pay4'], axis=1)
            Xnew = pd.DataFrame(transformer_x.transform(Xdropped), index=Xdropped.index, columns=Xdropped.columns)
            res = model.predict(Xnew)
        predicted_ptype = transformer_y.inverse_transform(np.reshape(res, (res.shape[0], 1))).flatten()
        io.save_model(model, filename)
        return predicted_ptype


    def predict(self,X=None):
        if X == None:
            distance = self.predict_distance_queens()
            fare = self.predict_fare_queens()
            type = self.predict_payment_type_queens()
        else:
            distance = self.predict_distance_queens(X)
            fare = self.predict_fare_queens(X)
            type = self.predict_payment_type_queens(X)
        df = pd.DataFrame(list(zip(distance,fare,type)),columns=['predicted_distance','predicted_fare','predicted_payment_type'])
        filename= 'predict_queens.pkl'
        io.save_model(df,filename)
        return df


