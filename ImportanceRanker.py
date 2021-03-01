#!/usr/bin/env python
# -*- coding:utf-8 _*-

from __future__ import print_function

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import csv

## 排序参数个数
configNum = 0

class Train_CV_SGBRT(object):

    def __init__(self):
        ## 数据路径
        self.data_path = ""
        ## 文件名称
        self.algorithm_name = "Normalized_confs"

    def train_sk_cv(self):

        if ".csv" in self.algorithm_name:
            algorithm = str(self.algorithm_name)
        else:
            algorithm = str(self.algorithm_name + '.csv')

        algorithm = os.path.join(self.data_path, algorithm)
        data = pd.read_csv(algorithm)

        X = data.iloc[1:, 0:configNum]
        y = data.iloc[1:, configNum]

        # Iteration times
        Itera = 24

        events_name = X.columns

        def feature_importances_(self):
            """
            Returns
            -------
            feature_importances_ : array of shape = [n_features]
            """
            b = self.booster()
            fs = b.get_fscore()
            all_features = [fs.get(f, 0.) for f in b.feature_names]
            all_features = np.array(all_features, dtype=np.float32)
            return all_features / all_features.sum()

        def data_process_remove_zero(ytest, predict):
            """ removing zero in ytest, predict for computing error
            :param ytest:
            :param predict:
            :return:
            """
            ytest = ytest.tolist()
            predict = predict.tolist()
            zero = []
            for i in range(len(ytest)):
                if ytest[i] == 0:
                    # print('-----> zero value in ytest: {}'.format(ytest[i]))
                    zero.append(i)
                    ytest[i] = "zero"
                    predict[i] = "zero"
            if "zero" in ytest:
                ytest.remove("zero")
                predict.remove("zero")
            ytest = np.asarray(ytest).astype(float)
            predict = np.asarray(predict).astype(float)
            return ytest, predict

        def cv_estimate(X_train, y_train):
            """
            corss validation: split the X_train to train dataset and valid dataset
            -------
            :param X_train: Event value. type: Pandas DataFrame
            :param y_train: IPC value. type: Pandas DataFrame
            :return: GBM model which has lowest error
            """
            n_splits = 9
            if X_train is None or y_train is None:
                return -1
            cv_clf_list = []
            err_list = []
            cv = KFold(n_splits=n_splits)
            forest = GradientBoostingRegressor()
            # parameter
            forest = GradientBoostingRegressor(n_estimators=800,  # 800,
                                               max_depth=20,  # 20,
                                               # min_samples_leaf=1,
                                               min_samples_split=2)

            for train, valid in cv.split(X_train, y_train):
                forest.fit(X_train.iloc[train], y_train.iloc[train])
                preds = forest.predict(X_train.iloc[valid])
                predicted = preds
                y_test = y_train.iloc[valid]
                assert len(predicted) == len(y_test)
                y_test = np.asarray(y_test).astype(float)
                predicted = np.asarray(predicted).astype(float)
                y_test, predicted = data_process_remove_zero(ytest=y_test, predict=predicted)
                err = np.mean(np.abs(y_test - predicted) / y_test)
                cv_clf_list.append(forest)
                err_list.append(err)

            cv_clf_lowest = cv_clf_list[np.argmin(err_list)]
            return cv_clf_lowest

        Err = []
        Importances = []
        Indices = []
        Events_Name = []
        importanceRanksCsv = []

        # Itera = 24
        forest = GradientBoostingRegressor(max_depth=0, n_estimators=1)
        for _ in range(Itera):
            """ Event Importance Refinement (EIR) """
            print("---------- the " + str(_) + "th training ----------")
            assert len(X) == len(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            forest = cv_estimate(X_train, y_train)
            forest = forest.fit(X_train, y_train)

            predicted = forest.predict(X_test)
            assert len(predicted) == len(y_test)
            y_test = np.asarray(y_test).astype(float)
            predicted = np.asarray(predicted).astype(float)
            y_test, predicted = data_process_remove_zero(ytest=y_test, predict=predicted)
            err = np.mean(np.abs(y_test - predicted) / y_test)

            importances = forest.feature_importances_
            indices = np.argsort(importances)[::-1]

            importances_sum = sum(importances)
            if (importances_sum < 0.9999) or (importances_sum > 1.0002):
                for i in range(len(Events_Name[_ - 1])):
                    importanceRanks = []
                    ## 排序对象名称
                    featureName = []
                    importanceRanks.append(featureName)
                    importanceRanks.append(Importances[_ - 1][i])
                    importanceRanksCsv.append(importanceRanks)
                ## 设定输出路径
                with open("输出文件路径") as importanceRanksFile:
                    importanceRanksFile_csv = csv.writer(importanceRanksFile)
                    for i in range(len(Events_Name[_ - 1])):
                        importanceRanksFile_csv.writerow(importanceRanksCsv[i])
                print("------- Sum of importance:  {}".format(importances_sum) + "--------")
                os._exit(1)

            Err.append(err)
            Indices.append(indices)
            events_Name = []
            importanceS = []
            for f in range(X.shape[1]):
                events_Name.append(events_name[indices[f]])
                importanceS.append(importances[indices[f]])
                # print("%d. feature %d  %s (%f)" % (f + 1, indices[f], events_name[indices[f]], importances[indices[f]]))
            Events_Name.append(events_Name)
            Importances.append(importanceS)

            if _ < Itera - 1:
                """ Deleting the least important 10 events in each Iteration 206 """
                X = pd.DataFrame(X)
                X[X.columns[indices[-10 * (_ + 1):]]] = 0
                # X = np.array(X)
            print(' Error: ', err * 100, '%')
        for i in range(len(Events_Name[_ - 1])):
            importanceRanks = []
            #
            featureName = []
            # print(featureName)
            importanceRanks.append(featureName)
            importanceRanks.append(Importances[_ - 1][i])
            importanceRanksCsv.append(importanceRanks)
        with open("输出文件路径") as importanceRanksFile:
            importanceRanksFile_csv = csv.writer(importanceRanksFile)
            for i in range(len(Events_Name[_ - 1])):
                importanceRanksFile_csv.writerow(importanceRanksCsv[i])

    def build(self):
        self.train_sk_cv()

    def build_loop(self):
        """
        Traverse the entire folder
        :return:
        """
        path_list = os.listdir(self.data_path)
        # print(path_list)

        for i in path_list:
            # print(i)
            self.algorithm_name = i
            self.train_sk_cv()


if __name__ == '__main__':
    train_cv_sgbrt = Train_CV_SGBRT()
    train_cv_sgbrt.__init__()
    train_cv_sgbrt.build()
