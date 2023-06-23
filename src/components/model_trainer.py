import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            x_train, y_train, x_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            print ("x_train shape:", x_train.shape)

            print ("x_test shape:", x_test.shape)

            models = {
                "RFC": RandomForestClassifier(),
                "DTC": DecisionTreeClassifier(),
                "GBC": GradientBoostingClassifier(),
                "XGBC": XGBClassifier(),
                "CBC": CatBoostClassifier(verbose=False),
                "ABC": AdaBoostClassifier(),
                "LDAM" : LinearDiscriminantAnalysis(),
                "LR" : LogisticRegression()
            }
            params={
                "DTC": {
                    'criterion':['entropy', 'log_loss', 'gini'],
                },
                "XGBC": {
                    'max_depth': [1,2,3,4,5,6,7,8,10,11,12], 
                    'n_estimators':[8, 12, 16, 20, 24, 28, 32]

                },

                "RFC":{

                    'max_depth': [1,2,3,4,5,6,7,8,10,11,12], 
                    'n_estimators':[8, 12, 16, 20, 24, 28, 32]
                },

                "GBC":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                   'n_estimators':[8, 12, 16, 20, 24, 28, 32]
                },
                "CBC":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30]
                },
                "ABC":{
                    'learning_rate':[.1,.01,0.5,.001],
                     'n_estimators':[8, 12, 16, 20, 24, 28, 32]
                },
                
                "LDAM":{

                    'solver' : ['svd', 'lsqr', 'eigen'],
                    'tol' : [0.00001, 0.0001, 0.001, 0.01, 0.1]
                },

                "LR":{
                    'solver': ['liblinear', 'sag', 'saga'],
                    'tol' : [0.00001, 0.0001, 0.001, 0.01, 0.1]
                },
                
            }

            model_report : dict=evaluate_models (x_train = x_train, y_train = y_train, x_test = x_test,
                                                 y_test=y_test, models = models, param = params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.1:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            y_pred_train = best_model.predict(x_train)

            logging.info("Made prediction on training data")
            
            y_pred_test = best_model.predict (x_test)
            
            logging.info("Made prediction on test data")
            
            training_data_accuracy = accuracy_score(y_pred_train, y_train)
            
            print ("Accuracy On Training Data:", training_data_accuracy)
            
            logging.info("Train Accuracy obtained")

            test_data_accuracy = accuracy_score(y_pred_test, y_test)

            print ("Accuracy On Test Data:", test_data_accuracy)

            logging.info("test data acurracy obtained")
            
            print ("Correct predictions Train:", sum (y_train == y_pred_train))

            logging.info("Correct predictions calculated on x_train")
            
            print ("Correct predictions Test:", sum (y_test == y_pred_test))

            logging.info("Correct predictions calculated on x_test")
            
            print ("Incorrect predictions Train:", sum (y_train != y_pred_train))

            logging.info("Incorrect predictions calculated on x_train")
            
            print ("Incorrect predictions Test:", sum (y_test != y_pred_test))

            logging.info("Incorrect predictions calculated on x_test")
            
            print ("F1 Score Train:", f1_score(y_train, y_pred_train))

            logging.info("F1 score calculated on x_train")
            
            print ("F1 Score Test:", f1_score(y_test, y_pred_test))

            logging.info("F1 score calculated on x_test")
            
            print('Precision Train: %.3f' % precision_score(y_train, y_pred_train))

            logging.info("Precision calculated on x_train")
            
            print('Precision Test: %.3f' % precision_score(y_test, y_pred_test))

            logging.info("Precision calculated on x_test")
            
            print('Recall Train: %.3f' % recall_score(y_train, y_pred_train))

            logging.info("Recall calculated on x_train")
            
            print('Recall Test: %.3f' % recall_score(y_test, y_pred_test))

            logging.info("Recall calculated on x_test")
            
            FPR, TPR, threshold = roc_curve(y_train, y_pred_train)
            
            print('roc_auc_score train: ', roc_auc_score(y_train, y_pred_train))

            logging.info("ROC AUC Score calculated on x_train")

            print('roc_auc_score test: ', roc_auc_score(y_test, y_pred_test))

            logging.info("ROC AUC Score calculated on x_test")
            
            print ("Confusion Matrix Train:\n", confusion_matrix(y_train , y_pred_train))

            logging.info("Confusion matrix calculated on x_train")

            print ("Confusion Matrix Test:\n", confusion_matrix(y_test , y_pred_test))

            logging.info("Confusion matrix calculated on x_test")

            print ("Classification Report Train:\n", classification_report (y_train, y_pred_train, digits = 4))

            logging.info("Classification report obtained on x_train")

            print ("Classification Report Test:\n", classification_report (y_test, y_pred_test, digits = 4))

            logging.info("Classification report obtained on x_test")
            
            return training_data_accuracy, test_data_accuracy, best_model
            
        except Exception as e:
            raise CustomException(e,sys)