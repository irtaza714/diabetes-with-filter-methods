import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")
    trainos_data_path: str=os.path.join('artifacts',"train_os.csv")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):

        try:
            outliers = ["hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level"]

            cat = ['gender', 'smoking_history']
            
            no_outliers = ['age']
            
            outliers_pipeline= Pipeline( steps=
                                        [
                                         ("rs", RobustScaler())])
            
            no_outliers_pipeline = Pipeline( steps=
                                        [
                                         ("ss", StandardScaler())])

            cat_pipeline = Pipeline( steps=
                                  [
                                   ('ohe', OneHotEncoder())])
            
            preprocessor = ColumnTransformer(
                [
                    ("outliers_pipeline", outliers_pipeline, outliers),
                    ("no_outliers_pipeline", no_outliers_pipeline, no_outliers),
                    ("cat_pipeline", cat_pipeline, cat)
                ]
            )



            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train = pd.read_csv(train_path)

            logging.info("Read train data")
            
            test = pd.read_csv(test_path)

            logging.info("Read test data")

            os.makedirs(os.path.dirname(self.data_transformation_config.trainos_data_path),exist_ok=True)

            logging.info ("directory made for df_os")

            x_train_transf = train.drop('diabetes',axis=1)

            logging.info("Dropped target column from the train set to make the input data frame for model training")

            y_train_transf = train['diabetes']

            logging.info("Target feature obtained for model training")

            x_test_transf = test.drop('diabetes', axis=1)

            logging.info("Dropped target column from the test set to make the input data frame for model testing")
        
            y_test_transf = test['diabetes']

            logging.info("Target feature obtained for model testing")

            # print ("y_train classes:", y_train_transf.value_counts())

            # print ("y_test classes:", y_test_transf.value_counts())

            preprocessor = self.get_data_transformer_object()
            
            logging.info("Preprocessing object obtained")

            x_train_transf_preprocessed = preprocessor.fit_transform(x_train_transf)

            logging.info("Preprocessor applied on x_train_transf")

            x_train_transf_preprocessed_df = pd.DataFrame(x_train_transf_preprocessed)

            logging.info('''x_train_transf dataframe formed for feature selection by filter methods, VIF and dimensionality
                         reduction by LDA''')
            
            for i in range(len(x_train_transf_preprocessed_df.columns)):
                
                x_train_transf_preprocessed_df = x_train_transf_preprocessed_df.rename(columns={x_train_transf_preprocessed_df.columns[i]: f'c{i+1}'})

            logging.info('''x_train_transf dataframe columns renamed''')
            
            # print ("x_train_preprocessed head:", x_train_transf_preprocessed_df.head(5))
            
            over_samp = SMOTE(k_neighbors=1)
            
            logging.info ("oversampling initiated")
            
            x_train_os, y_train_os = over_samp.fit_resample (x_train_transf_preprocessed_df, y_train_transf)

            logging.info ("oversampling completed")
            
            constant_filter = VarianceThreshold(threshold=0)
            
            logging.info ('feature selection by filter methods initiated')
            
            x_train_c = constant_filter.fit_transform(x_train_os)

            logging.info ('constant filter applied on os x_train')
            
            print ('After dropping the constant features x_train_os shape is:', x_train_c.shape)

            constant_columns = [column for column in x_train_os.columns
                     if column not in x_train_os.columns [constant_filter.get_support()]]
            
            logging.info ('constant columns check completed')
            
            print ("Length Of Constant Features:", len (constant_columns))
            
            print ("Constant Features Are:", constant_columns)

            x_train_nc = x_train_os.drop (constant_columns, axis =1)

            quasicons_filter = VarianceThreshold(threshold=0.01)
            
            x_train_qc = quasicons_filter.fit_transform(x_train_nc)
            
            print ('After dropping the quasi constant features x_train shape is:', x_train_qc.shape)
            
            quasi_constant_features = [column for column in x_train_nc.columns
                     if column not in x_train_nc.columns [quasicons_filter.get_support()]]
            
            print ("Length Of Quasi Constant Features:", len (quasi_constant_features))
            
            print ("Quasi Constant Features Are:", quasi_constant_features)

            x_train_nqc = x_train_nc.drop (quasi_constant_features, axis =1)

            print ('shape after dropping quasi constant features:', x_train_qc.shape)

            duplicate_columns = []
            
            for i in range (0, len (x_train_nqc.columns)):
                 
                 col_1 = x_train_nqc.columns[i]
                 
                 for col_2 in x_train_nqc.columns[i+1:]:
                      if x_train_nqc[col_1].equals(x_train_nqc[col_2]):
                           duplicate_columns.append(col_2)
                           
            print ("Length Of Duplicate Features:", len (duplicate_columns))
            
            print ("Duplicate Features Are:", duplicate_columns)

            x_train_nd =  x_train_nqc.drop (duplicate_columns, axis =1)

            # vif = pd.DataFrame()
            
            # vif['vif'] = [variance_inflation_factor(x_train_os, i) for i in range (x_train_os.shape[1])]

            # logging.info("VIF iniitiated")
            
            # vif['features'] = x_train_os.columns

            # logging.info("VIF completed")

            # print (vif)
            
            # high_vif_columns = vif[vif['vif'] > 10]['features'].tolist()
            
            # print ("columns with vif>10:", high_vif_columns)

            # x_train_vif = x_train_os.drop (high_vif_columns, axis=1)

            # logging.info ("columns with vif>10 dropped")

            # print ("x_train shape after dropping high vif columns", x_train_vif.shape)

            x_test_transf_preprocessed = preprocessor.transform(x_test_transf)

            logging.info("Preprocessor applied on x_test_transf")

            x_test_transf_preprocessed_df = pd.DataFrame(x_test_transf_preprocessed)

            logging.info('''x_test_transf dataframe formed for feature selection by filter methods, vif and dimensionality
                         reduction by LDA''')
            
            for i in range(len(x_test_transf_preprocessed_df.columns)):
                
                x_test_transf_preprocessed_df = x_test_transf_preprocessed_df.rename(columns={x_test_transf_preprocessed_df.columns[i]: f'c{i+1}'})

            logging.info('''x_test_transf dataframe columns renamed''')

            x_test_nc = x_test_transf_preprocessed_df.drop (constant_columns, axis =1)

            logging.info('''constant columns dropped from x_test''')

            x_test_nqc = x_test_nc.drop (quasi_constant_features, axis =1)

            logging.info('''quasi constant columns dropped from x_test''')

            x_test_nd = x_test_nqc.drop (duplicate_columns, axis =1)

            logging.info('''duplicate columns dropped from x_test''')

            # x_test_vif = x_test_transf_preprocessed_df.drop (high_vif_columns, axis=1)

            # logging.info('''high VIF columns dropped from x_test''')

            # rf = RandomForestClassifier()

            # logging.info('''RFC initiated for boruta''')
            
            # rf.fit(x_train_os, y_train_os)

            # logging.info('''x_train and y_train fitted in rf''')
            
            # y_prediction_train_rf = rf.predict (x_train_os)

            # logging.info('''Prediction made on x_train''')

            # y_prediction_test_rf = rf.predict (x_test_transf_preprocessed_df)

            # logging.info('''Prediction made on x_test''')

            # train_data_accuracy_rf = accuracy_score(y_prediction_train_rf, y_train_os)

            # logging.info('''train data acuracy printed''')
            
            # test_data_accuracy_rf = accuracy_score(y_prediction_test_rf, y_test_transf)

            # logging.info('''test data acuracy printed''')

            # print ("RF Accuracy On Train Data:", train_data_accuracy_rf)
            
            # print ("RF Accuracy On Test Data:", test_data_accuracy_rf)

            # print ("RF F1 Score Train:", f1_score(y_train_os, y_prediction_train_rf, average='macro'))

            # logging.info("RF F1 score calculated on x_train")
            
            # print ("RF F1 Score Test:", f1_score(y_test_transf, y_prediction_test_rf, average='macro'))

            # logging.info("RF F1 score calculated on x_test")

            # print ("RF Classification Report Train:\n", classification_report (y_train_os, y_prediction_train_rf, digits = 4))

            # logging.info("RF Classification report obtained on x_train")

            # print ("RF Classification Report Test:\n", classification_report (y_test_transf, y_prediction_test_rf, digits = 4))

            # logging.info("RF Classification report obtained on x_test")

            # x_train_os_np = np.array (x_train_os)

            # x_test_transf_preprocessed_df_os = np.array (x_test_transf_preprocessed_df)

            # y_train_os_np = np.array(y_train_os)

            # y_test_transf_np = np.array(y_test_transf)

            # feat_selector = BorutaPy (rf, n_estimators = 'auto', verbose =2, random_state =1)

            # logging.info ("Boruta Feature Selection Started")

            # feat_selector.fit (x_train_os_np, y_train_os_np)

            # logging.info ("x train and y train fiited in boruta feature selector")

            # print (feat_selector.support_)

            # logging.info ("boruta feature selector printed")

            # print ("Ranking:", feat_selector.ranking_)

            # logging.info ("boruta feature selector ranking printed")

            # print ("No Of Significant Features:", feat_selector.n_features_)

            # logging.info ("boruta significant features printed")

            # selected_rf_features = pd.DataFrame ({'Features': list (x_train_os.columns), 'Ranking': feat_selector.ranking_})
            
            # logging.info ("df made by the features and their ranking")
            
            # selected_rf_features.sort_values(by='Ranking')

            # logging.info ("df made by the features and sorted by their ranking")

            # best_features = feat_selector.transform (x_train_os_np)

            # logging.info ("best features selected by boruta")
            
            # print (best_features.shape)

            # logging.info ("best features shape printed")
            
            # print (best_features)

            # logging.info ("best features printed")

            # boruta_columns = ['c3', 'c4', 'c5', 'c6']

            # x_train_boruta = x_train_os [boruta_columns]
            
            # print (x_train_boruta.head(2))

            # x_test_boruta = x_test_transf_preprocessed_df [boruta_columns]

            # print (x_test_boruta.head(2))

            # lda = LinearDiscriminantAnalysis(n_components=6)
            
            # ## number of components is equals to the number of unique classes y minus 1

            # x_train_lda = lda.fit_transform(x_train_nd_vif_np, y_train_transf_np)

            # x_test_lda = lda.transform(x_test_nd_vif_np)

            # print (lda.explained_variance_ratio_)

            # print ("x_train_nd_vif_np shape:", x_train_nd_vif_np.shape)

            # print ("x_test_nd_vif_np shape:", x_test_nd_vif_np.shape)

            # print ("x_train_lda dtype:", type (x_train_lda.dtype))

            # print ("x_test_lda dtype:", type (x_test_lda.dtype))
            
            # # print (x_train_nd.shape)
            
            train_arr = np.c_[np.array(x_train_nd), np.array(y_train_os)]
            
            logging.info("Combined the input features and target feature of the train set as an array.")
            
            test_arr = np.c_[np.array(x_test_nd), np.array(y_test_transf)]
            
            logging.info("Combined the input features and target feature of the test set as an array.")
            
            save_object(
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessor)
            
            logging.info("Saved preprocessing object.")
            
            return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path,)
        
        except Exception as e:
            raise CustomException(e, sys)