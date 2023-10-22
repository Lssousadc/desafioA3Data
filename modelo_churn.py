# Importação das bibliotecas e módulos

from libraries import *
from funcoes_churn import *

# READING DATA

data_churn = pd.read_csv('../dataRaw/customer_churn.csv', index_col = 'customerID')

# Aplicando o tratamento dos dados


data_churn_ = data_churn.copy()
data_churn_ = Tratamento.alteraTipoDado(data_churn_, columns = ['TotalCharges', 'MonthlyCharges'])
data_churn_ = Tratamento.fillNa(data_churn_, column = 'TotalCharges')
data_churn_ = Tratamento.alteraValores(df = data_churn_
                                       ,columns = ['MultipleLines'
                                                   ,'OnlineSecurity'
                                                   ,'OnlineBackup'
                                                   ,'DeviceProtection'
                                                   ,'TechSupport'
                                                   ,'StreamingTV'
                                                   ,'StreamingMovies'
                                                   ]
                                       ,dictToMap = {'No internet service': 'No'
                                                    ,'No phone service': 'No'})

data_churn_ = Tratamento.alteraValores(df = data_churn_
                                       ,columns = ['gender'                                            
                                                    , 'Partner'
                                                    , 'Dependents'                                                   
                                                    , 'PhoneService'
                                                    , 'MultipleLines'
                                                    , 'InternetService'
                                                    , 'OnlineSecurity'
                                                    , 'OnlineBackup'
                                                    , 'DeviceProtection'
                                                    , 'TechSupport'
                                                    , 'StreamingTV'
                                                    , 'StreamingMovies'
                                                    ,'PaperlessBilling'
                                                    ,'Churn'
                                                   ]
                                       ,dictToMap = {'No': '0'
                                                    ,'Yes': '1'
                                                    ,'Female': '1'
                                                    ,'Male': '0'})

data_churn_ = Tratamento.replaceCharacters(df = data_churn_)


# Preparacao da modelagem

X_train, X_test, y_train, y_test = Modelagem().prepara_variaveis(df = data_churn_)

#Treinamento

modelTrained = Modelagem().trainModel(model = RandomForestClassifier(n_estimators=100, random_state=123)
                                     ,X=X_train
                                     ,y=y_train)

# Salvando modelo para posterior aplicacao

joblib.dump(modelo, 'modelo_churn.joblib')

# Avaliação das metricas

prediction, y_pred = Modelagem().predicMetrics(modelTrained =modelTrained
                                              ,X_test = X_test
                                              ,y_test = y_test)

