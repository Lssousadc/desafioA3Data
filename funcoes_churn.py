"""
Este arquivo compreende as classes e metodos para desenvolvimento da analise
"""

from libraries import * 

class EDA:
    
    """
    Engloba as funcoes referentes as visualizacoes - exploratory data analysis
    """
    
    def visualizaValoresCategoricos(self, df):
        
        """
        Objetiva criar a visualizacao, em boxplot, dos valores unicos das variaveis categoricas
        """

        num_cols = 3
        num_plots = len(df.columns)
        num_rows = (num_plots - 1) // num_cols + 1

        # Cria subplots em uma matriz
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4*num_rows))

        # Cria gráficos de barras para cada coluna
        for i, column in enumerate(df.columns):
            ax = axes[i // num_cols, i % num_cols]
            sns.countplot(data=df, x=column, ax=ax, palette="viridis")
            ax.set_title(f'{column}')
            ax.set_xlabel('Valores')
            ax.set_ylabel('Contagem') 
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        for i in range(num_plots, num_rows * num_cols):
            fig.delaxes(axes.flatten()[i])

        plt.tight_layout()
        fig.savefig("valoresUnicosVariaveisCategoricas.png")
        plt.show
        
        
    def createHeatMap(self, df, columns, yColumn):
        
        """
        Cria um HeatMap entre as variaveis categoricas e a variavel resposta (Churn)
        """
        
        num_columns = len(columns)
        num_rows = int(np.ceil(num_columns / 3))  

        fig, axes = plt.subplots(num_rows, 3, figsize=(15, 4 * num_rows))

        for i, column in enumerate(columns):
            row = i // 3  
            col = i % 3   

            crosstab = pd.crosstab(df[column], df[yColumn])
            ax = axes[row, col]  

            sns.heatmap(crosstab, annot=True, cmap="YlGnBu", fmt='d', ax=ax, cbar=False)
            ax.set_title(f'{column} x {yColumn}')

        # Remove subplots não utilizados
        for i in range(num_columns, num_rows * 3):
            fig.delaxes(axes.flatten()[i])

        plt.tight_layout()
        fig.savefig("heatMapVariaveisCategoricas.png")
        plt.show()
        
    
    def visualizaVariaveisContinuas(self, df):
        
        """
        Observa a distribuicao da variavel resposta (Churn) com relacao as variaveis continuas
        """

        columns_num = ['TotalCharges', 'MonthlyCharges', 'tenure']
        num_cols = 3
        num_plots = len(columns_num)

        custom_palette = ["#5a7e9e", "#00cc99"]

        if num_plots > num_cols:
            num_cols = min(num_plots, num_cols)
            num_rows = (num_plots - 1) // num_cols + 1
        else:
            num_cols = num_plots
            num_rows = 1

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))

        for i, column in enumerate(columns_num):
            if num_rows == 1:
                ax = axes[i % num_cols]
            else:
                ax = axes[i // num_cols, i % num_cols]

            sns.boxplot(data=df, x=df['Churn'], y=df[column], ax=ax, palette=custom_palette)

        fig.savefig("distribuicaoVariaveisContinuas.png")
        plt.show()
        
        
    def chi2Analysis(self, df, columns, yColumn):
        
        """
        Realiza o teste de Qui-Quadrado para avaliar possiveis associacoes entre variaveis
        """
        df_chi = pd.DataFrame(columns=['VARIAVEL', 'P-VALOR', 'CORRELACAO'])
        dict_chi_list = []

        for column in columns:
            crosstab = pd.crosstab(df[column], df[yColumn])
            chi2, p, dof, expected = chi2_contingency(crosstab)

            # Adicionar valores ao DataFrame df_chi
            correlation = 'Sim' if p < 0.05 else 'Não'
            dict_chi = {'VARIAVEL': column, 'P-VALOR': p, 'CORRELACAO': correlation}
            dict_chi_list.append(dict_chi)


        df_chi = pd.DataFrame.from_dict(dict_chi_list)

        return df_chi
    
class Tratamento:
    
    """
    Engloba todas as funcoes referentes ao tratamento da base
    """
    
    def alteraValores(df, columns, dictToMap):
        
        """
        Realiza um map para alteracao de valores das colunas desejadas
        """

        for column in df[columns]:

            df[column] = df[column].replace(dictToMap)


        return df
        
    def replaceCharacters(df):
        """
        Substitui espaços em branco, '(' e ')' em todas as colunas do DataFrame.
        """
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = df[column].str.replace(' ', '_')
                df[column] = df[column].str.replace('(', '_')
                df[column] = df[column].str.replace(')', '')

        return df
    
      
    
    def fillNa(df, column):
        
        """
        Preenchimento de nulos: se for numerico, usar a mediana, se for str, usar o valor mais frequente
        """

        if df[column].dtype != 'object':

            df[column] = df[column].fillna(df[column].median())

        else:

            df[column] = df[column].fillna(df[column].mode().iloc[0])

        return df
            
        
    def alteraTipoDado(df, columns, to_int = True):
        
        """
        Alterando o tipo de dados
        """
    
            
        if to_int:

            for column in columns:

                df[column] = df[column].str.replace(',', '.').astype(float)

        else:

            df[column] = df[column].astype(str)

        return df
            

class Modelagem:
    
    

    def prepara_variaveis(self, df):
        
        """
        Preparacao dos dados para Modelagem
        """

        df = df.copy()
        
            
        # Dividindo em X e y
        
        X = df.drop(columns = ['Churn'])
        y = df[['Churn']]
        
        
        # Treino e teste
        
        X_train, X_test, y_train, y_test = train_test_split(X
                                                            , y
                                                            , test_size=0.25
                                                            , random_state=123
                                                            , shuffle=True)
        
             
        # encoding X_train
        
        X_train_encoded = pd.get_dummies(X_train
                                        ,columns = ['InternetService'
                                                    ,'Contract'                                                
                                                    ,'PaymentMethod'
                                                   ]
                                        ,drop_first=True
                                         ,dtype=int
                                         )

        X_test_encoded = pd.get_dummies(X_test
                                        ,columns = ['InternetService'
                                                    ,'Contract'                                                
                                                    ,'PaymentMethod'
                                                   ]
                                        ,drop_first=True
                                        ,dtype=int
                                       )
        
        
        
        
        # CRIANDO COLUNAS EM X_TEST QUE  APAREÇAM NO TREINO

        for col in X_train_encoded.columns:
            if col not in X_test_encoded.columns:
                X_test_encoded[col] = 0

        X_test_encoded = X_test_encoded[X_train_encoded.columns]
        
        # BALANCEAMENTO DOS DADOS - UNDERSAMPLING
        
        us = RandomUnderSampler()
        X_train_resampled, y_train_resampled = us.fit_resample(X_train_encoded, y_train)
        
        X_train_resampled = X_train_resampled.astype(float)
        


        print("Quantidade de dados de treino: {0}".format(X_train_resampled.shape[0]))
        print("Quantidade de dados de teste: {0}".format(X_test_encoded.shape[0]))
        return X_train_resampled, X_test_encoded, y_train_resampled, y_test
    
    
    def checkScore(self, X, y):
        
        model_rf = RandomForestClassifier(n_estimators=100, random_state=123)       
        kf = KFold(n_splits=5, shuffle=True, random_state=123)
        scorer = make_scorer(f1_score)
        scores = cross_val_score(model_rf, X, y, cv=kf, scoring=scorer)
        
        return scores, scores.mean()
    
    
    def trainModel(self, model, X, y):
        
        """
        Realiza o treinamento do modelo
        """
        
        modelTrained = model.fit(X, y)
        
        return modelTrained
    
    
    def predicMetrics(self, modelTrained, X_test, y_test):
        
        """
        Faz a predicao e visualizacao das metricas do modelo
        """

        pred = modelTrained.predict(X_test)
        pred_DF = pd.DataFrame(pred).set_index(y_test.index)
        df_model = y_test.merge(pred_DF, right_index = True, left_index = True)
        
        print(classification_report(y_test, pred))
        
        return df_model, pred
       
    

            
        
            
            
    