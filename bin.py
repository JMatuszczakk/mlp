from wykrywacz_paternów import zwróćDFall
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib  # Importuj bibliotekę joblib do zapisu modelu
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def trenujModel(akcja):
    dfAll = zwróćDFall(akcja)
    #dump to txt
    dfAll.to_csv(f'{akcja}.txt', sep='\t', encoding='utf-8')
    # Dodanie kolumny target
    dfAll['target'] = (dfAll['close'].shift(-1) > dfAll['close']).astype(int) #BINIARNY
    #Zamień obliczanie czy rośnie i dawanie 0 lub 1 na obliczanie o ile procent rośnie i dawanie tych procentów, uwzględnij wartość ujemną
    #dfAll['target'] = (dfAll['close'].shift(-1) / dfAll['close'] - 1) * 100 #ANALOGOWY

    # Podział danych
    features = dfAll.drop(['target'], axis=1)
    #usuń kolumny high, low, open
    features = features.drop(['high', 'low', 'open'], axis=1)
    print(features.head())
    target = dfAll['target']
    print("Target")
    print(target.head())
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=873)


    # Imputacja brakujących wartości
    imputer = SimpleImputer(strategy='mean')  # Możesz dostosować strategię imputacji
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    # Impuda brakujących wartości dla Y
    y_train_imputed = imputer.fit_transform(y_train.values.reshape(-1, 1))
    y_test_imputed = imputer.transform(y_test.values.reshape(-1, 1))
    #print head of X_train_imputed
    print(pd.DataFrame(X_train_imputed).head())   
    #pint head of Y_train
    print(pd.DataFrame(y_train).head())


    # Utworzenie i trenowanie modelu
    model = GaussianNB()
    model.fit(X_train_imputed, y_train_imputed)



    # Ocenianie modelug
    predictions = model.predict(X_test_imputed)
    accuracy = accuracy_score(y_test_imputed, predictions)
    print(f"Dokładność modelu: {accuracy}")

    # Zapisz model do pliku, podaj nazwe modelu
    makapaka = input("Podaj nazwe modelu: ")
    joblib.dump(model, f'{akcja}_dokl-{accuracy}_analogowy_{makapaka}.joblib')

    return model

if __name__ == "__main__":
    wejście = "AAPL"
    model = trenujModel(wejście)
    print("Model został pomyślnie trenowany i zapisany do pliku '")
