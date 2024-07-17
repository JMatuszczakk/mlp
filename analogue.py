from wykrywacz_paternów import zwróćDFall
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from pobieracz_danych import get_data
import datetime as dt
from pbWykrywacz_świeczek import wykryjŚwieczki
def trenujModel(akcja, x, y):
    # podziel dane na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=52, shuffle=False)

    # uzupełnianie danych, gdzie wartość = NaN wartością średnią
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Tworzenie modelu
    model = LinearRegression()

    # Treanowanie modelu
    model.fit(X_train_imputed, y_train)

    # Usuwanie rzędów z wartościami NaN
    y_test = y_test.dropna()
    X_test_imputed = X_test_imputed[:len(y_test)]

    # generowanie predykcji dla zbioru testowego
    predictions = model.predict(X_test_imputed)

    # obiczanie mae, mse, r2
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    # Zapisywanie modelu do pliku
    joblib.dump(model, f'{akcja}_mae={mae:.4f}_mse={mse:.4f}_r2={r2:.4f}_model.joblib')

    # Zwracanie modelu i wartości mae, mse, r2
    return model, mae, mse, r2
