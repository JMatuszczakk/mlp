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


# Funkcja do zaokrąglania liczb
def truncate(n, decimals=0):
    multiplier = 10**decimals
    return int(n * multiplier) / multiplier

if __name__ == "__main__":
    # Ustawianie wejścia
    wejście = "AAPL"
    # Zwracanie df z wykrywacza paternów
    x = get_data(wejście, '2015-01-01', '2023-01-12', interval="1d")
    x.columns = map(str.lower, x.columns)
    df = x.copy()
    df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)
    x = df.copy()





    # przesunięcie danych o 1 w górę i dodanie do y (wartości, które chcemy przewidzieć), chodzi o to żeby przewidywało następne zamknięcie
    y = x['close'].shift(-1)
    # usunięcie high, low i close z x (x to dane wejściowe do modelu)
    x = x.drop(['high', 'low', 'close'], axis=1)
    # edit timestamp to datetime to timestamp
    x['date'] = pd.to_datetime(x['date'])
    x['date'] = x['date'].map(dt.datetime.toordinal)
    # naprawienie indexów, żeby się zgadzały
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)
    # trenowanie modelu i zwrócenie modelu, mae, mse, r2 do zmiennych
    model, mae, mse, r2 = trenujModel(wejście, x, y)
    # wypisanie informacji o modelu
    print("Model został \033[92m pomyślnie \033[0mtrenowany i zapisany do pliku.")
    print(f"Średnia pomyłka modelu regresyjnego: \033[93m{truncate(mae, 2)}$ \033[0m")
    print(f"Dokładność procentowa modelu regresyjnego: \033[96m{truncate(100 - (mae/165)*100, 2)}% \033[0m")
    print(f"Dopasowanie modelu regresyjnego: \033[92m{truncate(r2, 4)*100}% \033[0m")
    # wypisanie procentu pomyłki dla różnicy dwóch ostatnich wartości zamknięcia
  
    #obliczanie procentu pomyłki
    różnica = 2.3
    procent_pomyłki = mae/różnica*100
    # wypisanie informacji o procentach pomyłki
    print(f"Procent pomyłki: \033[93m{procent_pomyłki}% \033[0m")
    print(f"Różnica: \033[93m{różnica} \033[0m")