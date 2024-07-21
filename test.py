
from datetime import datetime
import pandas as pd
from pobieracz_danych import get_data
from pbWykrywacz_świeczek import wykryjŚwieczki
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

dane = get_data('AAPL', '2015-01-01', '2023-01-12', override=True)

# sprawienie, że kolumny są z małych liter
dane.columns = map(str.lower, dane.columns)
# przygotowanie danych do wykrywania świeczek
dane_do_świeczek = dane[['open', 'high', 'low', 'close']]
# wykrywanie świeczek
dane_ze_świeczkami = wykryjŚwieczki(dane_do_świeczek)
# usunięcie pierwszego wiersza, bo jest NaN
dane_ze_świeczkami = dane_ze_świeczkami.iloc[1:]

# przygotowanie danych do trenowania modelu
X = dane_ze_świeczkami
X = X.drop(['open', 'high', 'low'], axis=1)

Y = dane_ze_świeczkami['close']

# podziel dane na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=52, shuffle=False)

# uzupełnianie danych, gdzie wartość = NaN wartością średnią
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# deklaracja modelu
model = LinearRegression()

# trenowanie modelu
model.fit(X_train_imputed, y_train)

# generowanie predykcji dla zbioru testowego
predictions = model.predict(X_test_imputed)

# obiczanie mae, mse, r2
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


print(dane)
print(dane_ze_świeczkami)
