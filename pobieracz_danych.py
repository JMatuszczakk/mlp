import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import os

def get_data(ticker, start, end, override=False, interval="1d"):
    nazwa_pliku = f'yf_{ticker}_{start}_{end}.csv'
    # sprawdź czy istnieje plik csv z danymi
    if not os.path.exists(nazwa_pliku) or override:
        # jeśli nie istnieje, pobierz dane z yfinance
        df = yf.download(ticker, start, end, interval=interval)
        # zapisz dane do pliku csv
        df.to_csv(nazwa_pliku)
    if not override:
        # jeśli istnieje, wczytaj dane z pliku csv
        try:
            df = pd.read_csv(nazwa_pliku)
        except e as Exception:
            print(e)
    return df