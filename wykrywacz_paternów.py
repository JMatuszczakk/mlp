
from polygon import RESTClient #API do danych giełdowych
import datetime #Do pobierania daty i czasu
import pandas as pd #Do tworzenia df
import streamlit as st #Do tworzenia aplikacji webowej
from candlestick import candlestick #Do wykrywania formacji świecowych
import plotly
import plotly.graph_objects as go
import talib
import os
#########Pobieranie danych z API i dodawanie zwracanie gotowego df z open, high, low, close i timestamp jako index#########

import time

def pobierzDaneZAPI(akcja):
    # Check if CSV file exists
    csv_file = f'{akcja}_DUMP.txt'
    if not os.path.exists(csv_file):
        # Generate the CSV file
        df = pd.DataFrame(columns=["open", "high", "low", "close", "timestamp"])
        current_date = datetime.datetime.now()
        date_two_years_ago = current_date - datetime.timedelta(days=10)
        client = RESTClient(api_key="DpF3sM4OKCg_CrYbr5vTbTdimzmatVhB") # Tworzenie klienta API
        while date_two_years_ago < current_date:
            try:
                date_start = date_two_years_ago.strftime("%Y-%m-%d") # Data początkowa
                date_end = (date_two_years_ago + datetime.timedelta(days=60)).strftime("%Y-%m-%d") # Data końcowa (30 dni później)
                aggs = client.get_aggs(
                    akcja, # Symbol akcji
                    1, # Mnożnik
                    "hour", # Co ile jedna dana
                    date_start, # Data początkowa
                    date_end, # Data końcowa
                    limit=50000
                )
                temp_df = pd.DataFrame(aggs, columns=["open", "high", "low", "close", "timestamp"]) # Przetwarzanie danych z API do tymczasowego df
                temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'], unit='ms') # Zamiana timestamp na datę i czas
                df = pd.concat([df, temp_df]) # Dodawanie tymczasowego df do głównego df
                date_two_years_ago += datetime.timedelta(days=60) # Przesunięcie daty początkowej o 30 dni
                # Dump to CSV
                df.to_csv(csv_file, sep='\t', encoding='utf-8')
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Waiting for 1 minute...")
                time.sleep(60) # Oczekiwanie 1 minuty
    else:
        print(f"CSV file '{csv_file}' already exists.")
        # Retrieve data from CSV file
        df = pd.read_csv(csv_file, sep='\t', encoding='utf-8')
        # Process the data as needed

    df.set_index("timestamp", inplace=True)  # Set timestamp as index
    df = df[~df.index.duplicated(keep='first')] 

    return df # Zwracanie df

def wykryjŚwieczki(akcja):
    df0 = pobierzDaneZAPI(akcja) #Zaposywanie df z funkcji do zmiennej globalnej
    df = candlestick.inverted_hammer(df0, target='inverted_hammer') #Wykrywanie formacji świecowych - inverted_hammer
    df2 = candlestick.hammer(df0, target='hammer') #Wykrywanie formacji świecowych - hammer
    df3 = candlestick.bullish_engulfing(df0, target='bullish_engulfing') #Wykrywanie formacji świecowych - bullish_engulfing
    df4 = candlestick.doji(df0, target='doji') #Wykrywanie formacji świecowych - doji
    df5 = candlestick.doji_star(df0, target='doji_star') #Wykrywanie formacji świecowej -doji_star
    df6 = candlestick.dragonfly_doji(df0, target='dragonfly_doji') #Wykrywanie formacji świecowej - dragonfly_doji
    df7 = candlestick.piercing_pattern(df0, target='piercing_pattern') #Wykrywanie formacji świecowej - piercing_pattern
    df8 = candlestick.morning_star(df0, target='morning_star') #Wykrywanie formacji świecowej - morning_star
    df9 = candlestick.rain_drop(df0, target='rain_drop') #Wykrywanie formacji świecowej - rain_drop
    df10 = candlestick.hanging_man(df0, target='hanging_man')
    df11 = candlestick.gravestone_doji(df0, target='shooting_star')
    df12 = candlestick.bearish_engulfing(df0, target='bearish_engulfing')
    df13 = candlestick.morning_star_doji(df0, target='morning_star_doji')
    df14 = candlestick.rain_drop_doji(df0, target='rain_drop_doji')
    df15 = candlestick.star(df0, target='star')
    df16 = candlestick.shooting_star(df0, target='shooting_star')
    df17 = candlestick.bearish_harami(df0, target='bearish_harami')
    df18 = candlestick.bullish_harami(df0, target='bullish_harami')
    df19 = candlestick.dark_cloud_cover(df0, target='dark_cloud_cover')

    dfAll = df0.copy()
    dfAll['inverted_hammer'] = df['inverted_hammer']
    dfAll['hammer'] = df2['hammer']
    dfAll['bullish_engulfing'] = df3['bullish_engulfing']
    dfAll['doji'] = df4['doji']
    dfAll['doji_star'] = df5['doji_star']
    dfAll['dragonfly_doji'] = df6['dragonfly_doji']
    dfAll['piercing_pattern'] = df7['piercing_pattern']
    dfAll['morning_star'] = df8['morning_star']
    dfAll['rain_drop'] = df9['rain_drop']
    dfAll['hanging_man'] = df10['hanging_man']
    dfAll['gravestone_doji'] = df11['shooting_star']
    dfAll['bearish_engulfing'] = df12['bearish_engulfing']
    dfAll['morning_star_doji'] = df13['morning_star_doji']
    dfAll['rain_drop_doji'] = df14['rain_drop_doji']
    dfAll['star'] = df15['star']
    dfAll['shooting_star'] = df16['shooting_star']
    dfAll['bearish_harami'] = df17['bearish_harami']
    dfAll['bullish_harami'] = df18['bullish_harami']
    dfAll['dark_cloud_cover'] = df19['dark_cloud_cover']


    return df, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19, dfAll #Zwracanie df

import pandas as pd
import os

def zwróćDFall(akcja):
    filename = f"{akcja}_data.csv"  # Define the filename based on the action
    if os.path.isfile(filename):  # Check if the file exists
        dfAll = pd.read_csv(filename)  # Read data from the file
    else:
        df0 = pobierzDaneZAPI(akcja)  # Zaposywanie df z funkcji do zmiennej globalnej
        df = candlestick.inverted_hammer(df0, target='inverted_hammer')  # Wykrywanie formacji świecowych - inverted_hammer
        df2 = candlestick.hammer(df0, target='hammer')  # Wykrywanie formacji świecowych - hammer
        df3 = candlestick.bullish_engulfing(df0, target='bullish_engulfing')  # Wykrywanie formacji świecowych - bullish_engulfing
        df4 = candlestick.doji(df0, target='doji')  # Wykrywanie formacji świecowych - doji
        df5 = candlestick.doji_star(df0, target='doji_star')  # Wykrywanie formacji świecowej -doji_star
        df6 = candlestick.dragonfly_doji(df0, target='dragonfly_doji')  # Wykrywanie formacji świecowej - dragonfly_doji
        df7 = candlestick.piercing_pattern(df0, target='piercing_pattern')  # Wykrywanie formacji świecowej - piercing_pattern
        df8 = candlestick.morning_star(df0, target='morning_star')  # Wykrywanie formacji świecowej - morning_star
        df9 = candlestick.rain_drop(df0, target='rain_drop')  # Wykrywanie formacji świecowej - rain_drop
        df10 = candlestick.hanging_man(df0, target='hanging_man')
        df11 = candlestick.gravestone_doji(df0, target='shooting_star')
        df12 = candlestick.bearish_engulfing(df0, target='bearish_engulfing')
        df13 = candlestick.morning_star_doji(df0, target='morning_star_doji')
        df14 = candlestick.rain_drop_doji(df0, target='rain_drop_doji')
        df15 = candlestick.star(df0, target='star')
        df16 = candlestick.shooting_star(df0, target='shooting_star')
        df17 = candlestick.bearish_harami(df0, target='bearish_harami')
        df18 = candlestick.bullish_harami(df0, target='bullish_harami')
        df19 = candlestick.dark_cloud_cover(df0, target='dark_cloud_cover')

        dfAll = df0.copy()
        dfAll['inverted_hammer'] = df['inverted_hammer']
        dfAll['hammer'] = df2['hammer']
        dfAll['bullish_engulfing'] = df3['bullish_engulfing']
        dfAll['doji'] = df4['doji']
        dfAll['doji_star'] = df5['doji_star']
        dfAll['dragonfly_doji'] = df6['dragonfly_doji']
        dfAll['piercing_pattern'] = df7['piercing_pattern']
        dfAll['morning_star'] = df8['morning_star']
        dfAll['rain_drop'] = df9['rain_drop']
        dfAll['hanging_man'] = df10['hanging_man']
        dfAll['gravestone_doji'] = df11['shooting_star']
        dfAll['bearish_engulfing'] = df12['bearish_engulfing']
        dfAll['morning_star_doji'] = df13['morning_star_doji']
        dfAll['rain_drop_doji'] = df14['rain_drop_doji']
        dfAll['star'] = df15['star']
        dfAll['shooting_star'] = df16['shooting_star']
        dfAll['bearish_harami'] = df17['bearish_harami']
        dfAll['bullish_harami'] = df18['bullish_harami']
        dfAll['dark_cloud_cover'] = df19['dark_cloud_cover']


        dfAll.to_csv(filename, index=False)  # Save dfAll to a file

    return dfAll  # Zwracanie df

def wyświetlWykresZKropkami(akcja):
    df, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19, dfAll = wykryjŚwieczki(akcja) #Zapisywanie df z funkcji do zmiennych globalnych
    fig = go.Figure() #Tworzenie wykresu
    #dodawanie nazwy wykresu
    st.write(dfAll)
    fig.update_layout(
        title={
            'text': "Formacje świecowe dla akcji " + akcja,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    #Dodawanie danych do wykresu - close price
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close Price'))
    #Dodawanie formacji świecowych do wykresu - inverted_hammer, hammer, bullish_engulfing, doji, doji_star
    fig.add_trace(go.Scatter(x=df[df['inverted_hammer'] == True].index, y=df[df['inverted_hammer'] == True]['close'], name='Inverted Hammer', mode='markers', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df2[df2['hammer'] == True].index, y=df2[df2['hammer'] == True]['close'], name='Hammer', mode='markers', marker=dict(color='green')))
    fig.add_trace(go.Scatter(x=df3[df3['bullish_engulfing'] == True].index, y=df3[df3['bullish_engulfing'] == True]['close'], name='Bullish Engulfing', mode='markers', marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=df4[df4['doji'] == True].index, y=df4[df4['doji'] == True]['close'], name='Doji', mode='markers', marker=dict(color='yellow')))
    fig.add_trace(go.Scatter(x=df5[df5['doji_star'] == True].index, y=df5[df5['doji_star'] == True]['close'], name='Doji Star', mode='markers', marker=dict(color='purple')))
    fig.add_trace(go.Scatter(x=df6[df6['dragonfly_doji'] == True].index, y=df6[df6['dragonfly_doji'] == True]['close'], name='Dragonfly Doji', mode='markers', marker=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df7[df7['piercing_pattern'] == True].index, y=df7[df7['piercing_pattern'] == True]['close'], name='Piercing Pattern', mode='markers', marker=dict(color='pink')))
    fig.add_trace(go.Scatter(x=df8[df8['morning_star'] == True].index, y=df8[df8['morning_star'] == True]['close'], name='Morning Star', mode='markers', marker=dict(color='brown')))
    fig.add_trace(go.Scatter(x=df9[df9['rain_drop'] == True].index, y=df9[df9['rain_drop'] == True]['close'], name='Rain Drop', mode='markers', marker=dict(color='black')))
    fig.add_trace(go.Scatter(x=df10[df10['hanging_man'] == True].index, y=df10[df10['hanging_man'] == True]['close'], name='Hanging Man', mode='markers', marker=dict(color='gray')))
    fig.add_trace(go.Scatter(x=df11[df11['shooting_star'] == True].index, y=df11[df11['shooting_star'] == True]['close'], name='Shooting Star', mode='markers', marker=dict(color='cyan')))
    fig.add_trace(go.Scatter(x=df12[df12['bearish_engulfing'] == True].index, y=df12[df12['bearish_engulfing'] == True]['close'], name='Bearish Engulfing', mode='markers', marker=dict(color='magenta')))
    fig.add_trace(go.Scatter(x=df13[df13['morning_star_doji'] == True].index, y=df13[df13['morning_star_doji'] == True]['close'], name='Morning Star Doji', mode='markers', marker=dict(color='olive')))
    fig.add_trace(go.Scatter(x=df14[df14['rain_drop_doji'] == True].index, y=df14[df14['rain_drop_doji'] == True]['close'], name='Rain Drop Doji', mode='markers', marker=dict(color='salmon')))
    fig.add_trace(go.Scatter(x=df15[df15['star'] == True].index, y=df15[df15['star'] == True]['close'], name='Star', mode='markers', marker=dict(color='teal')))
    fig.add_trace(go.Scatter(x=df16[df16['shooting_star'] == True].index, y=df16[df16['shooting_star'] == True]['close'], name='Shooting Star', mode='markers', marker=dict(color='gold')))
    fig.add_trace(go.Scatter(x=df17[df17['bearish_harami'] == True].index, y=df17[df17['bearish_harami'] == True]['close'], name='Bearish Harami', mode='markers', marker=dict(color='darkblue')))
    fig.add_trace(go.Scatter(x=df18[df18['bullish_harami'] == True].index, y=df18[df18['bullish_harami'] == True]['close'], name='Bullish Harami', mode='markers', marker=dict(color='darkgreen')))
    fig.add_trace(go.Scatter(x=df19[df19['dark_cloud_cover'] == True].index, y=df19[df19['dark_cloud_cover'] == True]['close'], name='Dark Cloud Cover', mode='markers', marker=dict(color='darkred')))
    dfAll
    st.plotly_chart(fig) #Wyświetlanie wykresu

