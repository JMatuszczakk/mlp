import streamlit as st
import plotly.graph_objects as go
from wykrywacz_paternów import wyświetlWykresZKropkami
import requests
from pbWykrywacz_świeczek import wykryjŚwieczki
from pobieracz_danych import get_data

def wyślijWiadomośc(wiadomość):
    requests.post("https://ntfy.sh/xxxxx",
    data=wiadomość.encode(encoding='utf-8'))

st.title("Przewidywanie akcji")
st.header("Regresja liniowa")

if __name__ == "__main__":
    wejście = st.text_input("Wpisz nazwę akcji", "AAPL")

    dane = get_data('AAPL', '2022-12-01', '2023-01-12', override=True)

    # sprawienie, że kolumny są z małych liter
    dane.columns = map(str.lower, dane.columns)
    # przygotowanie danych do wykrywania świeczek
    dane_do_świeczek = dane[['open', 'high', 'low', 'close']]
    # wykrywanie świeczek
    dane_ze_świeczkami = wykryjŚwieczki(dane_do_świeczek)
    # usunięcie pierwszego wiersza, bo jest NaN
    dane_ze_świeczkami = dane_ze_świeczkami.iloc[1:]

    # Line chart of closing price
    fig = go.Figure(data=go.Scatter(x=dane_ze_świeczkami.index, y=dane_ze_świeczkami['close'], mode='lines', name='Closing Price'))

    # Add points for each candlestick formation
    formations = ['inverted_hammer', 'hammer', 'bullish_engulfing', 'doji', 'doji_star', 'dragonfly_doji', 'piercing_pattern', 'morning_star', 'rain_drop', 'hanging_man', 'gravestone_doji', 'bearish_engulfing', 'morning_star_doji', 'rain_drop_doji', 'star', 'shooting_star', 'bearish_harami', 'bullish_harami', 'dark_cloud_cover']
    
    for index, row in dane_ze_świeczkami.iterrows():
        for formation in formations:
            if row[formation]:
                fig.add_trace(go.Scatter(x=[index], y=[row['close']], mode='markers', name=formation.replace('_', ' ').title()))

    fig.update_layout(xaxis_title='Date', yaxis_title='Closing Price', title='Closing Price with Candlestick Formations')

    # Display the chart on Streamlit
    st.plotly_chart(fig)