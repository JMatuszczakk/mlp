from wykrywacz_paternów import zwróćDFall
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer

def create_model_1(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_model_2(input_dim):
    model = Sequential()
    model.add(Embedding(input_dim, output_dim=64, input_length=input_dim))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_model_3(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(action, model_generator, epochs=50, batch_size=64):
    df_all = zwróćDFall(action)
    # dump to txt
    df_all.to_csv(f'{action}.txt', sep='\t', encoding='utf-8')

    # Add target column
    df_all['target'] = (df_all['close'].shift(-1) > df_all['close']).astype(int)  # BINARY

    # Data split
    features = df_all.drop(['target'], axis=1)
    target = df_all['target']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=873)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')  # You can adjust the imputation strategy
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    y_train_imputed = imputer.fit_transform(y_train.values.reshape(-1, 1))
    y_test_imputed = imputer.transform(y_test.values.reshape(-1, 1))
