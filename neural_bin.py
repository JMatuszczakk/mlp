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

    best_accuracy = 0
    best_model = None

    for i, model_generator_func in enumerate([create_model_1, create_model_2, create_model_3], start=1):
        model = model_generator_func(X_train_imputed.shape[1])
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = model.fit(X_train_imputed, y_train_imputed, epochs=epochs, batch_size=batch_size,
                            validation_split=0.2, callbacks=[early_stopping])

        _, accuracy = model.evaluate(X_test_imputed, y_test_imputed)
        print(f"Model {i} Accuracy: {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    # Save the best model to a file, provide the model name
    model_name = input("Enter the best model name: ")
    best_model.save(f'{action}_best_acc-{best_accuracy}_{model_name}.h5')

    print(f"\nBest Model Information:\n")
    print(f"Model Name: {model_name}")
    print(f"Best Accuracy: {best_accuracy}")
    print(f"Architecture: {best_model.summary()}")

    return best_model

if __name__ == "__main__":
    input_data = "AAPL"
    trained_model = train_model(input_data, create_model_1)
    print("Best model has been successfully trained and saved to the file.")
