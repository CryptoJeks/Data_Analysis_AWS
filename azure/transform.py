import argparse

import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", dest='input_data', type=str)
    parser.add_argument("--output_datastore", dest='output_datastore',
                        type=str)
    
    args = parser.parse_args()
    
    diamond_prices = pd.read_csv(args.input_data)

    target = 'price'
    numeric_features = ['carat']
    categorical_features = ['shape', 'cut', 'color', 'clarity', 'report', 'type']

    X = diamond_prices.drop(columns=[target, 'id', 'url', 'date_fetched'])
    y = diamond_prices[target]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
    
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown='ignore', sparse=False), 
         categorical_features)
    )
    
    transformed_Xtrain = preprocessor.fit_transform(Xtrain)
    transformed_Xtest = preprocessor.transform(Xtest)
    
    output_data_store = args.output_datastore

    train_features_output_path = Path(output_data_store, "train_features.csv")
    train_labels_output_path = Path(output_data_store, "train_labels.csv")
    
    test_features_output_path = Path(output_data_store, "test_features.csv")
    test_labels_output_path = Path(output_data_store, "test_labels.csv")

    pd.DataFrame(transformed_Xtrain).to_csv(train_features_output_path, index=False)
    pd.DataFrame(transformed_Xtest).to_csv(test_features_output_path, index=False)

    ytrain.to_csv(train_labels_output_path, index=False)
    ytest.to_csv(test_labels_output_path, index=False)