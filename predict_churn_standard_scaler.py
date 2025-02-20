import pandas as pd
import numpy as np
from pycaret.classification import predict_model, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

model = load_model('Best_Model_Churn_Standard_Scaler')

def load_data(filepath):
    """
    Loads churn data into a DataFrame from a string filepath.
    """
    df = pd.read_csv(filepath, index_col='customerID')
    return df


def make_predictions(df, threshold=0.70):
    """
    Uses the pycaret best model to make predictions on data in the df dataframe.
    Rounds up to 1 if greater than or equal to the threshold.
    """
    predictions = predict_model(model, data=df)
    predictions['Churn_prediction'] = (predictions['prediction_score'] >= threshold)
    predictions['Churn_prediction'].replace({True: 'Churn', False: 'No Churn'}, inplace=True)
    drop_cols = predictions.columns.tolist()
    drop_cols.remove('Churn_prediction')
    return predictions.drop(drop_cols, axis=1)


if __name__ == "__main__":
    df = load_data('./data/new_churn_data.csv')
    #df_copy = df.copy()
    #df = df.drop('tenure', axis=1)
    if 'charge_per_tenure' in df.columns:
        df = df.drop('charge_per_tenure', axis=1)
    #df = df.drop('PhoneService', axis=1)
    #df = df.drop('Contract', axis=1)
    #df = df.drop('PaymentMethod', axis=1)
    scaler = StandardScaler()
    #df_copy = df.copy()
    df.iloc[:,:] = scaler.fit_transform(df)
    #scaled_data
    #df.iloc[:,:] = Normalizer(norm='l1').fit_transform(df)
    #df['Churn'] = df_copy['Churn']
    #df['tenure'] = df['tenure'] / df['tenure'].max()
    #df['MonthlyCharges'] = df['MonthlyCharges'] / df['MonthlyCharges'].max()
    #df['TotalCharges'] = df['TotalCharges'] / df['TotalCharges'].max()
    #column_names = ['tenure', 'MonthlyCharges', 'TotalCharges']
    column_names = ['tenure', 'PhoneService', 'Contract', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)