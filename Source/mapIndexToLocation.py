import pandas as pd


def mapIndexToLocation(index):
    return pd.read_csv('../Data/Clean/cleaned.csv')['location'].get(index)
