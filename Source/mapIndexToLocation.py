import pandas as pd


def mapIndexToLocation(index):
    return pd.read_csv('../Data/Raw/locations.csv')['location'].get(index)
