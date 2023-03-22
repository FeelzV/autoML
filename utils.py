from sklearn.model_selection import train_test_split
import pandas as pd


def load(name):
    data = pd.read_csv(f"datasets/{name}.csv")
    return train_test_split(data, test_size=0.2, random_state=0)


def create_test_train_datasets():
    files = ["blobs", "circles", "moons", "regression", "emotion", "titanic"]
    for file in files:
        df = pd.read_csv(f"datasets/{file}.csv")
        train, test = train_test_split(df, test_size=0.2, random_state=0)
        train.to_csv(f"datasets/{file}_train.csv", index=False)
        test.to_csv(f"datasets/{file}_test.csv", index=False)
