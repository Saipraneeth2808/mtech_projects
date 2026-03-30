import pandas as pd

def prepare_dataframe(X, y):
    # Combine into one DataFrame
    df = pd.concat([X, y], axis=1)
    df.columns = [c.strip().lower() for c in df.columns]  # normalize column names

    # Inspect the data:
    print(df.shape)
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())  # check missing values

    # Handle missing values:
    # If any column has missing values, decide:
    # Drop rows with missing values (if few).
    # Or impute (mean/median for numeric, mode for categorical).
    df.fillna(df.median(), inplace=True)
    return df

X = pd.DataFrame({
    "age": [50, None],
    "sex": [1, 0]
})
y = pd.Series([0, 1], name="num")

df = prepare_dataframe(X, y)

assert df.isnull().sum().sum() == 0
