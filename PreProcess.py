import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocessData(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['Unnamed: 0', 'generation_name', 'city'])
    df = df.sample(frac = 1)
    dfX = df.drop(columns=['price'])
    dfY = df.iloc[:,-1]
    full_pipeline = ColumnTransformer([("num", StandardScaler(), ['vol_engine', 'year', 'mileage']), ("cat", OneHotEncoder(handle_unknown='ignore'), ['mark', 'model', 'fuel', 'province'])])
    x_train = full_pipeline.fit_transform(dfX.iloc[:int(0.8*len(df)),:])
    x_test = full_pipeline.transform(dfX.iloc[int(0.8*len(df)):,:])
    x_train = x_train.toarray()
    x_test = x_test.toarray()
    y_train = dfY.iloc[:int(0.8*len(df))]
    y_test = dfY.iloc[int(0.8*len(df)):]
    return x_train, y_train, x_test, y_test