from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pandas as pd
import pickle


def standardize_numerical_features(df, numerical_features):
    df = df.copy()
    scaler = StandardScaler()

    scaler.fit(df[numerical_features])

    scaler_data = pd.DataFrame(
        data=[scaler.mean_, scaler.var_, scaler.scale_],
        index=['mean', 'variance', 'scale'],
        columns=df[numerical_features].columns
    ).T

    scaler_data.index.name = 'feature'
    df[numerical_features] = scaler.transform(df[numerical_features])

    return df, scaler_data


def one_hot_encoding_on_categorical_features(df, categorical_features):
    for feature in categorical_features:
        one_hot_encoded_feature = pd.get_dummies(df[feature], drop_first=True, prefix=feature)
        df = df.join(one_hot_encoded_feature)

    return df.drop(columns=categorical_features)


def split_train_test(df, target_name):
    target = df[target_name]
    features = df.drop([target_name], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        train_size=0.70,
        test_size=0.30,
        random_state=42,
        stratify=target
    )
    print(f'Train size: {X_train.shape}')
    print(f'Test size: {X_test.shape}')

    return X_train, X_test, y_train, y_test


def train(X_train, y_train):
    model = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        penalty='l2',
        solver='sag',
        verbose=2
    )
    model.fit(X_train, y_train)

    return model


def predict_and_report(model, X, y, dataset_type):
    y_pred = model.predict_proba(X)
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred[:, 1])
    auc = metrics.auc(fpr, tpr)
    proportion = round(100 * y.mean(), 2)
    print(f"AUC {dataset_type} {auc}")
    print(f"Proporção de dianóstico tardio: {proportion}%")

    return auc


def get_categorical_features():
    return [
        'AP_CONDIC',
        'AP_TPUPS',
        'AP_TIPPRE',
        'AP_MN_IND',
        'AP_SEXO',
        'AP_RACACOR',
        'AP_UFDIF',
        'AQ_TRANTE',
        'AQ_CONTTR'
    ]

def get_numerical_features(df):
    numerical_variables = df.select_dtypes(include=['float64', 'int']).columns.tolist()
    numerical_variables.remove('tardio')
    numerical_variables.remove('AP_TPUPS')
    numerical_variables.remove('AP_TIPPRE')
    numerical_variables.remove('AP_RACACOR')
    numerical_variables.remove('AP_UFDIF')

    return numerical_variables


def main():
    df_mama = pd.read_csv('data/Banco_Datathon/Banco_Datathon/processed/mama.csv', index_col=0)
    print(f'Proporção de diagnóstico tardio (Mama): {df_mama.tardio.mean()}')

    numerical_features = get_numerical_features(df_mama)
    categorical_features = get_categorical_features()

    X_train, X_test, y_train, y_test = split_train_test(df_mama, 'tardio')

    X_train, _ = standardize_numerical_features(X_train, numerical_features)
    X_test, _ = standardize_numerical_features(X_test, numerical_features)

    X_train = one_hot_encoding_on_categorical_features(X_train, categorical_features)
    X_test = one_hot_encoding_on_categorical_features(X_test, categorical_features)

    X_train.fillna(-999, inplace=True)
    X_test.fillna(-999, inplace=True)

    best_model = train(X_train, y_train)
    _ = predict_and_report(best_model, X_train, y_train, 'Train')
    _ = predict_and_report(best_model, X_test, y_test, 'Test')

    with open('model/model_logit.pkl', 'wb') as f:
        pickle.dump(best_model, f)


if __name__ == '__main__':
    main()
