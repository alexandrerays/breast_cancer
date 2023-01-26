from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import pickle
from lightgbm import LGBMClassifier


def dtype_category():
    return {
        'AP_TPUPS': 'category',
        'AP_TIPPRE': 'category',
        'AP_MN_IND': 'category',
        'AP_SEXO': 'category',
        'AP_RACACOR': 'category',
        'AP_UFDIF': 'bool',
        'AQ_TRANTE': 'category',
        'AQ_CONTTR': 'category'
    }


def append_target(X, y):
    complete_dataset = X.copy()
    y = y.reset_index(drop=True)
    complete_dataset['tardio'] = y

    return complete_dataset


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
    model = LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        class_weight='balanced',
        random_state=42,
        verbose=2,
        n_estimators=3000
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


def get_numerical_features(df):
    infra_columns = df.filter(regex='^rf_|equipes_|rh_|est_|hosp_|ubs_|dianose|hop_').columns.tolist()

    return infra_columns + ['AP_NUIDADE']


def main():
    df_mama = pd.read_csv(
        'data/Banco_Datathon/Banco_Datathon/processed/mama.csv',
        dtype=dtype_category(),
        index_col=0
    )
    print(f'Proporção de diagnóstico tardio (Mama): {df_mama.tardio.mean()}')

    X_train, X_test, y_train, y_test = split_train_test(df_mama, 'tardio')

    model = train(X_train, y_train)
    _ = predict_and_report(model, X_train, y_train, 'Train')
    _ = predict_and_report(model, X_test, y_test, 'Test')

    df_train = append_target(X_train, y_train)
    df_test = append_target(X_test, y_test)

    df_train.to_csv('data/Banco_Datathon/Banco_Datathon/processed/train_v4.csv')
    df_test.to_csv('data/Banco_Datathon/Banco_Datathon/processed/test_v4.csv')

    with open('model/model_lgbm_v4.pkl', 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    main()
