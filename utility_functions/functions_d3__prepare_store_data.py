import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from utility_functions.functions_0__common import get_columns
from utility_functions.functions_b__get_the_data import get_source_dataframe

def create_train_test_data(df_orig, categories, RANDOM_STATE=[], p_train_size=0.9, return_index=False, drop_nulls=True, no_dummies=False):
    df = df_orig.copy()

    if drop_nulls:
        df.dropna(inplace=True)

    if return_index:
        df.reset_index(inplace=True)

    if not no_dummies:
        df = convert_to_dummied(df, categories)
    else:
        # one_hot_max_size
        # for column in categories:
        #    df[column] = df[column].astype('category')
        pass

    try:
        ins = df.pop('index')
    except:
        # we must be at version 11
        ins = df.pop('iddd')

    df.insert(1, 'index2', ins)
    df.insert(0, 'index', ins)

    df_features = df[df.columns[2:]]
    df_labels = df.iloc[:, 0:2]

    features = df[df.columns[2:]].values
    labels = df.iloc[:, 0:2].values

    if not return_index:
        return train_test_split(features, labels, train_size=p_train_size, random_state=RANDOM_STATE)
    else:
        X_train1, X_test1, y_train1, y_test1 = train_test_split(features, labels, train_size=p_train_size,
                                                                random_state=RANDOM_STATE)
        X_train_index = X_train1[:, 0].reshape(-1, 1)
        y_train_index = y_train1[:, 0].reshape(-1, 1)
        X_test_index = X_test1[:, 0].reshape(-1, 1)
        y_test_index = y_test1[:, 0].reshape(-1, 1)
        X_train1 = X_train1[:, 1:]
        y_train1 = y_train1[:, 1].reshape(-1, 1)
        X_test1 = X_test1[:, 1:]
        y_test1 = y_test1[:, 1].reshape(-1, 1)

        return X_train1, X_test1, y_train1, y_test1, X_train_index, X_test_index, y_train_index, y_test_index, df_features, df_labels


def convert_to_dummied(df, categories):
    for column in categories:
        df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
        df.drop([column], axis=1, inplace=True)  # now drop the original column (you don't need it anymore),
    return df


def this_test_data(VERSION, test_data_only=False, drop_nulls=True, cloud_or_webapp_run=False, versioned=False):
    if not versioned:
        raise AttributeError("need to check all calls that don't use versioned=True, and rewrite if necessary")

    suffix = "_no_nulls" if drop_nulls else ""

    try:
        if versioned:
            print('getting test data for version', VERSION)
            #X_train = np.loadtxt(f"webapp_deployment/cache/X_train_v{VERSION}.csv", delimiter=",")
            #y_train = np.loadtxt(f"webapp_deployment/cache/y_train_v{VERSION}.csv", delimiter=",")
            X_test = np.loadtxt(f"webapp_deployment/cache/X_test_v{VERSION}.csv", delimiter=",")
            y_test = np.loadtxt(f"webapp_deployment/cache/y_test_v{VERSION}.csv", delimiter=",")
            #feature_names = np.loadtxt(f"webapp_deployment/cache/feature_names_v{VERSION}.csv", delimiter=",")
            feature_names_str = np.genfromtxt(f"webapp_deployment/cache/feature_names_v{VERSION}.csv",dtype='str')

            x = str(feature_names_str)
            #x = x[2:-2].replace("' '"," ")
            x = x.replace("' '"," ").replace("['","").replace("']","")

            feature_names = x.split(',')
            feature_names = [each.replace("'","") for each in feature_names]
            #print("feature_names",type(feature_names), feature_names)
        elif not test_data_only:
            print('getting suffix test data', VERSION)
            X_train = np.loadtxt(f"webapp_deployment/cache/X_train{suffix}.csv", delimiter=",")
            y_train = np.loadtxt(f"webapp_deployment/cache/y_train{suffix}.csv", delimiter=",")
            X_test = np.loadtxt(f"webapp_deployment/cache/X_test{suffix}.csv", delimiter=",")
            y_test = np.loadtxt(f"webapp_deployment/cache/y_test{suffix}.csv", delimiter=",")
    except Exception as e:
        print('ENDED UP IN GENERAL EXCEPTION', e)
        print(e)
        df, retrieval_type = get_source_dataframe(cloud_or_webapp_run=cloud_or_webapp_run, version=VERSION, folder_prefix='')

        if drop_nulls:
            df.dropna(inplace=True)

        # xxxfeatures = df[df.columns[:-1]].values
        # features = df[FEATURES].values
        # labels = df[LABEL].values
        # X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.9, random_state=RANDOM_STATE)
        X_train, X_test, y_train, y_test, feature_names = tt_split(VERSION, df)
        #print("feature_names", "feature_names")

        print('test_data_only', test_data_only)
        print('drop_nulls', drop_nulls)

        if versioned:
            if not test_data_only:
                np.savetxt(f"webapp_deployment/cache/X_train_v{VERSION}.csv", X_train, delimiter=",", fmt="%f")
                np.savetxt(f"webapp_deployment/cache/y_train_v{VERSION}.csv", y_train, delimiter=",", fmt="%f")
            np.savetxt(f"webapp_deployment/cache/X_test_v{VERSION}.csv", X_test, delimiter=",", fmt="%f")
            np.savetxt(f"webapp_deployment/cache/y_test_v{VERSION}.csv", y_test, delimiter=",", fmt="%f")
            np.savetxt(f"webapp_deployment/cache/feature_names_v{VERSION}.csv", [feature_names], delimiter=",", fmt="%s")
        else:
            if not test_data_only:
                suffix = '_no_nulls' if drop_nulls else ''
                print('suffix:', suffix)
                print('text:', f"webapp_deployment/cache/X_train{suffix}.csv")
                print()
                print(X_train)
                print()
                np.savetxt("webapp_deployment/cache/X_train_no_nulls.csv", X_train, delimiter=",")
                np.savetxt(f"webapp_deployment/cache/y_train{suffix}.csv", y_train, delimiter=",")

            np.savetxt(f"webapp_deployment/cache/X_test{suffix}.csv", X_test, delimiter=",")
            np.savetxt(f"webapp_deployment/cache/y_test{suffix}.csv", y_test, delimiter=",")

    if not test_data_only:
        return X_train, X_test, y_train, y_test

    return X_test, y_test, feature_names


def tt_split(VERSION, df, RANDOM_STATE=101, LABEL='Price'):
    debug_mode=False

    columns, booleans, floats, categories, custom, wildcard = get_columns(version=VERSION)

    for column in categories:
        df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
        df.drop([column], axis=1, inplace=True)  # now drop the original column (you don't need it anymore),
        if debug_mode:print("updated df!!!")
        if debug_mode:print(df.head(1))
        if debug_mode:print("df.columns[1:]", df.columns[1:])
        feature_names_internal = df.columns[1:]
        if debug_mode:print("returnable_columns", feature_names_internal)

    # features = df[df.columns[:-1]].values
    features = df[df.columns[1:]].values
    # features = df[FEATURES].values
    labels = df[LABEL].values
    X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.9, random_state=RANDOM_STATE)

    #print("feature_names_internal", feature_names_internal)
    return X_train, X_test, y_train, y_test, feature_names_internal

