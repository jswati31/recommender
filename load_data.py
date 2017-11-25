# processing data

import pandas as pd
import numpy as np

data_path = 'data/'

#features to be converted to one-hot encodings.
cols_for_one_hot = ['channel', 'ci_month', 'ci_quarter', 'co_month', 'co_quarter', 'hotel_continent', 'ci_day',
                    'ci_dayofweek', 'co_day', 'co_dayofweek',
                    'hotel_country', 'hotel_market', 'month', 'posa_continent', 'quarter',
                    'site_name', 'srch_adults_cnt', 'srch_children_cnt', 'srch_destination_type_id', 'srch_rm_cnt',
                    'stay_span', 'user_location_city', 'user_location_country', 'user_location_region']

#features - no one-hot encodings.
drop_list = ['user_id', 'srch_destination_id', 'orig_destination_distance', 'srch_destination_iddest']


# Convert the data into one-hot encodings format
def one_hot(df, cols):
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df


# convert the datetime fields to day, month, quarter and day of week
def calc_fast_features(df, destinations):
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["srch_ci"] = pd.to_datetime(df["srch_ci"], format='%Y-%m-%d', errors="coerce")
    df["srch_co"] = pd.to_datetime(df["srch_co"], format='%Y-%m-%d', errors="coerce")

    props = {}
    for prop in ["month", "quarter"]:
        props[prop] = getattr(df["date_time"].dt, prop)

    carryover = [p for p in df.columns if p not in ["date_time", "srch_ci", "srch_co"]]
    for prop in carryover:
        props[prop] = df[prop]

    date_props = ["month", "day", "dayofweek", "quarter"]
    for prop in date_props:
        props["ci_{0}".format(prop)] = getattr(df["srch_ci"].dt, prop)
        props["co_{0}".format(prop)] = getattr(df["srch_co"].dt, prop)
    props["stay_span"] = (df["srch_co"] - df["srch_ci"]).astype('timedelta64[h]')

    ret = pd.DataFrame(props)

    ret = ret.join(destinations, on="srch_destination_id", how='left', rsuffix="dest")
    ret = ret.drop(drop_list, axis=1)
    return ret


def load():
    destinations = pd.read_csv(data_path + "destinations.csv")
    test = pd.read_csv(data_path + "test.csv", index_col=0)
    train = pd.read_csv(data_path + "train.csv", index_col=0)
    test_labels = pd.read_csv(data_path + "testlabels.csv", index_col=0)

    most_common_clusters = list(train.hotel_cluster.value_counts().head().index)

    # Concatenate test and train data for converting into one-hot encodings form
    merge_df = pd.concat([train, test])
    merge_df.drop(['cnt', 'is_booking'], axis=1, inplace=True)

    # Convert the date time columns
    merge_df = calc_fast_features(merge_df, destinations)
    merge_df.fillna(-1, inplace=True)

    one_hot_rep = one_hot(merge_df, cols_for_one_hot)
    one_hot_rep.drop(cols_for_one_hot, axis=1, inplace=True)
    one_hot_rep.to_csv("One_hot_encodings_x.csv", sep='\t')
    size = 1000
    y_labels = one_hot_rep.filter(['hotel_cluster'])
    y_data = y_labels.as_matrix().astype(int)

    # Separating the train and test data
    one_hot_rep.drop(['hotel_cluster'], axis=1, inplace=True)
    one_hot_rep.to_csv("One_hot_encodings_x.csv", sep='\t')
    x_data = one_hot_rep.as_matrix()
    x_train = x_data[0:size, :]
    y_data = y_data[0:size, :]

    x_test = x_data[size: x_data.shape[0], :]
    y_test = test_labels.filter(['hotel_cluster'])
    y_test = y_test.as_matrix().astype(int)

    y_data_oh = np.zeros((y_data.shape[0], 100))
    y_test_oh = np.zeros((y_test.shape[0], 100))

    for i in range(y_data.shape[0]):
        ind = y_data[i]
        y_data_oh[i][ind] = 1

    for i in range(y_test.shape[0]):
        ind = y_test[i]
        y_test_oh[i][ind] = 1

    target = [[l] for l in test_labels["hotel_cluster"]]

    return x_train, x_test, y_data_oh, y_test_oh, target, most_common_clusters


def f5(seq, idfun=None):
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result
