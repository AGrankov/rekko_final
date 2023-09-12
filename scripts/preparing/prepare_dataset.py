import sys
sys.path.append('..')
import os
import json
import numpy as np
import pandas as pd
import argparse
from scripts.utils.utils import labeling_col, normalize_int_col

train_filename = 'transactions.csv'
content_meta_filename = 'catalogue.json'

def main():
    parser = argparse.ArgumentParser(description='Preprocess initial dataset')

    parser.add_argument('-i', '--input', default='../input/', help='input data path')
    parser.add_argument('-o', '--output', default='../p_input/', help='output data path')

    args = parser.parse_args()

    train_df = pd.read_csv(os.path.join(args.input, train_filename))
    train_df = train_df.sort_values('ts').reset_index(drop=True)

    cols_for_drop = [
        'device_manufacturer',
    ]

    train_df.drop(cols_for_drop, 1, inplace=True)

    labeling_col(train_df, 'consumption_mode')
    labeling_col(train_df, 'device_type')

    train_df.loc[train_df.watched_time > 86400, 'watched_time'] = 86400

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    cont_meta_dict = json.load(open(os.path.join(args.input, content_meta_filename)))
    cont_meta_keys = sorted(list(cont_meta_dict.keys()))

    good_cols = [
        'duration',
        'feature_1',
        'feature_2',
        'feature_3',
        'feature_4',
        'feature_5',
        'type'
    ]

    cont_meta_values = []
    for cont_id in cont_meta_keys:
        dd = cont_meta_dict[cont_id]
        cont_meta_values.append([cont_id] + [dd[col] for col in good_cols])

    cont_meta_df = pd.DataFrame(data=cont_meta_values, columns=['element_uid'] + good_cols)
    cont_meta_df['element_uid'] = cont_meta_df['element_uid'].astype(np.int64)

    train_df = train_df.merge(cont_meta_df[['element_uid', 'duration']],
                              how='left',
                              on='element_uid')
    train_df['watch_percent'] = train_df['watched_time'] / (train_df['duration'] * 60)
    train_df.loc[train_df['watch_percent'] == np.inf, 'watch_percent'] = 1
    train_df.loc[pd.isna(train_df['watch_percent']), 'watch_percent'] = 1
    train_df.loc[train_df['watch_percent'] == 0, 'watch_percent'] = 0.01
    train_df.drop(['watched_time', 'duration'], 1, inplace=True)

    type_dict = {'movie':0, 'series':1, 'multipart_movie':2}
    cont_meta_df['type'] = cont_meta_df['type'].map(type_dict)

    cont_meta_df['duration'] = cont_meta_df['duration'] // 10
    cont_meta_df['feature_1'] = np.log(cont_meta_df['feature_1'])

    cols_for_normalize = [
        'duration',
        'feature_1',
    ]

    for col in cols_for_normalize:
        normalize_int_col(cont_meta_df, col)

    train_df.to_csv(os.path.join(args.output, train_filename), index=False)
    cont_meta_df.to_csv(os.path.join(args.output, content_meta_filename.replace('.json', '.csv')), index=False)


if __name__ == '__main__':
    main()
