import sys
sys.path.append('..')
import os
import argparse
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
import torch
import scipy
import pickle
import json

from sklearn.model_selection import KFold

from scripts.utils.utils import labeling_col, process_sequence
from scripts.dataset.user_movie_multi_target_dataset import UserMovieMultiTargetDataset
from scripts.models.movie_features_seq_model import MovieRNNModel

tqdm.monitor_interval = 0

def load_vectors(input_file):
    vectors = {}
    with open(input_file) as file:
        file.readline()
        for line in file:
            line_list = line.strip().split()
            if not line_list[0].isdigit():
                continue
            movie_id = int(line_list[0])
            vec = np.array([float(_) for _ in line_list[1:]], dtype=float)
            if not movie_id in vectors:
                vectors[movie_id] = vec
    return vectors

def prepare():
    parser = argparse.ArgumentParser(description='Train lightgbm model')

    parser.add_argument('-i', '--input', default='../p_input/transactions.csv', help='input data file name')
    parser.add_argument('-mi', '--meta-information', default='../p_input/catalogue.csv', help='meta information file name')
    parser.add_argument('-v', '--vectors', default='../models/fasttext_cbow_model.vec', help='path to pretrained vectors')
    parser.add_argument('-dim', '--dimension', default=64, help='dimention of embeddings')

    parser.add_argument('-kf', '--kfolds', default=6, help='kfold count for split')
    parser.add_argument('-s', '--seed', default=42, help='seed')

    parser.add_argument('-om1', '--origin-model-1', default='../models/movies_seq_emb_mt_model_kfold_{}_v9', help='original model path')
    parser.add_argument('-sl', '--sequence-length', default=100, help='length of sequences')
    parser.add_argument('-bs', '--batch-size', default=256, help='size of batches')
    parser.add_argument('-tn', '--top-n', default=100, help='get top N preds from neural network')

    parser.add_argument('-mp', '--model-path', default='../models/lightgbm/', help='path to models directory')
    parser.add_argument('-m', '--model-name', default='lgb_v4', help='name of trained model')

    args = parser.parse_args()


    df = pd.read_csv(args.input)
    movie_id_trans = labeling_col(df, 'element_uid')
    user_id_trans = labeling_col(df, 'user_uid')
    movies_sequences = process_sequence(df, 'element_uid')
    tss = process_sequence(df, 'ts')
    wp = process_sequence(df, 'watch_percent')
    cm = process_sequence(df, 'consumption_mode')   # 3
    dt = process_sequence(df, 'device_type')        # 7

    targets_count = len(movie_id_trans.classes_)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    rat_df = pd.read_csv('../input/ratings.csv')
    rat_df = rat_df.sort_values('ts').reset_index(drop=True)
    rat_df = rat_df[rat_df['ts'] < 43447000].reset_index(drop=True)
    rat_df_inner = rat_df[rat_df['element_uid'].isin(set(movie_id_trans.classes_))].reset_index()
    rat_df_inner['element_uid'] = movie_id_trans.transform(rat_df_inner['element_uid'])
    # rat_df_inner = rat_df_inner[rat_df_inner['rating'] >= 9].reset_index(drop=True)
    rat_df_inner_seq = process_sequence(rat_df_inner, 'element_uid')
    rat_df_inner_users = sorted(rat_df_inner.user_uid.unique())
    rat_df_inner_dict = {rat_df_inner_users[idx]: rat_df_inner_seq[idx] for idx in range(len(rat_df_inner_seq))}
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    book_df = pd.read_csv('../input/bookmarks.csv')
    book_df = book_df.sort_values('ts').reset_index(drop=True)
    book_df = book_df[book_df['ts'] < 43447000].reset_index(drop=True)
    book_df_inner = book_df[book_df['element_uid'].isin(set(movie_id_trans.classes_))].reset_index()
    book_df_inner['element_uid'] = movie_id_trans.transform(book_df_inner['element_uid'])
    book_df_inner_seq = process_sequence(book_df_inner, 'element_uid')
    book_df_inner_users = sorted(book_df_inner.user_uid.unique())
    book_df_inner_dict = {book_df_inner_users[idx]: book_df_inner_seq[idx] for idx in range(len(book_df_inner_seq))}
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    result_users = sorted(df.user_uid.unique())
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    df_meta = pd.read_csv(args.meta_information)
    int_cols = [
        'duration',
        'feature_1',
        'feature_2',
        'feature_4',
        'feature_5',
    ]
    ohe_cols = [
        'type',
        'feature_3',
    ]
    ohe_cols_size = [3, 51]

    vectors = load_vectors(args.vectors)

    movies_dict = dict()
    for name, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
        processed = []
        processed.append(row[int_cols].values)
        for oidx, oc in enumerate(ohe_cols):
            ar = np.zeros((ohe_cols_size[oidx]), dtype=np.uint8)
            if type(row[oc]) == str:
                ar[np.array([int(x) for x in row[oc].split(',')])] = 1
            if type(row[oc]) == int:
                ar[row[oc]] = 1
            processed.append(ar)
        if name in vectors:
            processed.append(vectors[name])
        else:
            processed.append(np.zeros((args.dimension), dtype=np.float32))
        movies_dict[name+1] = np.concatenate(processed)

    def ohe_seq(seq, size):
        res = np.zeros((len(seq), size))
        for idx, val in enumerate(seq):
            if val >= 0:
                res[idx, val] = 1
        return res

    def make_glob_from_data(seq, seq_data, max_len=len(movie_id_trans.classes_)):
        res = np.zeros((max_len), dtype=np.float32)
        for idx, sv in enumerate(seq):
            res[sv] += seq_data[idx]
        return res

    all_users_seq = []
    all_users_additional = []

    for si, seq in tqdm(enumerate(movies_sequences), total=len(movies_sequences)):
        ts = [x+1 for x in seq]
        cmts = ohe_seq(cm[si], 3)
        dtts = ohe_seq(dt[si], 7)
        adds = np.hstack((np.reshape(wp[si], (-1, 1)), cmts, dtts,))
        adds[adds > 1] = 1
        all_users_seq.append(ts)
        all_users_additional.append(adds)

    all_users_seq = np.array(all_users_seq)
    all_users_additional = np.array(all_users_additional)

    oof_preds_idxes = np.zeros((len(movies_sequences), args.top_n), dtype=np.int32)
    oof_preds_values = np.zeros((len(movies_sequences), args.top_n), dtype=np.float32)

    folds = KFold(n_splits=args.kfolds, shuffle=False, random_state=args.seed)
    for fold_idx, (trn_idx, val_idx) in enumerate(folds.split(np.arange(len(movies_sequences)))):

        validation_tuples = []
        wp_all = []

        for si, seq in tqdm(enumerate(movies_sequences), total=len(movies_sequences)):
            if si in val_idx:
                for idx in range(len(seq)):
                    if tss[si][idx] > 43447000:
                        break

                wp_all.append(make_glob_from_data(seq[:idx], wp[si][:idx]))

                targets = []
                for sub_idx in range(idx, len(seq)):
                    # see to movie type ohe, zero index 'is movie' flag
                    if ((movies_dict[seq[sub_idx]+1][len(int_cols)] == 1) and (wp[si][sub_idx] > 0.5)) or\
                        ((movies_dict[seq[sub_idx]+1][len(int_cols)] == 0) and (wp[si][sub_idx] > 0.3)):
                        targets.append(seq[sub_idx])
                validation_tuples.append((si, idx, targets))

        wp_all = np.array(wp_all)

        validset1 = UserMovieMultiTargetDataset(all_users_seq,
                                                all_users_additional,
                                                movies_dict,
                                                seq_len=args.sequence_length,
                                                targets=validation_tuples,
                                                targets_size=targets_count,
                                                )

        model1 = MovieRNNModel(
            seq_len=args.sequence_length,
            features_dim=validset1[0][0].shape[1],
            result_classes=targets_count,
            rnn_units=128,
            mid_dim=2048,
            mid_dim2=2048,
            pool_count=1,
            num_layers=1
        )
        model1 = torch.nn.DataParallel(model1).cuda()

        checkpoint = torch.load(os.path.join(args.origin_model_1.format(fold_idx), "best_map_model.t7"))
        model1.load_state_dict(checkpoint['net'])
        model1.eval()


        batches_count = len(validset1) // args.batch_size + 1
        valid_predictions_idxes = []
        valid_predictions_values = []
        with torch.no_grad():
            for bidx in tqdm(range(batches_count)):
                inp1_arr = []
                for sidx in range(args.batch_size):
                    if bidx*args.batch_size+sidx < len(validset1):
                        inp1_arr.append(validset1[bidx*args.batch_size+sidx][0])

                input1 = torch.Tensor(np.array(inp1_arr)).cuda()
                y_pred = model1(input1)

                mask_tens = torch.ByteTensor((wp_all[bidx*args.batch_size:(bidx+1)*args.batch_size] > 0).astype(np.uint8)).cuda()
                y_pred.masked_fill_(mask_tens, -1000)

                sort_preds, sort_indices = y_pred.sort(dim=1)
                sort_indices = sort_indices[:, -args.top_n:]
                sort_preds = sort_preds[:, -args.top_n:]
                for i in range(len(sort_preds)):
                    sort_preds[i] = y_pred[i, sort_indices[i]]
                valid_predictions_idxes.extend(sort_indices.cpu().numpy()[:, ::-1])
                valid_predictions_values.extend(sort_preds.cpu().numpy()[:, ::-1])

        valid_predictions_idxes = np.array(valid_predictions_idxes)
        valid_predictions_values = np.array(valid_predictions_values)

        oof_preds_idxes[val_idx] = valid_predictions_idxes
        oof_preds_values[val_idx] = valid_predictions_values


    validation_tuples = []
    for si, seq in tqdm(enumerate(movies_sequences), total=len(movies_sequences)):
        for idx in range(len(seq)):
            if tss[si][idx] > 43447000:
                break

        targets = []
        for sub_idx in range(idx, len(seq)):
            # see to movie type ohe, zero index 'is movie' flag
            if ((movies_dict[seq[sub_idx]+1][len(int_cols)] == 1) and (wp[si][sub_idx] > 0.5)) or\
                ((movies_dict[seq[sub_idx]+1][len(int_cols)] == 0) and (wp[si][sub_idx] > 0.3)):
                targets.append(seq[sub_idx])
        validation_tuples.append((si, idx, targets))

    train_users_idxes = []

    with open('../p_input/{}.train.query'.format(args.model_name), 'w') as query_output_file:
        with open('../p_input/{}.train'.format(args.model_name), 'w') as output_file:
            for gi, (si, idx, tar) in tqdm(enumerate(validation_tuples), total=len(validation_tuples)):
                if len(tar) > 0:
                    seq = movies_sequences[si][:idx]
                    adds = wp[si][:idx]

                    user_features = ' '.join(['{}:{:.3f}'.format(seq[i], adds[i]) for i in range(len(seq))])

                    pred_idxes = oof_preds_idxes[gi]
                    pred_values = oof_preds_values[gi]

                    is_correct = [i in tar for i in pred_idxes]

                    if np.sum(is_correct) > 0:
                        train_users_idxes.append(si)

                        uidx = result_users[si]
                        ratings = rat_df_inner_dict[uidx] if uidx in rat_df_inner_dict else None
                        books = book_df_inner_dict[uidx] if uidx in book_df_inner_dict else None

                        for sub_idx in range(args.top_n):
                            md = movies_dict[pred_idxes[sub_idx] + 1]
                            movie_features = ' '.join(['{}:{:.4f}'.format(i+targets_count, md[i]) for i in range(len(md))])

                            is_rating = (pred_idxes[sub_idx] in ratings) if ratings is not None else False
                            is_book = (pred_idxes[sub_idx] in books) if books is not None else False

                            br_feature = '{}:{} {}:{}'.format(targets_count+len(md), int(is_rating), targets_count+len(md)+1, int(is_book))

                            res_str = '{} {} {} {}\n'.format(int(is_correct[sub_idx]), user_features, movie_features, br_feature)
                            output_file.write(res_str)

                        query_output_file.write('{}\n'.format(args.top_n))

    with open('../p_input/{}.train.users'.format(args.model_name), 'wb') as users_file:
        pickle.dump(train_users_idxes, users_file, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../p_input/{}.validate.idxes'.format(args.model_name), 'wb') as file:
        pickle.dump(oof_preds_idxes, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../p_input/{}.validate.values'.format(args.model_name), 'wb') as file:
        pickle.dump(oof_preds_values, file, protocol=pickle.HIGHEST_PROTOCOL)



def training():
    parser = argparse.ArgumentParser(description='Train lightgbm model')

    parser.add_argument('-i', '--input', default='../p_input/transactions.csv', help='input data file name')

    parser.add_argument('-kf', '--kfolds', default=5, help='kfold count for split')
    parser.add_argument('-s', '--seed', default=42, help='seed')

    parser.add_argument('-mp', '--model-path', default='../models/lightgbm/', help='path to models directory')
    parser.add_argument('-m', '--model-name', default='lgb_v4', help='name of trained model')

    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        os.mkdir(args.model_path)

    train_fn = '../p_input/{}.train'.format(args.model_name)
    train_data = lgb.Dataset(train_fn, free_raw_data=False)
    train_data.construct()

    total_users_count = train_data.get_group().shape[0]
    groups = train_data.get_group()

    folds = KFold(n_splits=args.kfolds, shuffle=False, random_state=args.seed)
    for fold_idx, (trn_idx, val_idx) in enumerate(folds.split(np.arange(total_users_count))):
        train_idxes = []
        valid_idxes = []
        pass_idxes = 0
        for gri in tqdm(range(total_users_count)):
            if gri in trn_idx:
                train_idxes.append(np.arange(groups[gri])+pass_idxes)
            else:
                valid_idxes.append(np.arange(groups[gri])+pass_idxes)
            pass_idxes += groups[gri]

        trd = train_data.subset(np.concatenate(train_idxes))
        trd.construct()
        vld = train_data.subset(np.concatenate(valid_idxes))
        vld.construct()


        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'lambdarank',
            'metric': 'map',
            'eval_at': [20],
            'max_position': 20,

            'metric_freq': 1,
            'is_training_metric': True,
            'max_bin': 64,
            'learning_rate': 0.1,
            'num_leaves': 63,
            'bagging_fraction': 0.5,
            'bagging_freq': 3,
            'min_data_in_leaf': 50,
            'min_sum_hessian_in_leaf': 5.0,
            'is_enable_sparse': True,
            'verbosity': -1,
            'seed':42,

            'num_threads': 10
        }

        clf = lgb.train(params,
                        trd,
                        num_boost_round=2000,
                        valid_sets = [vld],
                        verbose_eval=1,
                        early_stopping_rounds=100,
                        )

        clf.save_model(os.path.join(args.model_path, '{}_fold_{}'.format(args.model_name, fold_idx)))


if __name__ == '__main__':
    prepare()
    training()
