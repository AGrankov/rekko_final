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

def test_predict():
    parser = argparse.ArgumentParser(description='Test predict with lightgbm ranker')

    parser.add_argument('-i', '--input', default='../p_input/transactions.csv', help='input data file name')
    parser.add_argument('-mi', '--meta-information', default='../p_input/catalogue.csv', help='meta information file name')
    parser.add_argument('-v', '--vectors', default='../models/fasttext_cbow_model.vec', help='path to pretrained vectors')
    parser.add_argument('-dim', '--dimension', default=64, help='dimention of embeddings')

    parser.add_argument('-ss', '--sample-submission', default='../input/test_users.json', help='sample submission file name')

    parser.add_argument('-om1', '--origin-model-1', default='../models/movies_seq_emb_mt_model_kfold_{}_v9', help='original model path')
    parser.add_argument('-sl', '--sequence-length', default=100, help='length of sequences')
    parser.add_argument('-bs', '--batch-size', default=256, help='size of batches')
    parser.add_argument('-tn', '--top-n', default=100, help='get top N preds from neural network')

    parser.add_argument('-mp', '--model-path', default='../models/lightgbm/', help='path to models directory')
    parser.add_argument('-m', '--model-name', default='lgb_v4', help='name of trained model')

    args = parser.parse_args()


    df = pd.read_csv(args.input)
    movie_id_trans = labeling_col(df, 'element_uid')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    ss = json.load(open(args.sample_submission))
    res_users = set(ss['users'])
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    df = df[df.user_uid.isin(res_users)].reset_index(drop=True)

    movies_sequences = process_sequence(df, 'element_uid')
    tss = process_sequence(df, 'ts')
    wp = process_sequence(df, 'watch_percent')
    cm = process_sequence(df, 'consumption_mode')   # 3
    dt = process_sequence(df, 'device_type')        # 7

    targets_count = len(movie_id_trans.classes_)

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
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    rat_df = pd.read_csv('../input/ratings.csv')
    rat_df = rat_df.sort_values('ts').reset_index(drop=True)
    rat_df_inner = rat_df[rat_df['element_uid'].isin(set(movie_id_trans.classes_))].reset_index()
    rat_df_inner['element_uid'] = movie_id_trans.transform(rat_df_inner['element_uid'])
    # rat_df_inner = rat_df_inner[rat_df_inner['rating'] >= 9].reset_index(drop=True)
    rat_df_inner_seq = process_sequence(rat_df_inner, 'element_uid')
    rat_df_inner_users = sorted(rat_df_inner.user_uid.unique())
    rat_df_inner_dict = {rat_df_inner_users[idx]: rat_df_inner_seq[idx] for idx in range(len(rat_df_inner_seq))}
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    book_df = pd.read_csv('../input/bookmarks.csv')
    book_df = book_df.sort_values('ts').reset_index(drop=True)
    book_df_inner = book_df[book_df['element_uid'].isin(set(movie_id_trans.classes_))].reset_index()
    book_df_inner['element_uid'] = movie_id_trans.transform(book_df_inner['element_uid'])
    book_df_inner_seq = process_sequence(book_df_inner, 'element_uid')
    book_df_inner_users = sorted(book_df_inner.user_uid.unique())
    book_df_inner_dict = {book_df_inner_users[idx]: book_df_inner_seq[idx] for idx in range(len(book_df_inner_seq))}
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    result_users = sorted(df.user_uid.unique())
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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

    testing_seq = []
    testing_additional = []

    wp_all = np.zeros((len(movies_sequences), len(movie_id_trans.classes_)), dtype=np.float32)

    for si, seq in tqdm(enumerate(movies_sequences), total=len(movies_sequences)):
        wp_all[si] = make_glob_from_data(seq, wp[si])

        ts = [x+1 for x in seq]
        cmts = ohe_seq(cm[si], 3)
        dtts = ohe_seq(dt[si], 7)
        adds = np.hstack((np.reshape(wp[si], (-1, 1)), cmts, dtts,))
        adds[adds > 1] = 1

        testing_seq.append(ts)
        testing_additional.append(adds)

    testing_seq = np.array(testing_seq)
    testing_additional = np.array(testing_additional)


    folds = [0, 1, 2, 3, 4, 5]

    testset1 = UserMovieMultiTargetDataset(testing_seq,
                                          testing_additional,
                                          movies_dict,
                                          seq_len=args.sequence_length)

    models = []
    for fi in folds:
        model = MovieRNNModel(
            seq_len=args.sequence_length,
            features_dim=testset1[0].shape[1],
            result_classes=len(movie_id_trans.classes_),
            rnn_units=128,
            mid_dim=2048,
            mid_dim2=2048,
            pool_count=1,
            num_layers=1
        )
        model = torch.nn.DataParallel(model).cuda()

        checkpoint = torch.load(os.path.join(args.origin_model_1.format(fi), "best_map_model.t7"))
        model.load_state_dict(checkpoint['net'])
        model.eval()

        models.append(model)

    batches_count = len(testset1) // args.batch_size + 1

    test_predictions_idxes = []
    test_predictions_values = []
    with torch.no_grad():
        for bidx in tqdm(range(batches_count)):
            inp1_arr = []
            for sidx in range(args.batch_size):
                if bidx*args.batch_size+sidx < len(testset1):
                    inp1_arr.append(testset1[bidx*args.batch_size+sidx])

            input1 = torch.Tensor(np.array(inp1_arr)).cuda()
            y_preds_1 = []
            for model in models:
                y_pred_1 = model(input1)
                y_preds_1.append(y_pred_1)
            y_pred = torch.mean(torch.stack(y_preds_1), dim=0)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            mask_tens = torch.ByteTensor((wp_all[bidx*args.batch_size:(bidx+1)*args.batch_size] > 0).astype(np.uint8)).cuda()
            y_pred.masked_fill_(mask_tens, -1000)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

            sort_preds, sort_indices = y_pred.sort(dim=1)
            sort_indices = sort_indices[:, -args.top_n:]
            sort_preds = sort_preds[:, -args.top_n:]
            for i in range(len(sort_preds)):
                sort_preds[i] = y_pred_1[i, sort_indices[i]]
            test_predictions_idxes.extend(sort_indices.cpu().data.numpy()[:, ::-1])
            test_predictions_values.extend(sort_preds.cpu().data.numpy()[:, ::-1])

    test_predictions_idxes = np.array(test_predictions_idxes)
    test_predictions_values = np.array(test_predictions_values)


    with open('../p_input/{}.test.query'.format(args.model_name), 'w') as query_output_file:
        with open('../p_input/{}.test'.format(args.model_name), 'w') as output_file:
            for si in tqdm(range(len(movies_sequences))):
                seq = movies_sequences[si]
                adds = wp[si]

                user_features = ' '.join(['{}:{:.3f}'.format(seq[i], adds[i]) for i in range(len(seq))])

                pred_idxes = test_predictions_idxes[si]
                pred_values = test_predictions_values[si]

                uidx = result_users[si]
                ratings = rat_df_inner_dict[uidx] if uidx in rat_df_inner_dict else None
                books = book_df_inner_dict[uidx] if uidx in book_df_inner_dict else None

                for sub_idx in range(args.top_n):
                    md = movies_dict[pred_idxes[sub_idx] + 1]
                    movie_features = ' '.join(['{}:{:.4f}'.format(i+targets_count, md[i]) for i in range(len(md))])

                    is_rating = (pred_idxes[sub_idx] in ratings) if ratings is not None else False
                    is_book = (pred_idxes[sub_idx] in books) if books is not None else False

                    br_feature = '{}:{} {}:{}'.format(targets_count+len(md), int(is_rating), targets_count+len(md)+1, int(is_book))

                    res_str = '{} {} {}\n'.format(user_features, movie_features, br_feature)
                    output_file.write(res_str)

                query_output_file.write('{}\n'.format(args.top_n))


    lgb_preds = np.zeros((len(result_users) * args.top_n), dtype=np.float32)
    test_fn = '../p_input/{}.test'.format(args.model_name)

    folds = [0, 1, 2, 3, 4]
    for fold_idx in folds:
        print("Predicting {}".format(fold_idx))

        clf = lgb.Booster(model_file=os.path.join(args.model_path, '{}_fold_{}'.format(args.model_name, fold_idx)))
        preds = clf.predict(test_fn)
        lgb_preds += preds

    lgb_preds = lgb_preds / len(folds)
    lgb_preds = lgb_preds.reshape((-1, args.top_n))

    with open('../p_input/{}.test_preds'.format(args.model_name), 'wb') as file:
        pickle.dump(lgb_preds, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    test_predict()
