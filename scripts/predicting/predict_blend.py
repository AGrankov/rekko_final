import sys
sys.path.append('..')
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import json
import pickle

from scripts.utils.utils import labeling_col, process_sequence
from scripts.utils.torch_train_utils import progress_bar
from scripts.models.movie_features_seq_model import MovieRNNModel
from scripts.dataset.user_movie_multi_target_dataset import UserMovieMultiTargetDataset

tqdm.monitor_interval = 0
torch.backends.cudnn.benchmark = True


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


def main():
    parser = argparse.ArgumentParser(description='Predict meaned values from two models')

    parser.add_argument('-i', '--input', default='../p_input/transactions.csv', help='input data file name')
    parser.add_argument('-mi', '--meta-information', default='../p_input/catalogue.csv', help='meta information file name')
    parser.add_argument('-v', '--vectors', default='../models/fasttext_cbow_model.vec', help='path to pretrained vectors')
    parser.add_argument('-ss', '--sample-submission', default='../input/test_users.json', help='sample submission file name')
    parser.add_argument('-dim', '--dimension', default=64, help='dimention of embeddings')
    parser.add_argument('-sl', '--sequence-length', default=100, help='length of sequences')

    parser.add_argument('-mp1', '--model-path-1', default='../models/movies_seq_emb_mt_model_kfold_{}_v9/', help='third model file name')
    parser.add_argument('-bs', '--batch-size', default=256, help='size of batches')

    parser.add_argument('-lm', '--lgb-model-name', default='lgb_v4', help='name of trained lgb model')

    parser.add_argument('-o', '--output', default='../submissions/movies_mt_kfold_full_v9_lightfm_5_v2_lgb_ranker_v4_books_binrat_2.json', help='output submission file name')

    args = parser.parse_args()


    df = pd.read_csv(args.input)
    movie_id_trans = labeling_col(df, 'element_uid')
    user_id_trans = labeling_col(df[['user_uid', 'ts']].copy(), 'user_uid')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    ss = json.load(open(args.sample_submission))
    res_users = set(ss['users'])
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    df = df[df.user_uid.isin(res_users)].reset_index(drop=True)
    movies_sequences = process_sequence(df, 'element_uid')
    wp = process_sequence(df, 'watch_percent')
    cm = process_sequence(df, 'consumption_mode')   # 3
    dt = process_sequence(df, 'device_type')        # 7


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

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    rat_df = pd.read_csv('../input/ratings.csv')
    # 42840000
    rat_df_inner = rat_df[rat_df['element_uid'].isin(set(movie_id_trans.classes_))].reset_index()
    rat_df_inner['element_uid'] = movie_id_trans.transform(rat_df_inner['element_uid'])
    rat_df_inner_neg = rat_df_inner[rat_df_inner['ts'] < 42840000].reset_index(drop=True)
    rat_df_inner_pos = rat_df_inner[(rat_df_inner['ts'] > 42840000) & (rat_df_inner['rating'] >= 8)].reset_index(drop=True)

    rat_df_inner_pos_seq = process_sequence(rat_df_inner_pos, 'element_uid')
    rat_df_inner_pos_users = sorted(rat_df_inner_pos.user_uid.unique())
    rat_df_inner_pos_dict = {rat_df_inner_pos_users[idx]: rat_df_inner_pos_seq[idx] for idx in range(len(rat_df_inner_pos_seq))}

    rat_df_inner_neg_seq = process_sequence(rat_df_inner_neg, 'element_uid')
    rat_df_inner_neg_users = sorted(rat_df_inner_neg.user_uid.unique())
    rat_df_inner_neg_dict = {rat_df_inner_neg_users[idx]: rat_df_inner_neg_seq[idx] for idx in range(len(rat_df_inner_neg_seq))}
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    book_df = pd.read_csv('../input/bookmarks.csv')
    book_df_inner = book_df[book_df['element_uid'].isin(set(movie_id_trans.classes_))].reset_index()
    book_df_inner['element_uid'] = movie_id_trans.transform(book_df_inner['element_uid'])
    book_df_inner_seq = process_sequence(book_df_inner, 'element_uid')
    book_df_inner_users = sorted(book_df_inner.user_uid.unique())
    book_df_inner_dict = {book_df_inner_users[idx]: book_df_inner_seq[idx] for idx in range(len(book_df_inner_seq))}
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    result_users = sorted(df.user_uid.unique())
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    testing_seq = []
    testing_additional = []

    wp_all = np.zeros((len(movies_sequences), len(movie_id_trans.classes_)), dtype=np.float32)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    rat_pos_all = np.zeros((len(movies_sequences), len(movie_id_trans.classes_)), dtype=np.uint8)
    rat_neg_all = np.zeros((len(movies_sequences), len(movie_id_trans.classes_)), dtype=np.uint8)
    books_all = np.zeros((len(movies_sequences), len(movie_id_trans.classes_)), dtype=np.uint8)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def make_glob_from_data(seq, seq_data, max_len=len(movie_id_trans.classes_)):
        res = np.zeros((max_len), dtype=np.float32)
        for idx, sv in enumerate(seq):
            res[sv] += seq_data[idx]
        return res

    for si, seq in tqdm(enumerate(movies_sequences), total=len(movies_sequences)):
        wp_all[si] = make_glob_from_data(seq, wp[si])
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        uidx = result_users[si]
        if uidx in rat_df_inner_pos_dict:
            rat_pos_all[si] = make_glob_from_data(rat_df_inner_pos_dict[uidx], [1]*len(rat_df_inner_pos_dict[uidx]))
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        uidx = result_users[si]
        if uidx in rat_df_inner_neg_dict:
            rat_neg_all[si] = make_glob_from_data(rat_df_inner_neg_dict[uidx], [1]*len(rat_df_inner_neg_dict[uidx]))
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        uidx = result_users[si]
        if uidx in book_df_inner_dict:
            books_all[si] = make_glob_from_data(book_df_inner_dict[uidx], [1]*len(book_df_inner_dict[uidx]))
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        ts = [x+1 for x in seq]
        cmts = ohe_seq(cm[si], 3)
        dtts = ohe_seq(dt[si], 7)
        adds = np.hstack((np.reshape(wp[si], (-1, 1)), cmts, dtts,))
        adds[adds > 1] = 1

        testing_seq.append(ts)
        testing_additional.append(adds)

    testing_seq = np.array(testing_seq)
    testing_additional = np.array(testing_additional)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    folds = [0, 1, 2, 3, 4, 5]

    testset1 = UserMovieMultiTargetDataset(testing_seq,
                                          testing_additional,
                                          movies_dict,
                                          seq_len=args.sequence_length)

    models_1 = []

    for fi in folds:
        model1 = MovieRNNModel(
            seq_len=args.sequence_length,
            features_dim=testset1[0].shape[1],
            result_classes=len(movie_id_trans.classes_),
            rnn_units=128,
            mid_dim=2048,
            mid_dim2=2048,
            pool_count=1,
            num_layers=1
        )
        model1 = torch.nn.DataParallel(model1).cuda()

        checkpoint = torch.load(os.path.join(args.model_path_1.format(fi), "best_map_model.t7"))
        model1.load_state_dict(checkpoint['net'])
        model1.eval()
        print('val mapk {} {}'.format(checkpoint['mapk20'], fi))

        models_1.append(model1)

    lightfm_models = []

    for fi in range(5):
        with open('../models/lightfm_v2_fold_{}.pickle'.format(fi), 'rb') as file:
            model = pickle.load(file)
            lightfm_models.append(model)


    light_res_users = res_users & set(user_id_trans.classes_)
    result_users = sorted(list(light_res_users))
    light_res_users = user_id_trans.transform(result_users)

    light_preds = []
    for sidx in tqdm(range(len(light_res_users))):
        tmp_preds = []
        for light_model in lightfm_models:
            userId = light_res_users[sidx]
            preds = light_model.predict(userId,
                                    np.arange(len(movie_id_trans.classes_)),
                                    num_threads=12)
            tmp_preds.append(preds)
        preds = np.mean(tmp_preds, axis=0)
        light_preds.append(preds)
    light_preds = np.array(light_preds)

    with open('../p_input/{}.test_preds'.format(args.lgb_model_name), 'rb') as file:
        lgb_preds = pickle.load(file)
    lgb_preds = (lgb_preds + 5) / 10


    batches_count = len(testset1) // args.batch_size + 1

    test_predictions = []
    with torch.no_grad():
        for bidx in tqdm(range(batches_count)):
            inp1_arr = []
            for sidx in range(args.batch_size):
                if bidx*args.batch_size+sidx < len(testset1):
                    inp1_arr.append(testset1[bidx*args.batch_size+sidx])

            mask_tens = torch.ByteTensor((wp_all[bidx*args.batch_size:(bidx+1)*args.batch_size] > 0).astype(np.uint8)).cuda()

            input1 = torch.Tensor(np.array(inp1_arr)).cuda()
            y_preds_1 = []
            for model_1 in models_1:
                y_pred_1 = model_1(input1)
                y_preds_1.append(y_pred_1)
            y_pred_1 = torch.mean(torch.stack(y_preds_1), dim=0)
            y_pred_1 = (y_pred_1 + 30) / 30
            y_pred_1.masked_fill_(mask_tens, -1000)

            sort_preds, sort_indices = y_pred_1.sort(dim=1, descending=True)
            cropped_sort_indices = sort_indices[:, :lgb_preds.shape[1]]

            lgb_pred_1 = torch.zeros(y_pred_1.shape).cuda()
            for sidx in range(len(lgb_pred_1)):
                lgb_pred_1[sidx, cropped_sort_indices[sidx]] = torch.Tensor(lgb_preds[bidx*args.batch_size+sidx]).cuda()

            lpr = torch.Tensor(light_preds[bidx*args.batch_size:(bidx+1)*args.batch_size]).cuda()
            lpr = (lpr + 7) / 66

            rat_pred = torch.Tensor((rat_pos_all[bidx*args.batch_size:(bidx+1)*args.batch_size] > 0) * 0.5).cuda()
            book_pred = torch.Tensor((books_all[bidx*args.batch_size:(bidx+1)*args.batch_size] > 0) * 0.73).cuda()

            y_pred = torch.mean(torch.stack([y_pred_1, lgb_pred_1, lpr, rat_pred, book_pred]), dim=0)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            y_pred.masked_fill_(mask_tens, -1000)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            rat_neg_tens = torch.ByteTensor((rat_neg_all[bidx*args.batch_size:(bidx+1)*args.batch_size] > 0).astype(np.uint8)).cuda()
            y_pred.masked_fill_(rat_neg_tens, -1000)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

            sort_preds, sort_indices = y_pred.sort(dim=1)
            sort_indices = sort_indices[:, -20:]
            test_predictions.extend(sort_indices.cpu().data.numpy()[:, ::-1])
    test_predictions = np.array(test_predictions)

    result_preds = []
    for idx in tqdm(range(len(test_predictions))):
        reversed = movie_id_trans.inverse_transform(test_predictions[idx])
        result_preds.append([int(j) for j in reversed])

    result_users = sorted(df.user_uid.unique())
    predicted_dict = {result_users[idx]: result_preds[idx] for idx in range(len(result_preds))}

    sub_name = args.output
    with open(sub_name, 'w') as fp:
        json.dump({str(k): v for k,v in predicted_dict.items()}, fp)



if __name__ == '__main__':
    main()
