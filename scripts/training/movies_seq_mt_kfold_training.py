import sys
sys.path.append('..')
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import gc
from torch.autograd import Variable, Function
import torch.nn as nn
import random

from sklearn.model_selection import KFold

from scripts.utils.utils import labeling_col, process_sequence
from scripts.utils.torch_train_utils import progress_bar
from scripts.models.movie_features_seq_model import MovieRNNModel
from scripts.dataset.user_movie_multi_target_dataset import UserMovieMultiTargetDataset, UserMovieMultiTargetCheckDataset

tqdm.monitor_interval = 0
torch.backends.cudnn.benchmark = True

def apk(actual, predicted, k=20):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if len(actual) == 0:
        return 0.0

    return score / min(len(actual), k)

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
    parser = argparse.ArgumentParser(description='Train sequence model based on pretrained Item2Vec and videos meta information')

    parser.add_argument('-i', '--input', default='../p_input/transactions.csv', help='input data file name')
    parser.add_argument('-mi', '--meta-information', default='../p_input/catalogue.csv', help='meta information file name')
    parser.add_argument('-v', '--vectors', default='../models/fasttext_cbow_model.vec', help='path to pretrained vectors')
    parser.add_argument('-dim', '--dimension', default=64, help='dimention of embeddings')
    parser.add_argument('-sl', '--sequence-length', default=100, help='length of sequences')
    parser.add_argument('-ss', '--sequence-step', default=5, help='length of sequences')
    parser.add_argument('-sm', '--sequence-min', default=3, help='minimum sequence length')

    parser.add_argument('-mp', '--model-path', default='../models/', help='path to models directory')
    parser.add_argument('-m', '--model-name', default='movies_seq_emb_mt_model_kfold_{}_v9', help='name of trained model')
    parser.add_argument('-bs', '--batch-size', default=2048, help='size of batches')
    parser.add_argument('-lr', '--learning-rate', default=0.001, help='learning rate')
    parser.add_argument('-e', '--epochs', default=30, help='epochs count')

    parser.add_argument('-kf', '--kfolds', default=6, help='kfold count for split')
    parser.add_argument('-s', '--seed', default=42, help='seed')

    args = parser.parse_args()


    df = pd.read_csv(args.input)
    movie_id_trans = labeling_col(df, 'element_uid')
    movies_sequences = process_sequence(df, 'element_uid')
    wp = process_sequence(df, 'watch_percent')
    cm = process_sequence(df, 'consumption_mode')   # 3
    dt = process_sequence(df, 'device_type')        # 7
    dfts = process_sequence(df, 'ts')

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


    folds = KFold(n_splits=args.kfolds, shuffle=False, random_state=args.seed)
    run_folds = [0, 1, 2, 3, 4, 5]

    for fold_idx, (trn_idx, val_idx) in enumerate(folds.split(np.arange(len(movies_sequences)))):
        if fold_idx not in run_folds:
            continue

        training_tuples = []
        validation_tuples = []

        for si, seq in tqdm(enumerate(movies_sequences), total=len(movies_sequences)):
            if si in val_idx:
                for idx in range(len(seq)):
                    if dfts[si][idx] > 43447000:
                        break
                targets = []
                for sub_idx in range(idx, len(seq)):
                    # see to movie type ohe, zero index 'is movie' flag
                    if ((movies_dict[seq[sub_idx]+1][len(int_cols)] == 1) and (wp[si][sub_idx] > 0.5)) or\
                        ((movies_dict[seq[sub_idx]+1][len(int_cols)] == 0) and (wp[si][sub_idx] > 0.3)):
                        targets.append(seq[sub_idx])
                validation_tuples.append((si, idx, targets))
            else:
                for idx in range(args.sequence_min, len(seq), args.sequence_step):
                    targets = []
                    for sub_idx in range(idx, len(seq)):
                        # see to movie type ohe, zero index 'is movie' flag
                        if ((movies_dict[seq[sub_idx]+1][len(int_cols)] == 1) and (wp[si][sub_idx] > 0.5)) or\
                            ((movies_dict[seq[sub_idx]+1][len(int_cols)] == 0) and (wp[si][sub_idx] > 0.3)):
                            targets.append(seq[sub_idx])

                    if len(targets) > 0:
                        training_tuples.append((si, idx, targets))

        gc.collect()


        model_path = os.path.join(args.model_path, args.model_name.format(fold_idx))

        if not os.path.isdir(model_path):
            os.mkdir(model_path)

        trainset = UserMovieMultiTargetDataset(all_users_seq,
                                              all_users_additional,
                                              movies_dict,
                                              seq_len=args.sequence_length,
                                              targets=training_tuples,
                                              targets_size=len(movie_id_trans.classes_),
                                              is_shuffling=False)

        validset = UserMovieMultiTargetDataset(all_users_seq,
                                              all_users_additional,
                                              movies_dict,
                                              seq_len=args.sequence_length,
                                              targets=validation_tuples,
                                              targets_size=len(movie_id_trans.classes_))

        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=6,
                                                  pin_memory=False)

        validloader = torch.utils.data.DataLoader(validset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=4,
                                                  pin_memory=False)

        checkset = UserMovieMultiTargetCheckDataset(
                 users_movies_sequences=all_users_seq,
                 additional_info=all_users_additional,
                 movies_dict=movies_dict,
                 seq_len=args.sequence_length,
                 targets=validation_tuples,
                 targets_size=len(movie_id_trans.classes_),
        )

        checkloader = torch.utils.data.DataLoader(checkset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=4,
                                                  pin_memory=False)

        model = MovieRNNModel(
            seq_len=args.sequence_length,
            features_dim=trainset[0][0].shape[1],
            result_classes=len(movie_id_trans.classes_),

            rnn_units=128,
            mid_dim=2048,
            mid_dim2=2048,
            pool_count=1,
            num_layers=1
        )

        model = torch.nn.DataParallel(model).cuda()

        best_loss = np.finfo(np.float32).max
        best_score = 0

        criterion = torch.nn.BCEWithLogitsLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        lr_sch = torch.optim.lr_scheduler.StepLR(optimizer, 4, 0.6)

        def train(epoch, model, trainloader, optimizer, criterion):
            '''
            Train function for each epoch
            '''
            model.train()
            train_loss = 0

            print('Training Epoch {} optimizer LR {}'.format(epoch, optimizer.param_groups[0]['lr']))

            for batch_idx, (seq_data, targets) in enumerate(trainloader):
                seq_data = seq_data.cuda()
                targets = targets.cuda()

                outputs = model(seq_data)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                curr_batch_loss = loss.item()
                train_loss += curr_batch_loss

                progress_bar(batch_idx,
                             len(trainloader),
                             'Loss: {l:.8f}'.format(l = train_loss/(batch_idx+1)))

        def validate(epoch, model, validloader, criterion, best_loss):
            '''
            Validate function for each epoch
            '''
            print('\nValidate Epoch: %d' % epoch)
            model.eval()
            eval_loss = 0

            with torch.no_grad():
                for batch_idx, (seq_data, targets) in enumerate(validloader):
                    seq_data = seq_data.cuda()
                    targets = targets.cuda()

                    outputs = model(seq_data)
                    loss = criterion(outputs, targets)

                    curr_batch_loss = loss.item()
                    eval_loss += curr_batch_loss

                    progress_bar(batch_idx,
                                 len(validloader),
                                 'Loss: {l:.8f}'.format(l = eval_loss/(batch_idx+1)))

            if eval_loss < best_loss:
                print('Saving..')
                state = {
                    'net': model.state_dict(),
                    'loss': eval_loss,
                    'epoch': epoch,
                }
                session_checkpoint = os.path.join(model_path, "best_model_chkpt.t7")
                torch.save(state, session_checkpoint)
                best_loss = eval_loss

            return best_loss

        def validate_mapk(epoch, model, checkloader, truth_labels, best_map):
            '''
            Validate function for each epoch
            '''
            print('\nValidate Epoch: %d' % epoch)
            model.eval()

            preds = []

            with torch.no_grad():
                for batch_idx, (seq_data, al_viewed) in enumerate(checkloader):
                    seq_data = seq_data.cuda()
                    al_viewed = al_viewed.cuda()
                    outputs = model(seq_data)

                    outputs.masked_fill_(al_viewed, -100)
                    sort_dist, sort_indices = outputs.sort(dim=1)

                    preds.append(sort_indices[:, -20:].cpu().numpy()[:, ::-1])

            preds = np.concatenate(preds)

            score = []
            for idx, truth in enumerate(truth_labels):
                score.append(apk(truth, preds[idx]))
            score = np.mean(score)
            print("MAPK20 score is {}".format(score))


            if score > best_map:
                print('Saving..')
                state = {
                    'net': model.state_dict(),
                    'mapk20': score,
                    'epoch': epoch,
                }
                session_checkpoint = os.path.join(model_path, "best_map_model.t7")
                torch.save(state, session_checkpoint)
                best_map = score

            return best_map

        try:
            for epoch in range(args.epochs):
                lr_sch.step(epoch)
                train(epoch, model, trainloader, optimizer, criterion)
                best_loss = validate(epoch, model, validloader, criterion, best_loss)
                best_score = validate_mapk(epoch, model, checkloader, [x for _,_,x in validation_tuples], best_score)
        except Exception as e:
            print (e.message)

        del model
        del trainloader
        del validloader
        del trainset
        del validset
        del criterion
        del optimizer

        gc.collect()


if __name__ == '__main__':
    main()
