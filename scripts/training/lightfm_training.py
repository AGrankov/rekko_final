import sys
sys.path.append('..')
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy
import pickle

from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split

from scripts.utils.utils import labeling_col

tqdm.monitor_interval = 0

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


def main():
    parser = argparse.ArgumentParser(description='Train lightfm model')

    parser.add_argument('-i', '--input', default='../p_input/transactions.csv', help='input data file name')

    parser.add_argument('-fc', '--folds-count', default=5, help='count of folds')
    parser.add_argument('-fss', '--folds-seed', default=4242, help='seed')

    args = parser.parse_args()


    df = pd.read_csv(args.input)
    movie_id_trans = labeling_col(df, 'element_uid')
    user_id_trans = labeling_col(df, 'user_uid')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    wp_matrix = scipy.sparse.coo_matrix(
        (df['watch_percent'].astype(np.float32),
            (
                df['user_uid'].values.copy(),
                df['element_uid'].values.copy(),
            )
        )
    )

    for fold_idx in range(args.folds_count):
        train_interactions, test_interactions = random_train_test_split(wp_matrix,
                                        test_percentage=1/args.folds_count,
                                        random_state=np.random.RandomState(seed=args.folds_seed + fold_idx))

        test_ints = test_interactions.tocsr()
        test_ints_dict = dict()
        for i in tqdm(range(test_ints.shape[0])):
            idxes = (test_ints[i] > 0.3).indices
            if len(idxes) > 0:
                test_ints_dict[i] = idxes

        train_ints = train_interactions.tocsr()

        def evaluate(model, test_dict, train_ints):
            item_count = train_ints.shape[1]
            mapk20_scores = []
            for userId, movies in tqdm(test_dict.items()):
                preds = model.predict(userId,
                                        np.arange(item_count),
                                        num_threads=12)
                preds[train_ints[userId].indices] = -10
                preds = np.argsort(preds)[::-1]
                score = apk(movies, preds, 20)
                mapk20_scores.append(score)
            return np.mean(mapk20_scores)

        alpha = 1e-05
        num_components = 32

        model = LightFM(no_components=num_components,
                            loss='warp',
                            learning_schedule='adagrad',
                            learning_rate=0.01,
                            max_sampled=100,
                            user_alpha=alpha,
                            item_alpha=alpha
                            )

        print('training start')
        model.fit_partial(train_interactions,
                        epochs=150,
                        num_threads=12,
                        verbose=True
                        )

        # print('validation start')
        # valid_score = evaluate(model, test_ints_dict, train_ints)
        # print('valid score {}'.format(valid_score))
        # print('adopted valid score {}'.format(valid_score * len(test_ints_dict) / test_interactions.shape[0]))

        with open('../models/lightfm_v2_fold_{}.pickle'.format(fold_idx), 'wb') as file:
            pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    main()
