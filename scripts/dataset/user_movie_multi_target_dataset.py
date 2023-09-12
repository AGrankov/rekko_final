from torch.utils import data
import numpy as np

class UserMovieMultiTargetDataset(data.Dataset):
    def __init__(self,
                 users_movies_sequences,
                 additional_info,
                 movies_dict,
                 seq_len=10,
                 targets=None,
                 targets_size=None,
                 is_shuffling=False):
        self.users_movies_sequences = users_movies_sequences
        self.additional_info = additional_info
        self.movies_dict = movies_dict
        self.targets = targets
        self.targets_size = targets_size
        self.is_shuffling = is_shuffling

        self.data_size = len(list(self.movies_dict.values())[0])
        self.additional_size = len(additional_info[0][0])
        self.seq_len = seq_len

    def __len__(self):
        return len(self.targets) if self.targets is not None else len(self.users_movies_sequences)

    def __getitem__(self, index):
        if self.targets is None:
            uidx = index
            stopidx = len(self.users_movies_sequences[uidx])
            target = None
        else:
            uidx, stopidx, targets_seq = self.targets[index]
            target = np.zeros((self.targets_size), dtype=np.float32)
            for val in targets_seq:
                target[val] = 1

        res_arr = np.zeros((self.seq_len, self.data_size + self.additional_size),
                            dtype=np.float32)

        seq = self.users_movies_sequences[uidx]
        seq = np.array(seq[:stopidx][::-1])

        add_seq = self.additional_info[uidx]
        add_seq = add_seq[:stopidx][::-1]

        if self.is_shuffling:
            shuffled_idxes = np.arange(len(seq))
            np.random.shuffle(shuffled_idxes)
            seq = seq[shuffled_idxes]
            add_seq = add_seq[shuffled_idxes]

        seq = seq[:self.seq_len]
        add_seq = add_seq[:self.seq_len]

        for idx in range(len(seq)):
            res_arr[self.seq_len-idx-1, :self.data_size] = self.movies_dict[seq[idx]]
            res_arr[self.seq_len-idx-1, self.data_size:self.data_size+self.additional_size] =\
                add_seq[idx]

        if self.targets is not None:
            return res_arr, target

        return res_arr

class UserMovieMultiTargetCheckDataset(data.Dataset):
    def __init__(self,
                 users_movies_sequences,
                 additional_info,
                 movies_dict,
                 seq_len=10,
                 targets=None,
                 targets_size=None):
        self.users_movies_sequences = users_movies_sequences
        self.additional_info = additional_info
        self.movies_dict = movies_dict
        self.targets = targets
        self.targets_size = targets_size

        self.data_size = len(list(self.movies_dict.values())[0])
        self.additional_size = len(additional_info[0][0])
        self.seq_len = seq_len

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        uidx, stopidx, targets_seq = self.targets[index]

        res_arr = np.zeros((self.seq_len, self.data_size + self.additional_size),
                            dtype=np.float32)

        seq = self.users_movies_sequences[uidx]
        seq = np.array(seq[:stopidx][::-1])
        already_viewed = np.zeros((self.targets_size), dtype=np.uint8)
        for val in seq:
            already_viewed[val-1] = 1

        add_seq = self.additional_info[uidx]
        add_seq = add_seq[:stopidx][::-1]

        seq = seq[:self.seq_len]
        add_seq = add_seq[:self.seq_len]

        for idx in range(len(seq)):
            res_arr[self.seq_len-idx-1, :self.data_size] = self.movies_dict[seq[idx]]
            res_arr[self.seq_len-idx-1, self.data_size:self.data_size+self.additional_size] =\
                add_seq[idx]

        return res_arr, already_viewed
