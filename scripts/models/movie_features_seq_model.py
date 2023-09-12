import torch.nn as nn
import torch

class MovieRNNModel(nn.Module):
    def __init__(self,
                 seq_len,
                 features_dim,
                 result_classes,
                 rnn_units=32,
                 mid_dim=16,
                 mid_dim2=16,
                 pool_count=1,
                 num_layers=1
                ):
        super(MovieRNNModel, self).__init__()


        self.gru = nn.GRU(
                            input_size=features_dim,
                            hidden_size=rnn_units,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            )

        self.avg_pooling = nn.AdaptiveAvgPool1d(pool_count)
        self.max_pooling = nn.AdaptiveMaxPool1d(pool_count)

        dense_first = seq_len * pool_count * 2 + rnn_units * 2

        self.classifier = nn.Sequential(
            nn.Linear(dense_first, mid_dim),
            nn.Tanh(),
            nn.Dropout(0.25),
            nn.Linear(mid_dim, mid_dim2),
            nn.Tanh(),
            nn.Dropout(0.25),
            nn.Linear(mid_dim2, result_classes),
        )

    def forward(self, x):
        B = x.shape[0]

        self.gru.flatten_parameters()
        gru_res, _ = self.gru(x)
        avg_gru_res = self.avg_pooling(gru_res).view(B, -1)
        max_gru_res = self.max_pooling(gru_res).view(B, -1)
        last_gru_res = gru_res[:, -1]

        conc_gru_res = torch.cat([
            avg_gru_res,
            max_gru_res,
            last_gru_res
        ], dim=1)

        out = self.classifier(conc_gru_res)
        return out
