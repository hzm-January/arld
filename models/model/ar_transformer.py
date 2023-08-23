#  Copyright (c) 3.2022. Yinyu Nie
#  License: MIT
import torch
import torch.nn as nn
from external.fast_transformers.fast_transformers.builders import TransformerEncoderBuilder, TransformerDecoderBuilder


class AutoregressiveTransformer(nn.Module):
    # def __init__(self, cfg, optim_spec=None, device='cuda'):
    def __init__(self, cfg, device='cuda'):
        '''
        Encode scene priors from embeddings
        :param cfg: configuration file.
        :param optim_spec: optimizer parameters.
        '''
        super(AutoregressiveTransformer, self).__init__()
        '''Optimizer parameters used in training'''
        # self.optim_spec = optim_spec
        self.device = device

        '''Network'''
        # Parameters
        self.z_dim = cfg.tf_config['z_dim']  # vector length 512
        self.inst_latent_len = cfg.tf_config['latent_len']  # 1024
        self.max_n_lane = cfg.tf_config['max_n_lane']  # 53
        d_model = cfg.tf_config['z_dim']  # 512 total length of q,k,v
        n_head = cfg.tf_config['n_head']  # the number of heads

        # Build Networks
        # empty room token in transformer encoder
        # self.empty_token_embedding = nn.Embedding(len(cfg.room_types), self.z_dim)
        self.empty_token_embedding = nn.Embedding(100, self.z_dim)

        # Build a transformer encoder
        self.transformer_encoder = TransformerEncoderBuilder.from_kwargs(
            n_layers=1,
            n_heads=n_head,
            query_dimensions=d_model // n_head,
            value_dimensions=d_model // n_head,
            feed_forward_dimensions=d_model,
            attention_type="full",
            activation="gelu",
        ).get()

        self.transformer_decoder = TransformerDecoderBuilder.from_kwargs(
            n_layers=1,
            n_heads=n_head,
            query_dimensions=d_model // n_head,
            value_dimensions=d_model // n_head,
            feed_forward_dimensions=d_model,
            self_attention_type="full",
            cross_attention_type="full",
            activation="gelu",
        ).get()

        self.flatten = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(512 * 25 * 5, 2048),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            # nn.Tanh()
        )
        self.encoders = nn.ModuleList([nn.Sequential(nn.Linear(self.z_dim, self.z_dim), nn.ReLU(), nn.Linear(self.z_dim, self.z_dim), nn.ReLU()) for _ in range(self.max_n_lane)])

        self.mlp_bbox = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, self.inst_latent_len))
        self.mlp_comp = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1))
    def forward(self, latent_z):
        n_batch = latent_z.size(0)
        latent_z = latent_z.contiguous().view(latent_z.size(0), 1, -1)  # bev (8,512,25,5) -> (8,1,64000)
        latent_z = self.flatten(latent_z)  # (8,1,64000) -> (8,1,1024)
        obj_feats = [self.empty_token_embedding(torch.LongTensor([0]).cuda())[:, None] for _ in range(n_batch)]

        for idx in range(self.max_n_lane):
            X = torch.cat(obj_feats, dim=1)
            # if idx > 0:
            #     X = X.detach()
            X = self.transformer_encoder(X, length_mask=None)  # 得到Fk，作为key, value (1,1,512)
            last_feat = self.transformer_decoder(latent_z, X)  # latent_z 作为query
            obj_feats.append(self.encoders[idx](last_feat))

        obj_feats = torch.cat(obj_feats[1:], dim=1)[:, :self.max_n_lane]
        box_feat = self.mlp_bbox(obj_feats)
        completenesss_feat = self.mlp_comp(obj_feats)
        return box_feat, completenesss_feat
    # def forward(self, latent_z):
    #     latent_z = latent_z.contiguous().view(latent_z.size(0), -1)  # bev (8,512,25,5) -> (8,64000)
    #     latent_z = self.flatten(latent_z)  # (8,64000) -> (8,1024)
    #     batch_size = latent_z.shape[0]
    #     b_obj_feats = []
    #     b_completenesss_feats = []
    #
    #     obj_feats = [self.empty_token_embedding(torch.LongTensor([0]).cuda())[:, None] for _ in range(batch_size)]
    #     for idx in range(self.max_n_lane):
    #         # X = torch.cat(obj_feats, dim=1)
    #         X = torch.cat(obj_feats, dim=1)
    #         # if idx > 0:
    #         #     X = X.detach()
    #         X = self.transformer_encoder(X, length_mask=None)
    #         last_feat = self.transformer_decoder(latent_z, X)
    #         obj_feats.append(self.encoders[idx](last_feat))
    #
    #     for index in range(batch_size):
    #         # TODO: trans latent_z to vector
    #         # obj_feats = [self.empty_token_embedding(room_type_idx[:, 0])[:, None]]
    #         obj_feats = [self.empty_token_embedding(torch.LongTensor([0]).cuda())]  #list [(1,512)]
    #         # obj_feats = [self.empty_token_embedding(torch.LongTensor([0]).cuda())[:, None]] # list [(1,1,512)]
    #
    #         for idx in range(self.max_n_lane):
    #             # X = torch.cat(obj_feats, dim=1)
    #             X = torch.cat(obj_feats, dim=0)
    #             # if idx > 0:
    #             #     X = X.detach()
    #             X = self.transformer_encoder(X, length_mask=None)
    #             last_feat = self.transformer_decoder(latent_z, X)
    #             obj_feats.append(self.encoders[idx](last_feat))
    #
    #         obj_feats = torch.cat(obj_feats[1:], dim=1)[:, :self.max_n_lane]
    #         box_feat = self.mlp_bbox(obj_feats)
    #         completenesss_feat = self.mlp_comp(obj_feats)
    #         b_obj_feats.append(box_feat)
    #         b_completenesss_feats.append(completenesss_feat)
    #     return b_obj_feats, b_completenesss_feats

    @torch.no_grad()
    def generate_boxes(self, latent_codes, room_type_idx, pred_gt_matching=None, self_end=False, threshold=0.5,
                       **kwargs):
        '''Generate boxes from latent codes'''
        if pred_gt_matching is None:
            self_end = True

        assert self_end != (pred_gt_matching is not None)

        n_batch = latent_codes.size(0)
        output_feats = []
        for batch_id in range(n_batch):
            latent_z = latent_codes[[batch_id]]

            # prepare encoder input: start with empty token, will accumulate with more objects
            obj_feats = [self.empty_token_embedding(room_type_idx[[batch_id], 0])[:, None]]

            for idx in range(self.max_obj_num):
                X = torch.cat(obj_feats, dim=1)
                X = self.transformer_encoder(X, length_mask=None)
                last_feat = self.transformer_decoder(latent_z, X)
                last_feat = self.encoders[idx](last_feat)
                obj_feats.append(last_feat)

                if self_end:
                    completeness = self.mlp_comp(last_feat).sigmoid()
                    if completeness > threshold:
                        break

            obj_feats = torch.cat(obj_feats[1:], dim=1)
            box_feat = self.mlp_bbox(obj_feats)

            if pred_gt_matching is not None:
                box_feat = box_feat[:, pred_gt_matching[batch_id][0]]

            output_feats.append(box_feat)

        return output_feats
