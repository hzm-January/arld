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
            nn.Linear(512 * 25 * 5, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(1024, 512),
            # nn.Tanh()
        )
        self.encoders = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.z_dim, self.z_dim), nn.ReLU(), nn.Linear(self.z_dim, self.z_dim), nn.ReLU())
             for _ in range(self.max_n_lane)])

        self.mlp_bbox = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, self.inst_latent_len))
        self.mlp_comp = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1))

    # def forward(self, latent_z):
    #     n_batch = latent_z.size(0)
    #     latent_z = latent_z.contiguous().view(n_batch, 1, -1)  # bev (8,512,25,5) -> (8,1,64000)
    #     latent_z = self.flatten(latent_z)  # (8,1,64000) -> (8,1,1024)
    #     index = torch.arange(n_batch, dtype=torch.long).view(n_batch,1)
    #     obj_feats = self.empty_token_embedding(index)  # 取出batch size个word，作为每个样本的初始车道线
    #
    #     for idx in range(self.max_n_lane):
    #         X = torch.cat(obj_feats, dim=1)
    #         # if idx > 0:
    #         #     X = X.detach()
    #         X = self.transformer_encoder(X, length_mask=None)  # 得到Fk，作为key, value (1,1,512)
    #         last_feat = self.transformer_decoder(latent_z, X)  # latent_z 作为query
    #         obj_feats.append(self.encoders[idx](last_feat))
    #
    #     obj_feats = torch.cat(obj_feats[1:], dim=1)[:, :self.max_n_lane]
    #     box_feat = self.mlp_bbox(obj_feats)
    #     completenesss_feat = self.mlp_comp(obj_feats)
    #     return box_feat, completenesss_feat

    def forward(self, latent_codes):
        n_batch = latent_codes.size(0)
        latent_codes = latent_codes.contiguous().view(n_batch, -1)  # bev (8,512,25,5) -> (8,64000)
        latent_codes = self.flatten(latent_codes).unsqueeze(1)  # (8,64000) -> (8,1,1024)
        output_box_feats = []
        output_completenesss_feat = []
        for batch_id in range(n_batch):
            latent_z = latent_codes[[batch_id]]  # (1,1,1024)
            # prepare encoder input: start with empty token, will accumulate with more objects
            #  [(1,1,512)]
            obj_feats = [self.empty_token_embedding(torch.LongTensor([batch_id]).cuda())[:, None]]

            for idx in range(self.max_n_lane):
                X = torch.cat(obj_feats, dim=1)  # (1,k,512)
                X = self.transformer_encoder(X, length_mask=None)  # (1,k,512) -> (1,k,512)
                last_feat = self.transformer_decoder(latent_z, X)  # latent_z (1,1,512), X (1,k,512) -> (1,1,512)
                last_feat = self.encoders[idx](last_feat)  # (1,1,512) -> (1,1,512)
                obj_feats.append(last_feat)

            obj_feats = torch.cat(obj_feats[1:], dim=1)  # [(1,1,512),(1,1,512),...] -> (1,K,512)
            box_feat = self.mlp_bbox(obj_feats)  # (1,K,512) -> (1,K,1024)
            completenesss_feat = self.mlp_comp(obj_feats)  # (1,K,512) -> (1,K,1)

            output_box_feats.append(box_feat)
            output_completenesss_feat.append(completenesss_feat)

        return output_box_feats, output_completenesss_feat

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
