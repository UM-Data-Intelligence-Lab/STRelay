import torch
from torch import nn


class Temporal(nn.Module):
    def __init__(self, config):
        super(Temporal, self).__init__()
        self.base_dim = config.hidden_dim
        self.num_users = config.user_count
        self.poi_vocab_size = config.poi_count
        self.temporal_intervals = config.temporal_intervals

        self.num_heads = config.head
        self.head_dim = self.base_dim // self.num_heads
        self.poi_embedding = nn.Embedding(self.poi_vocab_size + 1, self.base_dim)
        self.timeslot_embedding = nn.Embedding(168, self.base_dim)
        self.timeinterval_embedding = nn.Embedding(self.temporal_intervals+1, self.base_dim)
        self.user_preference = nn.Embedding(self.num_users, self.base_dim)
        self.w_q = nn.ModuleList(
            [nn.Linear(self.base_dim + self.base_dim, self.head_dim) for _ in range(self.num_heads)])
        self.w_k = nn.ModuleList(
            [nn.Linear(self.base_dim, self.head_dim) for _ in range(self.num_heads)])
        self.w_v = nn.ModuleList(
            [nn.Linear(self.base_dim, self.head_dim) for _ in range(self.num_heads)])
        self.unify_heads = nn.Linear(self.base_dim, self.base_dim)


    def forward(self, poi_ids, t_slot, active_user):
        user_x = active_user
        hour_x = t_slot.permute(1, 0)  # (batch_size, seq_len)

        timeslot_embedded = self.timeslot_embedding(torch.arange(end=168, dtype=torch.int, device=t_slot.device))
        timeinterval_embedded = self.timeinterval_embedding(torch.arange(end=self.temporal_intervals, dtype=torch.int, device=t_slot.device))

        batch_size, sequence_length = hour_x.shape

        hour_x = hour_x.reshape(batch_size * sequence_length)
        user_preference = self.user_preference(user_x)
        user_feature = user_preference.unsqueeze(1).repeat(1, sequence_length, 1)
        time_feature = timeslot_embedded[hour_x].view(batch_size, sequence_length, -1)

        head_outputs = []

        query = torch.cat([user_feature, time_feature], dim=-1)
        key = timeinterval_embedded
        for i in range(self.num_heads):
            query_i = self.w_q[i](query)
            key_i = self.w_k[i](key)
            value_i = self.w_v[i](key)
            attn_scores_i = torch.matmul(query_i, key_i.T)
            scale = 1.0 / (key_i.size(-1) ** 0.5)
            attn_scores_i = attn_scores_i * scale
            attn_scores_i = torch.softmax(attn_scores_i, dim=-1)
            weighted_values_i = torch.matmul(attn_scores_i, value_i)
            head_outputs.append(weighted_values_i)
        head_outputs = torch.cat(head_outputs, dim=-1)
        head_outputs = head_outputs.view(batch_size, sequence_length, -1)
        return self.unify_heads(head_outputs).permute(1, 0, 2)

class Spatial(nn.Module):
    def __init__(self, config):
        super(Spatial, self).__init__()
        self.base_dim = config.hidden_dim
        self.num_users = config.user_count
        self.poi_vocab_size = config.poi_count
        self.spatial_intervals = config.spatial_intervals

        self.num_heads = config.head
        self.head_dim = self.base_dim // self.num_heads
        self.poi_embedding = nn.Embedding(self.poi_vocab_size + 1, self.base_dim)
        self.spatial_interval_embedding = nn.Embedding(self.spatial_intervals+1, self.base_dim)

        self.user_preference = nn.Embedding(self.num_users, self.base_dim)
        self.w_q = nn.ModuleList(
            [nn.Linear(self.base_dim + self.base_dim + self.base_dim, self.head_dim) for _ in range(self.num_heads)])
        self.w_k = nn.ModuleList(
            [nn.Linear(self.base_dim, self.head_dim) for _ in range(self.num_heads)])
        self.w_v = nn.ModuleList(
            [nn.Linear(self.base_dim, self.head_dim) for _ in range(self.num_heads)])
        self.unify_heads = nn.Linear(self.base_dim, self.base_dim)


    def forward(self, poi_ids, temporal_embedded, active_user):
        user_x = active_user

        disinterval_embedded = self.spatial_interval_embedding(torch.arange(end=self.spatial_intervals, dtype=torch.int, device=poi_ids.device))

        sequence_length, batch_size = poi_ids.shape

        user_preference = self.user_preference(user_x)
        user_feature = user_preference.unsqueeze(1).repeat(1, sequence_length, 1)

        time_feature = temporal_embedded.transpose(0, 1)
        
        poi_feature = self.poi_embedding(poi_ids).view(batch_size, sequence_length, -1)  # [batch, seq_len, embed_dim]
        head_outputs = []

        query = torch.cat([user_feature, time_feature, poi_feature], dim=-1)
        key = disinterval_embedded
        for i in range(self.num_heads):
            query_i = self.w_q[i](query)
            key_i = self.w_k[i](key)
            value_i = self.w_v[i](key)
            attn_scores_i = torch.matmul(query_i, key_i.T)
            scale = 1.0 / (key_i.size(-1) ** 0.5)
            attn_scores_i = attn_scores_i * scale
            attn_scores_i = torch.softmax(attn_scores_i, dim=-1)
            weighted_values_i = torch.matmul(attn_scores_i, value_i)
            head_outputs.append(weighted_values_i)
        head_outputs = torch.cat(head_outputs, dim=-1)
        head_outputs = head_outputs.view(batch_size, sequence_length, -1)
        return self.unify_heads(head_outputs).permute(1, 0, 2)
    

class STRelay(nn.Module):
    def __init__(self, config):
        super(STRelay, self).__init__()
        self.Temporal_predictor = Temporal(config)
        self.Spatial_predictor = Spatial(config)

    def forward(self, poi_ids, t_slot, active_user):
        temporal_context = self.Temporal_predictor(poi_ids, t_slot, active_user)
        spatial_context = self.Spatial_predictor(poi_ids, temporal_context, active_user)
        return temporal_context, spatial_context