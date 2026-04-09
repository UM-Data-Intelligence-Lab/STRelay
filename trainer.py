import time

import torch
import torch.nn as nn
import numpy as np
from utils import *
from network import Flashback
from scipy.sparse import csr_matrix


class FlashbackTrainer():
    """ Instantiates Flashback module with spatial and temporal weight functions.
    Performs loss computation and prediction.
    """

    def __init__(self, lambda_t, lambda_s, lambda_loc, lambda_user, use_weight, transition_graph, spatial_graph,
                 friend_graph, use_graph_user, use_spatial_graph, interact_graph):
        """ The hyper parameters to control spatial and temporal decay.
        """
        self.lambda_t = lambda_t
        self.lambda_s = lambda_s

        self.lambda_loc = lambda_loc
        self.lambda_user = lambda_user
        self.use_weight = use_weight
        self.use_graph_user = use_graph_user
        self.use_spatial_graph = use_spatial_graph
        self.graph = transition_graph
        self.spatial_graph = spatial_graph
        self.friend_graph = friend_graph
        self.interact_graph = interact_graph

    def __str__(self):
        return 'Use flashback training.'

    def count_parameters(self):
        param_count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_count += param.numel()
        return param_count
    
    def parameters(self):
        return self.model.parameters()

    def prepare(self, loc_count, user_count, hidden_size, gru_factory, device, config=None):
        def f_t(delta_t, user_len): return ((torch.cos(delta_t * 2 * np.pi / 86400) + 1) / 2) * torch.exp(
            -(delta_t / 86400 * self.lambda_t))  # hover cosine + exp decay

        # exp decay  2个functions
        def f_s(delta_s, user_len): return torch.exp(-(delta_s * self.lambda_s))
        self.loc_count = loc_count
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.model = Flashback(loc_count, user_count, hidden_size, f_t, f_s, gru_factory, self.lambda_loc,
                               self.lambda_user, self.use_weight, self.graph, self.spatial_graph, self.friend_graph,
                               self.use_graph_user, self.use_spatial_graph, self.interact_graph, config).to(device)
        self.config = config


    def evaluate(self, x, t, t_slot, s, y_t, y_t_slot, y_s, y_timeintervals, y_distances, active_users):

        self.model.eval()
        out, t_linear, d_linear = self.model(x, t, t_slot, s, y_t, y_t_slot, y_s, y_timeintervals, y_distances, active_users)
        out_t = out.transpose(0, 1)
        return out_t 
    
    def loss(self, x, t, t_slot, s, y, y_t, y_t_slot, y_s, y_timeintervals, y_distances, active_users):

        self.model.train()
        out, t_linear, d_linear = self.model(x, t, t_slot, s, y_t, y_t_slot, y_s, y_timeintervals, y_distances, active_users)  
        out = out.view(-1, self.loc_count)  
        y = y.view(-1)  
        l = self.cross_entropy_loss(out, y)
        if self.config.STRelay:
            t_linear = t_linear.view(-1, self.config.temporal_intervals + 1)
            l_t = self.cross_entropy_loss(t_linear, y_timeintervals.view(-1)) * 0.5
            l += l_t

            d_linear = d_linear.view(-1, self.config.spatial_intervals + 1)
            l_d = self.cross_entropy_loss(d_linear, y_distances.view(-1)) * 0.5
            l += l_d           
        return l