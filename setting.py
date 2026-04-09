import torch
import argparse
import sys

from network import RnnFactory


class Setting:
    """ Defines all settings in a single place using a command line interface.
    """

    def parse(self):
        self.guess_foursquare = any(['4sq' in argv for argv in sys.argv])  # foursquare has different default args.

        parser = argparse.ArgumentParser()
        if self.guess_foursquare:
            self.parse_foursquare(parser)
        else:
            self.parse_gowalla(parser)
        self.parse_arguments(parser)
        args = parser.parse_args()

        ###### settings ######
        # training
        self.gpu = args.gpu
        self.hidden_dim = args.hidden_dim  # 10
        self.weight_decay = args.weight_decay  # 0.0
        self.learning_rate = args.lr  # 0.01
        self.epochs = args.epochs  # 100
        self.rnn_factory = RnnFactory(args.rnn)  # RNN:0, GRU:1, LSTM:2
        self.is_lstm = self.rnn_factory.is_lstm()  # True or False
        self.lambda_t = args.lambda_t  # 0.01
        self.lambda_s = args.lambda_s  # 100 or 1000
        self.STRelay = args.STRelay == 'True'  # True or False
        self.temporal_intervals = args.temporal_intervals  # 24
        self.spatial_intervals = args.spatial_intervals  #
        self.head = args.head 

        # data management
        self.dataset = args.dataset
        self.dataset_file = './data/{}'.format(args.dataset)
        self.friend_file = './data/{}'.format(args.friendship)
        self.max_users = 0  # 0 = use all available users
        self.sequence_length = 20  # 将用户的所有check-in轨迹划分成固定长度为20的多个子轨迹
        self.batch_size = args.batch_size
        self.min_checkins = 101

        # evaluation        
        self.validate_epoch = args.validate_epoch  # 每5轮验证一次
        self.report_user = args.report_user  # -1

        # log
        self.log_file = args.log_file

        self.trans_loc_file = args.trans_loc_file  # 时间POI graph
        self.trans_loc_spatial_file = args.trans_loc_spatial_file  # 空间POI graph
        self.trans_user_file = args.trans_user_file
        self.trans_interact_file = args.trans_interact_file

        self.lambda_user = args.lambda_user
        self.lambda_loc = args.lambda_loc

        self.use_weight = args.use_weight
        self.use_graph_user = args.use_graph_user
        self.use_spatial_graph = args.use_spatial_graph

        ### CUDA Setup ###
        self.device = torch.device('cpu') if args.gpu == -1 else torch.device('cuda', args.gpu)

    def parse_arguments(self, parser):
        # training
        parser.add_argument('--gpu', default=1, type=int, help='the gpu to use')  # -1
        parser.add_argument('--hidden-dim', default=10, type=int, help='hidden dimensions to use')  # 10
        parser.add_argument('--weight_decay', default=0, type=float, help='weight decay regularization')
        parser.add_argument('--lr', default=0.01, type=float, help='learning rate')  # 0.01
        parser.add_argument('--epochs', default=100, type=int, help='amount of epochs')  # 100
        parser.add_argument('--rnn', default='lstm', type=str, help='the GRU implementation to use: [rnn|gru|lstm]')
        parser.add_argument('--STRelay', type=str, default='True', choices=['True', 'False'], help='Use STRelay or not (True/False)')
        parser.add_argument('--temporal_intervals', type=int, default=24, help='the number of temporal intervals')
        parser.add_argument('--spatial_intervals', type=int, default=30, help='the number of spatial intervals')
        parser.add_argument('--head', type=float, default=2, help='attention head')
        # data management
        parser.add_argument('--dataset', default='Istanbul.txt', type=str,
                            help='the dataset under ./data/<dataset.txt> to load')
        parser.add_argument('--friendship', default='', type=str,
                            help='the friendship file under ../data/<edges.txt> to load')
        # evaluation        
        parser.add_argument('--validate-epoch', default=5, type=int,
                            help='run each validation after this amount of epochs')
        parser.add_argument('--report-user', default=-1, type=int,
                            help='report every x user on evaluation (-1: ignore)')

        # log
        parser.add_argument('--log_file', default='./results/log', type=str,
                            help='存储结果日志')
        parser.add_argument('--trans_loc_file', default='./KGE/Graphs/gowalla_scheme2_transe_loc_temporal_100.pkl', type=str,
                            help='使用transe方法构造的时间POI转换图')
        parser.add_argument('--trans_user_file', default='', type=str,
                            help='使用transe方法构造的user转换图')
        parser.add_argument('--trans_loc_spatial_file', default='', type=str,
                            help='使用transe方法构造的空间POI转换图')
        parser.add_argument('--trans_interact_file', default='./KGE/Graphs/gowalla_scheme2_transe_user-loc_100.pkl', type=str,
                            help='使用transe方法构造的用户-POI交互图')
        parser.add_argument('--use_weight', default=False, type=bool, help='应用于GCN的AXW中是否使用W')
        parser.add_argument('--use_graph_user', default=False, type=bool, help='是否使用user graph')
        parser.add_argument('--use_spatial_graph', default=False, type=bool, help='是否使用空间POI graph')

    def parse_gowalla(self, parser):
        # defaults for gowalla dataset
        parser.add_argument('--batch-size', default=64, type=int,  
                            help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=1000, type=float, help='decay factor for spatial data')
        parser.add_argument('--lambda_loc', default=1.0, type=float, help='weight factor for transition graph')
        parser.add_argument('--lambda_user', default=1.0, type=float, help='weight factor for user graph')

    def parse_foursquare(self, parser):
        # defaults for foursquare dataset
        parser.add_argument('--batch-size', default=512, type=int,
                            help='amount of users to process in one pass (batching)')  # 1024
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')
        parser.add_argument('--lambda_loc', default=1.0, type=float, help='weight factor for transition graph')
        parser.add_argument('--lambda_user', default=1.0, type=float, help='weight factor for user graph')

    def __str__(self):        
        settings = [
            f"Dataset Settings:",
            f"  Dataset: {self.dataset}",
            f"  Batch Size: {self.batch_size}",
            f"",
            f"Training Settings:",
            f"  GPU: {self.gpu}",
            f"  Hidden Dimensions: {self.hidden_dim}",
            f"  Learning Rate: {self.learning_rate}",
            f"  Epochs: {self.epochs}",
            f"  Weight Decay: {self.weight_decay}",
            f"  Lambda Temporal: {self.lambda_t}",
            f"  Lambda Spatial: {self.lambda_s}",
            f"  Use STRelay: {self.STRelay}",
            f"  Temporal intervals Number: {self.temporal_intervals}",
            f"  Sptial intervals Number: {self.spatial_intervals}",
        ]
        return "\n".join(settings)