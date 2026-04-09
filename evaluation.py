import torch
import numpy as np
from utils import log_string
import math

class Evaluation:
    """
    Handles evaluation on a given POI dataset and loader.

    The two metrics are MRR and ACC@n. Our model predicts sequence of
    next locations determined by the sequence_length at one pass. During evaluation we
    treat each entry of the sequence as single prediction. One such prediction
    is the ranked list of all available locations and we can compute the two metrics.

    As a single prediction is of the size of all available locations,
    evaluation takes its time to compute. The code here is optimized.

    Using the --report_user argument one can ACCess the statistics per user.
    """

    def __init__(self, dataset, dataloader, user_count, h0_strategy, trainer, setting, log):
        self.dataset = dataset
        self.dataloader = dataloader
        self.user_count = user_count
        self.h0_strategy = h0_strategy
        self.trainer = trainer
        self.setting = setting
        self._log = log


    def evaluate(self):
        self.dataset.reset()
        h = self.h0_strategy.on_init(self.setting.batch_size, self.setting.device)
        
        with torch.no_grad():        
            iter_cnt = 0
            ACC5 = 0
            ACC10 = 0
            ndcg5 = 0
            ndcg10 = 0
            average_precision = 0.
            
            u_iter_cnt = np.zeros(self.user_count)
            u_ACC5 = np.zeros(self.user_count)
            u_ACC10 = np.zeros(self.user_count)
            u_ndcg5 = np.zeros(self.user_count)
            u_ndcg10 = np.zeros(self.user_count)
            u_average_precision = np.zeros(self.user_count)        
            reset_count = torch.zeros(self.user_count)
            
            for i, (x, t, t_slot, s, y, y_t, y_t_slot, y_s, reset_h, y_timeintervals, y_distances, active_users) in enumerate(self.dataloader):
                active_users = active_users.squeeze()
                for j, reset in enumerate(reset_h):
                    if reset:
                        if self.setting.is_lstm:
                            hc = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                            h[0][0, j] = hc[0]
                            h[1][0, j] = hc[1]
                        else:
                            h[:, j] = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                        reset_count[active_users[j]] += 1
                        
                # squeeze for reasons of "loader-batch-size-is-1"
                x = x.squeeze().to(self.setting.device)
                t = t.squeeze().to(self.setting.device)
                t_slot = t_slot.squeeze().to(self.setting.device)
                s = s.squeeze().to(self.setting.device)

                y = y.squeeze()
                y_t = y_t.squeeze().to(self.setting.device)
                y_t_slot = y_t_slot.squeeze().to(self.setting.device)
                y_s = y_s.squeeze().to(self.setting.device)
                active_users = active_users.to(self.setting.device)         
                y_timeintervals = y_timeintervals.squeeze().to(self.setting.device)
                y_distances = y_distances.squeeze().to(self.setting.device)
                # evaluate:
                out = self.trainer.evaluate(x, t, t_slot, s, y_t, y_t_slot, y_s, y_timeintervals, y_distances, active_users)
                
                for j in range(self.setting.batch_size):  
                    # o contains a per user list of votes for all locations for each sequence entry
                    o = out[j]                 
                                        
                    # partition elements
                    o_n = o.cpu().detach().numpy()
                    ind = np.argpartition(o_n, -10, axis=1)[:, -10:] # top 10 elements
                                       
                    y_j = y[:, j]
                    
                    for k in range(len(y_j)):                    
                        if (reset_count[active_users[j]] > 1):
                            continue # skip already evaluated users.
                                                                                                            
                        # resort indices for k:
                        ind_k = ind[k]
                        r = ind_k[np.argsort(-o_n[k, ind_k], axis=0)] # sort top 10 elements descending
                                                    
                        r = torch.tensor(r)
                        t = y_j[k]
                        
                        # compute MRR:
                        r_kj = o_n[k, :]
                        t_val = r_kj[t]
                        upper = np.where(r_kj > t_val)[0]
                        precision = 1. / (1+len(upper))
                        
                        # compute NDCG:
                        dcg5 = 0.0
                        dcg10 = 0.0
                        for pos, item in enumerate(r[:10], 1):
                            rel = 1.0 if item == t else 0.0
                            if pos <= 5:
                                dcg5 += rel / math.log2(pos + 1)
                            dcg10 += rel / math.log2(pos + 1)

                        # IDCG for a single relevant item (rel=1 at position 1)
                        idcg5 = 1.0 / math.log2(2)
                        idcg10 = 1.0 / math.log2(2)
                        ndcg5_val = dcg5 / idcg5 if dcg5 > 0 else 0.0
                        ndcg10_val = dcg10 / idcg10 if dcg10 > 0 else 0.0
                        
                        # store
                        u_iter_cnt[active_users[j]] += 1
                        u_ACC5[active_users[j]] += t in r[:5]
                        u_ACC10[active_users[j]] += t in r[:10]
                        u_ndcg5[active_users[j]] += ndcg5_val
                        u_ndcg10[active_users[j]] += ndcg10_val
                        u_average_precision[active_users[j]] += precision                        

            formatter = "{0:.8f}"
            for j in range(self.user_count):
                iter_cnt += u_iter_cnt[j]
                ACC5 += u_ACC5[j]
                ACC10 += u_ACC10[j]
                ndcg5 += u_ndcg5[j]
                ndcg10 += u_ndcg10[j]
                average_precision += u_average_precision[j]

                if (self.setting.report_user > 0 and (j+1) % self.setting.report_user == 0):
                    log_string(self._log, f'Report user  {j}, preds: {u_iter_cnt[j]}, \
                          ACC@5: {formatter.format(u_ACC5[j]/u_iter_cnt[j] if u_iter_cnt[j] > 0 else 0.0)}, \
                          ACC@10: {formatter.format(u_ACC10[j]/u_iter_cnt[j] if u_iter_cnt[j] > 0 else 0.0)},\
                          NDCG@5: {formatter.format(u_ndcg5[j]/u_iter_cnt[j] if u_iter_cnt[j] > 0 else 0.0)}, \
                          NDCG@10: {formatter.format(u_ndcg10[j]/u_iter_cnt[j] if u_iter_cnt[j] > 0 else 0.0)}, \
                          MRR: {formatter.format(u_average_precision[j]/u_iter_cnt[j] if u_iter_cnt[j] > 0 else 0.0)}')
            
            
            log_string(self._log,f'ACC@5: {formatter.format(ACC5/iter_cnt)}')
            log_string(self._log,f'ACC@10: {formatter.format(ACC10/iter_cnt)}')
            log_string(self._log,f'NDCG@5: {formatter.format(ndcg5/iter_cnt)}')
            log_string(self._log,f'NDCG@10: {formatter.format(ndcg10/iter_cnt)}')
            log_string(self._log,f'MRR: {formatter.format(average_precision/iter_cnt)}')
            log_string(self._log,f'predictions: {iter_cnt}')