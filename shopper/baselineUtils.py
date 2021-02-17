import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import datetime
import numpy as np
from tqdm.auto import tqdm
import argparse
from joblib import Parallel, delayed
import multiprocessing
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import random


class IndividualTfDataset(Dataset):
    def __init__(self,data):
        super(IndividualTfDataset,self).__init__()

        self.data=data

    def __len__(self):
        return self.data['src'].shape[0]


    def __getitem__(self,index):
        return {'src':torch.Tensor(self.data['src'][index]),
                'trg':torch.Tensor(self.data['trg'][index]),
                'tag_id':self.data['tag_id'][index],
                'seq_idx':self.data['seq_idx'][index],
                }

def create_dataset(full_df, n_obs=40, n_preds=60):
    
    data_pos=[]
    data_speed=[]
    data_rel_pos=[]
    
    info_tag_id=[]
    info_seq_idx=[]
    
    data = {}
    
    splitted_df = {n: full_df.iloc[n:n+100, :] 
           for n in range(0, len(full_df), 100)}

    info_tag_id = list(map(lambda kv: kv[1]['tag_id'].iloc[0], splitted_df.items()))
    info_seq_idx = list(map(lambda kv: kv[1]['seq_idx'].iloc[0], splitted_df.items()))


    data_pos = list(map(lambda kv: kv[1][['x','y']].values.astype(np.float32), splitted_df.items()))
    data_speed = list(map(lambda kv: kv[1][['x','y']].diff().values.astype(np.float32), splitted_df.items()))
    data_rel_pos = list(map(lambda kv: (kv[1][['x','y']] - kv[1][['x','y']].iloc[0,:]).values.astype(np.float32), splitted_df.items()))                
    
    data_pos_stack = np.stack(data_pos)
    data_speed_stack = np.stack(data_speed)
    data_rel_pos_stack = np.stack(data_rel_pos)
    info_tag_id_stack = np.stack(info_tag_id)
    info_seq_idx_stack = np.stack(info_seq_idx)
    
    all_data = np.concatenate((data_pos_stack, data_speed_stack, data_rel_pos_stack), 2)
    inp = all_data[:,:n_obs,:]
    out = all_data[:,n_obs:,:]
    
    data['src'] = inp
    data['trg'] = out
    data['tag_id'] = info_tag_id_stack
    data['seq_idx'] = info_seq_idx_stack
    
    return IndividualTfDataset(data)



def distance_metrics(gt,preds):
    errors = np.zeros(preds.shape[:-1])
    for i in range(errors.shape[0]):
        for j in range(errors.shape[1]):
            errors[i, j] = scipy.spatial.distance.euclidean(gt[i, j], preds[i, j])
    return errors.mean(),errors[:,-1].mean(),errors



if __name__=='__main__':

    parser=argparse.ArgumentParser(description='Preprocess the Shopper dataset')
    parser.add_argument('--max_number_files',type=int, default=None)  

    args=parser.parse_args()

	files = os.listdir('./preprocessed')
	full_df = pd.DataFrame(columns=['x', 'y', 'tag_id', 'seq_idx'])
	for f in files[:args.max_number_files]:
	    curr_df = pd.read_csv(os.path.join('./preprocessed', f))
	    full_df = pd.concat([full_df, curr_df])

	
	train_dataset = create_dataset(full_df)

