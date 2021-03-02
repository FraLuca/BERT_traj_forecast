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


def cast_time(x):
		return datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')

def compute_delta_time(x):
		date_1 = sub_df.iloc[0]['datetime']
		date_2 = x
		time_delta = (date_2 - date_1)
		return time_delta.total_seconds()



def extract_sequences(dataframe, threshold, min_seq_time_length, min_num_points):
    
    # threshold = posizioni più distanti in tempo di questa threshold spezzano la sequenza
    # min_seq_time_length = selezioniaimo sequenze t.c. il punto finale e iniziale distano almeno min_seq_time_length secondi 
    # min_num_points = selezioniaimo sequenze t.c. il numero di punti sia almeno min_num_points
    
    curr_seq = []
    seq_idx = 0

    new_seq_df = pd.DataFrame(columns = ['tag_id', 'time', 'x', 'y', 'description', 'datetime', 'deltatime', 'seq_idx'])

    
    for tag_id in tqdm(dataframe['tag_id'].unique()):
        
        sub_df = dataframe[dataframe['tag_id'] == tag_id]
        sub_df = sub_df.sort_values(by='time')
        sub_df['datetime'] = sub_df['time'].apply(cast_time)
        first = sub_df.iloc[0]['datetime']
        sub_df['deltatime'] = (sub_df['datetime'] - first).dt.total_seconds()
    
        for row in range(len(sub_df)-1):
            if (sub_df.iloc[row+1]['deltatime'] - sub_df.iloc[row]['deltatime']) <= threshold:
                if len(curr_seq) == 0:
                    curr_seq.append(sub_df.iloc[row])
                    curr_seq.append(sub_df.iloc[row+1])
                else:
                    curr_seq.append(sub_df.iloc[row+1])
            else:
                if len(curr_seq) >= min_num_points:
                    #list_of_seq.append(curr_seq)
                    out_df = pd.DataFrame(curr_seq)
                    if (out_df.iloc[-1]['deltatime'] - out_df.iloc[0]['deltatime']) >= min_seq_time_length:
                        out_df = out_df.drop_duplicates(subset=['datetime'])
                        out_df['seq_idx'] = [seq_idx]*len(out_df)
                        seq_idx += 1
                        new_seq_df = pd.concat([new_seq_df, out_df], ignore_index=True, sort=False)

                curr_seq = []
            
    return new_seq_df
    

def pad_missing_values(curr_seq):
		curr_seq['deltatime'] = curr_seq['deltatime'] - curr_seq['deltatime'].iloc[0]
		last_timestamp = int(curr_seq['deltatime'].iloc[-1])+1
		filled_seq = dict(zip(list(range(last_timestamp)), [float("NAN")]*last_timestamp))
		for row in range(len(curr_seq)):
				filled_seq[curr_seq['deltatime'].iloc[row]] = curr_seq[['x','y']].iloc[row]
		filled_df = pd.DataFrame.from_dict(filled_seq).T    
		return filled_df.astype(np.float32)


def sequence_creator(filled_curr_seq, tag_id, seq_idx, threshold=75, stride=5, len_seq=100):
		
		filled_df = filled_curr_seq.interpolate(method='ffill', axis=0)

		all_sequences = []
		accepted_seq = 0
		
		if len(filled_df)<threshold:
				return pd.DataFrame(columns=['x', 'y', 'tag_id', 'seq_idx'])
		elif len(filled_df) <= len_seq:
				# Pad at the begginning and add to list
				df_nan = pd.DataFrame(np.nan, index=np.arange(100-(len(filled_df)+1)), columns=['x', 'y'])
				if len(filled_df) != len_seq:
						new_df = pd.concat([filled_df.head(1), df_nan, filled_df])
				else: 
						new_df = filled_df
				# Add seq_idx
				new_seq_idx = str(seq_idx)+'_'+str(accepted_seq)
				accepted_seq += 1
				new_df['seq_idx'] = np.repeat([new_seq_idx], len_seq)

				# Add tag_id
				new_df['tag_id'] = np.repeat([tag_id], len_seq)

				all_sequences.append(new_df)
		elif len(filled_df) > len_seq:
				for i in range(1+(len(filled_df) - len_seq) // stride):
						new_df = filled_df.iloc[i*stride : i*stride+len_seq, [0,1]].reset_index(drop=True)

						# Add seq_idx
						new_seq_idx = str(seq_idx)+'_'+str(accepted_seq)
						accepted_seq += 1
						try:
								new_df['seq_idx'] = np.repeat([new_seq_idx], len_seq)
						except:
								print(len(new_seq_idx), len_seq)
						# Add tag_id
						new_df['tag_id'] = np.repeat([tag_id], len_seq)

						all_sequences.append(new_df)
		if len(all_sequences) > 0:
				return pd.concat(all_sequences).reset_index(drop=True)
		else:
				return pd.DataFrame(columns=['x', 'y', 'tag_id', 'seq_idx'])


def general_function(sub_list):
		
		sssseq = []
		splitted_df = {seq_idx: seq_df[seq_df['seq_idx']==seq_idx] for seq_idx in sub_list}

		prova = list(map(lambda kv: sequence_creator(pad_missing_values(kv[1]), kv[1]['tag_id'].iloc[0], kv[1]['seq_idx'].iloc[0]), splitted_df.items()))
		sssseq += prova
		final_df = pd.concat(sssseq).reset_index(drop=True)
		final_df.to_csv("./preprocessed/sequences_from_"+str(sub_list[0])+"_to_"+str(sub_list[-1])+".csv")




if __name__=='__main__':

		parser=argparse.ArgumentParser(description='Preprocess the Shopper dataset')
		parser.add_argument('--dataset_file',type=str)  # edeka, aldi, globus, rewe 
		parser.add_argument('--dataset_folder',type=str, default='raw_trajectories') 
		parser.add_argument('--preprocess_part',type=int, default=2)  

		args=parser.parse_args()

		outdir=f'preprocessed'
		try:
				os.mkdir(outdir)
		except:
				pass

		if args.preprocess_part == 1:

			df = pd.read_csv('./'+args.dataset_folder+'/'+args.dataset_file+'_dataset.txt', sep=';')
			seq_df = extract_sequences(dataset=df, threshold=120, min_seq_time_length=10, min_num_points=5)
			seq_df.to_csv('./'+args.dataset_folder+'/'+args.dataset_file+'_extracted_sequences.csv', index=False) 
	
		else:

			seq_df = pd.read_csv('./'+args.dataset_folder+'/'+args.dataset_file+'_extracted_sequences.csv')
		
			len_processed_seq = 200
			tot_seq = len(seq_df['seq_idx'].unique()) # //5 # Per ora ne processiamo un quinto
			inputs = [np.arange(i*len_processed_seq, min((i+1)*len_processed_seq, tot_seq)) for i in range(tot_seq//len_processed_seq + 1)]


			num_cores = multiprocessing.cpu_count() - 2
			results = Parallel(n_jobs=num_cores)(delayed(general_function)(i) for i in inputs)