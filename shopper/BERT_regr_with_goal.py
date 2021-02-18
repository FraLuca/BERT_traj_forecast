import argparse
import baselineUtils
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from transformer.batch import subsequent_mask
from torch.optim import Adam,SGD,RMSprop,Adagrad
from transformer.noam_opt import NoamOpt
import numpy as np
import scipy.io
import json
import pickle
import pandas as pd


from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig, AdamW
from individual_TF import LinearEmbedding as NewEmbed,Generator as GeneratorTS

from goal_estimator import GoalEstimator

from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split

# An example of line to run the script
#
# !CUDA_VISIBLE_DEVICES=0 python BERT_with_goal.py --dataset_name eth --name eth --max_epoch 50 --batch_size 128 --data_type 1 --goal_type 0 --verbose 1


def main():
    parser=argparse.ArgumentParser(description='Train the individual regressive BERT model')
    parser.add_argument('--obs',type=int,default=8)
    parser.add_argument('--preds',type=int,default=12)
    parser.add_argument('--emb_size',type=int,default=768)
    #parser.add_argument('--heads',type=int, default=8)
    #parser.add_argument('--layers',type=int,default=6)
    #parser.add_argument('--dropout',type=float,default=0.1)
    parser.add_argument('--cpu', action='store_true')
    #parser.add_argument('--output_folder',type=str,default='Output')
    #parser.add_argument('--val_size',type=int, default=50)
    #parser.add_argument('--gpu_device',type=str, default="0")
    parser.add_argument('--verbose', type=int, default=1)  # 0: False | 1: True
    parser.add_argument('--max_epoch',type=int, default=100)
    parser.add_argument('--batch_size',type=int,default=256)
    #parser.add_argument('--validation_epoch_start', type=int, default=30)
    #parser.add_argument('--resume_train',action='store_true')
    parser.add_argument('--delim',type=str,default='\t')
    #parser.add_argument('--name', type=str, default="zara1")
    parser.add_argument('--factor', type=float, default=0.1)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--inter_point', nargs='+', type=int, default=None)
    parser.add_argument('--data_type', type=int, default=2) # 0: Positions | 1: Speeds | 2: Relative Positions
    parser.add_argument('--goal_type', type=int, default=2) # 0: NO Goal | 1: True Goal | 2: Estimated Goal
    parser.add_argument('--max_number_files',type=int, default=None)  

    
    ##### INITIAL SETUP #####

    args=parser.parse_args()

    if args.verbose == 0:
        args.verbose = False
    elif args.verbose == 1:
        args.verbose = True


    # Create folders to save results 
    outdir=f'results'
    try:
        os.mkdir(outdir)
    except:
        pass


    outdir=f'./results/Regressive'
    try:
        os.mkdir(outdir)
    except:
        pass


    dict_folder_data = {0:"Posit", 1:"Speed", 2:"RelPos"}
    dict_folder_goal = {0:"NoGoal", 1:"TrGoal", 2:"EsGoal"}

    outdir=f'./results/Regressive/'+'BERT_'+dict_folder_data[args.data_type]+'_'+dict_folder_goal[args.goal_type]
    try:
        os.mkdir(outdir)
    except:
        pass
    

    # Set device where to run the model
    device=torch.device("cuda")
    if args.cpu or not torch.cuda.is_available():
        device=torch.device("cpu")

    files = os.listdir('./preprocessed')
    full_df = pd.DataFrame(columns=['x', 'y', 'tag_id', 'seq_idx'])
    for f in files[:args.max_number_files]:
        curr_df = pd.read_csv(os.path.join('./preprocessed', f))
        full_df = pd.concat([full_df, curr_df])

    train, test = train_test_split(full_df['seq_idx'].unique(), test_size = 0.3, random_state = 0)
    test, val = train_test_split(test, test_size = 0.5, random_state = 0)

    ##### DATALOADER CREATION #####

    train_dataset = baselineUtils.create_dataset(full_df[full_df['seq_idx'].isin(train)], args.obs, args.preds)
    val_dataset = baselineUtils.create_dataset(full_df[full_df['seq_idx'].isin(val)], args.obs, args.preds)
    test_dataset = baselineUtils.create_dataset(full_df[full_df['seq_idx'].isin(test)], args.obs, args.preds)

    tr_dl=torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)



    ##### MODELS INITIALIZATION #####

    if args.goal_type == 2:
        goal_model = GoalEstimator()


    config= BertConfig(vocab_size=30522, hidden_size=args.emb_size, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='relu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12)
    model = BertModel(config).to(device)

    a=NewEmbed(3, args.emb_size).to(device)
    model.set_input_embeddings(a)
    generator=GeneratorTS(config.hidden_size, 2).to(device)

    
    if args.goal_type == 2:
        optim = NoamOpt(args.emb_size, args.factor, len(tr_dl)*args.warmup, torch.optim.Adam(list(goal_model.parameters())+list(model.parameters())+list(generator.parameters()), lr=0, betas=(0.9, 0.98), eps=1e-9))
    else:
        optim = NoamOpt(args.emb_size, args.factor, len(tr_dl)*args.warmup, torch.optim.Adam(list(model.parameters())+list(generator.parameters()), lr=0, betas=(0.9, 0.98), eps=1e-9))
                 


    ##### DATA TYPE SELECTION AND NORMALIZATION #####

    idx1 = args.data_type*2
    idx2 = args.data_type*2+2

    mean=train_dataset[:]['src'][:,:,idx1:idx2].mean((0,1))*0 # per ora non viene fatta nessuna normalizzazione
    std=train_dataset[:]['src'][:,:,idx1:idx2].std((0,1))*0+1
    
    mean_val=val_dataset[:]['src'][:,:,idx1:idx2].mean((0,1))*0 # per ora non viene fatta nessuna normalizzazione
    std_val=val_dataset[:]['src'][:,:,idx1:idx2].std((0,1))*0+1

    
    ##### INITIALIZATION OF INTERMEDIATE POSITIONS #####

    if args.inter_point:
      list_inter_point = np.array(args.inter_point) - args.obs
    else:
      list_inter_point = np.array([])


    
    print()
    print('Using Data mode:', str(args.data_type))
    print('Using Goal mode:', str(args.goal_type))
    print('Using Intermediate Positions:',' '.join(str(x) for x in list_inter_point))
    print()



    ##### TRAIN #####

    epoch=0

    tr_loss_list = []
    val_loss_list = []

    mad_val = []
    fad_val = []
    mad_test = []
    fad_test = []


    while epoch<args.max_epoch:

        epoch_loss=0

        model.train()
        if args.goal_type == 2:
          goal_model.train()

        for id_b, batch in enumerate(tr_dl):

            optim.optimizer.zero_grad()

            inp=((batch['src'][:,:,idx1:idx2]-mean)/std).to(device)

            # Creation of mask tokens for target
            if args.goal_type != 0:
                trg_masked=torch.zeros((inp.shape[0],args.preds-1,2)).to(device) - 1
            else:
                trg_masked=torch.zeros((inp.shape[0],args.preds,2)).to(device) - 1

            # For the regressive approach, is needed a mask/embedding of what should be predicted (=0) and what is given (=1).
            # So we add a new column (x, y, mask) to help model distinguishing these cases.
            inp_cls=torch.ones(inp.shape[0],inp.shape[1],1).to(device)
            trg_cls= torch.zeros(trg_masked.shape[0], trg_masked.shape[1], 1).to(device)
            inp_cat=torch.cat((inp,trg_masked),1)
            cls_cat=torch.cat((inp_cls,trg_cls),1)
            net_input=torch.cat((inp_cat,cls_cat),2)

            # If intermediate positions were specified then we add to the input, swapping also the relative mask embedding from 0 to 1
            inter_pos_cls = torch.ones(batch['trg'].shape[0], list_inter_point.shape[0], 1).to(device)
            inter_pos_new = batch['trg'][:, list_inter_point, idx1:idx2].to(device)
            net_input[:, list_inter_point+args.obs, :] = torch.cat((inter_pos_new, inter_pos_cls),2)

            # Here we add the goal.
            # If we use ground truth one (--goal_type 1) then simply change last mask token with true goal.
            # Also in this case changing the mask embedding to 1.
            # Otherwise, if we want to estimate the goal (--goal_type 2), we use goal_model and we add to the input the best of K goal.
            if args.goal_type == 1:
                best_pred_goal = batch['trg'][:,-1:,idx1:idx2]

                last_speed=((best_pred_goal-mean)/std).to(device)
                last_one=torch.ones(last_speed.shape[0],last_speed.shape[1],1).to(device)
                last_speed=torch.cat((last_speed,last_one),2)
                net_input=torch.cat((net_input,last_speed),1)

            elif args.goal_type == 2:
                pred_goal, kld_loss = goal_model(batch['src'][:,:,idx1:idx2], batch['trg'][:,:,idx1:idx2], training=True, K=args.K)
                error_goal = torch.sum((pred_goal - batch['trg'][:,-1:,idx1:idx2])**2, dim=-1)
                best_goal_idx = error_goal.min(dim=-1)
                best_pred_goal =  pred_goal[torch.arange(pred_goal.size(0)), best_goal_idx.indices, :].unsqueeze(1)

                last_speed=((best_pred_goal-mean)/std).to(device)
                last_one=torch.ones(last_speed.shape[0],last_speed.shape[1],1).to(device)
                last_speed=torch.cat((last_speed,last_one),2)
                net_input=torch.cat((net_input,last_speed),1)

            net_input = torch.Tensor(np.nan_to_num(net_input.numpy(), 0))

            # Postional Embedding
            position = torch.arange(0, net_input.shape[1]).repeat(inp.shape[0],1).long().to(device)
            # Sentence Embedding
            token = torch.zeros((inp.shape[0],net_input.shape[1])).long().to(device)
            # Attention Mask for possible padding
            attention_mask = (~torch.isnan(torch.cat((batch['src'][:,:,idx1:idx2], batch['trg'][:,:,idx1:idx2]), dim=1))*1).long().to(device)

            # BERT
            out=model(input_ids=net_input, position_ids=position, token_type_ids=token, attention_mask=attention_mask)
            # Out linear layer
            pred=generator(out[0])

          
            # In debugging mode, check for Input, True and Predicted sequences
            if id_b==0 and args.verbose:
                print("INPUT")
                print(net_input[0,:,:])
                print("TRUE DATA")
                print(torch.cat((batch['src'][:,:,idx1:idx2],batch['trg'][:,:,idx1:idx2]),1)[0,:,:])
                print("PRED DATA")
                print(pred[0,:,:])

            
            # MSE to train prediction of the sequence
            loss_traj = F.pairwise_distance(pred[:,:].contiguous().view(-1, 2), torch.matmul(torch.cat((batch['src'][:,:,idx1:idx2],batch['trg'][:, :,idx1:idx2]),1).contiguous().view(-1, 2).to(device), torch.from_numpy(rot_mat).float().to(device)) ).mean()

            # In case we estimate goal, we add loss_goal (MSE on last position) and kld_loss (to bring closer train and test goal distribution)
            if args.goal_type != 2:
                loss = loss_traj 
            elif args.goal_type == 2:
                loss_goal = best_goal_idx.values.mean()
                loss = loss_traj + loss_goal + kld_loss

            loss.backward()

            optim.step()

            if args.verbose:
                print("epoch %03i/%03i  frame %04i / %04i loss: %7.4f" % (epoch, args.max_epoch, id_b, len(tr_dl), loss.item()))

            epoch_loss += loss.item()

        print("\nEPOCH:", epoch, " - TRAIN - LOSS:", epoch_loss/len(tr_dl))

        tr_loss_list.append(epoch_loss/len(tr_dl))



        ##### VALIDATION #####

        with torch.no_grad():

            model.eval()

            if args.goal_type == 2:
               goal_model.eval()

            gt=[]
            pr=[]
            val_loss=0

            for id_b, batch in enumerate(val_dl):
                inp = ((batch['src'][:,:,idx1:idx2]-mean_val)/std_val).to(device)

                if args.goal_type != 0:
                    trg_masked=torch.zeros((inp.shape[0],args.preds-1,2)).to(device) - 1
                else:
                    trg_masked=torch.zeros((inp.shape[0],args.preds,2)).to(device) - 1

                inp_cls = torch.ones(inp.shape[0], inp.shape[1], 1).to(device)
                trg_cls = torch.zeros(trg_masked.shape[0], trg_masked.shape[1], 1).to(device)
                inp_cat = torch.cat((inp, trg_masked), 1)
                cls_cat = torch.cat((inp_cls, trg_cls), 1)
                net_input = torch.cat((inp_cat, cls_cat), 2)

                inter_pos_cls = torch.ones(batch['trg'].shape[0], list_inter_point.shape[0], 1).to(device)
                inter_pos_new = batch['trg'][:, list_inter_point, idx1:idx2].to(device)
                net_input[:, list_inter_point+args.obs, :] = torch.cat((inter_pos_new, inter_pos_cls),2)


                if args.goal_type == 1:
                    best_pred_goal = batch['trg'][:,-1:,idx1:idx2]

                    last_speed=((best_pred_goal-mean_val)/std_val).to(device)
                    last_one=torch.ones(last_speed.shape[0],last_speed.shape[1],1).to(device)
                    last_speed=torch.cat((last_speed,last_one),2)
                    net_input=torch.cat((net_input,last_speed),1)

                elif args.goal_type == 2:
                    pred_goal, kld_loss = goal_model(batch['src'][:,:,idx1:idx2], batch['trg'][:,:,idx1:idx2], training=False, K=args.K)
                    error_goal = torch.sum((pred_goal - batch['trg'][:,-1:,idx1:idx2])**2, dim=-1)
                    best_goal_idx = error_goal.min(dim=-1)
                    best_pred_goal =  pred_goal[torch.arange(pred_goal.size(0)), best_goal_idx.indices, :].unsqueeze(1)

                    last_speed=((best_pred_goal-mean_val)/std_val).to(device)
                    last_one=torch.ones(last_speed.shape[0],last_speed.shape[1],1).to(device)
                    last_speed=torch.cat((last_speed,last_one),2)
                    net_input=torch.cat((net_input,last_speed),1)

                net_input = torch.Tensor(np.nan_to_num(net_input.numpy(), 0))

                position = torch.arange(0, net_input.shape[1]).repeat(inp.shape[0], 1).long().to(device)
                token = torch.zeros((inp.shape[0], net_input.shape[1])).long().to(device)
                # attention_mask = torch.ones((inp.shape[0], net_input.shape[1])).long().to(device)
                attention_mask = (~torch.isnan(torch.cat((batch['src'][:,:,idx1:idx2], batch['trg'][:,:,idx1:idx2]), dim=1))*1).long().to(device)

                out = model(input_ids=net_input, position_ids=position, token_type_ids=token, attention_mask=attention_mask)

                pred = generator(out[0])


                loss_traj = F.pairwise_distance(pred[:,:].contiguous().view(-1, 2), torch.matmul(torch.cat((batch['src'][:,:,idx1:idx2],batch['trg'][:, :,idx1:idx2]),1).contiguous().view(-1, 2).to(device), torch.from_numpy(rot_mat).float().to(device)) ).mean()

                if args.goal_type != 2:
                    loss = loss_traj 
                elif args.goal_type == 2:
                    loss_goal = best_goal_idx.values.mean()
                    loss = loss_traj + loss_goal + kld_loss

                val_loss += loss.item()


                # Here we compute back the positions in (x,y). 
                # In this way we can look at the metrics of FAD and MAD.
                # So we save ground truth and predicted position to compare them.
                gt_b=batch['trg'][:,:,0:2]
                gt.append(gt_b)

                # Depending on what data type we chose, we have to reconstruct the position properly
                if args.data_type == 0:
                    preds_tr_b=pred[:,args.obs:-1].to('cpu').detach() 
                elif args.data_type == 1:
                    preds_tr_b=pred[:,args.obs:-1].to('cpu').detach().cumsum(1).to('cpu').detach()+batch['src'][:,-1:,0:2]
                elif args.data_type == 2:
                    preds_tr_b=pred[:,args.obs:].to('cpu').detach()+batch['src'][:,:1,0:2]

                pr.append(preds_tr_b)


            gt=np.concatenate(gt,0)
            pr=np.concatenate(pr,0)
            mad,fad,errs=baselineUtils.distance_metrics(gt,pr)

            print("EVAL - LOSS:", val_loss/len(val_dl), "MAD:", mad, "FAD:", fad)

            val_loss_list.append(val_loss/len(val_dl))
            mad_val.append(mad)
            fad_val.append(fad)



            ##### TEST #####

            model.eval()
            
            if args.goal_type == 2:
               goal_model.eval()

            gt=[]
            pr=[]

            for id_b, batch in enumerate(test_dl):
                inp = ((batch['src'][:,:,idx1:idx2]-mean)/std).to(device)

                if args.goal_type != 0:
                    trg_masked=torch.zeros((inp.shape[0],args.preds-1,2)).to(device) - 1
                else:
                    trg_masked=torch.zeros((inp.shape[0],args.preds,2)).to(device) - 1

                inp_cls = torch.ones(inp.shape[0], inp.shape[1], 1).to(device)
                trg_cls = torch.zeros(trg_masked.shape[0], trg_masked.shape[1], 1).to(device)
                inp_cat = torch.cat((inp, trg_masked), 1)
                cls_cat = torch.cat((inp_cls, trg_cls), 1)
                net_input = torch.cat((inp_cat, cls_cat), 2)

                inter_pos_cls = torch.ones(batch['trg'].shape[0], list_inter_point.shape[0], 1).to(device)
                inter_pos_new = batch['trg'][:, list_inter_point, idx1:idx2].to(device)
                net_input[:, list_inter_point+args.obs, :] = torch.cat((inter_pos_new, inter_pos_cls),2)


                if args.goal_type == 1:
                    best_pred_goal = batch['trg'][:,-1:,idx1:idx2]

                    last_speed=((best_pred_goal-mean_val)/std_val).to(device)
                    last_one=torch.ones(last_speed.shape[0],last_speed.shape[1],1).to(device)
                    last_speed=torch.cat((last_speed,last_one),2)
                    net_input=torch.cat((net_input,last_speed),1)

                elif args.goal_type == 2:
                    pred_goal, kld_loss = goal_model(batch['src'][:,:,idx1:idx2], batch['trg'][:,:,idx1:idx2], training=False, K=args.K)
                    error_goal = torch.sum((pred_goal - batch['trg'][:,-1:,idx1:idx2])**2, dim=-1)
                    best_goal_idx = error_goal.min(dim=-1)
                    best_pred_goal =  pred_goal[torch.arange(pred_goal.size(0)), best_goal_idx.indices, :].unsqueeze(1)

                    last_speed=((best_pred_goal-mean_val)/std_val).to(device)
                    last_one=torch.ones(last_speed.shape[0],last_speed.shape[1],1).to(device)
                    last_speed=torch.cat((last_speed,last_one),2)
                    net_input=torch.cat((net_input,last_speed),1)

                net_input = torch.Tensor(np.nan_to_num(net_input.numpy(), 0))

                position = torch.arange(0, net_input.shape[1]).repeat(inp.shape[0], 1).long().to(device)
                token = torch.zeros((inp.shape[0], net_input.shape[1])).long().to(device)
                # attention_mask = torch.ones((inp.shape[0], net_input.shape[1])).long().to(device)
                attention_mask = (~torch.isnan(torch.cat((batch['src'][:,:,idx1:idx2], batch['trg'][:,:,idx1:idx2]), dim=1))*1).long().to(device)

                out = model(input_ids=net_input, position_ids=position, token_type_ids=token, attention_mask=attention_mask)

                pred = generator(out[0])



                gt_b=batch['trg'][:,:,0:2]

                if args.data_type == 0:
                    preds_tr_b=pred[:,args.obs:-1].to('cpu').detach() 
                elif args.data_type == 1:
                    preds_tr_b=pred[:,args.obs:-1].to('cpu').detach().cumsum(1).to('cpu').detach()+batch['src'][:,-1:,0:2]
                elif args.data_type == 2:
                    preds_tr_b=pred[:,args.obs:].to('cpu').detach()+batch['src'][:,:1,0:2]

                # In debugging mode, check for prediction on an test sequence example
                if id_b==0 and args.verbose:
                    print("TRUE TRG POSITION")
                    print(gt_b[0,:,:])
                    print("PRED TRG POSITION")
                    print(preds_tr_b[0,:,:])

                
                gt.append(gt_b)
                pr.append(preds_tr_b)

            gt=np.concatenate(gt,0)
            pr=np.concatenate(pr,0)
            mad,fad,errs=baselineUtils.distance_metrics(gt,pr)

            print("TEST - MAD:", mad, "FAD:", fad, "\n")

            mad_test.append(mad)
            fad_test.append(fad)


        epoch+=1

    
    #torch.save(model.state_dict(), './models/BERT/model'+'_'+str(args.pos_sp)+'.pth')
    #torch.save(goal_model.state_dict(), './models/BERT/goal_model'+'_'+str(args.pos_sp)+'.pth')
    #torch.save(generator.state_dict(), './models/BERT/generator'+'_'+str(args.pos_sp)+'.pth')


    # At the end we save in a .csv values for all losses and metrics to plot and analyze results
    df_results = pd.DataFrame({'tr_loss_list': tr_loss_list,
                               'val_loss_list': val_loss_list,
                                'mad_val': mad_val,
                                'fad_val': fad_val,
                                'mad_test': mad_test,
                                'fad_test': fad_test})
    
    save_folder = './results/Regressive/'+'BERT_'+dict_folder_data[args.data_type]+'_'+dict_folder_goal[args.goal_type]+'/'
    file_name =  'regr_shopper_data'+str(args.data_type)+'_goal'+str(args.goal_type)+'_Epoch'+str(args.max_epoch)+'.csv'
    df_results.to_csv(save_folder+file_name, index=False)




























if __name__=='__main__':
    main()
