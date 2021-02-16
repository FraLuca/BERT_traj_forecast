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


# An example of line to run the script
#
# !CUDA_VISIBLE_DEVICES=0 python BERT_quant_with_goal.py --dataset_name eth --name eth --max_epoch 50 --batch_size 128 --num_clusters 1000 --data_type 2 --goal_type 2 --verbose 1


def main():
    parser=argparse.ArgumentParser(description='Train the individual quantized BERT model')
    parser.add_argument('--dataset_folder',type=str,default='datasets')
    parser.add_argument('--dataset_name',type=str,default='eth')
    parser.add_argument('--obs',type=int,default=8)
    parser.add_argument('--preds',type=int,default=12)
    parser.add_argument('--emb_size',type=int,default=1024)
    #parser.add_argument('--heads',type=int, default=8)
    #parser.add_argument('--layers',type=int,default=6)
    #parser.add_argument('--dropout',type=float,default=0.1)
    parser.add_argument('--cpu',action='store_true')
    #parser.add_argument('--output_folder',type=str,default='Output')
    #parser.add_argument('--val_size',type=int, default=50)
    #parser.add_argument('--gpu_device',type=str, default="0")
    parser.add_argument('--verbose', type=int, default=1)  # 0: False | 1: True
    parser.add_argument('--max_epoch',type=int, default=100)
    parser.add_argument('--batch_size',type=int,default=256)
    #parser.add_argument('--validation_epoch_start', type=int, default=30)
    #parser.add_argument('--resume_train',action='store_true')
    parser.add_argument('--delim',type=str,default='\t')
    #parser.add_argument('--name', type=str, default="eth_0.1")
    parser.add_argument('--factor', type=float, default=0.1)
    parser.add_argument('--warmup', type=int, default=1)
    #parser.add_argument('--save_step', type=int, default=1)
    parser.add_argument('--num_clusters', type=int, default=1000)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--data_type', type=int, default=2) # 0: Positions | 1: Speeds | 2: Relative Positions
    parser.add_argument('--goal_type', type=int, default=2) # 0: NO Goal | 1: True Goal | 2: Estimated Goal


    ##### INITIAL SETUP #####

    args=parser.parse_args()
    # model_name=args.name

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


    outdir=f'./results/Quantized'
    try:
        os.mkdir(outdir)
    except:
        pass


    outdir=f'./results/Quantized/'+str(args.num_clusters)+'_class'
    try:
        os.mkdir(outdir)
    except:
        pass


    dict_folder_data = {0:"Posit", 1:"Speed", 2:"RelPos"}
    dict_folder_goal = {0:"NoGoal", 1:"TrGoal", 2:"EsGoal"}
      

    outdir=f'./results/Quantized/'+str(args.num_clusters)+'_class/'+'BERT_'+dict_folder_data[args.data_type]+'_'+dict_folder_goal[args.goal_type]
    try:
        os.mkdir(outdir)
    except:
        pass


    # Set device where to run the model
    device=torch.device("cuda")
    if args.cpu or not torch.cuda.is_available():
        device=torch.device("cpu")




    ##### DATALOADER CREATION #####

    train_dataset,_ = baselineUtils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=True,verbose=args.verbose)
    val_dataset, _ = baselineUtils.create_dataset(args.dataset_folder, args.dataset_name, 0, args.obs, args.preds,delim=args.delim, train=False,verbose=args.verbose)
    test_dataset,_ =  baselineUtils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=False,eval=True,verbose=args.verbose)


    tr_dl=torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Here we load centroids of each cluster to classify each position, speed or realtive position
    mat = scipy.io.loadmat("./clusters/"+dict_folder_data[args.data_type]+"/clusters_"+args.dataset_name+"_"+str(args.num_clusters)+".mat")
    clusters = mat['centroids']



    ##### MODELS INITIALIZATION #####

    if args.goal_type == 2:
        goal_model = GoalEstimator()

    config = BertConfig(vocab_size=clusters.shape[0] + 1)
    mask_token = 1000
    gen = nn.Linear(config.hidden_size, clusters.shape[0]).to(device)
    model = BertModel(config).to(device)


    if args.goal_type == 2:
        optim = NoamOpt(args.emb_size, args.factor, len(tr_dl)*args.warmup, torch.optim.Adam(list(goal_model.parameters()) + list(model.parameters()) + list(gen.parameters()), lr=0, betas=(0.9, 0.98), eps=1e-9))
    else:
        optim = NoamOpt(args.emb_size, args.factor, len(tr_dl)*args.warmup, torch.optim.Adam(list(model.parameters()) + list(gen.parameters()), lr=0, betas=(0.9, 0.98), eps=1e-9))
         


    ##### DATA TYPE SELECTION AND NORMALIZATION #####

    idx1 = args.data_type*2
    idx2 = args.data_type*2+2



    ##### TRAIN ##### 

    epoch=0

    tr_loss_list = []
    val_loss_list = []

    mad_val = []
    fad_val = []
    mad_test = []
    fad_test = []


    while epoch < args.max_epoch:

        epoch_loss=0

        model.train()
        if args.goal_type == 2:
            goal_model.train()

        for id_b, batch in enumerate(tr_dl):

            optim.optimizer.zero_grad()

            # Scale factor for augmentation
            scale = np.random.uniform(0.75, 1.5)

            # Cast positions (x,y) into classes
            n_in_batch = batch['src'].shape[0]
            speeds_inp = batch['src'][:, :, idx1:idx2] * scale
            inp = torch.tensor(scipy.spatial.distance.cdist(speeds_inp.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,-1)).to(device)
            speeds_trg = batch['trg'][:, :, idx1:idx2] * scale
            target = torch.tensor(scipy.spatial.distance.cdist(speeds_trg.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,-1)).to(device)

            # Creation of mask tokens for target
            if args.goal_type != 0:
                trg_masked=torch.tensor([mask_token]).repeat(n_in_batch, args.preds-1).to(device)
            else:
                trg_masked=torch.tensor([mask_token]).repeat(n_in_batch, args.preds).to(device)

            net_input=torch.cat((inp,trg_masked),1)

            # Here we add the goal.
            # If we use ground truth one (--goal_type 1) then simply change last mask token with true goal.
            # Also in this case changing the mask embedding to 1.
            # Otherwise, if we want to estimate the goal (--goal_type 2), we use goal_model and we add to the input the best of K goal.
            if args.goal_type == 1:
                last_speed = target[:,-1:]
                net_input=torch.cat((net_input, last_speed),1).to(torch.long)

            elif args.goal_type == 2:
                pred_goal, kld_loss = goal_model(batch['src'][:,:,idx1:idx2], batch['trg'][:,:,idx1:idx2], training=True, K=args.K)
                error_goal = torch.sum((pred_goal - batch['trg'][:,-1:,idx1:idx2])**2, dim=-1)
                best_goal_idx = error_goal.min(dim=-1)
                best_pred_goal =  pred_goal[torch.arange(pred_goal.size(0)), best_goal_idx.indices, :].unsqueeze(1).detach().numpy()
                last_speed = torch.tensor(scipy.spatial.distance.cdist(best_pred_goal.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,-1)).to(device)
                net_input=torch.cat((net_input, last_speed),1).to(torch.long)


            # Postional Embedding
            position = torch.arange(0, net_input.shape[1]).repeat(inp.shape[0],1).long().to(device)
            # Sentence Embedding
            token = torch.zeros((inp.shape[0],net_input.shape[1])).long().to(device)
            # Attention Mask for possible padding
            attention_mask = torch.ones((inp.shape[0], net_input.shape[1])).long().to(device)

            # BERT + linear layer
            out=gen(model(input_ids=net_input, position_ids=position, token_type_ids=token, attention_mask=attention_mask)[0])


            # In debugging mode, check for Input, True and Predicted sequences
            if id_b==0 and args.verbose:
                print("INPUT")
                print(net_input[0, :])
                print("TRUE RELAT POSITION CLASS")
                print(torch.cat((inp, target),1)[0, :])
                print("PRED RELAT POSITION CLASS")
                print(F.softmax(out, dim=-1).argmax(dim=-1).cpu().numpy()[0, :])


            # Cross Entropy to train prediction of the sequence
            loss_traj = F.cross_entropy(out.view(-1, out.shape[-1]), torch.cat((inp, target), 1).view(-1), reduction='mean')


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

            gt = []
            pr = []
            val_loss=0

            for id_b, batch in enumerate(val_dl):

                n_in_batch = batch['src'].shape[0]
                speeds_inp = batch['src'][:, :, idx1:idx2]
                inp = torch.tensor(scipy.spatial.distance.cdist(speeds_inp.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,-1)).to(device)
                speeds_trg = batch['trg'][:, :, idx1:idx2]
                target = torch.tensor(scipy.spatial.distance.cdist(speeds_trg.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,-1)).to(device)

                if args.goal_type != 0:
                    trg_masked=torch.tensor([mask_token]).repeat(n_in_batch, args.preds-1).to(device)
                else:
                    trg_masked=torch.tensor([mask_token]).repeat(n_in_batch, args.preds).to(device)

                net_input=torch.cat((inp,trg_masked),1)


                if args.goal_type == 1:
                    last_speed = target[:,-1:]
                    net_input=torch.cat((net_input, last_speed),1).to(torch.long)

                elif args.goal_type == 2:
                    pred_goal, kld_loss = goal_model(batch['src'][:,:,idx1:idx2], batch['trg'][:,:,idx1:idx2], training=False, K=args.K)
                    error_goal = torch.sum((pred_goal - batch['trg'][:,-1:,idx1:idx2])**2, dim=-1)
                    best_goal_idx = error_goal.min(dim=-1)
                    best_pred_goal =  pred_goal[torch.arange(pred_goal.size(0)), best_goal_idx.indices, :].unsqueeze(1).detach().numpy()
                    last_speed = torch.tensor(scipy.spatial.distance.cdist(best_pred_goal.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,-1)).to(device)
                    net_input=torch.cat((net_input, last_speed),1).to(torch.long)



                position = torch.arange(0, net_input.shape[1]).repeat(inp.shape[0],1).long().to(device)
                token = torch.zeros((inp.shape[0],net_input.shape[1])).long().to(device)
                attention_mask = torch.ones((inp.shape[0], net_input.shape[1])).long().to(device)

                out=gen(model(input_ids=net_input, position_ids=position, token_type_ids=token, attention_mask=attention_mask)[0])



                loss_traj = F.cross_entropy(out.view(-1, out.shape[-1]), torch.cat((inp, target), 1).view(-1), reduction='mean')

                if args.goal_type != 2:
                    loss = loss_traj 
                elif args.goal_type == 2:
                    loss_goal = best_goal_idx.values.mean()
                    loss = loss_traj + loss_goal + kld_loss

                val_loss += loss.item()


                # Here we compute back the positions in (x,y). 
                # In this way we can look at the metrics of FAD and MAD.
                # So we save ground truth and predicted position to compare them.
                gt_b = batch['trg'][:, :, 0:2]
                gt.append(gt_b)

                # Depending on what data type we chose, we have to reconstruct the position properly
                if args.data_type == 0:
                    preds_tr_b = clusters[F.softmax(out, dim=-1).argmax(dim=-1).cpu().numpy()][:, -args.preds:]
                elif args.data_type == 1:
                    preds_tr_b = clusters[F.softmax(out, dim=-1).argmax(dim=-1).cpu().numpy()][:, -args.preds:].cumsum(1) + batch['src'][:,-1:,0:2].cpu().numpy()
                elif args.data_type == 2:
                    preds_tr_b = clusters[F.softmax(out, dim=-1).argmax(dim=-1).cpu().numpy()][:, -args.preds:] + batch['src'][:,:1,0:2].cpu().numpy()

                pr.append(preds_tr_b)


            gt = np.concatenate(gt, 0)
            pr = np.concatenate(pr, 0)
            mad, fad, errs = baselineUtils.distance_metrics(gt, pr)

            print("EVAL - LOSS:", val_loss/len(val_dl), "MAD:", mad, "FAD:", fad)

            val_loss_list.append(val_loss/len(val_dl))
            mad_val.append(mad)
            fad_val.append(fad)



            ##### TEST #####

            model.eval()

            if args.goal_type == 2:
                goal_model.eval()

            gt = []
            pr = []

            for id_b, batch in enumerate(test_dl):
                
                n_in_batch = batch['src'].shape[0]
                speeds_inp = batch['src'][:, :, idx1:idx2]
                inp = torch.tensor(scipy.spatial.distance.cdist(speeds_inp.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,-1)).to(device)
                speeds_trg = batch['trg'][:, :, idx1:idx2]
                target = torch.tensor(scipy.spatial.distance.cdist(speeds_trg.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,-1)).to(device)

                if args.goal_type != 0:
                    trg_masked=torch.tensor([mask_token]).repeat(n_in_batch, args.preds-1).to(device)
                else:
                    trg_masked=torch.tensor([mask_token]).repeat(n_in_batch, args.preds).to(device)

                net_input=torch.cat((inp,trg_masked),1) 

                if args.goal_type == 1:
                    last_speed = target[:,-1:]
                    net_input=torch.cat((net_input, last_speed),1).to(torch.long)

                elif args.goal_type == 2:
                    pred_goal, kld_loss = goal_model(batch['src'][:,:,idx1:idx2], batch['trg'][:,:,idx1:idx2], training=False, K=args.K)
                    error_goal = torch.sum((pred_goal - batch['trg'][:,-1:,idx1:idx2])**2, dim=-1)
                    best_goal_idx = error_goal.min(dim=-1)
                    best_pred_goal =  pred_goal[torch.arange(pred_goal.size(0)), best_goal_idx.indices, :].unsqueeze(1).detach().numpy()
                    last_speed = torch.tensor(scipy.spatial.distance.cdist(best_pred_goal.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,-1)).to(device)
                    net_input=torch.cat((net_input, last_speed),1).to(torch.long)


                position = torch.arange(0, net_input.shape[1]).repeat(inp.shape[0],1).long().to(device)
                token = torch.zeros((inp.shape[0],net_input.shape[1])).long().to(device)
                attention_mask = torch.ones((inp.shape[0], net_input.shape[1])).long().to(device)

                out=gen(model(input_ids=net_input,position_ids=position,token_type_ids=token,attention_mask=attention_mask)[0])

                
                gt_b = batch['trg'][:, :, 0:2]
                gt.append(gt_b)
                
                if args.data_type == 0:
                    preds_tr_b = clusters[F.softmax(out, dim=-1).argmax(dim=-1).cpu().numpy()][:, -args.preds:]
                elif args.data_type == 1:
                    preds_tr_b = clusters[F.softmax(out, dim=-1).argmax(dim=-1).cpu().numpy()][:, -args.preds:].cumsum(1) + batch['src'][:,-1:,0:2].cpu().numpy()
                elif args.data_type == 2:
                    preds_tr_b = clusters[F.softmax(out, dim=-1).argmax(dim=-1).cpu().numpy()][:, -args.preds:] + batch['src'][:,:1,0:2].cpu().numpy()


                # In debugging mode, check for prediction on an test sequence example. Both on classes and positions
                if id_b==0 and args.verbose:
                  print("TRUE RELAT POSITION CLASS")
                  print(torch.cat((inp, target),1)[0, :])
                  print("PRED RELAT POSITION CLASS")
                  print(F.softmax(out, dim=-1).argmax(dim=-1).cpu().numpy()[0, :])
                  print("TRUE TRG POSITION")
                  print(gt_b[0, :])
                  print("PRED TRG POSITION")
                  print(preds_tr_b[0, :])


                pr.append(preds_tr_b)


            gt = np.concatenate(gt, 0)
            pr = np.concatenate(pr, 0)
            mad, fad, errs = baselineUtils.distance_metrics(gt, pr)

            print("TEST - MAD:", mad, "FAD:", fad, "\n")

            mad_test.append(mad)
            fad_test.append(fad)


        epoch+=1


    # At the end we save in a .csv values for all losses and metrics to plot and analyze results
    df_results = pd.DataFrame({'tr_loss_list': tr_loss_list,
                       'val_loss_list': val_loss_list,
                        'mad_val': mad_val,
                        'fad_val': fad_val,
                        'mad_test': mad_test,
                        'fad_test': fad_test})
    
    save_folder = './results/Quantized/'+str(args.num_clusters)+'_class/'+'BERT_'+dict_folder_data[args.data_type]+'_'+dict_folder_goal[args.goal_type]+'/'
    file_name =  'quant_'+str(args.num_clusters)+'_'+args.dataset_name+'_data'+str(args.data_type)+'_goal'+str(args.goal_type)+'_Epoch'+str(args.max_epoch)+'.csv'
    df_results.to_csv(save_folder+file_name, index=False)

























if __name__=='__main__':
    main()
