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

from numpy.random import seed
from numpy.random import randint


# !CUDA_VISIBLE_DEVICES=0 python BERT_quant_with_goal.py --dataset_name eth --name eth --max_epoch 50 --batch_size 128 --num_clusters 1000 --data_type 2 --goal_type 2 --verbose 1



def main():
    parser=argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('--dataset_folder',type=str,default='datasets')
    parser.add_argument('--dataset_name',type=str,default='eth')
    parser.add_argument('--obs',type=int,default=8)
    parser.add_argument('--preds',type=int,default=12)
    parser.add_argument('--emb_size',type=int,default=1024)
    parser.add_argument('--heads',type=int, default=8)
    parser.add_argument('--layers',type=int,default=6)
    parser.add_argument('--dropout',type=float,default=0.1)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--output_folder',type=str,default='Output')
    parser.add_argument('--val_size',type=int, default=50)
    parser.add_argument('--gpu_device',type=str, default="0")
    parser.add_argument('--verbose', type=int, default=1)  # 0: False | 1: True
    parser.add_argument('--max_epoch',type=int, default=100)
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--validation_epoch_start', type=int, default=30)
    parser.add_argument('--resume_train',action='store_true')
    parser.add_argument('--delim',type=str,default='\t')
    parser.add_argument('--name', type=str, default="eth_0.1")
    parser.add_argument('--factor', type=float, default=0.1)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--save_step', type=int, default=1)
    parser.add_argument('--num_clusters', type=int, default=1000)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--data_type', type=int, default=2) # 0: Positions | 1: Speeds | 2: Relative Positions
    parser.add_argument('--goal_type', type=int, default=2) # 0: NO Goal | 1: True Goal | 2: Estimated Goal



    args=parser.parse_args()
    model_name=args.name

    if args.verbose == 0:
      args.verbose = False
    elif args.verbose == 1:
      args.verbose = True


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



    device=torch.device("cuda")
    if args.cpu or not torch.cuda.is_available():
        device=torch.device("cpu")



    idx1 = args.data_type*2
    idx2 = args.data_type*2+2


    ## creation of the dataloaders for train and validation
    train_dataset,_ = baselineUtils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=True,verbose=args.verbose)
    val_dataset, _ = baselineUtils.create_dataset(args.dataset_folder, args.dataset_name, 0, args.obs, args.preds,delim=args.delim, train=False,verbose=args.verbose)
    test_dataset,_ =  baselineUtils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=False,eval=True,verbose=args.verbose)


    tr_dl=torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)



    epoch=0
    mat = scipy.io.loadmat("./clusters/"+dict_folder_data[args.data_type]+"/clusters_"+args.dataset_name+"_"+str(args.num_clusters)+".mat")
    clusters = mat['centroids']

    if args.goal_type == 2:
        goal_model = GoalEstimator()

    config = BertConfig(vocab_size=clusters.shape[0] + 1)
    mask_token = 1000
    gen = nn.Linear(config.hidden_size, 1).to(device)
    model = BertModel(config).to(device)


    if args.goal_type == 2:
        optim = NoamOpt(args.emb_size, args.factor, len(tr_dl)*args.warmup, torch.optim.Adam(list(goal_model.parameters()) + list(model.parameters()) + list(gen.parameters()), lr=0, betas=(0.9, 0.98), eps=1e-9))
    else:
        optim = NoamOpt(args.emb_size, args.factor, len(tr_dl)*args.warmup, torch.optim.Adam(list(model.parameters()) + list(gen.parameters()), lr=0, betas=(0.9, 0.98), eps=1e-9))
         



    tr_loss_list = []
    val_loss_list = []

    mad_val = []
    fad_val = []
    mad_test = []
    fad_test = []

    
    
    # Select random intermediate positions
    seed(123)

    if args.goal_type != 0:
        true_int_pos_tr = randint(0, args.preds, size=(len(tr_dl), args.batch_size))
        true_int_pos_val = randint(0, args.preds, size=(len(val_dl), args.batch_size))
        true_int_pos_test = randint(0, args.preds, size=(len(test_dl), args.batch_size))

        selected_int_pos_tr = randint(args.obs, args.obs+args.preds, size=(len(tr_dl), args.batch_size))
        selected_int_pos_val = randint(args.obs, args.obs+args.preds, size=(len(val_dl), args.batch_size))
        selected_int_pos_test = randint(args.obs, args.obs+args.preds, size=(len(test_dl), args.batch_size))
    else:
        true_int_pos_tr = randint(0, args.preds+1, size=(len(tr_dl), args.batch_size))
        true_int_pos_val = randint(0, args.preds+1, size=(len(val_dl), args.batch_size))
        true_int_pos_test = randint(0, args.preds+1, size=(len(test_dl), args.batch_size))

        selected_int_pos_tr = randint(args.obs, args.obs+args.preds+1, size=(len(tr_dl), args.batch_size))
        selected_int_pos_val = randint(args.obs, args.obs+args.preds+1, size=(len(val_dl), args.batch_size))
        selected_int_pos_test = randint(args.obs, args.obs+args.preds+1, size=(len(test_dl), args.batch_size))



    while epoch < args.max_epoch:
        
        model.train()
        if args.goal_type == 2:
            goal_model.train()

        epoch_loss=0
        abs_dist = 0
        denom_dist = 0
        metric_list_train = []

        for id_b, batch in enumerate(tr_dl):

            optim.optimizer.zero_grad()
            scale = np.random.uniform(0.75, 1.5)

            n_in_batch = batch['src'].shape[0]
            speeds_inp = batch['src'][:, :, idx1:idx2] * scale
            inp = torch.tensor(scipy.spatial.distance.cdist(speeds_inp.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,-1)).to(device)
            speeds_trg = batch['trg'][:, :, idx1:idx2] * scale
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
                pred_goal, kld_loss = goal_model(batch['src'][:,:,idx1:idx2], batch['trg'][:,:,idx1:idx2], training=True, K=args.K)
                error_goal = torch.sum((pred_goal - batch['trg'][:,-1:,idx1:idx2])**2, dim=-1)
                best_goal_idx = error_goal.min(dim=-1)
                best_pred_goal =  pred_goal[torch.arange(pred_goal.size(0)), best_goal_idx.indices, :].unsqueeze(1).detach().numpy()
                last_speed = torch.tensor(scipy.spatial.distance.cdist(best_pred_goal.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,-1)).to(device)
                net_input=torch.cat((net_input, last_speed),1).to(torch.long)

            true_posit = target[np.arange(target.shape[0]), true_int_pos_tr[id_b, :n_in_batch]]
            net_input[np.arange(net_input.shape[0]), selected_int_pos_tr[id_b, :n_in_batch]] = true_posit

            #true_label = np.zeros(net_input.shape)
            #true_label[np.arange(net_input.shape[0]), args.obs+true_int_pos_tr[id_b, :]] = 1
            true_label = torch.tensor(args.obs+true_int_pos_tr[id_b, :n_in_batch]).to(device)

            position = torch.arange(0, net_input.shape[1]).repeat(inp.shape[0],1).long().to(device)
            token = torch.zeros((inp.shape[0],net_input.shape[1])).long().to(device)
            attention_mask = torch.ones((inp.shape[0], net_input.shape[1])).long().to(device)

            out=gen(model(input_ids=net_input, position_ids=position, token_type_ids=token, attention_mask=attention_mask)[0]).squeeze(-1)


            if id_b==0 and args.verbose and False:
                print("INPUT")
                print(net_input[0, :])
                print("TRUE RELAT POSITION CLASS")
                print(torch.cat((inp, target),1)[0, :])
                print("PRED RELAT POSITION CLASS")
                print(F.softmax(out, dim=-1).argmax(dim=-1).cpu().numpy()[0])


            #print(out.shape, true_label.shape)
            #print(out.view(-1, out.shape[-1]).shape)
            #print((args.obs+true_int_pos_tr[id_b, :]))


            loss_traj = F.cross_entropy(out, true_label, reduction='mean')

            abs_dist += np.sum(np.abs(F.softmax(out, dim=-1).argmax(dim=-1).cpu().numpy() - true_label.cpu().numpy()))
            denom_dist += n_in_batch


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

        print("\nEPOCH:", epoch, " - TRAIN - LOSS:", epoch_loss/len(tr_dl), "METRIC:", abs_dist/denom_dist)
        tr_loss_list.append(epoch_loss/len(tr_dl))
        metric_list_train.append(abs_dist/denom_dist)
        


        with torch.no_grad():

            model.eval()
            if args.goal_type == 2:
                goal_model.eval()

            metric_list_val = []
            abs_dist = 0
            denom_dist = 0

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


                true_posit = target[np.arange(target.shape[0]), true_int_pos_val[id_b, :n_in_batch]]
                net_input[np.arange(net_input.shape[0]), selected_int_pos_val[id_b, :n_in_batch]] = true_posit

                true_label = torch.tensor(args.obs+true_int_pos_val[id_b, :n_in_batch]).to(device)



                position = torch.arange(0, net_input.shape[1]).repeat(inp.shape[0],1).long().to(device)
                token = torch.zeros((inp.shape[0],net_input.shape[1])).long().to(device)
                attention_mask = torch.ones((inp.shape[0], net_input.shape[1])).long().to(device)

                out=gen(model(input_ids=net_input, position_ids=position, token_type_ids=token, attention_mask=attention_mask)[0]).squeeze(-1)


                loss_traj = F.cross_entropy(out, true_label, reduction='mean')


                if args.goal_type != 2:
                    loss = loss_traj 
                elif args.goal_type == 2:
                    loss_goal = best_goal_idx.values.mean()
                    loss = loss_traj + loss_goal + kld_loss


                val_loss += loss.item()

                abs_dist += np.sum(np.abs(F.softmax(out, dim=-1).argmax(dim=-1).cpu().numpy() - true_label.cpu().numpy()))
                denom_dist += n_in_batch
                
                '''gt_b = batch['trg'][:, :, 0:2]
                gt.append(gt_b)

                if args.data_type == 0:
                    preds_tr_b = clusters[F.softmax(out, dim=-1).argmax(dim=-1).cpu().numpy()][:, -args.preds:]
                elif args.data_type == 1:
                    preds_tr_b = clusters[F.softmax(out, dim=-1).argmax(dim=-1).cpu().numpy()][:, -args.preds:].cumsum(1) + batch['src'][:,-1:,0:2].cpu().numpy()
                elif args.data_type == 2:
                    preds_tr_b = clusters[F.softmax(out, dim=-1).argmax(dim=-1).cpu().numpy()][:, -args.preds:] + batch['src'][:,:1,0:2].cpu().numpy()

                pr.append(preds_tr_b)'''

            '''gt = np.concatenate(gt, 0)
            pr = np.concatenate(pr, 0)
            mad, fad, errs = baselineUtils.distance_metrics(gt, pr)'''

            print("EVAL - LOSS:", val_loss/len(val_dl), "METRIC:", abs_dist/denom_dist)

            val_loss_list.append(val_loss/len(val_dl))
            metric_list_val.append(abs_dist/denom_dist)
            '''mad_val.append(mad)
            fad_val.append(fad)'''




            model.eval()
            if args.goal_type == 2:
                goal_model.eval()

            metric_list_test = []
            abs_dist = 0
            denom_dist = 0


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


                true_posit = target[np.arange(target.shape[0]), true_int_pos_test[id_b, :n_in_batch]]
                net_input[np.arange(net_input.shape[0]), selected_int_pos_test[id_b, :n_in_batch]] = true_posit

                true_label = torch.tensor(args.obs+true_int_pos_test[id_b, :n_in_batch]).to(device)


                position = torch.arange(0, net_input.shape[1]).repeat(inp.shape[0],1).long().to(device)
                token = torch.zeros((inp.shape[0],net_input.shape[1])).long().to(device)
                attention_mask = torch.ones((inp.shape[0], net_input.shape[1])).long().to(device)

                out=gen(model(input_ids=net_input,position_ids=position,token_type_ids=token,attention_mask=attention_mask)[0]).squeeze(-1)


                abs_dist += np.sum(np.abs(F.softmax(out, dim=-1).argmax(dim=-1).cpu().numpy() - true_label.cpu().numpy()))
                denom_dist += n_in_batch

                
                '''gt_b = batch['trg'][:, :, 0:2]
                gt.append(gt_b)
                
                if args.data_type == 0:
                    preds_tr_b = clusters[F.softmax(out, dim=-1).argmax(dim=-1).cpu().numpy()][:, -args.preds:]
                elif args.data_type == 1:
                    preds_tr_b = clusters[F.softmax(out, dim=-1).argmax(dim=-1).cpu().numpy()][:, -args.preds:].cumsum(1) + batch['src'][:,-1:,0:2].cpu().numpy()
                elif args.data_type == 2:
                    preds_tr_b = clusters[F.softmax(out, dim=-1).argmax(dim=-1).cpu().numpy()][:, -args.preds:] + batch['src'][:,:1,0:2].cpu().numpy()


                if b_id==0 and args.verbose:
                  print("TRUE RELAT POSITION CLASS")
                  print(torch.cat((inp, target),1)[0, :])
                  print("PRED RELAT POSITION CLASS")
                  print(F.softmax(out, dim=-1).argmax(dim=-1).cpu().numpy()[0, :])
                  print("TRUE TRG POSITION")
                  print(gt_b[0, :])
                  print("PRED TRG POSITION")
                  print(preds_tr_b[0, :])


                pr.append(preds_tr_b)'''


            '''gt = np.concatenate(gt, 0)
            pr = np.concatenate(pr, 0)
            mad, fad, errs = baselineUtils.distance_metrics(gt, pr)'''

            print("TEST - METRIC:", abs_dist/denom_dist, "\n")
            metric_list_test.append(abs_dist/denom_dist)
            '''mad_test.append(mad)
            fad_test.append(fad)'''


        epoch+=1

    #ab=1
    df_results = pd.DataFrame({'tr_loss_list': tr_loss_list,
                       'val_loss_list': val_loss_list,
                        'metric_train': metric_list_train,
                        'metric_val': metric_list_val,
                        'metric_test': metric_list_test})
    
    save_folder = './results/'
    file_name =  'quant_correct_checkpoint'+str(args.num_clusters)+'_'+args.name+'_data'+str(args.data_type)+'_goal'+str(args.goal_type)+'_Epoch'+str(args.max_epoch)+'.csv'
    df_results.to_csv(save_folder+file_name, index=False)

























if __name__=='__main__':
    main()
