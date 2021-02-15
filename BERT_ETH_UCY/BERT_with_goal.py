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


# !CUDA_VISIBLE_DEVICES=0 python BERT_with_goal.py --dataset_name eth --name eth --max_epoch 50 --batch_size 128 --data_type 1 --goal_type 0 --verbose 1


def main():
    parser=argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('--dataset_folder',type=str,default='datasets')
    parser.add_argument('--dataset_name',type=str,default='zara1')
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
    parser.add_argument('--name', type=str, default="zara1")
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--inter_point', nargs='+', type=int, default=None)
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
    


    device=torch.device("cuda")
    if args.cpu or not torch.cuda.is_available():
        device=torch.device("cpu")



    ## creation of the dataloaders for train and validation
    train_dataset,_ = baselineUtils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=True,verbose=args.verbose)
    val_dataset, _ = baselineUtils.create_dataset(args.dataset_folder, args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=False,verbose=args.verbose)
    test_dataset,_ =  baselineUtils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=False,eval=True,verbose=args.verbose)


    tr_dl=torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)



    if args.goal_type == 2:
        goal_model = GoalEstimator()



    config= BertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='relu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12)
    model = BertModel(config).to(device)

    a=NewEmbed(3, 768).to(device)
    model.set_input_embeddings(a)
    generator=GeneratorTS(768,2).to(device)

    
    if args.goal_type == 2:
        optim = NoamOpt(768, 0.1, len(tr_dl), torch.optim.Adam(list(goal_model.parameters())+list(a.parameters())+list(model.parameters())+list(generator.parameters()), lr=0, betas=(0.9, 0.98), eps=1e-9))
    else:
        optim = NoamOpt(768, 0.1, len(tr_dl), torch.optim.Adam(list(a.parameters())+list(model.parameters())+list(generator.parameters()), lr=0, betas=(0.9, 0.98), eps=1e-9))
                 






    epoch=0


    idx1 = args.data_type*2
    idx2 = args.data_type*2+2

    mean=train_dataset[:]['src'][:,:,idx1:idx2].mean((0,1))*0 # per ora non viene fatta nessuna normalizzazione
    std=train_dataset[:]['src'][:,:,idx1:idx2].std((0,1))*0+1
    
    mean_val=val_dataset[:]['src'][:,:,idx1:idx2].mean((0,1))*0 # per ora non viene fatta nessuna normalizzazione
    std_val=val_dataset[:]['src'][:,:,idx1:idx2].std((0,1))*0+1

    if args.inter_point:
      list_inter_point = np.array(args.inter_point) - args.obs
    else:
      list_inter_point = np.array([])
    
    print('Using Data mode:', str(args.data_type))
    print('Using Goal mode:', str(args.goal_type))
    print('Using Intermediate Positions:',' '.join(str(x) for x in list_inter_point))

    tr_loss_list = []
    val_loss_list = []

    mad_val = []
    fad_val = []
    mad_test = []
    fad_test = []


    traj_list = []

    while epoch<args.max_epoch:
        epoch_loss=0
        model.train()
        if args.goal_type == 2:
          goal_model.train()

        for id_b,batch in enumerate(tr_dl):

            optim.optimizer.zero_grad()
            r=0
            rot_mat = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])

            inp=((batch['src'][:,:,idx1:idx2]-mean)/std).to(device)

            if args.goal_type != 0:
                trg_masked=torch.zeros((inp.shape[0],args.preds-1,2)).to(device) - 1
            else:
                trg_masked=torch.zeros((inp.shape[0],args.preds,2)).to(device) - 1

            inp_cls=torch.ones(inp.shape[0],inp.shape[1],1).to(device)
            trg_cls= torch.zeros(trg_masked.shape[0], trg_masked.shape[1], 1).to(device)
            inp_cat=torch.cat((inp,trg_masked),1)
            cls_cat=torch.cat((inp_cls,trg_cls),1)
            net_input=torch.cat((inp_cat,cls_cat),2)

            inter_pos_cls = torch.ones(batch['trg'].shape[0], list_inter_point.shape[0], 1).to(device)
            inter_pos_new = batch['trg'][:, list_inter_point, idx1:idx2].to(device)
            net_input[:, list_inter_point+args.obs, :] = torch.cat((inter_pos_new, inter_pos_cls),2)


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

            

            position = torch.arange(0, net_input.shape[1]).repeat(inp.shape[0],1).long().to(device)
            token = torch.zeros((inp.shape[0],net_input.shape[1])).long().to(device)
            attention_mask = torch.ones((inp.shape[0], net_input.shape[1])).long().to(device)

            out=model(input_ids=net_input, position_ids=position, token_type_ids=token, attention_mask=attention_mask)

            pred=generator(out[0])

          

            if id_b==0 and args.verbose:
                print("INPUT")
                print(net_input[0,:,:])
                print("TRUE DATA")
                print(torch.cat((batch['src'][:,:,idx1:idx2],batch['trg'][:,:,idx1:idx2]),1)[0,:,:])
                print("PRED DATA")
                print(pred[0,:,:])

            

            loss_traj = F.pairwise_distance(pred[:,:].contiguous().view(-1, 2), torch.matmul(torch.cat((batch['src'][:,:,idx1:idx2],batch['trg'][:, :,idx1:idx2]),1).contiguous().view(-1, 2).to(device), torch.from_numpy(rot_mat).float().to(device)) ).mean()

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
        

        with torch.no_grad():
            model.eval()
            if args.goal_type == 2:
               goal_model.eval()

            gt=[]
            pr=[]
            val_loss=0

            for b_id, batch in enumerate(val_dl):
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

                '''if args.true_goal:
                    best_pred_goal = batch['trg'][:,-1:,idx1:idx2]
                else:
                    pred_goal, kld_loss = goal_model(batch['src'][:,:,idx1:idx2], batch['trg'][:,:,idx1:idx2], training=False, K=args.K)
                    error_goal = torch.sum((pred_goal - batch['trg'][:,-1:,idx1:idx2])**2, dim=-1)
                    best_goal_idx = error_goal.min(dim=-1)
                    best_pred_goal =  pred_goal[torch.arange(pred_goal.size(0)), best_goal_idx.indices, :].unsqueeze(1)


                last_speed=((best_pred_goal-mean_val)/std_val).to(device)
                last_one=torch.ones(last_speed.shape[0],last_speed.shape[1],1).to(device)
                last_speed=torch.cat((last_speed,last_one),2)
                net_input=torch.cat((net_input,last_speed),1)'''

                position = torch.arange(0, net_input.shape[1]).repeat(inp.shape[0], 1).long().to(device)
                token = torch.zeros((inp.shape[0], net_input.shape[1])).long().to(device)
                attention_mask = torch.zeros((inp.shape[0], net_input.shape[1])).long().to(device)

                out = model(input_ids=net_input, position_ids=position, token_type_ids=token, attention_mask=attention_mask)

                pred = generator(out[0])


                loss_traj = F.pairwise_distance(pred[:,:].contiguous().view(-1, 2), torch.matmul(torch.cat((batch['src'][:,:,idx1:idx2],batch['trg'][:, :,idx1:idx2]),1).contiguous().view(-1, 2).to(device), torch.from_numpy(rot_mat).float().to(device)) ).mean()

                if args.goal_type != 2:
                    loss = loss_traj 
                elif args.goal_type == 2:
                    loss_goal = best_goal_idx.values.mean()
                    loss = loss_traj + loss_goal + kld_loss

                val_loss += loss.item()

                gt_b=batch['trg'][:,:,0:2]

                if args.data_type == 0:
                    preds_tr_b=pred[:,args.obs:-1].to('cpu').detach() 
                elif args.data_type == 1:
                    preds_tr_b=pred[:,args.obs:-1].to('cpu').detach().cumsum(1).to('cpu').detach()+batch['src'][:,-1:,0:2]
                elif args.data_type == 2:
                    preds_tr_b=pred[:,args.obs:].to('cpu').detach()+batch['src'][:,:1,0:2]


                gt.append(gt_b)
                pr.append(preds_tr_b)

            gt=np.concatenate(gt,0)
            pr=np.concatenate(pr,0)
            mad,fad,errs=baselineUtils.distance_metrics(gt,pr)

            print("EVAL - LOSS:", val_loss/len(val_dl), "MAD:", mad, "FAD:", fad)

            val_loss_list.append(val_loss/len(val_dl))
            mad_val.append(mad)
            fad_val.append(fad)



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

                '''if args.true_goal:
                    best_pred_goal = batch['trg'][:,-1:,idx1:idx2]
                else:
                    pred_goal, kld_loss = goal_model(batch['src'][:,:,idx1:idx2], batch['trg'][:,:,idx1:idx2], training=False, K=args.K)
                    error_goal = torch.sum((pred_goal - batch['trg'][:,-1:,idx1:idx2])**2, dim=-1)
                    best_goal_idx = error_goal.min(dim=-1)
                    best_pred_goal =  pred_goal[torch.arange(pred_goal.size(0)), best_goal_idx.indices, :].unsqueeze(1)

                last_speed=((best_pred_goal-mean)/std).to(device)
                last_one=torch.ones(last_speed.shape[0],last_speed.shape[1],1).to(device)
                last_speed=torch.cat((last_speed,last_one),2)
                net_input=torch.cat((net_input,last_speed),1)'''

                position = torch.arange(0, net_input.shape[1]).repeat(inp.shape[0], 1).long().to(device)
                token = torch.zeros((inp.shape[0], net_input.shape[1])).long().to(device)
                attention_mask = torch.zeros((inp.shape[0], net_input.shape[1])).long().to(device)

                out = model(input_ids=net_input, position_ids=position, token_type_ids=token, attention_mask=attention_mask)

                pred = generator(out[0])

                gt_b=batch['trg'][:,:,0:2]

                if args.data_type == 0:
                    preds_tr_b=pred[:,args.obs:-1].to('cpu').detach() 
                elif args.data_type == 1:
                    preds_tr_b=pred[:,args.obs:-1].to('cpu').detach().cumsum(1).to('cpu').detach()+batch['src'][:,-1:,0:2]
                elif args.data_type == 2:
                    preds_tr_b=pred[:,args.obs:].to('cpu').detach()+batch['src'][:,:1,0:2]



                if id_b==0 and args.verbose:
                    print("TRUE TRG POSITION")
                    print(gt_b[0,:,:])
                    print("PRED TRG POSITION")
                    print(preds_tr_b[0,:,:])


                '''if id_b==0:

                    _inp=batch['src'][:,:,idx1:idx2]
                    #inp=torch.matmul(inp, torch.from_numpy(rot_mat).float().to(device))
                    _trg_masked=torch.zeros((_inp.shape[0],args.preds-1,2)) - 1
                    _inp_cls=torch.ones(_inp.shape[0],_inp.shape[1],1)
                    _trg_cls= torch.zeros(_trg_masked.shape[0], _trg_masked.shape[1], 1)
                    _inp_cat=torch.cat((_inp,_trg_masked),1)
                    _cls_cat=torch.cat((_inp_cls,_trg_cls),1)
                    _net_input=torch.cat((_inp_cat,_cls_cat),2)

                    _inter_pos_cls = torch.ones(batch['trg'].shape[0], list_inter_point.shape[0], 1)
                    _inter_pos_new = batch['trg'][:, list_inter_point, idx1:idx2]
                    _net_input[:, list_inter_point+args.obs, :] = torch.cat((_inter_pos_new, _inter_pos_cls),2)

                    _last_speed=best_pred_goal
                    _last_one=torch.ones(_last_speed.shape[0],_last_speed.shape[1],1)
                    _last_speed=torch.cat((_last_speed,_last_one),2)
                    _net_input=torch.cat((_net_input,_last_speed),1)

                    print("GOAL")
                    print(best_pred_goal[0,:,:])

                    print("INPUT")
                    print(_net_input[0,:,:])

                    print("OUTPUT")
                    print(pred[0,:,:])

                    print("TRUE POSITIONS")
                    print(torch.cat((batch['src'][:1,:,0:2],batch['trg'][:1,:,0:2]),1))

                    print("PRED POSITIONS")
                    _preds_tr_b = pred[:,:].to('cpu').detach()+batch['src'][:,:1,0:2]
                    print(_preds_tr_b[0,:,:])'''

                
                gt.append(gt_b)
                pr.append(preds_tr_b)

            gt=np.concatenate(gt,0)
            pr=np.concatenate(pr,0)
            mad,fad,errs=baselineUtils.distance_metrics(gt,pr)

            print("TEST - MAD:", mad, "FAD:", fad, "\n")

            mad_test.append(mad)
            fad_test.append(fad)


        epoch+=1

    #ab=1

    
    #torch.save(model.state_dict(), './models/BERT/model'+'_'+str(args.pos_sp)+'.pth')
    #torch.save(goal_model.state_dict(), './models/BERT/goal_model'+'_'+str(args.pos_sp)+'.pth')
    #torch.save(generator.state_dict(), './models/BERT/generator'+'_'+str(args.pos_sp)+'.pth')


    df_results = pd.DataFrame({'tr_loss_list': tr_loss_list,
                               'val_loss_list': val_loss_list,
                                'mad_val': mad_val,
                                'fad_val': fad_val,
                                'mad_test': mad_test,
                                'fad_test': fad_test})
    
    save_folder = './results/Regressive/'+'BERT_'+dict_folder_data[args.data_type]+'_'+dict_folder_goal[args.goal_type]+'/'
    file_name =  'regr_'+args.name+'_data'+str(args.data_type)+'_goal'+str(args.goal_type)+'_Epoch'+str(args.max_epoch)+'.csv'
    df_results.to_csv(save_folder+file_name, index=False)




























if __name__=='__main__':
    main()
