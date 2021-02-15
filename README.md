# BERT for Trajectory Forecasting

======

## Scripts and Folders


**BERT_with_goal.py** main script that train a Regressive BERT for trajectory prediction. 

Different inputs can be choosen: *--data_type 0* for Positions, *--data_type 1* for Speed and *--data_type 2* for Relative Positions.

Moreover, model could be train with different level of information on the goal: *--goal_type 0* for No Goal, *--goal_type 1* for True Goal and *--goal_type 2* for Estimated Goal.

An example of line to run it is:

CUDA_VISIBLE_DEVICES=0 python BERT_with_goal.py --dataset_name eth --name eth --max_epoch 50 --batch_size 128 --data_type 2 --goal_type 2 --verbose 0



**BERT_quant_with_goal.py** main script that train a Quantized BERT for trajectory prediction. 

Basically, that is the quantized version of previous script. 
Then data_type and goal_type can be switched with the same method.

An example of line to run it is:

CUDA_VISIBLE_DEVICES=0 python BERT_quant_with_goal.py --dataset_name eth --name eth --max_epoch 50 --batch_size 128 --num_clusters 1000 --data_type 2 --goal_type 2 --verbose 0


======


**cluster:** classes obtained with k-means algorithm. Those clusters needs to move to quantized/classification approach. Clusters are divided by number of classes (500, 1000) and data type (Speed, Relative Positions);

**dataset:** original ETH-UCY dataset;


**kmeans_pytorch:** files for k-means algo;


**transformer:** files for transformer and BERT;


