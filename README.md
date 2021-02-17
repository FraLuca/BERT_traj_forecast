# BERT for Trajectory Forecasting

---

## Scripts 


+ **BERT_regr_with_goal.py** main script that train a Regressive BERT for trajectory prediction. 

Different inputs can be choosen: *--data_type 0* for Positions, *--data_type 1* for Speed and *--data_type 2* for Relative Positions.

Moreover, model can be train with different level of information on the goal: *--goal_type 0* for No Goal, *--goal_type 1* for True Goal and *--goal_type 2* for Estimated Goal.

An example of line to run it is:

```
CUDA_VISIBLE_DEVICES=0 python BERT_regr_with_goal.py --dataset_name eth --name eth --max_epoch 50 --batch_size 128 --data_type 2 --goal_type 2 --verbose 0
```



+ **BERT_quant_with_goal.py** main script that train a Quantized BERT for trajectory prediction. 

Basically, that is the quantized version of previous script. 
Then data_type and goal_type can be switched with the same method.

An example of line to run it is:

```
CUDA_VISIBLE_DEVICES=0 python BERT_quant_with_goal.py --dataset_name eth --name eth --max_epoch 50 --batch_size 128 --num_clusters 1000 --data_type 2 --goal_type 2 --verbose 0
```


+ **BERT_quant_classification_int_pos.py - BERT_regr_classification_int_pos.py** these scripts refer to experiments about understanding if BERT can localize (in time) a checkpoint. Basically we select an intermediate position in the target and we move it in another one. 
Then we train BERT to classify its original position. Here some examples, for the quantized approach:

```
CUDA_VISIBLE_DEVICES=0 python BERT_quant_classification_int_pos.py --dataset_name eth --max_epoch 50 --batch_size 128 --num_clusters 1000 --data_type 2 --goal_type 1 --verbose 1 --warmup 3 --factor 0.01
```

and for the regressive one:

```
CUDA_VISIBLE_DEVICES=0 python BERT_regr_classification_int_pos.py --dataset_name eth --max_epoch 50 --batch_size 128 --data_type 2 --goal_type 1 --verbose 1 --warmup 1 --factor 0.01
```

+ **BERT_with_goal_interm.py** da sistemare;


+ **baselineUtils.py** script to preprocess data from raw format to pytorch dataset;


+ **goal_estimator.py** class with model that learns how to estimate the goal;


+ **individual_TF.py** some usefull model functions;


+ **kmeans.py** with this script you can generate the classes for the different quantized approaches;


+ **utils.py** some utility function for visualization.





***

## Folders

+ **clusters:** classes obtained with k-means algorithm. Those clusters needs to move to quantized/classification approach. Clusters are divided by number of classes (500, 1000) and data type (Speed, Relative Positions);


+ **datasets:** original ETH-UCY dataset;


+ **kmeans_pytorch:** files for k-means algo;


+ **results:** folder where each file, with loss and metric, is saved;


+ **transformer:** files for Transformer and BERT.


```
TODO

- uniformare k-means con data-type
- correggere BERT_with_goal_interm.py
- commenti in BERT_regr_classification_int_pos.py & BERT_quant_classification_int_pos.py
```


