This is the implementation of the manuscript entitled:
STRelay: A Universal Spatio-Temporal Relaying Framework for Location Prediction over Human Trajectory Data

We upload the STRelay with Graph-Flashback (the best-performing base model) for your reference. 

# DATA and Pretrained Graphs

The datasets are pretrained graphs available here: https://drive.google.com/drive/folders/1WJOfoMPUj3nVJOqJKwPtDS-N8G8-pI6s?usp=drive_link

Please download from the link above, and place the two folders ``data`` and ``POI_graph`` in the current directory.

# Requirements
```
python: 3.8
torch: 2.3.0
numpy
tqdm
scipy
```

# To reproduce STRelay with the best-performing base model in our paper, please run the following commands:

```
python -u train.py --dataset Istanbul.txt --trans_loc_file POI_graph/Istanbul_scheme2_transe_loc_temporal_20.pkl --trans_interact_file POI_graph/Istanbul_scheme2_transe_user-loc_100.pkl --gpu 0 --batch-size 128 --hidden-dim 10 --rnn lstm --head 2 --temporal_intervals 24 --spatial_intervals 30 --STRelay True --lambda_temporal 0.5 --lambda_spatial 0.5 --lr 0.01 --weight_decay 0 --epochs 100 --validate-epoch 5 --seed 0 --log_file results/strelay_istanbul
```

```
python -u train.py --dataset Tokyo.txt --trans_loc_file POI_graph/Tokyo_scheme2_transe_loc_temporal_20.pkl --trans_interact_file POI_graph/Tokyo_scheme2_transe_user-loc_100.pkl --gpu 0 --batch-size 64 --hidden-dim 10 --rnn lstm --head 2 --temporal_intervals 24 --spatial_intervals 30 --STRelay True --lambda_temporal 0.5 --lambda_spatial 0.5 --lr 0.01 --weight_decay 0 --epochs 100 --validate-epoch 5 --seed 0 --log_file results/strelay_tokyo
```

```
python -u train.py --dataset Singapore.txt --trans_loc_file POI_graph/Singapore_scheme2_transe_loc_temporal_20.pkl --trans_interact_file POI_graph/Singapore_scheme2_transe_user-loc_100.pkl --gpu 0 --batch-size 64 --hidden-dim 10 --rnn lstm --head 2 --temporal_intervals 24 --spatial_intervals 30 --STRelay True --lambda_temporal 0.5 --lambda_spatial 1.0 --lr 0.002 --weight_decay 0 --epochs 100 --validate-epoch 5 --seed 0 --log_file results/strelay_singapore
```

```
python -u train.py --dataset Moscow.txt --trans_loc_file POI_graph/Moscow_scheme2_transe_loc_temporal_20.pkl --trans_interact_file POI_graph/Moscow_scheme2_transe_user-loc_100.pkl --gpu 0 --batch-size 128 --hidden-dim 10 --rnn lstm --head 2 --temporal_intervals 24 --spatial_intervals 30 --STRelay True --lambda_temporal 0.5 --lambda_spatial 1.0 --lr 0.002 --weight_decay 0 --epochs 100 --validate-epoch 5 --seed 0 --log_file results/strelay_moscow
```

