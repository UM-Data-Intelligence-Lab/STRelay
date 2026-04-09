This is the implementation of the manuscript entitled:
STRelay: A Universal Spatio-Temporal Relaying Framework for Location Prediction over Human Trajectory Data

We upload the STRelay with Graph-Flashback (the best-performing base model) for your reference. 

# Requirements
```
python: 3.8
torch: 2.3.0
numpy
tqdm
scipy
```

# To reproduce STRelay with best performing base model in our paper, please run following scripts:

```
python -u train.py --dataset Istanbul.txt --trans_loc_file POI_graph/Istanbul_scheme2_transe_loc_temporal_20.pkl --trans_interact_file POI_graph/Istanbul_scheme2_transe_user-loc_100.pkl  --gpu 0  --batch-size 128 --STRelay True --log_file results/strelay_istanbul
```

```
python -u train.py --dataset Tokyo.txt --trans_loc_file POI_graph/Tokyo_scheme2_transe_loc_temporal_20.pkl --trans_interact_file POI_graph/Tokyo_scheme2_transe_user-loc_100.pkl --gpu 0 --batch-size 64 --STRelay True --log_file results/strelay_tokyo
```

```
python -u train.py --dataset Singapore.txt --trans_loc_file POI_graph/Singapore_scheme2_transe_loc_temporal_20.pkl --trans_interact_file POI_graph/Singapore_scheme2_transe_user-loc_100.pkl  --gpu 0  --batch-size 64 --STRelay True --log_file results/strelay_singapore 
```

```
python -u train.py --dataset Moscow.txt --trans_loc_file POI_graph/Moscow_scheme2_transe_loc_temporal_20.pkl --trans_interact_file POI_graph/Moscow_scheme2_transe_user-loc_100.pkl --gpu 0 --batch-size 128 --STRelay True --log_file results/strelay_moscow
```

