
python construct_loc_loc_graph.py --model_type transe --dataset Istanbul --pretrain_model ./pretrain/Istanbul-transe-1751604860.ckpt_final --version scheme2 --threshold 100 --user_count 8750 --loc_count 13194
python construct_user_loc_graph.py --model_type transe --dataset Istanbul --pretrain_model ./pretrain/Istanbul-transe-1751604860.ckpt_final --version scheme2 --threshold 100 --user_count 8750 --loc_count 13194


python construct_loc_loc_graph.py --model_type transe --dataset Moscow --pretrain_model ./pretrain/Moscow-transe-1751604815.ckpt_final --version scheme2 --threshold 100 --user_count 2665 --loc_count 12527
python construct_user_loc_graph.py --model_type transe --dataset Moscow --pretrain_model ./pretrain/Moscow-transe-1751604815.ckpt_final --version scheme2 --threshold 100 --user_count 2665 --loc_count 12527

python construct_loc_loc_graph.py --model_type transe --dataset Singapore --pretrain_model ./pretrain/Singapore-transe-1752130534.ckpt_final --version scheme2 --threshold 100 --user_count 1626 --loc_count 11867
python construct_user_loc_graph.py --model_type transe --dataset Singapore --pretrain_model ./pretrain/Singapore-transe-1752130534.ckpt_final --version scheme2 --threshold 100 --user_count 1626 --loc_count 11867

python construct_loc_loc_graph.py --model_type transe --dataset Tokyo --pretrain_model ./pretrain/Tokyo-transe-1751605563.ckpt_final --version scheme2 --threshold 100 --user_count 2484 --loc_count 8196
python construct_user_loc_graph.py --model_type transe --dataset Tokyo --pretrain_model ./pretrain/Tokyo-transe-1751605563.ckpt_final --version scheme2 --threshold 100 --user_count 2484 --loc_count 8196