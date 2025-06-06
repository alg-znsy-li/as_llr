nohup deepspeed --include=localhost:0,1,2,3,4,5,6,7 train.py \
 --deepspeed \
 --ckpt_interval 5120 \
 --lora_r 16 \
 --k 128 \
 --lr 0.0005 \
 --max_len 20 \
 --data_aug_n 1 \
 --dropout_rate 0.5 \
 --warmup_step_num 20 \
 --pretrained_model /lpai/volumes/jfs-ppl-alg-disk-bd-ga/songjunjie/ppl_rank/model/Qwen1___5-1___8B \
 --out_dir  /lpai/output/models/Qwen1___5-1___8B__lr1E-3__lora_128_240805_iteration \
 --data_file /lpai/volumes/jfs-ppl-alg-disk-bd-ga/songjunjie/ppl_rank/model_set_abstract_qwen/dataset_20240824.pkl \
 --deepspeed_config ../cfg/ds_config_mixprecision_stage2.json 2>&1 > /lpai/output/logs/log.txt &
