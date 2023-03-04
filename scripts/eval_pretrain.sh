CUDA_VISIBLE_DEVICES=$1 \
python main.py --split train --eval 1 --load pretrained_models/sem_exp.pth \
--auto_gpu_config 0 --num_processes 25 --num_processes_per_gpu 5 --sim_gpu_id 1 \
--num_eval_episodes 50 --exp_name 2023-03-04/pretrain_rollouts