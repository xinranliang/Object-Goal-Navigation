CUDA_VISIBLE_DEVICES=$1 \
python main.py --split train --eval 1 --load pretrained_models/sem_exp.pth \
--auto_gpu_config 0 --num_processes 5 --num_processes_per_gpu 1 \
--num_eval_episodes 1 --exp_name 2023-03-04/debug_pretrain --print_images 1 \
--seed 120