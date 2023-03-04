export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

CUDA_VISIBLE_DEVICES=$1 \
python main.py \
--exp_name 2023-03-03/train_scratch