#!/usr/bin/sh
# source Shell/train.sh $data_path
model='New_net'
optimizer='adam'
##自给路径
data_path=${1-'/root/autodl-tmp/Datasets_20000/'}
batch_size=${2-16}
lr=${3-0.001}
lr_policy='warmup'
nepochs=30
patience=5
nsamples=${5-20000}
multi=${6-0}
out_dir='Saved'

export OMP_NUM_THREADS=1
cd ..
python DFusetrain.py --mod $model --data_path $data_path --optimizer $optimizer --learning_rate $lr --lr_policy $lr_policy --batch_size $batch_size --nepochs $nepochs --no_tb true --lr_decay_iters $patience --num_samples $nsamples --multi $multi --nworkers 15 --save_path $out_dir

echo "python has finisched its "$nepochs" epochs!"
echo "Job finished"
shutdown
