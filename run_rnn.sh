model='rnn'
dt=`date '+%Y%m%d_%H%M%S'`
mkdir ./result_model
python train.py --model_fn ./result_model/review.native.${model}.${dt}.pth --train_fn ./data/review.sorted.uniq.refined.tok.shuf.train.tsv --gpu_id 2 --batch_size 128 --n_epochs 10 --word_vec_size 256 --dropout .3 --${model} --hidden_size 512 --n_layers 4
