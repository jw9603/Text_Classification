dt=`date '+%Y%m%d_%H%M%S'`

python finetune_native.py --model_fn ./result_model/review.native.kcbert.${dt}.pth --train_fn ./data/review.sorted.uniq.refined.shuf.train.tsv --gpu_id 0 --batch_size 80 --n_epochs 5 --pretrained_model_name 'beomi/kcbert-base'