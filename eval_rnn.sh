## look result in-person
# 모델 인자는 ensemble 모델이므로 rnn만의 성능을 보기 위해서는 drop_cnn인자 사용
cut -f2 ./data/review.sorted.uniq.refined.shuf.test.tsv | shuf | head -n 20 | python ./classify.py --model_fn ./result_model/model.20230415_232506.pth --gpu_id 1 --drop_cnn