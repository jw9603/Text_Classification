cat ./data/review.sorted.uniq.refined.shuf.test.tsv | awk -F'\t' '{print $2}' | python classify.py --model_fn ./result_model/model.20230415_232506.pth --gpu_id 0 --drop_cnn | awk -F'\t' '{print $1}' > ./model/model.202301415_232506_rnn.pth.result.txt
