## look result in-person

cut -f2 ./data/review.sorted.uniq.refined.shuf.test.tsv | shuf | head -n 20 | python ./classify.py --model_fn ./result_model/model.20230415_232506.pth --gpu_id 1