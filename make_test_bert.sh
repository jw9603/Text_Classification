cat ./data/review.sorted.uniq.refined.shuf.test.tsv | shuf | head -n 20 | awk -F'\t' '{print $2}' | python classify_plm.py --model_fn ./result_model/review.native.kcbert.20230406_231408.pth --gpu_id 0 | awk -F'\t' '{print $1}' > ./model/review.native.kcbert.20230406_231408.pth.result.txt