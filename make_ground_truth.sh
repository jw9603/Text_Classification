#make ./model/ground_truth.result.txt

awk -F'\t' '{print $1}' ./data/review.sorted.uniq.refined.shuf.test.tsv > ./model/ground_truth.result.txt