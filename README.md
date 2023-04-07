# Text_Classification
This is a classification repository for movie review datasets using rnn, cnn, and bert.

It is still incomplete.
## Usage
### 0. Dependencies
Run the following commands to create a conda environment (assuming CUDA10.1):
```bash
conda create -n ntc python=3.8 ipykernel
source activate ntc
conda install pytorch=1.9.0 torchvision=0.10.0 torchaudio=0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install torch_optimizer
conda install -c pytorch torchtext==0.10.0
conda install ignite -c pytorch
```













## Reference

- Kim, Convolutional neural networks for sentence classification, EMNLP, 2014
- Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, ACL, 2019
- [Lee, KcBERT: Korean comments BERT, GitHub, 2020](https://github.com/Beomi/KcBERT)

Also, I wrote & studied it while referring to this person's code.
- https://github.com/kh-kim

Many thanks to him
