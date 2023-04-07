# Text_Classification
This is a classification repository for movie review datasets using rnn, cnn, and bert.

It is still incomplete.
## Usage
### 0. Dependencies
Run the following commands to create a conda environment (assuming RTX A6000):
```bash
conda create -n ntc python=3.8 ipykernel
source activate ntc
conda install pytorch=1.9.0 torchvision=0.10.0 torchaudio=0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install torch_optimizer
conda install -c pytorch torchtext==0.10.0
conda install ignite -c pytorch
```
```
bash ./setup.sh
```
### 1. Download data

Data on movie reviews has been crawled and saved as tsv files. 

The first column is the label (positive or negative), and the second column is the review corpus.

```bash
positive	나름 괜찬항요 막 엄청 좋은건 아님 그냥 그럭저럭임... 아직 까지 인생 디퓨져는 못찾은느낌
negative	재질은플라스틱부분이많고요...금방깨질거같아요..당장 물은나오게해야하기에..그냥설치했어요..지금도 조금은후회중.....
positive	평소 신던 신발보다 크긴하지만 운동화라 끈 조절해서 신으려구요 신발 이쁘고 편하네요
positive	두개사서 직장에 구비해두고 먹고있어요 양 많아서 오래쓸듯
positive	생일선물로 샀는데 받으시는 분도 만족하시구 배송도 빨라서 좋았네요
positive	아이가 너무 좋아합니다 크롱도 좋아라하지만 루피를 더..
negative	배송은 기다릴수 있었는데 8개나 주문했는데 샘플을 너무 적게보내주시네요ㅡㅡ;;
positive	너무귀여워요~~ㅎ아직사용은 못해? f지만 이젠 모기땜에 잠설치는일은 ? j겟죠
positive	13개월 아가 제일좋은 간식이네요
positive	지인추천으로 샀어요~ 싸고 가성비 좋다해서 낮기저귀로 써보려구요~
```

### 2. Tokenization (Optional)

You need to tokenize sentences in the corpus. You need to select your own tokenizer based on the language. (e.g. Mecab for Korean)

Bert can do tokenization from the library in huggingface.
```
bash tokenize.sh
```





## Reference

- Kim, Convolutional neural networks for sentence classification, EMNLP, 2014
- Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, ACL, 2019
- [Lee, KcBERT: Korean comments BERT, GitHub, 2020](https://github.com/Beomi/KcBERT)

Also, I wrote & studied it while referring to this person's code.
- https://github.com/kh-kim

Many thanks to him
