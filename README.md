# Pytorch_Text_Classification
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
conda install transformers
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
### 3. Train
Shell script files have arbitrarily fixed hyperparameters, but you can change them as you like.

For RNN, run
```
bash run_rnn.sh
```
For ensemble(RNN+CNN), run
```
bash run.sh
```
For CNN, run
```
bash run_cnn.sh
```
For BERT, run
```
bash run_bert.sh
```

However, since both rnn and cnn models were learned through ensemble learning(run.sh), if drop_cnn or drop_rnn factor is given in the test process,
there is no need to learn rnn and cnn separately.

Rnn and Cnn are learned through train.py, and the arguments are as follows.
```
$ python train.py --asdf
usage: train.py [-h] --model_fn MODEL_FN --train_fn TRAIN_FN [--gpu_id GPU_ID] [--verbose VERBOSE] [--min_vocab_freq MIN_VOCAB_FREQ] [--max_vocab_size MAX_VOCAB_SIZE] [--batch_size BATCH_SIZE]
                [--n_epochs N_EPOCHS] [--word_vec_size WORD_VEC_SIZE] [--dropout DROPOUT] [--max_length MAX_LENGTH] [--rnn] [--hidden_size HIDDEN_SIZE] [--n_layers N_LAYERS] [--cnn] [--use_batch_norm]
                [--window_sizes [WINDOW_SIZES [WINDOW_SIZES ...]]] [--n_filters [N_FILTERS [N_FILTERS ...]]]
train.py: error: the following arguments are required: --model_fn, --train_fn
```
Similarly, BERT is learned with finetune_native.py, and the arguments are as follows.
```
$ python finetune_native.py --asdf
usage: finetune_native.py [-h] --model_fn MODEL_FN --train_fn TRAIN_FN [--pretrained_model_name PRETRAINED_MODEL_NAME] [--use_albert] [--gpu_id GPU_ID] [--verbose VERBOSE] [--batch_size BATCH_SIZE]
                          [--n_epochs N_EPOCHS] [--lr LR] [--warmup_ratio WARMUP_RATIO] [--adam_epsilon ADAM_EPSILON] [--use_radam] [--valid_ratio VALID_RATIO] [--max_length MAX_LENGTH]
finetune_native.py: error: the following arguments are required: --model_fn, --train_f
```
### 4. Evaluate trained model

The order is as follows.

1. eval_{model}.sh
- Look the real result in-person.
```
bash eval_{}.sh
```
- Since 20 sentences were selected and shuffling was performed, you can see the results of different sentences each time.
- The result is : 
```
RNN,
positive        너무빠른배송좋고 상품도굿이예요ㅎ
positive        아기가 잘먹어요♡
positive        배송 빨라요~~~~ 포장도 꼼꼼하고 좋네요!!! 많이 파세요~
negative        모니터 사진과 실제 제품의 모습이 차이가 많았습니다. 일단 정장구두라서 반치수 크게 주문했더니 너무 헐겁게 커서 못 신겠더라구요, 싼맛에 한번 신어보려구 했는데 도저히 못신겠어서 처박아 놨어요. 실망이 커서 반품이나 교환을 생각도 못한채 시간이 지나버렸네요. 정녕,,,싼게 비지떡 인가요?ㅜ
negative        주머니봉제가 제대로 안되어있어 구멍난 상태군. 고객에게 보낼 상품이라면 적어도 검수는 해야하지 않을까? 참으로 답답하네!!!
positive        블랙프라이스 덕분에 아주 저렴하게 잘샀어요~유통기한도 22년도라 최근제조상품으로 보내주셨구요^^저는 본품으로 제로를 시켰고사은품으로는 오리지날로 왔네요~그동안 쭉 제로만썼는데 오리지날을 오랜만에 써보니확실히 화~ 한감이 있습니다^^
negative        그린..... 받아보니 형광연두입니다 헐 ㅠ
positive        유통기한 넉넉하고. 굿
positive        가게에 필요해서 산건데 이거뭐 비밀을 제가 만들어써도 더빨리 만들겠네요 배송이 너무 늦습니다. 물론 택배회사 잘못이겠지만요
positive        자석 힘이 매우 강하네요 붙으면 손으로 떼기 힘든정도네요 ㅋ
positive        조아용!
negative        느려요 배송
negative        길이는 적당한데 발목이 너무 좁아서 쫄바지 같음..
positive        유통기한이 걱정되서 신제품으로만 주문했는데 맘에 들구요 저는 롤보다는 스틱이 더 좋은거 같아요. 바로 옷 입어도 되니까 ㅎㅎ 근데 뭐 취향에 따라 다를듯. 스프레이는 뿌릴때 겁나서 ㅋㅋ 롤이나 스틱 추천해드릴게요 특히 복숭아 모양 그려져있는 신제품 향이 좋아요 ㅎㅎ
negative        누가 받아보고 반품한건지 왠지 사용한 흔적이 지문도 여기저기 묻어있고 건전지도 빠져있고 별로한달후 후기 주말에 한번쓸까말까 하는데그 전에산거는 괜찮았는데 이거 세번째쯤 쓰는데 본체에서 쇠긁는소리나고나만불량인건지 뭔지 겁나서 못쓰겠어요
positive        생각보다 젤리케이스가 하드해서 대만족이예요^^ 케이스 구매에서 망설였던 점이 젤리 케이스라서 몇번쓰면 늘어나고 헐렁해질까봐 고민좀 했었는데, 막상 받아보니, 생각보다 젤리케이스도 하드하고 겉에 가죽케이스도 사진보다 더 고급스럽고 퀄리티가 낮지 않네요^^ 대만족이예요~~ 많이 파세요~~^^
positive        재질이좀...
negative        쓰기엔 그냥저냥 편합니다 잘쓰겠습다.
positive        이건 말이 안되는 가격이당.. 두꺼운 종이에 100일 인쇄해서 2800원이나 받아쳐먹다니. 28원을 100배나 해먹다니.. 어이가 없네.. 애기 머리가 작다고 하지만.. 이건 너무 한거 아니가요.. 작아도 넘 작고 2800원에 설마 했는데 완전 최악입니다
positive        여러개 붙이고 세탁기에 한번 돌리니 테두리 죄다 들뜨고 올풀리고 보풀 막생기고 처음에만 좋았지 이거 완전히 1회용인데요
```
```
CNN,
positive        빠른배송따따봉
positive        안전하게 포장되어 잘 받았습니다. 태풍예보에도 불구하고...배송해주신 택배기사님 감사합니다ㅠㅠ
positive        상당히 마음에 안들어요 1번밖에 안신엇는데신발에 쓸려서 피도나고 괜히삿다는생각이 떠나질않아요 돈낭비한것같아요
positive        좋습니다 용량도크고 잘쓸께요^^
negative        생각 보다 두께가 있어 좋네요 2장으로 보내 달라고 했는데 1장으로 와서 제가 잘라 써야 했지만 베란다에 깔고 나니 바닥에서 냉기가 올라오지 않아 너무 좋아요
positive        싸고 좋아요 양도 많고 잘 지워집니다. 저는 이것만 사용하네요..
positive        10일 사용해 본 솔직후기 ~~~♡ 후기도 좋고 해서 살짜~쿵 기대를 가지고 아침에 잠깐 저녁에 열심히 사용해봤지요 클렌징할때는 솜을 대고 하라는데 솜이 불량인건지 자꾸 튿어지고 밀리고 ... 기기의 거칠한 부분에 솜이 끼기도하고 해서 한 번 하고는 안합니다 그냥 마사지용으로 써요 이온모드는 건조한피부용으로 해놓음 더 따끔거리더라구요 효과가 있는지는 잘 모르겠구요 마사지모드는 불만 들어온다는거지 클렌징모드와 동일한 진동모드인 것 같아 잘 안 씁니다 크기가 커서 코옆이라던지 인중 눈과 눈썹사이 부분은 마사지가 힘들다는 단점도 있구요 10일간 부지런히 사용해 본 결과 ~~~ 큰 효과는 없습니당ㅠㅠ? 모공이 작아지지도 주름이 펴지는것도 얼굴이 환해지는 것도 아니더라구요 근데 함께 보내주신 팩은 효과가 좋더라구요 한 번 했더니 피부가 맑아보이고 기미잡티도 덜해지더라구요 팩이나 열심히 해야겠다는 결론에 도달했습니당 ~~~^^
negative        풀커버는 맞으나 터치 잘안됨.겉 테두리 부분쪽은 세게눌러야 터치됨.사서한시간쓰고 바로버렸음
positive        착용감이 보드랍습니다.
positive        너무작아 애들용인줄.. 반품비아까와그냥 두는데...진짜너무작네요
positive        엄마가 고혈압이셔서 비트가 혈압에 좋다고 하길래 사드렸어요 맛은 괜찮다고 하시는데 효과가 좋다하시면 계속 사드리려구요
positive        붙이면 용하다
positive        완전좋아요ㅋㅋ
positive        좋은 제품 잘 샀어요~
positive        저렴하고좋아여
positive        끈적이지않아서 샘플써보고 본품샀어요 좋네요
negative        과일맛 떡뻥에서 탄맛이 나요. 타사 제품들 먹을 땐 단 한 번도 안 그랬는데 여기 제품 먹은지 30분도 안 되어서 입가에 알러지 반응 올라오고 먹다가 머리와 귀를 긁네요. 그동안 다양한 맛으로 떡뻥 몇 박스를 먹였는데 이런 반응은 처음이라 너무 황당해요. 배송 하나는 빠르네요.
positive        상하차하는 과정에서 깨진건지 피죤이 흘러나와 포장박스가 젖어서 찢어지고 너덜너덜해서 도착햇는데 싼값에 좋은상태로 받았음 좋았을텐데ㅡㅡ
positive        와이프는 한달도 안되서 다 깨져버려서 진즉에 딴걸로 갈았구요 잔 좀 조심히 쓰는 편이라 아직버티고 있지만.... ㅡㅡㅋㅋ 내구성 어마어마하게 약합니다 구매하시려는 분들 참고하세요.... 판매자분... 광고는 엄청 튼튼한거처럼 하셨던데요... 판매자분께서도 이 제품 사용하시는지 의문이네요... 아주 미세한 충격에 모서리란 모서리는 죄다 깨져나가고 좀 심하게 약한거 아닌가요?????
negative        사진에 비해 별로..엄마가 작은가방 필요하데서 그냥 엄마쓰라고 주려고요..다른 쇼핑몰은 쿠폰할인받아서 총 만원에 샀는데..그게 훨나아요..
```
```
ensemble,
negative        그냥 투명케이스 살걸 그랬어요. 금방 더러워지는 느낌
positive        가격만큼이네요 가격만큼이네요 가격만큼이네요 가격만큼이네요 가격만큼이네요 가격만큼이네요 가격만큼이네요
negative        일주일이 지나도 아무 연락도 없고 배송중이라고 떠있고 기다리다 못해 문의하니 아직 입고가 안?다는둥 ,, 배송 느린거 짜증나고 대응도 불쾌해서 환불 요청을 해놨습니다. 속이 후련하더라고요. 그런데, 이게 왠일.? 반품신청 클릭해놓고 2일이 지났는데 분명 환불 신청 뻔히 한거 알면서도 물건을 막무가내로 보내셨네요?? 하도 물건이 도착안해서 반품신청하고 다른 곳에서 구두 하나 구매해놓은 상태였는데 그구두 도착한 날, 같이 오더라고요?(거긴 2일만에 도착했고 당일배송못했다고 사과문자도 보냅디다) 환불 신청해놨는데 막무가내로 물건 보내는건 대체 뭔짓입니까?? 반품 귀찮고 돈도 들고 짜증나서 그냥 신지만, 너무 불쾌합니다. 괜히 검정구두 두켤레나 산 꼴이네요. 디자인고 딴곳에서 산게더 고급스럽고 그것만 신고 다니고 있는데도요. 구두도 맘에 안들고, 디자인도 사진보다 별로입니다. 딱 봐도 싸구려같고요, 솔직히 물건 자체보다,,받아놓지도 않아 있지도 않는 물건 무작정 기다리게 하고 반품/교환처리도 형편없는 점이 참 화가 납니다. 사이트 하면서환불/반품신청 해본것도 처음이고, 신청후 아무 통보없이 몇일 지나서 물건 보내는것도 ..너무 황당하네요. 여기서 구매한거 몇푼안되도 참~~ 돈아깝네요 ㅉㅉ
positive        다른 사람들처럼 긴 머리에 이쁘게 하고 싶어서 구입했어요. 저렴하기도 하고 그런데 하자마자내려가고 고정도 안되고 한번하고 나면 다시 형태를 잡아야 하고해서 불편해요..
negative        사진이랑 넘 달리 많이 크네요
negative        곡물맛나요 숭늉맛이라네요 실패
positive        다 터져서와서 교환해달라니까 사은품 다떨어졌다고 다른거 보내주고. 정말 넘하네요T.T
positive        좋아요 아주좋아요 완전좋아요 좋아요 완전조으다
positive        많이파셔요!
positive        항상 쓰던거라ㅋㅋ믿고 구매합니다
positive        사진이랑 완전틀려요 무슨 고무신온줄알았네 귀찮아서 반품은안하고 작업신발로 써야겠어요
positive        그라노떼 괜찮아요 아주 맛있어요 사진 인화권도 감사요
positive        제일비싼암막 콤비블라인드 구매했는데 빛다비침 아침에 눈저절로 떠집니다
negative        10개월 아이 머리감을때마다 울어서 놀리면서 목욕할려고 씌웠는데 앞쪽으로 물을 흘리면 아래쪽으로 물이 떨어져 내리네요. 결국 잡아 뜯어버려서 여전히 울면서 머리감습니다 ㅠㅜ
positive        터치가...영 안됨ㅜ
positive        사이즈가 작고 신축성이 없어요 실망스럽네요
positive        첨사봤는데 믿고 구매했네요.
positive        천연세제라 안심은 되는데 얼룩이 완전히 지워지진 않아요 얼룩은 따로 손빨래후 세탁기에 넣어야해요ㅜ
positive        인트라젠이 새로 들어왔다고해서 판매가격을물어보니까12,000원이였는데 다른 인테넷 가격보다 꽤 많이 받아서 별로였습니다.
```
```
BERT,
negative        기대보다 향이별루네요
negative        저 말고 다른 분들의 후기를 봐도 배송은 어쩔수없이 느린 부분인가봐요. 전 제대하고 제방에 걸려고 샀는데 알루미늄 커버가 찌그러져 있네요. 혹시나 하고 배송 박스를 같이 봤는데 역시나 찌그러져 있더군요 배송사고 문제를 어떻게 해결해줄지.. 모르겠네요
negative        배송이 늦어요 ~~~~
negative        한쪽이 찌그러져 왔어요..ㅡ.ㅡ
negative        파란색이 은은한색인줄 알았는데 제가볼땐 촌스런 파랑이에요 실제색이랑 거의 흡사하게 찍혔어요남편과 아들이 좋다해서 그냥씁니다배송은 빨랐어요
positive        색상이 너무맘에들어서 삿어요~ 비밀번호 설정도 어렵진않을까 걱정했는데 쉬웠어요 ㅎㅎ 아버지도 선물드릴꺼예요
positive        뽁뽁이로 잘 감싸져 배송. 좋아요
negative        베니를 주문했는데 블랙베리를 보내면 어떻하나요?? 판매자가 확인도 안하고 막 보내요???????
positive        배송빠르고 맛있네요
positive        제품도 맘에 들고 배송도 맘에 들어요
negative        배송 3주
positive        너무 만족해요. 배송도 빠르고 상품 정확하게 잘 받았어요 재구매의사 있습니다~
positive        빠른배송으로 잘받았습니다
positive        아직 안먹여봤네요
negative        강화유리인줄 알고 샀는데 받아보니 조금 두꺼운 필름이더군요.괜히 샀다는 생각이 확드네요.7500원 사기당한 느낌이...
negative        제품이 리퍼네요
negative        모서리윗쪽끝부분 그리고 부분적으로 터치안됨 붙이고 십분쓰고 바로 때버렸음 돈만날림
positive        상품은 만족스럽고 배송도 빠르고 좋아요
positive        가성비 짱가성비 짱가성비 짱가성비 짱 가성비 짱 가성비 짱 가성비 짱 가성비 짱 가성비 짱 가성비 짱 가성비 짱 가성비 짱 가성비 짱 가성비 짱 가성비 짱 가성비 짱 가성비 짱 가성비 짱 가성비 짱 가성비 짱 가성비 짱 가성비 짱 가성비 짱 가성비 짱 가성비 짱 가성비 짱
positive        저렴하게 잘 구매했습니당
```
2. make_test_{model}.sh
- make the result text file included with only columns(labels)
3. make_ground_truth.sh
- make ground_truth test data included with only columns(labels)
4. get_accuracy.sh
- take the Accuracy

## Reference

- Kim, Convolutional neural networks for sentence classification, EMNLP, 2014
- Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, ACL, 2019
- [Lee, KcBERT: Korean comments BERT, GitHub, 2020](https://github.com/Beomi/KcBERT)


