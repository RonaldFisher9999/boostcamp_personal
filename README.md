부스트캠프 AI Tech 5기에서 진행한 3개의 대회에서 구현한 코드입니다.

## 사용방법
#### 1. 가상환경 설정
    ```
    conda init
    conda create -n boostcamp python==3.10.11 -y
    conda activate boostcamp
    pip install -r requirements.txt
    ```

#### 2. `data` 폴더의 파일들 압축해제

#### 3. `main.py` 파일 실행  
code 폴더 아래 book, dkt, movie 폴더의 main.py 파일 실행
  ```
  python main.py
  ```


## Movie Recommendation (2023.05.31 ~ 2023.06.22)

### 1. 개요
- **영화 시청 이력**(유저, 영화, 시간)과 영화에 대한 데이터(장르, 감독, 개봉연도 등) 제공
- 유저의 영화 시청 이력 중 **마지막 N개, 중간 랜덤 M개 누락**
- **10개의 영화를 추천**해 누락된 영화에 대한 예측 (**Recall@10**)

### 2. 역할
#### 2.1 **BERT4Rec** 모델 구현 및 실험
- 대회의 task는 **Masked Language Modeling(MLM)과 유사**
- **MLM에 사용하는 BERT 모델을 추천 task에 적용**한 BERT4Rec 모델 선택
- 논문을 먼저 그대로 구현한 후 **모델을 task에 적합하도록 수정**
- PyTorch 사용
  
<details>
<summary>상세</summary>

- 모델 구현
  - Valid data 생성
    - 유저 sequence마다 마지막 5개, 중간 5개의 랜덤 아이템 valid label로 사용
    - 남은 데이터에서 유저마다 `max_len - 10` 길이의 **sequence 샘플링**해서 valid data로 사용
  - Train data 생성
    -	Valid label을 제외한 데이터에서 유저마다 `max_len` 길이의 sequence `n_samples` 개수만큼 샘플링
    - 이때 `tail_ratio`의 비율만큼 마지막 아이템 사용
      ```
      ex) total sequnce length = 200, max_len = 100, tail_ratio = 0.25  
      마지막 25개 제외한 475개의 아이템 중 75개 + 마지막 25개 아이템
      ```
    - 각 샘플마다 일정 비율로(`mask_prob`) 마스킹
    - 마스킹된 step의 원래 아이템을 train label로 사용
  - 모델 구조
    - Input layer, Encoder layer, Output layer로 구성
    - Input layer
      - Sequence의 아이템을 embedding vector로 변환
      - `max_len`만큼의 positional embedding 사용
      - 유저, 아이템의 side information도 활용
    - Encoder layer
      - **Transformer의 encoder layer**(self-attention)와 동일
      - 따라서 attention mask를 적용하지 않음
    - Output layer
      - 1개의 linear layer를 사용
      - **마스킹된 step마다** 전체 아이템에 대한 **score 계산**
    - Loss function
      - train label과 score를 비교
      - **Cross Entropy loss** 사용
  - Validation
    -	앞서 생성한 valid data에 마스크를 섞어줌 (마지막 5개, 중간 랜덤 위치 5개)
    - 모델에 valid data를 입력해 마스킹된 step의 score 계산
    - Score를 표준화하기 위해 softmax를 적용한 후 합산
    - **합산 score 기준 상위 10개의 아이템 선택**해 valid label과 비교
    - Recall@10으로 평가
  - Inference
    - Valid label을 제외하지 않은 전체 데이터에서 `max_len - 10` 길이의 sequence 샘플링
    - 이후는 validation과 동일하게 10개의 아이템 선택
- 실혐 결과
  - **WandB**로 실험 관리
  - 유저, 아이템 정보를 추가해도 성능이 나아지지 않음
  - 파라미터 튜닝
    - `max_len`: 200 
    - `n_samples`: 10
    - `tail_raitio`: 0.5나 1.0 (`max_len`에 따라 변화)
    - `mask_prob`: 0.5

</details>

### 3. 성능 향상 (Recall@10)
- Public score 기준
- 0.06(최초 구현) → 0.1236


## DKT(Deep Knowledge Tracing) (2023.05.03 ~ 2023.05.25)

### 1. 대회 개요
-	**유저의 문제 풀이 이력**(유저, 문제, 시간), 문제 관련 정보(테스트 번호, 태그)가 제공
-	test 데이터에는 train 데이터에 등장하지 않은 유저의 문제 풀이 이력이 주어짐
-	test 데이터 **유저의 마지막 문제에 대한 정답 여부 예측**하는 대회 (**AUROC**)

### 2. 역할
#### 2.1 **EDA와 Feature Engineering**
- 유저와 문제의 그룹에 관련된 **feature 생성**
- 유저의 그룹 정보는 **팀에서 실험한 모든 모델에 사용**
#### 2.2 **LightGCN** 모델 구현 및 실험
- 유저와 문제를 node로, 정답 여부를 edge로 간주하면 **edge-prediction task와 유사**
- GNN 계열 모델 중 **구조가 단순하면서도, 괜찮은 성능**을 보인다고 알려진 LightGCN 모델 선택
- PyTorch, PyTroch Geometric 사용

<details>
<summary>상세</summary>

- 모델 구현 
  - Train/Valid split
    - Test 데이터와 동일하게 일정 비율 유저 데이터 전부를 valid 데이터로 분리
  - **Cold start 대응**
    - 원래의 LightGCN 모델은 학습하지 않은 node에 대해 예측이 어려움
    - 유저 그룹 feature 사용
    - **신규 유저**(train에 없었던 유저) node의 embedding vector를 **해당 유저 그룹의 평균값으로 초기화**
  - Train/Validation
    - 매번의 train epoch마다, 모델을 복사해 valid data의 유저 node 추가
    - 복사한 모델을 **`valid_n_epochs`만큼 다시 훈련**
    - 신규 유저의 마지막 문제에 대해 정답 예측
    - AUROC로 평가
  - Inference
    - 마찬가지로 train이 끝난 모델을 복사해 test data의 유저 node 추가
    - 복사한 모델을 `valid_n_epochs`만큼 다시 훈련해 예측
- 실험 결과
  - **WandB**로 실험 관리
  - **주로 푼 문제의 종류에 따라 유저 그룹**을 사용한게 가장 성능이 좋았음
  - 파라미터 튜닝
    - `n_layers`: 3
    - `embed_dim`: 128
    - `train_lr`: 0.005
    - `train_n_epochs`: 250
    - `valid_lr`: 0.005
    - `valid_n_epochs`: 50

</details>

#### 2.3 **Boosting 계열 모델**(LightGBM, CatBoost) 실험
- **정형 데이터에서 좋은 성능**을 낸다고 알려진 모델
- **feature 30개 이상 생성 후** lofo-importance 패키지 사용하여 **15개 선택**
- 팀원이 작성한 **Hyperopt**를 사용하는 코드로 **하이퍼파라미터 튜닝**
#### 2.4 Saint+ 모델 실험 보조
-	팀원이 구현한 Saint+ 모델에서 오류 수정
-	유저의 그룹 정보를 feature로 사용할 것을 제안

### 3. 성능 향상 (AUROC)
- Public score 기준
- LightGCN: 0.6791 → 0.8058
- LightGBM: 0.7345 → 0.7879
-	Saint+: 0.7857 → 0.8073


## Book Rating Prediction (2023.04.12 ~ 2023.04.20)

### 1. 개요
- **유저별 책의 평점**, 유저 관련 정보(지역, 나이), 책 관련 정보(저자, 출판사, ISBN, 카테고리)가 제공
- **유저가 책에 부여할 평점을 예측**하는 대회 (**RMSE**)

### 2. 역할 및 기여
#### 2.1 **EDA와 Feature Engineering**
- ISBN을 사용해 책이 출판된 지역에 관련된 **feature 생성**
- 범주형 feature에서 데이터 개수가 적은 **class는 통합**
- 다른 팀원들도 **모두 동일한 전처리 코드 사용**
#### 2.2 **DeepFM** 모델 구현 및 실험
-	**주어진 유저와 책 정보를 활용**할 수 있는 모델이라 선택
- PyTorch 사용
<details>
<summary>상세</summary>

- 모델 구현
  - **Layer를 따로 분리**해 FM, FFM 구현에 재활용
  - Pairwise-interaction을 계산할때 FM 논문에서 제시한 방법을 사용해 **시간복잡도 감소**
- 실험 결과
  - 5 epoch 이내에 빠르게 수렴
  - 파라미터 튜닝
    - `embed_dim`: 4
    - `mlp_dims`: [64, 32, 16]
    - `lr`: 0.005
</details>

### 3. 성능 향상 (RMSE)
-	Public score 기준
- 제공된 여러 baseline모델들 : 2.3 ~ 2.6 → 2.2 ~ 2.3 (전처리 적용)
-	DeepFM : 2.1862 → 2.1766