부스트캠프 AI Tech 5기에서 진행한 3개의 대회에서 구현한 코드입니다.

## 사용방법
#### 1. 가상환경 설정
    ```
    conda init
    conda create -n boostcamp python==3.10.11 -y
    conda activate boostcamp
    pip install -r requirements.txt
    ```

#### 2. data 폴더의 압축파일들 압축해제

#### 3. `main.py` 파일 실행
- code 폴더 아래 book, dkt, movie 폴더의 main.py 파일 실행
  ```
  python main.py
  ```

## Movie Recommendation (2023.06)
### 1. 개요
- 부스트캠프 교육 과정 중 참여한 팀 대회
- 영화 시청 이력(유저, 영화, 시간)과 영화에 대한 데이터(장르, 감독, 개봉연도 등) 제공
- 유저의 영화 시청 이력 중 **마지막 N개, 중간 랜덤 M개 추출**
- **10개의 영화를 추천**해 추출된 영화에 대한 예측 (**Recall@10**으로 평가)

### 2. 역할 및 기여
- **BERT4Rec** 모델 구현 및 실험
  - **대회의 task가 Masked Language Modeling과 유사**하다고 생각하여 해당 모델 선택
  - 논문을 먼저 그대로 구현한 후 **모델을 task에 적합하도록 수정**
  - 구현
    - Valid data 생성
      - 유저 sequence마다 마지막 5개, 중간 5개의 랜덤 아이템 valid label로 사용
      - 남은 데이터에서 유저마다 `max_len - 10` 길이의 sequence 샘플링해서 valid data로 사용
    - Train data 생성
      -	Valid label을 제외한 데이터에서 유저마다 `max_len` 길이의 sequence `n_samples` 개수만큼 샘플링
      - 이때 `tail_ratio`의 비율만큼 마지막 아이템 사용
        ```
        ex) total sequnce length = 200, max_len = 100, tail_ratio = 0.25  
        마지막 25개 제외한 475개의 아이템 중 75개 + 마지막 25개 아이템
        ```
      - 각 샘플마다 마지막 5개, 중간 랜덤 5개 아이템에 마스킹
    - 모델 구조
      - Input layer, Encoder layer, Output layer로 구성
      - Input layer
        - Sequence의 아이템을 embedding vector로 변환
        - 
  - Validation
    -	valid label을 제외한 데이터에서 sequence length - 10개의 데이터 샘플링
    - 이 샘플링 된 데이터의 마지막 5자리, 중간 랜덤 5자리에 mask 10개를 섞어줌
    - masking된 아이템에 대해 score 계산
    - 각 스텝의 score에 softmax 적용 -> 더해서 전체 스코어 반환
    -	합산 score 기준 상위 10개 아이템 선택, valid label과 비교해 Recall@10으로 성능 평가
  - Inference
    - 전체 데이터에서 sequence length - 10개의 데이터 샘플링
    - 이후는 validation과 동일
  -	WandB를 사용하여 하이퍼파라미터 튜닝
  - 실혐 결과
    - Valid score, Public score, Private score가 비례하는 강건한 모델 구축
    - 유저, 아이템 정보를 추가해도 성능이 나아지지 않음
    - 파라미터 튜닝
      - max_len은 400에서 성능이 제일 좋았음
      - n_samples는 10에서 성능이 제일 좋았음
      - tail_raitio은 0.5나 1.0 에서 성능이 제일 좋았음


### 3. 성능 향상 (Recall@10)
- Public score 기준
  - 0.06(최초 구현) → 0.1236(모델 수정 및 튜닝)


## DKT(Deep Knowledge Tracing) (2023.05)
### 1. 대회 개요
-	부스트캠프 교육 과정 중 참여한 팀 대회
-	유저, 유저가 푼 수학 문제 및 관련 정보(정답 여부, 테스트 번호, 태그), 문제를 푼 시간이 제공
-	test 데이터에는 train 데이터에 등장하지 않은 유저의 문제 풀이 이력이 주어짐
-	test 데이터 유저가 푼 마지막 문제에 대한 정답 여부 예측하는 대회 (AUROC로 평가)

### 2. 역할 및 기여
- EDA와 Feature Engineering을 통해 유저와 문제의 그룹에 관련된 feature 생성
  - 유저의 그룹 정보는 팀에서 실험한 모든 모델(LightGCN, Boosting계열, Saint+)에 사용
- LightGCN 모델 구현 및 고도화
  - 모델 선정 이유
    - 유저와 아이템을 node로, 정답 여부를 edge로 간주하면 edge-prediction task와 유사
    -	GNN 계열 모델 중 구조가 단순하면서도, 괜찮은 성능을 보인다고 알려진 LightGCN 모델 선택
  - 사용 library
    - PyTorch, PyTroch Geometric
  - 제공된 baseline LightGCN 모델의 문제
    -	전체 train 데이터에서 단순히 랜덤하게 valid 데이터 분리해서 사용
  -	개선 방안
    - test 데이터와 동일하게 일정 비율의 유저 데이터 전부를 valid 데이터로 분리
    - train 데이터에 등장하지 않는 유저에 대해 예측해야 하는 cold start문제 발생
    - 신규 유저 임베딩은 해당 유저가 속한 그룹의 평균 임베딩 벡터값으로 초기화
    - 정답 여부가 알려진 신규 유저의 데이터를 사용해 모델 추가 학습
    - 추가 학습한 모델로 마지막 문제의 정답 여부 예측
-	Boosting 모델(LightGBM, CatBoost) 실험
  - 모델 선정 이유
    - tabular 데이터에서 좋은 성능을 낸다고 알려진 tree-based gradient boosting 계열 모델
  -	feature 30개 이상 생성 후 lofo-importance 패키지 사용하여 15개 선택
  - 팀원이 작성한 Hyperopt 패키지를 사용하는 코드로 하이퍼파라미터 튜닝
-	Saint+ 모델 실험 보조
  -	팀원이 구현한 Saint+ 모델에서 오류 수정
  -	유저의 그룹 정보를 input feature로 사용할 것을 제안

### 3. 성능 향상 (AUROC)
- Public score 기준
  - LightGCN : 0.6791 → 0.8058
  - LightGBM : 0.7345 → 0.7879
  -	Saint+ : 0.7857 → 0.8073


## Book Rating Prediction (2023.04)
### 1. 개요
- 부스트캠프 교육 과정 중 참여한 팀 대회
- 유저별 책의 평점, 유저 관련 정보(지역, 나이), 책 관련 정보(저자, ISBN, 카테고리 등)가 제공
- 유저가 책에 부여할 평점을 예측하는 대회 (RMSE로 평가)

### 2. 역할 및 기여
-	EDA와 Feature Engineering 코드를 작성해 팀원들이 공통적으로 사용
-	DeepFM 모델 구현
  - 모델 선정 이유
    -	주어진 유저와 책 정보를 활용할 수 있는 모델
  - 사용 library
    - PyTorch

### 3. 성능 향상 (RMSE)
-	Public score 기준
  - 제공된 여러 baseline모델들 : 2.3~2.5 → 2.2~2.3 (전처리 적용 후)
  -	DeepFM : 2.1822







