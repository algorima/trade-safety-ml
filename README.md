# trade-safety-ml (Custom ML classifier)

이 레포는 **LLM 기반 사기탐지(Trade Safety) 아키텍처의 뒷단**에 붙일 **custom ML classifier**(tabular+text)를 학습/저장/추론하는 코드입니다.

- 입력: Reddit 거래글 metadata + title/selftext (+ optional image presence)
- 출력: `risk_score` (0~1), `fraud/legit` 분류

> ✅ 데이터 포맷은 `라벨링_가이드.md`의 컬럼 설명을 그대로 사용합니다.

---

## 0) 빠른 시작 (Windows / PowerShell)

```powershell
cd C:\Users\chaehyun\Projects
git clone https://github.com/algorima/trade-safety-ml.git
cd trade-safety-ml

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 1) Whole dataset 만들기 (xlsx + zip 합치기)

### (1) xlsx/zip 파일들을 레포 루트에 복사
예:
- `drive-download-20251223T073611Z-3-001.zip`
- `drive-download-20251223T073624Z-3-001.zip`
- `labeling_채현님_20개_labeled.xlsx`
- (추가 xlsx들도 가능)

### (2) build 스크립트 실행
```powershell
python scripts\build_dataset.py ^
  --inputs drive-download-20251223T073611Z-3-001.zip drive-download-20251223T073624Z-3-001.zip labeling_채현님_20개_labeled.xlsx ^
  --out data\whole_dataset.csv
```

결과:
- `data/whole_dataset.csv` : 전체  합본 (라벨 비어있어도 포함)
- `data/labeled.csv` : `is_scam` 있는 행만
- `data/unlabeled.csv` : `is_scam` 비어있는 행만 (추가 라벨링 대상)

---

## 2) 학습 / 검증 / 테스트 split + 5겹 Stratified CV

- Supervised 학습에는 `data/labeled.csv`만 사용합니다.
- split: train/val/test = 0.7 / 0.15 / 0.15 (stratified)
- train split 내부에서 **Stratified 5-fold CV**로 하이퍼파라미터 탐색 후,
  val로 1차 확인, 마지막에 train+val로 재학습해서 최종 ckpt 저장합니다.

학습 실행:
```powershell
python scripts\train.py --csv data\labeled.csv --out models\custom_ml_ckpt --candidates xgb logreg --cv 5
```

출력 (ckpt):
- `models/custom_ml_ckpt/model.joblib` : 전체 파이프라인(전처리+모델) 저장
- `models/custom_ml_ckpt/metadata.json` : 모델/파라미터/학습정보
- `models/custom_ml_ckpt/metrics_val.json` : val 성능
- `models/custom_ml_ckpt/metrics_test.json` : test 성능
- `models/custom_ml_ckpt/splits/*.csv` : 재현용 split 저장

---

## 3) 평가만 다시 실행

```powershell
python scripts\evaluate.py --ckpt models\custom_ml_ckpt --csv data\labeled.csv --out outputs\eval_report.json
```

---

## 4) 추론(LLM 뒤에 붙이는 형태)

```powershell
python scripts\demo_infer.py --ckpt models\custom_ml_ckpt
```

코드에서 직접 호출:

```python
from trade_safety_ml.custom_ml import CustomML

ml = CustomML.load("models/custom_ml_ckpt")

post = {
  "title": "[WTS][USA] Twice Photocards - PayPal G&S only",
  "selftext": "Prices: $5 each. Timestamp included. Shipping with tracking.",
  "author_flair": "Trades: 52",
  "transaction_type": "WTS",
  "country": "USA",
  "flair": "Photocard",
  "score": 5,
  "comment_count": 2,
  "first_image_url": "https://i.redd.it/xxx.jpg",
  "is_gallery": True,
}
result = ml.analyze_post(post, threshold=0.5)
print(result)
```

---

## 5) 사용 feature (라벨링 가이드 기반)

`trade_safety_ml/features.py`와 `trade_safety_ml/model.py`에서 자동으로 다음을 생성합니다.

- Tabular
  - `trades_count` (author_flair에서 Trades: N 파싱)
  - `has_image` (is_gallery / first_image_url 기반)
  - `score`, `comment_count`
  - `transaction_type`, `country`, `flair` (one-hot)

- Text (title+selftext)
  - TF-IDF (1~2gram, max_features=5000)
  - keyword signals:
    - urgent/asap 등 급처
    - 외부 연락(WhatsApp/Telegram/KakaoTalk/Line)
    - 선입금 유도(payment first 등)
    - PayPal G&S, timestamp/proof, shipping/tracking
    - 가격($/usd) 존재

---

## 6) Git push (원격 덮어쓰기 OK)

> ⚠️ 원격 main을 강제로 덮어쓰려면 `--force`가 필요합니다.

```powershell
git add -A
git commit -m "Add custom ML pipeline (dataset merge + train/eval + ckpt)"
git push origin main --force
```

---

## 참고

- 데이터가 아직 작으면(라벨 적음) 성능이 불안정할 수 있습니다.
  이 구조는 **라벨이 늘어날수록**(특히 true/false가 충분히 쌓일수록) 안정적으로 좋아집니다.
