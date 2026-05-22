# 🎯 LOTTO & Pension Lottery Prediction Project

> **"머신러닝은 완벽한 무작위성(Randomness)의 벽을 넘을 수 있을까?"**
> 로또(비복원 추출)와 연금복권(복원 추출)의 수학적 특성을 비교 분석하고, 머신러닝과 통계적 모델링을 통해 무작위성 속의 패턴을 탐구하는 데이터 분석 프로젝트입니다.

## 📌 Project Overview
본 프로젝트는 단순한 '번호 찍기'가 아닌, 철저한 통계적 검증과 머신러닝 파이프라인(XGBoost, LSTM, Optuna, TSCV)을 통해 복권 데이터의 본질적 한계와 가능성을 실험한 기록입니다.

- **Lotto 6/45 (비복원 추출):** 과거 데이터의 시계열 모멘텀(Momentum)과 패턴을 분석하여 통계적 기댓값(0.8)을 상회하는 예측 모델 구축.
- **Pension 720+ (복원 추출):** 완벽한 독립동일분포(i.i.d)를 띠는 데이터에서, 개별 숫자가 아닌 '거시적 형태(Macro Properties)'를 예측하는 방향으로의 관점 전환 및 마르코프 연쇄(Markov Chain)를 통한 기계적 편향성 검증.

## 📂 Repository Structure
```text
LOTTO_PROJECT/
├── data/                  # 원본 CSV 데이터 (lotto.csv, pension.csv)
├── lotto/                 # 로또 6/45 머신러닝 실험 (Phase 1 ~ 7)
│   ├── lotto_phase5.py    # 🏆 [Champion] XGBoost + Optuna + TSCV 모델
│   └── experiment_log.tsv # 실험 결과 로그
└── pension/               # 연금복권 통계/분석 실험 (Phase 1 ~ 3)
    ├── pension_phase2.py  # 🏆 [Champion] 거시적 패턴(Sum, Odd, High) 예측 모델
    └── pension_phase3.py  # 마르코프 연쇄(Markov Chain) 전이 확률 분석


##🚀 Key Experiments & Results
Part 1: 로또 6/45 (Lotto) - 시계열 모멘텀의 발견
Phase 1~3 (Baseline & Feature Engineering): 단순 빈도와 인간의 직관(총합/홀짝 비율)을 강제한 모델은 오히려 성능이 저하됨을 확인.

Phase 4~5 (XGBoost + Optuna + TSCV): 각 번호의 시계열적 '생체 리듬(HistAvgGap)'과 '단기 모멘텀(Recency)' 특징 12가지를 추출.

##🏆 최종 결과: 5-fold TimeSeriesSplit 교차 검증 결과, 평균 적중률(CV Hit Rate) 0.8567을 달성하며 순수 무작위 기댓값(0.8)을 수학적으로 돌파. (SHAP 분석을 통해 Gap과 Recency의 중요도 증명)

Phase 6 (LSTM): 딥러닝 모델 적용 시도, 데이터 부족(1,200건)으로 인한 Underfitting 현상 증명.

Part 2: 연금복권 (Pension) - 무작위성의 통계적 증명
Phase 1 (개별 번호 예측): 각 자리 독립 복원 추출 특성으로 인해 머신러닝의 예측력이 수학적 기댓값(0.1)에 수렴함을 확인.

Phase 2 (거시적 패턴 예측): 개별 숫자가 아닌 6개 숫자의 '총합(Sum)', 'High 숫자 개수' 등 거시적 타겟으로 전환. 중심극한정리를 활용해 기준점 대비 최대 2.9배 높은 예측 정확도(Accuracy) 달성.

Phase 3 (Markov Chain): 카이제곱 검정(Chi-Square Test)을 통해 모든 전이 확률 행렬의 p-value > 0.05임을 확인. 대한민국 연금복권 추첨기에 어떠한 물리적 결함이나 메모리 효과(Memory Effect)가 없음을 완벽히 증명.

##🛠️ Tech Stack
Data Science: Python, Pandas, NumPy

Machine Learning & DL: XGBoost, Scikit-learn, PyTorch

Optimization & Explainability: Optuna, SHAP
