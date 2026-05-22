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
