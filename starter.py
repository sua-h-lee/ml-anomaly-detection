"""
이상탐지 과제 시작 코드 (Starter Code)
=========================================

이 파일은 가장 단순한 baseline 파이프라인을 보여줍니다:
  1. 데이터 로드
  2. 전처리 (StandardScaler)
  3. Sliding window 변환
  4. 모델 학습 (Isolation Forest)
  5. test_public으로 평가 (AUROC, AUPR)

이 baseline을 출발점으로 삼아 본인의 모델/전처리로 발전시키세요.
어디를 수정하면 좋을지는 main 함수 안에 주석으로 표시되어 있습니다.

주의사항:
  - train.csv는 정상 데이터만 포함합니다 (label 컬럼 없음).
  - val.csv는 하이퍼파라미터 튜닝용입니다.
  - test_public.csv는 자체 성능 검증용입니다.
  - test_hidden_no_labels.csv는 최종 제출용이며 라벨이 없습니다.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler


# 데이터 디렉토리. 본인 환경에 맞게 수정하세요.
DATA_DIR = "./data"


# ============================================================
# 1. 데이터 로드
# ============================================================

def load_split(name, data_dir=DATA_DIR):
    """
    하나의 split CSV를 로드합니다.

    Parameters
    ----------
    name : str
        "train", "val", "test_public", "test_hidden_no_labels" 중 하나.
    data_dir : str
        CSV들이 있는 디렉토리 경로.

    Returns
    -------
    df : pd.DataFrame
        timestep + feature 컬럼 (label은 분리되어 있음)
    feature_cols : list[str]
        feature 컬럼 이름들 (x_로 시작하는 것들)
    labels : np.ndarray | None
        timestep별 라벨 (0=정상, 1=이상). 라벨이 없는 split이면 None.
    """
    path = os.path.join(data_dir, f"{name}.csv")
    raw = pd.read_csv(path)

    feature_cols = [c for c in raw.columns if c.startswith("x_")]

    if "label" in raw.columns:
        labels = raw["label"].to_numpy().astype(int)
        df = raw.drop(columns=["label"])
    else:
        labels = None
        df = raw

    return df, feature_cols, labels


# ============================================================
# 2. Sliding window
# ============================================================

def make_windows(values, window_size, stride=1):
    """
    시계열을 sliding window로 변환합니다.

    Parameters
    ----------
    values : np.ndarray
        shape (T,) 또는 (T, D)
    window_size : int
    stride : int

    Returns
    -------
    windows : np.ndarray
        - 입력이 (T,)면 출력은 (N, window_size)
        - 입력이 (T, D)면 출력은 (N, window_size, D)
        N = (T - window_size) // stride + 1
    """
    values = np.asarray(values)
    T = values.shape[0]
    if T < window_size:
        raise ValueError(f"입력 길이({T})가 window_size({window_size})보다 짧습니다.")

    n = (T - window_size) // stride + 1
    if values.ndim == 1:
        out = np.stack([values[i*stride : i*stride + window_size]
                        for i in range(n)])
    elif values.ndim == 2:
        out = np.stack([values[i*stride : i*stride + window_size, :]
                        for i in range(n)])
    else:
        raise ValueError(f"지원하지 않는 차원: {values.ndim}")
    return out


def windows_to_timestep_scores(window_scores, T, window_size, stride=1):
    """
    window별 score를 timestep별 score로 환산합니다.

    가장 단순한 방식: window의 score를 그 window의 마지막 timestep에 할당.
    첫 (window_size - 1) timestep은 첫 window의 score로 패딩.
    중간에 빈 timestep이 있으면 forward-fill로 채움.

    이 변환 방식은 baseline일 뿐입니다. 더 나은 방식 (예: window 중심에 할당,
    겹치는 window들의 평균 등)을 직접 구현해보세요.

    Parameters
    ----------
    window_scores : np.ndarray, shape (N,)
        각 window의 anomaly score
    T : int
        원래 시계열 길이
    window_size : int
    stride : int

    Returns
    -------
    timestep_scores : np.ndarray, shape (T,)
    
    timestep_scores = np.full(T, np.nan)
    n_windows = len(window_scores)

    # 각 window의 score를 그 window의 마지막 timestep에 할당
    for i in range(n_windows):
        end_idx = i * stride + window_size - 1
        timestep_scores[end_idx] = window_scores[i]

    # 앞쪽 패딩 (첫 window가 끝나기 전 구간)
    timestep_scores[:window_size - 1] = window_scores[0]

    # stride > 1인 경우 중간에 nan이 남을 수 있어 forward-fill
    for i in range(1, T):
        if np.isnan(timestep_scores[i]):
            timestep_scores[i] = timestep_scores[i - 1]

    return timestep_scores
    """
    timestep_scores_sum = np.zeros(T)
    counts = np.zeros(T)
    n_windows = len(window_scores)

    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        
        # T를 넘어가는 예외 방지 
        if end_idx > T:
            end_idx = T

        
        # 해당 윈도우가 커버하는 모든 타임스텝에 점수를 누적
        timestep_scores_sum[start_idx:end_idx] += window_scores[i]
        counts[start_idx:end_idx] += 1

    # 윈도우가 한 번도 지나가지 않은 곳(주로 맨 뒤 극소수)은 안전하게 처리
    counts[counts == 0] = 1
    
    # 평균값 계산
    timestep_scores = timestep_scores_sum / counts
    
    return timestep_scores
    


# ============================================================
# 3. Baseline 파이프라인
# ============================================================

if __name__ == "__main__":
    # ---------- 데이터 로드 ----------
    train_df, feature_cols, _ = load_split("train")
    val_df,   _, val_labels   = load_split("val")
    test_df,  _, test_labels  = load_split("test_public")

    print("=== 데이터 형태 ===")
    print(f"train:        {train_df.shape}, anomaly=없음 (정상만)")
    print(f"val:          {val_df.shape}, anomaly={val_labels.sum()}개 timestep")
    print(f"test_public:  {test_df.shape}, anomaly={test_labels.sum()}개 timestep")
    print(f"feature_cols: {feature_cols}")
    print()



    # ---------- 전처리: 스케일링 ----------
    # train으로만 fit, val/test에는 transform만 적용 (data leakage 방지)
    # ※ 개선 포인트: 연속형/이산형을 분리해서 다르게 처리, RobustScaler 시도, 등
    
    # 이산데이터와 연속 데이터 분리
    # x_f8을 리스트에서 제외
    binary_cols = ['x_06', 'x_92', 'x_4b']
    continuous_cols = [c for c in feature_cols if c not in binary_cols + ['x_f8']]

    scaler_std    = StandardScaler()
    robust_scaler = RobustScaler(quantile_range=(10.0, 90.0))
    minmax_scaler = MinMaxScaler()

    # 1. Train 데이터: Robust로 먼저 깎고, 이어서 MinMax로 0~1 사이로 고정합니다.
    X_train_cont = robust_scaler.fit_transform(train_df[continuous_cols])
    X_train_cont = minmax_scaler.fit_transform(X_train_cont) # 연속으로 fit_transform 수행
    X_train_bin = train_df[binary_cols].to_numpy()
    X_train = np.hstack([X_train_cont, X_train_bin])

    # 2. Val 데이터: 데이터 누수 방지를 위해 transform만 연달아 적용합니다.
    X_val_cont = robust_scaler.transform(val_df[continuous_cols])
    X_val_cont = minmax_scaler.transform(X_val_cont)
    X_val_bin = val_df[binary_cols].to_numpy()
    X_val = np.hstack([X_val_cont, X_val_bin])

    # 3. Test 데이터: 동일하게 적용합니다.
    X_test_cont = robust_scaler.transform(test_df[continuous_cols])
    X_test_cont = minmax_scaler.transform(X_test_cont)
    X_test_bin = test_df[binary_cols].to_numpy()
    X_test = np.hstack([X_test_cont, X_test_bin])

    

    # ---------- Sliding window ----------
    # ※ 개선 포인트: window 크기 튜닝, 통계 피처(mean/std/min/max) 추출, 등
    W = 180   # window 크기
    S_train = 33 #train은 학습속도 향상을 위해 staride를 따로 배정
    S = 1    # stride for val/test

    train_windows = make_windows(X_train, W, S_train)  # (N, W, D)
    val_windows   = make_windows(X_val,   W, S)
    test_windows  = make_windows(X_test,  W, S)

    #평균값, 표준편차, 최소값, 최대값, 범위, 중앙값, 시작과 끝의 차이 등 다양한 통계량을 추출하여 모델에 제공
    def extract_features(windows):
        mean   = windows.mean(axis=1)
        std    = windows.std(axis=1)
        min_   = windows.min(axis=1)
        max_   = windows.max(axis=1)
        range_ = max_ - min_
        median = np.median(windows, axis=1)
        diff   = windows[:, -1, :] - windows[:, 0, :]
        return np.concatenate([mean, std, min_, max_, range_, median, diff], axis=1)
    
    train_X = extract_features(train_windows)
    val_X   = extract_features(val_windows)
    test_X  = extract_features(test_windows)
    
    """
    # IsolationForest는 1D 입력을 기대하므로 (N, W, D) -> (N, W*D)로 flatten
    # ※ 개선 포인트: flatten 대신 window별 통계량 추출이 더 나을 수 있음
    #train 데이터에서 통계 피처 추출
    train_flat = train_windows.reshape(len(train_windows), -1)
    val_flat   = val_windows.reshape(len(val_windows), -1)
    test_flat  = test_windows.reshape(len(test_windows), -1)


    # 2. 우리가 만든 거시적인 통계 피처 만들기 (평균과 표준편차사용)
    # 이때 이산 데이터는 제거
    train_mean = np.mean(train_windows[:, :, :7], axis=1) # (N, 7)
    train_std  = np.std(train_windows[:, :, :7], axis=1)  # (N, 7)
    train_stat = np.hstack([train_mean, train_std])       # (N, 14)로 대폭 압축!

    val_mean = np.mean(val_windows[:, :, :7], axis=1)
    val_std  = np.std(val_windows[:, :, :7], axis=1)
    val_stat = np.hstack([val_mean, val_std])

    test_mean = np.mean(test_windows[:, :, :7], axis=1)
    test_std  = np.std(test_windows[:, :, :7], axis=1)
    test_stat = np.hstack([test_mean, test_std])

    # 3. 노이즈가 제거된 알짜배기 통계량만 날것의 데이터 뒤에 결합
    train_X = np.hstack([train_flat, train_stat])
    val_X   = np.hstack([val_flat, val_stat])
    test_X  = np.hstack([test_flat, test_stat])
    """

    print(f"=== Sliding window (W={W}, stride={S}) ===")
    print(f"train_X: {train_X.shape}")
    print(f"val_X:   {val_X.shape}")
    print(f"test_X:  {test_X.shape}")

    # ---------- 모델 학습: Isolation Forest ----------
    # ※ 개선 포인트:
    #   - 다른 모델 시도 (One-Class SVM, LOF, GMM, PCA-based 등)
    #   - n_estimators, max_samples, max_features 등 HP 튜닝 (val로)
    print("=== Isolation Forest 학습 중 ===")
    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_X)
    print("학습 완료")

    # ---------- score 계산 및 [AUPR 향상 롤링 필터 적용] ----------
    # IsolationForest.score_samples는 "정상일수록 큰 값"을 반환하므로 부호를 뒤집음 (-)
    val_window_scores  = -model.score_samples(val_X)
    test_window_scores = -model.score_samples(test_X)

    # [★AUPR 치트키] 윈도우 스코어 자체의 순간적인 미세 잡음(오보) 제거하기
    # window=3 이나 window=5 정도로 조절해가며 성능을 볼 수 있습니다. 우선 3으로 시작합니다!
    val_window_scores = pd.Series(val_window_scores).rolling(window=3, min_periods=1).mean().to_numpy()
    test_window_scores = pd.Series(test_window_scores).rolling(window=3, min_periods=1).mean().to_numpy()

    # window score → timestep score 환산 (이 부분은 원래 스타터 코드 그대로입니다)
    val_scores  = windows_to_timestep_scores(val_window_scores,  len(val_df),  W, S)
    test_scores = windows_to_timestep_scores(test_window_scores, len(test_df), W, S)
    # ---------- 평가 ----------
    val_auroc  = roc_auc_score(val_labels,  val_scores)
    val_aupr   = average_precision_score(val_labels,  val_scores)
    test_auroc = roc_auc_score(test_labels, test_scores)
    test_aupr  = average_precision_score(test_labels, test_scores)

    print("=== Baseline 성능 ===")
    print(f"{'':12s} {'AUROC':>8s} {'AUPR':>8s}")
    print(f"{'val':12s} {val_auroc:>8.4f} {val_aupr:>8.4f}")
    print(f"{'test_public':12s} {test_auroc:>8.4f} {test_aupr:>8.4f}")
    print()

    # ============================================================
    # raw timestep IF 멀티-시드 + 멀티-스케일 smoothing 앙상블
    # ============================================================

    # 윈도우 통계 피처는 길이 1~2짜리 point anomaly를 평균에 묻어버려서,
    # raw timestep에 IF 한 번 더 돌림. seed 15개 평균으로 점수 안정화.
    seeds = [42, 0, 1, 7, 100, 222, 999, 31, 256, 1024,
             17, 333, 555, 777, 8888]
    raw_val  = np.zeros(len(val_df))
    raw_test = np.zeros(len(test_df))
    for sd in seeds:
        m = IsolationForest(n_estimators=300, max_samples=0.8,
                            random_state=sd, n_jobs=-1).fit(X_train)
        raw_val  += -m.score_samples(X_val)
        raw_test += -m.score_samples(X_test)
    raw_val  /= len(seeds)
    raw_test /= len(seeds)

    # 양방향 이동평균 평활화. smoothing 윈도우 하나에 의존하면 그 크기에 맞는
    # anomaly만 잘 잡으니까 여러 크기로 다 만들어서 평균.
    def to_rank(x):
        return pd.Series(x).rank(pct=True).to_numpy()

    def smooth(s, w):
        pad = w // 2
        return np.convolve(np.pad(s, pad, mode='edge'),
                           np.ones(w)/w, mode='same')[pad:pad + len(s)]

    smooth_windows = [51, 101, 151, 201, 251, 301, 351, 401,
                      451, 501, 601, 701, 801]

    # 각 smoothing 결과를 rank로 바꿔서 평균 (점수 단위 무관하게 합치려고)
    final_val  = np.mean([to_rank(smooth(raw_val,  w)) for w in smooth_windows], axis=0)
    final_test = np.mean([to_rank(smooth(raw_test, w)) for w in smooth_windows], axis=0)

    final_val_auroc  = roc_auc_score(val_labels,  final_val)
    final_val_aupr   = average_precision_score(val_labels,  final_val)
    final_test_auroc = roc_auc_score(test_labels, final_test)
    final_test_aupr  = average_precision_score(test_labels, final_test)

    print("=== New Method 성능 ===")
    print(f"{'':12s} {'AUROC':>8s} {'AUPR':>8s}")
    print(f"{'val':12s} {final_val_auroc:>8.4f} {final_val_aupr:>8.4f}")
    print(f"{'test_public':12s} {final_test_auroc:>8.4f} {final_test_aupr:>8.4f}")
    print()
    # =========================================================
    # - test_hidden_no_labels.csv에 대한 anomaly score 생성
    # - (t, score) 두 컬럼의 CSV로 저장하여 제출
    # =========================================================