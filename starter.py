# 기계학습 이상탐지 과제

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
# 2. New Method 파이프라인
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

    robust_scaler = RobustScaler(quantile_range=(10.0, 90.0))

    # 1. Train 데이터: Robust사용
    X_train_cont = robust_scaler.fit_transform(train_df[continuous_cols])
    X_train_bin = train_df[binary_cols].to_numpy()
    X_train = np.hstack([X_train_cont, X_train_bin])

    # 2. Val 데이터: 데이터 누수 방지를 위해 transform만 연달아 적용합니다.
    X_val_cont = robust_scaler.transform(val_df[continuous_cols])
    X_val_bin = val_df[binary_cols].to_numpy()
    X_val = np.hstack([X_val_cont, X_val_bin])

    # 3. Test 데이터: 동일하게 적용합니다.
    X_test_cont = robust_scaler.transform(test_df[continuous_cols])
    X_test_bin = test_df[binary_cols].to_numpy()
    X_test = np.hstack([X_test_cont, X_test_bin])


    # ---------- raw timestep IF 멀티-시드 + 멀티-스케일 smoothing 앙상블 ---------- 
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