# 文件名：run_roc.py
# 需求套件：pandas, numpy, scikit-learn, matplotlib, openpyxl 或 xlsxwriter
# 在 Replit/本機第一次執行可先安裝：
# pip install pandas numpy scikit-learn matplotlib openpyxl xlsxwriter

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

INPUT_FILE = "data.xlsx"
RESULTS_XLSX = "ROC_results.xlsx"
OVERVIEW_PNG = "ROC_overview.png"
SINGLE_PNG_TMPL = "ROC_{var}.png"


# -------------------------
# 小工具：將 GROUP 轉成 0/1
# -------------------------
def to_binary_group(series):
    """
    嘗試把各式寫法的 GROUP 轉為 0/1：
      - patient/case/disease -> 1
      - healthy/control/normal -> 0
      - 也接受 0/1、'0'/'1'
    若無法辨識，將以排序後的第二個當 1、第一個當 0，並提示。
    """
    s = series.copy()
    # 先嘗試數值化
    try:
        sn = pd.to_numeric(s, errors="coerce")
        if set(pd.unique(sn.dropna())) <= {0, 1} and sn.notna().any():
            return sn.astype(int), {0: "0", 1: "1"}
    except Exception:
        pass

    # 字串化處理
    low = s.astype(str).str.strip().str.lower()
    pos_tokens = {
        "patient", "case", "disease", "positive", "pos", "1", "yes", "true"
    }
    neg_tokens = {
        "healthy", "control", "normal", "negative", "neg", "0", "no", "false"
    }

    y = pd.Series(index=s.index, dtype="float64")

    # 先標準可辨識集合
    y[low.isin(pos_tokens)] = 1
    y[low.isin(neg_tokens)] = 0

    # 還有沒標到的，做最終對應
    mask_un = y.isna()
    if mask_un.any():
        uniq = list(pd.unique(low[mask_un]))
        if len(uniq) == 2:
            u_sorted = sorted(uniq)
            # 依字典序：前者 0，後者 1
            y[low == u_sorted[0]] = 0
            y[low == u_sorted[1]] = 1
            print(
                f"[INFO] 無法明確辨識 GROUP，依字典序對應：{u_sorted[0]}→0, {u_sorted[1]}→1")
        else:
            # 若真的無法處理，就直接報錯
            raise ValueError(f"無法辨識 GROUP：發現非標準的值 {uniq}。"
                             "請將 GROUP 改為 patient/healthy 或 1/0 再試。")

    mapping_preview = {0: "0/healthy-like", 1: "1/patient-like"}
    return y.astype(int), mapping_preview


# -------------------------
# ROC 與最佳切點（Youden's J）
# -------------------------
def compute_roc_metrics(y_true, scores):
    """
    回傳：
      - auc_val: AUC
      - fpr, tpr, thr: ROC 曲線座標與對應閾值（thr 依據 sklearn 定義）
      - flipped: 是否有做反向（AUC<0.5 時以 -scores 重新計算）
      - direction: 用於報告最佳切點的方向字串，'>=' 或 '<='（以原始變數值為基準）
      - best_thr_orig: 以原始變數值衡量的最佳切點數值
      - sens, spec: 在最佳切點處的敏感度/特異度
    """
    # 第一次計算
    auc_raw = roc_auc_score(y_true, scores)
    flipped = False
    use_scores = scores.copy()
    auc_val = auc_raw

    # 若 AUC < 0.5，自動反向
    if auc_raw < 0.5:
        flipped = True
        use_scores = -scores
        # 重新計算，保險起見直接再跑一遍而不是 1-auc
        auc_val = roc_auc_score(y_true, use_scores)

    fpr, tpr, thr = roc_curve(y_true, use_scores)
    youden_j = tpr - fpr
    idx = int(np.argmax(youden_j))

    # sklearn 的 thr 是針對 "use_scores" 的門檻
    chosen_thr_use = thr[idx]

    # 轉回「原始變數」的切點與方向
    # 若未翻轉：決策規則為 「原始變數 >= cutoff」 預測為正
    # 若已翻轉：因為用的是 -score，條件 -x >= thr ⇔ x <= -thr
    if not flipped:
        direction = ">="
        best_thr_orig = chosen_thr_use
    else:
        direction = "<="
        best_thr_orig = -chosen_thr_use

    sens = float(tpr[idx])
    spec = float(1 - fpr[idx])

    return {
        "auc": float(auc_val),
        "fpr": fpr,
        "tpr": tpr,
        "thr_use": thr,
        "flipped": flipped,
        "direction": direction,
        "cutoff": float(best_thr_orig),
        "sensitivity": sens,
        "specificity": spec,
    }


# -------------------------
# 主流程
# -------------------------
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] 找不到 {INPUT_FILE}，請把資料放到同一資料夾。")
        sys.exit(1)

    df = pd.read_excel(INPUT_FILE)
    if df.shape[1] < 7:
        print("[ERROR] data.xlsx 欄位不足。請確保第1欄為 GROUP，後面緊接 6 個變數。")
        sys.exit(1)

    group_col = df.columns[0]
    var_cols = list(df.columns[1:7])

    print(f"[INFO] 讀入資料，共 {len(df)} 列")
    print(f"[INFO] GROUP 欄位：{group_col}")
    print(f"[INFO] 變數欄位：{', '.join(map(str, var_cols))}")

    # 轉成 0/1
    y, mapping_info = to_binary_group(df[group_col])
    print(f"[INFO] GROUP 映射（預覽）：{mapping_info}")

    results = []

    # 總覽圖初始化
    plt.figure(figsize=(8, 7), dpi=150)
    plt.plot([0, 1], [0, 1],
             linestyle="--",
             linewidth=1,
             label="Chance (AUC=0.500)")

    for v in var_cols:
        x = pd.to_numeric(df[v], errors="coerce")
        # 與該變數相關的資料（去除 NA）
        mask = y.notna() & x.notna()
        yv = y[mask].astype(int).values
        xv = x[mask].values

        if len(np.unique(yv)) < 2:
            print(f"[WARN] 變數 {v} 的有效樣本中，GROUP 只有單一類別，略過。")
            continue

        try:
            m = compute_roc_metrics(yv, xv)
        except Exception as e:
            print(f"[ERROR] 計算 {v} ROC 失敗：{e}")
            continue

        # 單一變數 ROC 圖
        fig = plt.figure(figsize=(6, 5), dpi=150)
        plt.plot(m["fpr"],
                 m["tpr"],
                 linewidth=2,
                 label=f"{v} (AUC={m['auc']:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        plt.xlabel("1 - Specificity (FPR)")
        plt.ylabel("Sensitivity (TPR)")
        flip_note = " (auto-flip)" if m["flipped"] else ""
        plt.title(f"ROC — {v}{flip_note}")
        plt.legend(loc="lower right")
        out_png = SINGLE_PNG_TMPL.format(var=str(v).replace('/', '_'))
        plt.tight_layout()
        plt.savefig(out_png, bbox_inches="tight")
        plt.close(fig)

        # 放到總覽圖
        plt.plot(m["fpr"],
                 m["tpr"],
                 linewidth=1.8,
                 label=f"{v} (AUC={m['auc']:.3f})")

        # 保存結果列
        results.append({
            "Variable": v,
            "AUC": round(m["auc"], 4),
            "Direction": m["direction"],  # 報告門檻方向（以原始變數值決策）
            "Best_Cutoff": m["cutoff"],  # 最佳切點（原始變數值）
            "Sensitivity": round(m["sensitivity"], 4),
            "Specificity": round(m["specificity"], 4),
            "Auto_Flipped": "Yes" if m["flipped"] else "No"
        })

    # 輸出總覽圖
    plt.xlabel("1 - Specificity (FPR)")
    plt.ylabel("Sensitivity (TPR)")
    plt.title("ROC Overview (All Variables)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OVERVIEW_PNG, bbox_inches="tight")
    plt.close()

    # 匯出 Excel
    if len(results) == 0:
        print("[ERROR] 沒有可用結果，請檢查資料內容。")
        sys.exit(1)

    res_df = pd.DataFrame(results)
    # 排序：AUC 由高到低
    res_df = res_df.sort_values("AUC", ascending=False, ignore_index=True)

    try:
        with pd.ExcelWriter(RESULTS_XLSX, engine="xlsxwriter") as writer:
            res_df.to_excel(writer, index=False, sheet_name="ROC_Summary")
    except Exception:
        # 若 xlsxwriter 不在，退回 openpyxl
        with pd.ExcelWriter(RESULTS_XLSX, engine="openpyxl") as writer:
            res_df.to_excel(writer, index=False, sheet_name="ROC_Summary")

    print("\n[完成] 已輸出檔案：")
    print(f"  - {OVERVIEW_PNG}")
    for v in res_df['Variable']:
        print(f"  - {SINGLE_PNG_TMPL.format(var=str(v).replace('/', '_'))}")
    print(f"  - {RESULTS_XLSX}")

    print("\n[說明]")
    print("  • Direction 表示以『原始變數值』判定陽性的方向：")
    print("      - '>=': 變數值 ≥ Best_Cutoff 判為陽性")
    print("      - '<=': 變數值 ≤ Best_Cutoff 判為陽性（此時代表模型自動偵測到需反向）")
    print("  • Auto_Flipped = Yes 代表原始 AUC < 0.5，已自動反向分數再計算。")


if __name__ == "__main__":
    main()
