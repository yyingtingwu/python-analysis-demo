# -*- coding: utf-8 -*-
"""
Logistic Regression 一鍵版（含進度輸出）
資料：CF16265B/CF16265B_映霆.(xlsx|xls)；工作表：Logistic回歸；範圍：A1:E224（第一欄為二元 y）
輸出：outputs/ 下列檔案
- logit_coef_OR.csv（coef, OR, 95%CI, p）
- model_summary.txt（statsmodels 摘要 + AUC + 報表 + 混淆矩陣）
- confusion_matrix.csv
- predictions.csv（y_true, y_prob, y_pred）
- roc_curve.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# =============== 使用者設定 ===============
DATA_DIR   = "CF16265B"
BASENAME   = "CF16265B_映霆"      # 不含副檔名；會自動嘗試 .xlsx / .xls
SHEET_NAME = "Logistic回歸"
USECOLS    = [0, 1, 2, 3, 4]      # A:E 用位置避免 Unnamed 問題
NROWS      = 224                  # 含標題
THRESH     = 0.5                  # 分類門檻
OUTPUT_DIR = "outputs"
# ========================================

def find_excel_path():
    print("[STEP] 檢查 Excel 檔案…")
    for ext in (".xlsx", ".xls"):
        p = os.path.join(DATA_DIR, BASENAME + ext)
        if os.path.exists(p):
            print(f"[OK] 找到：{p}")
            return p
    raise FileNotFoundError(f"找不到 {DATA_DIR}/{BASENAME}.xlsx 或 .xls，請確認檔名與位置。")

def read_clean_df(path):
    print(f"[STEP] 讀取工作表：{SHEET_NAME}（只抓 A:E、前 {NROWS} 列）")
    engine = "openpyxl" if path.lower().endswith(".xlsx") else None
    df = pd.read_excel(path, sheet_name=SHEET_NAME, usecols=USECOLS, nrows=NROWS, engine=engine)

    # 丟掉全空列/全空欄
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    # 欄名去空白
    df.columns = [str(c).strip() for c in df.columns]
    # 濾掉 Unnamed 欄（用 Index 的布林遮罩，避免對齊錯誤）
    cols = pd.Index(df.columns).astype(str)
    mask = ~cols.str.startswith("Unnamed")
    df = df.loc[:, mask]

    print(f"[CHECK] 目前資料形狀：{df.shape}")
    print(f"[CHECK] 欄名：{list(df.columns)}")
    if df.shape[1] < 2:
        raise ValueError("有效欄位少於 2（需要 1 個 y + 至少 1 個 X）。請確認 A:E 內有資料與標題。")
    return df

def coerce_binary_y(s):
    print("[STEP] 轉換 y 為二元 (0/1)…（支援 是/否、Y/N、Yes/No、陽性/陰性 等）")
    mapping = {
        "是":1, "否":0, "YES":1, "NO":0, "Y":1, "N":0,
        "陽性":1, "陰性":0, "POSITIVE":1, "NEGATIVE":0, "1":1, "0":0
    }
    s2 = s.astype(str).str.strip().str.upper().replace(mapping)
    y = pd.to_numeric(s2, errors="coerce")
    uniq = set(y.dropna().unique())
    print(f"[CHECK] y 唯一值：{sorted(list(uniq))}")
    if not uniq.issubset({0,1}):
        raise ValueError("第一欄 y 不是二元(0/1/是/否...)，請把 y 放在第一欄且為二元。")
    return y.astype(int)

def main():
    try:
        excel_path = find_excel_path()
        df = read_clean_df(excel_path)
    except Exception as e:
        print("[ERROR] 讀檔/清理失敗：", e)
        sys.exit(1)

    # y 與 X
    y_col = df.columns[0]
    print(f"[STEP] 設定目標變數 y = 第一欄：{y_col}")
    try:
        y = coerce_binary_y(df[y_col])
    except Exception as e:
        print("[ERROR] y 轉二元失敗：", e)
        sys.exit(1)

    print("[STEP] 只保留數值型 X 欄位…")
    X = df.drop(columns=[y_col]).apply(pd.to_numeric, errors="coerce")
    X = X.select_dtypes(include=[np.number])

    data = pd.concat([y, X], axis=1).dropna()
    if data.shape[1] < 2:
        print("[ERROR] 沒有可用的數值自變數 X。請確認 A:E 至少有 1 欄為數值。")
        sys.exit(1)

    y = data.iloc[:, 0].astype(int)
    X = data.iloc[:, 1:].copy()
    print(f"[CHECK] 最終 X 欄位：{list(X.columns)}（共 {X.shape[1]} 欄）")
    print(f"[CHECK] 可用樣本數：{X.shape[0]}")

    # 標準化
    print("[STEP] 標準化 X…")
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Logistic
    print("[STEP] 擬合 Logistic 模型（statsmodels.Logit）…")
    X_sm = sm.add_constant(Xs, prepend=True)
    try:
        res = sm.Logit(y, X_sm).fit(disp=False)
        print("[OK] 模型擬合完成。")
    except Exception as e:
        print("[ERROR] 模型擬合失敗：", e)
        sys.exit(1)

    # 係數 → OR / CI / p
    print("[STEP] 計算 OR、95% CI 與 p 值…")
    params = res.params
    conf = res.conf_int()
    conf.columns = ["2.5%", "97.5%"]
    table = pd.DataFrame({
        "coef": params,
        "OR": np.exp(params),
        "CI_low(95%)": np.exp(conf["2.5%"]),
        "CI_high(95%)": np.exp(conf["97.5%"]),
        "p_value": res.pvalues
    })
    const = table.loc[["const"]]
    others = table.drop(index=["const"], errors="ignore").sort_values("p_value")
    out_table = pd.concat([const, others])

    # 預測 / ROC / 混淆
    print("[STEP] 產生預測、ROC 與混淆矩陣…")
    y_prob = res.predict(X_sm)
    auc = roc_auc_score(y, y_prob)
    fpr, tpr, _ = roc_curve(y, y_prob)
    y_pred = (y_prob >= THRESH).astype(int)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, digits=3)

    # 輸出
    print("[STEP] 輸出結果到 outputs/ …")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    out_table.to_csv(os.path.join(OUTPUT_DIR, "logit_coef_OR.csv"), encoding="utf-8-sig")

    with open(os.path.join(OUTPUT_DIR, "model_summary.txt"), "w", encoding="utf-8") as f:
        f.write(res.summary2().as_text())
        f.write("\n\n---\n")
        f.write(f"AUC = {auc:.4f}\n")
        f.write(f"THRESH = {THRESH:.2f}\n\n")
        f.write("[Classification Report]\n")
        f.write(report)
        f.write("\n[Confusion Matrix]\n")
        f.write(str(cm))

    pd.DataFrame(cm, index=["Actual_0","Actual_1"], columns=["Pred_0","Pred_1"])\
      .to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"), encoding="utf-8-sig")

    pd.DataFrame({"y_true": y, "y_prob": y_prob, f"y_pred(TH={THRESH:.2f})": y_pred})\
      .to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False, encoding="utf-8-sig")

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Logistic Regression)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150)
    plt.close()

    print("\n========== 完成 ✅ ==========")
    print(f"AUC = {auc:.4f}｜THRESH = {THRESH:.2f}｜y 欄位 = {y_col}")
    print("輸出檔案：")
    print("- outputs/logit_coef_OR.csv")
    print("- outputs/model_summary.txt")
    print("- outputs/confusion_matrix.csv")
    print("- outputs/predictions.csv")
    print("- outputs/roc_curve.png")

if __name__ == "__main__":
    main()
