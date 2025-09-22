# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # ✅ 無頭環境：只存檔，不彈視窗
import matplotlib.pyplot as plt

excel_path = os.path.join("CE16265B", "ppp333.xlsx")
sheet_name = "一年前後P值總結"
out_dir = "outputs"
out_png = os.path.join(out_dir, "boxplot.png")

def log(msg):
    print(msg, flush=True)

try:
    log(f"📄 讀檔：{excel_path}")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"找不到檔案：{excel_path}")

    # 讀檔
    df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    log(f"✅ 已讀取工作表：{sheet_name}")

    # 清空列/欄 & 修剪欄名空白
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]

    # 第一欄當索引（mean/SD/minimum/IQR1/median/IQR3/maximum/P value）
    row_label_col = df.columns[0]
    df[row_label_col] = df[row_label_col].astype(str).str.strip()
    df = df.set_index(row_label_col)

    # 小工具：容忍空白/大小寫差異
    def pick_name(candidates, target):
        t = target.replace(" ", "").lower()
        for x in candidates:
            if str(x).replace(" ", "").lower() == t:
                return x
        raise KeyError(f"找不到「{target}」，實際名稱：{list(candidates)}")

    # 取欄/列名
    col_baseline = pick_name(df.columns, "Baseline")
    col_year1    = pick_name(df.columns, "1 year")

    r_min  = pick_name(df.index, "minimum")
    r_q1   = pick_name(df.index, "IQR1")
    r_med  = pick_name(df.index, "median")
    r_q3   = pick_name(df.index, "IQR3")
    r_max  = pick_name(df.index, "maximum")
    r_pval = pick_name(df.index, "P value")

    # 抽數值
    baseline = dict(
        min=float(df.at[r_min, col_baseline]),
        q1=float(df.at[r_q1, col_baseline]),
        median=float(df.at[r_med, col_baseline]),
        q3=float(df.at[r_q3, col_baseline]),
        max=float(df.at[r_max, col_baseline]),
    )
    year1 = dict(
        min=float(df.at[r_min, col_year1]),
        q1=float(df.at[r_q1, col_year1]),
        median=float(df.at[r_med, col_year1]),
        q3=float(df.at[r_q3, col_year1]),
        max=float(df.at[r_max, col_year1]),
    )
    p_value = float(df.at[r_pval, col_year1])
    log(f"🔢 Baseline: {baseline}")
    log(f"🔢 1 year  : {year1}")
    log(f"🔢 p-value : {p_value}")

    # 盒鬚圖 stats
    stats = [
        dict(whislo=baseline["min"], q1=baseline["q1"], med=baseline["median"],
             q3=baseline["q3"], whishi=baseline["max"], fliers=[]),
        dict(whislo=year1["min"], q1=year1["q1"], med=year1["median"],
             q3=year1["q3"], whishi=year1["max"], fliers=[]),
    ]

    log("🎨 開始繪圖…")
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#AAB8AB", "#8E9EAB"]  # Baseline / 1 year

    bp = ax.bxp(stats, showfliers=False, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
    for line in bp["whiskers"] + bp["caps"]:
        line.set_color("black")
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(2)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Baseline", "1 year"])
    ax.set_ylabel("Serum creatinine (mg/dL)")

    # 淺灰橫線
    ax.yaxis.grid(True, color="#cccccc", linestyle="-", linewidth=0.75, alpha=0.6)
    ax.set_axisbelow(True)

    # 讓 min 不貼 X 軸
    gmin = min(baseline["min"], year1["min"])
    gmax = max(baseline["max"], year1["max"])
    pad = max(0.05, (gmax - gmin) * 0.05)
    ax.set_ylim(gmin - pad, gmax + pad)

    # p 值 → 只顯示 *
    star = "*" if p_value <= 0.05 else ""
    ax.text(1.5, gmax + pad * 0.6, star, ha="center", va="bottom", fontsize=16)

    plt.tight_layout()

    # 存檔
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()
    log(f"✅ 盒鬚圖已輸出：{out_png}")

except Exception as e:
    print("❌ 發生錯誤：", e, file=sys.stderr)
    # 若要更詳細可加：import traceback; traceback.print_exc()
    sys.exit(1)
