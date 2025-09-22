import matplotlib.pyplot as plt
import os
from datetime import datetime

# 統計數據
mean = 1.21
sd = 0.330606
minimum = 0.86
q1 = 1.00
median = 1.13
q3 = 1.23
maximum = 1.92

# 🔹 設定字型
plt.rcParams["font.family"] = "Times New Roman"

# 建立輸出資料夾（假設在專案根目錄）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
output_dir = os.path.join(project_root, "outputs")
os.makedirs(output_dir, exist_ok=True)

# 畫箱型圖
fig, ax = plt.subplots(figsize=(6, 6))

box_data = {
    'whislo': minimum,
    'q1': q1,
    'med': median,
    'q3': q3,
    'whishi': maximum,
    'fliers': []
}

# 🔹 畫箱型圖（啟用填色）
bp = ax.bxp(
    [box_data],
    showfliers=False,
    boxprops=dict(color="black"),   # 邊框黑色
    whiskerprops=dict(color="black"),
    capprops=dict(color="black"),
    medianprops=dict(color="black", linewidth=2.5),
    patch_artist=True
)

# 🔹 填色：灰藍色
for patch in bp['boxes']:
    patch.set_facecolor("#6A7BA2")

# 軸標題
ax.set_ylabel("Time to renal function decline (years)")
ax.set_title("Distribution of renal function worsening after biologics")

# 平均值 → 小黑點
ax.scatter(1, mean, color="black", marker="o", s=40, label=f"Mean = {mean:.2f}")

# 背景 grid：淺灰細虛線
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="lightgray")
ax.xaxis.grid(False)

# 留空間：讓 min 不會壓到 x 軸
ax.set_ylim(minimum - 0.1, maximum + 0.1)

# 圖例
ax.legend()

# 產生日期時間戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 存檔
output_path = os.path.join(output_dir, f"renal_decline_boxplot_{timestamp}.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.close()
print(f"圖表已輸出到：{output_path}")
