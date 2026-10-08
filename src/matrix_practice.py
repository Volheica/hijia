# -*- coding: utf-8 -*-
r"""
仿射矩阵填空练习:三种写法的对比

题目:把图片放大 1.5 倍,并向左移 200 像素。

运行:
    python src/matrix_practice.py

三种写法:
    A 你的答案   [[1.5,   1, -200], [  1, -1.5,   0]]
    B 正确写法   [[1.5,   0, -200], [  0,  1.5,   0]]   <- 以左上角为原点缩放
    C 以中心缩放 [[1.5,   0,   tx], [  0,  1.5,   ty]]   <- 图不会跑偏
"""

import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_FILE = os.path.join(PROJECT_ROOT, "data", "example1.jpg")
OUT_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

img = cv2.imread(IMAGE_FILE)
if img is None:
    raise SystemExit("读不到图片:" + IMAGE_FILE)

h, w = img.shape[:2]
SCALE = 1.5
SHIFT_X = -200

# ---------- 三种写法 ----------
M_A = np.float32([[1.5, 1, -200],
                  [1, -1.5, 0]])

M_B = np.float32([[SCALE, 0, SHIFT_X],
                  [0, SCALE, 0]])

# 以中心为基准缩放:线性部分后面再补一个补偿平移,让中心保持不动
tx_c = (w / 2) * (1 - SCALE) + SHIFT_X
ty_c = (h / 2) * (1 - SCALE)
M_C = np.float32([[SCALE, 0, tx_c],
                  [0, SCALE, ty_c]])

cases = [
    ("A 你的答案", M_A),
    ("B 正确(以左上角为原点)", M_B),
    ("C 以中心为基准 + 左移 200", M_C),
]

print("=" * 60)
print("目标:放大 1.5 倍,并向左移 200 像素")
print("=" * 60)

for name, M in cases:
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    print()
    print(name)
    print("  矩阵:")
    print("    [[%7.2f, %7.2f, %8.2f]," % (M[0, 0], M[0, 1], M[0, 2]))
    print("     [%7.2f, %7.2f, %8.2f]]" % (M[1, 0], M[1, 1], M[1, 2]))
    print("  两条公式:")
    print("    新 x = %.2f*x + %.2f*y + %.2f" % (M[0, 0], M[0, 1], M[0, 2]))
    print("    新 y = %.2f*x + %.2f*y + %.2f" % (M[1, 0], M[1, 1], M[1, 2]))
    print("  行列式 det = %.2f" % det, end="  ")
    if det < 0:
        print("<- 负数!说明图像被镜像翻转了")
    else:
        print("<- 正数,面积放大 %.2f 倍" % det)

# ---------- 四个角被搬到哪里 ----------
rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
corner_names = ["左上", "右上", "右下", "左下"]

print()
print("=" * 60)
print("四个角被搬到哪里(这才是矩阵最直观的读法)")
print("=" * 60)
for name, M in cases:
    homo = np.hstack([rect, np.ones((4, 1), dtype=np.float32)])
    moved = (M @ homo.T).T
    print()
    print(name)
    for i, cname in enumerate(corner_names):
        px, py = rect[i]
        mx, my = moved[i]
        print("  %s (%.0f, %.0f) -> (%.0f, %.0f)" % (cname, px, py, mx, my))

# ---------- 画出来 ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, (name, M) in zip(axes, cases):
    homo = np.hstack([rect, np.ones((4, 1), dtype=np.float32)])
    moved = (M @ homo.T).T

    before = np.vstack([rect, rect[:1]])
    ax.plot(before[:, 0], before[:, 1], "--", color="gray", linewidth=2, label="原位置")

    after = np.vstack([moved, moved[:1]])
    ax.plot(after[:, 0], after[:, 1], "-o", color="tab:red", linewidth=2,
            markersize=7, label="变换后")

    for (px, py), cname in zip(moved, corner_names):
        ax.annotate(cname, (px, py), textcoords="offset points", xytext=(9, -12),
                    color="tab:red", fontsize=11)

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(name)
    ax.legend(loc="best", fontsize=9)

fig.suptitle("放大 1.5 倍 + 左移 200:三种写法的四个角", fontsize=15)
fig.tight_layout()
corners_path = os.path.join(OUT_DIR, "matrix_practice_corners.png")
fig.savefig(corners_path, dpi=110)
plt.close(fig)
print()
print("已保存四角对比图:", corners_path)

# ---------- 真正把图变换出来看一眼 ----------
big_w, big_h = int(w * 2.2), int(h * 2.2)
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, (name, M) in zip(axes, cases):
    canvas = np.zeros((big_h, big_w, 3), dtype=np.uint8)
    canvas[:] = (40, 40, 40)
    warped = cv2.warpAffine(img, M, (big_w, big_h),
                            borderMode=cv2.BORDER_TRANSPARENT,
                            dst=canvas.copy())
    ax.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    ax.set_title(name)
    ax.axis("off")
fig.suptitle("实际变换结果(深灰区域是画布上多出来的部分)", fontsize=15)
fig.tight_layout()
warp_path = os.path.join(OUT_DIR, "matrix_practice_warped.png")
fig.savefig(warp_path, dpi=110)
plt.close(fig)
print("已保存实际效果图:", warp_path)
