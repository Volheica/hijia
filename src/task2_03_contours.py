# -*- coding: utf-8 -*-
r"""
task-2 题三:检测 example1.jpg 的轮廓,保存结果并计算总面积、总周长

【在 VSCode 里怎么用】每个 "# %%" 是一个单元格,点 Run Cell 逐格运行
【整段运行】python src/task2_03_contours.py

这一题的流程是一条"流水线",每一步都为下一步服务:

    读图 -> 灰度 -> 去噪 -> 找边缘 -> 把断开的边缘连起来 -> 找轮廓 -> 算面积周长

【建议】先猜每一格在干什么,再运行;最后五个实验先写预测再跑。
"""

# %% [markdown]
# ## 第 1 格:配置

# %% 配置区
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


def to_rgb(image):
    """matplotlib 要 RGB,OpenCV 读进来是 BGR,显示前转一次。"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


print("输入图片:", IMAGE_FILE)
print("输出目录:", OUT_DIR)

# %% 读图
img = cv2.imread(IMAGE_FILE)
if img is None:
    raise SystemExit("读不到图片:" + IMAGE_FILE)

h, w = img.shape[:2]
print("图片尺寸:高 %d,宽 %d" % (h, w))

# %% [markdown]
# ## 第 2 格:先看看这张图能不能"直接二值化"
#
# 找轮廓的前提是**二值图**(只有黑和白)。最直觉的做法是设一个灰度阈值,
# 把亮的当物体、暗的当背景。但这张图是"白飞机 + 蓝天 + 云",
# 天空的灰度和飞机机身其实很接近 —— 直接切一刀,飞机和天空会被切到一起去。
#
# **先写下你的预测**:直接 `threshold` 之后,二值图里能看到一架完整的飞机吗?

# %% 灰度图 + 直方图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("灰度值的分布:")
print("  最小值 %d,最大值 %d,平均值 %.1f" % (gray.min(), gray.max(), gray.mean()))
hist = np.bincount(gray.ravel(), minlength=256)
print("  亮度在 100-200 之间的像素占 %.1f%%" % (hist[100:200].sum() / gray.size * 100))

plt.figure(figsize=(10, 3.5))
plt.plot(hist, color="tab:blue")
plt.title("灰度直方图:如果只在某个位置切一刀,飞机和天空很难分开")
plt.xlabel("灰度值")
plt.ylabel("像素个数")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "task2_03_histogram.png"), dpi=110)
plt.close()

# %% [markdown]
# ## 第 3 格:去噪
# 直接对原图做边缘检测,云朵的纹理也会被当成边缘。
# 高斯模糊**抹掉细节、保留大结构**,让后面的边缘检测只关注明显的边界。

# %% 高斯模糊
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
print("原图灰度标准差   : %.2f" % gray.std())
print("模糊后灰度标准差 : %.2f  <- 变小说明细节被抹平了" % blurred.std())

# %% [markdown]
# ## 第 4 格:Canny 边缘检测
#
# `cv2.Canny(img, low, high)` 的两个阈值是一对"门限":
#
# - 梯度**大于 high** 的像素:确定是边缘,保留
# - 梯度**小于 low** 的像素:肯定不是边缘,丢掉
# - 介于两者之间:看它是否和"确定的边缘"连在一起,连着就留
#
# 经验比例是 **1 : 2 到 1 : 3**,所以 50 / 150 是常见组合。

# %% Canny
edges = cv2.Canny(blurred, 50, 150)
print("边缘像素占全图的比例: %.2f%%" % (np.count_nonzero(edges) / edges.size * 100))
cv2.imwrite(os.path.join(OUT_DIR, "task2_03_edges.png"), edges)

# %% [markdown]
# ## 第 5 格:把断开的边缘连起来(形态学闭运算)
#
# Canny 出来的边缘常常是**断断续续**的:飞机的白色机身和天空对比低,
# 那一段边缘可能根本检测不到。边缘一断,轮廓就不闭合,`findContours` 就会
# 把一架飞机拆成几十段小轮廓。
#
# 解决办法是**闭运算**:先膨胀再腐蚀。膨胀负责把断口"糊上",腐蚀负责把线条恢复原样。

# %% 闭运算
kernel = np.ones((5, 5), np.uint8)
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

print("闭运算前边缘像素: %d" % np.count_nonzero(edges))
print("闭运算后边缘像素: %d  <- 变多了,因为断口被连上了" % np.count_nonzero(closed))
cv2.imwrite(os.path.join(OUT_DIR, "task2_03_closed.png"), closed)

# %% [markdown]
# ## 第 6 格:找轮廓
#
# `cv2.findContours(二值图, 检索模式, 近似方式)`:
#
# | 参数 | 常用值 | 含义 |
# |---|---|---|
# | 检索模式 | `RETR_EXTERNAL` | 只取**最外层**轮廓(飞机的外框,忽略机身上的窗户孔洞) |
# | | `RETR_LIST` | 取出**所有**轮廓,不分层级 |
# | | `RETR_TREE` | 取出所有轮廓,并给出父子层级 |
# | 近似方式 | `CHAIN_APPROX_SIMPLE` | 只保留拐点,省内存(一条直线只存两个端点) |
# | | `CHAIN_APPROX_NONE` | 保留每一个边界像素点 |
#
# 注意:`findContours` 会**修改传入的二值图**,所以后面还要用的话先 `.copy()`。

# %% 找轮廓
closed_for_contours = closed.copy()          # 保护原图,因为 findContours 会改它
contours, hierarchy = cv2.findContours(closed_for_contours, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

print("找到的轮廓数量:", len(contours))
areas = [cv2.contourArea(c) for c in contours]
order = np.argsort(areas)[::-1]              # 按面积从大到小排序
print("面积最大的前 5 个轮廓(平方像素):", [int(areas[i]) for i in order[:5]])
print("面积最小的前 5 个轮廓(平方像素):", [int(areas[i]) for i in order[-5:]])

# %% [markdown]
# ## 第 7 格:面积和周长怎么算
#
# - **面积**:`cv2.contourArea(c)`,单位是"平方像素"
# - **周长**:`cv2.arcLength(c, True)`,第二个参数 `True` 表示轮廓是**闭合**的
#   (如果传 `False`,它会当作一条开放曲线,少算最后一段)
#
# 题目要的"总面积和总周长",有两种理解,我两个都算给你看:
#
# 1. **所有轮廓加起来**(题目字面意思)
# 2. **最大轮廓(也就是飞机)单独的值**(实际更有意义)

# %% 计算面积周长
total_area = sum(cv2.contourArea(c) for c in contours)
total_perimeter = sum(cv2.arcLength(c, True) for c in contours)

biggest = contours[order[0]]
plane_area = cv2.contourArea(biggest)
plane_perimeter = cv2.arcLength(biggest, True)

print("=" * 56)
print("轮廓总数量: %d" % len(contours))
print("所有轮廓的总面积: %.0f 平方像素" % total_area)
print("所有轮廓的总周长: %.0f 像素" % total_perimeter)
print()
print("最大轮廓(飞机)的面积: %.0f 平方像素" % plane_area)
print("最大轮廓(飞机)的周长: %.0f 像素" % plane_perimeter)
print("最大轮廓的顶点数: %d" % len(biggest))
print("最大轮廓面积占整张图的比例: %.1f%%" % (plane_area / (w * h) * 100))

# %% [markdown]
# ## 第 8 格:把轮廓画出来并保存

# %% 画轮廓 + 保存
vis_all = img.copy()
cv2.drawContours(vis_all, contours, -1, (0, 255, 0), 2)              # -1 表示画全部

vis_biggest = img.copy()
cv2.drawContours(vis_biggest, [biggest], -1, (0, 0, 255), 3)         # 只画最大的
cv2.rectangle(vis_biggest, cv2.boundingRect(biggest), (255, 0, 0), 2)  # 再画个外接矩形

cv2.imwrite(os.path.join(OUT_DIR, "task2_03_contours_all.png"), vis_all)
cv2.imwrite(os.path.join(OUT_DIR, "task2_03_contour_biggest.png"), vis_biggest)

# 拼成一张总览图,方便放进报告
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
panels = [
    (to_rgb(img), "原图"),
    (gray, "灰度图"),
    (edges, "Canny 边缘"),
    (closed, "闭运算后(边缘被连起来)"),
    (to_rgb(vis_all), "所有轮廓:共 %d 个" % len(contours)),
    (to_rgb(vis_biggest), "最大轮廓:面积 %.0f,周长 %.0f" % (plane_area, plane_perimeter)),
]
for ax, (data, title) in zip(axes.ravel(), panels):
    ax.imshow(data, cmap=None if data.ndim == 3 else "gray")
    ax.set_title(title)
    ax.axis("off")
fig.suptitle("task-2 题三:轮廓检测流程", fontsize=15)
fig.tight_layout()
overview = os.path.join(OUT_DIR, "task2_03_overview.png")
fig.savefig(overview, dpi=105)
plt.close(fig)
print("已保存:")
print("  ", overview)
print("  ", os.path.join(OUT_DIR, "task2_03_contour_biggest.png"))

# %% [markdown]
# ## 实验 1:不做闭运算会怎样
# **先写下你的预测**:直接拿 Canny 的结果去找轮廓,轮廓数量会变多还是变少?

# %% 实验 1
raw_contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
raw_areas = [cv2.contourArea(c) for c in raw_contours]
print("不做闭运算:轮廓 %d 个,总面积 %.0f,最大轮廓面积 %.0f"
      % (len(raw_contours), sum(raw_areas), max(raw_areas) if raw_areas else 0))
print("做了闭运算:轮廓 %d 个,总面积 %.0f,最大轮廓面积 %.0f"
      % (len(contours), total_area, plane_area))
print()
print("结论:边缘断开时,一架飞机会被拆成很多碎块,而且'最大轮廓'也抓不到飞机。")

# %% [markdown]
# ## 实验 2:CHAIN_APPROX_SIMPLE 和 NONE 差在哪
# 这两个参数**不影响面积和周长**,只影响"存多少个点"。
# **先写下你的预测**:同一个轮廓,两种方式存下的点数会差几倍?

# %% 实验 2
c_simple, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c_none, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

b_simple = max(c_simple, key=cv2.contourArea)
b_none = max(c_none, key=cv2.contourArea)

print("SIMPLE 版本:轮廓 %d 个,最大轮廓存了 %d 个点" % (len(c_simple), len(b_simple)))
print("NONE   版本:轮廓 %d 个,最大轮廓存了 %d 个点" % (len(c_none), len(b_none)))
print()
print("面积对比: %.0f vs %.0f(几乎一样)" % (cv2.contourArea(b_simple), cv2.contourArea(b_none)))
print("周长对比: %.0f vs %.0f" % (cv2.arcLength(b_simple, True), cv2.arcLength(b_none, True)))
print("结论:近似方式只影响'存多少点',不影响测量结果。")

# %% [markdown]
# ## 实验 3:检索模式换一下
# `RETR_EXTERNAL` 只看最外层,`RETR_LIST` 会把内部的小轮廓也翻出来。
# **先写下你的预测**:换成 RETR_LIST,轮廓数量和总面积会怎么变?

# %% 实验 3
for mode, name in [(cv2.RETR_EXTERNAL, "RETR_EXTERNAL"), (cv2.RETR_LIST, "RETR_LIST"),
                   (cv2.RETR_TREE, "RETR_TREE")]:
    cs, _ = cv2.findContours(closed.copy(), mode, cv2.CHAIN_APPROX_SIMPLE)
    a = sum(cv2.contourArea(c) for c in cs)
    print("%-14s 轮廓 %3d 个,总面积 %.0f" % (name, len(cs), a))

print()
print("注意:换成 RETR_LIST 之后总面积反而变大了。")
print("因为机身上的孔洞(窗户、发动机口)也被当成一个个轮廓,面积被重复加了一遍。")
print("所以题目问总面积时应该用 RETR_EXTERNAL:只数最外层,才不会重复计算。")

# %% [markdown]
# ## 实验 4:面积过滤掉小噪点
# 天空的纹理会产生一些几十像素的小轮廓,它们对结果毫无意义。
# **先写下你的预测**:把面积小于 500 的轮廓都扔掉之后,总面积会明显变小吗?

# %% 实验 4
MIN_AREA = 500
list_contours, _ = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
list_areas = sorted((cv2.contourArea(c) for c in list_contours), reverse=True)

kept = [c for c in list_contours if cv2.contourArea(c) >= MIN_AREA]
dropped = [c for c in list_contours if cv2.contourArea(c) < MIN_AREA]

print("过滤前: %d 个轮廓" % len(list_contours))
print("  面积从大到小排前 6 个: %s" % [int(a) for a in list_areas[:6]])
print("过滤后: %d 个轮廓(扔掉了 %d 个小于 %d 平方像素的)"
      % (len(kept), len(dropped), MIN_AREA))
if dropped:
    area_kept = sum(cv2.contourArea(c) for c in kept)
    area_dropped = sum(cv2.contourArea(c) for c in dropped)
    print("  扔掉的那些加起来只有 %.0f 平方像素,保留的有 %.0f,相差 %.0f 倍"
          % (area_dropped, area_kept, area_kept / max(area_dropped, 1)))
    print("结论:数量少了 %d 个,但面积只损失 %.1f%% —— 扔掉的确实都是噪点。"
          % (len(dropped), area_dropped / (area_kept + area_dropped) * 100))

# %% [markdown]
# ## 实验 5:直接把阈值二值化拿来用
# 回到第 2 格那个问题:如果不用 Canny,直接 `threshold` 会怎样?

# %% 实验 5
_, binary = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
b_contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
b_areas = [cv2.contourArea(c) for c in b_contours]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (data, title) in zip(axes, [
        (binary, "threshold=150 直接二值化"),
        (edges, "Canny 边缘(低阈值 50)"),
        (closed, "闭运算后")]):
    ax.imshow(data, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
fig.suptitle("实验 5:为什么这题不适合用简单阈值")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "task2_03_exp5_threshold.png"), dpi=110)
plt.close(fig)
print("threshold=150 得到 %d 个轮廓,最大 %.0f 平方像素"
      % (len(b_contours), max(b_areas) if b_areas else 0))
print("已保存对比图,对着看就知道为什么选了 Canny。")

# %% [markdown]
# ## 收尾
# 产出:
# - `task2_03_contour_biggest.png`:画了轮廓 + 外接矩形(交这个)
# - `task2_03_contours_all.png`:所有轮廓
# - `task2_03_overview.png`:整条流水线的总览(放进报告最好)
#
# 报告里要填的数字在**第 7 格**的输出里,直接抄过去:
# 轮廓总数量、所有轮廓的总面积/总周长、最大轮廓的面积/周长。
#
# 另外值得写一句:**"先 Canny 再闭运算"是这题的关键** ——
# 直接用阈值二值化会把飞机和天空切到一起,而 Canny 的边缘又是断的,
# 所以必须用闭运算把断口连起来,否则飞机会被拆成几十个碎块。
