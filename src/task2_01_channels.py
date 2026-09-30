# -*- coding: utf-8 -*-
r"""
task-2 第一题:用 OpenCV + Pillow 打开图片、分离三个通道

【在 VSCode 里怎么用】
  每个 "# %%" 是一个单元格,单元格上方会出现「运行单元格」的按钮。
  点一下只跑那一段,变量会留在内存里,可以反复改、反复跑 —— 这是最高效的学习方式。
  想一次跑完:终端里执行  python src/task2_01_channels.py

【建议的用法】
  1. 先通读一遍,猜每段在做什么(不要先运行)
  2. 一段一段运行,每跑完一段,对照打印出来的数字检查你的猜测
  3. 走到最后三个实验单元格:先写下预测,再点运行

如果 data/test_image.png 不存在,脚本会自动生成一张六色测试图,
所以不需要提前准备素材。想换成自己的照片,只改下面 IMAGE_FILE 那一行。
"""

# %% [markdown]
# ## 第 1 格:环境和路径
# 所有路径和参数都集中在这一格,后面任何一格都不再出现硬编码路径。

# %% 配置区
import os

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")  # 只存文件不弹窗口,避免在没有显示器的环境里卡住
import matplotlib.pyplot as plt  # noqa: E402

# ---- 路径:想用自己的照片,只改 IMAGE_FILE 这一行 ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IMAGE = os.path.join(PROJECT_ROOT, "data", "test_image.png")
MY_IMAGE = ""      # 想用自己的照片:把路径填在这里,例如 r"D:\hijia\data\my_frog.png"
IMAGE_FILE = MY_IMAGE if MY_IMAGE else DEFAULT_IMAGE
OUT_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- matplotlib 的中文字体,不设置的话标题会变成一堆方框 ----
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 六个纯色 + 名称。注意顺序是 BGR,不是 RGB,这是本题的核心 ----
COLOR_BLOCKS = [
    ((0, 0, 255), "RED"),
    ((0, 255, 0), "GREEN"),
    ((255, 0, 0), "BLUE"),
    ((0, 255, 255), "YELLOW"),
    ((255, 255, 0), "CYAN"),
    ((255, 0, 255), "MAGENTA"),
]

print("图片路径:", IMAGE_FILE)
print("输出目录:", OUT_DIR)

# %% [markdown]
# ## 第 2 格:准备一张测试图
# 上半部分是六个纯色块,下半部分是从黑到白的灰度渐变。
# 渐变那条的用处:三个通道在同一位置的值应该完全相等,可以用它来验证通道没被弄错。

# %% 生成测试图
height, width = 360, 480
img_test = np.zeros((height, width, 3), dtype=np.uint8)  # dtype 必须是 uint8(0-255)

stripe_w = width // len(COLOR_BLOCKS)
for i, (bgr, name) in enumerate(COLOR_BLOCKS):
    x0 = i * stripe_w
    x1 = width if i == len(COLOR_BLOCKS) - 1 else x0 + stripe_w
    img_test[0:300, x0:x1] = bgr                     # 纯色块
    cv2.putText(img_test, name, (x0 + 8, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

gradient = np.linspace(0, 255, width, dtype=np.uint8)  # 从左到右的灰度渐变
img_test[300:360, :, :] = gradient[None, :, None]

# 六色测试图总是写到默认位置,这样你换了自己的照片,也随时能切回来对照
if not os.path.exists(DEFAULT_IMAGE):
    cv2.imwrite(DEFAULT_IMAGE, img_test)
    print("已生成六色测试图:", DEFAULT_IMAGE)
print("本次分析用的图片:", IMAGE_FILE)

# %% [markdown]
# ## 第 3 格:用 OpenCV 读图
# 重点看两个量:**shape** 和 **dtype**。
# shape 是 (高, 宽, 3),不是 (宽, 高, 3) —— 这个顺序经常把人绕晕。

# %% OpenCV 读图
img_bgr = cv2.imread(IMAGE_FILE)
if img_bgr is None:
    raise SystemExit("OpenCV 读不出这张图,常见原因是路径含中文或文件不存在")

print("shape =", img_bgr.shape, " dtype =", img_bgr.dtype)
print("左上角那个像素的 BGR 值 =", img_bgr[0, 0].tolist())
print("它的含义是 [蓝, 绿, 红]")

# %% [markdown]
# ## 第 4 格:用 Pillow 读同一张图
# 两个库读的是同一个文件,但通道顺序不同。这一步我们用**数字**证明它,而不是靠感觉。
# 下面这个验证对**任何**图片都成立,不依赖图片里有什么内容。

# %% Pillow 读图并对比
img_pil = Image.open(IMAGE_FILE)
print("PIL size =", img_pil.size, "(宽, 高),注意和 OpenCV 的顺序相反")
print("PIL mode =", img_pil.mode)

pil_rgb = np.asarray(img_pil)
cv_as_rgb = img_bgr[:, :, ::-1]      # 把 BGR 的顺序倒过来,就是 RGB

print()
print("[关键验证] 两个库读出来的像素是不是同一批?")
print("           np.array_equal(OpenCV 倒序后, Pillow) =", np.array_equal(cv_as_rgb, pil_rgb))
print("           这说明:数据完全相同,只是“第 0 个通道叫什么名字”不一样")

print()
print("[通道均值对照] 同一张图,各自按自己的顺序念一遍:")
print("  OpenCV 顺序: B=%.1f  G=%.1f  R=%.1f" % (img_bgr[:, :, 0].mean(),
                                                  img_bgr[:, :, 1].mean(),
                                                  img_bgr[:, :, 2].mean()))
print("  Pillow 顺序: R=%.1f  G=%.1f  B=%.1f" % (pil_rgb[:, :, 0].mean(),
                                                  pil_rgb[:, :, 1].mean(),
                                                  pil_rgb[:, :, 2].mean()))
print("  一一对应的数字完全一样 —— 同一次拍摄,两套叫法而已")

# %% [markdown]
# ## 第 5 格:分离三个通道
# 一张彩色图有三个通道:第 0 个是**蓝(B)**、第 1 个是**绿(G)**、第 2 个是**红(R)**。
# 三个通道叠在一起才构成你看到的彩色;单独拿出一个,得到的就是一张灰度图。
#
# `cv2.split` 和 numpy 切片 `img[:, :, 0]` 是等价的,后者更直观。
# 分离出来的单通道图是**灰度图**,越亮表示该通道数值越大。

# %% 分离通道
b, g, r = cv2.split(img_bgr)          # 方式一
b_by_slice = img_bgr[:, :, 0]         # 方式二,和上面结果完全一样
print("两种写法结果是否一致:", np.array_equal(b, b_by_slice))

cv2.imwrite(os.path.join(OUT_DIR, "task2_01_channel_B.png"), b)
cv2.imwrite(os.path.join(OUT_DIR, "task2_01_channel_G.png"), g)
cv2.imwrite(os.path.join(OUT_DIR, "task2_01_channel_R.png"), r)

print("三个通道均值: B=%.1f  G=%.1f  R=%.1f" % (b.mean(), g.mean(), r.mean()))
print("单通道图的 shape =", b.shape, "-> 只有两个数字,因为通道已经拆开了")

# %% [markdown]
# ## 第 6 格:灰度化到底是怎么算的
# 先说"灰度化是什么":把每个像素的**三个**数合成**一个**数,这个数表示"这个点有多亮"。
# 合成之后图就变成黑白的了 —— 因为一个数没法表示颜色,只能表示亮度。
#
# 为什么要这么做?两个原因:一是数据量直接减到三分之一;
# 二是很多算法(找轮廓、边缘检测、二值化)只关心亮暗,不关心是什么颜色。
#
# 它不是三个通道简单取平均,而是按人眼敏感度加权:
# **Gray = 0.114*B + 0.587*G + 0.299*R**(绿色的权重最高,因为人眼对绿色最敏感)
# 下面手工按公式重算一遍,和 OpenCV 的结果对一下。

# %% 灰度化
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
gray_manual = (0.114 * b.astype(np.float32)
               + 0.587 * g.astype(np.float32)
               + 0.299 * r.astype(np.float32))

print("OpenCV 结果与手工计算的最大误差: %.4f" % float(np.abs(gray.astype(np.float32) - gray_manual).max()))
cv2.imwrite(os.path.join(OUT_DIR, "task2_01_gray.png"), gray)

# %% [markdown]
# ## 第 7 格:拼成对比图
# matplotlib 显示图像时要求 RGB,所以 OpenCV 读进来的图必须先转一次,
# 否则红色和蓝色会互换 —— 这是最经典的一类"不报错但结果不对"。

# %% 拼图
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
axes[0, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("原图(转成 RGB 显示)")
for ax, data, name in zip(axes.ravel()[1:], (b, g, r), ("B 通道", "G 通道", "R 通道")):
    ax.imshow(data, cmap="gray", vmin=0, vmax=255)
    ax.set_title(name)
for ax in axes.ravel():
    ax.axis("off")
fig.suptitle("task-2 题一:原图与三个通道", fontsize=14)
fig.tight_layout()

montage_path = os.path.join(OUT_DIR, "task2_01_channels.png")
fig.savefig(montage_path, dpi=120)
plt.close(fig)
print("已保存:", montage_path)

# %% [markdown]
# ## 实验 1:忘记转换的后果
# **先写下你的预测**:如果把 OpenCV 读进来的图**直接**交给 matplotlib 显示
# (也就是新手最常犯的错——忘记转 RGB),红蓝会互换吗?黄色和青色会变吗?
#
# 原理其实就一句话:**同一个数组,两个库对"第 0 个位置"的理解不一样**。
#
# | 数组里的位置 | 第 0 个 | 第 1 个 | 第 2 个 |
# |---|---|---|---|
# | OpenCV 认为它是 | B 蓝 | G 绿 | R 红 |
# | matplotlib 认为它是 | R 红 | G 绿 | B 蓝 |
#
# 所以纯红色的数组是 [0, 0, 255],matplotlib 会读成"红=0、蓝=255",画出来就是蓝的。
# 而绿色的数组是 [0, 255, 0],两个库都读成"绿=255",所以绿色看起来没变;
# 品红是 [255, 0, 255],首尾一样,换了也还是自己。
#
# 顺带一个冷知识:`COLOR_BGR2RGB` 和 `COLOR_RGB2BGR` 其实是**同一个操作**
# (都是把三个通道的顺序倒过来),所以用哪个都一样。真正会出错的是**什么都不做**。

# %% 实验 1
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
axes[0].set_title("正确:先转成 RGB")
axes[1].imshow(img_bgr)                      # 直接塞进去,没有转换
axes[1].set_title("错误:直接用 BGR")
for ax in axes:
    ax.axis("off")
fig.suptitle("实验 1:转换方向搞反的后果")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "task2_01_exp1_wrong_convert.png"), dpi=120)
plt.close(fig)
print("已保存实验 1 的对比图")

# %% [markdown]
# ## 实验 2:整块替换 vs 按比例调暗
# **先写下你的预测**:下面两个动作都作用在**下方那条灰度渐变**上。
# 它们是"在原来的数值上做减法",还是"整块换成一个固定值"?B 和 G 会跟着变吗?
#
# 这一格的注释我第一版写得有歧义(写成了"调暗"),现在拆成两个动作对比给你看。

# %% 实验 2
# 动作 A:整块替换 —— 那一块所有像素被换成同一个值
img_replace = img_bgr.copy()
img_replace[300:360, 0:200] = (0, 0, 200)

# 动作 B:按比例调暗 —— 只动红色通道,原来的明暗关系保留
img_scale = img_bgr.copy()
img_scale[300:360, 0:200, 2] = (img_scale[300:360, 0:200, 2].astype(np.float32) * 0.8).astype(np.uint8)

region = (slice(300, 360), slice(0, 200))
print("下方灰度渐变那一条的三通道均值:")
print("  原图          B=%.1f G=%.1f R=%.1f" % (
    img_bgr[region][:, :, 0].mean(), img_bgr[region][:, :, 1].mean(), img_bgr[region][:, :, 2].mean()))
print("  A 整块替换后  B=%.1f G=%.1f R=%.1f" % (
    img_replace[region][:, :, 0].mean(), img_replace[region][:, :, 1].mean(), img_replace[region][:, :, 2].mean()))
print("  B 按比例调暗  B=%.1f G=%.1f R=%.1f" % (
    img_scale[region][:, :, 0].mean(), img_scale[region][:, :, 1].mean(), img_scale[region][:, :, 2].mean()))
print()
print("区别:替换会把 B、G 也强行改成 0(因为新值就是 0);")
print("      按比例调暗只动 R,B、G 保留原来的数值。")
print("这就是 img[区域] = 值 和 img[区域, 2] = img[区域, 2] * 0.8 的区别。")

# %% [markdown]
# ## 实验 3:用灰度模式读图
# **先写下你的预测**:`cv2.imread(路径, cv2.IMREAD_GRAYSCALE)` 读出来的数组,
# shape 会是几个数字?后面那些用 `img_bgr[:, :, 0]` 的代码还能跑吗?
#
# 提示:这里的两个数字是 **(高, 宽)**,和"刚才分离过通道"没有任何关系 ——
# 灰度图本来就只有两个维度。另外,这两个数字取决于**你用的是哪张图**。

# %% 实验 3
img_gray_mode = cv2.imread(IMAGE_FILE, cv2.IMREAD_GRAYSCALE)
print("灰度模式读入的 shape =", img_gray_mode.shape)
print("dtype =", img_gray_mode.dtype)
try:
    _ = img_gray_mode[:, :, 0]
    print("按三通道方式取第 0 通道:居然成功了?")
except IndexError as exc:
    print("按三通道方式取第 0 通道:报错 ->", exc)
    print("原因:灰度图只有两个维度,不存在第 3 个下标")

# %% [markdown]
# ## 收尾
# 把这张对比图放进报告:`results/task2_01_channels.png`
# 并用一句话说明结论:OpenCV 读入是 BGR,Pillow/matplotlib 是 RGB,
# 通道顺序不同会造成颜色异常,所以跨库使用时必须先转换。
