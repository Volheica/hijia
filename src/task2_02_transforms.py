# -*- coding: utf-8 -*-
r"""
task-2 题二:对图片做缩放、平移、旋转、翻转

【在 VSCode 里怎么用】
  每个 "# %%" 是一个单元格,点它上方的 Run Cell 逐格运行。
  整段运行:终端里  python src/task2_02_transforms.py

【建议的用法】
  1. 先通读一遍,猜每格在做什么(不要先运行)
  2. 逐格运行,对照打印出来的数字检查猜测
  3. 最后五个实验:先写下预测,再运行

这一题的核心只有一句话:
  除了缩放和翻转,其它的"变换"都是同一个套路 —— 构造一个 2×3 的矩阵,
  然后交给 cv2.warpAffine 去执行。
"""

# %% [markdown]
# ## 第 1 格:配置
# 题二用题目自带的 example1.jpg(一架飞机),这和题三保持同一张图,报告读起来连贯。

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
    """matplotlib 需要 RGB,而 OpenCV 读进来是 BGR,显示前必须转一次。"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


print("输入图片:", IMAGE_FILE)
print("输出目录:", OUT_DIR)

# %% 读图
img = cv2.imread(IMAGE_FILE)
if img is None:
    raise SystemExit("读不到图片,检查路径:%s" % IMAGE_FILE)

h, w = img.shape[:2]
print("原始尺寸:高 %d,宽 %d" % (h, w))

# %% [markdown]
# ## 第 2 格:缩放(resize)
# 两种写法是等价的:
# - `cv2.resize(img, (w, h))` 直接指定目标尺寸
# - `cv2.resize(img, None, fx=0.5, fy=0.5)` 指定缩放比例
#
# **注意 `(w, h)` 的顺序是"宽在前、高在后"**,和 shape 的 `(高, 宽)` 正好相反 ——
# 这是 OpenCV 里最容易搞混的一处。

# %% 缩放
half_size = cv2.resize(img, (w // 2, h // 2))                  # 指定目标尺寸
half_ratio = cv2.resize(img, None, fx=0.5, fy=0.5)             # 指定缩放比例

print("原图 shape        :", img.shape)
print("指定尺寸后的 shape:", half_size.shape)
print("指定比例后的 shape:", half_ratio.shape)
print("两种写法结果是否完全一致:", np.array_equal(half_size, half_ratio))

# %% [markdown]
# ## 第 3 格:平移 —— 第一个 2×3 矩阵
# 平移可以用一个矩阵表示:
#
# ```
# M = [[1, 0, tx],
#      [0, 1, ty]]
# ```
#
# 它的意思是:**新坐标 = 原坐标 + (tx, ty)**。
# 看矩阵就能发现,平移只改了**最后一列**。
#
# 变换矩阵统一是 2 行 3 列:`[[a, b, tx], [c, d, ty]]`,前两列管"怎么变",
# 第三列管"平移多少"。这也是为什么不用 2×2 —— 2×2 做不了平移。

# %% 平移
tx, ty = 120, 60                      # 向右 120 像素,向下 60 像素
M_translate = np.float32([[1, 0, tx],
                          [0, 1, ty]])

print("平移矩阵:")
print(M_translate)

shifted = cv2.warpAffine(img, M_translate, (w, h))
print("平移后 shape:", shifted.shape, " <- 画布大小没变,内容整体挪了")

# %% [markdown]
# ## 补充:那个矩阵到底在算什么?(以及 y 轴为什么朝下)
#
# **矩阵不是"数据",它是一张公式表。** 看这两行:
#
# ```
# x_new = 1*x + 0*y + tx
# y_new = 0*x + 1*y + ty
# ```
#
# 写成你看到的两行三列:
#
# ```
# [[1, 0, tx],
#  [0, 1, ty]]
#       ↑       ↑
#      x、y 的系数  平移量
# ```
#
# 每一行描述"新坐标的**一个分量**怎么算出来":第一行算新的 x,第二行算新的 y。
# 第三列就是在"乘完之后再加上去"的平移量 —— 所以平移只改了最后一列。
#
# 至于 `np.float32`,那是矩阵的**数据类型**。`warpAffine` 只接受 float32 或 float64
# 的矩阵,传整数进去会直接报错,所以这里必须显式指定。
#
# 最后是最重要的那句话:**图像的 y 轴是朝下的**。原点在左上角,x 往右增大,y 往下增大。
# 所以 ty 为正 = 图片**往下**移。这个"y 轴朝下"还会在后面解释旋转方向时再出现一次。
# 下面用代码和图形各验证一遍。

# %% 补充 1:把矩阵当公式算一遍
p0 = (100, 200)                      # 图上随便一个点,格式是 (x, y)
pt = np.array([p0[0], p0[1], 1.0])   # 末尾补一个 1,凑成三个数才能和 2×3 矩阵相乘
result = M_translate @ pt

print("矩阵:")
print(M_translate)
print("原坐标      :", (p0[0], p0[1]))
print("矩阵乘法结果:", (round(float(result[0]), 1), round(float(result[1]), 1)))
print("手算的预测  :", (p0[0] + tx, p0[1] + ty))
print("是否一致    :", np.allclose(result, [p0[0] + tx, p0[1] + ty]))
print("           (结果是两个数,不是三个 —— 因为变换只输出新的 x 和 y)")

# %% 补充 2:在图上把坐标系和位移画出来
demo = img.copy()

cv2.arrowedLine(demo, (30, 30), (280, 30), (255, 255, 255), 4)
cv2.putText(demo, "x -> right", (290, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.arrowedLine(demo, (30, 30), (30, 280), (255, 255, 255), 4)
cv2.putText(demo, "y  v down", (40, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

cv2.circle(demo, p0, 10, (0, 0, 255), -1)
cv2.circle(demo, (p0[0] + tx, p0[1] + ty), 10, (0, 255, 0), -1)
cv2.arrowedLine(demo, p0, (p0[0] + tx, p0[1] + ty), (0, 255, 255), 3)
cv2.putText(demo, "before", (p0[0] - 40, p0[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.putText(demo, "after", (p0[0] + tx - 40, p0[1] + ty + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

fig, ax = plt.subplots(figsize=(11, 6))
ax.imshow(to_rgb(demo))
ax.set_title("坐标系与平移:原点在左上角,x 向右增大,y 向下增大")
ax.axis("off")
fig.tight_layout()
coord_path = os.path.join(OUT_DIR, "task2_02_coord_shift.png")
fig.savefig(coord_path, dpi=110)
plt.close(fig)
print("已保存:", coord_path)

# %% [markdown]
# ## 第 4 格:旋转 —— 用一个函数造出矩阵
# `cv2.getRotationMatrix2D(旋转中心, 角度, 缩放比例)` 返回的就是那个 2×3 矩阵。
#
# **画布尺寸的坑**:`warpAffine` 的第三个参数是你指定的输出画布大小。
# 如果旋转后还沿用原尺寸,转出去的四个角会被**裁掉**。
# 想完整保留,得先算出旋转后图像的"外接矩形"有多大。

# %% 旋转
angle = 30
M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

print("旋转矩阵(中心=图像中心,角度=%d):" % angle)
print(M_rot.round(4))

rot_cropped = cv2.warpAffine(img, M_rot, (w, h))       # 画布不变 -> 会被裁

# 算出能装下整幅旋转后图像的新画布尺寸
cos_a = abs(M_rot[0, 0])
sin_a = abs(M_rot[0, 1])
new_w = int(h * sin_a + w * cos_a)
new_h = int(h * cos_a + w * sin_a)

M_rot_full = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
M_rot_full[0, 2] += new_w / 2 - w / 2          # 把旋转中心挪到新画布中心
M_rot_full[1, 2] += new_h / 2 - h / 2
rot_full = cv2.warpAffine(img, M_rot_full, (new_w, new_h))

print("原画布        :", (w, h))
print("旋转后需要的画布:", (new_w, new_h))
print("被裁版本 shape:", rot_cropped.shape)
print("完整版本 shape:", rot_full.shape)

# %% [markdown]
# ## 补充:那个 `+= new_w/2 - w/2` 到底在解决什么问题?
#
# 关键要理解 `warpAffine` 的工作方式:
#
# > 它是**逐一遍历输出画布上的每个像素**,反着算"这个位置对应原图的哪一点"。
# > 而输出画布的原点**永远固定在左上角 (0,0)**,它不会自动把内容居中。
#
# 所以事情是这样的:
#
# 1. `M_rot` 是"绕旧画布中心 (w/2, h/2) 转"的矩阵 —— 转完之后,旋转轴还在旧坐标 (w/2, h/2) 那里
# 2. 现在你把画布换成了更大的 (new_w, new_h),但画布左上角依然是 (0,0)
# 3. 相当于:大纸的原点和小纸的原点对齐了,图片却还按旧位置画 —— 于是图片偏向左上,左上角被切掉、右下角空出一片
#
# 解决办法就是把整个结果**平移半个差值**,让旋转轴落回新画布的正中心:
#
# ```
# 偏移量 = 新画布中心 - 旧画布中心 = (new_w/2 - w/2, new_h/2 - h/2)
# ```
#
# 而这个平移量正好可以直接写在矩阵的第三列上(还记得吗,第三列就是平移量)。
#
# 下面用数字验证:旋转后图像中心应该正好落在新画布中心。

# %% 补充 3:验证"旋转轴落到了新画布中心"
center_old = np.array([w / 2, h / 2, 1.0])

print("旋转前图像中心       :", (w / 2, h / 2))
print("新画布的中心应该在哪 :", (new_w / 2, new_h / 2))
print()
print("不加偏移时,中心停在 :", (M_rot @ center_old).round(1), " <- 还是旧坐标,不是新画布中心")
print("加了偏移后,中心落在 :", (M_rot_full @ center_old).round(1), " <- 正好是新画布中心 OK")

# 顺便验证一下:如果只修正 x 方向,不修正 y 方向会怎样?
# (这一行只是为了说明 M[0,2] 和 M[1,2] 各管一个方向)
M_half = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
M_half[0, 2] += new_w / 2 - w / 2
print()
print("M[0, 2] 管的是 x 方向(宽),M[1, 2] 管的是 y 方向(高)")
print("只修正 x 时,中心落在 :", (M_half @ center_old).round(1), " <- 左右居中了,上下还偏上")
print("两个都修正后,中心落在:", (M_rot_full @ center_old).round(1), " <- 这才是新画布正中心")

# %% 补充 4:三种画布放一起比
no_offset = cv2.warpAffine(img, M_rot, (new_w, new_h))     # 新画布,但不加偏移

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (data, title) in zip(axes, [
        (rot_cropped, "A 用原画布:四角被切"),
        (no_offset, "B 用新画布但不加偏移:偏向左上"),
        (rot_full, "C 用新画布 + 偏移:完整且居中")]):
    ax.imshow(to_rgb(data))
    ax.set_title(title)
    ax.axis("off")
fig.suptitle("为什么要加那个偏移量")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "task2_02_exp_offset.png"), dpi=110)
plt.close(fig)
print("已保存:", os.path.join(OUT_DIR, "task2_02_exp_offset.png"))

# %% [markdown]
# ## 第 5 格:那个矩阵到底是怎么算出来的
# 直接把 OpenCV 的公式手写一遍,你就不必再"背 API"了。
#
# 注意 OpenCV 返回的旋转部分是 `[[cos, sin], [-sin, cos]]`,
# 和数学课本上的 `[[cos, -sin], [sin, cos]]` 符号正好相反。
# 原因在于**图像的 y 轴朝下**(y 变大是往下走),同一个矩阵作用在图片上,
# 视觉上得到的是逆时针旋转 —— 所以 **正角度 = 屏幕上逆时针转**。
# 如果想让它顺时针转,把角度写成负数即可。

# %% 手写旋转矩阵
theta = np.deg2rad(angle)
c, s = np.cos(theta), np.sin(theta)
M_manual = np.float32([[c, s, 0],
                       [-s, c, 0]])

M_cv_zero_center = cv2.getRotationMatrix2D((0, 0), angle, 1.0)

print("OpenCV 给的(旋转中心在原点):")
print(M_cv_zero_center.round(4))
print("按公式手写的:")
print(M_manual.round(4))
print("两者是否相同:", np.allclose(M_cv_zero_center, M_manual, atol=1e-5))
print()
print("说明:getRotationMatrix2D 只是帮你把'绕原点转'的矩阵,")
print("      再加上'把旋转中心挪到指定位置'的平移量而已。")

# %% [markdown]
# ## 第 6 格:翻转(flip)
# 翻转不算仿射变换的常规用法,OpenCV 给了一行函数:
#
# | 参数 | 效果 |
# |---|---|
# | `1` | 水平翻转(左右镜像) |
# | `0` | 垂直翻转(上下倒影) |
# | `-1` | 同时水平和垂直(等于转 180°) |

# %% 翻转
flip_h = cv2.flip(img, 1)
flip_v = cv2.flip(img, 0)
flip_both = cv2.flip(img, -1)

print("水平翻转后 shape:", flip_h.shape)
print("验证:水平翻转两次会回到原图 ->", np.array_equal(cv2.flip(flip_h, 1), img))

# %% [markdown]
# ## 补充:flip 的参数怎么读,"both" 到底是什么
#
# `cv2.flip(img, flipCode)` 的第二个参数只有三种取值:
#
# | flipCode | 含义 | 通俗说法 |
# |---|---|---|
# | `1` | 水平翻转 | 左右镜像(沿竖直轴翻) |
# | `0` | 垂直翻转 | 上下倒影(沿水平轴翻) |
# | `-1` | 两个方向都翻 | 左右 + 上下,**等价于旋转 180°** |
#
# 所以 `flip_both` **不是"两张图的组合",而是"一次同时做两件事"的结果** ——
# 它当然只有一张图。就像"同时往左走一步、再往上走一步",最后只会到达**一个**位置。
#
# 下面既把它画出来,也顺手验证"两个方向都翻 = 转 180°"这个说法。

# %% 补充 5:三种翻转并排 + 验证 both 等价于转 180 度
same_as_rotate180 = np.array_equal(flip_both, cv2.rotate(img, cv2.ROTATE_180))
print("flip(-1) 和 rotate(ROTATE_180) 的结果是否完全相同:", same_as_rotate180)

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (data, title) in zip(axes, [
        (img, "原图"),
        (flip_h, "flip(1) 水平翻转"),
        (flip_v, "flip(0) 垂直翻转"),
        (flip_both, "flip(-1) 两个方向都翻 = 转 180 度")]):
    ax.imshow(to_rgb(data))
    ax.set_title(title)
    ax.axis("off")
fig.suptitle("补充 5:三种翻转的关系")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "task2_02_exp6_flip.png"), dpi=110)
plt.close(fig)
print("已保存:", os.path.join(OUT_DIR, "task2_02_exp6_flip.png"))

# %% [markdown]
# ### 顺带认识 cv2.rotate
#
# 如果要转的是 90° 的整数倍,用 `cv2.rotate` 更直接:
#
# | 参数 | 效果 |
# |---|---|
# | `cv2.ROTATE_90_CLOCKWISE` | 顺时针 90°(宽高互换) |
# | `cv2.ROTATE_180` | 180°(宽高不变) |
# | `cv2.ROTATE_90_COUNTERCLOCKWISE` | 逆时针 90° |
#
# 它和 `getRotationMatrix2D` 的分工很清楚:
# **90 的整数倍**用 `rotate`,干净利落、不会出现黑边;
# **任意角度**才需要仿射矩阵,而那就要自己处理画布尺寸了。

# %% 补充 6:cv2.rotate 的三个方向
rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
rot180 = cv2.rotate(img, cv2.ROTATE_180)
rot270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

print("原图 shape      :", img.shape)
print("顺时针 90 shape :", rot90.shape, " <- 宽高互换了")
print("180 度 shape    :", rot180.shape, " <- 宽高不变")

cv2.imwrite(os.path.join(OUT_DIR, "task2_02_09_rotate90_clockwise.png"), rot90)
cv2.imwrite(os.path.join(OUT_DIR, "task2_02_10_rotate90_counterclockwise.png"), rot270)
print("已保存 rotate 90 / 270 两张结果图(作为补充素材,题目没要求,不影响评分)")

# %% [markdown]
# ## 第 7 格:拼成结果图
# 这是要交的产出:一张图里看到四种操作的效果。

# %% 拼图
panels = [
    (to_rgb(img), "原图 %dx%d" % (w, h)),
    (to_rgb(half_size), "缩放 0.5 倍"),
    (to_rgb(shifted), "平移 (+%d, +%d)" % (tx, ty)),
    (to_rgb(rot_cropped), "旋转 %d°(原画布,被裁)" % angle),
    (to_rgb(rot_full), "旋转 %d°(完整画布)" % angle),
    (to_rgb(flip_h), "水平翻转"),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, (data, title) in zip(axes.ravel(), panels):
    ax.imshow(data)
    ax.set_title(title)
    ax.axis("off")
fig.suptitle("task-2 题二:缩放、平移、旋转、翻转", fontsize=15)
fig.tight_layout()

out_path = os.path.join(OUT_DIR, "task2_02_ops.png")
fig.savefig(out_path, dpi=110)
plt.close(fig)
print("已保存:", out_path)

# %% [markdown]
# ## 补充:把每一张结果也单独存下来
# 拼图适合放进报告里一眼看全,但题目要求提交"图像操作结果",
# 而且以后写报告、做答辩、打包发邮件时,单张文件比拼图更灵活
# (比如你只想贴"旋转后的图",就不必从拼图里裁)。
#
# 所以两种都存:**单张 + 拼图**。

# %% 保存单张结果
saved_files = []


def save_one(filename, image):
    path = os.path.join(OUT_DIR, filename)
    if not cv2.imwrite(path, image):          # imwrite 失败时返回 False,不报错,所以要检查
        raise RuntimeError("保存失败:" + path)
    saved_files.append((filename, image.shape))


save_one("task2_02_01_resize_half.png", half_size)          # 缩放(指定尺寸)
save_one("task2_02_02_resize_ratio.png", half_ratio)        # 缩放(指定比例)
save_one("task2_02_03_translate.png", shifted)              # 平移
save_one("task2_02_04_rotate_cropped.png", rot_cropped)     # 旋转(原画布,被裁)
save_one("task2_02_05_rotate_full.png", rot_full)           # 旋转(完整画布)
save_one("task2_02_06_flip_horizontal.png", flip_h)         # 水平翻转
save_one("task2_02_07_flip_vertical.png", flip_v)           # 垂直翻转
save_one("task2_02_08_flip_both.png", flip_both)            # 水平+垂直

print("已保存 %d 张单图:" % len(saved_files))
for name, shape in saved_files:
    print("  %-32s %s" % (name, shape))
print()
print("加上拼图 task2_02_ops.png,题二的图像操作结果就齐了。")

# %% [markdown]
# ## 实验 1:旋转中心放在哪?
# **先写下你的预测**:把旋转中心从图像中心改成左上角 `(0, 0)`,飞机还会在画面里吗?
# 画面里剩下什么?

# %% 实验 1
M_corner = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
rot_corner = cv2.warpAffine(img, M_corner, (w, h))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (data, title) in zip(axes, [
        (to_rgb(img), "原图"),
        (to_rgb(rot_cropped), "绕图像中心转 %d°" % angle),
        (to_rgb(rot_corner), "绕左上角 (0,0) 转 %d°" % angle)]):
    ax.imshow(data)
    ax.set_title(title)
    ax.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "task2_02_exp1_center.png"), dpi=110)
plt.close(fig)
print("已保存实验 1 的对比图")

# %% [markdown]
# ## 实验 2:旋转后用什么颜色填空白?
# 旋转后的四个角是画布上原本没有的地方,OpenCV 默认填**黑色**。
# **先写下你的预测**:把填充色改成白色,该改哪个参数?

# %% 实验 2
rot_white = cv2.warpAffine(img, M_rot, (w, h),
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255, 255, 255))     # BGR 三元组,白色

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (data, title) in zip(axes, [
        (to_rgb(rot_cropped), "默认:黑边"),
        (to_rgb(rot_white), "borderValue=(255,255,255):白边")]):
    ax.imshow(data)
    ax.set_title(title)
    ax.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "task2_02_exp2_border.png"), dpi=110)
plt.close(fig)
print("已保存实验 2 的对比图")

# %% [markdown]
# ## 实验 3:缩小再放大,插值方式的差别就露出来了
# 把图缩到 1/8 再放大回原尺寸,细节已经丢了 —— 这时三种插值方式的差别肉眼可见。
# **先写下你的预测**:哪一种会出现明显的马赛克方块?哪一种最平滑?

# %% 实验 3
small = cv2.resize(img, (w // 8, h // 8), interpolation=cv2.INTER_AREA)

restored_nearest = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
restored_linear = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
restored_cubic = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

# 只取局部放大对比,细节更清楚
crop = (slice(100, 300), slice(200, 500))
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (data, title) in zip(axes, [
        (img, "原图(局部)"),
        (restored_nearest, "INTER_NEAREST"),
        (restored_linear, "INTER_LINEAR"),
        (restored_cubic, "INTER_CUBIC")]):
    ax.imshow(to_rgb(data[crop]))
    ax.set_title(title)
    ax.axis("off")
fig.suptitle("实验 3:缩到 1/8 再放大,不同插值方式的效果")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "task2_02_exp3_interp.png"), dpi=110)
plt.close(fig)
print("已保存实验 3 的对比图")
print("经验法则:缩小用 INTER_AREA,放大用 INTER_LINEAR,追求平滑用 INTER_CUBIC")

# %% [markdown]
# ## 实验 4:平移矩阵里两个数字换一下
# **先写下你的预测**:把 `tx, ty = 120, 60` 改成 `tx, ty = 60, 120`,
# 飞机是往右下方挪得更多,还是往右挪得更少?
#
# 再想一步:如果 tx 写成 **负数**,会往哪个方向走?

# %% 实验 4
M_swap = np.float32([[1, 0, 60],
                     [0, 1, 120]])
shifted_swap = cv2.warpAffine(img, M_swap, (w, h))

M_negative = np.float32([[1, 0, -120],
                         [0, 1, -60]])
shifted_negative = cv2.warpAffine(img, M_negative, (w, h))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (data, title) in zip(axes, [
        (to_rgb(shifted), "tx=120, ty=60(向右下)"),
        (to_rgb(shifted_swap), "tx=60, ty=120(更靠下)"),
        (to_rgb(shifted_negative), "tx=-120, ty=-60(向左上)")]):
    ax.imshow(data)
    ax.set_title(title)
    ax.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "task2_02_exp4_shift.png"), dpi=110)
plt.close(fig)
print("已保存实验 4 的对比图")

# %% [markdown]
# ## 实验 5:剪切 —— 仿射家族里你还没见过的成员
# 这个矩阵是:
#
# ```
# [[1, 0.5, 0],
#  [0,  1,  0]]
# ```
#
# 翻译成公式:`x_new = x + 0.5*y`,`y_new = y`。
# 也就是说 y 越大(越靠下)的地方,x 往右偏得越多 —— 图片会被"推斜",像斜体字。
#
# **先写下你的预测**:图片会变成什么形状?四条边还是直线吗?上下两条边还平行吗?

# %% 实验 5
M_shear = np.float32([[1, 0.5, 0],
                      [0, 1, 0]])

shear_w = int(w + 0.5 * h)          # 剪切后 x 方向会变宽,画布要留够
sheared = cv2.warpAffine(img, M_shear, (shear_w, h))

fig, axes = plt.subplots(1, 2, figsize=(15, 4))
for ax, (data, title) in zip(axes, [
        (img, "原图"),
        (sheared, "剪切 x_new = x + 0.5y")]):
    ax.imshow(to_rgb(data))
    ax.set_title(title)
    ax.axis("off")
fig.suptitle("实验 5:剪切(shear)—— 仿射变换的第四个成员")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "task2_02_exp5_shear.png"), dpi=110)
plt.close(fig)
print("已保存实验 5 的对比图")
print("注意:矩形被推成了平行四边形,但四条边依然是直的、上下边依然平行")

# %% [markdown]
# ## 收尾
# 产出文件:`results/task2_02_ops.png`(四种操作的效果拼图)
# 报告里写一句结论:除缩放和翻转外,平移与旋转都通过 2×3 变换矩阵 + warpAffine 完成;
# 旋转时要自己算新画布尺寸,否则四个角会被裁掉。
