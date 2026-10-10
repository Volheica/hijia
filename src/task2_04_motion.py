# -*- coding: utf-8 -*-
r"""
task-2 题四:运动目标检测 —— MOG2 背景减除 vs 帧差法

【题目要求】对 example2.mp4 做运动目标检测,比较"用了背景减除(MOG2)"
和"没用背景减除(帧差法)"对运动物体识别的影响。

【在 VSCode 里怎么用】每个 "# %%" 是一个单元格,点 Run Cell 逐格运行。
【整段运行】python src/task2_04_motion.py

这个脚本要跑两遍视频(984 帧 x 2),大概几十秒到两分钟。
**适合挂机**:启动后去干别的,回来只看最后的统计表和对比图。

产出的视频不会进 git(.gitignore 里排除了 *.mp4),按题目要求走邮箱提交。
"""

# %% [markdown]
# ## 第 1 格:配置

# %% 配置区
import os
import time

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_FILE = os.path.join(PROJECT_ROOT, "data", "example2.mp4")
OUT_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 可调参数(想做实验就改这里)----
MOG2_HISTORY = 500          # 背景模型保留多少帧的历史
MOG2_THRESHOLD = 16         # 判定"和背景不一样"的阈值,越小越敏感
MOG2_DETECT_SHADOWS = True  # 是否检测阴影(开启后阴影像素值为 127)
DIFF_THRESHOLD = 25         # 帧差法的二值化阈值,越小越敏感
OPEN_KERNEL = 3             # 形态学开运算核大小(去噪点)
MIN_AREA = 300              # 小于这个面积的轮廓当作噪点丢掉
WARMUP_FRAMES = 30          # 前多少帧不算进统计(MOG2 背景模型需要预热)


def to_rgb(image):
    """matplotlib 需要 RGB;灰度图先转成三通道,也能正常显示。"""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def annotate(frame, mask, title):
    """开运算去噪 -> 找轮廓 -> 画目标框,返回(画好的帧, 干净掩码, 框列表)。"""
    kernel = np.ones((OPEN_KERNEL, OPEN_KERNEL), np.uint8)
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        boxes.append((x, y, bw, bh))
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    cv2.putText(frame, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return frame, clean, boxes


print("输入视频:", VIDEO_FILE)

# %% 读取视频信息
cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    raise SystemExit("打不开视频:" + VIDEO_FILE)

FPS = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
N_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

print("尺寸: %d x %d" % (W, H))
print("帧率: %.1f fps" % FPS)
print("总帧数: %d,时长 %.1f 秒" % (N_FRAMES, N_FRAMES / FPS))

# %% [markdown]
# ## 第 2 格:方法 A —— MOG2 背景减除
#
# **思路**:给每个像素维护一个历史模型,记录它"平时长什么样"。新来的帧和模型一比,
# 差得多的就认为是**前景(运动物体)**。
#
# | 参数 | 含义 |
# |---|---|
# | `history` | 用多少帧建立背景模型。越大越稳,但对新场景适应越慢 |
# | `varThreshold` | 判定阈值,**越小越敏感** |
# | `detectShadows` | 是否标记阴影。开启后阴影像素值是 **127**,要自己清成 0 |

# %% 方法 A:MOG2
VIDEO_A = os.path.join(OUT_DIR, "task2_04_01_mog2.mp4")
writer_a = cv2.VideoWriter(VIDEO_A, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=MOG2_HISTORY, varThreshold=MOG2_THRESHOLD, detectShadows=MOG2_DETECT_SHADOWS)

cap = cv2.VideoCapture(VIDEO_FILE)
stats_a = {"fg_pixels": [], "boxes": [], "box_area": []}
samples_a = {}
t0 = time.time()
idx = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    mask = fgbg.apply(frame)
    if MOG2_DETECT_SHADOWS:
        mask[mask == 127] = 0            # 阴影不是运动物体,清掉

    vis, clean, boxes = annotate(frame.copy(), mask, "MOG2 background subtraction")
    writer_a.write(vis)

    if idx >= WARMUP_FRAMES:             # 跳过预热期,统计才有意义
        stats_a["fg_pixels"].append(int(np.count_nonzero(clean)))
        stats_a["boxes"].append(len(boxes))
        stats_a["box_area"].append(sum(bw * bh for _, _, bw, bh in boxes))

    if idx in (100, 400, 800):           # 留几帧做报告插图
        samples_a[idx] = (frame.copy(), clean.copy())
    idx += 1

cap.release()
writer_a.release()
elapsed_a = time.time() - t0

print("MOG2 处理完成:%d 帧,耗时 %.1f 秒(约 %.1f fps)" % (idx, elapsed_a, idx / elapsed_a))
print("输出视频:", VIDEO_A)

# %% [markdown]
# ## 第 3 格:方法 B —— 帧差法(不用背景减除)
#
# **思路**:只比较**相邻两帧**。同一位置的像素值变化超过阈值,就算运动。
# 核心其实只有一行:
#
# ```python
# diff = cv2.absdiff(prev_gray, cur_gray)
# ```
#
# 比 MOG2 简单得多,但有一个天生的短板 —— 后面的实验里看。

# %% 方法 B:帧差法
VIDEO_B = os.path.join(OUT_DIR, "task2_04_02_framediff.mp4")
writer_b = cv2.VideoWriter(VIDEO_B, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

cap = cv2.VideoCapture(VIDEO_FILE)
stats_b = {"fg_pixels": [], "boxes": [], "box_area": []}
samples_b = {}
prev_gray = None
t0 = time.time()
idx = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is None:
        mask = np.zeros_like(gray)       # 第一帧没有"上一帧",只能算全黑
    else:
        diff = cv2.absdiff(prev_gray, gray)
        _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    prev_gray = gray

    vis, clean, boxes = annotate(frame.copy(), mask, "frame differencing")
    writer_b.write(vis)

    if idx >= WARMUP_FRAMES:
        stats_b["fg_pixels"].append(int(np.count_nonzero(clean)))
        stats_b["boxes"].append(len(boxes))
        stats_b["box_area"].append(sum(bw * bh for _, _, bw, bh in boxes))

    if idx in (100, 400, 800):
        samples_b[idx] = (frame.copy(), clean.copy())
    idx += 1

cap.release()
writer_b.release()
elapsed_b = time.time() - t0

print("帧差法处理完成:%d 帧,耗时 %.1f 秒(约 %.1f fps)" % (idx, elapsed_b, idx / elapsed_b))
print("输出视频:", VIDEO_B)

# %% [markdown]
# ## 第 4 格:两种方法的量化对比
#
# 光看视频容易"感觉差不多",所以用数字说话:
#
# - **平均前景像素数**:检测出的运动区域有多大
# - **标准差**:输出稳不稳定,越大说明时好时坏
# - **抓到目标的帧比例**:有多少帧真的检测到了东西

# %% 统计对比
def summarize(name, stats, elapsed, n_frames):
    fg = np.array(stats["fg_pixels"])
    bx = np.array(stats["boxes"])
    ba = np.array(stats["box_area"])
    print()
    print("【%s】" % name)
    print("  处理速度      : %.1f fps" % (n_frames / elapsed))
    print("  平均前景像素  : %.0f" % fg.mean())
    print("  前景像素标准差: %.0f  <- 越大说明输出越不稳定" % fg.std())
    print("  抓到目标的帧  : %d / %d(%.1f%%)"
          % (int((bx > 0).sum()), len(bx), (bx > 0).mean() * 100))
    print("  平均目标个数  : %.2f 个" % bx.mean())
    print("  平均框总面积  : %.0f 平方像素  <- 拖影会把框撑大" % ba.mean())
    return fg, bx, ba


print("=" * 60)
print("两种方法对比(已跳过前 %d 帧预热期)" % WARMUP_FRAMES)
print("=" * 60)

fg_a, bx_a, ba_a = summarize("MOG2 背景减除", stats_a, elapsed_a, idx)
fg_b, bx_b, ba_b = summarize("帧差法(不用背景减除)", stats_b, elapsed_b, idx)

print()
print("=" * 60)
print("对照结论")
print("=" * 60)
print("  速度      : 帧差法 %.1f fps 快于 MOG2 %.1f fps(约 %.1f 倍)"
      % (idx / elapsed_b, idx / elapsed_a, elapsed_a / elapsed_b))
print("  检出完整度: MOG2 有 %.1f%% 的帧检出目标,帧差法 %.1f%%"
      % ((bx_a > 0).mean() * 100, (bx_b > 0).mean() * 100))
print("  框的干净度: MOG2 平均 %.2f 个框,帧差法 %.2f 个 —— 帧差法多出的部分"
      % (bx_a.mean(), bx_b.mean()))
print("              大多是同一个目标被拆成两半(拖影),不是真的多了一辆车")
print("  框的大小  : MOG2 平均 %.0f,帧差法 %.0f 平方像素"
      % (ba_a.mean(), ba_b.mean()))

# %% [markdown]
# ## 第 5 格:并排对比视频
# 把两种方法的结果拼在一起写成第三个视频 —— 最直观的证据,提交时也最方便。

# %% 生成并排视频
VIDEO_AB = os.path.join(OUT_DIR, "task2_04_03_side_by_side.mp4")
half_w, half_h = W // 2, H // 2
writer_ab = cv2.VideoWriter(VIDEO_AB, cv2.VideoWriter_fourcc(*"mp4v"),
                            FPS, (half_w * 2, half_h))

cap_a = cv2.VideoCapture(VIDEO_A)
cap_b = cv2.VideoCapture(VIDEO_B)
while True:
    ok_a, fa = cap_a.read()
    ok_b, fb = cap_b.read()
    if not (ok_a and ok_b):
        break
    fa = cv2.resize(fa, (half_w, half_h))
    fb = cv2.resize(fb, (half_w, half_h))
    writer_ab.write(np.hstack([fa, fb]))

cap_a.release()
cap_b.release()
writer_ab.release()
print("并排对比视频:", VIDEO_AB)

# %% [markdown]
# ## 第 6 格:关键帧对比图(报告里放这张)

# %% 关键帧对比图
moments = sorted(set(samples_a) & set(samples_b))
fig, axes = plt.subplots(len(moments), 3, figsize=(16, 4.2 * len(moments)))
if len(moments) == 1:
    axes = axes.reshape(1, -1)

for row, m_idx in enumerate(moments):
    frame_a, mask_a = samples_a[m_idx]
    frame_b, mask_b = samples_b[m_idx]
    axes[row, 0].imshow(to_rgb(frame_a))
    axes[row, 0].set_title("原图(第 %d 帧)" % m_idx)
    axes[row, 1].imshow(mask_a, cmap="gray")
    axes[row, 1].set_title("MOG2 前景掩码")
    axes[row, 2].imshow(mask_b, cmap="gray")
    axes[row, 2].set_title("帧差法前景掩码")
    for ax in axes[row]:
        ax.axis("off")

fig.suptitle("MOG2 与帧差法在不同时刻的检测结果", fontsize=15)
fig.tight_layout()
frame_cmp = os.path.join(OUT_DIR, "task2_04_04_frame_compare.png")
fig.savefig(frame_cmp, dpi=105)
plt.close(fig)
print("关键帧对比图:", frame_cmp)

# %% [markdown]
# ## 实验 1:MOG2 的预热期
# **先写下你的预测**:MOG2 从第一帧就能正常输出吗?前几十帧的掩码是什么样子?

# %% 实验 1:预热期
cap = cv2.VideoCapture(VIDEO_FILE)
fgbg_warm = cv2.createBackgroundSubtractorMOG2(
    history=MOG2_HISTORY, varThreshold=MOG2_THRESHOLD, detectShadows=MOG2_DETECT_SHADOWS)

warm_masks = []
warm_counts = []
for i in range(60):
    ok, frame = cap.read()
    if not ok:
        break
    m = fgbg_warm.apply(frame)
    if MOG2_DETECT_SHADOWS:
        m[m == 127] = 0
    warm_counts.append(int(np.count_nonzero(m)))
    if i in (0, 5, 20, 50):
        warm_masks.append((i, m.copy()))
cap.release()

print("MOG2 前 12 帧的前景像素数:", warm_counts[:12])
print("第 0 帧 %d,第 10 帧 %d,第 30 帧 %d,第 59 帧 %d"
      % (warm_counts[0], warm_counts[10], warm_counts[30], warm_counts[59]))

fig, axes = plt.subplots(1, len(warm_masks), figsize=(4 * len(warm_masks), 3.5))
for ax, (i, m) in zip(axes, warm_masks):
    ax.imshow(m, cmap="gray")
    ax.set_title("第 %d 帧" % i)
    ax.axis("off")
fig.suptitle("实验 1:MOG2 的预热期(前几十帧背景模型还没建好)")
fig.tight_layout()
warm_path = os.path.join(OUT_DIR, "task2_04_exp1_warmup.png")
fig.savefig(warm_path, dpi=105)
plt.close(fig)
print("已保存:", warm_path)

# %% [markdown]
# ## 实验 2:阴影要不要算成前景?
# `detectShadows=True` 时,阴影像素的值是 **127**(不是 255)。
# **先写下你的预测**:如果不清掉 127,检测框会变大还是变小?

# %% 实验 2:阴影处理
cap = cv2.VideoCapture(VIDEO_FILE)
fgbg_shadow = cv2.createBackgroundSubtractorMOG2(
    history=MOG2_HISTORY, varThreshold=MOG2_THRESHOLD, detectShadows=True)

shadow_pair = None
for i in range(200):
    ok, frame = cap.read()
    if not ok:
        break
    m = fgbg_shadow.apply(frame)
    if i == 150:
        mask_with = m.copy()
        mask_without = m.copy()
        mask_without[mask_without == 127] = 0
        _, clean_with, boxes_with = annotate(frame.copy(), mask_with, "keep shadows")
        _, clean_without, boxes_without = annotate(frame.copy(), mask_without, "remove shadows")
        shadow_pair = (clean_with, boxes_with, clean_without, boxes_without)
        break
cap.release()

if shadow_pair:
    clean_with, boxes_with, clean_without, boxes_without = shadow_pair
    print("保留阴影:前景像素 %d 个,画出 %d 个框" % (np.count_nonzero(clean_with), len(boxes_with)))
    print("去掉阴影:前景像素 %d 个,画出 %d 个框" % (np.count_nonzero(clean_without), len(boxes_without)))
    print()
    print("结论:阴影(值 127)不清掉会和目标连成一片,导致框变大,甚至把多个目标粘成一个。")

# %% [markdown]
# ## 实验 3:把 MOG2 的敏感度调到两个极端
# `varThreshold` 越小越敏感。**先写下你的预测**:调到 4 会怎样?调到 64 又会怎样?

# %% 实验 3:varThreshold
cap = cv2.VideoCapture(VIDEO_FILE)
frames_cache = []
for i in range(200):
    ok, frame = cap.read()
    if not ok:
        break
    frames_cache.append(frame)
cap.release()

for th in (4, 16, 64):
    fgbg_t = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=th, detectShadows=True)
    m = None
    for f in frames_cache:
        m = fgbg_t.apply(f)
        m[m == 127] = 0
    clean = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((OPEN_KERNEL, OPEN_KERNEL), np.uint8))
    print("varThreshold = %2d -> 前景像素 %6d" % (th, np.count_nonzero(clean)))
print()
print("结论:阈值太小会把噪声、光照抖动都当成前景;太大又会漏掉真实目标。")

# %% [markdown]
# ## 实验 4:同样是检车,两种方法画出的"框"有什么不同?
#
# 你可能会想:帧差法只比较相邻两帧,那物体停下不动时岂不是检不到了?
# 这个担心是对的 —— 但**在这段视频里不成立**:车一直在开,所以帧差法也一直检得到。
# 实测"什么都没检测到的帧数",两种方法都是 0。
#
# 那差别体现在哪?看两个数字:
#
# | 指标 | 说明 |
# |---|---|
# | 平均目标个数 | 帧差法比 MOG2 多出约 1 个 |
# | 平均框总面积 | 帧差法的框更大 |
#
# **原因**:帧差法检测的是"两帧之间**发生过变化**的区域"。
# 一辆快速行驶的车,在第 N 帧和第 N+1 帧位置不同,于是它**前后两个位置之间的整条区域**
# 都被标成前景,形状被拉长(拖影),有时还会被切成两块、变成两个框。
#
# 而 MOG2 检测的是"和**背景模型**不同的区域",只要车还在背景之上,
# 报出来的就是车本身的位置,形状更贴近物体。

# %% 实验 4:框的数量与大小
print("帧差法前景像素(每 50 帧取一次):")
print("  ", [int(v) for v in fg_b[::50][:12]])
print("MOG2 前景像素(每 50 帧取一次):")
print("  ", [int(v) for v in fg_a[::50][:12]])

zero_b = int((fg_b == 0).sum())
zero_a = int((fg_a == 0).sum())
print()
print("'什么都没检测到'的帧数:")
print("  帧差法: %d 帧(%.1f%%)" % (zero_b, zero_b / len(fg_b) * 100))
print("  MOG2  : %d 帧(%.1f%%)" % (zero_a, zero_a / len(fg_a) * 100))
print("  (这段视频里车一直在动,所以两种方法都没出现空白帧)")
print()
print("框的数量:MOG2 平均 %.2f 个,帧差法平均 %.2f 个" % (bx_a.mean(), bx_b.mean()))
print("框的大小:MOG2 平均 %.0f 平方像素,帧差法平均 %.0f 平方像素"
      % (ba_a.mean(), ba_b.mean()))
print()
print("结论:帧差法多出来的那些框,大多是同一个目标被拖影切成两半,")
print("      并不是真的多了一辆车 —— 这是它'只比较相邻两帧'带来的固有结果。")
print()
print("再想一步:如果哪天画面里有一辆车停在路边不动,")
print("      帧差法会完全看不到它,而 MOG2 会一直把它报出来。")
print("      这就是两种判据的本质差别:比'上一帧'还是比'背景'。")

# %% [markdown]
# ## 收尾
# 产出文件:
#
# | 文件 | 内容 |
# |---|---|
# | `task2_04_01_mog2.mp4` | MOG2 检测结果视频 |
# | `task2_04_02_framediff.mp4` | 帧差法检测结果视频 |
# | `task2_04_03_side_by_side.mp4` | 两种方法并排对比(**交这个最直观**) |
# | `task2_04_04_frame_compare.png` | 关键帧对比图(放报告) |
# | `task2_04_exp1_warmup.png` | MOG2 预热期示意图 |
#
# 报告里要填:
# 1. 两种方法的优缺点对比表
# 2. 观察到的现象(用上面打印的数字支撑)
# 3. 结论:什么时候该用哪个
