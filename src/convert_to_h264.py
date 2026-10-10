r"""
把 mp4v 视频转成 H.264(用 ffmpeg)

【为什么需要】
OpenCV 的 VideoWriter 默认用 mp4v 编码(MPEG-4 Part 2)。这个编码:
  - VSCode 放不了(它的 Chromium 解码器不认)
  - 部分手机 / 浏览器 / 新版播放器也放不了
  - 老师那边可能同样打不开,有提交风险

转成 H.264 之后:VSCode 能直接预览、任何播放器都能放、体积通常还更小。

【怎么用】
    python src/convert_to_h264.py                   # 转 results 目录下所有 mp4
    python src/convert_to_h264.py 文件名.mp4         # 只转一个

转换结果放在 results/h264/ 目录里,原文件保留不动。
"""

import os
import shutil
import subprocess
import sys

import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.path.join(PROJECT_ROOT, "results")
OUT_DIR = os.path.join(VIDEO_DIR, "h264")

# ---- 画质与体积的平衡 ----
# CRF 越小画质越好、文件越大。23 是公认的"看起来无损"档位,18 约等于视觉无损。
# 想更小就调到 28;想更清晰就调到 18。
CRF = 23
PRESET = "medium"      # ultrafast / fast / medium / slow,越慢压得越小


def find_ffmpeg():
    """先看 PATH,再看好几个常见安装位置。"""
    found = shutil.which("ffmpeg")
    if found:
        return found

    candidates = [
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Links", "ffmpeg.exe"),
        r"D:\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\msys64\ucrt64\bin\ffmpeg.exe",
    ]
    # winget 的包目录里版本号会变,所以用通配找
    winget_pkgs = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages")
    if os.path.isdir(winget_pkgs):
        for root, _dirs, files in os.walk(winget_pkgs):
            if "ffmpeg.exe" in files:
                candidates.insert(0, os.path.join(root, "ffmpeg.exe"))
                break

    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def convert(ffmpeg, src, dst):
    """调用 ffmpeg 转码。返回 (是否成功, ffmpeg 的错误输出)。"""
    cmd = [
        ffmpeg, "-y",
        "-i", src,
        "-c:v", "libx264",          # H.264 编码器
        "-crf", str(CRF),           # 画质档位
        "-preset", PRESET,          # 压缩速度
        "-pix_fmt", "yuv420p",      # 兼容性最好的像素格式(手机、浏览器都认)
        "-movflags", "+faststart",  # 把索引放到文件开头,预览时不用等
        "-an",                      # 我们的视频没有声音,去掉音轨
        dst,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr


def frame_count(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def main():
    print("=" * 62)
    print("把 mp4v 视频转成 H.264(转完 VSCode 就能预览)")
    print("=" * 62)

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print()
        print("没找到 ffmpeg。请先在终端里确认这一行能跑:")
        print("    ffmpeg -version")
        print("如果报'不是内部或外部命令',说明 ffmpeg 没装好或者没加进 PATH。")
        return
    print("使用的 ffmpeg:", ffmpeg)
    os.makedirs(OUT_DIR, exist_ok=True)

    if len(sys.argv) > 1:
        names = sys.argv[1:]
    else:
        names = sorted(f for f in os.listdir(VIDEO_DIR)
                       if f.lower().endswith(".mp4") and os.path.isfile(os.path.join(VIDEO_DIR, f)))

    if not names:
        print("results 目录下没找到 mp4 文件。")
        return

    for name in names:
        src = os.path.join(VIDEO_DIR, name)
        if not os.path.exists(src):
            print("\n跳过(不存在):", name)
            continue

        dst = os.path.join(OUT_DIR, name)
        size_before = os.path.getsize(src) / 1024 / 1024
        print()
        print("转换:%s(%.1f MB)" % (name, size_before))

        ok, err = convert(ffmpeg, src, dst)
        if not ok:
            print("  失败。ffmpeg 的错误信息如下(最后 6 行):")
            for line in err.strip().splitlines()[-6:]:
                print("   ", line)
            continue

        size_after = os.path.getsize(dst) / 1024 / 1024
        n_before = frame_count(src)
        n_after = frame_count(dst)
        print("  %.1f MB -> %.1f MB(缩小到 %.0f%%)"
              % (size_before, size_after, size_after / size_before * 100))
        print("  帧数 %d -> %d %s" % (n_before, n_after, "OK" if n_before == n_after else "不一致!"))

    print()
    print("=" * 62)
    print("完成。转换后的文件都在:%s" % OUT_DIR)
    print("在 VSCode 里点开就能直接预览了;提交时打包这个目录里的文件。")
    print("=" * 62)


if __name__ == "__main__":
    main()
