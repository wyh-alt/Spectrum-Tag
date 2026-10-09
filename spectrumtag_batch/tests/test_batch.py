"""端到端自检 —— 造几个测试媒体，跑完整批处理，再回头验证音频里真的有水印。

直接运行：``python -m spectrumtag_batch.tests.test_batch``

验证思路是"用结果说话"：把处理后的音频重新做一遍时频分析，检查印章区间内
目标频段的能量是否确实被压了下去，而区间外基本没被动过。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

from ..batch import (
    SUPPORTED_EXTENSIONS,
    ItemStatus,
    TextMode,
    collect_files,
    run_batch,
)
from ..core.params import (
    DspSpec,
    EngraveMode,
    LoopSpec,
    PatternSource,
    PatternSpec,
    PlacementSpec,
    PositionMode,
    RenderJob,
)
from ..media.audio_io import probe_audio, read_audio
from ..media.video_io import find_ffmpeg

FREQ_LOW_NORM = 0.10
FREQ_HIGH_NORM = 0.25
STAMP_START = 0.5      # 秒
STAMP_DURATION = 1.0   # 秒


def _make_job(text: str = "TEST") -> RenderJob:
    return RenderJob(
        pattern=PatternSpec(source=PatternSource.TEXT, text=text, weight=0.25),
        dsp=DspSpec(fft_size=4096, mode=EngraveMode.CUT, strength=1.0),
        placement=PlacementSpec(
            freq_low_norm=FREQ_LOW_NORM,
            freq_high_norm=FREQ_HIGH_NORM,
            start=STAMP_START,
            duration=STAMP_DURATION,
            position_mode=PositionMode.ABSOLUTE,
        ),
        loop=LoopSpec(enabled=False),
    )


def _write_test_wav(path: str, seconds: float, sample_rate: int, channels: int) -> None:
    rng = np.random.default_rng(3)
    frames = int(seconds * sample_rate)
    data = (rng.standard_normal((frames, channels)) * 0.15).astype(np.float32)
    sf.write(path, data, sample_rate, subtype="PCM_16")


def _band_rms(x: np.ndarray, sr: int, lo_hz: float, hi_hz: float) -> float:
    """用 Welch 风格的分段谱估计来量某个频段的平均能量。"""
    spec = np.fft.rfft(x * np.hanning(len(x)))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    sel = (freqs >= lo_hz) & (freqs <= hi_hz)
    return float(np.sqrt(np.mean(np.abs(spec[sel]) ** 2)))


def check_collect_files(tmp: str) -> bool:
    """文件夹展开 + 去重 + 只收受支持的类型。"""
    nested = os.path.join(tmp, "inputs", "sub")
    os.makedirs(nested, exist_ok=True)
    _write_test_wav(os.path.join(tmp, "inputs", "a.wav"), 0.2, 44100, 1)
    _write_test_wav(os.path.join(nested, "b.wav"), 0.2, 44100, 1)
    with open(os.path.join(tmp, "inputs", "notes.txt"), "w", encoding="utf-8") as fh:
        fh.write("ignore me")

    found = collect_files([os.path.join(tmp, "inputs")])
    names = sorted(os.path.basename(p) for p in found)

    # 重复传入同一个文件夹不应产生重复项
    deduped = collect_files([os.path.join(tmp, "inputs"), os.path.join(tmp, "inputs")])

    # 单个文件也必须过扩展名 —— 曾经拖进一张图片也会被当成待处理素材
    stray = os.path.join(tmp, "inputs", "logo.png")
    with open(stray, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
    single_rejected = collect_files([stray]) == []
    mixed = collect_files([os.path.join(tmp, "inputs", "a.wav"), stray])
    mixed_ok = [os.path.basename(p) for p in mixed] == ["a.wav"]

    ok = (
        names == ["a.wav", "b.wav"]
        and len(deduped) == 2
        and single_rejected
        and mixed_ok
    )
    print(f"[1] 文件收集                找到 = {names}  去重后 = {len(deduped)}  "
          f"单文件过滤 = {single_rejected}  混合拖入只收音频 = {mixed_ok}"
          f"  ->  {'通过' if ok else '失败'}")
    return ok


def check_batch_audio_only(tmp: str) -> bool:
    """纯音频批量：三个不同参数的 WAV 全部处理成功。"""
    src = os.path.join(tmp, "audio_only")
    out = os.path.join(tmp, "out_audio")
    os.makedirs(src, exist_ok=True)
    _write_test_wav(os.path.join(src, "mono_44k.wav"), 2.0, 44100, 1)
    _write_test_wav(os.path.join(src, "stereo_48k.wav"), 2.0, 48000, 2)
    _write_test_wav(os.path.join(src, "mono_22k.wav"), 2.0, 22050, 1)

    files = collect_files([src])
    result = run_batch(files, _make_job(), out)

    if result.succeeded != 3:
        for it in result.items:
            if it.status is not ItemStatus.DONE:
                print(f"    ! {it.name}: {it.message}")

    # 输出应保持源采样率与声道数
    checks = []
    for name in ("mono_44k", "stereo_48k", "mono_22k"):
        src_probe = probe_audio(os.path.join(src, f"{name}.wav"))
        # 输出沿用源文件主名（后缀为空）；源在 src/、输出在 out/，不会撞车
        out_path = os.path.join(out, f"{name}.wav")
        if not os.path.exists(out_path):
            checks.append(False)
            continue
        out_probe = probe_audio(out_path)
        checks.append(
            out_probe.sample_rate == src_probe.sample_rate
            and out_probe.channels == src_probe.channels
            and out_probe.frames == src_probe.frames
        )

    ok = result.succeeded == 3 and all(checks)
    print(f"[2] 纯音频批量              成功 {result.succeeded}/3  "
          f"参数保持 = {checks}  ->  {'通过' if ok else '失败'}")
    return ok


def _make_solid_image(path: str, coverage: float = 0.98) -> None:
    """造一张几乎全黑的图 —— 亮度模式下"暗像素 = 图案本体"，覆盖率≈coverage。"""
    from PIL import Image, ImageDraw

    side = 200
    margin = max(1, int(side * (1.0 - coverage) / 2))
    img = Image.new("RGB", (side, side), (255, 255, 255))
    ImageDraw.Draw(img).rectangle(
        [margin, margin, side - 1 - margin, side - 1 - margin], fill=(0, 0, 0)
    )
    img.save(path, format="PNG")


def check_watermark_actually_embedded(tmp: str) -> bool:
    """关键验证：近乎实心的图案应把整个目标频段压下去。"""
    src = os.path.join(tmp, "verify_src")
    out = os.path.join(tmp, "out_verify")
    os.makedirs(src, exist_ok=True)
    path = os.path.join(src, "target.wav")
    image_path = os.path.join(src, "solid.png")
    _write_test_wav(path, 2.5, 44100, 1)
    _make_solid_image(image_path)

    job = RenderJob(
        pattern=PatternSpec(source=PatternSource.IMAGE, image_path=image_path),
        dsp=DspSpec(fft_size=4096, mode=EngraveMode.CUT, strength=1.0),
        placement=PlacementSpec(
            freq_low_norm=FREQ_LOW_NORM,
            freq_high_norm=FREQ_HIGH_NORM,
            start=STAMP_START,
            duration=STAMP_DURATION,
            position_mode=PositionMode.ABSOLUTE,
        ),
        loop=LoopSpec(enabled=False),
    )

    result = run_batch([path], job, out)
    if result.succeeded != 1:
        print(f"[3] 水印生效验证            处理失败：{result.items[0].message}"
              f"  ->  失败")
        return False

    original = read_audio(path)
    tagged = read_audio(result.items[0].output_path)
    sr = original.sample_rate
    # 测量区间往里收一点：图案描边会占住频段最外那一圈，而那一圈是按设计
    # 被**抬起来**的（峰谷对），量进去只会把衰减幅度冲淡
    lo_hz = (FREQ_LOW_NORM + 0.01) * sr / 2
    hi_hz = (FREQ_HIGH_NORM - 0.01) * sr / 2

    def slice_at(data, t0, t1):
        a, b = int(t0 * sr), int(t1 * sr)
        return data.samples[0, a:b]

    # 印章中段（避开两端各一个帧长 ≈93ms 的稀释区）
    inside_before = _band_rms(slice_at(original, 0.8, 1.2), sr, lo_hz, hi_hz)
    inside_after = _band_rms(slice_at(tagged, 0.8, 1.2), sr, lo_hz, hi_hz)
    inside_db = 20 * np.log10(max(inside_after, 1e-30) / max(inside_before, 1e-30))

    # 印章之外应几乎没变
    outside_delta = float(
        np.max(np.abs(tagged.samples[0, int(1.8 * sr):int(2.4 * sr)]
                      - original.samples[0, int(1.8 * sr):int(2.4 * sr)]))
    )

    # 时长必须严丝合缝（帧几何对齐的回归防线）
    length_ok = tagged.frames == original.frames

    # 阈值留出余量：图案边缘 2%、保护带与 3-tap 平滑都会留下一点残留能量
    ok = inside_db < -18.0 and outside_delta < 1e-5 and length_ok
    print(f"[3] 水印生效验证            印章内 {inside_db:+.1f} dB  "
          f"印章外偏差 {outside_delta:.2e}  时长一致 = {length_ok}"
          f"  ->  {'通过' if ok else '失败'}")
    return ok


def check_text_pattern_locality(tmp: str) -> bool:
    """文字图案：改动必须只发生在印章区间内，且确实改变了音频。"""
    src = os.path.join(tmp, "text_src")
    out = os.path.join(tmp, "out_text")
    os.makedirs(src, exist_ok=True)
    path = os.path.join(src, "text_target.wav")
    _write_test_wav(path, 2.5, 44100, 1)

    result = run_batch([path], _make_job("水印"), out)
    if result.succeeded != 1:
        print(f"[3b] 文字图案定位            处理失败：{result.items[0].message}"
              f"  ->  失败")
        return False

    original = read_audio(path).samples[0]
    tagged = read_audio(result.items[0].output_path).samples[0]
    sr = 44100

    def peak_delta(t0: float, t1: float) -> float:
        a, b = int(t0 * sr), int(t1 * sr)
        return float(np.max(np.abs(tagged[a:b] - original[a:b])))

    inside = peak_delta(0.7, 1.3)      # 印章区间 [0.5, 1.5] 的中段
    before = peak_delta(0.05, 0.45)    # 印章之前
    after = peak_delta(1.6, 2.4)       # 印章之后

    ok = inside > 1e-3 and before < 1e-5 and after < 1e-5
    print(f"[3b] 文字图案定位            印章内变化 {inside:.2e}  "
          f"前 {before:.2e}  后 {after:.2e}  ->  {'通过' if ok else '失败'}")
    return ok


def check_video_input(tmp: str) -> bool:
    """视频输入：抽音轨 → 水印 → 输出 WAV。"""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("[4] 视频输入                未找到 ffmpeg  ->  跳过")
        return True

    src = os.path.join(tmp, "video_src")
    out = os.path.join(tmp, "out_video")
    os.makedirs(src, exist_ok=True)
    video = os.path.join(src, "clip.avi")

    cmd = [
        ffmpeg, "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-c:v", "mpeg4", "-c:a", "pcm_s16le", "-shortest", video,
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not os.path.exists(video):
        print(f"[4] 视频输入                无法生成测试视频  ->  跳过"
              f" ({proc.stderr.decode('utf-8', 'replace')[:80]})")
        return True

    result = run_batch([video], _make_job(), out)
    item = result.items[0]
    if item.status is not ItemStatus.DONE:
        print(f"[4] 视频输入                失败：{item.message}  ->  失败")
        return False

    probe = probe_audio(item.output_path)
    ok = probe.frames > 0 and probe.sample_rate > 0 and item.duration_sec > 1.5
    print(f"[4] 视频输入                输出 {probe.frames} 帧 @ {probe.sample_rate} Hz"
          f"  音轨时长 {item.duration_sec:.2f}s  ->  {'通过' if ok else '失败'}")
    return ok


def check_filename_text_mode(tmp: str) -> bool:
    """按文件名生成文字：不同文件应得到不同的图案。"""
    src = os.path.join(tmp, "name_src")
    out = os.path.join(tmp, "out_name")
    os.makedirs(src, exist_ok=True)
    for name in ("Alpha", "Beta"):
        _write_test_wav(os.path.join(src, f"{name}.wav"), 1.5, 44100, 1)

    files = collect_files([src])
    result = run_batch(
        files, _make_job(), out, text_mode=TextMode.FROM_FILENAME
    )

    if result.succeeded != 2:
        print(f"[5] 按文件名生成文字        成功 {result.succeeded}/2  ->  失败")
        return False

    # 两份输出内容应当不同（文字不同 → 图案不同 → 音频不同）
    a = read_audio(result.items[0].output_path).samples
    b = read_audio(result.items[1].output_path).samples
    differ = float(np.max(np.abs(a - b))) > 1e-6

    # 但同一份源文件在固定文字模式下应当可复现
    out2 = os.path.join(tmp, "out_name2")
    again = run_batch([files[0]], _make_job("Alpha"), out2)
    same_text = float(
        np.max(np.abs(read_audio(again.items[0].output_path).samples - a))
    ) < 1e-6

    ok = differ and same_text
    print(f"[5] 按文件名生成文字        两文件不同 = {differ}  同文字可复现 = {same_text}"
          f"  ->  {'通过' if ok else '失败'}")
    return ok


def check_output_naming(tmp: str) -> bool:
    """输出沿用源文件名；但绝不能把源文件本身覆盖掉。"""
    src = os.path.join(tmp, "naming")
    dst = os.path.join(tmp, "naming_out")
    os.makedirs(src, exist_ok=True)

    # 1) 输出到另一个目录 → 文件名与源完全一致
    path = os.path.join(src, "song.wav")
    _write_test_wav(path, 1.0, 44100, 1)
    first = run_batch([path], _make_job(), dst)
    same_name = first.succeeded == 1 and os.path.exists(os.path.join(dst, "song.wav"))

    # 2) 输出目录就是源目录 → 必须退让成 _tagged，原始素材保持原样
    other = os.path.join(src, "other.wav")
    _write_test_wav(other, 1.0, 44100, 1)
    original = read_audio(other).samples.copy()

    second = run_batch([other], _make_job(), src)
    preserved = np.array_equal(read_audio(other).samples, original)
    fallback = os.path.join(src, "other_tagged.wav")

    ok = same_name and second.succeeded == 1 and preserved and os.path.exists(fallback)
    print(f"[8] 输出命名                同名导出 = {same_name}  "
          f"源目录退让 = {os.path.exists(fallback)}  源文件未被覆盖 = {preserved}"
          f"  ->  {'通过' if ok else '失败'}")
    return ok


def check_output_formats(tmp: str) -> bool:
    """音频格式选项：既能跟着源文件走，也能指定成别的容器。"""
    from ..core.params import AudioFormat, OutputSpec

    src = os.path.join(tmp, "fmt_src")
    os.makedirs(src, exist_ok=True)
    flac_path = os.path.join(src, "song.flac")
    rng = np.random.default_rng(3)
    data = (rng.standard_normal((44100, 1)) * 0.15).astype(np.float32)
    sf.write(flac_path, data, 44100, format="FLAC")

    expected = {
        AudioFormat.SOURCE: ".flac",
        AudioFormat.WAV: ".wav",
        AudioFormat.MP3: ".mp3",
        AudioFormat.OGG: ".ogg",
    }
    actual: dict[str, str] = {}
    failures: list[str] = []

    for fmt, want_ext in expected.items():
        out = os.path.join(tmp, f"fmt_{fmt.value}")
        result = run_batch([flac_path], _make_job(), out, OutputSpec(audio_format=fmt))
        item = result.items[0]
        got_ext = os.path.splitext(item.output_path)[1] if item.output_path else ""
        actual[fmt.value] = got_ext or f"失败({item.message[:30]})"
        if item.status is not ItemStatus.DONE or got_ext != want_ext:
            failures.append(f"{fmt.value}: 期望 {want_ext} 得到 {actual[fmt.value]}")

    ok = not failures
    detail = "  ".join(f"{k}→{v}" for k, v in actual.items())
    print(f"[9] 输出格式                {detail}  ->  {'通过' if ok else '失败'}")
    return ok


def check_video_mux(tmp: str) -> bool:
    """合成视频：画面原样保留，音轨换成处理过的。"""
    from ..core.params import OutputSpec, VideoFormat

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("[10] 合成视频               未找到 ffmpeg  ->  跳过")
        return True

    src = os.path.join(tmp, "mux_src")
    os.makedirs(src, exist_ok=True)
    video = os.path.join(src, "clip.avi")
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=12",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:v", "mpeg4", "-c:a", "pcm_s16le", "-shortest", video],
        capture_output=True, check=False,
    )
    if proc.returncode != 0 or not os.path.exists(video):
        print("[10] 合成视频               无法生成测试视频  ->  跳过")
        return True

    checks: dict[str, bool] = {}
    for label, spec, want_ext, want_video in (
        ("仅音频", OutputSpec(mux_video=False), ".wav", False),
        ("与源一致", OutputSpec(mux_video=True, video_format=VideoFormat.SOURCE), ".avi", True),
        ("转 mp4", OutputSpec(mux_video=True, video_format=VideoFormat.MP4), ".mp4", True),
    ):
        out = os.path.join(tmp, f"mux_{want_ext.lstrip('.')}_{abs(hash(label)) % 1000}")
        result = run_batch([video], _make_job(), out, spec)
        item = result.items[0]
        if item.status is not ItemStatus.DONE:
            checks[label] = False
            continue

        # 用 ffmpeg 探一遍有没有视频流 —— 光看扩展名说明不了问题
        probe = subprocess.run(
            [ffmpeg, "-v", "error", "-i", item.output_path, "-map", "0:v:0",
             "-f", "null", "-"],
            capture_output=True, check=False,
        )
        ext_ok = os.path.splitext(item.output_path)[1] == want_ext
        has_video = probe.returncode == 0
        checks[label] = ext_ok and (has_video is want_video)

    ok = all(checks.values())
    detail = "  ".join(f"{k}={v}" for k, v in checks.items())
    print(f"[10] 合成视频               {detail}  ->  {'通过' if ok else '失败'}")
    return ok


def check_failure_isolation(tmp: str) -> bool:
    """坏文件不该中断整批。"""
    src = os.path.join(tmp, "bad_src")
    out = os.path.join(tmp, "out_bad")
    os.makedirs(src, exist_ok=True)
    _write_test_wav(os.path.join(src, "good1.wav"), 1.0, 44100, 1)
    # 伪造一个扩展名合法但内容损坏的文件
    with open(os.path.join(src, "broken.wav"), "wb") as fh:
        fh.write(b"not a real wav file at all")
    _write_test_wav(os.path.join(src, "good2.wav"), 1.0, 44100, 1)

    files = collect_files([src])
    result = run_batch(files, _make_job(), out)

    ok = result.succeeded == 2 and len(result.failed) == 1
    print(f"[6] 单文件失败隔离          成功 {result.succeeded}  失败 {len(result.failed)}"
          f"  ->  {'通过' if ok else '失败'}")
    return ok


def check_cancel(tmp: str) -> bool:
    """取消后应立即停下，未处理的文件标为跳过。"""
    src = os.path.join(tmp, "cancel_src")
    out = os.path.join(tmp, "out_cancel")
    os.makedirs(src, exist_ok=True)
    for i in range(4):
        _write_test_wav(os.path.join(src, f"c{i}.wav"), 1.5, 44100, 1)

    files = collect_files([src])
    seen = {"n": 0}

    def on_progress(_):
        seen["n"] += 1
        return seen["n"] < 3      # 第 3 次回调起请求取消

    result = run_batch(files, _make_job(), out, progress=on_progress)
    processed = sum(1 for it in result.items if it.status is ItemStatus.DONE)

    ok = result.cancelled and processed < len(files)
    print(f"[7] 取消                    已取消 = {result.cancelled}  "
          f"完成 {processed}/{len(files)}  ->  {'通过' if ok else '失败'}")
    return ok


def main() -> int:
    print("=" * 68)
    print("批处理 —— 端到端自检")
    print(f"支持的扩展名：{len(SUPPORTED_EXTENSIONS)} 种")
    print("=" * 68)

    tmp = tempfile.mkdtemp(prefix="stbatch_test_")
    try:
        results = [
            check_collect_files(tmp),
            check_batch_audio_only(tmp),
            check_watermark_actually_embedded(tmp),
            check_text_pattern_locality(tmp),
            check_video_input(tmp),
            check_filename_text_mode(tmp),
            check_output_naming(tmp),
            check_output_formats(tmp),
            check_video_mux(tmp),
            check_failure_isolation(tmp),
            check_cancel(tmp),
        ]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    passed = sum(results)
    print("-" * 68)
    print(f"结果：{passed}/{len(results)} 项通过")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
