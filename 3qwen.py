import cv2
import subprocess
import base64
import threading
import time
from queue import Queue
from ultralytics import YOLO
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ==============================================
# 阿里云百炼 Qwen2.5-Omni-7B 配置
# ==============================================
chat_model = ChatOpenAI(
    api_key="sk-4780053072f645de9179696c3262b924",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen2.5-omni-7b",
    temperature=0.3,
    max_tokens=256,
    timeout=8,
)

# ==============================================
# 全局变量
# ==============================================
VIDEO_PATH = "1.mp4"
RTSP_URL = "rtsp://127.0.0.1:554/stream"

# 动作识别结果（线程安全）
current_action = ""
action_lock = threading.Lock()

# 帧队列：主线程生产，AI识别线程消费
action_queue = Queue(maxsize=3)

# 推流帧队列：主线程生产，推流线程消费
stream_queue = Queue(maxsize=60)  # 缓冲 60 帧，约 2 秒

# ==============================================
# 工具函数
# ==============================================
def frame_to_base64(frame):
    """视频帧转 Base64"""
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"


def put_chinese_text(img, text, position, font_size=36, color=(0, 255, 0)):
    """在 OpenCV 图像上绘制中文文字"""
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    font = None
    for fp in font_paths:
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except (IOError, OSError):
            continue
    if font is None:
        font = ImageFont.load_default()

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def detect_action(frame):
    """调用 Qwen 识别动作"""
    global current_action
    try:
        img_base64 = frame_to_base64(frame)
        messages = [HumanMessage(
            content=[
                {"type": "text", "text": "用8个字以内简洁描述图中人物的动作"},
                {"type": "image_url", "image_url": {"url": img_base64}},
            ]
        )]
        res = chat_model.invoke(messages)
        action_text = res.content.strip()
        print(f"[AI] 识别结果：{action_text}")
        with action_lock:
            current_action = action_text
    except Exception as e:
        print(f"[AI] 识别失败：{e}")
        with action_lock:
            current_action = ""


def action_worker():
    """动作识别子线程"""
    while True:
        frame = action_queue.get()
        if frame is None:
            break
        detect_action(frame)
        action_queue.task_done()


def stream_worker(ffmpeg_process):
    """推流子线程：专门负责写入 FFmpeg stdin，阻塞不影响主线程"""
    while True:
        item = stream_queue.get()
        if item is None:
            break
        try:
            ffmpeg_process.stdin.write(item)
        except BrokenPipeError:
            print("[推流] 管道断裂")
            break
        except Exception as e:
            print(f"[推流] 写入异常: {e}")
            break
        stream_queue.task_done()
    try:
        ffmpeg_process.stdin.close()
    except Exception:
        pass


# ==============================================
# FFmpeg RTSP 推流命令
# ==============================================
ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo", "-pix_fmt", "bgr24",
    "-s", "720x1280", "-r", "25", "-i", "-",
    "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
    "-b:v", "1500k",
    "-bufsize", "512k",
    "-f", "rtsp",
    "-rtsp_transport", "tcp",
    RTSP_URL,
]

# ==============================================
# 主程序
# ==============================================
if __name__ == "__main__":
    print("=" * 50)
    print("  动作识别 + RTSP 推流（V3 线程隔离版）")
    print("=" * 50)

    # 1. 加载 YOLO 模型
    print("[启动] 正在加载 YOLO 模型...")
    yolo = YOLO("yolov8n-pose.pt")
    print("[启动] YOLO 模型加载完成")

    # 2. 打开视频
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {VIDEO_PATH}")
        exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[启动] 视频: {w}x{h}, {fps}fps, 共{total}帧")

    # 3. 启动 FFmpeg 推流
    print(f"[启动] 正在启动 FFmpeg 推流到 {RTSP_URL} ...")
    ffmpeg = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    # 读取 stderr 的线程（防止管道满阻塞）
    def read_stderr():
        try:
            for line in ffmpeg.stderr:
                pass
        except:
            pass
    threading.Thread(target=read_stderr, daemon=True).start()

    # 4. 启动推流写入线程（关键！stdin.write 不再阻塞主线程）
    stream_thread = threading.Thread(target=stream_worker, args=(ffmpeg,), daemon=True)
    stream_thread.start()

    # 5. 启动 AI 识别线程
    ai_thread = threading.Thread(target=action_worker, daemon=True)
    ai_thread.start()

    print("[启动] 所有线程已启动")
    print("=" * 50)

    count = 0
    last_fps_time = time.time()
    last_fps_count = 0

    while True:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("[结束] 视频播放完毕")
            break

        # ---- YOLO 骨骼检测 ----
        yolo_start = time.time()
        try:
            results = yolo(frame, conf=0.5, verbose=False)
            annotated = results[0].plot()
        except Exception as e:
            annotated = frame
        yolo_time = time.time() - yolo_start

        # ---- 每 15 帧提交动作识别（异步）----
        count += 1
        if count % 15 == 0:
            if action_queue.qsize() < 2:
                action_queue.put(annotated.copy())

        # ---- 叠加动作文字 ----
        with action_lock:
            action_text = current_action
        if action_text:
            annotated = put_chinese_text(annotated, action_text, (10, 40))

        # ---- 推流：放入队列，由推流线程异步写入 ----
        if stream_queue.qsize() < 50:  # 队列未满才放入
            stream_queue.put(annotated.tobytes())
        else:
            print(f"[警告] 推流队列积压({stream_queue.qsize()})，跳帧")

        # ---- FPS 监控 ----
        now = time.time()
        if now - last_fps_time >= 1.0:
            real_fps = (count - last_fps_count) / (now - last_fps_time)
            print(f"[性能] FPS={real_fps:.1f}, YOLO={yolo_time*1000:.0f}ms, "
                  f"推流队列={stream_queue.qsize()}, AI队列={action_queue.qsize()}, "
                  f"帧={count}/{total}")
            last_fps_time = now
            last_fps_count = count

        # ---- 预览窗口 ----
        cv2.imshow("动作识别", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[退出] 用户按下 Q")
            break

    # ==============================================
    # 释放资源
    # ==============================================
    print("[清理] 正在释放资源...")
    action_queue.put(None)
    stream_queue.put(None)
    ai_thread.join(timeout=5)
    stream_thread.join(timeout=5)

    cap.release()
    cv2.destroyAllWindows()

    try:
        ffmpeg.terminate()
        ffmpeg.wait(timeout=5)
    except Exception:
        pass

    print("[完成] 所有资源已释放")
