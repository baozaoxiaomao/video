import cv2
import subprocess
import numpy as np
from ultralytics import YOLO

# --------------------------
# 1. 配置
video_path = "1.mp4"  # 你的本地视频
rtsp_output_url = "rtsp://127.0.0.1:25544/output"

# FFmpeg推流配置（适配彩色BGR帧，和OpenCV输出格式匹配）
ffmpeg_cmd = [
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-pix_fmt', 'bgr24',  # 适配OpenCV的BGR彩色帧
    '-s', '720x1280',     # 和你的视频分辨率一致
    '-r', '25',           # 匹配视频帧率，避免掉帧
    '-i', '-',
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-tune', 'zerolatency',
    '-crf', '28',
    '-threads', '4',
    '-f', 'rtsp',
    rtsp_output_url
]

# --------------------------
# 2. 换更快的模型（CPU首选，帧率直接翻倍）
# 第一次运行会自动下载，或者你也可以手动把n版模型放到这个路径
model = YOLO("yolov8n-pose.pt")  

# 打开本地视频
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"无法打开本地视频: {video_path}")
    exit()

# 启动推流
process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# 处理视频+骨骼点识别
while True:
    ret, frame = cap.read()
    if not ret:
        print("视频处理完成！")
        break

    # 推理骨骼点
    results = model(frame, conf=0.5)
    frame_with_pose = results[0].plot()  # 直接用彩色帧，不转灰度

    # 推流
    try:
        process.stdin.write(frame_with_pose.tobytes())
    except BrokenPipeError:
        print("推流关闭")
        break

    # 可选：本地预览（按q退出，不影响推流）
    cv2.imshow("实时预览", frame_with_pose)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理资源
cap.release()
cv2.destroyAllWindows()
if process.poll() is None:
    process.stdin.close()
    process.wait() 