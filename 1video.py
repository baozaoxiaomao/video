import cv2
import subprocess
import numpy as np

# EasyDarwin的RTSP流地址
rtsp_input_url = "rtsp://127.0.0.1:25544/input"

# 新的RTSP发布地址（你需要一个RTSP服务器来接收这个流，例如你也可以再运行一个EasyDarwin实例）
rtsp_output_url = "rtsp://127.0.0.1:25544/output"  # 假设你希望发布到这个地址和端口

# FFmpeg命令
ffmpeg_cmd = [
    'ffmpeg',
    '-y',  # 覆盖已存在的文件（如果需要）
    '-f', 'rawvideo',
    '-pix_fmt', 'gray',  # 指定输入像素格式为灰度
    '-s', f'{720}x{1280}',  # 指定输入视频尺寸（需要根据你的实际视频尺寸调整）
    '-i', '-',  # 从标准输入读取数据
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-tune', 'zerolatency',
    '-crf', '28',  # 设置视频质量（可以调整）
    '-threads', '4',  # 尝试使用多线程
    '-f', 'rtsp',
    rtsp_output_url
]

# 打开视频流
cap = cv2.VideoCapture(rtsp_input_url)

if not cap.isOpened():
    print(f"无法打开视频流: {rtsp_input_url}")
    exit()

# 启动FFmpeg进程
process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧，退出")
        break

    # 将彩色帧转换为灰度帧
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 将灰度帧转换为字节数据并写入FFmpeg的stdin
    try:
        process.stdin.write(gray_frame.tobytes())
    except BrokenPipeError:
        print("FFmpeg进程已关闭")
        break

    # 显示处理后的帧（可选，仅用于本地查看）
    # cv2.imshow("Processed Frame", gray_frame)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# 清理资源
cap.release()
cv2.destroyAllWindows()
if process.poll() is None:
    process.stdin.close()
    process.wait()