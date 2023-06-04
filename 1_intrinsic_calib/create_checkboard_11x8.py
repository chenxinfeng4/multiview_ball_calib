# Desc: 生成 11x8 的棋盘图像，每个方格大小为 20mm x 20mm，保存在 A4 纸上
import cv2
import numpy as np

# 定义棋盘大小
JOINT_SIZE = (8, 11)
CHESSBOARD_SIZE = (JOINT_SIZE[0] + 1, JOINT_SIZE[1] + 1)

# 定义方格大小
A4_SIZE_MM = (210, 297) #210mm x 297mm
# 设置 A4 纸大小（单位：像素）
A4_SIZE_PIXELS = (2480, 3508)

SQUARE_SIZE_MM = 20
SQUARE_SIZE = int(A4_SIZE_PIXELS[0] / (A4_SIZE_MM[0] / SQUARE_SIZE_MM))

# 计算棋盘的真实尺寸（单位：像素）
CHESSBOARD_SIZE_PIXELS = (CHESSBOARD_SIZE[0] * SQUARE_SIZE, CHESSBOARD_SIZE[1] * SQUARE_SIZE)

# 生成棋盘图像
chessboard = np.zeros((CHESSBOARD_SIZE_PIXELS[1], CHESSBOARD_SIZE_PIXELS[0], 3), dtype=np.uint8)
color = (255, 255, 255)

for y in range(CHESSBOARD_SIZE[1]):
    for x in range(CHESSBOARD_SIZE[0]):
        if (x + y) % 2 == 0:
            cv2.rectangle(chessboard, (x * SQUARE_SIZE, y * SQUARE_SIZE), ((x + 1) * SQUARE_SIZE, (y + 1) * SQUARE_SIZE), color, -1)



# 计算棋盘在 A4 纸上的位置和大小
x_offset = int((A4_SIZE_PIXELS[0] - CHESSBOARD_SIZE_PIXELS[0]) / 2)
y_offset = int((A4_SIZE_PIXELS[1] - CHESSBOARD_SIZE_PIXELS[1]) / 2)
x_end = x_offset + CHESSBOARD_SIZE_PIXELS[0]
y_end = y_offset + CHESSBOARD_SIZE_PIXELS[1]
chessboard_on_A4 = np.zeros((A4_SIZE_PIXELS[1], A4_SIZE_PIXELS[0], 3), dtype=np.uint8) + 255
chessboard_on_A4[y_offset:y_end, x_offset:x_end] = chessboard

# 保存棋盘在 A4 上的图像
cv2.imwrite("chessboard_11x8_20mm_on_A4.jpg", chessboard_on_A4)
