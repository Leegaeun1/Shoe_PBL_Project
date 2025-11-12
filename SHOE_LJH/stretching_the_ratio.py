import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 정답 신발 사이즈
CORRECT_SHOE_SIZE = 250
# 타겟 신발 사이즈
TARGET_SHOE_SIZE = 270
# 비율
ratio = TARGET_SHOE_SIZE / CORRECT_SHOE_SIZE

Data_DIR = "output_outlines_1030\L"

TARGET_SHOE_IMG = os.path.join(Data_DIR,"irish_270L.jpg")

# 이미지 불러오기
img = mpimg.imread(TARGET_SHOE_IMG* ratio)

# 이미지 출력
plt.imshow(img)
plt.axis('off')  # 축 레이블 끄기
plt.show()