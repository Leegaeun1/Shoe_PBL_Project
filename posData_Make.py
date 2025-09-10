import matplotlib.pyplot as plt
import pandas as pd

# 좌표 저장할 리스트
points = []
MAX_POINTS = 40  # 고정 개수

def onclick(event):
    if event.button == 1 and event.xdata is not None and event.ydata is not None:
        if len(points) < MAX_POINTS:
            x, y = event.xdata, event.ydata
            points.append((x, y))
            idx = len(points)

            print(f"{idx}번째 점: ({x:.2f}, {y:.2f})")
            # 빨간 점 + 번호 표시
            plt.plot(x, y, "ro")
            plt.text(x, y, str(idx), color="blue", fontsize=8)

            plt.draw()

            # 40개 다 찍으면 자동 저장 후 종료
            if len(points) == MAX_POINTS:
                save_and_close()

def save_and_close():
    output_csv = "clicked_points.csv"

    # 좌표를 x1,y1,x2,y2,... 형식으로 변환
    flat = []
    for (x, y) in points:
        flat.extend([x, y])

    # 컬럼명: x1,y1,x2,y2,...
    cols = [f"{c}{i}" for i in range(1, len(points) + 1) for c in ["x", "y"]]

    df = pd.DataFrame([flat], columns=cols)
    df.to_csv(output_csv, index=False)

    print(f"{MAX_POINTS}개 좌표가 {output_csv}에 저장되었습니다.")
    plt.close()  # 창 닫기

# figure 생성
fig, ax = plt.subplots()
ax.set_title(f"마우스로 좌표를 찍으세요 (총 {MAX_POINTS}개).")
ax.set_xlim(-50, 50)   # X축 범위
ax.set_ylim(0, 250)    # Y축 범위
ax.set_aspect("equal") # 비율 고정

cid = fig.canvas.mpl_connect("button_press_event", onclick)
plt.show()
