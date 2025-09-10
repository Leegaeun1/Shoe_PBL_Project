#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Insole Data Viewer
- CSV에서 X1,Y1,...,Xn,Yn 포맷의 좌표를 읽어 각 행(샘플)을 시각화
- 포인트 순번(1..N) 라벨 표기
- ←/→ 로 샘플 이동
- 마우스 클릭으로 개별 포인트 삭제/복원
- 'd' 로 행 전체 삭제/복원, 'r' 로 현재 행 포인트 삭제 초기화
- 's' 로 결과 저장(삭제 행 제거, 삭제 포인트는 NaN으로) + 로그 저장
- 'h' 도움말

Run:
    python insole_viewer.py --csv /path/to/your.csv
"""

import argparse
import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 불필요한 Unnamed 컬럼 제거
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, na=False)].copy()
    if df.empty:
        raise ValueError("CSV가 비어있습니다.")
    return df

def detect_xy_pairs(df: pd.DataFrame):
    """대소문자 무시하고 xk/yk 쌍을 탐지하여 (Xcols, Ycols, pair_nums) 반환"""
    colmap = {c.lower().strip(): c for c in df.columns}  # lower -> original
    pair_nums = []
    for c in df.columns:
        m = re.fullmatch(r"[xy](\d+)", c.strip(), flags=re.IGNORECASE)
        if m and c.lower().startswith("x"):
            k = int(m.group(1))
            if f"y{k}" in colmap:
                pair_nums.append(k)
    pair_nums = sorted(set(pair_nums))
    if not pair_nums:
        raise ValueError("Xk/Yk 포맷 컬럼을 찾지 못했습니다. (예: X1,Y1,... 또는 x1,y1,...)")
    Xcols = [colmap[f"x{k}"] for k in pair_nums]
    Ycols = [colmap[f"y{k}"] for k in pair_nums]
    return Xcols, Ycols, pair_nums

class InsoleViewer:
    def __init__(self, df: pd.DataFrame, Xcols, Ycols):
        self.df = df
        self.Xcols = Xcols
        self.Ycols = Ycols

        # 삭제 상태
        self.deleted_rows = set()     # 완전 삭제된 행 인덱스
        self.deleted_points = {}      # row_idx -> set({1-based point indices})

        # 뷰 상태
        self.current_row = 0

        # 플롯 준비
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.cid_clk = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.plot_row(self.current_row)
        plt.show()

    def get_row_points(self, i):
        xs = self.df.loc[i, self.Xcols].values.astype(float)
        ys = self.df.loc[i, self.Ycols].values.astype(float)
        return xs, ys

    def title_text(self, i):
        bits = [f"Row {i+1}/{len(self.df)}"]
        if i in self.deleted_rows:
            bits.append("[ROW DELETED]")
        dp = self.deleted_points.get(i, set())
        if dp:
            bits.append(f"[{len(dp)} PT{'S' if len(dp)>1 else ''} DELETED]")
        return " ".join(bits)

    def plot_row(self, i):
        self.ax.clear()
        self.ax.set_title(self.title_text(i))

        xs, ys = self.get_row_points(i)
        dp = self.deleted_points.get(i, set())
        mask = np.ones_like(xs, dtype=bool)
        if dp:
            idx0 = [k-1 for k in dp if 1 <= k <= len(xs)]
            mask[idx0] = False

        # 선(연결) 그리기
        if np.sum(mask) >= 2:
            pts = np.column_stack((xs[mask], ys[mask]))
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            self.ax.add_collection(LineCollection(segs))

        # 포인트 찍기
        self.ax.scatter(xs[mask], ys[mask])

        # 순번 라벨(삭제 포인트도 숫자는 표기, *로 표시)
        for idx, (x, y) in enumerate(zip(xs, ys), start=1):
            label = str(idx) + ("*" if idx in dp else "")
            self.ax.text(x, y, label, fontsize=8, ha="center", va="center")

        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.autoscale_view()
        self.ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        self.ax.figure.canvas.draw_idle()

    def nearest_point(self, i, x, y):
        xs, ys = self.get_row_points(i)
        d2 = (xs - x) ** 2 + (ys - y) ** 2
        k0 = int(np.argmin(d2))  # 0-based
        return k0 + 1            # 1-based

    def on_key(self, event):
        k = event.key
        if k == "right":
            if self.current_row < len(self.df) - 1:
                self.current_row += 1
                self.plot_row(self.current_row)
        elif k == "left":
            if self.current_row > 0:
                self.current_row -= 1
                self.plot_row(self.current_row)
        elif k == "d":
            # 행 전체 삭제/복원
            if self.current_row in self.deleted_rows:
                self.deleted_rows.remove(self.current_row)
            else:
                self.deleted_rows.add(self.current_row)
            self.plot_row(self.current_row)
        elif k == "r":
            # 현재 행 포인트 삭제 초기화
            self.deleted_points.pop(self.current_row, None)
            self.plot_row(self.current_row)
        elif k == "s":
            # 저장
            self.save_results()
        elif k == "h":
            print(
"""단축키:
  ← / → : 이전 / 다음 샘플
  d      : 현재 행 삭제/복원
  r      : 현재 행 포인트 삭제 초기화
  s      : 결과 저장 (CSV + 로그)
  h      : 도움말 출력
마우스:
  좌클릭 : 가장 가까운 포인트 삭제/복원
표기:
  숫자: 포인트 순서 (삭제된 포인트는 * 표시)
"""
            )

    def on_click(self, event):
        if event.inaxes != self.ax or event.button != 1 or event.xdata is None:
            return
        k1 = self.nearest_point(self.current_row, event.xdata, event.ydata)
        st = self.deleted_points.setdefault(self.current_row, set())
        if k1 in st:
            st.remove(k1)
        else:
            st.add(k1)
        self.plot_row(self.current_row)

    def save_results(self):
        out_df = self.df.copy()

        # 완전 삭제된 행 제거
        keep_idx = [i for i in range(len(out_df)) if i not in self.deleted_rows]
        out_df = out_df.iloc[keep_idx].copy()

        # 포인트 삭제는 NaN으로 마킹 + 컬럼에 기록
        deleted_points_col = []
        for j, src_idx in enumerate(keep_idx):
            dp = sorted(list(self.deleted_points.get(src_idx, set())))
            deleted_points_col.append(",".join(map(str, dp)) if dp else "")
            for k in dp:
                if 1 <= k <= len(self.Xcols):
                    out_df.at[out_df.index[j], self.Xcols[k-1]] = np.nan
                    out_df.at[out_df.index[j], self.Ycols[k-1]] = np.nan
        out_df["deleted_points"] = deleted_points_col

        out_csv = os.path.abspath("insole_viewer_output.csv")
        out_log = os.path.abspath("insole_viewer_log.txt")
        out_df.to_csv(out_csv, index=False)

        with open(out_log, "w", encoding="utf-8") as f:
            f.write("Deleted rows (1-based): " + ",".join(str(i+1) for i in sorted(self.deleted_rows)))
            f.write("\nDeleted points per original row (1-based):\n")
            for i in range(len(self.df)):
                dp = self.deleted_points.get(i, set())
                if dp:
                    f.write(f"  Row {i+1}: {sorted(dp)}\n")

        print(f"Saved CSV: {out_csv}")
        print(f"Saved LOG: {out_log}")

def main():
    # parser = argparse.ArgumentParser(description="Interactive Insole Viewer")
    # parser.add_argument("--csv", required=True, help="X1,Y1,... 스타일 좌표가 있는 CSV 경로")
    # args = parser.parse_args()

    df = load_dataframe('insole_viewer_output.csv')
    Xcols, Ycols, _ = detect_xy_pairs(df)
    InsoleViewer(df, Xcols, Ycols)

if __name__ == "__main__":
    main()
