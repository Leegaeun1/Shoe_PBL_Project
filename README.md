<div align="center">

  # 👟 Shoe Shape Prediction AI
  **머신러닝 기반 신발 사이즈별 비선형 형상 예측 모델**
  <br/>

  <br/>

  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>

</div>

<br/>

## 📝 Introduction

> **단순히 신발 사이즈를 키운다고(Scale-up) 실제 신발의 모양이 될까요? 답은 '아니오'입니다.**

예를 들어, 신발 사이즈가 230mm에서 280mm로 커질 때, 발볼의 너비는 단순 비율로 커지지 않습니다. 본 프로젝트는 이러한 비선형적인 형상 변화를 머신러닝으로 학습하여, 단일 사이즈 이미지 하나만으로 다른 모든 사이즈의 정밀한 외곽선 좌표를 예측하는 모델을 개발했습니다.

* **기간:** 2025.09 ~ 2025.11
* **유형:** 팀 프로젝트 (2인)
    * **이가은:** AI 모델링 알고리즘 개발, 성과 분석 및 시각화
    * **김민준:** 이미지 왜곡 처리, Contour 좌표 추출 파이프라인 구축

<br/>

## 🛠 Tech Stack

| Category | Technologies |
| :---: | :--- |
| **Language** | ![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white) |
| **Vision** | ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white) `Contour Extraction` `Image Processing` |
| **ML Model** | ![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) `PCA` `Kernel Ridge` `SVR` `GPR` |
| **Data** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) |
| **Tools** | ![VS Code](https://img.shields.io/badge/VS_Code-007ACC?style=flat-square&logo=visualstudiocode&logoColor=white) ![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white) |

<br/>

## 🔄 Project Pipeline

<div align="center">
  <table>
    <tr>
      <th width="25%">Phase 1. 이미지 전처리</th>
      <th width="50%">Phase 2. 모델링 & 예측</th>
      <th width="25%">Phase 3. 실험</th>
    </tr>
    <tr>
      <td align="center" valign="top">
        📸 <b>HSV Masking<br/>
        ⬇️<br/>
        🟢 <b>외곽선 좌표 추출<br/>
        ⬇️<br/>
        📏 <b>픽셀 → mm 변환
      </td>
      <td align="center" valign="top">
        🔄 <b>데이터 시작점/위상 정렬<br/>
        ⬇️<br/>
        🔀<b>학습 선택<br/>
        <table width="100%">
            <tr>
                <td width="50%" align="center"><b>🅰️ </b>전체 형상 주성분 분석</td>
                <td width="50%" align="center"><b>🅱️ </b>접선/법선<br/>벡터 변위량 계산</td>
            </tr>
        </table>
        ⬇️<br/>
        📈 회귀 진행(Machine Learning + Linear Extrapolation)
      </td>
      <td align="center" valign="top">
        📐 <b>실제 치수</b><br/>
        Length/Width Error<br/>
        ⬇️<br/>
        🎯 <b>영역 일치도</b><br/>
        IoU (Intersection over Union)<br/>
        ⬇️<br/>
        🧬 <b>형상 유사도</b><br/>
        Hausdorff Dist., Chamfer Mean
      </td>
    </tr>
  </table>
</div>

<br/>

## 🧠 Core Logic & Algorithms

본 프로젝트의 핵심은 **`Fin_shape_prediction_lib.py`** 에 구현된 커스텀 알고리즘입니다.

### 1. Hybrid Prediction System
> 학습 데이터 범위 밖에서도 모델이 무너지지 않도록 설계했습니다.

* **Logic:** 학습 범위 내에서는 RBF 커널 기반의 머신러닝(KRR/SVR)을 사용하여 정밀도를 높이고, 범위를 벗어나면 부드럽게 선형 회귀로 전환합니다.

### 2. Cyclic Alignment
> 같은 신발이라도 사진마다 좌표의 시작점(Index 0)이 다를 수 있습니다.

* **Logic:** 기준 형상(Base Shape)에 맞춰 타겟 형상의 좌표 인덱스를 회전(Rolling)시켜가며 L2 Norm 오차가 최소가 되는 최적의 정렬 포인트를 자동으로 탐색합니다.

<br/>

## 📊 Experimental Results

단순 비율 확대 방식(`RATIO_CTRL`)과 머신러닝 모델(`PCA_KRR`, `PCA_SVR` 등)의 성능을 비교했습니다.

![결과 그래프 설명](SHOE_LJH/performance_metrics_plot_combine__2.png)

> **Result Analysis:**
> * **Control Points:** 제어점이 **40~50개**일 때 성능 대비 연산 효율이 가장 좋음 (그래프 수렴 구간).
> * **Non-linearity:** 단순 Ratio 방식보다 ML 모델이 월등히 높은 결과들을 기록하며, 신발 사이즈 변화의 비선형성을 입증함.<br/> Length 오류는 Ratio 방식이 좋으나, Width 오류는 다른 모델들이 성능이 좋음.

<br/>

## 🚀 Troubleshooting

<details>
<summary><b>🚨 Issue 1: 데이터 정렬 불일치 (Misalignment)</b> (Click to expand)</summary>
<br/>

* **문제 상황:** 이미지 촬영 각도나 조명에 따라 외곽선 추출 시 시작점(Index 0)이 발가락 끝이 아닌 옆면으로 잡히는 경우가 발생하여, 모델이 엉뚱한 형상을 학습함.
* **해결 방법:** 전처리 단계에서 `np.roll`을 활용한 **Cyclic Alignment** 알고리즘을 도입. 모든 데이터의 좌표를 회전시켜 기준 데이터와 가장 유사한 위상으로 자동 정렬한 후 학습을 진행하여 오차를 획기적으로 줄임.

</details>

<details>
<summary><b>🚨 Issue 2: 외삽 구간의 예측 발산</b> (Click to expand)</summary>
<br/>

* **문제 상황:** 학습 데이터(230~270mm)에 없는 280mm 사이즈를 예측할 때, 비선형 모델(RBF Kernel)의 특성상 예측값이 급격히 튀는 현상 발생.
* **해결 방법:** 경계값(Boundary)에서 멀어질수록 선형 모델의 가중치를 높이는 **Blending Strategy**를 적용. 이를 통해 미지의 영역에서도 물리적으로 타당한 예측값을 얻음.

</details>

<details>
<summary><b>🚨 Issue 3: 물리적 모순</b> (Click to expand)</summary>
<br/>

* **문제 상황:** 280mm 예측 결과가 275mm보다 길이가 짧게 나오는 역전 현상 발생.
* **해결 방법:** PCA 주축(Principal Axis)을 기준으로, 사이즈가 커질 때 길이도 반드시 단조 증가하도록 강제하는 후처리 로직(`enforce_size_caps_monotone`)을 추가함.

</details>

<br/>

## 📂 Directory Structure

```bash
├── 1_Counter_Code.py           # 📸 이미지 전처리 및 외곽선 추출
├── 2_outline_points.py         # 💾 추출된 외곽선을 CSV 데이터로 변환
├── 2_CounterToExcel.py         # 💾 컨트롤 포인트를 CSV 데이터로 변환
├── Fin_shape_prediction_lib.py # 🧠 [Core] 형상 예측 라이브러리 (Model Class)
├── 4_main_controller.py        # 🎮 모델 학습 및 예측 실행 스크립트
├── 5_Result_Visual_V8_All.py   # 📊 결과 시각화 및 성능 평가
└── Fin_graph.py                # 📈 최종 성과 그래프 생성

