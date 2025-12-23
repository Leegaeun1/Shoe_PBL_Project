<div>

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

## 1. 프로젝트 개요 (Overview)

이 프로젝트에서 저희는 신발 제조 공정의 효율화를 위해, **단일 사이즈 이미지 하나만으로 다른 모든 사이즈의 정밀한 외곽선을 예측하는 머신러닝 시스템**을 설계하고 구현했습니다.

기존 신발 제조 현장에서는 사이즈를 확장(Grading)할 때 단순 비율 확대 방식을 주로 사용합니다. 하지만 저희는 **"발의 크기가 커진다고 해서 발볼의 너비가 정비례하게 커지지는 않는다"**는 인체공학적 비선형성에 주목했습니다. 단순 스케일업(Scale-up) 방식은 사이즈가 커질수록 신발의 모양이 기형적으로 변하는 문제를 야기합니다.

이를 해결하기 위해 저희는 머신러닝 기반의 비선형 회귀 모델을 도입했습니다. 사이즈 변화에 따른 미세한 형상 변화 패턴을 학습하여, 230mm 신발 사진 한 장만 있어도 280mm 신발의 정확한 라스트(Last, 신발 틀) 형상을 예측할 수 있는 기술을 개발했습니다.

* 개발 기간 : 2025.09~2025.11
* 참여 인원 : 2명
  * 김민준 : 데이터 전처리
  * 이가은 : 예측 및 시각화화

<br/>

## 2. 주요 기능 (Key Features)

저희가 구현한 시스템의 핵심 기능은 다음과 같습니다.

* **자동 외곽선 추출 (Auto Contour Extraction):** OpenCV의 HSV Masking 기법을 활용하여 다양한 배경의 신발 이미지에서 노이즈를 제거하고 정밀한 외곽선 좌표를 자동으로 추출합니다.
* **순환 정렬 알고리즘 (Cyclic Data Alignment):** 이미지마다 제각각인 좌표 시작점(Index 0) 문제를 해결하기 위해, 기준 형상에 맞춰 데이터의 위상을 자동으로 회전(Rolling)시켜 정렬하는 전처리 알고리즘을 구현했습니다.
* **하이브리드 예측 시스템 (Hybrid Prediction):** 학습 데이터 구간 내에서는 RBF 커널 기반의 비선형 모델(KRR/SVR)로 정밀도를 높이고, 외삽(Extrapolation) 구간에서는 선형 모델을 결합하여 예측 안정성을 확보했습니다.
* **물리적 모순 방지 (Monotonic Constraint):** 예측된 큰 사이즈의 신발이 작은 사이즈보다 길이가 짧아지는 역전 현상을 방지하기 위해, PCA 주축을 기준으로 단조 증가성을 강제하는 후처리 로직을 적용했습니다.

<br/>

## 3. 시스템 아키텍처 (System Pipeline)

저희는 데이터의 전처리부터 모델링, 평가까지 이어지는 파이프라인을 구축했습니다. 전체 데이터 처리 흐름은 다음과 같습니다.

<div>
  <table>
    <tr>
      <th width="25%">Phase 1. Preprocessing</th>
      <th width="50%">Phase 2. Modeling & Prediction</th>
      <th width="25%">Phase 3. Evaluation</th>
    </tr>
    <tr>
      <td align="center" valign="top">
        <br/>
        📸 <b>Image Processing</b><br/>
        (HSV Masking & Contour)<br/>
        ⬇️<br/>
        🟢 <b>Normalization</b><br/>
        (Pixel to mm conversion)<br/>
        ⬇️<br/>
        🔄 <b>Cyclic Alignment</b><br/>
        (Data Phase Matching)
      </td>
      <td align="center" valign="top">
        <br/>
        🔀 <b>Feature Engineering</b><br/>
        (PCA & Vector Calculation)<br/>
        ⬇️<br/>
        🧠 <b>Hybrid Learning</b><br/>
        (Non-linear Kernel Ridge + Linear Regression)<br/>
        ⬇️<br/>
        📈 <b>Shape Reconstruction</b><br/>
        (Inverse PCA Transform)
      </td>
      <td align="center" valign="top">
        <br/>
        📐 <b>Physical Metrics</b><br/>
        (Length/Width Error Check)<br/>
        ⬇️<br/>
        🎯 <b>Geometric Metrics</b><br/>
        (IoU, Hausdorff Distance)<br/>
        ⬇️<br/>
        📊 <b>Visualization</b><br/>
        (Error Heatmap)
      </td>
    </tr>
  </table>
</div>

1.  **전처리 (Preprocessing):** 입력된 신발 이미지에서 외곽선을 추출한 뒤, `Cyclic Alignment`를 통해 모든 데이터의 좌표 시작점을 통일합니다.
2.  **모델링 (Modeling):** 정렬된 좌표 데이터를 PCA(주성분 분석)로 차원 축소한 뒤, 사이즈 변화에 따른 주성분의 변화량을 머신러닝 모델(KRR, SVR)로 학습합니다.
3.  **예측 및 후처리 (Prediction):** 입력된 타겟 사이즈에 맞는 형상을 예측하고, 물리적 모순(길이 역전 등)이 발생하지 않도록 보정하여 최종 좌표를 출력합니다.

<br/>

## 4. 실험 결과 (Experimental Results)

저희는 개발한 모델의 성능을 검증하기 위해, 단순 비율 확대 방식(`Ratio Control`)과 머신러닝 모델(`PCA_KRR`, `PCA_SVR`)의 예측 정확도를 비교했습니다.

![결과 그래프](SHOE_LJH/performance_metrics_plot_combine__2.png)

**실험 분석 (Result Analysis)**
* **비선형성 입증:** 단순 비율 확대 방식(Ratio)은 길이가 커질수록 발볼이 과도하게 넓어지는 오차가 발생했습니다. 반면, 저희가 개발한 **ML 모델은 발볼(Width) 오차를 획기적으로 줄여** 실제 신발의 비선형적인 형상 변화를 잘 반영함을 확인했습니다.
* **최적 제어점:** 외곽선을 구성하는 제어점(Control Points)이 **40~50개**일 때 연산 효율 대비 예측 성능이 가장 우수함을 확인하여 시스템을 최적화했습니다.

<br/>

## 5. 한계점 및 향후 과제 (Limitations & Future Work)

**한계점**
**한계점 (Limitations)**
* **외삽(Extrapolation) 구간의 불안정성:** 학습 데이터(230~270mm) 범위를 벗어나는 280mm 이상의 사이즈를 예측할 때, 비선형 모델(RBF Kernel)의 특성상 예측값이 급격히 발산하는 현상이 발생했습니다.
* **데이터 정렬의 민감도:** `Cyclic Alignment` 알고리즘을 도입하여 위상차를 보정했으나, 촬영 각도가 심하게 틀어진(Skewed) 이미지의 경우 여전히 초기 정렬 오차가 모델 성능에 영향을 미쳤습니다.
* **전처리의 배경 의존성:** 현재의 OpenCV 기반 외곽선 추출 방식은 배경이 복잡하거나 그림자가 짙은 환경에서는 노이즈가 섞이거나 추출 품질이 저하되는 한계가 있습니다.
* **학습 데이터의 부족:** 확보된 신발 데이터의 수와 스타일이 한정적이어서, 다양한 디자인과 형태를 가진 신발에 대해 범용적인 예측 성능을 확보하는 데 어려움이 있었습니다.

**향후 과제 (Future Work)**
* **데이터 증강:** 회전(Rotation), 비틀림(Shearing), 시점 변환(Perspective Transform) 등의 Data Augmentation 기법을 적용하여, 다양한 촬영 환경에서도 모델이 잘 동작하도록 일반화 성능을 높일 계획입니다.
* **3D 형상 재구성으로의 확장:** 현재의 2D 평면 외곽선 예측을 넘어, 멀티뷰(Multi-view) 이미지나 심도(Depth) 정보를 활용하여 신발의 발등 높이와 전체 부피(Volume)까지 예측하는 3D 모델링 프로젝트로 확장하고자 합니다.
