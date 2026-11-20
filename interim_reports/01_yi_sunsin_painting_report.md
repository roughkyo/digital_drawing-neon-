# 충무공 이순신 영정 브라우저 페인팅 앱 구현 및 스트로크 최적화 보고서

## 1. 과제 개요 및 제약 조건
- **대상 이미지**: `../reference.png` (`1280 × 1932`, 비율 `0.662526`) — 충무공 이순신 표준영정
- **앱 요구사항**: 순수 HTML/CSS/JS(프레임워크·외부 에셋·이미지 생성 모델 미사용), 상단 바(`Paint` + 실시간 스트로크 카운터), 툴바(브러시 크기·투명도·현재 색상 스와치·브러시 모드), 좌측 동일 비율의 빈 캔버스, 우측 `Reference` 패널(`reference.png` + 4단계 진행 라벨 `Blocking in` / `Form and color` / `Refinement` / `Fine details`).
- **렌더링 제약**: `drawImage`, `putImageData`, CSS 이미지/필터/마스크 일절 금지. 오직 개별 Soft Round 및 Bristle 브러시 스트로크만으로 단계별 채색.

## 2. 방법론 및 설계 근거
1. **Python(Numba) ↔ Browser Canvas 2D 1:1 브러시 정합성**
   - 4종의 순수 Canvas 2D 벡터/그라디언트 브러시 프리미티브(`createRadialGradient` Soft Round 워시, 5가닥 필라멘트 Bristle 브러시, 2-pass Fine Soft Round, 세필 Hairline Round)를 Numba JIT 시뮬레이터와 `app.js`에 동일 수식으로 구현.
   - 좌표(0.5px), 반경(0.1px), 투명도(0.01), RGB(8-bit) 및 브리스틀 시드를 사전 양자화하여 시뮬레이션과 실제 브라우저 렌더링 간 드리프트를 제거.
2. **형태 추종 곡선 스트로크 (No Grid)**
   - 다중 스케일 구조 텐서(Structure Tensor) 접선장(`σ = 12, 6.5, 3.0, 1.4`)을 따라 색상 경계 감지 유선(Streamline)을 추적해 2차 베지어 곡선 스트로크 생성.
   - 수염 영역에는 하향 세필 바이어스, 좌·우 표피(표범 가죽) 영역에는 방사형 털 결 바이어스 적용.
3. **닫힌해 최적 안료색 및 피크 오차 샘플링**
   - 각 후보 스트로크의 유효 마스크 $M_{\text{eff}}$에 대해 가중 L2 오차를 최소화하는 닫힌해 브러시 색상 $S^*$를 계산(자연 안료 범위 내 클램핑)하고, 실제 오차 감소량이 기준 이상인 스트로크만 채택.
   - V2부터 4×4 셀 내 최대 오차 픽셀을 추적하는 피크 풀링 샘플링을 도입해 1px 수염·눈꺼풀선·의습선·쌍호흉배 링 반점 누락을 방지.

## 3. 반복 개선 결과 (V1 → V2 → V3)

| 지표 | V1 (26,573획) | V2 (50,358획) | V3 (75,822획) | V1→V3 개선폭 |
|---|---:|---:|---:|---:|
| Global PSNR (dB) | 26.90 | 28.19 | **29.23** | +2.33 dB |
| Global SSIM | 0.7057 | 0.7355 | **0.7551** | +0.0494 |
| Global MAE (0–255) | 6.82 | 6.29 | **5.79** | -1.03 |
| Eyes PSNR / MAE | 32.40 / 4.00 | 34.12 / 3.50 | **34.58 / 3.33** | +2.18 dB |
| Face PSNR / MAE | 23.74 / 9.01 | 26.34 / 7.32 | **28.74 / 6.02** | +5.00 dB |
| Beard PSNR / MAE | 19.88 / 16.08 | 22.89 / 11.91 | **25.51 / 9.25** | +5.63 dB |
| Badge (쌍호흉배) PSNR | 22.17 | 23.97 | **25.31** | +3.14 dB |
| Belt (각대) PSNR | 23.91 | 25.37 | **26.60** | +2.69 dB |
| Hat (사모) PSNR | 26.06 | 28.02 | **28.69** | +2.63 dB |
| Robe (홍단령) PSNR | 25.82 | 27.01 | **28.13** | +2.31 dB |
| Pelt (표피) PSNR | 26.96 | 27.81 | **28.58** | +1.62 dB |
| Background PSNR | 34.24 | 35.58 | **35.67** | +1.43 dB |

### V3 단계별 스트로크 구성 (총 75,822획)
1. **Blocking in**: `871`획 (2단 광폭 워시 + 주요 명암 덩어리)
2. **Form and color**: `2,891`획 (배경 한지/비단 톤 정돈 + 브리스틀 입체 조형 + 얼굴 명암)
3. **Refinement**: `10,318`획 (배경 최종 바림 + 형태 정교화 + 홍단령 운보문 기초)
4. **Fine details**: `61,742`획 (의습선 윤곽, 사모 양각 나뭇결, 표피 깃털·반점, 쌍호흉배 링 반점·파도·각대 격자, 2단 세필 수염, 이목구비 및 눈매 마감)

## 4. 실제 브라우저 검증 결과
- 로컬 정적 서버에서 Chromium 브라우저로 `index.html` 실행 검증 완료:
  - **콘솔/페이지 에러**: `0건` (모든 5개 로컬 에셋 HTTP 200)
  - **실시간 재생**: 빈 캔버스에서 시작해 4단계(`Blocking in` → `Form and color` → `Refinement` → `Fine details`)를 거쳐 약 6초 만에 `75,822 strokes` 완료, `Restart` / `Finish` / 수동 페인팅 정상 동작.
  - **캔버스 비율**: `#paintCanvas` 렌더링 크기 `514.30 × 776.27 px` (비율 `0.662530`), `#referenceImage` `209.87 × 316.77 px` (비율 `0.662528`), 원본 `1280/1932 = 0.662526`과 서브픽셀 수준 일치.
  - **금지 API 감사**: `index.html`, `styles.css`, `app.js` 내 `drawImage`, `putImageData`, `getImageData`, 외부 URL, CSS `filter`/`mask` 사용 없음.

## 5. 관련 산출물 및 시각화
- 앱 파일: `../index.html`, `../styles.css`, `../app.js`, `../painting-data.js`
- 파이프라인 스크립트: `../scripts/01_analyze_reference.py`, `../scripts/02_generate_strokes.py`
- 정량 메트릭: `../intermediate_results/01_reference_analysis.json`, `../intermediate_results/metrics_v1.json`, `../intermediate_results/metrics_v2.json`, `../intermediate_results/metrics_v3.json`
- 시각화:
  - ![V3 4단계 진행](../visualizations/stages_v3.png)
  - ![V3 원본 대비 비교 및 잔차 맵](../visualizations/compare_v3.png)
  - ![V3 6개 핵심 영역 확대 비교](../visualizations/zooms_v3.png)
  - ![실제 브라우저 V3 완료 화면](../visualizations/browser_final_v3.png)
