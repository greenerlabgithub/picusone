import os
if not hasattr(os, "add_dll_directory"):
    os.add_dll_directory = lambda x: None

import logging
import azure.functions as func

import sys
import json
import base64
import cv2
import numpy as np


# ---------------------------
# 전역: 색상 범위 (HSV)
# ---------------------------
color_ranges = {
    "검정":      ((0,   0,   0),   (10, 255,  50)),
    "갈색":      ((10,  80,  5),    (30, 255, 255)),
    "초록":      ((60,  50,  50),   (85, 255, 255)),
    "보라":      ((130, 50,  50),   (170, 255, 255)),
    "파랑":      ((90,  50,  50),   (114, 255, 255)),
    "나무둘레":  ((115, 50, 50),    (129, 255, 255)),
    "나무표시":  ((32,  50, 50),    (59, 255, 255)),
    "나무표시2": ((3,  240,150),    (10, 255, 255))
}

# ---------------------------
# 등급 판정 함수 (기존 그대로)
# ---------------------------
def calc_grade(ratio: float) -> str:
    """
    초록+보라+파랑 비율(ratio)에 따라 등급 산정
    """
    if 0 <= ratio < 1:
        return "A"
    elif 1 <= ratio <= 19:
        return "B"
    elif 20 <= ratio <= 39:
        return "C"
    elif 40 <= ratio <= 49:
        return "D"
    else:
        return "E"

# ---------------------------
# 단일 이미지 분석 (기존 그대로)
# ---------------------------
def analyze_one_image(tree_id: str, image_path: str):
    """
    단일 이미지를 분석하여 색상 픽셀수/등급 정보를 반환.
    실패 시 None.
    """
    if not os.path.exists(image_path):
        logging.info(f"[오류] 파일 없음: {image_path}")
        return None

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        logging.info(f"[오류] OpenCV로 읽지 못함: {image_path}")
        return None

    # BGR → HSV 변환
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 나무둘레 + 표시 + 표시2 합치기
    outer_mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
    for key in ["나무둘레", "나무표시", "나무표시2"]:
        lo, up = color_ranges[key]
        tmp = cv2.inRange(img_hsv, lo, up)
        outer_mask = cv2.bitwise_or(outer_mask, tmp)

    # 모폴로지
    kernel = np.ones((3,3), np.uint8)
    outer_mask = cv2.dilate(outer_mask, kernel, iterations=1)
    outer_mask = cv2.erode(outer_mask, kernel, iterations=1)

    # 컨투어 찾기
    contours, _ = cv2.findContours(outer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logging.info(f"[결과] 외곽 컨투어 없음: {image_path}")
        return None

    largest = max(contours, key=cv2.contourArea)
    roi_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(roi_mask, [largest], -1, 255, -1)

    # 5색 분석
    target_colors = ["검정", "갈색", "초록", "보라", "파랑"]
    color_counts = {}
    for c in target_colors:
        lo, up = color_ranges[c]
        mask_c = cv2.inRange(img_hsv, lo, up)
        final_mask = cv2.bitwise_and(mask_c, mask_c, mask=roi_mask)
        cnt = cv2.countNonZero(final_mask)
        color_counts[c] = cnt

    sum_of_5 = sum(color_counts.values())
    if sum_of_5 == 0:
        logging.info(f"[결과] 5색 픽셀 없음: {image_path}")
        return None

    # 검정+갈색
    black_brown = color_counts["검정"] + color_counts["갈색"]
    black_brown_ratio = round((black_brown / sum_of_5) * 100, 2)

    # 초록+보라+파랑
    gpb = color_counts["초록"] + color_counts["보라"] + color_counts["파랑"]
    gpb_ratio = round((gpb / sum_of_5) * 100, 2)

    overall_grade = calc_grade(gpb_ratio)

    return {
        "tree_id": tree_id,
        "image_path": image_path,
        "color_counts": color_counts,
        "sum_of_5": sum_of_5,
        "black_brown_count": black_brown,
        "black_brown_ratio": black_brown_ratio,
        "green_purple_blue_count": gpb,
        "green_purple_blue_ratio": gpb_ratio,
        "overall_grade": overall_grade
    }

# ---------------------------
# 여러 이미지 분석 후 JSON 반환 함수 (새로 추가)
# ---------------------------
def analyze_multiple_images_json(image_list):
    """
    여러 이미지를 한 번에 분석하여 결과 리스트를 반환합니다.
    image_list: [(수목번호, 이미지경로), (...), ...]
    """
    results = []
    for tree_id, img_path in image_list:
        result = analyze_one_image(tree_id, img_path)
        if result is not None:
            results.append(result)
    logging.info(f"[결과] 전체 {len(results)}개 이미지 분석 완료")
    return results

# ---------------------------
# JSON 디코딩 및 분석 (수정)
# ---------------------------
def decode_and_run(json_str):
    data = json.loads(json_str)
    image_list = []

    # 최대 15장 예시
    for i in range(1, 16):
        num_key = f"img{i}Num"
        img_key = f"img{i}"
        if num_key in data and img_key in data:
            tree_id = data[num_key]
            b64_str = data[img_key]
            if not b64_str:
                continue
            # 쓰기 권한이 있는 /tmp 디렉터리를 사용
            local_path = os.path.join("/tmp", f"temp_img{i}.jpg")
            with open(local_path, "wb") as f:
                f.write(base64.b64decode(b64_str))
            image_list.append((tree_id, local_path))

    if not image_list:
        logging.info("[결과] 디코딩된 이미지가 하나도 없습니다.")
        return None

    # 엑셀 파일 대신 JSON 결과 반환
    return analyze_multiple_images_json(image_list)

# ---------------------------
# 메인 함수 (HTTP Trigger) - JSON 출력 방식으로 수정
# ---------------------------
def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function의 HTTP Trigger 엔트리 포인트
    """
    logging.info("Python HTTP trigger function processed a request (JSON output).")

    try:
        body_str = req.get_body().decode('utf-8')
    except Exception as e:
        logging.error(f"Error reading request body: {e}")
        return func.HttpResponse("Invalid request body", status_code=400)

    results = decode_and_run(body_str)
    if results is None or len(results) == 0:
        return func.HttpResponse(json.dumps({"result": "no analysis data"}), status_code=200, headers={"Content-Type": "application/json"})
    
    # JSON 결과 반환 (전체 픽셀 대비 초록+보라+파랑 비율과 등급이 포함됨)
    return func.HttpResponse(json.dumps({"result": results}, ensure_ascii=False), status_code=200, headers={"Content-Type": "application/json"})
