import os
import argparse
import cv2
import numpy as np


# =========================
# 기본 유틸
# =========================
def normalize_to_uint8(channel: np.ndarray) -> np.ndarray:
    ch = channel.astype(np.float32)
    ch_min, ch_max = ch.min(), ch.max()
    if ch_max <= ch_min:
        return np.zeros_like(ch, dtype=np.uint8)
    ch = (ch - ch_min) / (ch_max - ch_min)
    ch = (ch * 255.0).clip(0, 255).astype(np.uint8)
    return ch


def make_colorized(gray: np.ndarray, color: str) -> np.ndarray:
    z = np.zeros_like(gray, dtype=np.uint8)
    if color == "red":
        return cv2.merge([z, z, gray])      # BGR
    elif color == "green":
        return cv2.merge([z, gray, z])
    elif color == "blue":
        return cv2.merge([gray, z, z])
    else:
        raise ValueError("color must be one of: red, green, blue")


def put_label(img, text, anchor_xy, text_xy, color=(255, 255, 255), thickness=2):
    """
    anchor_xy: annotation이 가리키는 실제 위치
    text_xy: 텍스트를 적을 위치
    """
    cv2.arrowedLine(img, text_xy, anchor_xy, color, 2, tipLength=0.03)
    cv2.putText(
        img, text, text_xy,
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness, cv2.LINE_AA
    )


# =========================
# B 채널: nucleus 검출
# =========================
def detect_nuclei(blue_u8: np.ndarray):
    blur = cv2.GaussianBlur(blue_u8, (0, 0), 1.5)

    _, otsu_mask = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    nuclei = []
    filtered_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 80:
            continue

        comp = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        nuclei.append({
            "contour": cnt,
            "center": (cx, cy),
            "bbox": (x, y, w, h),
            "area": area
        })
        filtered_mask[labels == i] = 255

    return filtered_mask, nuclei


def draw_blue_annotation(blue_u8: np.ndarray, nuclei):
    out = make_colorized(blue_u8, "blue")

    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    # nucleus contour
    for i, n in enumerate(nuclei, start=1):
        cv2.drawContours(out, [n["contour"]], -1, (255, 255, 255), 2)
        cx, cy = n["center"]
        cv2.circle(out, (cx, cy), 3, (255, 255, 255), -1)

    # 대표 nucleus 하나만 설명용 annotation
    if nuclei:
        target = max(nuclei, key=lambda x: x["area"])
        cx, cy = target["center"]
        text_xy = (max(10, cx - 120), max(25, cy - 40))
        put_label(out, "Nucleus", (cx, cy), text_xy, color=(255, 255, 255))

    return out


# =========================
# G 채널: FA point 검출
# =========================
def detect_fa_points(green_u8: np.ndarray,
                     min_distance_px: int = 8,
                     threshold_percentile: float = 99.2,
                     min_intensity: int = 20):
    """
    Green 채널에서 local maxima 기반 FA point 검출
    """
    blur_small = cv2.GaussianBlur(green_u8, (0, 0), 1.2)
    blur_bg = cv2.GaussianBlur(green_u8, (0, 0), 8.0)

    response = blur_small.astype(np.float32) - blur_bg.astype(np.float32)
    response = np.clip(response, 0, None)

    if np.count_nonzero(response) == 0:
        return [], response

    thr = np.percentile(response[response > 0], threshold_percentile) if np.any(response > 0) else 0
    thr = max(thr, float(min_intensity))

    # local maximum
    kernel_size = 2 * min_distance_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(response, kernel)

    local_max_mask = (response == dilated) & (response >= thr)
    ys, xs = np.where(local_max_mask)

    candidates = [(int(x), int(y), float(response[y, x])) for x, y in zip(xs, ys)]
    candidates = sorted(candidates, key=lambda t: t[2], reverse=True)

    selected = []
    for x, y, score in candidates:
        keep = True
        for px, py, _ in selected:
            if (x - px) ** 2 + (y - py) ** 2 < min_distance_px ** 2:
                keep = False
                break
        if keep:
            selected.append((x, y, score))

    return selected, response


def draw_green_annotation(green_u8: np.ndarray, points):
    out = make_colorized(green_u8, "green")

    # 모든 FA point 표시
    for x, y, _ in points:
        cv2.circle(out, (x, y), 3, (0, 255, 255), -1)   # yellow
        cv2.circle(out, (x, y), 6, (0, 255, 255), 1)

    # 대표 point 하나 설명
    if points:
        # 중앙에 가장 가까운 point를 대표로 선택
        h, w = green_u8.shape
        cx0, cy0 = w // 2, h // 2
        target = min(points, key=lambda p: (p[0] - cx0) ** 2 + (p[1] - cy0) ** 2)
        x, y, _ = target
        text_xy = (max(10, x - 120), max(25, y - 20))
        put_label(out, "FA point", (x, y), text_xy, color=(255, 255, 255))

    return out


# =========================
# R 채널: cell area 추정
# =========================
def detect_cell_area(red_u8: np.ndarray, green_u8: np.ndarray = None):
    """
    Red 채널 중심으로 cell area 추정.
    필요 시 green을 약하게 보조로 사용 가능.
    """
    red_f = red_u8.astype(np.float32)

    if green_u8 is not None:
        green_f = green_u8.astype(np.float32)
        combined = np.maximum(red_f, 0.6 * green_f)
    else:
        combined = red_f

    combined = normalize_to_uint8(combined)

    blur = cv2.GaussianBlur(combined, (0, 0), 2.0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    cell_contours = []
    filtered_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 1000:
            continue

        comp = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        cell_contours.append(cnt)
        filtered_mask[labels == i] = 255

    return filtered_mask, cell_contours


def draw_red_annotation(red_u8: np.ndarray, cell_contours):
    out = make_colorized(red_u8, "red")

    for cnt in cell_contours:
        cv2.drawContours(out, [cnt], -1, (0, 255, 255), 2)  # yellow contour

    if cell_contours:
        target = max(cell_contours, key=cv2.contourArea)
        M = cv2.moments(target)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            text_xy = (max(10, cx - 130), max(25, cy - 40))
            put_label(out, "Cell area", (cx, cy), text_xy, color=(255, 255, 255))

    return out


# =========================
# 메인
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Export annotated R/G/B channel images from merged microscopy image."
    )
    parser.add_argument("--input", required=True, help="Path to merged image")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--min_distance_px", type=int, default=8, help="Min distance between FA points")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.input}")

    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(f"Input must be a 3-channel merged image. Current shape: {img.shape}")

    # OpenCV = BGR
    b_raw = img[:, :, 0]
    g_raw = img[:, :, 1]
    r_raw = img[:, :, 2]

    b = normalize_to_uint8(b_raw)
    g = normalize_to_uint8(g_raw)
    r = normalize_to_uint8(r_raw)

    # ---- B: nucleus
    nucleus_mask, nuclei = detect_nuclei(b)
    blue_annot = draw_blue_annotation(b, nuclei)

    # ---- G: FA point
    fa_points, _ = detect_fa_points(g, min_distance_px=args.min_distance_px)
    green_annot = draw_green_annotation(g, fa_points)

    # ---- R: cell area
    cell_mask, cell_contours = detect_cell_area(r, g)
    red_annot = draw_red_annotation(r, cell_contours)

    # 저장
    cv2.imwrite(os.path.join(args.output_dir, "blue_dapi_annotated.png"), blue_annot)
    cv2.imwrite(os.path.join(args.output_dir, "green_fa_point_annotated.png"), green_annot)
    cv2.imwrite(os.path.join(args.output_dir, "red_cell_area_annotated.png"), red_annot)

    # 참고용 mask도 저장
    cv2.imwrite(os.path.join(args.output_dir, "blue_nucleus_mask.png"), nucleus_mask)
    cv2.imwrite(os.path.join(args.output_dir, "red_cell_mask.png"), cell_mask)

    print("[Done] Saved files:")
    print(f"  {os.path.join(args.output_dir, 'blue_dapi_annotated.png')}")
    print(f"  {os.path.join(args.output_dir, 'green_fa_point_annotated.png')}")
    print(f"  {os.path.join(args.output_dir, 'red_cell_area_annotated.png')}")
    print(f"  nuclei detected: {len(nuclei)}")
    print(f"  FA points detected: {len(fa_points)}")
    print(f"  cell components: {len(cell_contours)}")


if __name__ == "__main__":
    main()