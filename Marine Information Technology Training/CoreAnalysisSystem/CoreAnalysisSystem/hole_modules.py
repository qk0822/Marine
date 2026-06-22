import cv2
import numpy as np


def _safe_percentile(values, q, default=0):
    values = np.asarray(values)
    if values.size == 0:
        return default
    return float(np.percentile(values, np.clip(q, 0, 100)))


def _extract_material_mask(gray):
    grad = cv2.morphologyEx(
        gray,
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )

    # 非极亮区域 + 有纹理区域作为材料候选
    mask = ((gray < 248) | (grad > 4)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=1
    )

    # 去除小噪声，只保留较大的主体区域
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    min_area = max(100, int(0.003 * gray.shape[0] * gray.shape[1]))

    for idx in range(1, num):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == idx] = 255

    if np.count_nonzero(keep) == 0:
        keep = np.full_like(mask, 255)
    return keep


def _ring_contrast(mask_u8, gray, material_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    ring = (dilated > 0) & (mask_u8 == 0) & (material_mask > 0)
    inside = mask_u8 > 0

    if np.count_nonzero(inside) == 0 or np.count_nonzero(ring) < 6:
        return 0.0

    return float(np.mean(gray[ring]) - np.mean(gray[inside]))


def _build_hole_candidate(gray, threshold_val=85):
    material_mask = _extract_material_mask(gray)
    material_bool = material_mask > 0

    # 轻微平滑，减少孤立纹理点影响
    denoise = cv2.GaussianBlur(gray, (3, 3), 0)

    # 局部暗差：孔洞比周围更暗，因此 local_bg - gray 较大
    sigma = max(4.0, min(gray.shape[:2]) / 32.0)
    local_bg = cv2.GaussianBlur(denoise, (0, 0), sigmaX=sigma, sigmaY=sigma)
    dark_contrast = cv2.subtract(local_bg, denoise)

    # 黑帽变换：增强黑色小孔、深色边界和局部暗陷
    blackhat = np.zeros_like(gray)
    for k in (7, 11, 17, 25):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        blackhat = np.maximum(
            blackhat,
            cv2.morphologyEx(denoise, cv2.MORPH_BLACKHAT, kernel)
        )

    vals_gray = denoise[material_bool]
    vals_contrast = dark_contrast[material_bool]
    vals_blackhat = blackhat[material_bool]

    # 阈值映射：
    # 默认 threshold=85 时，整体比旧版 threshold=100 更保守，
    # 但仍保留图中明显黑孔和深灰孔。
    threshold_val = float(threshold_val)
    q = 15 + (threshold_val - 85.0) * 0.10
    percentile_thr = _safe_percentile(vals_gray, q, 58)

    # 绝对灰度阈值和百分位阈值取较合理者，避免不同亮度图像失效
    absolute_thr = max(percentile_thr, threshold_val * 0.72)
    absolute_thr = float(np.clip(absolute_thr, 35, 105))

    # 稍浅孔洞补偿阈值：只允许局部暗差强的区域进入
    soft_thr = min(absolute_thr + 28, 128)
    contrast_thr = max(5.0, _safe_percentile(vals_contrast, 72 - (threshold_val - 85.0) * 0.04, 7))
    blackhat_thr = max(5.0, _safe_percentile(vals_blackhat, 78 - (threshold_val - 85.0) * 0.04, 8))

    # 三路候选：
    # 1. 绝对深色孔洞
    # 2. 局部暗差明显的孔洞
    # 3. 黑帽响应明显的小孔
    candidate_abs = (denoise <= absolute_thr) & material_bool
    candidate_local = (denoise <= soft_thr) & (dark_contrast >= contrast_thr) & material_bool
    candidate_bh = (denoise <= soft_thr) & (blackhat >= blackhat_thr) & material_bool

    candidate = (candidate_abs | candidate_local | candidate_bh).astype(np.uint8) * 255

    # 去除孤立点，再闭合孔洞边界
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        iterations=1
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1
    )

    debug = {
        "material_mask": material_mask,
        "dark_contrast": dark_contrast,
        "blackhat": blackhat,
        "absolute_thr": absolute_thr,
        "soft_thr": soft_thr,
        "contrast_thr": contrast_thr,
        "blackhat_thr": blackhat_thr,
    }
    return candidate, debug


def _filter_components(candidate, gray, debug, min_area=8, max_area=10000, circularity_thresh=0.35):
    try:
        max_area = float(max_area)
    except Exception:
        max_area = np.inf

    min_area = max(1.0, float(min_area))
    material_mask = debug["material_mask"]

    num, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    binary = np.zeros_like(candidate)
    components = []
    h, w = gray.shape[:2]

    for idx in range(1, num):
        area_px = int(stats[idx, cv2.CC_STAT_AREA])
        if area_px < min_area or area_px > max_area:
            continue

        mask = labels == idx
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            continue

        # 允许边缘处的半孔洞被识别；只过滤极大、极满的边界伪影。
        touches_border = xs.min() <= 1 or ys.min() <= 1 or xs.max() >= w - 2 or ys.max() >= h - 2

        bbox_w = int(xs.max() - xs.min() + 1)
        bbox_h = int(ys.max() - ys.min() + 1)
        aspect = max(bbox_w, bbox_h) / (min(bbox_w, bbox_h) + 1e-6)
        extent = area_px / (bbox_w * bbox_h + 1e-6)

        if touches_border and area_px > 900 and extent > 0.70:
            continue

        cnts, _ = cv2.findContours(
            (mask.astype(np.uint8) * 255),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if not cnts:
            continue

        contour = max(cnts, key=cv2.contourArea)
        contour_area = max(1.0, float(cv2.contourArea(contour)))
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * contour_area) / (perimeter * perimeter + 1e-6) if perimeter > 0 else 0.0
        hull = cv2.convexHull(contour)
        hull_area = max(1.0, float(cv2.contourArea(hull)))
        solidity = contour_area / hull_area

        mean_gray = float(np.mean(gray[mask]))
        ring_delta = _ring_contrast(mask.astype(np.uint8) * 255, gray, material_mask)

        # 去掉明显细长纹理/划痕
        if aspect > 5.0 and area_px < 350:
            continue

        # 过满且不够圆、不够暗的候选更像纹理块
        if extent > 0.88 and circularity < 0.32 and mean_gray > debug["absolute_thr"] + 18:
            continue

        # 很小的候选必须足够暗或与周围差异明显
        if area_px < 15:
            if not (mean_gray <= debug["absolute_thr"] - 5 or ring_delta >= 12):
                continue

        # 形状允许不规则，但不能完全不像孔洞
        shape_ok = (
            circularity >= max(0.08, float(circularity_thresh) * 0.35) or
            (solidity >= 0.28 and aspect <= 3.8) or
            (mean_gray <= debug["absolute_thr"] - 8 and aspect <= 4.5)
        )
        if not shape_ok:
            continue

        # 孔洞应足够暗，或者至少比周围环带暗
        dark_ok = (
            mean_gray <= debug["soft_thr"] or
            ring_delta >= 8
        )
        if not dark_ok:
            continue

        binary[mask] = 255
        components.append({
            "area": float(area_px),
            "circularity": float(circularity),
            "solidity": float(solidity),
            "aspect": float(aspect),
            "mean_gray": mean_gray,
            "ring_delta": float(ring_delta),
            "contour": contour,
        })

    return binary, components



def _recover_left_border_holes(binary, gray, debug, min_area=8):
    h, w = gray.shape[:2]
    border_w = max(18, int(0.07 * w))      # 左侧补偿区域，约占图宽 7%
    ignore_edge = 2                         # 忽略最左侧连续黑边

    # 只在左边界附近寻找补偿孔洞，不影响其他区域。
    roi_gray = gray[:, :border_w].copy()
    material = debug.get("material_mask", np.full_like(gray, 255))

    # 采用比主体检测略宽松的深色阈值，但只作用在左边界窄区域。
    local_thr = min(debug["soft_thr"], debug["absolute_thr"] + 28)
    seed = (roi_gray <= local_thr).astype(np.uint8) * 255

    # 忽略最左侧连续黑边，避免所有边界暗区连成一条竖向大块。
    seed[:, :ignore_edge] = 0

    # 去除零散噪声，并轻微闭合半孔洞。
    seed = cv2.morphologyEx(
        seed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        iterations=1
    )
    seed = cv2.morphologyEx(
        seed,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1
    )

    num, labels, stats, _ = cv2.connectedComponentsWithStats(seed, 8)
    recovered = np.zeros_like(binary)

    for idx in range(1, num):
        x, y, bw, bh, area = stats[idx]
        area = int(area)

        # 只补偿边界附近的半孔洞，不处理较远区域。
        if x > border_w * 0.75:
            continue

        # 面积过滤：小于最小面积的噪声不要；过大的边界阴影不要。
        if area < max(5, int(min_area * 0.65)) or area > 420:
            continue

        # 形状过滤：边界半孔洞可以不完整，但不能是细长划痕。
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect > 3.8 and area < 250:
            continue
        if bh > 55 or bw > border_w:
            continue

        comp = labels == idx
        full_mask = np.zeros_like(binary)
        full_mask[:, :border_w][comp] = 255

        # 要求内部确实较暗，或相对周围明显更暗。
        mean_gray = float(np.mean(gray[full_mask > 0]))
        ring_delta = _ring_contrast(full_mask, gray, material)
        if not (mean_gray <= debug["soft_thr"] or ring_delta >= 6):
            continue

        # 如果这个补偿区域已经被原算法识别过，则不重复统计，但保留边界形状。
        # 膨胀一下，让半孔洞的轮廓能靠到图像边界。
        comp_u8 = full_mask.astype(np.uint8)
        comp_u8 = cv2.dilate(
            comp_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1
        )

        # 只允许写入左侧补偿区，确保不影响其他孔洞。
        mask_limit = np.zeros_like(binary)
        mask_limit[:, :border_w] = 255
        comp_u8 = cv2.bitwise_and(comp_u8, mask_limit)
        recovered = cv2.bitwise_or(recovered, comp_u8)

    # 只补充，不删除原检测结果。
    return cv2.bitwise_or(binary, recovered)

def analysis_holes(image, min_area=50, max_area=np.inf, threshold_val=85,
                   circularity_thresh=0.35, clahe_clip=2.0, morph_kernel_size=5):
    if image is None:
        return {}, None, None, None

    # 兼容灰度、BGR、BGRA 输入
    if len(image.shape) == 2:
        gray = image.copy()
        display_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        display_img = image.copy()

    candidate, debug = _build_hole_candidate(gray, threshold_val=threshold_val)
    binary, components = _filter_components(
        candidate,
        gray,
        debug,
        min_area=min_area,
        max_area=max_area,
        circularity_thresh=circularity_thresh
    )

    # 仅补偿左侧边界漏掉的半孔洞，不改变其他区域已有识别结果。
    binary = _recover_left_border_holes(binary, gray, debug, min_area=min_area)

    result_img = display_img.copy()
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 用较粗红色轮廓，接近期望图中标注效果
    cv2.drawContours(result_img, contours, -1, (0, 0, 255), 3)

    # 根据最终 binary 重新统计，确保左边界补偿孔洞也被计入结果。
    areas = []
    circularities = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area <= 0:
            continue
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter * perimeter + 1e-6) if perimeter > 0 else 0.0
        areas.append(area)
        circularities.append(float(circularity))

    total_area = float(sum(areas))
    count = len(areas)

    result = {
        "孔洞数量": count,
        "总面积": total_area,
        "平均面积": total_area / count if count > 0 else 0.0,
        "平均圆形度": float(np.mean(circularities)) if circularities else 0.0,
        "面积列表": areas,
        "最大孔洞面积": float(max(areas)) if areas else 0.0,
        "最小孔洞面积": float(min(areas)) if areas else 0.0,
    }

    return result, gray, binary, result_img
