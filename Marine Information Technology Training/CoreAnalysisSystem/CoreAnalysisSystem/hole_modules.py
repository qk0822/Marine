import cv2
import numpy as np


def _safe_percentile(values, q, default=0):
    values = np.asarray(values)
    if values.size == 0:
        return default
    return float(np.percentile(values, np.clip(q, 0, 100)))


def _extract_rock_support(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    mask = ((val < 248) & ((sat > 7) | (gray < 238) | (grad > 4))).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)), iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    support = np.zeros_like(mask)
    min_area = max(200, int(0.003 * image.shape[0] * image.shape[1]))
    for idx in range(1, num):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            support[labels == idx] = 255
    if np.count_nonzero(support) == 0:
        support = mask
    return support


def _build_candidates(image, threshold_val=100, clahe_clip=2.0, morph_kernel_size=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lab_l = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]

    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
    enhanced = clahe.apply(lab_l)
    enhanced = cv2.bilateralFilter(enhanced, 5, 40, 40)

    support = _extract_rock_support(image)
    support_bool = support > 0
    if np.count_nonzero(support_bool) == 0:
        support_bool = np.ones_like(gray, dtype=bool)

    sigma = max(4.5, min(image.shape[:2]) / 52.0)
    local_bg = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=sigma, sigmaY=sigma)
    dark_contrast = cv2.subtract(local_bg, enhanced)

    blackhat = np.zeros_like(enhanced)
    for k in (5, 7, 9, 13):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        blackhat = np.maximum(blackhat, cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel))

    vals_e = enhanced[support_bool]
    vals_g = gray[support_bool]
    vals_c = dark_contrast[support_bool]
    vals_b = blackhat[support_bool]

    sensitivity = np.clip(float(threshold_val) / 120.0, 0.55, 1.35)
    dark_thr = _safe_percentile(vals_e, 8 + 8 * sensitivity, 105)
    soft_dark_thr = _safe_percentile(vals_e, 18 + 9 * sensitivity, 140)
    gray_thr = _safe_percentile(vals_g, 10 + 8 * sensitivity, 118)
    contrast_thr = max(4.5, _safe_percentile(vals_c, 79 - 8 * sensitivity, 8))
    blackhat_thr = max(4.5, _safe_percentile(vals_b, 81 - 8 * sensitivity, 8))
    deep_gray_thr = _safe_percentile(vals_g, 7 + 6 * sensitivity, 95)

    cand_local = (dark_contrast >= contrast_thr) & (enhanced <= soft_dark_thr) & support_bool
    cand_bh = (blackhat >= blackhat_thr) & (enhanced <= soft_dark_thr + 5) & support_bool
    cand_strong_dark = (
        (enhanced <= dark_thr) &
        ((dark_contrast >= contrast_thr * 0.78) | (blackhat >= blackhat_thr * 0.78)) &
        support_bool
    )

    candidate = (cand_local | cand_bh | cand_strong_dark).astype(np.uint8) * 255
    candidate[(gray >= min(220, gray_thr + 80)) & (dark_contrast < contrast_thr * 1.05)] = 0

    # 恢复特别深的小黑孔候选（后续还会再做严格过滤）
    deep_spot = (
        (gray <= deep_gray_thr) &
        (dark_contrast >= contrast_thr * 0.62) &
        support_bool
    ).astype(np.uint8) * 255

    k = max(3, int(morph_kernel_size))
    if k % 2 == 0:
        k += 1
    open_k = min(k, 5)
    close_k = min(max(3, k), 5)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)), iterations=1)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)), iterations=1)

    deep_spot = cv2.morphologyEx(deep_spot, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    debug = {
        'support': support,
        'enhanced': enhanced,
        'gray': gray,
        'dark_contrast': dark_contrast,
        'blackhat': blackhat,
        'dark_thr': dark_thr,
        'soft_dark_thr': soft_dark_thr,
        'gray_thr': gray_thr,
        'deep_gray_thr': deep_gray_thr,
        'contrast_thr': contrast_thr,
        'blackhat_thr': blackhat_thr,
        'deep_spot': deep_spot,
    }
    return candidate, debug


def _ring_contrast(mask_u8, gray, support):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dil = cv2.dilate(mask_u8, k, iterations=1)
    ring = (dil > 0) & (mask_u8 == 0) & (support > 0)
    inside = mask_u8 > 0
    if np.count_nonzero(ring) < 8 or np.count_nonzero(inside) == 0:
        return 0.0, 0.0, 0.0
    inner_mean = float(np.mean(gray[inside]))
    ring_mean = float(np.mean(gray[ring]))
    return ring_mean - inner_mean, inner_mean, ring_mean


def _component_props(mask, debug):
    gray = debug['gray']
    enhanced = debug['enhanced']
    dark_contrast = debug['dark_contrast']
    blackhat = debug['blackhat']
    support = debug['support']
    ys, xs = np.nonzero(mask)
    h, w = gray.shape[:2]
    if xs.size == 0:
        return None
    if xs.min() <= 1 or ys.min() <= 1 or xs.max() >= w - 2 or ys.max() >= h - 2:
        return None

    area_px = int(np.count_nonzero(mask))
    bbox_w = int(xs.max() - xs.min() + 1)
    bbox_h = int(ys.max() - ys.min() + 1)
    aspect = max(bbox_w, bbox_h) / (min(bbox_w, bbox_h) + 1e-6)
    extent = area_px / (bbox_w * bbox_h + 1e-6)

    cnts, _ = cv2.findContours((mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    contour = max(cnts, key=cv2.contourArea)
    contour_area = max(1.0, float(cv2.contourArea(contour)))
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * contour_area) / (perimeter * perimeter + 1e-6) if perimeter > 0 else 0.0
    hull = cv2.convexHull(contour)
    hull_area = max(1.0, float(cv2.contourArea(hull)))
    solidity = contour_area / hull_area

    mean_gray = float(np.mean(gray[mask]))
    mean_enhanced = float(np.mean(enhanced[mask]))
    mean_contrast = float(np.mean(dark_contrast[mask]))
    mean_blackhat = float(np.mean(blackhat[mask]))
    ring_delta, inner_mean, ring_mean = _ring_contrast(mask.astype(np.uint8) * 255, gray, support)

    return {
        'mask': mask,
        'area': float(area_px),
        'bbox_w': bbox_w,
        'bbox_h': bbox_h,
        'aspect': float(aspect),
        'extent': float(extent),
        'circularity': float(circularity),
        'solidity': float(solidity),
        'mean_gray': mean_gray,
        'mean_enhanced': mean_enhanced,
        'mean_contrast': mean_contrast,
        'mean_blackhat': mean_blackhat,
        'ring_delta': float(ring_delta),
        'inner_mean': float(inner_mean),
        'ring_mean': float(ring_mean),
        'contour': contour,
    }


def _accept_component(props, debug, effective_circularity, recovery_mode=False):
    area_px = props['area']
    aspect = props['aspect']
    extent = props['extent']
    circularity = props['circularity']
    solidity = props['solidity']
    mean_gray = props['mean_gray']
    mean_enhanced = props['mean_enhanced']
    mean_contrast = props['mean_contrast']
    mean_blackhat = props['mean_blackhat']
    ring_delta = props['ring_delta']

    # 极小组件单独更严格，减少散点误检。
    if area_px <= 12:
        tiny_ok = (ring_delta >= 11.0 and mean_gray <= debug['gray_thr'] + 8 and (circularity >= 0.22 or solidity >= 0.45))
        if not tiny_ok:
            return False

    # 细长且面积不小的凹槽，更可能是非孔洞。比 v4 更严格。
    if aspect > 3.4 and area_px < 220:
        return False
    if aspect > 2.25 and area_px > 160 and ring_delta < 18.0 and mean_gray > 90:
        return False

    # extent 过滤放宽：对“小而深的黑孔”允许较高 extent；
    # 但对不够暗、不够紧凑的组件仍拒绝。
    if extent > 0.74 and area_px > 30:
        compact_dark = (circularity > 0.48 and ring_delta >= 10.0) or (mean_gray < 80 and ring_delta >= 8.0)
        if not compact_dark:
            return False
    elif extent > 0.68 and area_px > 45:
        if ring_delta < 10.0 and circularity < 0.55:
            return False

    local_dark_ok = (
        ring_delta >= (7.0 if not recovery_mode else 9.0) or
        mean_contrast >= debug['contrast_thr'] * (1.02 if not recovery_mode else 1.08) or
        mean_blackhat >= debug['blackhat_thr'] * (1.02 if not recovery_mode else 1.08)
    )
    if not local_dark_ok:
        return False

    dark_ok = (
        mean_enhanced <= debug['soft_dark_thr'] + (0 if not recovery_mode else -2) or
        mean_gray <= debug['gray_thr'] + (16 if not recovery_mode else 10) or
        ring_delta >= (12.0 if not recovery_mode else 14.0)
    )
    if not dark_ok:
        return False

    # 形状筛选：普通模式更平衡；恢复模式只接收小而深、较紧凑的暗孔。
    if recovery_mode:
        shape_ok = (
            area_px <= 180 and
            aspect < 2.7 and
            (circularity >= max(0.14, effective_circularity * 0.9) or solidity > 0.38) and
            ring_delta >= 9.0
        )
    else:
        shape_ok = (
            circularity >= effective_circularity or
            (aspect < 2.8 and solidity > 0.24 and extent < 0.72) or
            (area_px <= 20 and aspect < 3.0 and ring_delta >= 9.0) or
            (mean_gray < 78 and aspect < 3.4 and solidity > 0.18) or
            (circularity >= 0.42 and ring_delta >= 8.0)
        )
    if not shape_ok:
        return False

    return True


def _filter_components(candidate, debug, min_area, max_area, circularity_thresh):
    support = debug['support']
    support_area = max(1, int(np.count_nonzero(support)))

    try:
        max_area = float(max_area)
    except Exception:
        max_area = np.inf
    if max_area <= 1000:
        effective_max_area = max(max_area, 0.010 * support_area)
    else:
        effective_max_area = max_area
    effective_min_area = max(1.0, float(min_area))
    effective_circularity = max(0.06, float(circularity_thresh) * 0.36)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    binary = np.zeros_like(candidate)
    components = []

    for idx in range(1, num):
        area_px = int(stats[idx, cv2.CC_STAT_AREA])
        if area_px < effective_min_area or area_px > effective_max_area:
            continue
        mask = labels == idx
        props = _component_props(mask, debug)
        if props is None:
            continue
        if _accept_component(props, debug, effective_circularity, recovery_mode=False):
            binary[mask] = 255
            components.append(props)

    # 第二阶段：恢复漏掉的“小而深的黑孔”。
    extra = debug['deep_spot'].copy()
    if np.count_nonzero(binary) > 0:
        dil = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        extra[dil > 0] = 0

    num2, labels2, stats2, _ = cv2.connectedComponentsWithStats(extra, 8)
    for idx in range(1, num2):
        area_px = int(stats2[idx, cv2.CC_STAT_AREA])
        if area_px < effective_min_area or area_px > min(220.0, effective_max_area):
            continue
        mask = labels2 == idx
        props = _component_props(mask, debug)
        if props is None:
            continue
        if _accept_component(props, debug, effective_circularity, recovery_mode=True):
            binary[mask] = 255
            components.append(props)

    return binary, components


def analysis_holes(image, min_area=1, max_area=1000, threshold_val=100,
                   circularity_thresh=0.5, clahe_clip=2.0, morph_kernel_size=5):
    if image is None:
        return {}, None, None, None

    candidate, debug = _build_candidates(
        image,
        threshold_val=threshold_val,
        clahe_clip=clahe_clip,
        morph_kernel_size=morph_kernel_size,
    )
    binary, components = _filter_components(
        candidate,
        debug,
        min_area=min_area,
        max_area=max_area,
        circularity_thresh=circularity_thresh,
    )

    result_img = image.copy()
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_img, contours, -1, (0, 255, 0), 2)

    areas = [c['area'] for c in components]
    circularities = [c['circularity'] for c in components]
    total_hole_area = float(sum(areas))
    hole_count = len(components)

    result = {
        '孔洞数量': hole_count,
        '总面积': total_hole_area,
        '平均面积': total_hole_area / hole_count if hole_count > 0 else 0.0,
        '平均圆形度': float(np.mean(circularities)) if circularities else 0.0,
        '面积列表': areas,
        '最大孔洞面积': float(max(areas)) if areas else 0.0,
        '最小孔洞面积': float(min(areas)) if areas else 0.0,
    }
    return result, debug['enhanced'], binary, result_img
