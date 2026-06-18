import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def _odd(value, minimum=3):
    value = int(round(value))
    value = max(value, minimum)
    return value if value % 2 == 1 else value + 1


def _largest_component(mask):
    mask = (mask > 0).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num <= 1:
        return mask
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask)
    out[labels == idx] = 255
    return out


def _core_mask_from_image(image_bgr, core_dark_threshold=18):
    """提取岩心主体，避免黑色背景参与粒度统计。"""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    core = (l_channel > core_dark_threshold).astype(np.uint8) * 255

    h, w = core.shape[:2]
    k = _odd(max(5, min(h, w) // 40), 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, kernel, iterations=2)
    core = ndi.binary_fill_holes(core > 0).astype(np.uint8) * 255
    core = _largest_component(core)
    return core


def _enhance_for_matrix_detection(image_bgr):
    """降噪并增强亮暗差异，用于寻找颗粒之间的暗色胶结/基质边界。"""
    # 均值漂移能把岩心纹理变平滑，保留颗粒边界；比单纯高斯模糊更适合砾岩颗粒。
    h, w = image_bgr.shape[:2]
    if max(h, w) <= 1600:
        smooth = cv2.pyrMeanShiftFiltering(image_bgr, sp=8, sr=18, maxLevel=1)
    else:
        smooth = cv2.bilateralFilter(image_bgr, 7, 55, 55)

    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced, smooth


def _candidate_mask_by_dark_matrix(image_bgr, core_mask, matrix_percentile=35,
                                   matrix_dilate_size=2, morph_kernel_size=3,
                                   use_adaptive_matrix=False):
    """
    粒度图中颗粒常被暗色基质分隔。
    这里先提取暗色基质，再用“岩心主体 - 暗色基质”得到颗粒候选区。
    """
    enhanced, _ = _enhance_for_matrix_detection(image_bgr)
    core_pixels = enhanced[core_mask > 0]
    if core_pixels.size == 0:
        return np.zeros_like(core_mask), enhanced, 0

    matrix_thr = float(np.percentile(core_pixels, matrix_percentile))
    dark_matrix = (enhanced < matrix_thr).astype(np.uint8) * 255

    if use_adaptive_matrix:
        block = _odd(max(15, min(image_bgr.shape[:2]) // 12), 15)
        adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block, 3
        )
        dark_matrix = cv2.bitwise_or(dark_matrix, adaptive)

    dark_matrix = cv2.bitwise_and(dark_matrix, core_mask)

    # 轻微膨胀暗色基质，让颗粒之间留出边界，避免后续全部粘成一个大块。
    dk = _odd(matrix_dilate_size, 1)
    if dk > 1:
        dkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
        dark_matrix = cv2.dilate(dark_matrix, dkernel, iterations=1)

    candidate = cv2.bitwise_and(core_mask, cv2.bitwise_not(dark_matrix))

    k = _odd(morph_kernel_size, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)

    return candidate, enhanced, matrix_thr


def _dense_distance_watershed(binary, min_area, morph_kernel_size=3,
                              seed_min_distance=None, dist_peak_ratio=0.06):
    """用较密的距离变换种子分水岭，专门处理砾岩颗粒粘连。"""
    binary = (binary > 0).astype(np.uint8) * 255
    if np.count_nonzero(binary) == 0:
        return np.zeros(binary.shape, dtype=np.int32)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return np.zeros(binary.shape, dtype=np.int32)

    if seed_min_distance is None:
        # min_area 很小时，仍然给一个较合理的默认峰距，避免一个大颗粒被切得太碎。
        seed_min_distance = max(
            8,
            int(round(morph_kernel_size * 2)),
            int(round(np.sqrt(max(float(min_area), 1.0) / np.pi) * 1.2))
        )

    coords = peak_local_max(
        dist,
        min_distance=int(seed_min_distance),
        threshold_abs=float(dist.max()) * float(dist_peak_ratio),
        labels=(binary > 0),
        exclude_border=False
    )

    markers = np.zeros(binary.shape, dtype=np.int32)
    for idx, (r, c) in enumerate(coords, start=1):
        markers[r, c] = idx

    if markers.max() == 0:
        markers, _ = ndi.label(binary > 0)
        markers = markers.astype(np.int32)

    labels = watershed(-dist, markers, mask=(binary > 0))
    return labels.astype(np.int32)


def _collect_contours_from_labels(labels, min_area, max_area,
                                  min_circularity=0.05, max_aspect_ratio=12.0):
    contours = []
    stats = []

    for label in np.unique(labels):
        if label <= 0:
            continue
        component = (labels == label).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        if not (float(min_area) <= area <= float(max_area)):
            continue

        perimeter = float(cv2.arcLength(cnt, True))
        if perimeter <= 0:
            continue
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter + 1e-8))
        if circularity < float(min_circularity):
            continue

        if len(cnt) >= 5:
            (_, _), (axis_a, axis_b), _ = cv2.fitEllipse(cnt)
            major_axis = float(max(axis_a, axis_b))
            minor_axis = float(min(axis_a, axis_b))
        else:
            _, _, bw, bh = cv2.boundingRect(cnt)
            major_axis = float(max(bw, bh))
            minor_axis = float(min(bw, bh))

        aspect_ratio = major_axis / minor_axis if minor_axis > 0 else np.inf
        if aspect_ratio > float(max_aspect_ratio):
            continue

        eq_diameter = float(np.sqrt(4.0 * area / np.pi))
        contours.append(cnt)
        stats.append({
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "major_axis": major_axis,
            "minor_axis": minor_axis,
            "aspect_ratio": aspect_ratio,
            "eq_diameter": eq_diameter
        })

    return contours, stats




def _recover_compact_dark_grains(image_bgr, enhanced, core_mask, matrix_thr,
                                 min_area=5, max_area=5000,
                                 recovery_kernel_size=None,
                                 min_solidity=0.32,
                                 min_dark_circularity=0.06,
                                 max_dark_aspect_ratio=8.0,
                                 dark_min_l=20,
                                 reject_border_dark=True,
                                 border_margin=2,
                                 core_boundary_margin=4):

    dark = ((enhanced < float(matrix_thr)) & (core_mask > 0)).astype(np.uint8) * 255
    if np.count_nonzero(dark) == 0:
        return np.zeros_like(core_mask, dtype=np.uint8)

    if recovery_kernel_size is None:
        recovery_kernel_size = 7
    recovery_kernel_size = _odd(recovery_kernel_size, 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (recovery_kernel_size, recovery_kernel_size))
    thick_dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)

    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thick_dark = cv2.morphologyEx(thick_dark, cv2.MORPH_CLOSE, small_kernel, iterations=1)
    thick_dark = ndi.binary_fill_holes(thick_dark > 0).astype(np.uint8) * 255
    thick_dark = cv2.bitwise_and(thick_dark, core_mask)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    h, w = core_mask.shape[:2]
    if core_boundary_margin and core_boundary_margin > 0:
        bk = _odd(core_boundary_margin, 3)
        bkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bk, bk))
        eroded_core = cv2.erode(core_mask, bkernel, iterations=1)
        core_boundary_band = (core_mask > 0) & (eroded_core == 0)
    else:
        core_boundary_band = np.zeros_like(core_mask, dtype=bool)

    contours, _ = cv2.findContours(thick_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    recovered = np.zeros_like(core_mask, dtype=np.uint8)

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < max(float(min_area), 3.0):
            continue
        if np.isfinite(float(max_area)) and area > float(max_area):
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        comp_mask = np.zeros_like(core_mask, dtype=np.uint8)
        cv2.drawContours(comp_mask, [cnt], -1, 255, -1)
        pixels = comp_mask > 0
        if not np.any(pixels):
            continue

        l_values = l_channel[pixels]
        l_median = float(np.median(l_values))
        l_mean = float(np.mean(l_values))

        # 近黑色区域多为阴影、岩心破损边界、照片背景残留，不应作为深色颗粒补偿。
        if l_median < float(dark_min_l):
            continue

        if reject_border_dark:
            touches_image_border = (
                x <= int(border_margin) or y <= int(border_margin) or
                x + bw >= w - int(border_margin) or y + bh >= h - int(border_margin)
            )
            if touches_image_border:
                continue

            boundary_ratio = float(np.count_nonzero(core_boundary_band & pixels)) / float(np.count_nonzero(pixels))
            # 贴着岩心轮廓的一整块暗区更可能是边缘阴影或破损，不做深色颗粒补偿。
            if boundary_ratio > 0.12 and l_mean < float(dark_min_l) + 18:
                continue

        perimeter = float(cv2.arcLength(cnt, True))
        if perimeter <= 0:
            continue
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter + 1e-8))

        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 0 else 0.0

        if len(cnt) >= 5:
            (_, _), (axis_a, axis_b), _ = cv2.fitEllipse(cnt)
            major_axis = float(max(axis_a, axis_b))
            minor_axis = float(min(axis_a, axis_b))
        else:
            major_axis = float(max(bw, bh))
            minor_axis = float(min(bw, bh))
        aspect_ratio = major_axis / minor_axis if minor_axis > 0 else np.inf

        if solidity < float(min_solidity):
            continue
        if circularity < float(min_dark_circularity):
            continue
        if aspect_ratio > float(max_dark_aspect_ratio):
            continue

        cv2.drawContours(recovered, [cnt], -1, 255, -1)

    return recovered

def _draw_result(image_bgr, contours, fill_color=(180, 60, 180), alpha=0.62,
                 contour_color=(0, 255, 0), contour_thickness=1,
                 draw_index=False):
    result = image_bgr.copy()
    overlay = image_bgr.copy()

    for idx, cnt in enumerate(contours, start=1):
        cv2.drawContours(overlay, [cnt], -1, fill_color, -1)

    if contours:
        result = cv2.addWeighted(overlay, float(alpha), result, 1.0 - float(alpha), 0)

    for idx, cnt in enumerate(contours, start=1):
        cv2.drawContours(result, [cnt], -1, contour_color, contour_thickness)
        if draw_index:
            m = cv2.moments(cnt)
            if m["m00"] != 0:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
                cv2.putText(result, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def analyze_grains(image, min_area=5, max_area=5000,
                   matrix_percentile=28, dist_peak_ratio=0.06,
                   seed_min_distance=None, matrix_dilate_size=2,
                   min_circularity=0.05, max_aspect_ratio=12.0,
                   recover_dark_grains=True, dark_recovery_kernel_size=7,
                   dark_min_l=20, reject_border_dark=True,
                   draw_index=False, overlay_alpha=0.62):
    if image is None:
        return {}, None, None, None

    internal_morph_kernel_size = 3

    core_mask = _core_mask_from_image(image)
    candidate, enhanced, matrix_thr = _candidate_mask_by_dark_matrix(
        image,
        core_mask,
        matrix_percentile=matrix_percentile,
        matrix_dilate_size=matrix_dilate_size,
        morph_kernel_size=internal_morph_kernel_size,
        use_adaptive_matrix=False
    )

    # 先只对普通候选区做分水岭，避免深色补偿区域把基质边界桥接成大块。
    labels = _dense_distance_watershed(
        candidate,
        min_area=min_area,
        morph_kernel_size=internal_morph_kernel_size,
        seed_min_distance=seed_min_distance,
        dist_peak_ratio=dist_peak_ratio
    )

    contours, grain_stats = _collect_contours_from_labels(
        labels,
        min_area=min_area,
        max_area=max_area,
        min_circularity=min_circularity,
        max_aspect_ratio=max_aspect_ratio
    )

    base_binary = np.zeros(candidate.shape, dtype=np.uint8)
    for cnt in contours:
        cv2.drawContours(base_binary, [cnt], -1, 255, -1)

    # 深色颗粒补偿作为“独立追加对象”处理，而不是并入候选区后一起分水岭。
    dark_recovered = np.zeros_like(candidate, dtype=np.uint8)
    if recover_dark_grains:
        dark_recovered = _recover_compact_dark_grains(
            image,
            enhanced,
            core_mask,
            matrix_thr,
            min_area=min_area,
            max_area=max_area,
            recovery_kernel_size=dark_recovery_kernel_size,
            dark_min_l=dark_min_l,
            reject_border_dark=reject_border_dark
        )

        dark_labels, _ = ndi.label(dark_recovered > 0)
        dark_contours, dark_stats = _collect_contours_from_labels(
            dark_labels.astype(np.int32),
            min_area=min_area,
            max_area=max_area,
            min_circularity=max(0.04, min_circularity),
            max_aspect_ratio=max_aspect_ratio
        )

        # 与普通候选区重叠太多的补偿区域不重复计数。
        for cnt, st in zip(dark_contours, dark_stats):
            tmp = np.zeros(candidate.shape, dtype=np.uint8)
            cv2.drawContours(tmp, [cnt], -1, 255, -1)
            overlap = float(np.count_nonzero((tmp > 0) & (base_binary > 0)))
            tmp_area = float(np.count_nonzero(tmp > 0))
            if tmp_area > 0 and overlap / tmp_area > 0.35:
                continue
            contours.append(cnt)
            grain_stats.append(st)

    final_binary = np.zeros(candidate.shape, dtype=np.uint8)
    for cnt in contours:
        cv2.drawContours(final_binary, [cnt], -1, 255, -1)

    marked_img = _draw_result(
        image,
        contours,
        fill_color=(180, 60, 180),
        alpha=overlay_alpha,
        contour_color=(0, 255, 0),
        contour_thickness=1,
        draw_index=draw_index
    )

    areas = [s["area"] for s in grain_stats]
    eq_diameters = [s["eq_diameter"] for s in grain_stats]
    circularities = [s["circularity"] for s in grain_stats]
    aspect_ratios = [s["aspect_ratio"] for s in grain_stats]

    areas_np = np.asarray(areas, dtype=np.float64)
    diam_np = np.asarray(eq_diameters, dtype=np.float64)
    circ_np = np.asarray(circularities, dtype=np.float64)
    aspect_np = np.asarray(aspect_ratios, dtype=np.float64)

    if diam_np.size:
        d10, d50, d90 = np.percentile(diam_np, [10, 50, 90])
    else:
        d10 = d50 = d90 = 0.0

    core_area = float(np.count_nonzero(core_mask))
    grain_area = float(np.count_nonzero(final_binary))

    result = {
        "粒子数量": int(len(contours)),
        "总面积": float(np.sum(areas_np)) if areas_np.size else 0.0,
        "平均面积": float(np.mean(areas_np)) if areas_np.size else 0.0,
        "面积列表": areas,
        "等效直径列表": eq_diameters,
        "平均等效直径": float(np.mean(diam_np)) if diam_np.size else 0.0,
        "D10": float(d10),
        "D50": float(d50),
        "D90": float(d90),
        "平均圆形度": float(np.mean(circ_np)) if circ_np.size else 0.0,
        "平均长短轴比": float(np.mean(aspect_np)) if aspect_np.size else 0.0,
        "粒度面积占比": float(grain_area / core_area) if core_area > 0 else 0.0,
        "基质阈值": float(matrix_thr),
        "基质百分位": float(matrix_percentile),
        "分水岭峰值比例": float(dist_peak_ratio),
        "深色颗粒补偿": bool(recover_dark_grains),
        "深色补偿面积": float(np.count_nonzero(dark_recovered)) if recover_dark_grains else 0.0,
        "暗粒最低亮度": float(dark_min_l),
        "排除边缘暗区": bool(reject_border_dark),
        "轮廓列表": contours
    }

    return result, enhanced, final_binary, marked_img
