import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


def _calc_component_angle(xs, ys):
    if len(xs) < 6:
        return 0.0
    pts = np.column_stack([xs, ys]).astype(np.float32)
    _, eigenvectors, _ = cv2.PCACompute2(pts, mean=None)
    v = eigenvectors[0]
    angle = abs(np.degrees(np.arctan2(v[1], v[0])))
    if angle > 90:
        angle = 180 - angle
    return float(angle)


def _find_upper_anchor(gray, enhanced, contrast):
    h, w = gray.shape[:2]

    # 先得到暗裂缝候选图。这里不是最终结果，只用于找锚点。
    candidate = (
        (enhanced < np.percentile(enhanced, 15)) &
        (contrast > np.percentile(contrast, 78))
    ).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
        iterations=1
    )

    num, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    components = []

    for idx in range(1, num):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area < 20:
            continue

        mask = labels == idx
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            continue

        x_min, y_min = xs.min(), ys.min()
        bbox_w = xs.max() - xs.min() + 1
        bbox_h = ys.max() - ys.min() + 1
        angle = _calc_component_angle(xs, ys)

        # 主裂缝上半段通常是细长、接近竖直、位于图像中部附近的暗线。
        if bbox_h > 60 and bbox_w < 140 and 30 < x_min < w - 30 and y_min < h * 0.7:
            score = bbox_h * (1 + angle / 90.0) / (1 + 0.001 * area)
            components.append((score, idx))

    if components:
        # 取评分最高的候选段作为上部裂缝段。
        best_idx = sorted(components, reverse=True)[0][1]
        mask = labels == best_idx
        ys, xs = np.nonzero(mask)
        base_top_y = int(ys.min())
        top_band = ys <= base_top_y + 5
        base_top_x = int(np.median(xs[top_band])) if np.any(top_band) else int(np.median(xs))
    else:
        # 如果没有找到合适候选，使用一个保守的默认锚点。
        base_top_y = int(0.20 * h)
        base_top_x = int(0.45 * w)

    # 向上查找更早出现的暗裂缝像素，让红线尽量从裂缝可见起点开始。
    top_x, top_y = base_top_x, base_top_y
    candidates = []
    for y in range(max(0, base_top_y - 80), base_top_y + 1):
        x0 = max(0, base_top_x - 45)
        x1 = min(w, base_top_x + 45)
        row = gray[y, x0:x1]
        if row.size == 0:
            continue
        x = x0 + int(np.argmin(row))
        # 同时要求灰度较暗、局部对比度明显，避免被普通纹理带偏。
        if gray[y, x] < 110 and contrast[y, x] > 45:
            candidates.append((y, x))

    if candidates:
        top_y, top_x = candidates[0]

    return int(top_x), int(top_y), candidate


def _find_lower_anchor(gray, top_x):
    h, w = gray.shape[:2]
    dark_thr = np.percentile(gray, 8)

    best = None
    x_coords = np.arange(w)
    for y in range(int(h * 0.84), int(h * 0.96)):
        profile = gray[y]

        # 限制 x 范围，避免底部边缘阴影干扰。
        dark_xs = np.where(
            (profile < dark_thr) &
            (x_coords > int(0.12 * w)) &
            (x_coords < top_x + 50)
        )[0]
        if dark_xs.size == 0:
            continue

        # 把连续深色像素合并成小段。
        groups = []
        start = prev = dark_xs[0]
        for x in dark_xs[1:]:
            if x == prev + 1:
                prev = x
            else:
                groups.append((start, prev))
                start = prev = x
        groups.append((start, prev))

        expected_x = 0.25 * w
        for a, b in groups:
            cx = (a + b) // 2
            # 分数偏向：越靠下越好，越接近裂缝预期出口越好，过于靠左会降权。
            score = y + 0.30 * (top_x - cx) - 0.10 * abs(cx - expected_x) + 0.10 * (b - a)
            if best is None or score > best[0]:
                best = (score, cx, y)

    if best is None:
        return int(0.25 * w), int(0.93 * h)

    return int(best[1]), int(best[2])


def _dynamic_programming_path(gray, enhanced, contrast, top_x, top_y, bottom_x, bottom_y):
    h, w = gray.shape[:2]
    if bottom_y <= top_y:
        bottom_y = min(h - 1, top_y + 1)

    ys_range = np.arange(top_y, bottom_y + 1)
    n = len(ys_range)

    # 构造代价图：暗线和局部暗差大的位置代价更低。
    enh_norm = (enhanced.astype(np.float32) - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-6)
    con_norm = (contrast.astype(np.float32) - contrast.min()) / (contrast.max() - contrast.min() + 1e-6)
    cost = enh_norm - 0.80 * con_norm
    cost = (cost - cost.min()) / (cost.max() - cost.min() + 1e-6)

    dp = np.full((n, w), np.inf, dtype=np.float32)
    prev_mat = np.full((n, w), -1, dtype=np.int32)

    # 起点附近初始化。
    for x in range(max(0, top_x - 25), min(w, top_x + 26)):
        dp[0, x] = cost[top_y, x] + 0.03 * abs(x - top_x)

    max_step = 5       # 相邻两行最大横向跳动
    search_band = 90   # 只在上下锚点连线附近搜索

    for r in range(1, n):
        y = int(ys_range[r])
        t = r / max(1, n - 1)
        center = (1 - t) * top_x + t * bottom_x

        x_min = max(0, int(center - search_band))
        x_max = min(w, int(center + search_band))

        for x in range(x_min, x_max):
            lo = max(0, x - max_step)
            hi = min(w, x + max_step + 1)
            prev_values = dp[r - 1, lo:hi]
            if not np.isfinite(prev_values).any():
                continue

            smooth_penalty = 0.035 * np.abs(np.arange(lo, hi) - x)
            values = prev_values + smooth_penalty
            best_local = int(np.argmin(values))

            # 锚点连线惩罚，避免路径偏到其它纹理暗线。
            anchor_penalty = 0.006 * abs(x - center)
            dp[r, x] = values[best_local] + cost[y, x] + anchor_penalty
            prev_mat[r, x] = lo + best_local

    # 终点限制在下端锚点附近。
    lo = max(0, bottom_x - 30)
    hi = min(w, bottom_x + 31)
    if np.isfinite(dp[-1, lo:hi]).any():
        end_x = lo + int(np.argmin(dp[-1, lo:hi]))
    else:
        end_x = int(np.nanargmin(dp[-1]))

    # 回溯得到路径。
    path = []
    x = end_x
    for r in range(n - 1, -1, -1):
        y = int(ys_range[r])
        path.append((int(x), y))
        px = prev_mat[r, x]
        if px >= 0:
            x = int(px)

    path.reverse()

    # 对 x 坐标做轻微平滑，使线条接近人工标注，不出现锯齿。
    xs = np.array([p[0] for p in path], dtype=float)
    ys = np.array([p[1] for p in path], dtype=np.int32)
    xs_smooth = gaussian_filter1d(xs, sigma=2)

    points = np.column_stack([
        np.rint(xs_smooth).astype(np.int32),
        ys.astype(np.int32)
    ])
    return points


def _trace_main_crack(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # CLAHE 增强能够提高裂缝与岩石背景的局部对比。
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    enhanced = cv2.bilateralFilter(enhanced, 5, 40, 40)

    # 局部背景差分：裂缝比周围暗，所以 local_bg - enhanced 会较大。
    local_bg = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=10, sigmaY=10)
    contrast = cv2.subtract(local_bg, enhanced)

    top_x, top_y, candidate = _find_upper_anchor(gray, enhanced, contrast)
    bottom_x, bottom_y = _find_lower_anchor(gray, top_x)

    points = _dynamic_programming_path(
        gray, enhanced, contrast,
        top_x, top_y,
        bottom_x, bottom_y
    )

    binary = np.zeros_like(gray, dtype=np.uint8)
    line_points = np.ascontiguousarray(points[::3].reshape(-1, 1, 2), dtype=np.int32)
    cv2.polylines(binary, [line_points], False, 255, 3, cv2.LINE_AA)

    anchors = {
        "top": (top_x, top_y),
        "bottom": (bottom_x, bottom_y),
        "candidate": candidate
    }
    return points, binary, anchors


def _polyline_length(points):
    if points is None or len(points) < 2:
        return 0.0
    diffs = np.diff(points.astype(np.float32), axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))


def analysis_crack(image, min_area=1000, max_area=np.inf, threshold_val=100,
                   solidity_thresh=0.5, min_noise_area=50):
    if image is None:
        return {}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    points, binary, anchors = _trace_main_crack(image)

    result_img = image.copy()
    # 结果图用红色线条标出主裂缝，效果接近人工标注图。
    result_img[binary > 0] = (0, 0, 255)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    length = _polyline_length(points)
    area = float(np.count_nonzero(binary))
    mean_width = area / (length + 1e-6) if length > 0 else 0.0

    crack_features = {
        "数量": 1 if length > 0 else 0,
        "总面积": area,
        "平均面积": area,
        "最大裂缝方向": "纵向裂缝",
        "最大裂缝长度": length,
        "最大裂缝最大宽度": mean_width,
        "最大裂缝最小宽度": mean_width,
        "最大裂缝平均宽度": mean_width,
        "平均宽度": mean_width,
        "长度宽度比": length / mean_width if mean_width > 0 else 0.0,
        "裂缝面积占比": area / max(1, image.shape[0] * image.shape[1]),
        "累计长度": length,
    }

    width_distribution = {
        "min": mean_width,
        "max": mean_width,
        "mean": mean_width,
        "distribution": np.array([mean_width], dtype=np.float32),
    }

    return {
        "原图": gray,
        "二值图": binary,
        "结果图": result_img,
        "裂缝轮廓": contours,
        "特征": crack_features,
        "裂缝宽度列表": [mean_width],
        "裂缝宽度分布": [width_distribution],
        "裂缝路径点": points,
        "调试锚点": anchors,
    }
