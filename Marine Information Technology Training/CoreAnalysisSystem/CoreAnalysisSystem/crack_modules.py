import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import measure


def _safe_percentile(values, q, default=0):
    values = np.asarray(values)
    if values.size == 0:
        return default
    return float(np.percentile(values, np.clip(q, 0, 100)))


def _support_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = ((val < 248) & ((sat > 8) | (gray < 235) | (grad > 4))).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    thr = max(80, int(0.002 * gray.shape[0] * gray.shape[1]))
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= thr:
            out[labels == i] = 255
    return out if np.count_nonzero(out) > 0 else np.full_like(mask, 255)


def _line_kernel(length, angle_deg):
    if length % 2 == 0:
        length += 1
    ker = np.zeros((length, length), np.uint8)
    c = length // 2
    a = np.deg2rad(angle_deg)
    dx = int(round(np.cos(a) * c))
    dy = int(round(np.sin(a) * c))
    cv2.line(ker, (c - dx, c - dy), (c + dx, c + dy), 1, 1)
    return ker


def _candidate_mask(image, threshold_val=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    support = _support_mask(image)
    idx = support > 0

    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    enhanced = cv2.bilateralFilter(clahe, 5, 40, 40)
    bg = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=max(5, min(gray.shape) / 30.0), sigmaY=max(5, min(gray.shape) / 30.0))
    contrast = cv2.subtract(bg, enhanced)

    line_resp = np.zeros_like(gray)
    for L in (9, 15, 23, 31):
        for ang in (0, 20, 45, 70, 90, 110, 135, 160):
            line_resp = np.maximum(line_resp, cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, _line_kernel(L, ang)))

    # 分成两类候选：宽而深的裂缝 + 细而线性的裂缝
    dark_q = 8 + np.clip((threshold_val - 100) * 0.04, -2, 4)
    dark_thr = _safe_percentile(enhanced[idx], dark_q, 120)
    dark_soft = _safe_percentile(enhanced[idx], 35, 170)
    con_soft = max(4, _safe_percentile(contrast[idx], 60, 6))
    con_high = max(8, _safe_percentile(contrast[idx], 80, 10))
    line_high = max(10, _safe_percentile(line_resp[idx], 92, 14))

    cand_wide = ((enhanced <= dark_thr) & (contrast >= con_soft) & idx)
    cand_thin = ((line_resp >= line_high) & (contrast >= con_high) & (enhanced <= dark_soft) & idx)
    candidate = (cand_wide | cand_thin).astype(np.uint8) * 255

    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return gray, support, enhanced, candidate


def _endpoints(skel_bool):
    sk = skel_bool.astype(np.uint8)
    neigh = cv2.filter2D(sk, -1, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT)
    ys, xs = np.where((sk > 0) & (neigh == 2))
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


def _angle(xs, ys):
    if len(xs) < 6:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    _, eigvecs, _ = cv2.PCACompute2(pts, mean=None)
    v = eigvecs[0]
    a = abs(np.degrees(np.arctan2(v[1], v[0])))
    if a > 90:
        a = 180 - a
    return float(a)


def _build_segments(candidate, gray, support):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    segs = []
    img_h, img_w = gray.shape[:2]
    low_gray = _safe_percentile(gray[support > 0], 12, 100)

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 12:
            continue
        mask = labels == i
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            continue
        bbox_w = int(xs.max() - xs.min() + 1)
        bbox_h = int(ys.max() - ys.min() + 1)
        aspect = max(bbox_w, bbox_h) / (min(bbox_w, bbox_h) + 1e-6)
        skel = skeletonize(mask)
        length = float(np.count_nonzero(skel))
        if length < max(8, 0.015 * min(img_h, img_w)):
            continue
        mean_width = area / (length + 1e-6)
        angle = _angle(xs, ys)
        mean_gray = float(np.mean(gray[mask]))
        extent = area / (bbox_w * bbox_h + 1e-6)
        # 初筛，排掉明显块状噪声
        if mean_width > 26 and aspect < 2.5:
            continue
        if extent > 0.85 and aspect < 2.0:
            continue
        score = length * (1.0 + 0.20 * min(aspect, 12.0))
        score *= 1.0 + max(0.0, (low_gray - mean_gray) / 60.0)
        # 对接近边界的大块区域降权
        if xs.min() <= 1 or ys.min() <= 1 or xs.max() >= img_w - 2 or ys.max() >= img_h - 2:
            if bbox_w * bbox_h > 0.08 * img_h * img_w:
                score *= 0.35
        segs.append({
            'label': i,
            'mask': mask,
            'area': area,
            'length': length,
            'bbox_w': bbox_w,
            'bbox_h': bbox_h,
            'aspect': float(aspect),
            'mean_width': float(mean_width),
            'angle': angle,
            'mean_gray': mean_gray,
            'score': float(score),
            'endpoints': _endpoints(skel),
            'mean_y': float(np.mean(ys)),
            'mean_x': float(np.mean(xs)),
            'skeleton': skel,
        })
    return segs


def _segments_adjacent(seg_a, seg_b, connect_gap=22, angle_tol=35):
    # 掩膜膨胀后相交，认为属于同一主裂缝
    dil_a = cv2.dilate(seg_a['mask'].astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    if np.any((dil_a > 0) & seg_b['mask']):
        return True

    # 端点距离近，且方向相近，也合并
    if seg_a['endpoints'] and seg_b['endpoints']:
        min_d = 1e9
        for x1, y1 in seg_a['endpoints']:
            for x2, y2 in seg_b['endpoints']:
                d = (x1 - x2) ** 2 + (y1 - y2) ** 2
                if d < min_d:
                    min_d = d
        min_d = float(np.sqrt(min_d))
        if min_d <= connect_gap:
            a1, a2 = seg_a['angle'], seg_b['angle']
            if a1 is None or a2 is None:
                return True
            diff = abs(a1 - a2)
            diff = min(diff, 180 - diff)
            if diff <= angle_tol:
                return True
    return False


def _cluster_segments(segs, connect_gap=22):
    n = len(segs)
    if n == 0:
        return []
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _segments_adjacent(segs[i], segs[j], connect_gap=connect_gap):
                adj[i].append(j)
                adj[j].append(i)

    vis = [False] * n
    clusters = []
    for i in range(n):
        if vis[i]:
            continue
        stack = [i]
        vis[i] = True
        idxs = []
        while stack:
            u = stack.pop()
            idxs.append(u)
            for v in adj[u]:
                if not vis[v]:
                    vis[v] = True
                    stack.append(v)
        clusters.append([segs[k] for k in idxs])
    return clusters


def _cluster_feature(cluster, image_shape):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), np.uint8)
    total_score = 0.0
    total_length = 0.0
    total_area = 0.0
    for seg in cluster:
        mask[seg['mask']] = 255
        total_score += seg['score']
        total_length += seg['length']
        total_area += seg['area']
    ys, xs = np.nonzero(mask)
    bbox_w = int(xs.max() - xs.min() + 1)
    bbox_h = int(ys.max() - ys.min() + 1)
    angle = _angle(xs, ys)
    mean_y = float(np.mean(ys))
    mean_x = float(np.mean(xs))
    span = max(bbox_w, bbox_h)
    mean_width = float(total_area / (total_length + 1e-6)) if total_length > 0 else 0.0
    touches_border = bool(xs.min() <= 1 or ys.min() <= 1 or xs.max() >= w - 2 or ys.max() >= h - 2)

    skel = skeletonize(mask > 0)
    endpoint_count = len(_endpoints(skel))

    # 分支过多的 cluster 往往是纹理误检；适度降权。
    score = total_score / (1.0 + 0.18 * max(0, len(cluster) - 1))
    score = score / (1.0 + 0.04 * max(0, endpoint_count - 2))

    # 底边/外边界附近的大横向暗带，通常不是主裂缝。
    if touches_border and angle is not None and angle < 20 and mean_y > 0.75 * h:
        score *= 0.20
    elif touches_border and bbox_w * bbox_h > 0.08 * h * w:
        score *= 0.55

    # 过宽的暗带更像阴影或边界，不像单条主裂缝。
    if mean_width > 8:
        score *= 0.60

    # 把 cluster 再压成细线显示，避免大片涂色
    display = cv2.dilate(skel.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return {
        'segments': cluster,
        'mask': mask,
        'display': display,
        'score': float(score),
        'raw_score': float(total_score),
        'length': float(total_length),
        'angle': angle,
        'bbox_w': bbox_w,
        'bbox_h': bbox_h,
        'span': span,
        'mean_y': mean_y,
        'mean_x': mean_x,
        'mean_width': mean_width,
        'touches_border': touches_border,
        'segment_count': len(cluster),
        'endpoint_count': endpoint_count,
    }


def _select_main_clusters(clusters, image_shape, max_main=6):
    if not clusters:
        return []
    feats = [_cluster_feature(c, image_shape) for c in clusters]
    feats.sort(key=lambda c: c['score'], reverse=True)
    if not feats:
        return []

    best = feats[0]['score']
    second = feats[1]['score'] if len(feats) > 1 else 0.0
    h, w = image_shape[:2]

    # 若第一条主裂缝明显统治，并且跨度很大，则按单主裂缝图处理。
    if best > 1.75 * max(second, 1e-6) and feats[0]['span'] > 0.35 * max(h, w):
        return [feats[0]]

    kept = []
    for feat in feats:
        if feat['score'] < 0.55 * best:
            continue
        duplicate = False
        for old in kept:
            a1 = 90 if feat['angle'] is None else feat['angle']
            a2 = 90 if old['angle'] is None else old['angle']
            diff = abs(a1 - a2)
            diff = min(diff, 180 - diff)
            if diff < 18:
                # 水平裂缝按 y 抑制，竖向按 x 抑制
                if a1 < 25 and abs(feat['mean_y'] - old['mean_y']) < 10:
                    duplicate = True
                    break
                if a1 >= 25 and abs(feat['mean_x'] - old['mean_x']) < 12:
                    duplicate = True
                    break
        if duplicate:
            continue
        kept.append(feat)
        if len(kept) >= max_main:
            break
    return kept


def _direction_from_component(mask):
    labeled = measure.label(mask > 0)
    props = measure.regionprops(labeled)
    if not props:
        return '未知'
    prop = max(props, key=lambda p: p.area)
    angle_deg = np.rad2deg(prop.orientation)
    if abs(angle_deg) < 30:
        return '纵向裂缝'
    if abs(angle_deg) > 60:
        return '横向裂缝'
    return '斜向裂缝'


def analysis_crack(image, min_area=1000, max_area=np.inf, threshold_val=100, solidity_thresh=0.5,
                   min_noise_area=50, max_main_cracks=6, connect_gap=22):
    if image is None:
        return {}

    gray, support, enhanced, candidate = _candidate_mask(image, threshold_val=threshold_val)
    segments = _build_segments(candidate, gray, support)
    clusters = _cluster_segments(segments, connect_gap=connect_gap)
    main_clusters = _select_main_clusters(clusters, gray.shape, max_main=max_main_cracks)

    filtered_binary = np.zeros_like(gray, dtype=np.uint8)
    crack_contours = []
    crack_lengths = []
    crack_widths = []
    crack_width_distributions = []
    total_crack_area = 0.0

    for c in main_clusters:
        filtered_binary = cv2.bitwise_or(filtered_binary, c['display'])
        cnts, _ = cv2.findContours(c['display'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        crack_contours.extend(cnts)
        area = float(np.count_nonzero(c['display']))
        total_crack_area += area
        crack_lengths.append(c['length'])
        mean_w = area / (c['length'] + 1e-6) if c['length'] > 0 else 0.0
        crack_widths.append(mean_w)
        crack_width_distributions.append({'min': 1.0, 'max': max(1.0, mean_w * 1.5), 'mean': mean_w, 'distribution': np.array([mean_w], dtype=np.float32)})

    result_img = image.copy()
    result_img[filtered_binary > 0] = (0, 0, 255)

    crack_count = len(main_clusters)
    crack_features = {}
    if crack_count > 0:
        largest_idx = int(np.argmax(crack_lengths)) if crack_lengths else 0
        largest_widths = crack_width_distributions[largest_idx]
        max_length = max(crack_lengths) if crack_lengths else 0.0
        rock_area = max(1, int(np.count_nonzero(support)))
        crack_features = {
            '数量': crack_count,
            '总面积': total_crack_area,
            '平均面积': total_crack_area / crack_count,
            '最大裂缝方向': _direction_from_component(filtered_binary),
            '最大裂缝长度': max_length,
            '最大裂缝最大宽度': largest_widths['max'],
            '最大裂缝最小宽度': largest_widths['min'],
            '最大裂缝平均宽度': largest_widths['mean'],
            '平均宽度': float(np.mean(crack_widths)) if crack_widths else 0.0,
            '长度宽度比': (max_length / largest_widths['mean']) if largest_widths['mean'] > 0 else 0.0,
            '裂缝面积占比': total_crack_area / rock_area,
            '累计长度': float(np.sum(crack_lengths)) if crack_lengths else 0.0,
        }

    return {
        '原图': gray,
        '二值图': filtered_binary,
        '结果图': result_img,
        '裂缝轮廓': crack_contours,
        '特征': crack_features,
        '裂缝宽度列表': crack_widths,
        '裂缝宽度分布': crack_width_distributions,
        '候选图': candidate,
        '岩心区域': support,
        '主裂缝簇': main_clusters,
        '候选段数量': len(segments),
    }
