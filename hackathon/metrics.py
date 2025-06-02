def compute_map(preds, gt_bboxes, gt_labels, iou_threshold=0.5):
    """
    Computes a simplified mean Average Precision (mAP) at IoU=0.5

    Args:
        preds: List of dicts with keys: 'bbox', 'score', 'label'
        gt_bboxes: List of [cx, cy, w, h]
        gt_labels: List of int

    Returns:
        float: average precision for one image
    """
    matched = 0
    total_pred = len(preds)
    total_gt = len(gt_bboxes)

    used_gt = set()

    for pred in preds:
        pred_box = pred["bbox"]
        pred_label = pred["label"]

        for i, (gt_box, gt_label) in enumerate(zip(gt_bboxes, gt_labels)):
            if i in used_gt:
                continue
            if pred_label != gt_label:
                continue

            iou = compute_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                matched += 1
                used_gt.add(i)
                break

    precision = matched / (total_pred + 1e-6)
    recall = matched / (total_gt + 1e-6)
    ap = precision * recall  # Simplified stand-in

    return ap


def compute_iou(box1, box2):
    """
    Computes IoU between two bounding boxes in [cx, cy, w, h] format

    Returns:
        float: IoU value
    """
    def to_corners(box):
        cx, cy, w, h = box
        return [
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2,
        ]

    x1_min, y1_min, x1_max, y1_max = to_corners(box1)
    x2_min, y2_min, x2_max, y2_max = to_corners(box2)

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_area = max(0.0, inter_xmax - inter_xmin) * max(0.0, inter_ymax - inter_ymin)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union = area1 + area2 - inter_area

    return inter_area / (union + 1e-6)
