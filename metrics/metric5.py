
import os
import sys
import json

from e2e_preprocess_task2345_synthetic import preprocess_gt_result

def compute_iou(bb1, bb2):
    ax1 = bb1['x0']
    ay1 = bb1['y0']
    ax2 = bb1['x0'] + bb1['width']
    ay2 = bb1['y0'] + bb1['height']

    bx1 = bb2['x0']
    by1 = bb2['y0']
    bx2 = bb2['x0'] + bb2['width']
    by2 = bb2['y0'] + bb2['height']

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter

    iou = inter / float(union) if union else 0
    return iou


def metric_5(pred_legend_pairs, gt_legend_pairs, gt_type, debug=False):
    if len(gt_legend_pairs) == 0: 
        if len(pred_legend_pairs) == 0:
            return 1
        else:
            return 0

    gt_matched = []
    pred_matched = [False] * len(pred_legend_pairs)
    for i, gt_pair in enumerate(gt_legend_pairs):
        gt_id = gt_pair['id']
        match = None
        for j, pred_pair in enumerate(pred_legend_pairs):
            if pred_matched[j]:
                continue
            pred_id = pred_pair['id']
            if gt_id == pred_id:
                match = pred_pair
                pred_matched[j] = True
                break
        gt_matched.append(match)

    score = 0
    for gt_pair, pred_pair in zip(gt_legend_pairs, gt_matched):
        if pred_pair is None:
            continue
        iou = compute_iou(gt_pair['bb'], pred_pair['bb'])
        if debug:
            print("Text element ID %d IOU: %f" % (gt_pair['id'], iou))
        score += iou
                
    norm_score = score / max(len(gt_legend_pairs), len(pred_legend_pairs))
    return norm_score


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("USAGE: python metric5.py pred_file|pred_dir gt_file|gt_dir [debug] [True|False for e2e preprocessing]")
        exit()
    pred_infile = sys.argv[1]
    gt_infile = sys.argv[2]

    try:
        debug = sys.argv[3].lower() == 'true'
    except:
        debug = False

    try:
        preprocess = sys.argv[4].lower() == 'true'
    except:
        preprocess = False

    if os.path.isfile(pred_infile) and os.path.isfile(gt_infile):
        pred_json = json.load(open(pred_infile))
        gt_json = json.load(open(gt_infile))

        pred_outputs = pred_json['task5']['output']['legend_pairs']
        gt_outputs = gt_json['task5']['output']['legend_pairs']
        gt_type = gt_json['task1']['output']['chart_type']

        score = metric_5(pred_outputs, gt_outputs, gt_type, debug)
        print(score)
    elif os.path.isdir(pred_infile) and os.path.isdir(gt_infile):
        scores = []
        for x in os.listdir(pred_infile):
            pred_file = os.path.join(pred_infile, x)
            gt_file = os.path.join(gt_infile, x)

            pred_json = json.load(open(pred_file))
            gt_json = json.load(open(gt_file))

            if preprocess:
                preprocess_gt_result(gt_json, pred_json)

            pred_outputs = pred_json['task5']['output']['legend_pairs']
            gt_outputs = gt_json['task5']['output']['legend_pairs']
            gt_type = gt_json['task1']['output']['chart_type']

            score = metric_5(pred_outputs, gt_outputs, gt_type, debug)
            scores.append(score)
            print("%s: %f" % (x, score))
        avg_score = sum(scores) / len(scores)
        print("Average Score: %f" % avg_score)
    else:
        print("Error: pred_file and gt_file must both be files or both be directories")
        exit()


