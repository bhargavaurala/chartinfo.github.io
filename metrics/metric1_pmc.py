import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

def get_confusion_matrix(confusion, unique_labels):
    label_idx_map = {label : i for i, label in enumerate(unique_labels)}
    idx_label_map = {i : label for label, i in label_idx_map.items()}
    cmat = np.zeros((len(label_idx_map), len(label_idx_map)))
    for ID, pair in confusion.items():
        truth, pred = pair
        if pred is None or pred not in label_idx_map:
            continue
        t = label_idx_map[truth]
        p = label_idx_map[pred]
        cmat[t, p] += 1
    norm = cmat.sum(axis=1).reshape(-1, 1)
    cmat /= norm
    return cmat, idx_label_map


def plot_confusion_matrix(cm, classes, output_img_path):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    # fig.tight_layout()
    fig.savefig(output_img_path, bbox_inches='tight')
    plt.show()


def eval_task1(gt_folder, result_folder, output_img_path):
    gt_label_map = {}
    result_label_map = {}
    metrics = {}
    confusion = {}
    result_files = os.listdir(result_folder)
    gt_files = os.listdir(gt_folder)
    for gt_file in gt_files:
        gt_id = ''.join(gt_file.split('.')[:-1])
        with open(os.path.join(gt_folder, gt_file), 'r') as f:
            gt = json.load(f)
            truth = gt['task1']['output']['chart_type'].lower().strip()
        gt_label_map[truth] = gt_label_map[truth] + [gt_id] if truth in gt_label_map else [gt_id]
        confusion[gt_id] = [truth, None]
    for result_file in result_files:
        result_id = ''.join(result_file.split('.')[:-1])
        with open(os.path.join(result_folder, result_file), 'r') as f:
            result = json.load(f)
        try:
            pred = result['task1']['output']
            if 'chart_type' in pred:
                pred = result['task1']['output']['chart_type']
            if isinstance(pred, list):
                pred = pred[0]
            pred = pred.lower()
            if 'stacked' in pred or 'grouped' in pred:
                pred = ' '.join(pred.split(' ')[1:])
            pred = pred.strip()
        except Exception as e:
            print(e)
            print('invalid result json format in {} please check against provided samples'.format(result_file))
            continue
        result_label_map[pred] = result_label_map[pred] + [result_id] if pred in result_label_map else [result_id]
        confusion[result_id][1] = pred
    total_recall = 0.
    total_precision = 0.
    total_fmeasure = 0.
    for label, gt_imgs in gt_label_map.items():
        res_imgs = set(result_label_map[label])
        gt_imgs = set(gt_imgs)
        intersection = gt_imgs.intersection(res_imgs)
        recall = len(intersection) / float(len(gt_imgs))
        precision = len(intersection) / float(len(res_imgs))
        if recall == 0 and precision == 0:
            f_measure = 0.
        else:
            f_measure = 2 * recall * precision / (recall + precision)
        total_recall += recall
        total_precision += precision
        total_fmeasure += f_measure
        metrics[label] = (recall, precision, f_measure)
        if 'bar' in label:
            print('Grouped/Stacked will be ignored in PMC eval, only Horizontal/Vertical is considered')
        print('Recall for class {}: {}'.format(label, recall))
        print('Precision for class {}: {}'.format(label, precision))
        print('F-measure for class {}: {}'.format(label, f_measure))
    total_recall /= len(gt_label_map)
    total_precision /= len(gt_label_map)
    total_fmeasure /= len(gt_label_map)
    print('Average Recall across {} classes: {}'.format(len(gt_label_map), total_recall))
    print('Average Precision across {} classes: {}'.format(len(gt_label_map), total_precision))
    print('Average F-Measure across {} classes: {}'.format(len(gt_label_map), total_fmeasure))

    print('Computing Confusion Matrix')
    classes = sorted(list(gt_label_map.keys()))
    cm, idx_label_map = get_confusion_matrix(confusion, classes)
    plot_confusion_matrix(cm, classes, output_img_path)


if __name__ == '__main__':
    try:
        eval_task1(sys.argv[1], sys.argv[2], sys.argv[3])
    except Exception as e:
        print(e)
        print('Usage Guide: python metric1_pmc.py <ground_truth_folder> <result_folder> <confusion_matrix_path>')

