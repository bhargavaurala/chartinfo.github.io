import sys
import json

import numpy as np

from metric2 import bbox_iou

# this script will check task2 output of the input submission json, match it to the ground truth taskx[input][task2_output][text_blocks]
# to obtain mapping between ground truth ids and submitted ids. 
# all submitted ids not covering any gt will be given a unique negative id
# all submitted ids matching gt will be given the id of the gt it matches
# in the result json, unassigned ticklabels and legend markers are given negative indices
# these negative indices are retained in the preprocessing - they will cause a mismatch with the ground truth
# a new json will be generated for submitted json where the ids are changed in taskx['output']

iou_threshold_match = 0.5

def get_bbox(bb, mode='xywh'):
	x0 = bb['x0']
	y0 = bb['y0']
	width = bb['width']
	height = bb['height']
	if mode == 'xywh':		
		return (x0, y0, width, height)
	else:
		return (x0, y0, x0 + width, y0 + height)

def get_res_gt_id_map(gt, res):
	# get ground truth bboxes and ids
	gt_text_blocks = gt['task3']['input']['task2_output']['text_blocks']	
	gt_bboxes = np.array([get_bbox(tb['bb'], 'xyxy') for tb in gt_text_blocks])
	gt_ids = [tb['id'] for tb in gt_text_blocks]	
	# print(gt_bboxes)
	# print(gt_ids)
	# get result bboxes and ids
	res_text_blocks = res['task2']['output']['text_blocks']
	res_bboxes = np.array([get_bbox(tb['bb'], 'xyxy') for tb in res_text_blocks])
	res_ids = [tb['id'] for tb in res_text_blocks]
	# print(res_bboxes)
	# print(res_ids)
	# find mapping between result ids and gt ids
	res_gt_id_map = {}
	ious = bbox_iou(gt_bboxes, res_bboxes)
	for g, gt_bbox in enumerate(gt_bboxes):
		r = np.argmax(ious[g, :])
		if np.max(ious[g, :]) >= iou_threshold_match:
			res_gt_id_map[res_ids[r]] = gt_ids[g]
	# print(ious)
	# print('number of matched boxes', len(res_gt_id_map))
	# for r, g in res_gt_id_map.items():
	# 	print(r, ':', g)
	# fill unique negative ids for false positive result detections
	neg = 1
	for r, res_bbox in enumerate(res_bboxes):
		# if we have found a match to a gt, ignore
		if res_ids[r] in res_gt_id_map:
			continue
		# if a result text box is not found in the gt (false positive detection), allot it a unique negative id
		else:
			res_gt_id_map[res_ids[r]] = -neg
			neg += 1
	return res_gt_id_map

def correct_xy_axes_gt(anno):
	task4 = anno['task4']
	x_axis = task4['output']['axes']['x-axis']
	y_axis = task4['output']['axes']['y-axis']
	plot_bb = task4['output']['_plot_bb']
	px0, py0, ph, pw = plot_bb['x0'], plot_bb['y0'], plot_bb['height'], plot_bb['width']
	x_axis_xs = [pt['tick_pt']['x'] for pt in x_axis]
	x_axis_ys = [pt['tick_pt']['y'] for pt in x_axis]
	x_xvar, x_yvar = np.std(x_axis_xs), np.std(x_axis_ys)		
	# y coords is changing more than x coords in x-axis means x-axis is not horizontal
	if x_yvar > x_xvar:
		# print('swapping x and y axis ticks')
		task4['output']['axes']['x-axis'] = y_axis
		task4['output']['axes']['y-axis'] = x_axis
	else:			
		pass

def preprocess_gt_result(gt, res):
	# correct x and y axes in ground truth synthetic
	correct_xy_axes_gt(gt)
	res_gt_id_map = get_res_gt_id_map(gt, res)
	# change ids in task3 
	text_roles = res['task3']['output']['text_roles']
	for text_role in text_roles:
		text_role['id'] = res_gt_id_map[text_role['id']]
	# in the result json, unassigned ticklabels and legend markers are given negative indices
	# these negative indices are retained in the preprocessing - they will cause a mismatch with the ground truth
	# change ids in task4 x-axis
	x_axis_ticks = res['task4']['output']['axes']['x-axis']
	for x_axis_tick in x_axis_ticks:
		x_axis_tick['id'] = res_gt_id_map[x_axis_tick['id']] if x_axis_tick['id'] in res_gt_id_map else x_axis_tick['id']
	# change ids in task4 y-axis
	y_axis_ticks = res['task4']['output']['axes']['y-axis']
	for y_axis_tick in y_axis_ticks:
		y_axis_tick['id'] = res_gt_id_map[y_axis_tick['id']] if y_axis_tick['id'] in res_gt_id_map else y_axis_tick['id']
	# change ids in task5
	legend_pairs = res['task5']['output']['legend_pairs']
	for legend_pair in legend_pairs:
		legend_pair['id'] = res_gt_id_map[legend_pair['id']] if legend_pair['id'] in res_gt_id_map else legend_pair['id']

if __name__ == '__main__':
	gt_file = '/home/buralako/git/Charts_Competition_Submissions/gt_syn_jsons/task2/2.json'	
	res_file = '/home/buralako/git/Charts_Competition_Submissions/CUBS/task2-2345/2.json'
	with open(gt_file, 'r') as f:
		gt = json.load(f)
	with open(res_file, 'r') as f:
		res = json.load(f)
	preprocess_gt_result(gt, res)
	with open('/home/buralako/git/Charts_Competition_Submissions/gt_corr.json', 'w') as f:
		f.write(json.dumps(gt, indent=4, sort_keys=True))
	with open('/home/buralako/git/Charts_Competition_Submissions/res_corr.json', 'w') as f:
		f.write(json.dumps(res, indent=4, sort_keys=True))
