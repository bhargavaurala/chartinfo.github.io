from __future__ import print_function

import os
import sys
import json
import numpy as np

def main(input_dir, output_dir):
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)
	for anno_file in [fi for fi in os.listdir(input_dir) if '.json' in fi and '.swp' not in fi]:	
		if os.path.isfile(os.path.join(output_dir, anno_file)):
			continue
		print(anno_file)
		with open(os.path.join(input_dir, anno_file), 'r') as f:
			anno = json.load(f)		
		if 'task3' not in anno and 'task4' not in anno:
			print('both task3 and task4 annotation not present. cannot perform axis disambiguation')
		else:
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
				print('swapping x and y axis ticks for', anno_file)
				task4['output']['axes']['x-axis'] = y_axis
				task4['output']['axes']['y-axis'] = x_axis
			else:			
				pass			
			task3 = anno['task3']
			text_blocks = task3['input']['task2_output']['text_blocks']
			axis_titles = [role for role in task3['output']['text_roles'] if role['role'] == 'axis_title']		
			ID_disambiguation = {}						
			for axis_title in axis_titles:
				ID = axis_title['id']			
				text_bb = [text_block['bb'] for text_block in text_blocks if text_block['id'] == ID][0]
				x0, y0 = text_bb['x0'], text_bb['y0']
				w, h = text_bb['width'], text_bb['height']
				xc, yc = x0 + (w * 0.5), y0 + (h * 0.5)
				# if the centre is to the right of the plot left edge and below the plot bottom edge then it is X-axis title
				# if the centre is to the right of the plot left edge and above the plot top edge then it is X-axis title
				if (xc > px0 and yc > py0 + ph) or (xc > px0 and yc < py0):
					ID_disambiguation[ID] = 'x_axis_title'
				# if the centre is to the left of the plot left edge and above the plot bottom edge then it is Y-axis title
				# if the centre is to the right of the plot right edge and above the plot bottom edge then it is Y-axis title
				elif (xc < px0 and yc < py0 + ph) or (xc > px0 + pw and yc < py0 + ph):
					ID_disambiguation[ID] = 'y_axis_title'				
				else:
					print('failed to disambiguate:', anno_file, 'ID', ID, xc, yc, px0, py0, px0 + pw, py0 + ph)
					ID_disambiguation[ID] = 'axis_title'			
			x_axis_ids = [pt['id'] for pt in task4['output']['axes']['x-axis']]
			y_axis_ids = [pt['id'] for pt in task4['output']['axes']['y-axis']]
			for role in task3['output']['text_roles']:
				if role['id'] in ID_disambiguation:
					role['role'] = ID_disambiguation[role['id']]					
				elif role['id'] in x_axis_ids:
					role['role'] = 'x_tick_label'
				elif role['id'] in y_axis_ids:
					role['role'] = 'y_tick_label'
				else:
					pass					
		with open(os.path.join(output_dir, anno_file), 'w') as f:
			anno_str = json.dumps(anno, indent=4, sort_keys=True)
			f.write(anno_str)

if __name__ == '__main__':
	input_dir, output_dir = sys.argv[1:3]
	main(input_dir, output_dir)
