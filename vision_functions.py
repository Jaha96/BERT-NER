

import cv2
import os

import numpy as np
from collections import Counter
from scipy import ndimage
from scipy.stats import norm

from numpy.lib.function_base import average

from google.cloud.vision import types

import random

class Symbol:
	def __init__(self, text, x1, y1, x2, y2):
		self.text = text
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2
		self.height = y2 - y1
		self.width = x2 - x1
		self.subsymbols = []
	
	def __str__(self):
		return f'text: {self.text}, x1: {self.x1}, y1: {self.y1}, x2: {self.x2}, y2: {self.y2}'

	def __repr__(self):
		return f'text: {self.text}, x1: {self.x1}, y1: {self.y1}, x2: {self.x2}, y2: {self.y2}'

def img2texts(gray_image, client, lang="ja"):
	(_, binary_image) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	is_success, im_buf_arr = cv2.imencode(".png", gray_image)
	content = im_buf_arr.tobytes()
	image = types.Image(content=content)
	if lang == "":
		response = client.text_detection(image=image)
	else:
		response = client.text_detection(image=image,image_context={"language_hints": [lang]}) #ja = Japanese
	#texts = response.text_annotations
	# return (texts, response)
	
	# with open(os.path.join(output_path, "vision_original.txt"), "w", encoding="utf-8") as text_file:
        #     content = vision_response.text_annotations[0].description
        #     text_file.write(content)
	return response

def findRotation(response):
	texts = response.text_annotations
	document = response.full_text_annotation
	if not document:
		return 0
	# print (texts[0].description.split('\n'))
	scoring = []
	for page in document.pages:
		for block in page.blocks:
			for paragraph in block.paragraphs:
				for word in paragraph.words:
					if len(word.symbols) < 2:
						continue
					first_char = word.symbols[0]
					last_char = word.symbols[-1]
					(top_left, top_right, bottom_right, bottom_left) = first_char.bounding_box.vertices
					first_char_center = (np.mean([v.x for v in first_char.bounding_box.vertices]),np.mean([v.y for v in first_char.bounding_box.vertices]))
					last_char_center = (np.mean([v.x for v in last_char.bounding_box.vertices]),np.mean([v.y for v in last_char.bounding_box.vertices]))

					#upright or upside down
					if np.abs(first_char_center[1] - last_char_center[1]) < np.abs(top_right.y - bottom_right.y): 
						if first_char_center[0] <= last_char_center[0]: #upright
							# print (0)
							scoring.append(0)
						else: #updside down
							# print (180)
							scoring.append(180)
					else: #sideways
						if first_char_center[1] <= last_char_center[1]:
							# print (90)
							scoring.append(90)
						else:
							# print (270)
							scoring.append(270)
	c = Counter(scoring)
	# print(c.most_common(1)[0][0])	
	return c.most_common(1)[0][0]

def fix_orientation(gray_image, vision_response):
	
	degree = findRotation(vision_response)
	if degree != 0:
		gray_image = ndimage.rotate(gray_image, degree, cval=255)
		# print(degree)

	return gray_image



def text_annotation2format(vision_response, gray_image, fname,  output_path, is_debug = False):
	anno = vision_response.text_annotations
	gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
	all_results = gray_image.copy()
	image_h, image_w = gray_image.shape[:2]
	text_with_coordinates = []
	height_collector = []

	for i, a in enumerate(anno):
		vert = a.bounding_poly.vertices
		x1,y1 = vert[0].x, vert[0].y
		x2,y2 = vert[2].x, vert[2].y
		box_area = abs(x1 -x2) * abs(y1 - y2)
		image_area = image_w * image_h
		all_results = cv2.rectangle(all_results, (x1, y1), (x2, y2), (0,255,0), 3)
		if box_area / image_area > 0.35:
			# big paragraph boxes will be skipped
			continue
		height_collector.append(abs(y1 - y2)) 
		text_with_coordinates.append(Symbol(a.description, x1, y1, x2, y2))

	if is_debug:	cv2.imwrite(os.path.join(output_path, fname+ "_visionall.png"), all_results)

	# mean, standard deviation
	mu, std = norm.fit(height_collector)
	sigma_count = 1
	min_threshold = mu - sigma_count * std
	max_threshold = mu + sigma_count * std
	height_collector = [x for x in height_collector if (x >= min_threshold and x < max_threshold)]
	row_average = int(average(height_collector))

	max_thresh = row_average * 3
	# Filter extra big objects by height
	text_with_coordinates = list(filter(lambda sym: (abs(sym.y2 - sym.y1) <= max_thresh), text_with_coordinates))
	
	def _find_intersect_percentage(ly1, ly2, ry1, ry2):
		total_length = max(ly1, ly2, ry1, ry2) - min(ly1, ly2, ry1, ry2)
		intersection_pixels = min(ly2, ry2) - max(ly1, ry1)
		return intersection_pixels / total_length

	def _avg(a, b):
		return int((a + b) / 2)

	def _group_symbols(symbols_with_coordinates, curr_index):
		
		if len(symbols_with_coordinates) <= curr_index:
			return symbols_with_coordinates

		curr_sym = symbols_with_coordinates[curr_index]
		min_thresh = curr_sym.y1 - row_average
		max_thresh = curr_sym.y2 + row_average
		contestants = list(filter(lambda sym: (sym[1].y1 <= max_thresh and sym[1].y1 > min_thresh), enumerate(symbols_with_coordinates)))
		for c_index, contestant in contestants:
			if c_index == curr_index:
				continue
			intersect = _find_intersect_percentage(curr_sym.y1, curr_sym.y2, contestant.y1, contestant.y2)
			dist = abs(contestant.x1 - curr_sym.x2)
			if intersect >= 0.65 and dist < (curr_sym.height):
				joined = Symbol(curr_sym.text + contestant.text, min(curr_sym.x1, contestant.x1), _avg(curr_sym.y1, contestant.y1),
				max(curr_sym.x2, contestant.x2), _avg(curr_sym.y2, contestant.y2))

				#---------------Register subsymbols------------------------#
				if len(symbols_with_coordinates[curr_index].subsymbols) == 0:
					symbols_with_coordinates[curr_index].subsymbols.append(curr_sym)
				symbols_with_coordinates[curr_index].subsymbols.append(contestant)
				joined.subsymbols = symbols_with_coordinates[curr_index].subsymbols
				#---------------Register subsymbols------------------------#


				symbols_with_coordinates[curr_index] = joined
				symbols_with_coordinates.pop(c_index)
				return _group_symbols(symbols_with_coordinates, curr_index)

		# check index
		if len(symbols_with_coordinates) - 1 > curr_index:
			# Find inner guys here with coordinates
			curr_symbol = symbols_with_coordinates[curr_index]
			innerguys_withindex = list(filter(lambda sym: (sym[1].x1 > curr_symbol.x1 and sym[1].y1 > curr_symbol.y1 and sym[1].x2 < curr_symbol.x2 and sym[1].y2 < curr_symbol.y2), enumerate(symbols_with_coordinates)))
			if len(innerguys_withindex) > 0:
				innerguys = []
				innerguyindexes = []
				for innerguyindex, innerguy in innerguys_withindex:
					innerguys.append(innerguy)
					innerguyindexes.append(innerguyindex)
				allguys = curr_symbol.subsymbols + innerguys
				innerguyindexes = sorted(innerguyindexes, reverse=True)
				guys_sorted = sorted(allguys, key=lambda sym: average([sym.x1, sym.x2]))
				new_text = ""
				for guy in guys_sorted:
					new_text += guy.text
				symbols_with_coordinates[curr_index].text = new_text
				symbols_with_coordinates[curr_index].subsymbols = allguys
				for index in innerguyindexes:
					symbols_with_coordinates.pop(index)
		
			return _group_symbols(symbols_with_coordinates, curr_index + 1)

		return symbols_with_coordinates
		
	symbols_grouped = _group_symbols(text_with_coordinates, 0)
	
	save_image = gray_image.copy()
	for symbol in symbols_grouped:
		# print(symbol.text)
		save_image = cv2.rectangle(save_image, (symbol.x1, symbol.y1), (symbol.x2, symbol.y2), (255,0,0), 4)
		i += 1
	if is_debug:	cv2.imwrite(os.path.join(output_path, fname+ "1.png"), save_image)

	symbols_grouped = _group_symbols(symbols_grouped, 0)
	save_image = gray_image.copy()
	for symbol in symbols_grouped:
		# print(symbol.text)
		save_image = cv2.rectangle(save_image, (symbol.x1, symbol.y1), (symbol.x2, symbol.y2), (255,0,0), 4)
		i += 1
	if is_debug:	cv2.imwrite(os.path.join(output_path, fname+ "2.png"), save_image)

	return symbols_grouped, row_average



def find_overlap_percentage(a, b):
    y1 = min(a[1], b[1])
    y2 = max(a[0], b[0])
    if y1 < y2:
        return 0.
    else:
        if a[0] <= b[0] and a[1] >= b[1]:
            return 100.
        else:
            main = abs(a[0] - a[1])
            sub = abs(y2 - y1)
            return sub * 100 / main

def group_rows(text_anno, height_avarage, gray_image, out_dir, name, is_debug=False):
	max_thresh = height_avarage * 3
	text_anno = list(filter(lambda sym: (abs(sym.y2 - sym.y1) <= max_thresh), text_anno))
	text_anno = sorted(text_anno, key=lambda sym: (sym.y1, sym.x1))


	save_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
	for symbol in text_anno:
		# print(symbol.text)
		save_image = cv2.rectangle(save_image, (symbol.x1, symbol.y1), (symbol.x2, symbol.y2), (255,0,0), 4)
	if is_debug:	cv2.imwrite(os.path.join(out_dir, name+ "3.png"), save_image)

	sorder_characters = []
	y_group = []

	column_y1_y2 = []
	for index, ch in enumerate(text_anno):
		x1, y1, x2, y2 = ch.x1, ch.y1, ch.x2, ch.y2
		char_w = abs(x2 - x1)

		if not y_group:     # In the case of y_group empty
			y_group.append(ch)
			column_y1_y2 = [y1, y2]
			continue
		
		percentage = find_overlap_percentage(column_y1_y2, [y1, y2])
		if percentage > 50.:
			y_group.append(ch)
		else:
			y_group = sorted(y_group, key=lambda x: x.x1)
			sorder_characters.append(y_group)
			y_group = []
			y_group.append(ch)
			column_y1_y2 = [y1, y2]

		if len(text_anno) - 1 == index:
			y_group = sorted(y_group, key=lambda x: x.x1)
			sorder_characters.append(y_group)
			y_group = []
	
	save_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
	for row in sorder_characters:
		color = (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
		for symbol in row:
			save_image = cv2.rectangle(save_image, (symbol.x1, symbol.y1), (symbol.x2, symbol.y2), color, 4)
	if is_debug:	cv2.imwrite(os.path.join(out_dir, name+ "4.png"), save_image)


	return sorder_characters