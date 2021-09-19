import pickle 
import pandas as pd 
from pprint import pprint

def resize_bbox( xmin ,ymin ,xmax ,ymax ,W ,H):
	width = xmax - xmin 
	height = ymax - ymin

	center_x = (xmax + xmin)//2
	center_y = (ymax + ymin)//2
	
	new_center_x = ( center_x * 1.0 )/W
	new_center_y = ( center_y * 1.0 )/H
	new_width = ( width * 1.0 ) / W
	new_height = ( height * 1.0 ) / H

	return ( new_center_x ,new_center_y ,new_width ,new_height )


data = pd.read_csv('train_labels.csv')

print( data.iloc[1,0] , len(data) )

final_data = {}

for idx in range(len(data)):
	filename = data.iloc[idx ,0]
	W = data.iloc[idx,1]
	H = data.iloc[idx,2]
	cls_name = data.iloc[idx,3]

	xmin = data.iloc[idx,4]
	ymin = data.iloc[idx,5]
	xmax = data.iloc[idx,6]
	ymax = data.iloc[idx,7]

	if filename not in final_data.keys():
		final_data[filename] = [ [W, H , cls_name , resize_bbox(xmin ,ymin ,xmax ,ymax ,W,H) ] ]
	else :
		final_data[filename].append( [W, H , cls_name , resize_bbox(xmin ,ymin ,xmax ,ymax,W,H) ] )


with open('pickled_data', 'wb') as handle:
    pickle.dump(final_data , handle, protocol=pickle.HIGHEST_PROTOCOL)
