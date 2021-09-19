import torch.nn as nn
import torch 
import cv2
from PIL import Image
import pickle
from pprint import pprint
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse


################### Model Arch 
conv_arch = [
    (7 ,64 ,2 ,3) ,
    "M", 
    (3 ,192 ,1 ,1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNN_layer(nn.Module):

    def __init__(self ,in_size ,out_size , **kwargs):
        super(CNN_layer, self).__init__()
        self.conv = nn.Conv2d( in_size ,out_size ,bias=False ,**kwargs)
        self.bnorm = nn.BatchNorm2d(out_size)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self ,x):
        op = self.conv(x)
        op = self.bnorm(op)
        op = self.lrelu(op)
        return op

class Yolo(nn.Module):

    def __init__(self ,in_size=3 ,**kwargs):
        super(Yolo ,self).__init__()

        self.arch = conv_arch
        self.in_size = in_size 
        self.conv = self.conv_layers(conv_arch)
        self.head = self.fcs(**kwargs)

    def forward(self ,x):
        
        op = self.conv(x)
        temp = torch.flatten(op, start_dim=1)
        op = self.head(temp)

        return op

    # Creates YOLO Convolution Layer according to the Architechture
    def conv_layers(self, arch):
        layers = []
        ip = self.in_size 
        
        for i in arch:
            if type(i) == tuple:
                layers += [ CNN_layer(ip ,i[1] ,kernel_size=i[0] , stride=i[2] , padding=i[3]) ]
                ip = i[1]
            elif type(i) == str:
                layers += [ nn.MaxPool2d(kernel_size=(2,2) ,stride=(2,2)) ]

            else: # for List 
                c1 = i[0]
                c2 = i[1]
                loop = i[2]
                for temp in range(loop):
                    layers += [ CNN_layer(ip, c1[1], kernel_size=c1[0], stride=c1[2], padding=c1[3]) ]
                    layers += [ CNN_layer( c1[1] ,c2[1] ,kernel_size=c2[0] ,stride=c2[2] ,padding=c2[3])]
                    ip = c2[1]

        return nn.Sequential(*layers)

    # Creates Fully Connected Layer
    def fcs(self ,grid_split ,box_to_predict , classes ):
        
        fc= nn. Sequential(
            nn.Flatten(),
            nn.Linear( 1024 * grid_split * grid_split , 512),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(512 , grid_split * grid_split * ( classes + box_to_predict*5 ) )
            )
        return fc

################### Helper Fun

def iou(a,b):

    # a = [ ... , x_center , y_center , W , H ]
    a_x1 = a[..., 0:1] - a[..., 2:3]/2
    a_y1 = a[..., 1:2] - a[..., 3:4]/2
    a_x2 = a[..., 0:1] + a[..., 2:3]/2
    a_y2 = a[..., 1:2] + a[..., 3:4]/2

    # b = [ ... , x_center , y_center , W , H ]    
    b_x1 = b[..., 0:1] - b[..., 2:3]/2
    b_y1 = b[..., 1:2] - b[..., 3:4]/2
    b_x2 = b[..., 0:1] + b[..., 2:3]/2
    b_y2 = b[..., 1:2] + b[..., 3:4]/2

    x1 = torch.max(a_x1, b_x1)
    y1 = torch.max(a_y1, b_y1)
    x2 = torch.min(a_x2, b_x2)
    y2 = torch.min(a_y2, b_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    a_area = abs((a_x2 - a_x1) * (b_y2 - b_y1))
    b_area = abs((a_x2 - a_x1) * (b_y2 - b_y1))

    return intersection / (a_area + b_area - intersection + 1e-6)

def NMS( boxes, iou_thr, thr):
  
    boxes = [box for box in boxes if box[1] > thr]
    boxes = sorted(boxes, key=lambda x: x[1], reverse=True)
    nms_boxes = []

    while boxes:
        cur_box = boxes.pop(0)
        
        temp_boxes = [] 

        for box in boxes:
          iou_score = iou( torch.tensor(cur_box[2:]) , torch.tensor(box[2:])) 
          if box[0] != cur_box[0] or iou_score<iou_thr:
            temp_boxes.append(box)
        
        boxes = temp_boxes
        nms_boxes.append(cur_box)

    return nms_boxes

def tensor_to_image(x):
  temp  = x
  temp = temp.numpy()
  
  img =[]

  for i in range( temp.shape[2]):
    l = []
    for j in range(temp.shape[1]):
      r = temp[0][i][j]
      g = temp[1][i][j]
      b = temp[2][i][j]

      t = [ r,g,b]
      l.append(t)
    img.append(l)
  
  image = np.array(img)
  return image

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img ):
        for t in self.transforms:
            img = t(img)

        return img

def get_images(path ,transform ):
    
    img = Image.open(path)
    img = transform(img)
    return img

################### Prediction Functions

def gen_origbox_from_cellboxes( cell_box , cell=7 ):
    cell_box = cell_box.to("cpu")
    batch_size = cell_box.shape[0]
    cell_box = cell_box.reshape(batch_size, 7, 7, 11)


    cell_box.reshape(batch_size, 7, 7, 11)

    box1 = cell_box[..., 2:6]
    box2 = cell_box[..., 7:11]

    scores = torch.cat((cell_box[..., 1].unsqueeze(0), cell_box[..., 6].unsqueeze(0)), dim=0)
    best_box = scores.argmax(0).unsqueeze(-1)

    final_boxes = box1 * (1 - best_box) + best_box * box2

    '''
    Vectorize Adaptation of Converting X,Y and W,H WRT to Whole Image from WRT to perticular cell where X,Y lies
    '''
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)

    x = 1 / cell * (final_boxes[..., :1] + cell_indices)
    y = 1 / cell * (final_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / cell * final_boxes[..., 2:4]
    img_boxes = torch.cat((x, y, w_y), dim=-1)
    
    pred_class = cell_box[...,:1].argmax(-1).unsqueeze(-1)
    best_score = torch.max(cell_box[..., 1], cell_box[..., 6]).unsqueeze(-1)
    final_boxes = torch.cat((pred_class, best_score, img_boxes), dim=-1)
    
    converted_pred = final_boxes.reshape(cell_box.shape[0], cell * cell, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()

    all_bboxes = []

    for ex_idx in range(cell_box.shape[0]):
        bboxes = []

        for bbox_idx in range(cell * cell):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def predict_bboxes( loader, model, iou_threshold, threshold, pred_format="cells",device="cuda",):
    
    all_pred_boxes = []
    all_true_boxes = []

    model.eval()
    train_idx = 0
    
    for batch_idx, (x, labels ,img_name) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)
        
        batch_size = x.shape[0]
        true_bboxes = gen_origbox_from_cellboxes(labels)
        bboxes = gen_origbox_from_cellboxes(predictions)
        

        for idx in range(batch_size):
            nms_boxes = NMS(bboxes[idx], iou_thr=iou_threshold,thr=threshold)
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes

def predict( img1, model, iou_threshold, threshold, pred_format="cells",device="cuda",):
    img = img1.unsqueeze(0)

    all_pred_boxes = []

    model.eval()
    train_idx = 0
    
    img = img.to(device)
    
    with torch.no_grad():
        predictions = model(img)
    
    batch_size = img.shape[0]
    bboxes = gen_origbox_from_cellboxes(predictions)

    nms_boxes = NMS(bboxes[0], iou_thr=iou_threshold,thr=threshold)
    for nms_box in nms_boxes:
        all_pred_boxes.append([train_idx] + nms_box)

    train_idx += 1

    model.train()

    return all_pred_boxes

def save_output_img(path ,model):
  
  print('Predicting')
  transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])
  img = get_images(path ,transform)

  pred = predict(img ,model ,iou_threshold=0.3, threshold=0.2 )

  img = tensor_to_image(img)
  img = img*255

  x,y,W,H = pred[0][3] ,pred[0][4] ,pred[0][5] ,pred[0][6]
  upper_left_x = x - W / 2
  upper_left_y = y - H / 2
  #rect_pred = patches.Rectangle( (upper_left_x * 448, upper_left_y * 448), W * 448, H * 448, linewidth=1, edgecolor="r", facecolor="none",)
  
  cv2.rectangle(img, ( int(upper_left_x * 448), int(upper_left_y * 448) ), ( int(upper_left_x * 448 + W * 448) , int(upper_left_y * 448 + H * 448) ), (0,255,0), 2)

  cv2.imwrite( 'Output/op.jpg', img )
  #fig, ax = plt.subplots(1)
  #ax.imshow(img)
  #ax.add_patch(rect_pred)
  #plt.show()


################ Main Function 

if __name__ == "__main__":
	argp = argparse.ArgumentParser()
	argp.add_argument("-i", "--image", required=True,help="Image Path")
	argp.add_argument("-m", "--model", required=True,help="Model Path")

	args = vars(argp.parse_args())

	print('Loading Model')
	model = torch.load( args["model"] )
	model = model.to("cuda")

	save_output_img(args['image'] ,model)
	print('Output saved')