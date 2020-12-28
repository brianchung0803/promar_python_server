import torchvision
import os
# 讀取 label.csv
# 讀取圖片
import numpy as np
import sys
import torch
# Loss function
import torch.nn.functional as F
from PIL import Image
# 讀取資料
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
# 載入預訓練的模型
import torchvision.models as models
# 將資料轉換成符合預訓練模型的形式
import torchvision.transforms as T
# 顯示圖片
import cv2
import matplotlib.pyplot as plt

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.cuda()
model.eval()


COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table",
    "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def get_prediction(img, threshold):
  transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
  img = transform(img) # Apply the transform to the image
  img = img.cuda()
  pred = model([img]) # Pass the image to the model
  pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]["labels"].numpy())] # Get the Prediction Score
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]["boxes"].detach().numpy())] # Bounding boxes
  pred_score = list(pred[0]["scores"].detach().numpy())
  pred_t = [i for i,x in enumerate(pred_score) if x > threshold] # Get list of index with score greater than threshold.
  
  pred_boxes = pred_boxes[:len(pred_t)]
  pred_class = pred_class[:len(pred_t)]
  return pred_boxes, pred_class


def get_recogition(img, threshold=0.7):
    with torch.no_grad():
        transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
        img = transform(img) # Apply the transform to the image
        img = img.cuda()
        pred = model([img]) # Pass the image to the model
        pred_class = [i for i in list(pred[0]['labels'].detach().cpu().numpy())] # Get the Prediction Score
        
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())] # Bounding boxes
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())

        pred_t = [i for i,x in enumerate(pred_score) if x > threshold] # Get list of index with score greater than threshold.
        
        pred_boxes = pred_boxes[:len(pred_t)]
        pred_class = pred_class[:len(pred_t)]
        pred_score = pred_score[:len(pred_t)]
        return pred_boxes, pred_class, pred_score
    
    


def object_detection_api(img, threshold=0.5, rect_th=3, text_size=3, text_th=3):

  boxes, pred_cls = get_prediction(img, threshold) # Get predictions
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
  for i in range(len(boxes)):
    cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
    cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
  plt.figure(figsize=(20,30)) # display the output image
  plt.imshow(img)
  plt.xticks([])
  plt.yticks([])
  plt.show()


# object_detection_api("test.jpg")



import socket 
import select 
import sys 
import threading
import struct

# """The first argument AF_INET is the address domain of the 
# socket. This is used when we have an Internet Domain with 
# any two hosts The second argument is the type of socket. 
# SOCK_STREAM means that data or characters are read in 
# a continuous flow."""
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
  
IP_address = '192.168.50.74'
  

Port = 30000
  

server.bind((IP_address, Port)) 
  


server.listen(100) 
  
list_of_clients = [] 


def clientthread(conn, addr): 
  
    while True: 
        try:
            with open('receive.jpg', 'wb') as img:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    img.write(data)
        except: 
            continue
  
# """Using the below function, we broadcast the message to all 
# clients who's object is not the same as the one sending 
# the message """

# def broadcast(message, connection): 
#     for clients in list_of_clients: 
#         if clients!=connection: 
#             try: 
#                 clients.send(bytes(message,encoding='utf-8')) 
#             except: 
#                 clients.close() 
  
#                 # if the link is broken, we remove the client 
#                 remove(clients) 
  
# """The following function simply removes the object 
# from the list that was created at the beginning of  
# the program"""

import struct
def remove(connection): 
    if connection in list_of_clients: 
        list_of_clients.remove(connection) 



def readbytes(conn,l):
    payload=b''
    while l>0:
        tmp=conn.recv(l)
        l=l-len(tmp)
        payload=payload+tmp
    return payload

def readint(conn):
    return int.from_bytes(readbytes(conn,4), byteorder='big')




def send_bytes(bytes,conn):
    try:
        conn.send(bytes)
    except:
        print("can't send")
        conn.close() 
        remove(conn)
        
def send_int(message,conn): 
    # print((message).to_bytes(4, byteorder="big", signed=True))
    send_bytes((message).to_bytes(4, byteorder="big", signed=True),conn)

def send_float(message,conn): 
    send_bytes(struct.pack('>f', message),conn)
   

print("waiting")
tp = 0
 

# """Accepts a connection request and stores two parameters,  
# conn which is a socket object for that user, and addr  
# which contains the IP address of the client that just  
# connected"""
conn, addr = server.accept() 

# """Maintains a list of clients for ease of broadcasting 
# a message to all available people in the chatroom"""
# list_of_clients.append(conn) 

# prints the address of the user that just connected 
print(addr[0] + " connected")

# creates and individual thread for every user  
# that connects

# _thread.start_new_thread(clientthread,(conn,addr)借我
def image_process(img,conn):
    try:
        pred_boxes, pred_class, pred_score = get_recogition(img)
        length = len(pred_class)
        # print("send length")
        send_int(length,conn)
        print("length", length)
        # print("send rest")
        for i in range(length):
            x = pred_class[i].item()
            # print(x)
            send_int(x,conn)
            send_float(pred_score[i],conn)
            send_float(pred_boxes[i][0][0].item(),conn)
            send_float(pred_boxes[i][0][1].item(),conn)
            send_float(pred_boxes[i][1][0].item(),conn)
            send_float(pred_boxes[i][1][1].item(),conn)
        
    except e:
        print(e)
        return
    

def clientthread(conn,addr):
    while True:
        try:
            l = readint(conn)
            bimage = readbytes(conn,l)
            img = cv2.imdecode(np.frombuffer(bimage, np.uint8), -1)
            # print("image size: "+str(l))
            # print('Image:\n', image_array)
            #等等
            l = readint(conn)
            bpose = readbytes(conn,l)
            pose=bpose.decode('utf-8')
            # print("pose: "+str(pose))
            
            # print("============================="+str(threading.active_count()))
            if threading.active_count()<2:
                t = threading.Thread(target = image_process,args=(img,conn))
                t.start()
            #

        except:
            print("GG")
            break
        if(tp == 1):
            break

    conn.close() 
clientthread(conn,addr)
server.close() 