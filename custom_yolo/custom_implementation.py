import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

#detection from notebook
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections 
    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

##Detection from image

# img = "https://www.iii.org/sites/default/files/p_cars_highway_522785736.jpg"
img = "compressed.jpg"
results = model(img)
results.print()
final = cv2.cvtColor(np.squeeze(results.render()), cv2.COLOR_BGR2RGB)
# cv2.imshow('YOLO', final)
cv2.imwrite('test2.jpg', final)
# cv2.waitKey(0)
# cv2.destroyAllWindows() 

results.render()



