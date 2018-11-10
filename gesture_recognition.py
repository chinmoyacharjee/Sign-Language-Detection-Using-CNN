
import numpy as np
import cv2
import os
from color_detection import RangeColorDetector
from keras.models import load_model
from keras.preprocessing import image

#Firs image boundaries
# min_range = np.array([0, 48, 70], dtype = "uint8") #lower HSV boundary of skin color
# max_range = np.array([20, 150, 255], dtype = "uint8") #upper HSV boundary of skin color

min_range = np.array([0, 58, 50], dtype = "uint8") #lower HSV boundary of skin color
max_range = np.array([30, 255, 255], dtype = "uint8") #upper HSV boundary of skin color

my_skin_detector = RangeColorDetector(min_range, max_range) #Define the detector object

classifier = load_model('digit_0_10_OP.h5')
print("--> model loaded")
#----------------------

#----------------------
#class_value_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13,
#                     'O': 14, 'P': 15, 'Q': 16, 'T': 17, 'U': 18, 'V': 19, 'W': 20, 'Y': 21, 'Z': 22}

class_value_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '/': 10, '-': 11, '*': 12, '+': 13, '10': 14}
class_value_arr = list(class_value_dict.keys())



#digit ={'0': 0, '1': 1, '2': 10, '3': 4, '4': 5, '5': 6, '6': 7, '7': 8, '8': 9, '9': 10, 'D9': 10}



#-------------------------

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (80,150)
fontScale              = 5
fontColor              = (255,255,255)
lineType               = 10


#-------------------------


def return_skin(frame):
    min_range = np.array([0, 58, 50], dtype = "uint8") #lower HSV boundary of skin color
    max_range = np.array([30, 255, 255], dtype = "uint8") #upper HSV boundary of skin color
    image = frame 
    my_skin_detector.setRange(min_range, max_range)
    #For this image we use one iteration of the morph_opening and gaussian blur to clear the noise
    only_skin = my_skin_detector.returnFiltered(image, morph_opening=True, blur=True, kernel_size=3, iterations=1)
    binary_filtered = my_skin_detector.returnMask(image, morph_opening=True, blur=True, kernel_size=3, iterations=1)
    return only_skin, binary_filtered #Save the filtered image
	

def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result



posx, posy, width, height = 300, 10, 300, 300
	
cap = cv2.VideoCapture("http://192.168.0.102:4747/video", 0)
cap.set(cv2.CAP_PROP_FPS, 6000)

predicted_text = ""
frame_captured = 0
predicted_res_arr = []


finel_text_image = np.zeros((300, 400, 3), np.uint8)
finel_text_image[:] = (64, 55, 88)

while cap.isOpened():
    
    _, frame = cap.read()
    #    frame = cv2.flip(frame, 1)  
        
    frame = rotateImage(frame, 180)

    cv2.rectangle(frame, (posx,posy), (posx+width, posy+height), (45, 55, 188), 1)
    img = frame[posy:posy+height, posx:posx+width]

    only_skin, binary_filtered = return_skin(img)
    
    contours = cv2.findContours(binary_filtered.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    
    if len(contours) > 0:
        contour = max(contours, key = cv2.contourArea)
        if(cv2.contourArea(contour) > 10000):
            
            cv2.imwrite('saved_image.jpg',  cv2.resize(binary_filtered,(80,80)))
    
            test_image = image.load_img('saved_image.jpg', target_size = (80, 80))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = classifier.predict(test_image)

#            print(test_image.shape)
#            
            _, j = np.unravel_index(result.argmax(), result.shape)
            max_value_index = j
            
#            -----
#            if(max_value_index != 4):
#                predicted_ALPH = chr(max_value_index + 65)    
#            else: predicted_ALPH = "Nothing"
#            -----
            
            pred = class_value_arr[max_value_index]          
            
            
            predicted_res_arr.append(pred)
            
            if(frame_captured == 30):
                
                txt = max(predicted_res_arr, key = predicted_res_arr.count)
                predicted_text += txt
                predicted_res_arr = []
                frame_captured = 0
            
             
#            predicted_res_arr.append(pred)
            
            cv2.putText(frame, str(pred), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
            cv2.putText(finel_text_image, str(predicted_text), (10, 100), font, 1, fontColor, 3)
            
#            print("Predicted result is: {}, {}".format(max_value_index, pred))
            frame_captured += 1    
        
    cv2.imshow("Orginal", frame)
    cv2.imshow("Only Skin", only_skin)
    cv2.imshow("Binary Skin", binary_filtered)
    cv2.imshow("Final",finel_text_image)
    
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break


    
"""
img = cv2.imread('test.jpg')
img = cv2.resize(img,(320,240))
img = np.reshape(img,[1,320,240,3])

classes = model.predict_classes(img)
"""
#image = np.ones((300, 300, 3), np.uint8)
#image[:] = (0, 0, 0)
#cv2.imshow("ss",image)
#cv2.waitKey()





