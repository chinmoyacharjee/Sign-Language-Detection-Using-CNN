
import numpy as np
import cv2
from color_detection import RangeColorDetector
from keras.models import load_model
from keras.preprocessing import image
import os
from py_expression_eval import Parser

# -----------------Tensorflow session related---------------------
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

# -----------------Skin range---------------------
#skin range
#min_range = np.array([0, 58, 50], dtype = "uint8")         #lower HSV boundary of skin color
#max_range = np.array([30, 255, 255], dtype = "uint8")      #upper HSV boundary of skin color

min_range = np.array([0, 48, 70], dtype = "uint8")          #lower HSV boundary of skin color
max_range = np.array([20, 150, 255], dtype = "uint8")       #upper HSV boundary of skin color

my_skin_detector = RangeColorDetector(min_range, max_range) #Define the detector object

# -----------------Recntangle properties---------------------
posx = 300
posy = 10
width = 300
height = 300

# -----------------frames counting for calculator---------------------
frames_per_item = 30
detection_item_threshold = .8

# -----------------Calculator window properties---------------------
calc_screen = np.zeros((300, 500, 3), np.uint8)
calc_screen[:] = (64, 55, 88)

# -----------------Camera properties---------------------
angle = 180

# -----------------models---------------------
calcultaor_model = "digit_0_10_OP.h5"
digit_model = "digit_model_0_10.h5"
alphabet_model = "alphabetA-Z.h5"

# -----------------models dictionaries---------------------
calculator_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'del': 10, '/': 11, '=': 12, '-': 13, '*': 14, '+': 15, '10': 10}
digit_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10}
alphabet_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13,
                     'O': 14, 'P': 15, 'Q': 16, 'T': 17, 'U': 18, 'V': 19, 'W': 20, 'Y': 21, 'Z': 22}

# -----------------font properties---------------------
font = cv2.FONT_HERSHEY_SIMPLEX
main_window_text_pos, calc_text_pos, res_pos = (80,150), (10, 100), (10, 200)
main_window_font_scale, calc_font_scale = 5, 1
font_color = (255,255,255)
main_window_line_type, calc_line_type = 10, 2


# -----------------Mathmatical---------------------
parser = Parser()

# -----------------skin detect according to range defined avobe---------------------
def return_skin(frame):
    min_range = np.array([0, 58, 50], dtype = "uint8") 
    max_range = np.array([30, 255, 255], dtype = "uint8") 
    image = frame 
    my_skin_detector.setRange(min_range, max_range)    
    only_skin = my_skin_detector.returnFiltered(image, morph_opening=True, blur=True, kernel_size=3, iterations=1)
    binary_filtered = my_skin_detector.returnMask(image, morph_opening=True, blur=True, kernel_size=3, iterations=1)
    return only_skin, binary_filtered 
	
# -----------------Frame Rotation - [camera wise]---------------------
def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

# -----------------Operation Properties---------------------
class Operation(object):
    def __init__(self, model, dictionary, calc = False):
        self.model = model
        self.model_class_dictionary = dictionary
        self.calc = calc
        """
        model = which model to load
        model_class_dictionary = model wise dictionary ds
        calc = is operation is calcultor using sign language
        """
def split_expression_into_arr(expression):
    l = len(expression)
    
    if(expression[0] == "*" or expression[0] == "/"):
        expression = expression[1:l]
        l = l-1
        
    if(expression[l-1] == "*" or expression[l-1] == "/" or expression[l-1] == "+" or expression[l-1] == "-"):
        expression = expression[0:(l-1)]
        print(expression)
    
    
    operand = ""
    arr = []
    for char in expression:
        if(ord(char)>=ord('0') and ord(char)<=ord('9')):
            operand += char
        else:
            arr.append(int(operand))
            operand = ""
            arr.append(char)
    arr.append(int(operand))
    return arr

def get_res(expr):
    try:    
        x = parser.parse(expr).evaluate({})  # 6
        return x, True

    except:
        return "Some thing Wrong", False

       
# -----------------Capture Func---------------------    
def capture(camera_index, op):
    
    classifier = load_model(op.model)
    print("--> model loaded")
    
    class_value_arr = list(op.model_class_dictionary.keys())
    	
    cap = cv2.VideoCapture(camera_index, 0)
#    cap.set(cv2.CAP_PROP_FPS, 6000)
    
    predicted_text = ""
    frame_captured = 0
    predicted_res_arr = []
    
    ok = False
    err = False
    
    while cap.isOpened():
        
        _, frame = cap.read()
        #    frame = cv2.flip(frame, 1)  
            
        frame = rotateImage(frame, angle)
    
        cv2.rectangle(frame, (posx,posy), (posx+width, posy+height), (45, 55, 188), 1)
        img = frame[posy:posy+height, posx:posx+width]
    
        only_skin, binary_filtered = return_skin(img)
        
        contours = cv2.findContours(binary_filtered.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
        
        if(len(contours) > 0):
            contour = max(contours, key = cv2.contourArea)
            if(cv2.contourArea(contour) > 10000):
                
                x, y, w, h = cv2.boundingRect(contour)
                
                rectangle = binary_filtered[y:y+h, x:x+w]
                
                if(w > h):
                    rectangle = cv2.copyMakeBorder(rectangle, int((w-h)/2) , int((w-h)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                elif(h > w):
                    rectangle = cv2.copyMakeBorder(rectangle, 0, 0, int((h-w)/2) , int((h-w)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
                
                cv2.rectangle(frame,(x + posx, y + posy), (x + posx + w, y + posy + h), (0,255,0), 2)

                cv2.imwrite('temp.jpg',  cv2.resize(binary_filtered,(80,80)))

#                cv2.imwrite('temp.jpg',  cv2.resize(rectangle,(80,80)))
#        
                test_image = image.load_img('temp.jpg', target_size = (80, 80))
                test_image = image.img_to_array(test_image)
                test_image = np.expand_dims(test_image, axis = 0)
                result = classifier.predict(test_image)
    
                _, j = np.unravel_index(result.argmax(), result.shape)
                max_value_index = j
                
                pred = class_value_arr[max_value_index]          
                
                predicted_res_arr.append(pred)
                
                if(op.calc and frame_captured >= frames_per_item):
                    
                    max_time_occured_item = max(predicted_res_arr, key = predicted_res_arr.count)
                    max_time_occured_item_occurences = predicted_res_arr.count(max_time_occured_item)
                    
                    prob = max_time_occured_item_occurences / frames_per_item
                    
                    if(prob >= detection_item_threshold):
                        if(max_time_occured_item == "del" and len(predicted_text)!=0):
                            predicted_text = predicted_text[0: len(predicted_text)-1]
                        elif(max_time_occured_item == "="):
                            res, ok = get_res(predicted_text)
                            if(ok):
                                predicted_text  = predicted_text + " = " + str(res)
                                err = False                              
                            else:
                                print("er")
                                err = True
                                            
                        else:
                            if(ok):
                                predicted_text = ""
                                ok = False      
                            err = True
                            predicted_text += max_time_occured_item 

                        predicted_res_arr = []
                        frame_captured = 0  
                        
                cv2.putText(frame, str(pred), main_window_text_pos, font, main_window_font_scale, font_color, main_window_line_type)
                
                if(op.calc):
                    # clearing previous image
                    calc_screen[:] = (64, 55, 88)
                    cv2.putText(calc_screen, predicted_text, calc_text_pos, 
                                font, calc_font_scale, font_color, calc_line_type)
                    if(err):
                        cv2.putText(calc_screen, "Wrong Expression", res_pos, font, calc_font_scale, font_color, calc_line_type)

                frame_captured += 1    
            
        cv2.imshow("Orginal", frame)
        cv2.imshow("Only Skin", only_skin)
        cv2.imshow("Binary Skin", binary_filtered)
        if(op.calc): 
            cv2.imshow("Calculator Screen", calc_screen)
         
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break


def manage(camera_index = "http://192.168.0.102:4747/video", operation = "-a"):
    model = ""
    op_dict = {}
    
    if(operation == "-c"):
        model, op_dict, is_calc = calcultaor_model, calculator_dict, True
    elif(operation == "-d"):
        model, op_dict, is_calc = digit_model, digit_dict, False
    elif(operation == "-a"):
        model, op_dict, is_calc = alphabet_model, alphabet_dict, False
        
    op = Operation(model, op_dict, is_calc)
    capture(camera_index, op)
    
    
    
if(__name__ == "__main__"):
    manage()
    
    