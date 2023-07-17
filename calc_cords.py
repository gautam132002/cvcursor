import cv2
import mediapipe as mp
import math
from screeninfo import get_monitors
import numpy as np
import mouse
import pyautogui
import sys
import threading

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def calc_screen_data():
    for m in get_monitors():
        return (m.width, m.height)

    
def calc_euclidean_dist(p1, p2):

    
    print("recived to fun calc_eqi..")
    print(p1,p2)

    (p1_x, p1_y) = p1
    (p2_x, p2_y) = p2
    euc_dist = math.sqrt((p2_x - p1_x) ** 2 + (p2_y - p1_y) ** 2)
    return euc_dist


def correct_x(x,w):
    
    # code to correct and smooth the coordinates

    x = round(x,2)
    
    if x > w:
        x = w
        return x
    elif x < 0:
        x = 0
        return x
    else:
        return x
    
def correct_y(y,h):
    
    y = round(y,2)
    
    if y < 0:
        y = 0
        return 0
    elif y > h:
        y = h
        return y
    else:
        return y


# mouse action ====================================== <3

def move_to(x,y):
    #pyautogui.moveTo(x, y)
    mouse.move(x,y)
    
def left_click(x,y):
    pyautogui.click(x,y)
    #mouse.click(button='left')
    #mouse.is_pressed(button='left')

#def double_click(x,y):
    #pyautogui.click(x, y, clicks = 2, interval=0.1)

def right_click(x,y):
    pyautogui.click(x,y,button= "right")
    #mouse.click('right')
    #mouse.is_pressed(button='right')


def double_click(x,y):
    pyautogui.click(x,y,clicks=2)

def drag(x,y):
    #drag frm current pos to x,y
    pyautogui.dragTo(x,y,button="left")

def toggle():
    pyautogui.hotkey("alt","tab")
    
def copy():
    pyautogui.hotkey('ctrl', 'c')

def paste():
    pyautogui.hotkey('ctrl', 'v')

def scrol_up():
    pyautogui.hscroll(20)
def scrol_down():
    pyautogui.hscroll(-20)

    
#====================================================== <3

def start_cap(cam_inp, print_data, display = 0):

    #variables
    #frame reduction var
    
    frame_red = 100
    smooth_factor = 10

    prev_x, prev_y = 0,0
    cur_x, cur_y = 0,0

    #main loop
    cap = cv2.VideoCapture(cam_inp)
    with mp_hands.Hands(model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()

            height, width, _ = image.shape
            
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            #extracting data
            # {finger : [(tip_x,tip_y),(mcp_x,mcp_y),true or false]}

            left_fing_dict = {"l_ind":[],"l_mid":[],"l_ring":[],"l_pinky":[],"l_thumb":[]}
            right_fing_dict = {"r_ind":[],"r_mid":[],"r_ring":[],"r_pinky":[],"r_thumb":[]}
            cursor_pos = {"x":None,"y":None} #data of cursor
                
 
            
            if results.multi_hand_landmarks:
                 
               
                for hand_index, hand_info in enumerate(results.multi_handedness):
                    
                    hand_label = hand_info.classification[0].label
                    hand_landmarks = results.multi_hand_landmarks[hand_index]


                    if hand_label.upper() == "LEFT":

                        thumb_tip_x = correct_x(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width, width)
                        thumb_tip_y = correct_y(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height, height)
                        thumb_mcp_x = correct_x(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * width, width)
                        thumb_mcp_y = correct_y(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * height, height)

                        l_thumb_data = [(thumb_tip_x, thumb_tip_y),(thumb_mcp_x, thumb_mcp_y)]


                        index_tip_x = correct_x(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width, width)
                        index_tip_y = correct_y(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height, height)
                        index_mcp_x = correct_x(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * width, width) 
                        index_mcp_y = correct_y(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * height, height)

                        l_index_data = [(index_tip_x, index_tip_y),(index_mcp_x, index_mcp_y)]


                        mid_tip_x = correct_x(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * width, width) 
                        mid_tip_y = correct_y(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * height, height)
                        mid_mcp_x = correct_x(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * width, width) 
                        mid_mcp_y = correct_y(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * height, height)

                        l_mid_data = [(mid_tip_x, mid_tip_y),(mid_mcp_x, mid_mcp_y)]


                        ring_tip_x = correct_x(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * width, width) 
                        ring_tip_y = correct_y(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * height, height)
                        ring_mcp_x = correct_x(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * width, width) 
                        ring_mcp_y = correct_y(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * height, height)

                        l_ring_data = [(ring_tip_x, ring_tip_y), (ring_mcp_x, ring_mcp_y)]

                        pinky_tip_x = correct_x(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * width, width) 
                        pinky_tip_y = correct_y(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * height, height)
                        pinky_mcp_x = correct_x(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * width, width) 
                        pinky_mcp_y = correct_y(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * height, height)

                        l_pinky_data = [(pinky_tip_x, pinky_tip_y), (pinky_mcp_x, pinky_mcp_y)]

                        left_fing_dict["l_thumb"] = l_thumb_data
                        left_fing_dict["l_ind"] = l_index_data
                        left_fing_dict["l_mid"] = l_mid_data
                        left_fing_dict["l_ring"] = l_ring_data
                        left_fing_dict["l_pinky"] = l_pinky_data 

                        

                        
                    if hand_label.upper() == "RIGHT":

                        thumb_tip_x = correct_x(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width, width)
                        thumb_tip_y = correct_y(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height, height)
                        thumb_mcp_x = correct_x(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * width, width)
                        thumb_mcp_y = correct_y(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * height, height)

                        r_thumb_data = [(thumb_tip_x, thumb_tip_y),(thumb_mcp_x, thumb_mcp_y)]


                        index_tip_x = correct_x(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width, width)
                        index_tip_y = correct_y(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height, height)
                        index_mcp_x = correct_x(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * width, width)
                        index_mcp_y = correct_y(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * height, height)

                        r_index_data = [(index_tip_x, index_tip_y),(index_mcp_x, index_mcp_y)]


                        mid_tip_x = correct_x(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * width, width)
                        mid_tip_y = correct_y(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * height, height)
                        mid_mcp_x = correct_x(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * width, width)
                        mid_mcp_y = correct_y(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * height, height)

                        r_mid_data = [(mid_tip_x, mid_tip_y),(mid_mcp_x, mid_mcp_y)]


                        ring_tip_x = correct_x(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * width, width)
                        ring_tip_y = correct_y(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * height, height)
                        ring_mcp_x = correct_x(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * width, width)
                        ring_mcp_y = correct_y(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * height, height)

                        r_ring_data = [(ring_tip_x, ring_tip_y), (ring_tip_x, ring_tip_y)]


                        pinky_tip_x = correct_x(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * width, width)
                        pinky_tip_y = correct_y(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * height, height)
                        pinky_mcp_x = correct_x(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * width, width)
                        pinky_mcp_y = correct_y(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * height, height)

                        r_pinky_data = [(pinky_tip_x, pinky_tip_y), (pinky_mcp_x, pinky_mcp_y)]


                        right_fing_dict["r_thumb"] = r_thumb_data
                        right_fing_dict["r_ind"] = r_index_data
                        right_fing_dict["r_mid"] = r_mid_data
                        right_fing_dict["r_ring"] = r_ring_data
                        right_fing_dict["r_pinky"] = r_pinky_data
                    

                    
                    if hand_label.upper() == "LEFT":
                        wrist_x = correct_x(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width, width)
                        wrist_y = correct_y((hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height)-50, height)

                        cursor_pos["x"] = wrist_x
                        cursor_pos["y"] = wrist_y 

                if print_data == 0:
                    print("right = ",right_fing_dict)
                    print()
                    print("left = ",left_fing_dict)
                    print()
                    print("cursor at : " , cursor_pos)
                    print()
                    print("="*50)
                    print()


                else:
                    pass




                

                #rendering frames

                #correcting coordinates
                x_coord = cursor_pos["x"]
                y_coord = cursor_pos["y"]

                w_screen,h_screen = calc_screen_data()

                if x_coord != None and y_coord != None:

                    cv2.circle(image, (int(cursor_pos["x"]), int(cursor_pos["y"])), 20 , (3, 251, 255),2)

                    cv2.rectangle(image, (frame_red,frame_red), (width-frame_red,height-frame_red),(225,225,225),2)
                    
                    new_x = np.interp(width-x_coord, (frame_red, width-frame_red), (0,w_screen))
                    new_y = np.interp(y_coord, (frame_red, height-frame_red), (0,h_screen))

                    new_x =round(new_x,2)
                    new_y = round(new_y,2)

                    #smoothning landmark
                    cur_x = prev_x + (new_x-prev_x)/smooth_factor
                    cur_y = prev_y + (new_y-prev_y)/smooth_factor

                    print(cur_x,cur_y)
                    move_to(cur_x,cur_y)
                    
                    prev_x, prev_y = cur_x, cur_y

                    

                else:
                    pass


                #parsing left fingers
                #index and thumb
                #"l_ind":[],"l_mid":[]
                try:

                    i_tip,i_mcp = left_fing_dict["l_ind"]
                    t_tip,t_mcp = left_fing_dict["l_thumb"]
                    m_tip,m_mcp = left_fing_dict["l_mid"]
                    r_tip,r_mcp = left_fing_dict["l_ring"]
                    p_tip,p_mcp = left_fing_dict["l_pinky"]
                
                    #for left clickkkk
                    i_x, i_y = i_tip
                    t_x, t_y = t_tip

                    gap = calc_euclidean_dist((i_x, i_y),(t_x, t_y))
                    print(gap)
                    if gap < 50:
                        print("left_click")
                        left_click(cur_x,cur_y)

                    #for right click
                        
                    m_x, m_y = m_tip
                    m_m_x, m_m_y = m_mcp

                    if m_y > m_m_y:
                        print("right click")
                        right_click(cur_x,cur_y)
                    
                    #for double click
                    m_t_x, m_t_y = t_mcp

                    if t_x < m_t_x:
                        print("double click")
                        double_click(cur_x,cur_y)

                        
                    #scrool up
                    m_i_x,m_i_y = i_mcp
                    if i_y > m_i_y:
                        print("scroll up")
                        scrol_up()
                        
                    #scrool down
                    r_x,r_y = r_tip
                    m_r_x, m_r_y = r_mcp

                    print("mcp :: ",m_r_y,"tcp :: ",r_y)
                    if r_y > m_r_y:
                        print("scroll down")
                        scrol_down()
                    
                    #toggle
                    p_x,p_y = p_tip
                    m_p_x, m_p_y = p_mcp

                    if p_y > m_p_y:
                        print("toggling")
                        toggle()
                        
                    #copy
                    #paste
                except:
                    pass
                    

     
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())


                
            if display == 0:
                cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            else:
                pass
            
        cap.release()
    
             

              
                
            


