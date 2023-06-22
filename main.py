import cv2
import mediapipe as mp
import keras
from tensorflow.keras.models import load_model
import os
import numpy as np
import control_arduino

# id camera
cap=cv2.VideoCapture(0)

model=load_model('close_open_eyes1.h5')
# mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh()

def load_images_from_folder(folder,X):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),0)
        X.append(img)


list_output_warning=[]
#vòng lặp webcamera
while True:

    #đọc frame ảnh
    _,image=cap.read()
    #convert ảnh từ BGR-->RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Facial landmarks
    result = face_mesh.process(rgb_image)

    #kích thước bức ảnh
    height, width, _ = image.shape

    #nếu phát hiên mặt thì vẽ các điểm
    if result.multi_face_landmarks:
        #vẽ các điểm trên mắt
        for facial_landmarks in result.multi_face_landmarks:
            # vẽ hình chữ nhật
            x_start=facial_landmarks.landmark[68].x
            y_start=facial_landmarks.landmark[68].y
            x_end = facial_landmarks.landmark[195].x
            y_end = facial_landmarks.landmark[195].y
            x_start = int(x_start * width)
            y_start = int(y_start * height)
            x_end = int(x_end * width)
            y_end = int(y_end * height)
            if x_start<=x_end:
                cv2.rectangle(image,(x_start,y_start),(x_end,y_end),(255,0,0),thickness=1)
                img2 = image[y_start:y_end, x_start:x_end, :]
                cv2.imshow('right_eye', img2)
                img2_gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
                img2_gray = cv2.resize(img2_gray, (79, 79))
                x_test = np.array([img2_gray])
                # print(x_test.shape)
                # x_test = x_test.reshape(1, 79, 79, 1)
                # y_hat = model.predict(x_test)
                # y_label = np.argmax(y_hat, axis=1)
                # if y_label[0]==0:
                #     print('right: ',y_label)
            x_start1 = facial_landmarks.landmark[298].x
            y_start1 = facial_landmarks.landmark[298].y
            x_end1 = facial_landmarks.landmark[195].x
            y_end1 = facial_landmarks.landmark[195].y
            x_start1 = int(x_start1 * width)
            y_start1 = int(y_start1 * height)
            x_end1 = int(x_end1 * width)
            y_end1 = int(y_end1 * height)
            if x_start1>=x_end1:
                cv2.rectangle(image, (x_start1, y_start1), (x_end1, y_end1), (255, 0, 0), thickness=1)
                img3 = image[y_start1:y_end1, x_end1:x_start1, :]
                cv2.imshow('left_eye', img3)
                img3_gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                img3_gray = cv2.resize(img3_gray, (79, 79))
                x_test1 = np.array([img3_gray])
                # print(x_test.shape)
                x_test1 = x_test1.reshape(1, 79, 79, 1)
                y_hat1 = model.predict(x_test1)
                y_label1 = np.argmax(y_hat1, axis=1)
                if y_label1[0]==0:
                    print("left:" ,y_label1)
                    image = cv2.putText(image, 'close eye', (50,50), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 2, cv2.LINE_AA)
                    # control_arduino.open_warning()
                else:
                    image = cv2.putText(image, 'open eye', (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2,cv2.LINE_AA)
                    # control_arduino.close_warning()

                #control arduino
                list_output_warning.extend(list(y_label1))
                #30 khung hình thì kiểm tra xem số lần nhắm mắt nhiều hơn mở mắt thì bật cảnh báo không thi tắt cảnh báo
                if len(list_output_warning)==30:
                    if list_output_warning.count(0)>list_output_warning.count(1):
                        control_arduino.open_warning()
                    else:
                        control_arduino.close_warning()
                    list_output_warning=[]


    cv2.imshow('anh',image)
    #điều kiện thoát
    key=cv2.waitKey(1)
    if key==27:
        break
if __name__=="__main__":
    pass
    # X_thu = []
    # folder = 'test'
    # load_images_from_folder(folder, X_thu)
    # X_thu = np.array(X_thu)
    # print(X_thu.shape)
    # X_thu = X_thu.reshape(25, 79, 79, 1)
    # y_hat = model.predict(X_thu)
    # y_label = np.argmax(y_hat, axis=1)
    # print(y_label)