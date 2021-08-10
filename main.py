import cv2
import face_recognition
import numpy as np
from tensorflow import keras

model = keras.models.load_model("./model_v6_23.hdf5")
emotion_dict = {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}


def emotion_detect_in_img(img):
    img = cv2.imread(img)

    face_location = face_recognition.face_locations(img)

    # If faces were found, we will mark it on frame with blue dots
    for face_location in face_location:
        top, right, bottom, left = face_location

        face_image = img[top:bottom, left:right]
        face_image = cv2.resize(face_image, (48, 48))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
        predection_list = model.predict([face_image])
        predicted_class = np.argmax(predection_list)
        accurecy = predection_list[0][predicted_class]
        label_map = dict((v, k) for k, v in emotion_dict.items())
        predicted_label = label_map[predicted_class]

        print(face_location, accurecy, predicted_label)

        color = (0
                 , 255, 255)

        pt1 = (left, top)
        pt2 = (right, bottom)
        cv2.rectangle(img=img, pt1=pt1, pt2=pt2, color=color, thickness=2)
        cv2.putText(img, predicted_label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, str(accurecy)[0:4], (right, bottom), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

    cv2.imshow('image', img)

    # Maintain output window utill
    # user presses a key
    cv2.waitKey(0)

    # Destroying present windows on screen
    cv2.destroyAllWindows()


# emotion_detect_in_img('./2.jpg')

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    r_val, frame = vc.read()
else:
    r_val = False

while r_val:
    frame = cv2.flip(frame, 1)

    face_location = face_recognition.face_locations(frame)

    # If faces were found, we will mark it on frame with blue dots
    for face_location in face_location:
        top, right, bottom, left = face_location

        face_image = frame[top:bottom, left:right]
        face_image = cv2.resize(face_image, (48, 48))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
        predection_list = model.predict([face_image])
        predicted_class = np.argmax(predection_list)
        accurecy = predection_list[0][predicted_class]
        label_map = dict((v, k) for k, v in emotion_dict.items())
        predicted_label = label_map[predicted_class]

        print(face_location, accurecy, predicted_label)

        color = (0
                 , 255, 255)

        pt1 = (left, top)
        pt2 = (right, bottom)
        cv2.rectangle(img=frame, pt1=pt1, pt2=pt2, color=color, thickness=2)
        cv2.putText(frame, predicted_label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, str(accurecy)[0:3], (right + 10, bottom - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("preview", frame)
    r_val, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27 or cv2.getWindowProperty('preview', cv2.WND_PROP_VISIBLE) < 1:  # exit on ESC
        break
cv2.destroyWindow("preview")
