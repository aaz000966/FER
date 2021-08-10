# Libraries
import time

import cv2
from fer import FER

# Globals
TRACKER_FILE = "emotionsTracker.csv"


def detect_webcam():
    # Object
    detector = FER(mtcnn=True)  # or with mtcnn=False for Haar Cascade Classifier

    orange = (0, 155, 255)
    white = (255, 255, 255)

    # Webcam Init
    vc = cv2.VideoCapture(0)
    if vc.isOpened():  # try to get the first frame
        r_val, frame = vc.read()
    else:
        r_val = False

    fw = open(TRACKER_FILE, "w")

    while r_val:
        frame = cv2.flip(frame, 1)

        results = detector.detect_emotions(frame)
        # for face in faces
        for result in results:
            # bounding_box = result["box"]
            x1, y1, x2, y2 = result["box"]
            emotions = result["emotions"]

            top_emotion = str(max(emotions, key=emotions.get))
            top_emotion_score = str(emotions.get(top_emotion))

            # Draw Square on face
            cv2.rectangle(
                frame,
                (x1, y1),
                (x1 + x2, y1 + y2),
                orange,
                2,
            )

            if float(top_emotion_score) >= 0.55:
                # Write captures to file
                fw.write(top_emotion + "," + '1' + '\n')

                # Draw Label and background Square
                label = top_emotion + ' : ' + top_emotion_score
                # print(label)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)
                cv2.rectangle(
                    frame,
                    (x1, y1 - 20),
                    (x1 + w, y1),
                    orange, -1
                )
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .5,
                    white,
                    1,
                )

        cv2.imshow("preview", frame)
        r_val, frame = vc.read()
        key = cv2.waitKey(1)
        # time.sleep(1)
        if key == 27:  # exit on ESC
            break
    fw.close()
    cv2.destroyWindow("preview")
    analyze_file(TRACKER_FILE)


def analyze_file(file):
    emotions_dictionary = {}

    # Append dictionary
    fr = open(file, "r")
    for line in fr:
        emotions_count_split = line.split(',')
        if emotions_count_split[0] in emotions_dictionary:
            emotions_dictionary[emotions_count_split[0]] = emotions_dictionary[emotions_count_split[0]] + 1
        else:
            emotions_dictionary[emotions_count_split[0]] = 1
    fr.close()

    # Write Grouped by values
    fw = open(file, 'w')
    for key in emotions_dictionary:
        fw.write(key + ',' + str(emotions_dictionary[key]) + '\n')
    fw.close()

    # Calculate Sum
    fr = open(file, 'r')
    total = 0
    for line in fr:
        if ',' in line:
            split_line = line.split(',')
            total += int(split_line[1])
    fr.close()

    # Append Sum
    fa = open(file, 'a')
    fa.write(str(total))
    fa.close()

    # Rectify File and add percentage
    lines_list = []
    fr = open(file, 'r')
    for line in fr:
        if ',' in line:
            split_line = line.split(',')
            percentage = (int(split_line[1]) / total) * 100
            lines_list.append(line.replace('\n', '') + ',' + str(percentage)[:4] + '%')
    fr.close()

    fw = open(file, 'w')
    for item in lines_list:
        fw.write(item + '\n')
    fw.write('Total Captures,' + str(total) + '\n')
    fw.close()

    # Print File content
    fr = open(TRACKER_FILE, 'r')
    for line in fr:
        print(line, end='')
    fr.close()


if __name__ == "__main__":
    detect_webcam()
