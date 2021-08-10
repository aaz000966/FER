
def detect_image(img):
    import cv2
    from fer import FER
    detector = FER(mtcnn=True)  # or with mtcnn=False for Haar Cascade Classifier
    image = cv2.imread(img)
    results = detector.detect_emotions(image)
    white = (255, 255, 255)
    green = (0, 153, 0)
    orange = (0, 155, 255)

    # Result is an array with all the bounding boxes detected. We know that for 'justin.jpg' there is only one.
    for result in results:
        bounding_box = result["box"]
        x1, y1, x2, y2 = result["box"]

        emotions = result["emotions"]
        print(emotions)
        top_emotion = max(emotions, key=emotions.get)
        top_emotion_score = emotions.get(top_emotion)
        label = str(top_emotion) + ' : ' + str(top_emotion_score)
        color = green if float(top_emotion_score) >= 0.70 else orange
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)
        cv2.rectangle(
            image,
            (x1, y1 - 20),
            (x1 + w, y1),
            color, -1
        )
        cv2.rectangle(
            image,
            (bounding_box[0], bounding_box[1]),
            (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
            color,
            2,
        )
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            .5,
            (255, 255, 255),
            1,
        )

    cv2.imshow('lil', image)
    cv2.waitKey(0)

    image_split = img.split('/')
    new_name = image_split[-1].split('.')[0] + 'new.' + image_split[-1].split('.')[1]
    image_split[-1] = new_name
    new_name = ''
    for folder in image_split:
        new_name += folder + '/'
    new_name = new_name[:-1]
    cv2.imwrite(new_name, image)


def detect_webcam():
    import cv2
    from fer import FER
    detector = FER(mtcnn=True)  # or with mtcnn=False for Haar Cascade Classifier
    vc = cv2.VideoCapture(0)

    if vc.isOpened():  # try to get the first frame
        r_val, frame = vc.read()
    else:
        r_val = False

    while r_val:
        frame = cv2.flip(frame, 1)

        results = detector.detect_emotions(frame)

        # Result is an array with all the bounding boxes detected. We know that for 'justin.jpg' there is only one.
        for result in results:
            bounding_box = result["box"]
            x1, x2, y1, y2 = bounding_box
            emotions = result["emotions"]

            top_emotion = str(max(emotions, key=emotions.get))
            top_emotion_score = str(emotions.get(top_emotion))

            cv2.rectangle(
                frame,
                (bounding_box[0], bounding_box[1]),
                (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                (0, 155, 255),
                2,
            )
            color = (12, 255, 20)
            if float(top_emotion_score) > 0.01:
                cv2.putText(
                    frame,
                    top_emotion,
                    (bounding_box[0], bounding_box[1] + bounding_box[3] + 25 + 0 * 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    top_emotion_score,
                    (bounding_box[0], bounding_box[1] + bounding_box[3] + 25 + 1 * 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            # for idx, (emotion, score) in enumerate(emotions.items()):
            #     color = (211, 211, 211) if score < 0.01 else (0, 255, 0)
            #     emotion_score = "{}: {}".format(
            #         emotion, "{:.2f}".format(score) if score > 0.01 else ""
            #     )
            #     cv2.putText(
            #         frame,
            #         emotion_score,
            #         (bounding_box[0], bounding_box[1] + bounding_box[3] + 30 + idx * 15),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         .5,
            #         color,
            #         1,
            #         cv2.LINE_AA,
            #     )

        cv2.imshow("preview", frame)
        r_val, frame = vc.read()
        key = cv2.waitKey(1)
        if key == 27 or cv2.getWindowProperty('preview', cv2.WND_PROP_VISIBLE) < 1:  # exit on ESC
            break
    cv2.destroyWindow("preview")


def main():
    detect_image('./images/12.jpg')
    # detect_webcam()


if __name__ == '__main__':
    main()
