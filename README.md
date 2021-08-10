# FER
Facial Emotion Detection using fer


# Facial Emotion Detection

An R&amp;D project to test the feasibility of detecting emotions out of face images, as well as promoting the AI application within ANB. This application is meant to detect the faces of ANB customers during their sessions while talking to ANB branch customer service employees, and recognize their emotions based on their facial expressions during these session.

# Library observations

Used library: [https://github.com/justinshenk/fer](https://github.com/justinshenk/fer)

Highlights:

- Face detection using two methods: OpenCV&#39;s Haar Cascade classifier and MTCNN network
- Emotion recognition using tensorflow neural network
- MIT License

# Model observations:

- The model detects its best where the accuracy is above 70% based on our humble observations
- The facial detection method using MTCNN network is an absolute success for its flawless results
- The emotion recognition part requires a good camera or images with good resolution.

# Examples:

Ps. Detections with green color are these above 70% accuracy:

![](RackMultipart20210810-4-n0v4kc_html_bd0b45c9021f622f.jpg)

![](RackMultipart20210810-4-n0v4kc_html_45723fb109639caf.jpg)

![](RackMultipart20210810-4-n0v4kc_html_77856cc800adb88.jpg)

![](RackMultipart20210810-4-n0v4kc_html_a2d784756334ce96.png)

![](RackMultipart20210810-4-n0v4kc_html_213d0ea36e239431.jpg)

![](RackMultipart20210810-4-n0v4kc_html_bca0f0dcdd42ed69.jpg)

![](RackMultipart20210810-4-n0v4kc_html_e12fbee38594a0f.jpg)