import os
import cv2
import matplotlib.pyplot as plt

def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    # raise NotImplementedError("To be implemented")
    detectData = list(open(dataPath, "r"))
    line = 0
    while line < len(detectData):
        image_name, people_num = map(str, detectData[line].split())
        line += 1
        people = []
        for i in range( int(people_num) ):
            people.append( tuple( map( int, detectData[line].split() ) ) )
            line += 1
        image = cv2.imread("data/detect/" + image_name)
        image_cmp = cv2.imread("data/detect/" + image_name, cv2.IMREAD_GRAYSCALE)
        for face in people:
            face_image = cv2.resize(image_cmp[ face[1]:face[1]+face[3], face[0]:face[0]+face[2] ], (19, 19), interpolation=cv2.INTER_LINEAR )
            if clf.classify(face_image) == 1:
                cv2.rectangle(image, ( face[0], face[1] ), ( face[0] + face[2], face[1] + face[3] ), (0, 255, 0), thickness=2 )
            else:
                cv2.rectangle(image, ( face[0], face[1] ), ( face[0] + face[2], face[1] + face[3] ), (0, 0, 255), thickness=2 )
        cv2.imwrite("part6_"+image_name, image)
    # End your code (Part 4)
