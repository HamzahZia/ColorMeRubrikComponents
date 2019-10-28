import cv2
import dlib
import os.path
import sys

def detectfaces(path):
    filename, file_extension = os.path.splitext(path)
    filename = filename.split('/')[-1]
    # print(filename)
    # print(file_extension)
    
    dnnFaceDetector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
    
    img = cv2.imread(path, 1)
    height, width, channels = img.shape
    img_area = height * width

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = dnnFaceDetector(gray, 1)

    faces_area = 0
    count = 0

    for (i, rect) in enumerate(rects):
        faces_area += rect.rect.area()
        count += 1
        x1 = rect.rect.left()
        y1 = rect.rect.top()
        x2 = rect.rect.right()
        y2 = rect.rect.bottom()
        # Rectangle around the face
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    print(f"There is/are {count} face(s) in the frame.")
    print(f"{int((faces_area / img_area)*100)}% of frame is made up of faces.")

    cv2.imwrite("faces/" + filename + file_extension, img)

def main(path):
    detectfaces(path)
    print("Done.")
    exit()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python facedetection.py <path-to-file>")
    else:
        main(sys.argv[1])