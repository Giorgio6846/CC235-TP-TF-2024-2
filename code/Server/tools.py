import cv2

def saveFaces(facePath, depthPath, face1Path, depth1Path):
    print(facePath, depthPath, face1Path, depth1Path)

    faceIMG = cv2.imread(facePath)[..., ::-1]
    depthIMG = cv2.imread(depthPath)

    images = detect_faces(faceIMG, depthIMG)

    if images != None:
        faceRGB = images[0][..., ::-1]

        cv2.imwrite(f"./data/tmp/imgProc/F1.jpg", faceRGB)
        cv2.imwrite(f"./data/tmp/imgProc/D1.png", images[1])

    faceIMG = cv2.imread(face1Path)[..., ::-1]
    depthIMG = cv2.imread(depth1Path)

    images = detect_faces(faceIMG, depthIMG)

    if images != None:
        faceRGB = images[0][..., ::-1]

        cv2.imwrite(f"./data/tmp/imgProc/F2.jpg", faceRGB)
        cv2.imwrite(f"./data/tmp/imgProc/D2.png", images[1])


def detect_faces(image, depth):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "./haarcascade_frontalface_alt2.xml"
    )
    faces = face_cascade.detectMultiScale(
        image, scaleFactor=1.19, minNeighbors=15, minSize=(80, 80)
    )

    print(len(faces))
    print(faces)

    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]

    face_image = image[y : y + h, x : x + w]
    depth_image = depth[y : y + h, x : x + w]

    face_image_resized = cv2.resize(face_image, (480, 480))
    depth_image_resized = cv2.resize(depth_image, (480, 480))

    return face_image_resized, depth_image_resized
