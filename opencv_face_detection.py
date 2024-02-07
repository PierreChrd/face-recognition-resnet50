import cv2

def detect_and_save_face(image_path, output_path):
    # Load the pre-trained Haar Cascade model for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Loop over the faces and save the first one found
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        cv2.imwrite(output_path, face)
        break 
