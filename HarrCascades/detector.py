import cv2

# Define some of those HOT cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') CAUSE EYEGLASS ARE DOPE
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

# Hack into your webcam
cap = cv2.VideoCapture(0)

while True:
    # Take the img from the webcam
    ret, img = cap.read()

    # Convert the image to gray scale to be read easier and quicker
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect all the faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        # Draw a blue rectangle of line width 2 around the found x and y values of the face
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

        # Take the area that the face was in to search for eyes
        roi_gray = gray[y:y+h, x:x+w] # grayscale
        roi_color = img[y:y+h, x:x+w] # in color

        # Find all the eyes in the image
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            # At every place there is a eye draw a green rectangle around it of line width 2
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), ( 0, 255, 0), 2)

    # Display the image
    cv2.imshow('img', img)

    # Break if the user presses escape
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Stop recording and destroy all windows
cap.release()
cv2.destroyAllWindows()
