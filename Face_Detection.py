
import face_recognition
import cv2
import numpy as np

patch = "/home/pi/Desktop/Face Recognation/dataset/"
# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)






# Load a sample picture and learn how to recognize it.
milad_1_image = face_recognition.load_image_file(patch + "milad_1.png")
milad_2_image = face_recognition.load_image_file(patch + "milad_2.png")
maryam_1_image = face_recognition.load_image_file(patch + "maryam_1.png")
maryam_2_image = face_recognition.load_image_file(patch + "maryam_2.png")
damoon_1_image = face_recognition.load_image_file(patch + "damoon_0.png")
damoon_2_image = face_recognition.load_image_file(patch + "damoon_1.png")
farhad_1_image = face_recognition.load_image_file(patch + "FarhadFarahi_0.png")
farhad_2_image = face_recognition.load_image_file(patch + "FarhadFarahi_1.png")


milad_1_face_encoding = face_recognition.face_encodings(milad_1_image)[0]
milad_2_face_encoding = face_recognition.face_encodings(milad_2_image)[0]
maryam_1_face_encoding = face_recognition.face_encodings(maryam_1_image)[0]
maryam_2_face_encoding = face_recognition.face_encodings(maryam_2_image)[0]
damoon_1_face_encoding = face_recognition.face_encodings(damoon_1_image)[0]
damoon_2_face_encoding = face_recognition.face_encodings(damoon_2_image)[0]
farhad_1_face_encoding = face_recognition.face_encodings(farhad_1_image)[0]
farhad_2_face_encoding = face_recognition.face_encodings(farhad_2_image)[0]
#Create arrays of known face encodings and their names
known_face_encodings = [
    milad_1_face_encoding,
    milad_2_face_encoding,
    maryam_1_face_encoding,
    maryam_2_face_encoding,
    damoon_1_face_encoding,
    damoon_2_face_encoding,
    farhad_1_face_encoding,
    farhad_2_face_encoding
]
known_face_names = [
    "MILAD",
    "MILAD",
    "MARYAM HASSANI",
    "MARYAM HASSANI",
    "DAMOON AHMADI",
    "DAMOON AHMADI",
    "Farhad Farahi",
    "Farhad Farahi"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
