

print("Importing Library's...")

import face_recognition
import cv2
import numpy as np
import threading
import socket
import time
from os import listdir
from os.path import isfile, join


#Application is Running :
Running = True

#DataSet Patch :
patch = "D:/Projects/Face Detection/Face Detection/dataset"

#Creating a list of files in dataset folder :
onlyfiles = []
try:
    print("Listing files...")
    onlyfiles = [f for f in listdir(patch) if isfile(join(patch, f))]
except:
    print("Error listing files")



#Loading Dataset (Images and Labels) :
known_face_names = []
known_face_encodings = []
for filename in onlyfiles:
    try:
        # Load a sample picture and learn how to recognize it.
        image = face_recognition.load_image_file(patch + "/" + filename)
        encodedImage = face_recognition.face_encodings(image)[0]
        #and add it to known_face_encodings list :
        known_face_encodings.append(encodedImage)
        #Adding Label of the image to known_face_names list :
        Label = filename[0: filename.find('.')]
        known_face_names.append(Label)
        print(filename + "  -  OK  |  Label :" + Label)
    except:
        print(filename + "  -  Bad Image")

if len(known_face_names) == 0:
    print("No", end='')
else:
    print(len(known_face_names), end='')

print(" Valid Image's found")

if len(known_face_names) == 0:
    exit(0)

# init Socket and Opening Port:
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sck = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Checking Local IP if we connected to LAN:
localIP = socket.gethostname()

try:
    # Binding Socket on IP:
    serversocket.bind((localIP, 23))
    class SocketThread(threading.Thread):
        def run(self):
            global sck
            global Running
            print("{} Thread started!".format(self.getName()))
            # Waitnig for connection for 5 sec :
            serversocket.listen()
            serversocket.settimeout(5)
            print("Listening on " + localIP + " , port 23")
            #Checking while Application is running :
            while(Running): 
                try:
                    sck, addr = serversocket.accept()
                except:
                    continue
                sck.send(("Connected\r\n").encode())
                print('Socket Connected')
            print("{} Thread finished".format(self.getName()))

    #Starting Socket Thread :
    socketthread = SocketThread(name = "Socket")
    try:
        socketthread.start()
    except:
       print("Error: unable to start MQTT Thread")
except:
   print("Error: Socket error")


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

print("\n\nApplication Started")
while Running:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    #small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    small_frame = frame


    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        # # If a match was found in known_face_encodings, just use the first one.
        if matches.count(True) == 1:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        else:
            if matches.count(True) > 1: #Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

        face_names.append(name)
    
    FaceCount = len(face_names)


    print(FaceCount,end ='')
    print(" face detected :")
    for name in face_names:
        print("\t" + name)
    print("\n\n")

    if FaceCount ==0:
        time.sleep(3)
    else:
        time.sleep(2)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        Running = False

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
