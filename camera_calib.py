import cv2
import math
import socket
import pickle
import struct
import numpy as np

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('172.25.0.1', 9999))
print("Connected to server")
data = b""
payload_size = struct.calcsize("Q")

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard((7, 5), 0.03, 0.02, aruco_dict)
detector = cv2.aruco.CharucoDetector(board)

img = board.generateImage((864, 1080))
cv2.imshow("aruco", img)

# Define number of images needed
good_img = 10
good_img_cnt = 0

print("POSE ESTIMATION STARTS:")
allCorners = []
allIds = []
allImagePoints = []
allObjectPoints = []
# SUB PIXEL CORNER DETECTION CRITERION
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

camera = cv2.VideoCapture(0)

while good_img_cnt < good_img:
    while len(data) < payload_size:
        data += client_socket.recv(4096)
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]
    while len(data) < msg_size:
        data += client_socket.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    frame = pickle.loads(frame_data)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners = detector.detectBoard(gray)

    if len(corners) > 0 and (cv2.waitKey(1) == ord('t')):
        # SUB PIXEL DETECTION
        # for corner in corners:
        #     print("corner: ", corner)
        #     corner = cv2.cornerSubPix(gray, corner, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria)[0]
        #     print("corner after: ", corner)
        ids = corners[1]
        corners = corners[0]

        print("corners: ", corners)
        print("len(corners): ", len(corners))
        print("ids: ", ids)
        print("len(ids): ", len(ids))

        currentObjectPoints, currentImagePoints = board.matchImagePoints(corners, ids)               
        allCorners.append(corners)
        allIds.append(ids)
        allImagePoints.append(currentImagePoints)
        allObjectPoints.append(currentObjectPoints) 

        good_img_cnt += 1
        print(good_img_cnt)
    else:
        cv2.imshow("Video", frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        camera.release()
        cv2.destroyAllWindows()

imsize = gray.shape

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
cameraMatrixInit = np.array([[1000.,    0., imsize[0]/2.],
                             [0., 1000., imsize[1]/2.],
                             [0.,    0.,           1.]])

distCoeffsInit = np.zeros((5,1))
flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
#flags = (cv2.CALIB_RATIONAL_MODEL)
(ret, camera_matrix, distortion_coefficients0,
 rotation_vectors, translation_vectors,
 stdDeviationsIntrinsics, stdDeviationsExtrinsics,
 perViewErrors) = cv2.calibrateCameraExtended(
                  objectPoints=allObjectPoints, 
                  imagePoints=allImagePoints,
                  imageSize=imsize,
                  cameraMatrix=cameraMatrixInit,
                  distCoeffs=distCoeffsInit,
                  flags=flags,
                  criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

print("")
print("Total error: {}".format(perViewErrors))

print("")
print("Image size:")
print(imsize)
print("")
print("Camera matrix:")
print(camera_matrix)
print("")
print("distortion:")
print(distortion_coefficients0)
print("")
print("rotation vectors:")
print(rotation_vectors)
print("")
print("translation vectors:")
print(translation_vectors)
print("")

with open('ChArUcoCalib1.txt', 'w') as f:
    f.write("Re-projection Error:\n")
    f.write(str(perViewErrors))
    f.write(" ")
    f.write("Image size :\n")
    f.write(str(imsize))
    f.write("\n")
    f.write("Camera matrix :\n")
    f.write(str(camera_matrix))
    f.write("\n")
    f.write("dist :\n")
    f.write(str(distortion_coefficients0))
    f.write("\n")
    f.write("rvecs :\n")
    f.write(str(rotation_vectors))
    f.write("\n")
    f.write("tvecs :\n")
    f.write(str(translation_vectors))
    f.write("\n")