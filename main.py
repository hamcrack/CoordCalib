import cv2
import math
import socket
import pickle
import struct
import numpy as np

live = False
save_vid = False

rotationMatrix = np.identity(3, dtype=np.float32)
translationVector = np.array([[0], [0], [0]], dtype=np.float32)

data = b""
payload_size = struct.calcsize("Q")
image_size = (1152, 534)

focal_length = 800
cameraMatrix = np.array([[focal_length, 0, image_size[0] / 2],
                         [0, focal_length, image_size[1] / 2],
                         [0, 0, 1]], dtype=np.float32)
distCoeffs = np.zeros((5, 1), dtype=np.float32)

corner_no = 0
img_points = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32)
camera_angle = 20
world_coords = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32)

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

# determines the transform of the orientation of coordinate system with respect to camera coorindate system
def get_projected_coords():
    ret, rvec, tvec = cv2.solvePnP(world_coords, img_points, cameraMatrix, distCoeffs)
    rmat = cv2.Rodrigues(rvec)[0]
    rotations_euler = rotationMatrixToEulerAngles(rmat)

    print("Rotation vec:", math.degrees(rvec[0]), math.degrees(rvec[1]), math.degrees(rvec[2]))
    print("Rotation mat:", rmat)
    print("Rotations:", math.degrees(rotations_euler[0]), math.degrees(rotations_euler[1]), math.degrees(rotations_euler[2]))
    print("Translation:", tvec[0], tvec[1], tvec[2])
    print("")

    return rmat, tvec

def set_zone_points(event, x, y, flags, param):
    global corner_no, img_points, world_coords, rotationMatrix, translationVector
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Clicked at: ", x, y)
        img_points[corner_no][0] = x
        img_points[corner_no][1] = y
        world_coords[corner_no][0] = ((x - image_size[0] / 2) / 1000)
        world_coords[corner_no][1] = ((image_size[1] - y) / math.sin(math.radians(camera_angle)) / 1000)
        world_coords[corner_no][2] = 0
        print("World Coordinates: ", world_coords[corner_no])
        corner_no += 1
        if corner_no == 4:
            corner_no = 0
            print("Image Points: ", img_points)
            print("World Coordinates: ", world_coords)
            print("")
            rotationMatrix, translationVector = get_projected_coords()

if live:
    image_size = (640, 480)

    cameraMatrix = np.array([[888.37495946, 0, 664.61495248], [0, 888.37495946, 353.54694702], [0, 0, 1]], dtype=np.float32)
    distCoeffs = np.array([[-0.10650025, 1.0173892, 0.00192866, -0.0013329, 0.1139497, 0.29545028, 0.91682975, 0.51278569]], dtype=np.float32)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('172.25.0.1', 9999))
    print("Connected to server")

if save_vid:
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, image_size)

cv2.namedWindow('image')
cv2.setMouseCallback('image', set_zone_points)

while True:
    if live:
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
    else:
        frame = cv2.imread("1_empty_ramp.png")

    for i in range(4):
        x, y = img_points[i]
        x, y = int(x), int(y)
        if x != 0 and y != 0:
            if img_points[i-1][0] != 0 and img_points[i-1][1] != 0:
                cv2.line(frame, (int(img_points[i-1][0]), int(img_points[i-1][1])), (x, y), (255, 0, 0), 2)
            cv2.putText(frame, str(i) + " : " + str(img_points[i]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.circle(frame, (x,y), 5, (255,0,0), -1)

    # Define unit xyz axes. These are then projected to camera view using the rotation matrix and translation vector.
    unitv_points = np.array([[0,0,0], [0.02,0,0], [0,0.02,0], [0,0,-0.02]], dtype = 'float32').reshape((4,1,3))
    if not np.array_equal(rotationMatrix, np.identity(3)):
        points, jac = cv2.projectPoints(unitv_points, rotationMatrix, translationVector, cameraMatrix, distCoeffs)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]
        axis_points = points.reshape((4, 2))
        center_x = int((img_points[0][0] + img_points[2][0] + img_points[3][0] + img_points[1][0]) / 4)
        center_y = int((img_points[0][1] + img_points[2][1] + img_points[3][1] + img_points[1][1]) / 4)
        origin = (center_x, center_y)
        for p, c in zip(axis_points[1:], colors[:3]):
            p = (int(p[0]), int(p[1]))
            if 0 <= p[0] < frame.shape[1] and 0 <= p[1] < frame.shape[1]:
                # Sometimes qr detector will make a mistake and projected point will overflow integer value. We skip these cases.
                cv2.line(frame, origin, p, c, 5)

    if save_vid:
        out.write(frame)
    cv2.imshow('image', frame)
    if cv2.waitKey(1) == ord('q'):
        break

if save_vid:
    out.release()
cv2.destroyAllWindows()