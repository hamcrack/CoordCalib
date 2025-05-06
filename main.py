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

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detectorParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detectorParams)
id_positions = [[[0.092, 0.052, 0.01], [0.092, 0.072, 0.01], [0.072, 0.072, 0.01], [0.072, 0.052, 0.01]],  # 0 (Base)
                [[-0.162, 0.06, 0.01], [-0.162, 0.08, 0.01], [-0.182, 0.08, 0.01], [-0.182, 0.06, 0.01]],  # 1 (Base)
                [[-0.16, -0.08, 0.01], [-0.16, -0.06, 0.01], [-0.18, -0.06, 0.01], [-0.18, -0.08, 0.01]],  # 2 (Base)
                [[0.092, -0.082, 0.01], [0.092, -0.062, 0.01], [0.072, -0.062, 0.01], [0.072, -0.082, 0.01]]]  # 3 (Base)
        
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
mouse_pos = [0, 0]
world_pos = [0, 0, 0]
plane = []

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

# determines the transform of world coordinate system to camera coorindate system
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

def define_plane(points):
    if points.shape[0] < 3:
        return None

    p1 = points[0]
    p2 = points[1]
    p3 = points[2]

    v1 = p2 - p1
    v2 = p3 - p1

    normal = np.cross(v1, v2)
    if np.linalg.norm(normal) < 1e-6:  # Points are collinear
        return None

    A, B, C = normal
    D = -np.dot(normal, p1)

    return A, B, C, D

def pixel_to_world_ray(pixel_coords, camera_matrix, rotation_matrix, translation_vector):
    u, v = pixel_coords
    homogenous_pixel = np.array([[u], [v], [1]], dtype=np.float32)
    intrinsic_inv = np.linalg.inv(camera_matrix)
    ray_camera = intrinsic_inv @ homogenous_pixel
    ray_world_direction = rotation_matrix.T @ ray_camera
    camera_world_origin = -rotation_matrix.T @ translation_vector
    return camera_world_origin.flatten(), ray_world_direction.flatten()

def pixel_to_world_on_plane(pixel_coords, camera_matrix, rotation_matrix, translation_vector, plane_equation):
    origin, direction = pixel_to_world_ray(pixel_coords, camera_matrix, rotation_matrix, translation_vector)
    A, B, C, D = plane_equation

    denominator = np.dot(np.array([A, B, C]), direction)
    numerator = - (np.dot(np.array([A, B, C]), origin) + D)

    if np.abs(denominator) < 1e-6:  # Ray is parallel to the plane
        if np.abs(numerator) < 1e-6:
            print("Ray lies on the plane.")
        else:
            print("Ray does not intersect the plane.")
            return None
    else:
        lambda_val = numerator / denominator
        world_point = origin + lambda_val * direction
        return world_point

def set_zone_points(event, x, y, flags, param):
    global corner_no, img_points, world_coords, rotationMatrix, translationVector, plane, world_pos, mouse_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Clicked at: ", x, y)
        if corner_no < 4:
            img_points[corner_no][0] = x
            img_points[corner_no][1] = y
            world_coords[corner_no][0] = -((x - image_size[0] / 2) / 1000)
            world_coords[corner_no][1] = 0.01
            world_coords[corner_no][2] = (image_size[1] / math.sin(math.radians(camera_angle)) - (image_size[1] - y) / math.sin(math.radians(camera_angle))) / 1000
            print("World Coordinates: ", world_coords[corner_no])

        corner_no += 1

        if corner_no == 4:
            # corner_no = 0
            print("Image Points: ", img_points)
            print("World Coordinates: ", world_coords)
            print("")
            rotationMatrix, translationVector = get_projected_coords()
            plane = define_plane(world_coords[:3])

        if corner_no > 4:
            mouse_pos[0] = x
            mouse_pos[1] = y
            fake_world_x = -((x - image_size[0] / 2) / 1000)
            fake_world_y = 0.01
            fake_world_z = (image_size[1] / math.sin(math.radians(camera_angle)) - (image_size[1] - y) / math.sin(math.radians(camera_angle))) / 1000

            print("Fake World Coordinates: ", fake_world_x, fake_world_y, fake_world_z)
            A, B, C, D = plane
            world_point_on_plane = pixel_to_world_on_plane(mouse_pos, cameraMatrix, rotationMatrix, translationVector, plane)
            world_pos = [world_point_on_plane[0], world_point_on_plane[1], world_point_on_plane[2]]
            print("World Coordinates: ", world_point_on_plane)


if live:
    image_size = (640, 480)

    cameraMatrix = np.array([[642.70404546, 0, 315.2411113], [0, 642.70404546, 236.77473824], [0, 0, 1]], dtype=np.float32)
    distCoeffs = np.array([[13.2006295, -63.520978, -0.00165712, 0.00118068, -34.7385185, 13.111275, -63.2187353, -34.4029467]], dtype=np.float32)

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

        corners, ids, rejectedImgPoints = detector.detectMarkers(frame)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    else:
        frame = cv2.imread("1_empty_ramp.png")

        for i in range(4):
            x, y = img_points[i]
            x, y = int(x), int(y)
            if x != 0 and y != 0:
                if img_points[i-1][0] != 0 and img_points[i-1][1] != 0:
                    cv2.line(frame, (int(img_points[i-1][0]), int(img_points[i-1][1])), (x, y), (255, 0, 0), 2)
                cv2.putText(frame, str(i) + " : " + str(img_points[i]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, str(i) + " : " + str(world_coords[i]), (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (x,y), 5, (255,0,0), -1)
        
        if mouse_pos[0] != 0 and mouse_pos[1] != 0:
            cv2.putText(frame, str(mouse_pos), (mouse_pos[0], mouse_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, str(np.round(world_pos[0], 2)) + ", " + str(np.round(world_pos[1], 2)) + ", " + str(np.round(world_pos[2], 2)), (mouse_pos[0], mouse_pos[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (mouse_pos[0], mouse_pos[1]), 5, (0,0,255), -1)

        # Define unit xyz axes. These are then projected to camera view using the rotation matrix and translation vector.
        board_center_z = (image_size[1]/2 / math.sin(math.radians(camera_angle))) / 1000

        unitv_points = np.array([[0,0,board_center_z], [0.02,0,board_center_z], [0,0.02,board_center_z], [0,0,board_center_z+0.02]], dtype = 'float32').reshape((4,1,3))
        if not np.array_equal(rotationMatrix, np.identity(3)):
            points, jac = cv2.projectPoints(unitv_points, rotationMatrix, translationVector, cameraMatrix, distCoeffs)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]
            points = points.reshape((4, 2))
            origin = (int(points[0][0]), int(points[0][1]))
            for p, c in zip(points[1:], colors[:3]):
                p = (int(p[0]), int(p[1]))
                if origin[0] > 5 * frame.shape[1] or origin[1] > 5 * frame.shape[1]:
                    break
                if p[0] > 5 * frame.shape[1] or p[1] > 5 * frame.shape[1]:
                    break
                cv2.line(frame, origin, p, c, 5)

    if save_vid:
        out.write(frame)
    cv2.imshow('image', frame)
    if cv2.waitKey(1) == ord('q'):
        break

if save_vid:
    out.release()
cv2.destroyAllWindows()