import socket
import cv2
import pickle
import struct

live = False
save_vid = False

data = b""
payload_size = struct.calcsize("Q")
image_size = (1152, 534)

corner_no = 0
corners = [[0, 0], [0, 0], [0, 0], [0, 0]]

def draw_circle(event, x, y, flags, param):
    global corner_no, corners
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Clicked at: ", x, y)
        corners[corner_no] = [x, y]
        corner_no += 1
        if corner_no == 4:
            print("Corners: ", corners)
            corner_no = 0

if live:
    image_size = (640, 480)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('172.25.0.1', 9999))
    print("Connected to server")

if save_vid:
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, image_size)

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

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
        x, y = corners[i]
        if x != 0 and y != 0:
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.circle(frame, (x,y), 5, (255,0,0), -1)
    
    if save_vid:
        out.write(frame)
    cv2.imshow('image', frame)
    if cv2.waitKey(1) == ord('q'):
        break

if save_vid:
    out.release()
cv2.destroyAllWindows()