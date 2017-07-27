import socket
import cv2
import sys
import numpy
from model_server import *

#receive
address_r = ('10.85.125.105', 8002)
sock_r = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_r.bind(address_r)
sock_r.listen(True)
conn_r, addr_r = sock_r.accept()

#send
address_s = ('10.236.30.144', 8003)
sock_s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
sock_s.connect(address_s)
encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf
model_type = sys.argv[1]
init_model(model_type)
while True:
    length = recvall(conn_r,16)
    stringData = recvall(conn_r, int(length))
    data = numpy.fromstring(stringData, dtype='uint8')
    frame=cv2.imdecode(data,1)

    print 'Receive pic %s' %(str(frame.shape))
    frame = predict_image(frame, model_type)
    print 'Send pic %s' %(str(frame.shape))

    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    data = numpy.array(imgencode)
    stringData = data.tostring()
    sock_s.send( str(len(stringData)).ljust(16));
    sock_s.send( stringData );

sock_r.close()
sock_s.close()
