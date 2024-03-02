import zmq
import json
import time
def zmq_action(socket, action):
    message = socket.recv_string()
    # 将浮点数数组编码为JSON字符串
    message = json.dumps(action)
    socket.send_string(message)

def zmq_obs(socket):
    message = "get obs"
    socket.send_string(message)

    # 等待服务器响应
    reply = socket.recv_string()
    obs = json.loads(reply)
    return obs

if __name__ == "__main__":
    context_rep = zmq.Context()
    rep = context_rep.socket(zmq.REP)
    rep.bind("tcp://*:5555")  # 绑定端口5555

    context_req = zmq.Context()
    print("Connecting to server...")
    req = context_req.socket(zmq.REQ)
    req.connect("tcp://192.168.1.101:5555")  # 连接到服务器

    action = [0.00351818, 0.00524153, -0.0130234, -0.000813822, 0.999987, 0.00422609, 0.00145553]

    while True:
      zmq_action(rep, action)
      aa = time.time()
      time.sleep(0.1)
      a = zmq_obs(req)

# import zmq
# import time
# import threading

# def zmq_subscriber_thread():
#   context = zmq.Context()
#   socket = context.socket(zmq.PULL)
#   socket.connect("tcp://192.168.1.101:5556")

#   while True:
#     message = socket.recv()
#     print("Received: %s" % message)

# def zmq_publisher_thread():
#   context = zmq.Context()
#   socket_pub = context.socket(zmq.PUSH)
#   socket_pub.bind("tcp://*:5556")


#   context = zmq.Context()
#   socket = context.socket(zmq.PULL)
#   socket.connect("tcp://192.168.1.101:5556")

#   while True:
#     socket_pub.send(b"hello")
#     time.sleep(1)


# if __name__ == '__main__':
#     zmq_sub_thread = threading.Thread(target=zmq_subscriber_thread)
#     zmq_sub_thread.start()

#     zmq_pub_thread = threading.Thread(target=zmq_publisher_thread)
#     zmq_pub_thread.start()