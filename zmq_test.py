import zmq
import json
import time
def zmq_action(socket):
    # print(time.time())
    # context = zmq.Context()
    # socket = context.socket(zmq.REP)
    # socket.bind("tcp://*:5555")  # 绑定端口5555
    # print(time.time())

    print("Server is running...")
    # 等待客户端消息
    message = socket.recv_string()
    print(message)
    float_array = [0.05, 0.01, 0.01, 0.99993896484375, -0.0021783041302114725, -0.0061051067896187305, -0.008672064170241356]
    # 将浮点数数组编码为JSON字符串
    message = json.dumps(float_array)
    socket.send_string(message)

def zmq_obs(socket):

    # 定义一个浮点数数组
    # float_array = [1.1, 2.2, 3.3]
    # 将浮点数数组编码为JSON字符串
    message = "get obs"
    socket.send_string(message)
    print(time.time())

    # 等待服务器响应
    reply = socket.recv_string()
    print(f"Received obs: {reply}")
    print(time.time())


if __name__ == "__main__":
    context = zmq.Context()
    rep = context.socket(zmq.REP)
    rep.bind("tcp://*:5555")  # 绑定端口5555

    context = zmq.Context()
    print("Connecting to server...")
    req = context.socket(zmq.REQ)
    req.connect("tcp://192.168.1.101:5555")  # 连接到服务器
    while True:
      zmq_action(rep)
      time.sleep(0.1)
      zmq_obs(req)

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