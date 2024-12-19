import socket

def client_program():
    host = "10.60.10.26"
    port = 50000  # socket server port number
    
    key = 0b1010100010101010010010010110001011010001
    pwd_plain = "pera"
    pwd_cipher = encrypt_to_bytes(pwd_plain, key)
    
    client_socket = socket.socket()  # instantiate
    client_socket.connect((host, port))  # connect to the server
    
    
    client_socket.send(pwd_cipher)  # send message
    auth_msg = client_socket.recv(1024).decode()  # receive response
    print(auth_msg)
    
    message = input(" -> ")  # take input

    while message.lower().strip() != 'bye':
        client_socket.send(message.encode())  # send message
        data = client_socket.recv(1024).decode()  # receive response

        print('Received from server: ' + data)  # show in terminal

        message = input(" -> ")  # again take input

    client_socket.close()  # close the connection



def encrypt_to_bytes(plain_text: str, key: bin):
    byteorder = "big"
    plain_text_int = int.from_bytes(plain_text.encode(), byteorder)
    cipher_text = key ^ plain_text_int
    
    return cipher_text.to_bytes(1024, byteorder)

if __name__ == '__main__':
    client_program()