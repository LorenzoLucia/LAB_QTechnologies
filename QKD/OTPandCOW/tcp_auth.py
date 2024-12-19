# Programma server 
import socket

AUTH_FAILED_MSG = "AUTH_FAILED"
AUTH_SUCCESS_MSG = "AUTH_SUCCESS"
CLOSE_CONNECTION_MSG = "CLOSE"

auth_key = 0b1010100010101010010010010110001011010001
auth_word = "pera"
auth_word_int = int.from_bytes(auth_word.encode("utf-8"), "big")
HOST = '10.60.10.26'       # Nome o IP che rappresenta il server locale
PORT = 50000              # Porta non privilegiata  tra 49152 e 65535
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
    
def client_auth():
    
    print("Waiting on: ", (HOST, PORT))
    s.listen(1)
    conn, addr = s.accept()
    print ('Connected from: ', addr)

    while True:    
        auth_data = conn.recv(1024)
        
        auth_data_int = int.from_bytes(auth_data, "big")
        
        
        if auth_word_int != (auth_data_int ^ auth_key):
            
            print("Authentication failed")
            
            return conn, False
        else:
            print ("Authentication success")
            return conn, True
    
    
def recv_from_client(connection):
    while True:
        data = connection.recv(1024)
        
    
        if not data :
            break
        else:
            data_decoded = data.decode("utf-8")
            if data_decoded == CLOSE_CONNECTION_MSG:
                connection.close()
            else:
                print ("Client said: "+ data_decoded)
            
        



conn, is_auth = client_auth()

if is_auth:
    conn.send(AUTH_SUCCESS_MSG.encode())
    recv_from_client(conn)
else:
    conn.send(AUTH_FAILED_MSG.encode())
    conn.close()
    