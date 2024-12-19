# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:07:49 2024

@author: Studenti
"""

# Programma server 
import socket

AUTH_FAILED_MSG = "AUTH_FAILED"
AUTH_SUCCESS_MSG = "AUTH_SUCCESS"
CLOSE_CONNECTION_MSG = "CLOSE"

root_path = "C:\\Users\\Studenti\\Desktop\\keys\\"
key_file_extension = ".key"
key_file_label = "00acaa72-9cd0-409b-b89f-78b043d31526"



auth_key = 0b1010100010101010010010010110001011010001
auth_word = "pera"
auth_word_int = int.from_bytes(auth_word.encode("utf-8"), "big")
HOST = '10.60.10.26'       # Nome o IP che rappresenta il server locale
PORT = 50000              # Porta non privilegiata  tra 49152 e 65535
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
    
def client_connect():
    
    print("Waiting on: ", (HOST, PORT))
    s.listen(1)
    conn, addr = s.accept()
    print ('Connected from: ', addr)
    
    return conn
    



def get_key_as_int(key_label):
    with open(root_path + key_label + key_file_extension, "rb") as key_file:
        
        
        return int.from_bytes(key_file.read(), "big")
            
            

def send_to_client(connection):
    msg_to_send = input("Enter message to send:")
    
    key_as_int = get_key_as_int(key_file_label)
    
    ecnrypted_msg_int = key_as_int ^ int.from_bytes(msg_to_send.encode(), "big")
    
    connection.send(ecnrypted_msg_int.to_bytes(1024, "big"))
    
    connection.send(key_file_label.encode())
        

    
    
def recv_from_client(connection):
    while True:
        encrypted_data = connection.recv(1024)
        
        key_data = connection.recv(1024)
        
    
        if not encrypted_data or not key_data:
            print("One of the two data object is None")
            break
        else:
            
            key = get_key_as_int(key_data.decode())
            
            message = key ^ int.from_bytes(encrypted_data, "big")
            
            if message== CLOSE_CONNECTION_MSG:
                connection.close()
                return False
            else:
                print("Client said: "+ message.to_bytes(1024, "big").decode())
                return True
        
        
connection = client_connect()

while True:
    is_still_connected = recv_from_client(connection)
    
    if is_still_connected:
        send_to_client(connection)
    else:
        break

