# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:00:01 2024

@author: f.piacentini
"""

import socket
from os import listdir
from os.path import isfile, join

def client_program():
    host = "10.60.10.26"
    port = 50000  # socket server port number
    
    client_socket = socket.socket()  # instantiate
    client_socket.connect((host, port))  # connect to the server
    
    message = input(" -> ")  # take input

    while message.lower().strip() != 'bye':
        # send messages
        keylabel = "00acaa72-9cd0-409b-b89f-78b043d31526"
        keyvalue, _ = otp_key_gen(keylabel)
        
        msg_cipher = encrypt_to_bytes(message.encode(), keyvalue)
        
        client_socket.send(msg_cipher)  # send message
        client_socket.send(keylabel.encode())  # send message
        
        # receive message
        msg_recv_cipher = client_socket.recv(1024)
        keylabel = client_socket.recv(1024).decode()  # receive response
        keyvalue, _ = otp_key_gen(keylabel)
        
        msg_recv_plain = encrypt_to_bytes(msg_recv_cipher, keyvalue)
        msg_recv = msg_recv_plain.decode()

        print('Received from server: ' + msg_recv)  # show in terminal
        
        # again take input restart loop if not "bye"
        message = input(" -> ")

    client_socket.close()  # close the connection


def otp_key_gen(keylabel: str):
    path = "C:\\Users\\f.piacentini\\Desktop\\keys\\"
    ext = ".key"
    total_path = (path + keylabel + ext)
    file = open(total_path, 'rb')
    keyvalue = file.read()
    file.close()
    
    return keyvalue, keylabel


def encrypt_to_bytes(plain_text, key):
    byteorder = "big"
    plain_text_int = int.from_bytes(plain_text, byteorder)
    key = int.from_bytes(key, byteorder)
    cipher_text = key ^ plain_text_int
    
    return cipher_text.to_bytes(1024, byteorder)

if __name__ == '__main__':
    client_program()