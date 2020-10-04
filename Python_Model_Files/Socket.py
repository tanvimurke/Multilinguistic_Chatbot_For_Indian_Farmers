# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:12:29 2020

@author: Admin
"""

# first of all import the socket library 
import socket

from Functions import clean_text, get_weather_information, knn_predictor, get_answer
     
def get_reply(question,location):
    s=question
    city=location
    #print(city)
    s=clean_text(s)
    s_original = s
    
    predict_type = knn_predictor(s)
    
    if (predict_type[0]=='weather'):
        answer = get_weather_information(city)
        answer =  "CURRENT WEATHER INFORMATION: " + answer
    else:
        answer = get_answer(s_original)
        answer ="Answer: " + answer
    return answer

# next create a socket object 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)		 
print ("Socket successfully created")

# reserve a port on your computer in our 
# case it is 12345 but it can be anything 
PORT = 7000				
HOST = "192.168.225.29"

# Next bind to the port 
# we have not typed any ip in the ip field 
# instead we have inputted an empty string 
# this makes the server listen to requests 
# coming from other computers on the network 
s.bind((HOST,PORT))		 
print ("socket binded to %s" %(PORT) )

# put the socket into listening mode 
s.listen(5)	 
print ("socket is listening")			

# a forever loop until we interrupt it or 
# an error occurs 
while True: 
    i=0
    i=i+1
    # Establish connection with client. 
    connection, addr = s.accept()	 
    print ('Got connection from', addr) 
    data = str(connection.recv(1024),'utf-8') #how many bytes of data will the server receive
    #data="What is the givernment scheme for me+Pune"
    question, loc = data.split('+')
    print ("Received: ", str(data))
    if(question == "null"):
        question = "Sorry"
    output = get_reply(question,loc)
    reply = bytes(" {}".format(output), 'utf-8') #server's reply to the client
    connection.send(reply)
    connection.close() 
    if(i==10):
        break
"""
# send a thank you message to the client. 
c.send('Thank you for connecting') 

# Close the connection with the client 
c.close() 
cd "B. E\B. E. Project\Codes"
"""
