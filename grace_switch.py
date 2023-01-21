#coding:utf-8

from socket import *
from time import ctime
import time

#This class controls the network switch attached to the power button of grace.
#Note that to use this class, we must first configure the network switch to 
#look for the right pc and right gateway, following the instruction of the manufacturer of that switch
class Grace_Switch:
    HOST = '192.168.99.242'
    PORT = 6000 
    BUFSIZ = 1024 
    ADDR = (HOST, PORT)

    def __init__(self):
        #Create tcp server socket with the network switch
        self.tcp_ser_soc = socket(AF_INET, SOCK_STREAM)
        self.tcp_ser_soc.bind(self.ADDR)
        self.tcp_ser_soc.listen(5)
        #Wait for the replay switch connection request
        self.tcp_cli_soc, self.cli_addr = self.tcp_ser_soc.accept()  
        #Log info
        print("Self server ip: %s" % str(self.HOST))
        print("Switch client ip: %s" % str(self.cli_addr))
    
    def toggle_replay_switch_state(self,state):
        #state: True for connected, False for disconnected.
        state_string = ""
        if(state):
            state_string = "1\r\n"
        else:
            state_string = "0\r\n"
        msg_to_switch_cli = "AT+STACH1="+state_string
        self.tcp_cli_soc.send(msg_to_switch_cli.encode())
        recv_data = self.tcp_cli_soc.recv(self.BUFSIZ)
        print("Switch response %s" % recv_data)
        time.sleep(5)


if __name__ == '__main__':
    switch_test = Grace_Switch()
    switch_test.toggle_replay_switch_state(False)