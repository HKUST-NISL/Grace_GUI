import logging
import argparse
import sys
import threading
import time
import numpy
from signal import signal
from signal import SIGINT
import rospy
import sensor_msgs.msg
import std_msgs.msg
import hr_msgs.msg
import rosbag
import geometry_msgs.msg
import dynamic_reconfigure.client
import cv2
from cv_bridge import CvBridge
from copy import deepcopy
import os
from tkinter import *
import PIL.Image, PIL.ImageTk

class Grace_GUI:
    node_name = "grace_gui"
    cv_bridge = CvBridge()
    topic_queue_size = 100
    
    stop_topic = "/grace_proj/stop"
    toggle_attention_topic = "/grace_proj/toggle_attention"
    toggle_aversion_topic = "/grace_proj/toggle_aversion"
    aversion_text_topic = "/grace_proj/aversion_text"
    toggle_nodding_topic = "/grace_proj/toggle_nodding"
    nodding_text_topic = "/grace_proj/nodding_text"
    attention_target_topic = "/grace_proj/attention_target_idx"
    attention_target_img_topic = "/grace_proj/attention_target_img"
    annotated_tracking_stream_topic = "/grace_proj/annotated_tracking_stream"


    def __init__(self):
        rospy.init_node(self.node_name)
        
        self.stop_pub = rospy.Publisher(self.stop_topic, std_msgs.msg.Bool, queue_size=self.topic_queue_size)
        self.toggle_attention_pub = rospy.Publisher(self.toggle_attention_topic, std_msgs.msg.Bool, queue_size=self.topic_queue_size)
        self.toggle_aversion_pub = rospy.Publisher(self.toggle_aversion_topic, std_msgs.msg.Bool, queue_size=self.topic_queue_size)
        self.toggle_nodding_pub = rospy.Publisher(self.toggle_nodding_topic, std_msgs.msg.Bool, queue_size=self.topic_queue_size)
        self.attention_target_pub = rospy.Publisher(self.attention_target_topic, std_msgs.msg.Int16, queue_size=self.topic_queue_size)

        self.stop_sub = rospy.Subscriber(self.stop_topic, std_msgs.msg.Bool, self.__stopMsgCallback, queue_size=self.topic_queue_size)
        self.toggle_attention_sub = rospy.Subscriber(self.toggle_attention_topic, std_msgs.msg.Bool, self.__toggleAttentionMsgCallback, queue_size=self.topic_queue_size)
        self.toggle_aversion_sub = rospy.Subscriber(self.toggle_aversion_topic, std_msgs.msg.Bool, self.__toggleAversionMsgCallback, queue_size=self.topic_queue_size)
        self.aversion_text_sub = rospy.Subscriber(self.aversion_text_topic, std_msgs.msg.String, self.__aversionTextMsgCallback, queue_size=self.topic_queue_size)
        self.toggle_nodding_sub = rospy.Subscriber(self.toggle_nodding_topic, std_msgs.msg.Bool, self.__toggleNoddingMsgCallback, queue_size=self.topic_queue_size)
        self.nodding_text_sub = rospy.Subscriber(self.nodding_text_topic, std_msgs.msg.String, self.__noddingTextMsgCallback, queue_size=self.topic_queue_size)
        self.attention_target_img_sub = rospy.Subscriber(self.attention_target_img_topic, sensor_msgs.msg.Image, self.__attentionTargetImgMsgCallback, queue_size=self.topic_queue_size)
        self.annotated_tracking_stream_sub = rospy.Subscriber(self.annotated_tracking_stream_topic, sensor_msgs.msg.Image, self.__annotatedTrackingStreamMsgCallback, queue_size=self.topic_queue_size)

    def __stopBtnCallback(self):
        #Broadcast a stop signal to everyone
        self.stop_pub.publish(std_msgs.msg.Bool(True))

    def __stopMsgCallback(self, msg):
        #Adjustments related to other submodules would be invoked as those submodules
        #receive the stop msg and broadcast the sub-stop message to turn off individual functions / vis
        pass

    def __toggleAttentionBtnCallback(self):
        self.toggle_attention_pub.publish(std_msgs.msg.Bool(self.attention_enabled_tk.get()))

    def __toggleAttentionMsgCallback(self, msg):
        #GUI Adjustment upon receiving the attention toggling message
        self.attention_enabled_tk.set(msg.data)
        #Clear target person vis upon toggling attention
        self.__clearAttentionTargetImg()

    def __toggleAversionBtnCallback(self):
        self.toggle_aversion_pub.publish(
            std_msgs.msg.Bool(self.aversion_enabled_tk.get()))

    def __toggleAversionMsgCallback(self, msg):
        #GUI Adjustment upon receiving the aversion toggling message
        self.aversion_enabled_tk.set(
            self.attention_enabled_tk.get()
            and
            msg.data)

    def __aversionTextMsgCallback(self,msg):
        self.aversionStateText.config(text = msg.data)
    
    def __toggleNoddingBtnCallback(self):
        self.toggle_nodding_pub.publish(std_msgs.msg.Bool(self.nodding_enabled_tk.get()))

    def __toggleNoddingMsgCallback(self, msg):
        #GUI Adjustment upon receiving the nodding toggling message
        self.nodding_enabled_tk.set(
            self.attention_enabled_tk.get()
            and
            msg.data)

    def __noddingTextMsgCallback(self,msg):
        self.noddingStateText.config(text = msg.data)
    
    def __guiClose(self):
        self.grace_monitor_frame.destroy()

    def __targetRegBtnCallback(self): 
        if(self.attention_enabled_tk.get()):
            #Clear the visualization of any previously selected target
            self.__clearAttentionTargetImg()

            try:
                #Retrieve text input from the ui
                target_raw_idx = int(self.target_reg_input.get(1.0, "end-1c"))
                
                #Publish this target index to the attention module for registration
                self.attention_target_pub.publish(std_msgs.msg.Int16(target_raw_idx))
            except Exception as e:
                print(e)

    def __clearAttentionTargetImg(self):
        self.source_0_target_img_canvas.itemconfig(self.source_0_target_img_container,image=self.blank_img)

    def __attentionTargetImgMsgCallback(self,msg):
        #Visualize the image of the target person upon successful registration
        cv2_img_from_ros_img_msg = self.cv_bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        self.source_0_target_img_corrected = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv2.cvtColor(cv2_img_from_ros_img_msg, cv2.COLOR_BGR2RGB)))
        self.source_0_target_img_canvas.itemconfig(self.source_0_target_img_container,image=self.source_0_target_img_corrected)

    def __annotatedTrackingStreamMsgCallback(self,msg):
        #Show the img steam from the source of interest annotated by the attention module
        cv2_img_from_ros_img_msg = self.cv_bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        self.source_0_annotated_img_vis = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv2.cvtColor(cv2_img_from_ros_img_msg, cv2.COLOR_BGR2RGB)))
        self.source_0_img_canvas.itemconfig(self.source_0_img_container,image=self.source_0_annotated_img_vis)

    def constructGUI(self):
        #UI Frame
        self.grace_monitor_frame = Tk()
        self.grace_monitor_frame.title("Grace Monitor")
        self.grace_monitor_frame.geometry('1920x1080')

        #STOP
        stop_btn = Button(self.grace_monitor_frame,
                            text = "STOP", 
                            command = self.__stopBtnCallback)
        stop_btn.place(y=50, x=50)

        #Attention enabled state
        self.attention_enabled_tk = BooleanVar()
        enable_attention = Checkbutton(
            self.grace_monitor_frame,
            text = "Enable Attention", 
            variable= self.attention_enabled_tk,
            onvalue = True, offvalue = False,
            command = self.__toggleAttentionBtnCallback)
        enable_attention.place(y=50, x= 200)


        #Aversion: enable, disable, state
        self.aversion_enabled_tk = BooleanVar()
        enableAversion = Checkbutton(
            self.grace_monitor_frame,
            text = "Enable Aversion", 
            variable= self.aversion_enabled_tk,
            onvalue = True, offvalue = False,
            command = self.__toggleAversionBtnCallback)
        enableAversion.place(y=150, x=50)

        self.aversionStateText = Label(self.grace_monitor_frame, text = '')
        self.aversionStateText.place(y=150, x=400)

        #Nodding: enable, disable, state
        self.nodding_enabled_tk = BooleanVar()
        enableNodding = Checkbutton(
            self.grace_monitor_frame,
            text = "ENABLE Nodding", 
            variable= self.nodding_enabled_tk,
            onvalue = True, offvalue = False,
            command = self.__toggleNoddingBtnCallback)
        enableNodding.place(y=250, x=50)
        
        self.noddingStateText = Label(self.grace_monitor_frame, text = '')
        self.noddingStateText.place(y=250, x=400)


        #Target Selection
        self.target_reg_input = Text(
            self.grace_monitor_frame,
            height = 2,
            width = 10)
        self.target_reg_input.pack()
        self.target_reg_input.place(y=50, x= 650)

        confirm_target_reg = Button(self.grace_monitor_frame,
                                text = "REG", 
                                command = self.__targetRegBtnCallback)
        confirm_target_reg.place(y=50, x= 550)


        self.blank_img = PIL.ImageTk.PhotoImage(image=PIL.Image.new("RGB", (640, 480)))
        
        #Visualize the image of the target person
        self.source_0_target_img_canvas = Canvas(self.grace_monitor_frame, width = 640, height = 480)
        self.source_0_target_img_container = self.source_0_target_img_canvas.create_image(0,0, anchor=NW, image=self.blank_img)
        self.source_0_target_img_canvas.place(y=350, x=50)


        #Visualize the image of the camera view
        self.source_0_img_canvas = Canvas(self.grace_monitor_frame, width = 640, height = 480)
        self.source_0_img_container = self.source_0_img_canvas.create_image(0,0, anchor=NW, image=self.blank_img)
        self.source_0_img_canvas.place(y=350, x=750)
            

        #Blocks
        self.grace_monitor_frame.mainloop()


def handle_sigint(signalnum, frame):
    # terminate
    LOGGER.warning('Main interrupted! Exiting.')
    sys.exit()

if __name__ == '__main__':

    signal(SIGINT, handle_sigint)

    grace_gui = Grace_GUI()

    grace_gui.constructGUI()







