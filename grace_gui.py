import logging
import argparse
import sys
from signal import signal
from signal import SIGINT
import threading
import time
from tkinter import font as tkFont
import numpy
import rospy
import sensor_msgs.msg
import std_msgs.msg
import hr_msgs.msg
import grace_attn_msgs.msg
import rosbag
import geometry_msgs.msg
import dynamic_reconfigure.client
import cv2
from cv_bridge import CvBridge
from copy import deepcopy
import os
from tkinter import *
from tkinter import ttk

import PIL.Image
import PIL.ImageTk


class Grace_GUI:
    node_name = "grace_gui"
    cv_bridge = CvBridge()
    topic_queue_size = 100
    
    stop_topic = "/grace_proj/stop"
    toggle_attention_topic = "/grace_proj/toggle_attention"
    toggle_aversion_topic = "/grace_proj/toggle_aversion"
    aversion_text_topic = "/grace_proj/aversion_text"
    # #Deprecated: now nodding is controlled directly by the dialogue system
    # toggle_nodding_topic = "/grace_proj/toggle_nodding"
    # nodding_text_topic = "/grace_proj/nodding_text"
    attention_target_topic = "/grace_proj/attention_target_idx"
    attention_target_img_topic = "/grace_proj/attention_target_img"
    annotated_tracking_stream_topic = "/grace_proj/annotated_tracking_stream"
    target_state_estimation_topic = "/grace_proj/emotion_attention_target_person_output_topic" 


    estimated_attention_generic_text = "Paying Attention? "
    estimated_emotion_generic_text = "Emotion: "

    def __init__(self):
        rospy.init_node(self.node_name)
        
        self.stop_pub = rospy.Publisher(self.stop_topic, std_msgs.msg.Bool, queue_size=self.topic_queue_size)
        self.toggle_attention_pub = rospy.Publisher(self.toggle_attention_topic, std_msgs.msg.Bool, queue_size=self.topic_queue_size)
        self.toggle_aversion_pub = rospy.Publisher(self.toggle_aversion_topic, std_msgs.msg.Bool, queue_size=self.topic_queue_size)
        # #Deprecated: now nodding is controlled directly by the dialogue system
        # self.toggle_nodding_pub = rospy.Publisher(self.toggle_nodding_topic, std_msgs.msg.Bool, queue_size=self.topic_queue_size)
        self.attention_target_pub = rospy.Publisher(self.attention_target_topic, std_msgs.msg.Int16, queue_size=self.topic_queue_size)

        self.stop_sub = rospy.Subscriber(self.stop_topic, std_msgs.msg.Bool, self.__stopMsgCallback, queue_size=self.topic_queue_size)
        self.toggle_attention_sub = rospy.Subscriber(self.toggle_attention_topic, std_msgs.msg.Bool, self.__toggleAttentionMsgCallback, queue_size=self.topic_queue_size)
        self.toggle_aversion_sub = rospy.Subscriber(self.toggle_aversion_topic, std_msgs.msg.Bool, self.__toggleAversionMsgCallback, queue_size=self.topic_queue_size)
        self.aversion_text_sub = rospy.Subscriber(self.aversion_text_topic, std_msgs.msg.String, self.__aversionTextMsgCallback, queue_size=self.topic_queue_size)
        # #Deprecated: now nodding is controlled directly by the dialogue system
        # self.toggle_nodding_sub = rospy.Subscriber(self.toggle_nodding_topic, std_msgs.msg.Bool, self.__toggleNoddingMsgCallback, queue_size=self.topic_queue_size)
        # self.nodding_text_sub = rospy.Subscriber(self.nodding_text_topic, std_msgs.msg.String, self.__noddingTextMsgCallback, queue_size=self.topic_queue_size)
        self.attention_target_img_sub = rospy.Subscriber(self.attention_target_img_topic, sensor_msgs.msg.Image, self.__attentionTargetImgMsgCallback, queue_size=self.topic_queue_size)
        self.annotated_tracking_stream_sub = rospy.Subscriber(self.annotated_tracking_stream_topic, sensor_msgs.msg.Image, self.__annotatedTrackingStreamMsgCallback, queue_size=self.topic_queue_size)
        self.target_state_estimation_sub = rospy.Subscriber(self.target_state_estimation_topic, grace_attn_msgs.msg.EmotionAttentionResult, self.__targetStateEstimationMsgCallback, queue_size=self.topic_queue_size)

        # UI Frame Elements
        self.grace_monitor_frame = None
        self.attention_enabled_tk = None
        self.aversion_enabled_tk = None
        self.aversionStateText = None
        self.source_0_annotated_img_vis = None
        self.source_0_state_estimation_img_vis = None
        self.source_0_target_img_corrected = None
        self.target_reg_input = None
        self.source_0_img_canvas = None
        self.blank_img = None
        self.source_0_img_canvas = None
        self.source_0_img_container = None
        self.source_0_target_img_canvas = None
        self.source_0_target_img_container = None
        self.estimatedAttentionText = None
        self.estimatedEmotionText = None
        self.source_0_state_estimation_img_canvas = None
        self.source_0_state_estimation_img_container = None
        self.logInfo_message_box = None
        self.dialogue_transcript_box = None
        self.orientation_progress_bar = None

        # UI Font Size
        self.helv = None

        # Default Messages
        self.logInfo_text = "Log line 1\nLog line 2: hahahahaha\nLog line 3: this is a long long log message\n\nI will try to maintain about 16-20 lines of log messages under this TKINTER GUI\n7\n8\n9\n10\n11\n12\n13\n14\n15\n16\n17\n18\n19\n20"
        
        self.dialogue_transcript = 'A says: A young girl named Alice sits bored by a riverbank, where she suddenly spots a White Rabbit with a pocket watch and waistcoat lamenting that he is late. The surprised Alice follows him down a rabbit hole, which sends her down a lengthy plummet but to a safe landing.\n\nB says: Inside a room with a table, she finds a key to a tiny door, beyond which is a beautiful garden. As she ponders how to fit through the door, she discovers a bottle reading "Drink me".\n\nA says:Alice hesitantly drinks a portion of the bottle\'s contents, and to her astonishment, she shrinks small enough to enter the door. However, she had left the key upon the table and is unable to reach it. Alice then discovers and eats a cake, which causes her to grow to a tremendous size. As the unhappy Alice bursts into tears, the passing White Rabbit flees in a panic, dropping a fan and pair of gloves.'

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


    # #Deprecated: now nodding is controlled directly by the dialogue system

    # def __toggleNoddingBtnCallback(self):
    #     self.toggle_nodding_pub.publish(std_msgs.msg.Bool(self.nodding_enabled_tk.get()))

    # def __toggleNoddingMsgCallback(self, msg):
    #     #GUI Adjustment upon receiving the nodding toggling message
    #     self.nodding_enabled_tk.set(
    #         self.attention_enabled_tk.get()
    #         and
    #         msg.data)

    # def __noddingTextMsgCallback(self,msg):
    #     self.noddingStateText.config(text = msg.data)
    
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

    def __targetStateEstimationMsgCallback(self,msg):
        #Show the visualization image output from the state estiamtor including gaze, expression, headpose, etc.
        cv2_img_from_ros_img_msg = self.cv_bridge.imgmsg_to_cv2(msg.visualization_frame,desired_encoding='bgr8')
        self.source_0_state_estimation_img_vis = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv2.cvtColor(cv2_img_from_ros_img_msg, cv2.COLOR_BGR2RGB)))
        self.source_0_state_estimation_img_canvas.itemconfig(self.source_0_state_estimation_img_container,image=self.source_0_state_estimation_img_vis)
        #Display text on the gui as well
        self.estimatedAttentionText.config(text = self.estimated_attention_generic_text + msg.attention.data)
        self.estimatedEmotionText.config(text = self.estimated_emotion_generic_text + msg.emotion.data)

    def __endConversationCallback(self):
        # endConversation
        pass

    def progress_bar_step(self):
        self.orientation_progress_bar["value"] += 25

    def constructGUI(self):
        #UI Frame
        self.grace_monitor_frame = Tk()
        self.grace_monitor_frame.title("Grace Monitor")
        self.grace_monitor_frame.geometry('1920x1080')

        # Define font styles
        self.helv = tkFont.Font(family='Helvetica', size=10, weight='bold')

        # STOP BOTTON
        stop_btn = Button(self.grace_monitor_frame,
                            text = "STOP",
                            font=self.helv,
                            command = self.__stopBtnCallback)
        stop_btn.place(y=50, x=50)

        # END BOTTON
        end_btn = Button(
            self.grace_monitor_frame, text="END_CONVERSATION", font=self.helv,
            command=self.__endConversationCallback
        )
        end_btn.place(y=50, x=150)

        #Attention enabled state
        self.attention_enabled_tk = BooleanVar()
        enable_attention = Checkbutton(
            self.grace_monitor_frame,
            text = "Enable Attention",
            font = self.helv,
            variable= self.attention_enabled_tk,
            onvalue = True, offvalue = False,
            command = self.__toggleAttentionBtnCallback)
        enable_attention.place(y=150, x= 50)


        #Aversion: enable, disable, state
        self.aversion_enabled_tk = BooleanVar()
        enableAversion = Checkbutton(
            self.grace_monitor_frame,
            text = "Enable Aversion",
            font = self.helv,
            variable= self.aversion_enabled_tk,
            onvalue = True, offvalue = False,
            command = self.__toggleAversionBtnCallback)
        enableAversion.place(y=250, x=50)

        self.aversionStateText = Label(
            self.grace_monitor_frame, text = '', font=self.helv,
            )
        self.aversionStateText.place(y=250, x=450)


        # #Deprecated: now nodding is controlled directly by the dialogue system
        # #Nodding: enable, disable, state
        # self.nodding_enabled_tk = BooleanVar()
        # enableNodding = Checkbutton(
        #     self.grace_monitor_frame,
        #     text = "ENABLE Nodding", 
        #     variable= self.nodding_enabled_tk,
        #     onvalue = True, offvalue = False,
        #     command = self.__toggleNoddingBtnCallback)
        # enableNodding.place(y=250, x=50)
        
        # self.noddingStateText = Label(self.grace_monitor_frame, text = '')
        # self.noddingStateText.place(y=250, x=400)


        # Target Selection
        self.target_reg_input = Text(
            self.grace_monitor_frame,
            font=self.helv,
            height = 1,
            width = 12)
        self.target_reg_input.pack()
        self.target_reg_input.place(y=855, x=400)

        confirm_target_reg = Button(self.grace_monitor_frame,
                                text = "REG",
                                font=self.helv,
                                command = self.__targetRegBtnCallback)
        confirm_target_reg.place(y=850, x=550)

        target_selection_hint = Label(
            self.grace_monitor_frame, text="Enter Target ID to Focus on:", font=self.helv,
        )
        target_selection_hint.place(y=855, x=80)


        self.blank_img = PIL.ImageTk.PhotoImage(image=PIL.Image.new("RGB", (640, 480)))

        # Visualize raw camera view
        self.source_0_img_canvas = Canvas(self.grace_monitor_frame, width = 640, height = 480)
        self.source_0_img_container = self.source_0_img_canvas.create_image(0,0, anchor=NW, image=self.blank_img)
        self.source_0_img_canvas.place(y=350, x=50)
        source_0_img_canvas_label = Label(
            self.grace_monitor_frame, text="raw_camera_view",)
        source_0_img_canvas_label.place(y=360, x=50)
        

        #Visualize the image of the target person
        self.source_0_target_img_canvas = Canvas(self.grace_monitor_frame, width = 400, height = 300)
        self.source_0_target_img_container = self.source_0_target_img_canvas.create_image(0,0, anchor=NW, image=self.blank_img)
        self.source_0_target_img_canvas.place(y=660, x=700)
        source_0_target_img_label = Label(
            self.grace_monitor_frame, text="image_of_the_target_person"
        )
        source_0_target_img_label.place(y=670, x=700)


        # Display the estimated state
        # Texts
        self.estimatedAttentionText = Label(
            self.grace_monitor_frame,
            text = self.estimated_attention_generic_text,
            font=self.helv,
            )
        self.estimatedAttentionText.place(y=150, x=850)
        self.estimatedEmotionText = Label(
            self.grace_monitor_frame,
            text = self.estimated_emotion_generic_text,
            font=self.helv,
            )
        self.estimatedEmotionText.place(y=250, x=850)


        # Annotated image
        self.source_0_state_estimation_img_canvas = Canvas(self.grace_monitor_frame, width = 400, height = 300)
        self.source_0_state_estimation_img_container = self.source_0_state_estimation_img_canvas.create_image(0,0, anchor=NW, image=self.blank_img)
        self.source_0_state_estimation_img_canvas.place(y=350, x=700)
        source_0_state_estimation_img_label = Label(
            self.grace_monitor_frame, text="Annotated_image",
        )
        source_0_state_estimation_img_label.place(y=360, x=700)

        # Display the log information
        log_info_label = Label(
            self.grace_monitor_frame, text="Logging_Info_Area", font=self.helv
        )
        log_info_label.place(x=1150, y=20)
        self.logInfo_message_box = Message(
            self.grace_monitor_frame, anchor='nw', bg='white', font=self.helv,
            text=self.logInfo_text,
            width=700,
        )
        self.logInfo_message_box.place(x=1150, y=50)

        # Display the dialog transcript
        dialogue_transcipt_label = Label(
            self.grace_monitor_frame, text="Dialogue_Transcript",
            font=self.helv,
        )
        dialogue_transcipt_label.place(x=1150, y=450)
        self.dialogue_transcript_box = Message(
            self.grace_monitor_frame, anchor=NW, bg='white',
            font=self.helv,
            text=self.dialogue_transcript,
            width=700,
        )
        self.dialogue_transcript_box.place(x=1150, y=480)


        # Progress bar
        progress_bar_label = Label(
            self.grace_monitor_frame, text="Progress_Bar",
            font=self.helv,
        )
        progress_bar_label.place(x=1150, y=810)
        self.orientation_progress_bar = ttk.Progressbar(
            self.grace_monitor_frame, orient=HORIZONTAL, length=600, mode="indeterminate"
        )
        self.orientation_progress_bar.place(x=1150, y=850)
        self.orientation_progress_bar["value"] = 0

        test_progress_bar_button=Button(
            self.grace_monitor_frame, text="Progress", command=self.progress_bar_step
        )
        test_progress_bar_button.place(x=1800, y=850)
        ## progress bar status
        progress_message_0 = Label(
            self.grace_monitor_frame, text="start", font=self.helv
        )
        progress_message_0.place(x=1150, y=890)
        progress_message_1 = Label(
            self.grace_monitor_frame, text="Question 1", font=self.helv, anchor=CENTER
        )
        progress_message_1.place(x=1150+80+25, y=890)

        progress_message_2 = Label(
            self.grace_monitor_frame, text="Question 2", font=self.helv, anchor=CENTER
        )
        progress_message_2.place(x=1150+80+120+50, y=890)

        progress_message_3 = Label(
            self.grace_monitor_frame, text="Question 3", font=self.helv, anchor=CENTER
        )
        progress_message_3.place(x=1350+120+50+25, y=890)

        progress_message_4 = Label(
            self.grace_monitor_frame, text="End", font=self.helv, anchor=CENTER
        )
        progress_message_4.place(x=1150+560, y=890)



        #Blocks
        self.grace_monitor_frame.mainloop()


def handle_sigint(signalnum, frame):
    # terminate
    LOGGER.warning('Main interrupted! Exiting.')
    sys.exit()

if __name__ == '__main__':

    # Add LOGGER
    LOGGER = logging.getLogger()

    signal(SIGINT, handle_sigint)

    grace_gui = Grace_GUI()

    grace_gui.constructGUI()







