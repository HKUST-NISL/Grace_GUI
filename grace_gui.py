import logging
import argparse
import sys
from signal import signal
from signal import SIGINT
import threading
import time
from tkinter import font as tkFont
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
from datetime import datetime

import numpy as np
import re
from .grace_switch import Grace_Switch

class Grace_GUI:
    node_name = "grace_gui"
    cv_bridge = CvBridge()
    topic_queue_size = 100

    stop_topic = "/grace_proj/stop"
    start_topic = "/grace_proj/start"
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

    hr_CAM_cfg_server = "/hr/perception/camera_angle"
    default_grace_chest_cam_motor_angle = 0.55
    dynamic_reconfig_request_timeout = 0.5

    # Topic of dialogue log
    dialogue_log_topic = "/grace_proj/dialogue_log_topic"


    estimated_attention_generic_text = "Paying Attention? "
    estimated_emotion_generic_text = "Emotion: "

    def __init__(self):
        rospy.init_node(self.node_name)

        self.switch_class = Grace_Switch()

        self.stop_pub = rospy.Publisher(self.stop_topic, std_msgs.msg.Bool, queue_size=self.topic_queue_size)
        self.start_pub = rospy.Publisher(self.start_topic, std_msgs.msg.Bool, queue_size=self.topic_queue_size)

        self.toggle_attention_pub = rospy.Publisher(self.toggle_attention_topic, std_msgs.msg.Bool, queue_size=self.topic_queue_size)
        self.toggle_aversion_pub = rospy.Publisher(self.toggle_aversion_topic, std_msgs.msg.Bool, queue_size=self.topic_queue_size)
        # #Deprecated: now nodding is controlled directly by the dialogue system
        # self.toggle_nodding_pub = rospy.Publisher(self.toggle_nodding_topic, std_msgs.msg.Bool, queue_size=self.topic_queue_size)
        self.attention_target_pub = rospy.Publisher(self.attention_target_topic, std_msgs.msg.Int16, queue_size=self.topic_queue_size)

        self.stop_sub = rospy.Subscriber(self.stop_topic, std_msgs.msg.Bool, self.__stopMsgCallback, queue_size=self.topic_queue_size)
        self.start_sub = rospy.Subscriber(
            self.start_topic, std_msgs.msg.Bool, callback=self.__startOfConvMsgCallback, queue_size=self.topic_queue_size
        )
        self.toggle_attention_sub = rospy.Subscriber(self.toggle_attention_topic, std_msgs.msg.Bool, self.__toggleAttentionMsgCallback, queue_size=self.topic_queue_size)
        self.toggle_aversion_sub = rospy.Subscriber(self.toggle_aversion_topic, std_msgs.msg.Bool, self.__toggleAversionMsgCallback, queue_size=self.topic_queue_size)
        self.aversion_text_sub = rospy.Subscriber(self.aversion_text_topic, std_msgs.msg.String, self.__aversionTextMsgCallback, queue_size=self.topic_queue_size)
        # #Deprecated: now nodding is controlled directly by the dialogue system
        # self.toggle_nodding_sub = rospy.Subscriber(self.toggle_nodding_topic, std_msgs.msg.Bool, self.__toggleNoddingMsgCallback, queue_size=self.topic_queue_size)
        # self.nodding_text_sub = rospy.Subscriber(self.nodding_text_topic, std_msgs.msg.String, self.__noddingTextMsgCallback, queue_size=self.topic_queue_size)
        self.attention_target_img_sub = rospy.Subscriber(self.attention_target_img_topic, sensor_msgs.msg.Image, self.__attentionTargetImgMsgCallback, queue_size=self.topic_queue_size)
        self.annotated_tracking_stream_sub = rospy.Subscriber(self.annotated_tracking_stream_topic, sensor_msgs.msg.Image, self.__annotatedTrackingStreamMsgCallback, queue_size=self.topic_queue_size)
        self.target_state_estimation_sub = rospy.Subscriber(self.target_state_estimation_topic, grace_attn_msgs.msg.EmotionAttentionResult, self.__targetStateEstimationMsgCallback, queue_size=self.topic_queue_size)

        ## Logging subscriber and listener
        self.dialogue_log_publisher = rospy.Publisher(self.dialogue_log_topic, grace_attn_msgs.msg.DialogueLog, queue_size=self.topic_queue_size)
        self.dialogue_log_subscriber = rospy.Subscriber(
            self.dialogue_log_topic, grace_attn_msgs.msg.DialogueLog, callback= self.__dialogueLogCallback, queue_size=self.topic_queue_size)

        self.dynamic_CAM_cfg_client = dynamic_reconfigure.client.Client(self.hr_CAM_cfg_server, timeout=self.dynamic_reconfig_request_timeout, config_callback=self.__configureGraceCAMCallback)


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
        # self.logInfo_text = "Log line 1: starting conversation\nLog line 2: asking question 1\nLog line 3: feedback detected! emotion normal! intent continue! Apply policy \"ask_next\"\n\nI will try to maintain about 16-20 lines of log messages under this TKINTER GUI\n7\n8\n9\n10\n11\n12\n13\n14\n15\n16\n17\n18\n19\n20"
        self.logInfo_text = ""

        # self.dialogue_transcript_text = 'A says: A young girl named Alice sits bored by a riverbank, where she suddenly spots a White Rabbit with a pocket watch and waistcoat lamenting that he is late. The surprised Alice follows him down a rabbit hole, which sends her down a lengthy plummet but to a safe landing.\n\nB says: Inside a room with a table, she finds a key to a tiny door, beyond which is a beautiful garden. As she ponders how to fit through the door, she discovers a bottle reading "Drink me".\n\nA says:Alice hesitantly drinks a portion of the bottle\'s contents, and to her astonishment, she shrinks small enough to enter the door. However, she had left the key upon the table and is unable to reach it. Alice then discovers and eats a cake, which causes her to grow to a tremendous size. As the unhappy Alice bursts into tears, the passing White Rabbit flees in a panic, dropping a fan and pair of gloves.\n\nA says: A young girl named Alice sits bored by a riverbank, where she suddenly spots a White Rabbit with a pocket watch and waistcoat lamenting that he is late. The surprised Alice follows him down a rabbit hole, which sends her down a lengthy plummet but to a safe landing.\n\nB says: Inside a room with a table, she finds a key to a tiny door, beyond which is a beautiful garden. As she ponders how to fit through the door, she discovers a bottle reading "Drink me".\n\nA says:Alice hesitantly drinks a portion of the bottle\'s contents, and to her astonishment, she shrinks small enough to enter the door. However, she had left the key upon the table and is unable to reach it. Alice then discovers and eats a cake, which causes her to grow to a tremendous size. As the unhappy Alice bursts into tears, the passing White Rabbit flees in a panic, dropping a fan and pair of gloves.\n\n你依家幾多歲？'
        self.dialogue_transcript_text = ""

    def __stopBtnCallback(self):
        self.switch_class.toggle_replay_switch_state(False)

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
        try:
            self.aversionStateText.config(text = msg.data)
        except Exception as e:
            print(e)


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
    
    # def __guiClose(self):
    #     self.grace_monitor_frame.destroy()

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

    def __endOfConvBtnCallback(self):
        #Broadcast a stop signal to everyone
        self.stop_pub.publish(std_msgs.msg.Bool(True))

    def __startOfConvMsgCallback(self, msg):
        #GUI update upon receiving the start message
        pass

    def __startOfConvBtnCallback(self):
        # Broadcast a start signal to everyone
        self.start_pub.publish(std_msgs.msg.Bool(True))

    def __dialogueLogCallback(self, msg):
        # emergency_signal = msg.emergency
        # disengage_signal = msg.disengage
        # log_message = msg.log

        # Handle log message
        log_message = self.__update_log_message(msg.emergency.data, msg.disengage.data, msg.log.data)
        self.logInfo_message_box.config(state='normal')
        self.logInfo_message_box.insert('end', log_message)
        self.logInfo_message_box.config(state='disabled')
        self.logInfo_message_box.see('end')

        # Handle dialogue transcript
        dialogue_transcript = self.__update_dialogue_transcript(msg.transcript.data)
        self.dialogue_transcript_box.config(state='normal')
        self.dialogue_transcript_box.insert('end', dialogue_transcript)
        self.dialogue_transcript_box.config(state='disabled')
        self.dialogue_transcript_box.see('end')


    def __update_dialogue_transcript(self, new_transcript):
        
        new_message = f"\n\n[{str(datetime.now())}]\n" + new_transcript

        # TODO: write all dialogue to file in the future
        # self.dialogue_transcript_text += new_message
        return new_message

    def __update_log_message(self, emergency_signal, disengage_signal, log_message):

        self.set_progress_bar(log_message)
        
        new_message = f"\n[{str(datetime.now())}]\nEmergency {emergency_signal}, disengage {disengage_signal}; " + log_message

        # TODO: write all log to file in the future
        # self.logInfo_text = self.logInfo_text + new_message
        
        return new_message

    def set_progress_bar(self, log_message):
        pattern = re.compile(r"[nN][oO]:[ ]*(\d+)")
        search_result = pattern.search(log_message)
        if search_result:
            progress_number = search_result.group(1)
            self.orientation_progress_bar["value"] = 25*int(progress_number)

    def __clearLog(self):
        self.logInfo_message_box.config(state='normal')
        self.logInfo_message_box.delete(1.0, END)
        self.logInfo_message_box.config(state='disabled')
        

    def progress_bar_step(self):
        self.orientation_progress_bar["value"] += 25

    def __setCameraAngle(self):
		#tilt chest cam to a given angle
        try:
            target_angle = float(self.cam_ang_input.get(1.0, "end-1c"))
            self.dynamic_CAM_cfg_client.update_configuration({"motor_angle":target_angle})
        except Exception as e:
            print(e)

    def __configureGraceCAMCallback(self,config):
        # # hr sdk seems to be repeatedly throwing back this response
        # rospy.loginfo("Config set to {motor_angle}".format(**config))
        pass


    def constructGUI(self):
        #UI Frame
        self.grace_monitor_frame = Tk()
        self.grace_monitor_frame.title("Grace Monitor")
        self.grace_monitor_frame.geometry('1920x1080')

        ## set DPI for the window
        dpi = self.grace_monitor_frame.winfo_fpixels('1i')
        self.grace_monitor_frame.call('tk', 'scaling', dpi/72)

        # Define font styles
        self.helv = tkFont.Font(family='Helvetica', size=9, weight='bold')

        # STOP BOTTON
        stop_btn = Button(self.grace_monitor_frame,
                            text = "STOP",
                            font=self.helv,
                            command = self.__stopBtnCallback)
        stop_btn.place(y=50, x=50)

        # END BOTTON
        end_btn = Button(
            self.grace_monitor_frame, text="END_CONVERSATION",
            font=self.helv,
            command=self.__endOfConvBtnCallback
        )
        end_btn.place(y=50, x=180)

        # START BUTTON
        start_btn = Button(
            self.grace_monitor_frame, text="START_CONVERSATION",
            font=self.helv,
            command=self.__startOfConvBtnCallback
        )
        start_btn.place(y=50, x=370)

        #Camera angle adjustment
        self.cam_ang_input = Text(
            self.grace_monitor_frame,
            font=self.helv,
            height = 1,
            width = 12)
        self.cam_ang_input.pack()
        self.cam_ang_input.insert(END,str(self.default_grace_chest_cam_motor_angle))
        self.cam_ang_input.place(y=50, x=650)
        cam_btn = Button(
            self.grace_monitor_frame, text="SET CAM",
            font=self.helv,
            command=self.__setCameraAngle
        )
        cam_btn.place(y=50, x=750)
        self.__setCameraAngle()

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
        self.target_reg_input.place(y=855, x=410)

        confirm_target_reg = Button(self.grace_monitor_frame,
                                text = "REG",
                                font=self.helv,
                                command = self.__targetRegBtnCallback)
        confirm_target_reg.place(y=850, x=580)

        target_selection_hint = Label(
            self.grace_monitor_frame, text="Enter Target ID to Focus on:", font=self.helv,
        )
        target_selection_hint.place(y=855, x=60)


        self.blank_img = PIL.ImageTk.PhotoImage(image=PIL.Image.new("RGB", (640, 480)))

        # Visualize raw camera view
        self.source_0_img_canvas = Canvas(self.grace_monitor_frame, width = 640, height = 480)
        self.source_0_img_container = self.source_0_img_canvas.create_image(0,0, anchor=NW, image=self.blank_img)
        self.source_0_img_canvas.place(y=350, x=50)
        source_0_img_canvas_label = Label(
            self.grace_monitor_frame, text="Camera View & Tracking Result",)
        source_0_img_canvas_label.place(y=320, x=50)
        

        #Visualize the image of the target person
        self.source_0_target_img_canvas = Canvas(self.grace_monitor_frame, width = 400, height = 300)
        self.source_0_target_img_container = self.source_0_target_img_canvas.create_image(0,0, anchor=NW, image=self.blank_img)
        self.source_0_target_img_canvas.place(y=670, x=700)
        source_0_target_img_label = Label(
            self.grace_monitor_frame, text="Registered Target"
        )
        source_0_target_img_label.place(y=650, x=700)


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
            self.grace_monitor_frame, text="Head pose & Emotion",
        )
        source_0_state_estimation_img_label.place(y=320, x=700)

        ## Display the log information
        clear_log_button = Button(
            self.grace_monitor_frame, text="CLEAR LOG",
            font=tkFont.Font(family='Helvetica', size=6),
            command=self.__clearLog
        )
        clear_log_button.place(x=1400, y=15)

        log_info_label = Label(
            self.grace_monitor_frame, text="Logging_Info_Area", font=self.helv
        )
        log_info_label.place(x=1150, y=20)

        log_info_frame = Frame(self.grace_monitor_frame, width=710, height=380, bg='white')
        log_info_frame.place(x=1150, y=50)
        log_info_frame.pack_propagate(False)

        self.logInfo_message_box = Text(
            log_info_frame, bg='white', 
            font=tkFont.Font(family="Helvetica", size=8),
            wrap=WORD
            # text=self.logInfo_text,
            # width=70,
            # height=20,
        )
        self.logInfo_message_box.insert("end", self.logInfo_text)
        self.logInfo_message_box.see("end")
        # self.logInfo_message_box.place(x=1150, y=50)
        self.logInfo_message_box.config(state="disabled")
        self.logInfo_message_box.pack(expand=YES, fill=BOTH)


        ## Display the dialog transcript

        dialogue_transcipt_label = Label(
            self.grace_monitor_frame, text="Dialogue_Transcript",
            font=self.helv,
        )
        dialogue_transcipt_label.place(x=1150, y=450)

        dialogue_transcript_scroll_bar = Scrollbar(self.grace_monitor_frame, orient='vertical')
        # dialogue_transcript_scroll_bar.pack(side=RIGHT, fill='y')

        dialogue_transcipt_frame = Frame(self.grace_monitor_frame, width=710, height=310, bg='white')
        dialogue_transcipt_frame.place(x=1150, y=480)
        dialogue_transcipt_frame.pack_propagate(False)

        self.dialogue_transcript_box = Text(
            dialogue_transcipt_frame,
            bg='white',
            font=tkFont.Font(family="Helvetica", size=8),
            # width=70,
            # height=17,
            yscrollcommand=dialogue_transcript_scroll_bar.set
        )
        self.dialogue_transcript_box.insert("end", self.dialogue_transcript_text)
        self.dialogue_transcript_box.see("end")
        # self.dialogue_transcript_box.place(x=1150, y=480)
        self.dialogue_transcript_box.config(state='disabled')
        self.dialogue_transcript_box.pack(expand=YES, fill=BOTH)


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

        ## This is a test button only
        test_progress_bar_button=Button(
            self.grace_monitor_frame, text="Progress\nTest Only", command=self.progress_bar_step
        )
        test_progress_bar_button.place(x=1800, y=850)
        ## progress bar status
        progress_message_0 = Label(
            self.grace_monitor_frame, text="START", font=self.helv
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
            self.grace_monitor_frame, text="END", font=self.helv, anchor=CENTER
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







