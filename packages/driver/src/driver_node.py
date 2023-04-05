#!/usr/bin/env python3

import rospy

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32
from turbojpeg import TurboJPEG
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped, BoolStamped, VehicleCorners, LEDPattern
from duckietown_msgs.srv import ChangePattern, SetCustomLEDPattern
from dt_apriltags import Detector, Detection
import yaml

YELLOW_MASK = [(20, 60, 0), (50, 255, 255)]  # for yellow mask
BLUE_MASK = [(90, 100, 50), (140, 255, 255)]  # for blue mask
ORANGE_MASK = [(0, 100, 50), (20, 255, 255)]  # for orange mask
DEBUG = False
ENGLISH = False


class DriverNode(DTROS):

    def __init__(self, node_name):
        super(DriverNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")
        self.stall = None # TODO: Get stall number as launch parameter

        # Subscribers
        self.sub_camera = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed", CompressedImage, self.img_callback, queue_size=1, buff_size="20MB")

        # Odometry Subscribers
        # self.sub_distance_left = rospy.Subscriber(f'/{self.veh}/odometry_node/integrated_distance_left', Float32, self.cb_distance_left)
        # self.sub_distance_right = rospy.Subscriber(f'/{self.veh}/odometry_node/integrated_distance_right', Float32, self.cb_distance_right)

        # Publishers
        self.vel_pub = rospy.Publisher("/" + self.veh + "/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
        self.pub_wheel_commands = rospy.Publisher(f'/{self.veh}/wheels_driver_node/wheels_cmd', WheelsCmdStamped, queue_size=1)
        self.pub_mask = rospy.Publisher("/" + self.veh + "/output/image/mask/compressed", CompressedImage, queue_size=1)
        self.pub_img_bool = DEBUG

        # image processing tools
        self.image_msg = None
        self.bridge = CvBridge()
        self.jpeg = TurboJPEG()

        # info from subscribers
        self.intersection_detected = False

        # find the calibration parameters
        # for detecting apriltags
        camera_intrinsic_dict = self.readYamlFile(
            f'/data/config/calibrations/camera_intrinsic/{self.veh}.yaml')

        self.K = np.array(
            camera_intrinsic_dict["camera_matrix"]["data"]).reshape((3, 3))
        self.R = np.array(
            camera_intrinsic_dict["rectification_matrix"]["data"]).reshape((3, 3))
        self.DC = np.array(
            camera_intrinsic_dict["distortion_coefficients"]["data"])
        self.P = np.array(
            camera_intrinsic_dict["projection_matrix"]["data"]).reshape((3, 4))
        self.h = camera_intrinsic_dict["image_height"]
        self.w = camera_intrinsic_dict["image_width"]

        f_x = camera_intrinsic_dict['camera_matrix']['data'][0]
        f_y = camera_intrinsic_dict['camera_matrix']['data'][4]
        c_x = camera_intrinsic_dict['camera_matrix']['data'][2]
        c_y = camera_intrinsic_dict['camera_matrix']['data'][5]
        self.camera_params = [f_x, f_y, c_x, c_y]

        # initialize apriltag detector
        self.at_detector = Detector(searchpath=['apriltags'],
                                    families='tag36h11',
                                    nthreads=1,
                                    quad_decimate=1.0,
                                    quad_sigma=0.0,
                                    refine_edges=1,
                                    decode_sharpening=0.25,
                                    debug=0)
        self.at_distance = 0
        self.led_colors = {
            0: 'WHITE',
            1: 'RED'
        }
        self.apriltags = {
            38: 'STOP',
            163: 'STOP, PEDUCKSTRIANS',
            48: 'RIGHT TURN',
            50: 'LEFT TURN',
            56: 'STRAIGHT',
            207: 'PARKING 1',
            226: 'PARKING 2',
            227: 'TRAFFIC LIGHT',
            228: 'PARKING 3',
            75: 'PARKING 4'
        }
        self.at_detected = False
        self.closest_at = None

        # apriltag detection filters
        self.decision_threshold = 10
        self.z_threshold = 0.35

        # PID Variables for driving
        self.proportional = None
        self.offset = 170
        if ENGLISH:
            self.offset = -170
        self.velocity = 0.25
        self.twist = Twist2DStamped(v=self.velocity, omega=0)
        self.turn_speed = 0.15

        self.P = 0.025
        self.D = -0.007
        self.last_error = 0
        self.last_time = rospy.get_time()
        self.calibration = 0.5

        self.camera_center = 340
        self.leader_x = self.camera_center
        self.straight_threshold = 75

        # Service proxies
        rospy.wait_for_service(
            f'/{self.veh}/led_emitter_node/set_custom_pattern')
        self.led_service = rospy.ServiceProxy(
            f'/{self.veh}/led_emitter_node/set_custom_pattern', SetCustomLEDPattern, persistent=True)

        self.loginfo("Initialized")
        # Shutdown hook
        rospy.on_shutdown(self.hook)

    def img_callback(self, msg):
        self.image_msg = msg

    def run_(self):
        rate = rospy.Rate(8)  # 8hz
        i = 0
        while not rospy.is_shutdown():
            if i % 4 == 0:
                if self.image_msg is not None:
                    self.detect_lane(self.image_msg)
            if self.intersection_detected:
                self.intersection_sequence()
            self.drive()
            rate.sleep()
            i += 1

    def run(self):
        # self.stage1()
        self.stage2()
        self.stage3()
        self.stop()
        rospy.signal_shutdown("Program terminating.")

    def drive_to_intersection(self): # and stop
        # new intersection behaviour defined
        # TODO: drive till we reach red line
        # while red line not in bottom crop...
            # self.drive()
        rate = rospy.Rate(4)  # 8hz
        while not self.intersection_detected: # TODO: implement intersection (red line) detection
            if self.image_msg is not None:
                self.detect_lane(self.image_msg)
                # self.detect_intersection(self.image_msg) # TODO: replace with red line detections
            self.drive()
            rate.sleep()

        self.stop()
        self.pass_time(7)

    def sprint(self):
        # keep in mind that this keeps checking for april tags
        rate = rospy.Rate(4)  # 8hz
        while not self.at_detected: # TODO: adjust distance for detecting apriltag
            if self.image_msg is not None:
                self.detect_lane(self.image_msg)
                self.at_detected = self.detect_apriltag(self.image_msg)
            self.drive()
            rate.sleep()
        # TODO: how are we currently handling the derivative kick problem?

    def correct(self):
        # TODO: currently when the bot stops it stops at an angle that prevents it from seeing
        # the yellow tape. I was thinking that if we add code for it to "correct" itself
        # i.e. re-orient itself so it sees the yellow lines properly
        # we won't have the problem of it driving off the road when it stops detecting the yellow
        # we could also add a flag for this within the drive function,
        # i.e. if we can't see yellow, stop, correct, then keep driving.
        return

    def route1(self):
        # new intersection behaviour defined
        # After crossing red line...
        self.right_turn()
        self.sprint()
        if self.closest_at != 50:
            rospy.loginfo("WARNING: Detecting wrong apriltag for route 1.")
        self.drive_to_intersection()
        rospy.loginfo("Turning left.")
        self.left_turn()

        self.sprint()
        if self.closest_at == 56:
            rospy.loginfo("YAY: Almost done with Stage 1. :)")
        else:
            rospy.loginfo("WARNING: Detecting wrong apriltag for route 1.")
        self.drive_to_intersection()
        self.pub_straight()

    def route2(self):
        # new intersection behaviour defined
        # After crossing red line...
        self.pub_straight() # TODO: needs fixing, (check function's comments)
        rate = rospy.Rate(4)  # 8hz
        self.sprint()
        if self.closest_at != 48:
            rospy.loginfo("WARNING: Detecting wrong apriltag for route 2.")
        self.drive_to_intersection()
        rospy.loginfo("Turning right.")
        self.right_turn()

        self.sprint()
        if self.closest_at != 50:
            rospy.loginfo("WARNING: Detecting wrong apriltag for route 1.")
        self.drive_to_intersection()
        rospy.loginfo("Turning left.")
        self.left_turn()

    def stage1(self):
        rate = rospy.Rate(4)  # 8hz
        self.sprint()

        if self.closest_at == 48:
            self.route1()
        elif self.closest_at == 56:
            self.route2()
        else:
            rospy.loginfo("ERROR: Detecting wrong apriltag for stage 1.")
            rospy.loginfo("REPEATING stage 1.") # TODO: Should we do this?
            self.stage1()

    def drive_to_blue_line(self):
        # like drive_to_intersection but stop farther away to avoid duck murder
        rate = rospy.Rate(4)
        self.close_to_blue = False
        while not self.close_to_blue:
            if self.image_msg is not None:
                self.detect_lane(self.image_msg)
                self.detect_blue_line(self.image_msg)
            self.drive()
            rate.sleep()

        self.stop()

    def check_for_ducks(self, msg):
        found_ducks = False
        
        img = self.jpeg.decode(msg.data)
        crop = img[:, :, :]
        crop_height = crop.shape[0]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        yellow_mask = cv2.inRange(hsv, YELLOW_MASK[0], YELLOW_MASK[1])
        orange_mask = cv2.inRange(hsv, ORANGE_MASK[0], ORANGE_MASK[1])
        combined_mask = cv2.bitwise_or(yellow_mask, orange_mask)
        crop = cv2.bitwise_and(crop, crop, mask=combined_mask)
        
        yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Search for nearest ducks
        min_area = 50
        max_idx = -1
        for yellow_contour in yellow_contours:
            for orange_contour in orange_contours:
                
                area = cv2.contourArea(yellow_contour)
                if area > min_area:
                    M1 = cv2.moments(yellow_contour)
                    M2 = cv2.moments(orange_contour)
                    

                    try:
                        cx1 = int(M1['m10'] / M1['m00'])
                        cy1 = int(M1['m01'] / M1['m00'])
                        cx2 = int(M2['m10'] / M2['m00'])
                        cy2 = int(M2['m01'] / M2['m00'])
                    
                        distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                    
                        if distance < 40:
                            found_ducks = True
                        
                            if DEBUG:
                                cv2.drawContours(crop, [yellow_contour], -1, (0, 255, 0), 3)
                                cv2.drawContours(crop, [orange_contour], -1, (0, 255, 0), 3)
                                cv2.circle(crop, (cx1, cy1), 7, (0, 0, 255), -1)

                    except:
                        pass

        # debugging
        if DEBUG:
            rect_img_msg = CompressedImage(
                format="jpeg", data=self.jpeg.encode(crop))
            self.pub_mask.publish(rect_img_msg)
            
        return found_ducks

    def check_for_bot(self):
        rate = rospy.Rate(4)
        self.close_to_blue = False
        while not self.close_to_blue:
            if self.image_msg is not None:
                self.detect_lane(self.image_msg)
                # self.detect_intersection(self.image_msg) # TODO: replace with blue contour/bot detections
            self.drive()
            rate.sleep()
        # TODO: stop at a suitable distance so we have time to switch lanes
        self.stop()

    def switch_lanes(self):
        # TODO: Probably switch to English driver. Keep moving till (?). Then switch back.
        pass

    def avoid_ducks(self):
        self.sprint()
        if self.closest_at != 163:
            rospy.loginfo("WARNING: Detecting wrong apriltag for stage 2.")
        self.drive_to_blue_line()
        self.stop() # just to make sure
        ducks_present = self.check_for_ducks(self.image_msg)
        if ducks_present:
            while self.check_for_ducks(self.image_msg):
                self.pass_time(1)
            self.pass_time(5)
        else:
            self.pass_time(5)

    def stage2(self):
        self.avoid_ducks()
        self.check_for_bot()
        self.switch_lanes()
        self.avoid_ducks()
        self.drive()

    def park(self, stall):
        # TODO
        if stall == 1:
            pass
        elif stall == 2:
            pass
        elif stall == 3:
            pass
        elif stall == 4:
            pass
        else:
            rospy.loginfo("WARNING: Wrong input for parking stall number.")
            rospy.loginfo("Automatically backing into stall 3.")

    def stage3(self):
        self.sprint()
        if self.closest_at != 38:
            rospy.loginfo("WARNING: Detecting wrong apriltag for stage 3.")
        rospy.loginfo("PARKING.")
        self.drive_to_intersection()
        self.pub_straight()
        while self.at_distance > 0.4:
            continue
        if self.closest_at != 227:
            rospy.loginfo("WARNING: Detecting wrong apriltag for stage 3.")
        self.park(self.stall)

    def detect_lane(self, msg):
        # decodes the image for lane following
        img = self.jpeg.decode(msg.data)
        # self.at_detected = self.detect_apriltag(msg)
        crop = img[:, :, :]
        crop_width = crop.shape[1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, YELLOW_MASK[0], YELLOW_MASK[1])
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Search for lane in front
        max_area = 20
        max_idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area

        if max_idx != -1:
            M = cv2.moments(contours[max_idx])
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                self.proportional = cx - int(crop_width / 2) + self.offset
                if DEBUG:
                    cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass
        else:
            self.proportional = None

        # debugging
        if DEBUG:
            rect_img_msg = CompressedImage(
                format="jpeg", data=self.jpeg.encode(crop))
            self.pub_mask.publish(rect_img_msg)

    def drive(self): # TODO: improve so it stops moving off the road at turns
        if self.proportional is None:
            self.twist.omega = 0
        else:
            # P Term
            P = -self.proportional * self.P

            # D Term
            d_error = (self.proportional - self.last_error) / \
                (rospy.get_time() - self.last_time)
            self.last_error = self.proportional
            self.last_time = rospy.get_time()
            D = d_error * self.D

            self.twist.v = self.velocity
            self.twist.omega = P + D

        self.vel_pub.publish(self.twist)

    def intersection_sequence(self):
        # outdated
        if DEBUG:
            rospy.loginfo("INTERSECTION DETECTED at {}".format(str(self.at_distance)))

        # latency between detecting intersection and stopping
        self.pub_straight(linear=0)
        wait_time = self.at_distance * 2  # 1.5 # seconds
        self.pass_time(wait_time)
        self.stop()
        self.pass_time(6)

    def right_turn(self):
        self.set_lights("right")
        # needs some odometry code, using wheels_cmd

    def left_turn(self):
        self.set_lights("left")
        # needs some odometry code, using wheels_cmd

    def right_turn_(self):
        self.set_lights("right")
        self.twist.v = 0
        self.twist.omega = -12
        self.vel_pub.publish(self.twist)
        start_time = rospy.get_time()
        while rospy.get_time() < start_time + 0.6:
            continue

    def left_turn_(self):
        self.set_lights("left")
        self.twist.v = 0
        self.twist.omega = 12
        self.vel_pub.publish(self.twist)
        start_time = rospy.get_time()
        while rospy.get_time() < start_time + 0.6:
            continue

    def stop(self):
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        self.set_lights("stop")

    def pub_straight(self, linear=None):
        # TODO: needs fixing, shouldn't be skewed. Get to move straight. Maybe add wheel calibration.
        self.twist.v = self.velocity
        if linear is not None:
            self.twist.omega = linear
        else:
            self.twist.omega = self.calibration
        self.vel_pub.publish(self.twist)

    def pass_time(self, t):
        start_time = rospy.get_time()
        while rospy.get_time() < start_time + t:
            continue

    def detect_apriltag(self, img_msg):
        # detect an intersection by finding the corresponding apriltags
        cv_image = None
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(img_msg)
        except CvBridgeError as e:
            self.log(e)
            return []

        # undistort the image
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.K, self.DC, (self.w, self.h), 0, (self.w, self.h))
        image_np = cv2.undistort(cv_image, self.K, self.DC, None, newcameramtx)

        # convert the image to black and white
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # detect tags present in image
        tags = self.at_detector.detect(
            image_gray, estimate_tag_pose=True, camera_params=self.camera_params, tag_size=0.065)

        closest_tag_z = 1000
        closest = None

        for tag in tags:
            # ignore distant tags and tags with bad decoding
            z = tag.pose_t[2][0]
            if tag.decision_margin < self.decision_threshold or z > self.z_threshold:
                continue

            # update the closest-detected tag if needed
            if z < closest_tag_z:
                closest_tag_z = z
                closest = tag

        if closest:
            if closest.tag_id in self.apriltags:
                self.at_distance = closest.pose_t[2][0]
                self.closest_at = closest.tag_id
                return True
        return False

    def detect_blue_line(self, msg):
        img = self.jpeg.decode(msg.data)
        crop = img[:, :, :]
        crop_height = crop.shape[0]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, BLUE_MASK[0], BLUE_MASK[1])
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Search for nearest blue line
        max_area = 20
        max_idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area

        if max_idx != -1:
            M = cv2.moments(contours[max_idx])
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if DEBUG:
                    cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
                    
                if cy > int(crop_height * (2/3)):
                    self.close_to_blue = True
                else:
                    self.close_to_blue = False
            except:
                pass


        # debugging
        if DEBUG:
            rect_img_msg = CompressedImage(
                format="jpeg", data=self.jpeg.encode(crop))
            self.pub_mask.publish(rect_img_msg)
            
    def set_lights(self, state):
        msg = LEDPattern()
        if state == "left":
            msg.color_list = ['yellow','yellow','switchedoff','switchedoff','switchedoff']
            msg.color_mask = [0, 0, 0, 0, 0]
            msg.frequency = 2
            msg.frequency_mask = [1, 1, 0, 0, 0]
        elif state == "right":
            msg.color_list = ['switchedoff','switchedoff','yellow','yellow','switchedoff']
            msg.color_mask = [0, 0, 0, 0, 0]
            msg.frequency = 2
            msg.frequency_mask = [0, 0, 1, 1, 0]
        elif state == "stop":
            msg.color_list = ['switchedoff','red','switchedoff','red','switchedoff']
            msg.color_mask = [0, 0, 0, 0, 0]
            msg.frequency = 0
            msg.frequency_mask = [0, 0, 0, 0, 0]
        elif state == "off":
            msg.color_list = ['switchedoff','switchedoff','switchedoff','switchedoff','switchedoff']
            msg.color_mask = [0, 0, 0, 0, 0]
            msg.frequency = 0
            msg.frequency_mask = [0, 0, 0, 0, 0]
        self.led_service(msg)

    def hook(self):
        print("SHUTTING DOWN")
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        for i in range(8):
            self.vel_pub.publish(self.twist)

    def readYamlFile(self, fname):
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         % (fname, exc), type='fatal')
                rospy.signal_shutdown()
                return


if __name__ == "__main__":
    node = DriverNode("driver_node")
    node.run()
