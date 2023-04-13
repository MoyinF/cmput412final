#!/usr/bin/env python3

import rospy, cv2, yaml, math
import numpy as np

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header, ColorRGBA
from duckietown_msgs.msg import Twist2DStamped, LEDPattern
from turbojpeg import TurboJPEG
from cv_bridge import CvBridge, CvBridgeError
from dt_apriltags import Detector

# Color masks
YELLOW_MASK = [(20, 60, 0), (50, 255, 255)]  # for lane mask
DUCK_MASK = [(7, 0, 0), (26, 255, 255)] # for duck mask
BLUE_MASK = [(100, 130, 100), (120, 255, 175)]  # for blue mask
ORANGE_MASK = [(0, 0, 0), (4, 255, 255)]  # for orange mask, lower range
ORANGE_MASK2 = [(127, 60, 110), (179, 255, 255)] # upper range
STOP_MASK1 = [(0, 75, 150), (5, 150, 255)] # for stop lines
STOP_MASK2 = [(175, 75, 150), (179, 150, 255)] # for stop lines
ROBOT_MASK = [(100, 90, 60), (140, 190, 130)] # bot detection

# debugging flags
DEBUG = False # this is only used for the image publishing.
AT_DEBUG = False # this is only used for april tag detections
LINES_DEBUG = False # check if red and blue lines were detected
DUCK_DEBUG = False # duck detections
BOT_DEBUG = False # bot detections
SWITCH_LANE_DEBUG = False

# flags that affect behavior
ENGLISH = False
AT_SYNCHRONOUS = False
CALLBACK_PROCESSING = True
FORWARD_PARKING = True

# Notes: 2023-04-09
# Moves too slow? Check offset value
# Turns when meant to move straight? Check straight_duration

# TODO:
# get stall num from command line
# change so it stops a bit closer to the blue line
# or increase the size of the crop, problem is it might see far ducks and stop.
# Maybe: implement correct

class DriverNode(DTROS):

    def __init__(self, node_name):
        super(DriverNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")

        # Get constants
        self.constants = self.readYamlFile(f'/data/final_config.yaml')

        # Get stall number from /data/stall file
        self.stall = self.constants['stall']

        # Shutdown hook
        rospy.on_shutdown(self.hook)

        # Subscribers
        self.sub_camera = rospy.Subscriber(f"/{self.veh}/camera_node/image/compressed", CompressedImage, self.img_callback, queue_size=1, buff_size="20MB")

        # Publishers
        self.vel_pub = rospy.Publisher(f"/{self.veh}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
        self.pub_mask = rospy.Publisher(f"/{self.veh}/output/image/mask/compressed", CompressedImage, queue_size=1)
        self.led_publisher = rospy.Publisher(f"/{self.veh}/led_emitter_node/led_pattern", LEDPattern, queue_size=1)

        # image processing tools
        self.image_msg = None
        self.bridge = CvBridge()
        self.jpeg = TurboJPEG()

        # info from subscribers
        self.intersection_detected = False

        # find the calibration parameters for detecting apriltags
        camera_intrinsic_dict = self.readYamlFile(f'/data/config/calibrations/camera_intrinsic/{self.veh}.yaml')

        self.K = np.array(camera_intrinsic_dict["camera_matrix"]["data"]).reshape((3, 3))
        self.R = np.array(camera_intrinsic_dict["rectification_matrix"]["data"]).reshape((3, 3))
        self.DC = np.array(camera_intrinsic_dict["distortion_coefficients"]["data"])
        self.P = np.array(camera_intrinsic_dict["projection_matrix"]["data"]).reshape((3, 4))
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
        self.at_detector_2 = Detector(searchpath=['apriltags'],
                                    families='tag36h11',
                                    nthreads=1,
                                    quad_decimate=1.0,
                                    quad_sigma=0.0,
                                    refine_edges=1,
                                    decode_sharpening=0.25,
                                    debug=0)
        self.at_distance = 0
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

        # Initialize LED color-changing
        self.pattern = LEDPattern()
        self.pattern.header = Header()
        if self.veh != "csc22906":
            self.set_LEDs(True)

        # apriltag detection filters
        self.decision_threshold = self.constants['decision_threshold']
        self.z_threshold = self.constants['z_threshold']

        # PID Variables for driving
        self.proportional = None
        self.offset = 170
        if ENGLISH:
            self.offset = -170
        self.velocity = self.constants['velocity']
        self.twist = Twist2DStamped(v=self.velocity, omega=0)
        self.turn_speed = 0.15

        self.P = 0.025
        self.D = -0.007
        self.last_error = 0
        self.last_time = rospy.get_time()
        self.calibration = 0

        if self.veh == "csc22906":
            self.calibration = 0.5

        if self.veh == "csc22907":
            self.P = self.constants['P']
            self.D = self.constants['D']
            self.offset = self.constants['offset']
            if ENGLISH:
                self.offset = -self.constants['offset']
            self.calibration = self.constants['calibration']
            self.turn_speed = self.constants['turn_speed']

        # Turning variables
        self.left_turn_duration = self.constants['left_turn_duration']
        self.right_turn_duration = self.constants['right_turn_duration'] # was 0.5 before, but changed because kept bumping into apriltag
        self.turn_in_place_duration = self.constants['turn_in_place_duration']
        self.straight_duration = self.constants['straight_duration']
        self.started_action = None

        # Crosswalk variables
        self.crosswalk_detected = False
        self.crosswalk_threshold_area = self.constants['crosswalk_threshold_area'] # minimum area of blue to detect

        # Stop variables
        self.last_stop_time = None # last time we stopped
        self.stop_cooldown = 2 # how long should we wait after detecting a stop sign to detect another
        self.stop_duration = 3 # how long to stop for
        self.stop_threshold_area = self.constants['stop_threshold_area'] # minimum area of red to stop at

        # Bot detection
        self.bot_detected = False
        # Number of dots in the pattern, two elements: [number of columns, number of rows]
        self.circlepattern_dims = [7, 3]
        # Parameters for the blob detector
        # passed to `SimpleBlobDetector <https://docs.opencv.org/4.3.0/d0/d7a/classcv_1_1SimpleBlobDetector.html>`
        params = cv2.SimpleBlobDetector_Params()
        params.minArea = 10
        params.minDistBetweenBlobs = 3 # changed from 2 to 3 so closer
        self.simple_blob_detector = cv2.SimpleBlobDetector_create(params)

        self.bot_threshold_dist = self.constants['bot_threshold_dist']
        self.bot_threshold_area = self.constants['bot_threshold_area']
        self.switch_duration = self.constants['switch_duration']

        # Parking lot variables
        self.near_stall_distance = self.constants['near_stall_distance'] # metres
        self.far_stall_distance = self.constants['far_stall_distance'] # metres
        self.opposite_at_distance = self.constants['opposite_at_distance'] # metres
        self.clockwise = 'CLOCKWISE'
        self.counterclockwise = 'COUNTERCLOCKWISE'

        self.FORWARD_PARKING = self.constants['FORWARD_PARKING']
        self.BOT_DEBUG = self.constants['BOT_DEBUG']

        # Apriltag detection timer
        if not AT_SYNCHRONOUS:
            self.apriltag_hz = 3
            self.timer = rospy.Timer(rospy.Duration(1 / self.apriltag_hz), self.cb_detect_apriltag)

        self.stage = 1
        self.loginfo("Initialized")

    def img_callback(self, msg):
        self.image_msg = msg

        if not self.image_msg or not CALLBACK_PROCESSING:
            return

        img = self.jpeg.decode(msg.data) # decode image

        self.detect_lane(img) # detect lane

        if self.stage in [1, 3]:
            self.detect_intersection(img) #detect intersection

        if self.stage == 2:
            self.detect_crosswalk(img) # detect crosswalk
            self.detect_bot() # detect robot

    def run(self):
        self.stage1()
        self.stage2()
        self.stage3()
        self.stop()
        rospy.signal_shutdown("Program terminating.")

    def stage1(self):
        rate = rospy.Rate(8)  # 8hz
        while not rospy.is_shutdown() and self.closest_at not in [163, 38]:
            if self.intersection_detected:
                self.intersection_sequence()
            else:
                self.lane_follow()
                rate.sleep()

        self.stage = 2
        self.loginfo("Finished stage 1 :)")

    def stage2(self):
        rate = rospy.Rate(10) # increased from 8 to 10 to prevent jerky movements
        self.velocity = self.constants['stage_2_velocity']
        while not rospy.is_shutdown() and self.closest_at != 38:
            if self.bot_detected:
                self.switch_lanes()
            elif self.crosswalk_detected:
                self.avoid_ducks()
            else:
                self.lane_follow()
                rate.sleep()

        self.stage = 3
        self.loginfo("Finished stage 2!")

    def stage3(self):
        self.velocity = self.constants['stage_3_velocity']
        self.velocity = 0.25
        self.drive_to_intersection()
        self.intersection_sequence()
        rospy.loginfo("Parking in stall no. {}".format(str(self.stall)))
        self.park(self.stall)

    def drive_to_intersection(self):
        # and stop
        rate = rospy.Rate(8)
        while not rospy.is_shutdown() and not self.intersection_detected:
            self.lane_follow()
            rate.sleep()
        self.stop()

    def correct(self):
        # TODO: will be used when the duckiebot stops, before continuing
        # check if yellow contours in frame (or in crop of frame)
        # if not, turn left (for right lane following) and then continue lane following

        # useful after self.straight in check_for_ducks, but not always needed
        return

    def lane_follow(self):
        if self.proportional is None:
            self.twist.omega = 0
            self.twist.v = 0.1 # drive slowly until we see the yellow lines again
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
        self.loginfo(f"Closest apriltag is {self.closest_at}")
        self.stop()
        self.pass_time(self.stop_duration)

        # Determine which direction to go, based on apriltag
        if self.closest_at in [94, 48]:
            self.right_turn()
        elif self.closest_at in [50, 169]:
            self.left_turn()
        elif self.closest_at in [56, 200]:
            self.straight()
        elif self.closest_at == 38:
            pass
        elif self.closest_at in [163, 21]:
            self.avoid_ducks()
        else:
            self.stop()
            self.pass_time(self.stop_duration)

        self.last_stop_time = rospy.get_time()
        self.intersection_detected = False

    def right_turn(self):
        """
        Publish right-angle right turn
        """
        self.loginfo("Turning right")
        twist = Twist2DStamped()
        twist.v = self.turn_speed
        twist.omega = self.constants['right_turn_omega']
        start_time = rospy.get_time()
        rate = rospy.Rate(4)
        while not rospy.is_shutdown() and rospy.get_time() - start_time < self.right_turn_duration:
            self.vel_pub.publish(twist)
            rate.sleep()

    def left_turn(self):
        """
        Publish right-angle left turn
        """
        self.loginfo("Turning left")
        twist = Twist2DStamped()
        twist.v = self.turn_speed
        twist.omega = self.constants['left_turn_omega']
        start_time = rospy.get_time()
        rate = rospy.Rate(4)
        while not rospy.is_shutdown() and rospy.get_time() - start_time < self.left_turn_duration:
            self.vel_pub.publish(twist)
            rate.sleep()

    def stop(self):
        """
        Publish stop command
        """
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)

    def straight(self, linear=None, duration=None):
        """
        Publish straight command
        """
        twist = Twist2DStamped()
        twist.v = self.turn_speed
        if linear is not None:
            twist.omega = linear
        else:
            twist.omega = self.calibration

        self.loginfo("Going straight")
        start_time = rospy.get_time()
        rate = rospy.Rate(8) # changed from 4 to 8
        if duration is None:
            duration = self.straight_duration
        while not rospy.is_shutdown() and rospy.get_time() - start_time < duration:
            self.vel_pub.publish(twist)
            rate.sleep()

    def pass_time(self, t):
        """
        Idle for time t
        """
        start_time = rospy.get_time()
        while not rospy.is_shutdown() and rospy.get_time() < start_time + t:
            continue

    def avoid_ducks(self):
        self.stop() # just to make sure
        rate = rospy.Rate(2)
        self.check_for_ducks()
        # Stay still while ducks are still crossing
        while not rospy.is_shutdown() and self.check_for_ducks():
            self.stop()
            rate.sleep()
        self.pass_time(5) # wait a bit longer after ducks have moved

        self.straight(duration = 2)
        self.crosswalk_detected = False

    def switch_lanes(self):
        # increase velocity to avoid wheels getting stuck
        original_vel = self.velocity

        if SWITCH_LANE_DEBUG:
            rospy.loginfo("Moving close to see if it needs help")
        # keep moving till we're close to the bot
        rate = rospy.Rate(14) # TODO: test with 8
        while not rospy.is_shutdown() and not self.detect_bot_contour() and self.bot_detected:
            self.lane_follow()
            rate.sleep()
        # stop to see if robot needs help
        self.stop()
        self.pass_time(5)

        self.velocity = self.constants['switch_lane_velocity']
        if SWITCH_LANE_DEBUG:
            rospy.loginfo("Switching lanes")
        # Switch lanes
        self.offset = -self.offset
        # left turn with twist.omega = 1.25
        # English lane follow
        rate = rospy.Rate(8)
        start_time = rospy.get_time()
        while not rospy.is_shutdown() and rospy.get_time() - start_time < self.switch_duration:
            if not CALLBACK_PROCESSING:
                self.detect_lane()
            self.lane_follow()
            rate.sleep()
        self.stop()

        # Switch lanes back
        self.offset = -self.offset
        self.bot_detected = False
        if SWITCH_LANE_DEBUG:
            rospy.loginfo("Done switching lanes")

        # switch back to original velocity
        self.velocity = original_vel

    def detect_bot(self):
        self.bot_detected = False
        image_cv = self.bridge.compressed_imgmsg_to_cv2(self.image_msg, "bgr8")
        (detection, centers) = cv2.findCirclesGrid(
            image_cv,
            patternSize=tuple(self.circlepattern_dims),
            flags=cv2.CALIB_CB_SYMMETRIC_GRID,
            blobDetector=self.simple_blob_detector,
        )

        if detection > 0:
            self.bot_detected = True
            if self.BOT_DEBUG:
                rospy.loginfo("Bot detected")

    def detect_bot_contour(self):
        msg = self.image_msg
        if not msg:
            return

        found_robot = False

        img = self.jpeg.decode(msg.data)
        crop = img[100:320, :, :]

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        robot_mask = cv2.inRange(hsv, ROBOT_MASK[0], ROBOT_MASK[1])
        if DEBUG:
            crop = cv2.bitwise_and(crop, crop, mask=robot_mask)

        contours, _ = cv2.findContours(robot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Search for robot
        max_area = self.bot_threshold_area
        max_idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area
                if not DEBUG:
                    break

        if max_idx != -1:
            found_robot = True

        return found_robot

    def apriltag_follow(self, apriltag, direction, distance, euc=False):
        rate = rospy.Rate(8)
        last_error = 0
        last_time = rospy.get_time()

        parking_P = self.constants['parking_P']
        parking_D = self.constants['parking_D']
        parking_forward_velocity = self.constants['parking_forward_velocity']
        parking_backward_velocity = self.constants['parking_backward_velocity']

        while not rospy.is_shutdown():
            x, y, z, theta = self.detect_apriltag_by_id(apriltag)

            if x == 0:
                self.twist.omega = 0
                self.twist.v = 0
                self.vel_pub.publish(self.twist)

                rate.sleep()

                continue

            p_error = 0

            distance_from_tag = z if not euc else math.sqrt(x**2 + z**2)

            if direction == "FORWARD":
                if distance_from_tag <= distance:
                    break
                else:
                    p_error = x
                    self.twist.v = parking_forward_velocity

            elif direction == "REVERSE":
                if distance_from_tag >= distance:
                    break
                else:
                    p_error = -x
                    self.twist.v = -parking_backward_velocity

            # P Term
            P = -p_error * parking_P

            # D Term
            d_error = (p_error - last_error) / (rospy.get_time() - last_time)
            last_error = p_error
            last_time = rospy.get_time()
            D = d_error * parking_D

            self.twist.omega = P + D
            self.vel_pub.publish(self.twist)

            rate.sleep()

        self.stop()

    def park(self, stall):
        # advance into parking lot until perpendicular to desired stall
        target_distance = 0
        if stall == 1 or stall == 3:
            target_distance = self.far_stall_distance
        else:
            target_distance = self.near_stall_distance

        self.apriltag_follow(227, "FORWARD", target_distance)

        # turn the vehicle such that it faces away from the target stall
        at_opposite = None
        turn_direction = None
        at = None
        if stall == 1:
            at = 207
            at_opposite = 228 # stall 3
            turn_direction = self.counterclockwise

        elif stall == 2:
            at = 226
            at_opposite = 75 # stall 4
            turn_direction = self.counterclockwise

        elif stall == 3:
            at = 228
            at_opposite = 207 # stall 1
            turn_direction = self.clockwise

        elif stall == 4:
            at = 75
            at_opposite = 226 # stall 2
            turn_direction = self.clockwise

        self.face_apriltag(turn_direction, at)

        if self.FORWARD_PARKING:
            # advance forward to stall
            self.apriltag_follow(at, "FORWARD", self.constants['forward_distance_from_at'])
        else:
            # advance forward to stall
            self.apriltag_follow(at, "FORWARD", self.constants['backward_distance_from_at'])

            # Turn backwards
            self.face_apriltag(turn_direction, at_opposite)

            # reverse into parking stall
            self.apriltag_follow(at_opposite, "REVERSE", self.constants['opposite_at_distance'], True)

    def reverse_to_stall(self, at_opposite):
        self.twist.v = -self.velocity
        self.twist.omega = -self.calibration
        rate = rospy.Rate(8)
        while not rospy.is_shutdown() and self.detect_apriltag_by_id(at_opposite)[3] < self.opposite_at_distance:
            self.vel_pub.publish(self.twist)
            rate.sleep()

        self.stop()

    def face_apriltag(self, turn_direction, apriltag):
        """
        Turn until apriltag is in center of image
        """
        self.loginfo(f"Turning to face apriltag {apriltag}")
        rate = rospy.Rate(2)

        angular_vel = self.constants['angular_vel']

        self.twist.v = 0
        while not rospy.is_shutdown() and self.detect_apriltag_by_id(apriltag)[0] <= 0:
            self.twist.omega = angular_vel if turn_direction == self.counterclockwise else -angular_vel
            self.vel_pub.publish(self.twist)

            rate.sleep()

            self.twist.omega = 0
            self.vel_pub.publish(self.twist)

    def detect_lane(self, img):
        """
        Detect the lane to get P for lane following
        """
        crop = img[300:-1, :, :]
        crop_width = crop.shape[1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, YELLOW_MASK[0], YELLOW_MASK[1])

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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
            rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
            self.pub_mask.publish(rect_img_msg)

    def detect_intersection(self, img):
        # Don't detect if we recently detected an intersection
        if self.last_stop_time and rospy.get_time() - self.last_stop_time < self.stop_cooldown:
            return

        # Mask for stop lines
        crop = img[400:-1, :, :]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        stopMask1 = cv2.inRange(hsv, STOP_MASK1[0], STOP_MASK1[1])
        stopMask2 = cv2.inRange(hsv, STOP_MASK2[0], STOP_MASK2[1])
        stopMask = cv2.bitwise_or(stopMask2, stopMask1)

        stopContours, _ = cv2.findContours(stopMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        max_area = self.stop_threshold_area
        max_idx = -1
        for i in range(len(stopContours)):
            area = cv2.contourArea(stopContours[i])
            if area > max_area:
                max_idx = i
                max_area = area
                if not DEBUG:
                    break

        if max_idx == -1:
            return

        self.intersection_detected = True

        if DEBUG:
            M = cv2.moments(stopContours[max_idx])
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.drawContours(crop, stopContours, max_idx, (0, 255, 0), 3)
                cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass

            rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
            self.pub_mask.publish(rect_img_msg)

    def detect_crosswalk(self, img):
        # Mask for blue lines
        crop = img[400:-1, :, :]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, BLUE_MASK[0], BLUE_MASK[1])

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Search for nearest blue line
        max_area = self.crosswalk_threshold_area
        max_idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area
                if not DEBUG:
                    break

        if max_idx == -1:
            return

        self.crosswalk_detected = True

        if DEBUG:
            M = cv2.moments(contours[max_idx])
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass

            rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
            self.pub_mask.publish(rect_img_msg)

    def detect_apriltag_by_id(self, apriltag):
        # Returns the x, y, z coordinate of a specific apriltag in metres, and its pitch in radians
        img_msg = self.image_msg
        if not img_msg:
            return (0, 0, 0, 0)

        cv_image = None
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(img_msg)
        except CvBridgeError as e:
            self.log(e)
            return (0, 0, 0, 0)

        # undistort the image
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(
            self.K, self.DC, (self.w, self.h), 0, (self.w, self.h))
        image_np = cv2.undistort(cv_image, self.K, self.DC, None, newcameramtx)

        # convert the image to black and white
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # detect tags present in image
        tags = self.at_detector_2.detect(
            image_gray, estimate_tag_pose=True, camera_params=self.camera_params, tag_size=0.065)

        if len(tags) == 0:
            return (0, 0, 0, 0)

        for tag in tags:
            if tag.tag_id == apriltag:
                theta = np.arctan2(-tag.pose_R[2][0], np.sqrt(tag.pose_R[2][1]**2 + tag.pose_R[2][2]**2))
                if DEBUG:
                    for i in range(len(tag.corners)):
                        point_x = tuple(tag.corners[i-1, :].astype(int))
                        point_y = tuple(tag.corners[i, :].astype(int))
                        cv2.line(image_np, point_x, point_y, (0, 255, 0), 5)
                    rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(image_np))
                    self.pub_mask.publish(rect_img_msg)
                return (tag.pose_t[0][0], tag.pose_t[1][0], tag.pose_t[2][0], theta)
        
        return (0, 0, 0, 0)

    def cb_detect_apriltag(self, _):
        if self.image_msg is None:
            return False
        img_msg = self.image_msg
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

        if closest and closest.tag_id in self.apriltags:
            self.at_distance = closest.pose_t[2][0]
            self.closest_at = closest.tag_id
            self.at_detected = True
        else:
            self.at_detected = False

    def check_for_ducks(self):
        msg = self.image_msg
        if not msg:
            return

        found_ducks = False

        crop = self.jpeg.decode(msg.data)
        crop = crop[260:-1, :, :]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        yellow_mask = cv2.inRange(hsv, DUCK_MASK[0], DUCK_MASK[1])
        orange_mask = cv2.inRange(hsv, ORANGE_MASK[0], ORANGE_MASK[1])
        orange_mask2 = cv2.inRange(hsv, ORANGE_MASK2[0], ORANGE_MASK2[1])
        combined_orange_mask = cv2.bitwise_or(orange_mask2, orange_mask)
        combined_mask = cv2.bitwise_or(combined_orange_mask, yellow_mask)
        crop = cv2.bitwise_and(crop, crop, mask=combined_mask)

        if DUCK_DEBUG:
            rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
            self.pub_mask.publish(rect_img_msg)

        yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        orange_contours, _ = cv2.findContours(combined_orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Search for nearest ducks
        min_area = 50
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
                            
                            break
                    except:
                        pass

        # debugging
        if DUCK_DEBUG:
            rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
            self.pub_mask.publish(rect_img_msg)
            if found_ducks:
                rospy.loginfo("Ducks detected.")
            else:
                rospy.loginfo("Ducks not detected.")

        return found_ducks

    def set_LEDs(self, on = True):
        '''
        Code for this function was inspired by
        "duckietown/dt-core", file "led_emitter_node.py"
        Link: https://github.com/duckietown/dt-core/blob/daffy/packages/led_emitter/src/led_emitter_node.py
        Author: GitHub user liampaull
        '''
        self.pattern.header.stamp = rospy.Time.now()
        rgba = ColorRGBA()

        value = 1.0 if on else 0.0

        # All white
        rgba.r = value
        rgba.g = value
        rgba.b = value
        rgba.a = value

        self.pattern.rgb_vals = [rgba] * 5

        for _ in range(5):
            self.led_publisher.publish(self.pattern)

    def hook(self):
        print("SHUTTING DOWN")
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        for _ in range(8):
            self.vel_pub.publish(self.twist)
        if self.veh != "csc22906":
            self.set_LEDs(False)

    def readYamlFile(self, fname):
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file, Loader=yaml.FullLoader)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         % (fname, exc), type='fatal')
                rospy.signal_shutdown()
                return

if __name__ == "__main__":
    node = DriverNode("driver_node")
    node.run()
    rospy.spin()
