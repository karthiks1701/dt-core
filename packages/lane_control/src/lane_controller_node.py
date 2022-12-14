#!/usr/bin/env python3
import numpy as np
import math
import rospy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Twist, Pose, Point, Vector3, TransformStamped, Transform
from duckietown_msgs.msg import WheelEncoderStamped
import message_filters
from tf2_ros import TransformBroadcaster
from tf import transformations as tr

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import (
    Twist2DStamped,
    LanePose,
    WheelsCmdStamped,
    BoolStamped,
    FSMState,
    StopLineReading,
)

from lane_controller.controller import LaneController


class LaneControllerNode(DTROS):
    """Computes control action.
    The node compute the commands in form of linear and angular velocities, by processing the estimate error in
    lateral deviationa and heading.
    The configuration parameters can be changed dynamically while the node is running via ``rosparam set`` commands.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use
    Configuration:
        ~v_bar (:obj:`float`): Nominal velocity in m/s
        ~k_d (:obj:`float`): Proportional term for lateral deviation
        ~k_theta (:obj:`float`): Proportional term for heading deviation
        ~k_Id (:obj:`float`): integral term for lateral deviation
        ~k_Iphi (:obj:`float`): integral term for lateral deviation
        ~d_thres (:obj:`float`): Maximum value for lateral error
        ~theta_thres (:obj:`float`): Maximum value for heading error
        ~d_offset (:obj:`float`): Goal offset from center of the lane
        ~integral_bounds (:obj:`dict`): Bounds for integral term
        ~d_resolution (:obj:`float`): Resolution of lateral position estimate
        ~phi_resolution (:obj:`float`): Resolution of heading estimate
        ~omega_ff (:obj:`float`): Feedforward part of controller
        ~verbose (:obj:`bool`): Verbosity level (0,1,2)
        ~stop_line_slowdown (:obj:`dict`): Start and end distances for slowdown at stop lines

    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        ~lane_pose (:obj:`LanePose`): The lane pose estimate from the lane filter
        ~intersection_navigation_pose (:obj:`LanePose`): The lane pose estimate from intersection navigation
        ~wheels_cmd_executed (:obj:`WheelsCmdStamped`): Confirmation that the control action was executed
        ~stop_line_reading (:obj:`StopLineReading`): Distance from stopline, to reduce speed
        ~obstacle_distance_reading (:obj:`stop_line_reading`): Distancefrom obstacle virtual stopline, to reduce speed
    """

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        # Add the node parameters to the parameters dictionary
        # TODO: MAKE TO WORK WITH NEW DTROS PARAMETERS
        self.params = dict()
        self.params["~v_bar"] = DTParam("~v_bar", param_type=ParamType.FLOAT, min_value=0.0, max_value=5.0)
        self.params["~k_d"] = DTParam("~k_d", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0)
        self.params["~k_theta"] = DTParam(
            "~k_theta", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0
        )
        self.params["~k_Id"] = DTParam("~k_Id", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0)
        self.params["~k_Iphi"] = DTParam(
            "~k_Iphi", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0
        )
        self.params["~theta_thres"] = rospy.get_param("~theta_thres", None)
        self.params["~d_thres"] = rospy.get_param("~d_thres", None)
        self.params["~d_offset"] = rospy.get_param("~d_offset", None)
        self.params["~integral_bounds"] = rospy.get_param("~integral_bounds", None)
        self.params["~d_resolution"] = rospy.get_param("~d_resolution", None)
        self.params["~phi_resolution"] = rospy.get_param("~phi_resolution", None)
        self.params["~omega_ff"] = rospy.get_param("~omega_ff", None)
        self.params["~verbose"] = rospy.get_param("~verbose", None)
        self.params["~stop_line_slowdown"] = rospy.get_param("~stop_line_slowdown", None)

        # Need to create controller object before updating parameters, otherwise it will fail
        self.controller = LaneController(self.params)
        # self.updateParameters() # TODO: This needs be replaced by the new DTROS callback when it is implemented

        # Initialize variables
        self.fsm_state = None
        self.wheels_cmd_executed = WheelsCmdStamped()
        self.pose_msg = LanePose()
        self.pose_initialized = False
        self.pose_msg_dict = dict()
        self.last_s = None
        self.stop_line_distance = None
        self.stop_line_detected = False
        self.at_stop_line = False
        self.obstacle_stop_line_distance = None
        self.obstacle_stop_line_detected = False
        self.at_obstacle_stop_line = False

        self.current_pose_source = "intersection_navigation"

        # Construct publishers
        self.pub_car_cmd = rospy.Publisher(
            "car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1, dt_topic_type=TopicType.CONTROL
        )

        # Construct subscribers
        self.sub_lane_reading = rospy.Subscriber(
            "~lane_pose", LanePose, self.cbAllPoses, "lane_filter", queue_size=1
        )
        self.sub_intersection_navigation_pose = rospy.Subscriber(
            "~intersection_navigation_pose",
            LanePose,
            self.cbAllPoses,
            "intersection_navigation",
            queue_size=1,
        )
        self.sub_wheels_cmd_executed = rospy.Subscriber(
            "~wheels_cmd", WheelsCmdStamped, self.cbWheelsCmdExecuted, queue_size=1
        )
        self.sub_stop_line = rospy.Subscriber(
            "~stop_line_reading", StopLineReading, self.cbStopLineReading, queue_size=1
        )
        self.sub_obstacle_stop_line = rospy.Subscriber(
            "~obstacle_distance_reading", StopLineReading, self.cbObstacleStopLineReading, queue_size=1
        )
        
        # introducing deadreckoning
        self.state = 0
        self.left_encoder_last = None
        self.right_encoder_last = None
        self.encoders_timestamp_last = None
        self.encoders_timestamp_last_local = None
        self.timestamp = None
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.yaw = 0.0
        self.q = [0.0, 0.0, 0.0, 1.0]
        self.tv = 0.0
        self.rv = 0.0

        self.ticks_per_meter = 656.0
        self.debug = False
        self.wierd = True
        self.wheelbase = 0.108
        self.iter_ = 0
        
        #NOTE
        self.target_states = np.array([  [0.09,0.0], [0.12,0.0], [0.15,0.0], [0.3,0.2], [0.3,0.3], [0.3,0.4] ,  [0.3,0.5]  ]) #---  left turn
        #self.target_states = np.array([  [0.09,0.0], [0.15,0.0], [0.15,-0.15], [0.15,-0.25], [0.15,-0.30], [0.15, -0.4]  ]) #-- right turn
        #self.target_states =  np.array([  [0.1,0.0], [0.15,0.0], [0.2,0.0], [0.25,0.0] ]) #-- straoigth 
        self.len_states = self.target_states.shape[0]
        print(self.len_states)
        self.final_state = 0 
        
        # subscirber
        self.sub_encoder_left = message_filters.Subscriber("/korra/left_wheel_encoder_node/tick", WheelEncoderStamped)
        self.sub_encoder_right = message_filters.Subscriber("/korra/right_wheel_encoder_node/tick", WheelEncoderStamped)
        
        self.ts_encoders = message_filters.ApproximateTimeSynchronizer(
            [self.sub_encoder_left, self.sub_encoder_right], 1, 1
        )
            
        #if path is created then only check for this 
        # IF 
        #self.callback(0,0)
        self.alpha = 0.0
        self.beta = 0.0
        
        self.log("Initialized controller")
        
    def reset_odometry(self):
        self.state = 0
        self.left_encoder_last = None
        self.right_encoder_last = None
        self.encoders_timestamp_last = None
        self.encoders_timestamp_last_local = None
        self.timestamp = None
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.yaw = 0.0
        self.q = [0.0, 0.0, 0.0, 1.0]
        self.tv = 0.0
        self.rv = 0.0

        self.ticks_per_meter = 656.0
        self.debug = False
        self.wierd = True
        self.wheelbase = 0.108
        self.iter_ = 0
        self.final_state = 0
    
    def setwaypoint(self, waypoint):
        self.target_states = waypoint
    
    def cb_ts_encoders(self, left_encoder, right_encoder):
        timestamp_now = rospy.get_time()

        # Use the average of the two encoder times as the timestamp
        left_encoder_timestamp = left_encoder.header.stamp.to_sec()
        right_encoder_timestamp = right_encoder.header.stamp.to_sec()
        timestamp = (left_encoder_timestamp + right_encoder_timestamp) / 2

        if not self.left_encoder_last:
            self.left_encoder_last = left_encoder
            self.right_encoder_last = right_encoder
            self.encoders_timestamp_last = timestamp
            self.encoders_timestamp_last_local = timestamp_now
            return

        # Skip this message if the time synchronizer gave us an older message
        dtl = left_encoder.header.stamp - self.left_encoder_last.header.stamp
        dtr = right_encoder.header.stamp - self.right_encoder_last.header.stamp
        if dtl.to_sec() < 0 or dtr.to_sec() < 0:
            self.loginfo("Ignoring stale encoder message")
            return

        left_dticks = left_encoder.data - self.left_encoder_last.data
        right_dticks = right_encoder.data - self.right_encoder_last.data

        left_distance = left_dticks * 1.0 / self.ticks_per_meter
        right_distance = right_dticks * 1.0 / self.ticks_per_meter

        # Displacement in body-relative x-direction
        distance = (left_distance + right_distance) / 2

        # Change in heading
        dyaw = (right_distance - left_distance) / self.wheelbase

        dt = timestamp - self.encoders_timestamp_last

        if dt < 1e-6:
            #self.logwarn("Time since last encoder message (%f) is too small. Ignoring" % dt)
            dt = 1e-6
            #return

        self.tv = distance / dt
        self.rv = dyaw / dt

        if self.debug:
            self.loginfo(
                "Left wheel:\t Time = %.4f\t Ticks = %d\t Distance = %.4f m"
                % (left_encoder.header.stamp.to_sec(), left_encoder.data, left_distance)
            )

            self.loginfo(
                "Right wheel:\t Time = %.4f\t Ticks = %d\t Distance = %.4f m"
                % (right_encoder.header.stamp.to_sec(), right_encoder.data, right_distance)
            )

            self.loginfo(
                "TV = %.2f m/s\t RV = %.2f deg/s\t DT = %.4f" % (self.tv, self.rv * 180 / math.pi, dt)
            )

        dist = self.tv * dt
        dyaw = self.rv * dt

        self.yaw = self.angle_clamp(self.yaw + dyaw)
        self.x = self.x + dist * math.cos(self.yaw)
        self.y = self.y + dist * math.sin(self.yaw)
        self.q = tr.quaternion_from_euler(0, 0, self.yaw)
        self.timestamp = timestamp

        self.left_encoder_last = left_encoder
        self.right_encoder_last = right_encoder
        self.encoders_timestamp_last = timestamp
        self.encoders_timestamp_last_local = timestamp_now

        if self.wierd and self.final_state == 0:
            print("iterator ",self.iter_)
            start_time = rospy.Time.now()
            
            self.state = 0

            car_control_msg = Twist2DStamped()
            car_control_msg.header.stamp = rospy.Time.now()
            car_control_msg.header.seq = 0

        # Add commands to car message
            #vpx = 0.5*(self.target_states[self.iter_][0]-self.x - self.alpha)
            #vpy = 0.06*(self.target_states[self.iter_][1]-self.y)
            car_control_msg.v = 0.12
            car_control_msg.omega = self.compute_omega(self.target_states[self.iter_],self.x,self.y,self.yaw,dt)
            #= vpy/self.alpha
            
                
            print( " car commands  ",car_control_msg.omega)
            print("car control given",car_control_msg)

            if self.check_point( np.array([self.x,self.y]),self.target_states[self.iter_] ):
                    self.iter_ += 1
                    car_control_msg.v = 0.0

                    if self.iter_ == len(self.target_states):
                        self.final_state = 1 # DONE
                        car_control_msg.v = 0
                        car_control_msg.omega = 0
                        self.publishCmd(car_control_msg)
                        #self.reset_odometry()
                        return 

            print("target_state_ ",self.target_states[self.iter_])
            self.publishCmd(car_control_msg)
            print("cur state ",self.x,self.y,self.yaw)
            self.state = 1


    def cbObstacleStopLineReading(self, msg):
        """
        Callback storing the current obstacle distance, if detected.

        Args:
            msg(:obj:`StopLineReading`): Message containing information about the virtual obstacle stopline.
        """
        self.obstacle_stop_line_distance = np.sqrt(msg.stop_line_point.x ** 2 + msg.stop_line_point.y ** 2)
        self.obstacle_stop_line_detected = msg.stop_line_detected
        self.at_stop_line = msg.at_stop_line

    def cbStopLineReading(self, msg):
        """Callback storing current distance to the next stopline, if one is detected.

        Args:
            msg (:obj:`StopLineReading`): Message containing information about the next stop line.
        """
        self.stop_line_distance = np.sqrt(msg.stop_line_point.x ** 2 + msg.stop_line_point.y ** 2)
        self.stop_line_detected = msg.stop_line_detected
        self.at_obstacle_stop_line = msg.at_stop_line

    def cbMode(self, fsm_state_msg):

        self.fsm_state = fsm_state_msg.state  # String of current FSM state

        rospy.loginfo(" We received an fsm state %s", fsm_state_msg.state)

        if self.fsm_state == "INTERSECTION_CONTROL":
            self.current_pose_source = "intersection_navigation"
        else:
            self.current_pose_source = "lane_filter"

        if self.params["~verbose"] == 2:
            self.log("Pose source: %s" % self.current_pose_source)

    def cbAllPoses(self, input_pose_msg, pose_source):
        """Callback receiving pose messages from multiple topics.

        If the source of the message corresponds with the current wanted pose source, it computes a control command.

        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
            pose_source (:obj:`String`): Source of the message, specified in the subscriber.
        """

        if pose_source == self.current_pose_source:
            self.pose_msg_dict[pose_source] = input_pose_msg

            self.pose_msg = input_pose_msg

            self.getControlAction(self.pose_msg)

    def cbWheelsCmdExecuted(self, msg_wheels_cmd):
        """Callback that reports if the requested control action was executed.

        Args:
            msg_wheels_cmd (:obj:`WheelsCmdStamped`): Executed wheel commands
        """
        self.wheels_cmd_executed = msg_wheels_cmd

    def publishCmd(self, car_cmd_msg):
        """Publishes a car command message.

        Args:
            car_cmd_msg (:obj:`Twist2DStamped`): Message containing the requested control action.
        """
        #rospy.loginfo("Commands car sent %s, %s", str(car_cmd_msg.v), str(car_cmd_msg.omega))
        self.pub_car_cmd.publish(car_cmd_msg)

    def getControlAction(self, pose_msg):
        """Callback that receives a pose message and updates the related control command.

        Using a controller object, computes the control action using the current pose estimate.

        Args:
            pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """
        current_s = rospy.Time.now().to_sec()
        dt = None
        if self.last_s is not None:
            dt = current_s - self.last_s

        if self.at_stop_line or self.at_obstacle_stop_line:
            v = 0
            omega = 0
        else:

            # Compute errors
            d_err = pose_msg.d - self.params["~d_offset"]
            phi_err = pose_msg.phi

            # We cap the error if it grows too large
            if np.abs(d_err) > self.params["~d_thres"]:
                self.log("d_err too large, thresholding it!", "error")
                d_err = np.sign(d_err) * self.params["~d_thres"]

            wheels_cmd_exec = [self.wheels_cmd_executed.vel_left, self.wheels_cmd_executed.vel_right]
            if self.obstacle_stop_line_detected:
                v, omega = self.controller.compute_control_action(
                    d_err, phi_err, dt, wheels_cmd_exec, self.obstacle_stop_line_distance
                )
                # TODO: This is a temporarily fix to avoid vehicle image detection latency caused unable to stop in time.
                v = v * 0.25
                omega = omega * 0.25

            else:
                #v, omega = self.controller.compute_control_action(
                #    d_err, phi_err, dt, wheels_cmd_exec, self.stop_line_distance
                #)
                self.ts_encoders.registerCallback(self.cb_ts_encoders)

            # For feedforward action (i.e. during intersection navigation)
            self.params["~omega_ff"] = rospy.get_param("~omega_ff", None)
            #rospy.loginfo("current computed omega %s sent omega_ff %s",str(omega), str(self.params["~omega_ff"]))
            #omega += self.params["~omega_ff"]

        # Initialize car control msg, add header from input message
        car_control_msg = Twist2DStamped()
        car_control_msg.header = pose_msg.header

        # Add commands to car message
        #car_control_msg.v = v
        #car_control_msg.omega = omega

        #self.publishCmd(car_control_msg)
        self.last_s = current_s

    def cbParametersChanged(self):
        """Updates parameters in the controller object."""

        self.controller.update_parameters(self.params)
    
    @staticmethod
    def angle_clamp(theta):
        if theta > 2 * math.pi:
            return theta - 2 * math.pi
        elif theta < -2 * math.pi:
            return theta + 2 * math.pi
        else:
            return theta      

    def path_plan(self,obstacle,lane):
            return 0

    def compute_omega(self,targetxy,x,y,current,dt):
        factor = 1 # PARAM 

        print("compute omega targetxy", targetxy)
        print("compute omega current",x,y)

        target_yaw = np.arctan2( (targetxy[1] - y),(targetxy[0]- x) )

        print("target yaw ", np.rad2deg(target_yaw))
        print("currnt_yaw ", np.rad2deg(current))

        omega = factor* ((target_yaw - current))

        return omega

    def check_point(self,current_point,target_point):
        threshold = 0.08
        threshold_x = 0.05
        dist_x = np.zeros((1,2))
        dist_x[0,0] = (current_point[0]-self.alpha) - target_point[0]
        dist_x[0,1] = (current_point[1]) - target_point[1]
        dist = np.sqrt(((current_point[0]-self.alpha) - target_point[0])**2 + ((current_point[1]-self.alpha) - target_point[1])**2 )

        if self.iter_ == (self.len_states - 1):

            if abs(dist) < threshold_x:
                return True

            return False

        else:
            print("check if we reached ")
            print("dist ",dist)
            print(" current and target point",current_point,target_point)
            print("-"*10)

            if (abs(dist_x[0,0])) < threshold_x or (dist) < threshold:
                return True

            return False


if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name="lane_controller_node")
    # Keep it spinning
    rospy.spin()
