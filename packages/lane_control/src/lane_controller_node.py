#!/usr/bin/env python
import math
import numpy as np
import rospy

from duckietown import DTROS, DTPublisher, DTSubscriber
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading

from lane_controller.controller import LaneController


class LaneControllerNode(DTROS):
    """Computes control action.
    The node compute the commands in form of linear and angular velocities, by processing the estimate error in
    lateral deviationa and heading.
    The configuration parameters can be changed dynamically while the node is running via `rosparam set` commands.
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
        ~velocity_to_m_per_s (:obj:`float`): Conversion factor
        ~omega_to_rad_per_s (:obj:`float`): Conversion factor
        ~integral_bounds (:obj:`dict`): Bounds for integral term
        ~d_resolution (:obj:`float`): Resolution of lateral position estimate
        ~phi_resolution (:obj:`float`): Resolution of heading estimate
        ~omega_ff (:obj:`float`): Feedforward part of controller
        ~verbose (:obj:`bool`): Verbosity level (0,1,2)

    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        ~lane_pose (:obj:`LanePose`): The lane pose estimate from the lane filter
        ~intersection_navigation_pose (:obj:`LanePose`): The lane pose estimate from intersection navigation
        ~wheels_cmd_executed (:obj:`WheelsCmdStamped`): Confirmation that the control action was executed
        ~stop_line_reading (:obj:`StopLineReading`): Distance from stopline, to reduce speed
    """

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(node_name=node_name)

        # Add the node parameters to the parameters dictionary
        self.parameters['~v_bar'] = None
        self.parameters['~k_d'] = None
        self.parameters['~k_theta'] = None
        self.parameters['~k_Id'] = None
        self.parameters['~k_Iphi'] = None
        self.parameters['~theta_thres'] = None
        self.parameters['~d_thres'] = None
        self.parameters['~d_offset'] = None
        self.parameters['~velocity_to_m_per_s'] = None
        self.parameters['~omega_to_rad_per_s'] = None
        self.parameters['~integral_bounds'] = None
        self.parameters['~d_resolution'] = None
        self.parameters['~phi_resolution'] = None
        self.parameters['~omega_ff'] = None
        self.parameters['~verbose'] = None

        # Need to create controller object before updating parameters, otherwise it will fail
        self.controller = LaneController(self.parameters)
        self.cbParametersChanged()

        # Initialize variables
        self.fsm_state = None
        self.wheels_cmd_executed = WheelsCmdStamped()
        self.pose_msg = LanePose()
        self.pose_initialized = False
        self.pose_msg_dict = dict()
        self.last_s = None
        self.stop_line_distance = None
        self.stop_line_detected = False

        self.current_pose_source = 'lane_filter'

        # Construct publishers
        self.pub_car_cmd = DTPublisher("~car_cmd", Twist2DStamped, queue_size=1)

        # Construct subscribers
        self.sub_lane_reading = DTSubscriber("~lane_pose",
                                             LanePose,
                                             self.cbAllPoses,
                                             "lane_filter",
                                             queue_size=1)
        self.sub_intersection_navigation_pose = DTSubscriber("~intersection_navigation_pose",
                                                             LanePose,
                                                             self.cbAllPoses,
                                                             "intersection_navigation",
                                                             queue_size=1)
        self.sub_wheels_cmd_executed = DTSubscriber("~wheels_cmd_executed",
                                                    WheelsCmdStamped,
                                                    self.cbWheelsCmdExecuted,
                                                    queue_size=1)
        self.sub_stop_line = rospy.Subscriber("~stop_line_reading",
                                              StopLineReading,
                                              self.cbStopLineReading,
                                              queue_size=1)
        self.log("Initialized!")

    def cbStopLineReading(self, msg):
        """Callback storing current distance to the next stopline, if one is detected.

        Args:
            msg (:obj:`StopLineReading`): Message containing information about the next stop line.
        """
        self.stop_line_distance = np.sqrt(msg.stop_line_point.x**2 + msg.stop_line_point.y**2 + msg.stop_line_point.z**2)
        self.stop_line_detected = msg.stop_line_detected

    def cbMode(self, fsm_state_msg):

        self.fsm_state = fsm_state_msg.state    # String of current FSM state

        if self.fsm_state == 'INTERSECTION_CONTROL':
            self.current_pose_source = 'intersection_navigation'
        else:
            self.current_pose_source = 'lane_filter'

        if self.parameters['~verbose'] == 2:
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
        self.pub_car_cmd.publish(car_cmd_msg)

    def getControlAction(self, pose_msg):
        """Callback that receives a pose message and updates the related control command.

        Using a controller object, computes the control action using the current pose estimate.

        Args:
            pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """

        # Compute errors
        d_err = pose_msg.d - self.parameters['~d_offset']
        phi_err = pose_msg.phi

        # We cap the error if it grows too large
        if math.fabs(d_err) > self.parameters['~d_thres']:
            self.log("d_err too large, thresholding it!", 'error')
            d_err = np.sign(d_err) * self.parameters['~d_thres']

        current_s = rospy.Time.now().to_sec()

        dt = None
        if self.last_s is not None:
            dt = (current_s - self.last_s)

        wheels_cmd_exec = [self.wheels_cmd_executed.vel_left, self.wheels_cmd_executed.vel_right]
        v, omega = self.controller.compute_control_action(d_err, phi_err, dt, wheels_cmd_exec, self.stop_line_distance)

        # For feedforward action (i.e. during intersection navigation)
        omega += self.parameters['~omega_ff']

        # Initialize car control msg, add header from input message
        car_control_msg = Twist2DStamped()
        car_control_msg.header = pose_msg.header

        # Add commands to car message
        car_control_msg.v = v * self.parameters['~velocity_to_m_per_s']
        car_control_msg.omega = omega

        self.publishCmd(car_control_msg)
        self.last_s = current_s

    def cbParametersChanged(self):
        """Updates parameters in the controller object."""
        super(LaneControllerNode, self).cbParametersChanged()
        self.controller.update_parameters(self.parameters)


if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name='lane_controller_node')
    # Keep it spinning
    rospy.spin()
