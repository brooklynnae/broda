#! /usr/bin/env python3

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class TimeTrialsBot():
    def __init__(self):
        rospy.init_node('time_trials_bot')
        
        self.vel_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
        self.score_pub = rospy.Publisher('/score_tracker', String, queue_size=1)

    def run(self):
        start_timer = String()
        start_timer.data = "Broda,adorb,0,NA"
        self.score_pub.publish(start_timer)

        move = Twist()
        move.linear.x = 0.5
        self.vel_pub.publish(move)

        rospy.sleep(10)

        move.linear.x = 0
        self.vel_pub.publish(move)

        end_timer = String()
        end_timer.data = "Broda,adorb,-1,NA"
        self.score_pub.publish(end_timer)

if __name__ == '__main__':
    try:
        my_bot = TimeTrialsBot()
        rospy.sleep(1)
        my_bot.run()
    except rospy.ROSInterruptException:
        pass

