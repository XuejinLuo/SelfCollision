import numpy as np

class BoundsCalculator:
    def __init__(self, cycletime):
        """
        初始化 BoundsCalculator 类。

        :param cycletime: 周期时间
        """
        self.joint_max_position = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float64)
        self.joint_min_position = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float64)
        self.jerk_limit = np.array([7500, 3750, 5000, 6250, 7500, 10000, 10000], dtype=np.float64)
        self.acc_limit = np.array([15, 7.5, 10, 12.5, 15, 20, 20], dtype=np.float64)
        self.vel_limit = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100], dtype=np.float64)
        self.cycletime = cycletime

        # 根据周期时间调整限制
        self.vel_limit *= cycletime * 0.001  # rad/20ms
        self.acc_limit *= cycletime * cycletime * 0.001 * 0.001  # rad/20ms^2
        self.jerk_limit *= cycletime * cycletime * cycletime * 0.001 * 0.001 * 0.001  # rad/20ms^3

    def get_bounded_value(self, value, bound):
        """
        确保值在给定的界限内。

        :param value: 输入的值
        :param bound: 界限值
        :return: 被限制在界限内的值
        """
        if value > bound:
            return bound
        if value < -bound:
            return -bound
        return value

    def calculate_bounds(self, new_robot_actual_angle, old_robot_actual_angle, prev_vel_robot1=0, prev_vel_robot2=0, prev_acc_robot1=0, prev_acc_robot2=0):
        """
        计算并返回 bounds。

        :param new_robot_actual_angle: 机器人 1 的当前实际角度
        :param old_robot_actual_angle: 机器人 2 的当前实际角度
        :param prev_vel_robot1: 机器人 1 上一时刻的速度
        :param prev_vel_robot2: 机器人 2 上一时刻的速度
        :param prev_acc_robot1: 机器人 1 上一时刻的加速度
        :param prev_acc_robot2: 机器人 2 上一时刻的加速度
        :return: 计算得到的 bounds
        """
        robot1_bounds = []
        robot2_bounds = []
        for i in range(len(self.joint_max_position)):
            limited_vel_robot1 = self.get_bounded_value(prev_vel_robot1, self.vel_limit[i])
            limited_vel_robot2 = self.get_bounded_value(prev_vel_robot2, self.vel_limit[i])
            limited_acc_robot1 = self.get_bounded_value(prev_acc_robot1, self.acc_limit[i])
            limited_acc_robot2 = self.get_bounded_value(prev_acc_robot2, self.acc_limit[i])

            min_step_robot1 = limited_vel_robot1 + limited_acc_robot1 - self.jerk_limit[i]
            max_step_robot1 = limited_vel_robot1 + limited_acc_robot1 + self.jerk_limit[i]
            min_step_robot2 = limited_vel_robot2 + limited_acc_robot2 - self.jerk_limit[i]
            max_step_robot2 = limited_vel_robot2 + limited_acc_robot2 + self.jerk_limit[i]

            robot1_min_val = new_robot_actual_angle[i] + min_step_robot1
            robot1_max_val = new_robot_actual_angle[i] + max_step_robot1
            robot2_min_val = old_robot_actual_angle[i] + min_step_robot2
            robot2_max_val = old_robot_actual_angle[i] + max_step_robot2

            robot1_min_val = max(self.joint_min_position[i], robot1_min_val)
            robot1_max_val = min(self.joint_max_position[i], robot1_max_val)
            robot2_min_val = max(self.joint_min_position[i], robot2_min_val)
            robot2_max_val = min(self.joint_max_position[i], robot2_max_val)

            # 归一化界限
            robot1_min_bound = (robot1_min_val - self.joint_min_position[i]) / (self.joint_max_position[i] - self.joint_min_position[i])
            robot1_max_bound = (robot1_max_val - self.joint_min_position[i]) / (self.joint_max_position[i] - self.joint_min_position[i])
            robot2_min_bound = (robot2_min_val - self.joint_min_position[i]) / (self.joint_max_position[i] - self.joint_min_position[i])
            robot2_max_bound = (robot2_max_val - self.joint_min_position[i]) / (self.joint_max_position[i] - self.joint_min_position[i])
            robot1_bounds.append((robot1_min_bound, robot1_max_bound))
            robot2_bounds.append((robot2_min_bound, robot2_max_bound))

        bounds = robot1_bounds + robot2_bounds
        return bounds

if __name__ == "__main__":
    cycletime = 20
    bounds_calculator = BoundsCalculator(cycletime)

    # 模拟机器人的实际角度
    new_robot_actual_angle = np.random.rand(7)
    old_robot_actual_angle = np.random.rand(7)

    # 计算 bounds
    bounds = bounds_calculator.calculate_bounds(new_robot_actual_angle, old_robot_actual_angle)

    print("计算得到的 bounds:\n", bounds)