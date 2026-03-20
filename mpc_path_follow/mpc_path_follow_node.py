#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import casadi as ca
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from scipy.signal import cont2discrete
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

class MPCController(Node):
    def __init__(self):
        super().__init__('mpc_controller')

        # Reference path
        self.leader = True
        self.path_to_follow = "/sim_ws/src/pure_pursuit/racelines/s1_fiit.csv"
        self.predecessor_topic = None
        self.leader_topic = None
        self.cycle = True
        self.current_s = 0.0
        self.ref_speed = 1
        self.last_idx = 0
        # Predicted path
        self.predicted_path_topic = '/ego_racecar/predicted'
        # ROS
        self.vehicle_drive_topic = '/drive'
        self.vehicle_odom_topic = '/ego_racecar/odom'
        self.pose_received = False
        # MPC
        self.L = 0.33
        self.horizon = 20
        self.dt = 0.1
        self.num_states = 4
        self.num_control = 2

        # Control

        # Reference path
        self.prepare_source_of_path()

        self.get_logger().info("Reference path loaded.")

        # Predicted path
        self.pred_path_pub = self.create_publisher(Path, self.predicted_path_topic, 10)
        self.get_logger().info("Prediction publisher inicialized.")

        # ROS
        self.drive = self.create_publisher(AckermannDriveStamped, self.vehicle_drive_topic, 10)
        self.get_logger().info("Drive publisher inicialized.")

        self.create_subscription(Odometry, self.vehicle_odom_topic, self.odom_callback, 10)
        self.get_logger().info("Odometry subscription inicialized.")

        # MPC
        self.solver = self.build_mpc()
        self.get_logger().info("MPC controller initialized.")

        # Control
        self.timer = self.create_timer(self.dt, self.control)
        self.get_logger().info("Control Started.")

    def prepare_source_of_path(self):
        if self.leader and self.path_to_follow is not None:
            self.ref_path = self.load_path()
        elif self.leader and self.path_to_follow is None:
            raise Exception("Reference path for leader has to be defined.")
        else:
            if self.leader_topic is None:
                raise Exception("Predecessor topic has to be defined.")
            #start subscription odom callback
            
    def load_path(self): #x,y,s,yaw
        path = np.loadtxt(self.path_to_follow, delimiter=';', skiprows=1)
        n = len(path)
        extra_cols = np.zeros((n, 4 - path.shape[1]))
        path = np.hstack((path, extra_cols))
        path[0,2] = 0.0  
        path[0,3] = 0.0
        for i in range(1, n):
            dx = path[i,0] - path[i-1,0]
            dy = path[i,1] - path[i-1,1]
            path[i,2] =  path[i-1,2] + np.hypot(dx, dy)

        for i in range(0, n-1):
            dx = path[i+1,0] - path[i,0]
            dy = path[i+1,1] - path[i,1]
            path[i,3] = np.arctan2(dy, dx)

        path[n-1,3] = path[n-2,3]

        self.s_max = path[-1,2]

        print(path)
        return path

    def odom_callback(self, msg: Odometry):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
        self.current_yaw = np.arctan2(siny_cosp, cosy_cosp)

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_v = np.hypot(vx, vy)

        self.pose_received = True

    def predecessor_callback(self):
        pass

    def create_model(self):
        model = AcadosModel()
        model.name = "model"

        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        v = ca.MX.sym('v')
        yaw = ca.MX.sym('yaw')

        states = ca.vertcat(x,y,v,yaw)

        a = ca.MX.sym('a')
        delta = ca.MX.sym('delta')

        controls = ca.vertcat(a,delta)

        x_dot = v * ca.cos(yaw)
        y_dot = v * ca.sin(yaw)
        v_dot = a
        yaw_dot = v / self.L * ca.tan(delta)

        dynamics = ca.vertcat(x_dot, y_dot, v_dot, yaw_dot)

        model.x = states
        model.u = controls
        model.f_expl_expr = dynamics

        return model
    

    def build_mpc(self):
        ocp = AcadosOcp()
        ocp.model = self.create_model()

        num_states = 4
        num_control = 2
        
        ocp.dims.N = self.horizon
        ocp.solver_options.tf = self.horizon*self.dt

        Q = np.diag([10,10,50,10])
        R = np.diag([10,10])

        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        ocp.cost.W = np.diag(np.concatenate([Q.diagonal(), R.diagonal()]))
        ocp.cost.Vx = np.zeros((num_states + num_control, num_states))
        ocp.cost.Vx[:num_states, :num_states] = np.eye(num_states)

        ocp.cost.Vu = np.zeros((num_states + num_control, num_control))
        ocp.cost.Vu[num_states:, :] = np.eye(num_control)
        
        ocp.cost.yref = np.zeros(num_states + num_control)

        # Initial stage cost (stage 0)
        ocp.cost.W_0 = Q                     # only states
        ocp.cost.Vx_0 = np.eye(num_states)   # only states
        ocp.cost.Vu_0 = np.zeros((num_states,num_control))  # no controls at t=0
        ocp.cost.yref_0 = np.zeros(num_states)     # initial ref

        # Terminal cost
        ocp.cost.W_e = Q
        ocp.cost.Vx_e = np.eye(num_states)
        ocp.cost.yref_e = np.zeros(num_states)
        
        ocp.constraints.lbu = np.array([-2.0,-0.41])
        ocp.constraints.ubu = np.array([2.0,0.41])
        ocp.constraints.idxbu = np.array([0, 1])

        ocp.constraints.lbx_0 = np.zeros(num_states)
        ocp.constraints.ubx_0 = np.zeros(num_states)
        ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3])

        ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.integrator_type = "ERK"

        solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

        return solver

    def solve_mpc(self, ref_traj):
        N = self.horizon

        x0 = np.array([self.current_x,self.current_y,self.current_v,self.current_yaw])

        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

        for k in range(N):
            yref = np.hstack((ref_traj[:, k], [0, 0])) 
            self.solver.set(k, "yref", yref)

        self.solver.set(N, "yref", ref_traj[:, -1])

        status = self.solver.solve()

        if status != 0:
            self.get_logger().info("Solver failed.")


        X_pred = np.zeros((N+1, self.num_states))
        for k in range(N+1):
            X_pred[k, :] = self.solver.get(k, "x")

        return X_pred

    def find_closest_index(self, x, y):
        dx = self.ref_path[:, 0] - x
        dy = self.ref_path[:, 1] - y
        dist = dx**2 + dy**2
        return np.argmin(dist)
    

    def get_ref(self):
        if self.leader:
            P = self.horizon
            X_ref = np.zeros((4, P + 1))

            idx0 = self.find_closest_index(self.current_x, self.current_y)

            s = self.ref_path[idx0, 2]
            prev_yaw = self.current_yaw

            for k in range(P + 1):

                s += self.ref_speed * self.dt
                if s > self.s_max:
                    s -= self.s_max

                idx = np.searchsorted(self.ref_path[:, 2], s) % len(self.ref_path)
                X_ref[0, k] = self.ref_path[idx, 0]   # x
                X_ref[1, k] = self.ref_path[idx, 1]   # y
                X_ref[2, k] = self.ref_speed          # v
                X_ref[3, k] = self.ref_path[idx, 3]   # yaw
                yaw_ref = self.ref_path[idx, 3]

                while yaw_ref - prev_yaw > np.pi:
                    yaw_ref -= 2*np.pi
                while yaw_ref - prev_yaw < -np.pi:
                    yaw_ref += 2*np.pi

                X_ref[3, k] = yaw_ref
                prev_yaw = yaw_ref
            self.get_logger().info(
                f"Ref = {X_ref}"
            )
            return X_ref
    
    def publish_path(self):
        pass

    def apply_control(self,speed,steering):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = float(speed)                # m/s
        drive_msg.drive.steering_angle = float(steering)       # rad
        self.get_logger().info(
            f"Publishing Drive: Speed={float(speed):.2f} m/s, Steer={float(steering):.2f} rad"
        )
        self.drive.publish(drive_msg)

    def control(self):
        if not self.pose_received: 
            self.get_logger().info(
                f"No pose received"
            )
            return
        ref = self.get_ref()
        pred_states = self.solve_mpc(ref)
        u_0 = self.solver.get(0, "u")
        x_1 = self.solver.get(1, "x")
        next_speed = x_1[2]
        steering_cmd = u_0[1]
        self.get_logger().info(
                f"pred = {pred_states}"
            )
        self.apply_control(next_speed,steering_cmd)
        #publish_prediction(pred[:][2:])







def main(args=None):
    rclpy.init(args=args)
    node = MPCController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()