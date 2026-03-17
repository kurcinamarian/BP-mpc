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

class MPCController(Node):
    def __init__(self):
        super().__init__('mpc_controller')

        # ──────────────── tunable parameters ────────────────
        self.dt                = 0.1
        self.L                 = 0.33      # wheelbase [m]
        self.ref_speed         = 0.8       # m/s  ← increase if too slow
        self.P = 10
        self.N   = 5        

        self.max_accel         = 2.0
        self.min_accel         = -1.8
        self.max_steering      = 0.418     # ~24 deg
        self.min_steering      = -0.418

        self.max_v             = 4.0
        self.min_v             = 0.0

        # Cost weights
        self.w_pos             = 50.0
        self.w_yaw             = 10.0
        self.w_vel             = 10.0
        self.w_accel           = 2.0
        self.w_steer           = 2.0

        # ──────────────── internal state ────────────────
        self.current_x    = 0.0
        self.current_y    = 0.0
        self.current_yaw  = 0.0
        self.current_v    = 0.0
        self.current_s    = 0.0
        self.pose_received = False

        self.u_prev = np.zeros(2)          # last applied [a, delta]

        # Load path
        self.load_reference_path("/sim_ws/src/pure_pursuit/racelines/s1_fiit.csv")
        self.compute_path_properties()
        print(self.ref_path)

        # ROS
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.pred_path_pub = self.create_publisher(Path, '/mpc_predicted_path', 10)

        self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_callback, 10)

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.build_mpc_solver()

        self.get_logger().info("MPC controller initialized")

    def load_reference_path(self, file_path):
        data = np.loadtxt(file_path, delimiter=';')
        data = data[4:]  # skip header if needed
        self.ref_x = data[:, 0]
        self.ref_y = data[:, 1]

    def compute_path_properties(self):
        n = len(self.ref_x)
        s = np.zeros(n)
        yaw = np.zeros(n)
        for i in range(1, n):
            dx = self.ref_x[i] - self.ref_x[i-1]
            dy = self.ref_y[i] - self.ref_y[i-1]
            s[i] = s[i-1] + np.hypot(dx, dy)

        for i in range(n-1):
            dx = self.ref_x[i+1] - self.ref_x[i]
            dy = self.ref_y[i+1] - self.ref_y[i]
            yaw[i] = np.arctan2(dy, dx)
        yaw[-1] = yaw[-2]

        self.ref_path = np.column_stack((self.ref_x, self.ref_y, s, yaw))
        self.s_max = s[-1]
        self.get_logger().info(f"Track length: {self.s_max:.2f} m")

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

        self.current_s = self.get_closest_s(self.current_x, self.current_y)
        self.pose_received = True

    def get_closest_s(self, x, y):
        dist_sq = (self.ref_x - x)**2 + (self.ref_y - y)**2
        idx = np.argmin(dist_sq)
        return self.ref_path[idx, 2]

    def build_ref_traj(self):
        P = self.prediction_horizon
        X_ref = np.zeros((4, P + 1))

        s = self.current_s
        for k in range(P + 1):
            if k > 0:
                s += self.ref_speed * self.dt
                if s > self.s_max:
                    s -= self.s_max

            idx = np.searchsorted(self.ref_path[:, 2], s) % len(self.ref_path)
            X_ref[0, k] = self.ref_path[idx, 0]   # x
            X_ref[1, k] = self.ref_path[idx, 1]   # y
            X_ref[2, k] = self.ref_path[idx, 3]   # yaw
            X_ref[3, k] = self.ref_speed          # v

        return X_ref

    def publish_pred_path(self, X_opt):
        pass

    def build_mpc_solver(self):
        nx, nu = 4, 2
        X = ca.SX.sym('X', nx, self.P + 1)
        U = ca.SX.sym('U', nu, self.N)
        X_ref = ca.SX.sym('X_ref', nx, self.P + 1)
        
        # Linearization parameters for each step k
        # Ad_seq stores flattened (nx*nx) matrices, Bd_seq stores (nx*nu), Dd_seq stores (nx)
        Ad_params = ca.SX.sym('Ad', nx * nx * self.P)
        Bd_params = ca.SX.sym('Bd', nx * nu * self.P)
        Dd_params = ca.SX.sym('Dd', nx * self.P)
        x_init = ca.SX.sym('x_init', nx)

        obj = 0
        g = [X[:, 0] - x_init]

        Q = ca.diag([80.0, 80.0, 30.0, 10.0]) # x, y, yaw, v
        R = ca.diag([1.0, 5.0])              # accel, steer

        for k in range(self.P):
            # Cost
            err = X[:, k] - X_ref[:, k]
            obj += ca.mtimes([err.T, Q, err])
            
            uk = U[:, min(k, self.N-1)]
            obj += ca.mtimes([uk.T, R, uk])

            # Dynamics Constraints using pre-computed Ad, Bd, Dd
            Ak = ca.reshape(Ad_params[k*nx*nx : (k+1)*nx*nx], nx, nx)
            Bk = ca.reshape(Bd_params[k*nx*nu : (k+1)*nx*nu], nx, nu)
            Dk = Dd_params[k*nx : (k+1)*nx]
            
            g.append(X[:, k+1] - (ca.mtimes(Ak, X[:, k]) + ca.mtimes(Bk, uk) + Dk))

        # Constraints and Solver
        params = ca.vertcat(x_init, ca.reshape(X_ref, -1, 1), Ad_params, Bd_params, Dd_params)
        vars_all = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))
        
        # Bounds
        lbu = np.tile([self.min_accel, -self.max_steering], self.N)
        ubu = np.tile([self.max_accel,  self.max_steering], self.N)
        lbx = np.tile([-np.inf, -np.inf, -100, self.min_v], self.P + 1)
        ubx = np.tile([ np.inf,  np.inf,  100, self.max_v], self.P + 1)

        self.lbw = np.concatenate([lbu, lbx])
        self.ubw = np.concatenate([ubu, ubx])
        self.lbg = np.zeros((self.P + 1) * nx)
        self.ubg = np.zeros((self.P + 1) * nx)

        self.solver = ca.nlpsol('S', 'ipopt', {'f': obj, 'x': vars_all, 'g': ca.vertcat(*g), 'p': params}, 
                               {'ipopt.print_level': 0, 'print_time': 0})

    def get_linear_dynamics(self, x, u):
        """ Returns discretized Ad, Bd, and affine Dd """
        # Continuous Jacobians
        A = np.array([
            [0, 0, -x[3]*np.sin(x[2]), np.cos(x[2])],
            [0, 0,  x[3]*np.cos(x[2]), np.sin(x[2])],
            [0, 0, 0, np.tan(u[1])/self.L],
            [0, 0, 0, 0]
        ])
        B = np.array([
            [0, 0],
            [0, 0],
            [0, x[3]/(self.L * np.cos(u[1])**2)],
            [1, 0]
        ])
        # f(x, u)
        f = np.array([
            x[3] * np.cos(x[2]),
            x[3] * np.sin(x[2]),
            x[3] / self.L * np.tan(u[1]),
            u[0]
        ])
        
        # Discretize
        Ad = np.eye(4) + A * self.dt
        Bd = B * self.dt
        # Dd is the affine residual: f(x,u)dt - (A*x + B*u)dt
        Dd = (f - A @ x - B @ u) * self.dt
        return Ad, Bd, Dd

    def build_mpc_solver(self):
        nx, nu = 4, 2
        X = ca.SX.sym('X', nx, self.P + 1)
        U = ca.SX.sym('U', nu, self.N)
        X_ref = ca.SX.sym('X_ref', nx, self.P + 1)
        
        # Linearization parameters for each step k
        # Ad_seq stores flattened (nx*nx) matrices, Bd_seq stores (nx*nu), Dd_seq stores (nx)
        Ad_params = ca.SX.sym('Ad', nx * nx * self.P)
        Bd_params = ca.SX.sym('Bd', nx * nu * self.P)
        Dd_params = ca.SX.sym('Dd', nx * self.P)
        x_init = ca.SX.sym('x_init', nx)

        obj = 0
        g = [X[:, 0] - x_init]

        Q = ca.diag([80.0, 80.0, 30.0, 10.0]) # x, y, yaw, v
        R = ca.diag([1.0, 5.0])              # accel, steer

        for k in range(self.P):
            # Cost
            err = X[:, k] - X_ref[:, k]
            obj += ca.mtimes([err.T, Q, err])
            
            uk = U[:, min(k, self.N-1)]
            obj += ca.mtimes([uk.T, R, uk])

            # Dynamics Constraints using pre-computed Ad, Bd, Dd
            Ak = ca.reshape(Ad_params[k*nx*nx : (k+1)*nx*nx], nx, nx)
            Bk = ca.reshape(Bd_params[k*nx*nu : (k+1)*nx*nu], nx, nu)
            Dk = Dd_params[k*nx : (k+1)*nx]
            
            g.append(X[:, k+1] - (ca.mtimes(Ak, X[:, k]) + ca.mtimes(Bk, uk) + Dk))

        # Constraints and Solver
        params = ca.vertcat(x_init, ca.reshape(X_ref, -1, 1), Ad_params, Bd_params, Dd_params)
        vars_all = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))
        
        # Bounds
        lbu = np.tile([self.min_accel, -self.max_steering], self.N)
        ubu = np.tile([self.max_accel,  self.max_steering], self.N)
        lbx = np.tile([-np.inf, -np.inf, -100, self.min_v], self.P + 1)
        ubx = np.tile([ np.inf,  np.inf,  100, self.max_v], self.P + 1)

        self.lbw = np.concatenate([lbu, lbx])
        self.ubw = np.concatenate([ubu, ubx])
        self.lbg = np.zeros((self.P + 1) * nx)
        self.ubg = np.zeros((self.P + 1) * nx)

        self.solver = ca.nlpsol('S', 'ipopt', {'f': obj, 'x': vars_all, 'g': ca.vertcat(*g), 'p': params}, 
                               {'ipopt.print_level': 0, 'print_time': 0})
            
    def control_loop(self):
        if not self.pose_received: return

        # 1. Reference Trajectory
        x_curr = np.array([self.current_x, self.current_y, self.current_yaw, self.current_v])
        X_ref = np.zeros((4, self.P + 1))
        
        # Find closest point on path to start reference
        dist = np.hypot(self.ref_x - x_curr[0], self.ref_y - x_curr[1])
        idx = np.argmin(dist)
        
        # 2. Sequential Linearization along the horizon
        Ad_list, Bd_list, Dd_list = [], [], []
        x_lin = x_curr.copy()
        for k in range(self.P):
            # We linearize around our previous solution's prediction
            u_lin = self.u_prev[:, k] if k < self.N else self.u_prev[:, -1]
            Ad, Bd, Dd = self.get_linear_dynamics(x_lin, u_lin)
            Ad_list.append(Ad.flatten())
            Bd_list.append(Bd.flatten())
            Dd_list.append(Dd.flatten())
            # Advance linearization point
            x_lin = Ad @ x_lin + Bd @ u_lin + Dd

        # 3. Solve
        p = np.concatenate([x_curr, X_ref.flatten(), np.concatenate(Ad_list), np.concatenate(Bd_list), np.concatenate(Dd_list)])
        # Warm start using zeros or u_prev
        x0 = np.zeros(self.N*2 + (self.P+1)*4) 
        
        sol = self.solver(x0=x0, p=p, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg)
        u_out = np.array(sol['x'][:self.N*2]).reshape(self.N, 2)
        
        # 4. Publish
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = x_curr[3] + u_out[0, 0] * self.dt 
        drive_msg.drive.steering_angle = float(u_out[0, 1])
        self.drive_pub.publish(drive_msg)
        
        # Update u_prev for next warm start/linearization
        self.u_prev = u_out.T

def main(args=None):
    rclpy.init(args=args)
    node = MPCController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()