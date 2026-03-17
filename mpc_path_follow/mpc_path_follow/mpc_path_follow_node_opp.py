#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import casadi as ca
from nav_msgs.msg import Odometry, Path
from ackermann_msgs.msg import AckermannDriveStamped

class MPCController(Node):
    def __init__(self):
        super().__init__('mpc_controller')

        # Parameters
        self.control_horizon    = 5
        self.prediction_horizon = 10
        self.dt                 = 0.1
        self.L                  = 0.33
        self.ref_speed          = 0.5

        self.min_accel    = -2.0
        self.max_accel    =  2.0
        self.min_steering = -0.418
        self.max_steering =  0.418

        self.pose_received = False

        # Current state
        self.current_x    = 0.0
        self.current_y    = 0.0
        self.current_yaw  = 0.0
        self.current_v    = 0.0
        self.current_s    = 0.0

        self.leader_path = None


        # ROS
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/opp_drive', 10)
        self.create_subscription(Odometry, '/opp_racecar/odom', self.odom_callback, 10)
        self.create_subscription(Path, '/leader_predicted_path', self.leader_path_callback, 10)
        self.build_mpc_solver()

        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info("MPC Node Initialized")

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.current_yaw = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))
        self.current_v = msg.twist.twist.linear.x
        self.pose_received = True

    def leader_path_callback(self, msg: Path):
        if not msg.poses:
            return

        lx, ly, lyaw = [], [], []
        for pose_stamped in msg.poses:
            p = pose_stamped.pose
            lx.append(p.position.x)
            ly.append(p.position.y)
            
            # Convert orientation to yaw
            q = p.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
            lyaw.append(np.arctan2(siny_cosp, cosy_cosp))

        # Save as numpy arrays for speed
        self.leader_path = (np.array(lx), np.array(ly), np.array(lyaw))

    def build_ref_traj(self):
        X_ref = np.zeros((4, self.prediction_horizon + 1))
        
        if self.leader_path is None:
            # If no path yet, reference the current state so the car stays still
            X_ref[0, :] = self.current_x
            X_ref[1, :] = self.current_y
            X_ref[2, :] = self.current_yaw
            X_ref[3, :] = 0.0
            return X_ref

        lx, ly, lyaw = self.leader_path
        n_points = len(lx)

        for k in range(self.prediction_horizon + 1):
            # Always start from index 0 of the received message
            # We step forward k steps into the message
            idx = min(k, n_points - 1)
            
            X_ref[0, k] = lx[idx]
            X_ref[1, k] = ly[idx]
            X_ref[2, k] = lyaw[idx]
            
            # Target velocity: either use a constant or the leader's speed
            X_ref[3, k] = self.ref_speed

        return X_ref

    def build_mpc_solver(self):
        N = self.control_horizon
        P = self.prediction_horizon
        dt = self.dt
        L = self.L

        states   = ca.SX.sym('states',   4)   # x, y, yaw, v
        controls = ca.SX.sym('controls', 2)   # a, delta

        rhs = ca.vertcat(
            states[3] * ca.cos(states[2]),
            states[3] * ca.sin(states[2]),
            states[3] / L * ca.tan(controls[1]),
            controls[0]
        )
        f = ca.Function('f', [states, controls], [rhs])

        U = ca.SX.sym('U', 2, N)
        X = ca.SX.sym('X', 4, P+1)

        X_ref  = ca.SX.sym('X_ref',  4, P+1)
        x_init = ca.SX.sym('x_init', 4, 1)

        # Weights (balanced for stability)
        Q  = ca.diag([220.0, 220.0, 0.0, 150])   # x, y, yaw, v,
        R  = ca.diag([30,  30])
        Qf = ca.diag([600.0, 600.0, 0.0, 3.0])

        obj = 0.0
        d = 1
        g = []

        g.append(X[:,0] - x_init)

        for k in range(P):
            st = X[:, k]
            con = U[:, k] if k < N else U[:, -1]
            ref = X_ref[:, k]

            # Wrapped yaw error
            yaw_err = ca.atan2(ca.sin(st[2] - ref[2]), ca.cos(st[2] - ref[2]))

            err = ca.vertcat(
                st[0] - ref[0] + d * ca.cos(ref[2]),
                st[1] - ref[1] + d * ca.sin(ref[2]),
                yaw_err,
                st[3] - ref[3],
            )

            obj += err.T @ Q @ err

            # Gentle incentive to keep moving
            obj += -0.4 * st[3]   # small positive velocity bias

            if k < N:
                obj += con.T @ R @ con

            st_next = st + dt * f(st, con)
            g.append(st_next - X[:, k+1])

        # Terminal cost
        yaw_err_f = ca.atan2(ca.sin(X[2,-1] - X_ref[2,-1]), ca.cos(X[2,-1] - X_ref[2,-1]))
        err_f = ca.vertcat(
            X[0,-1] - X_ref[0,-1],
            X[1,-1] - X_ref[1,-1],
            yaw_err_f,
            X[3,-1] - X_ref[3,-1]
        )
        obj += err_f.T @ Q @ err_f

        g_all = ca.vertcat(*g)
        vars_all = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))

        nlp = {
            'x': vars_all,
            'f': obj,
            'g': g_all,
            'p': ca.vertcat(x_init, ca.reshape(X_ref, -1, 1))
        }

        lbx_ctrl = np.tile([self.min_accel, self.min_steering], N)
        ubx_ctrl = np.tile([self.max_accel, self.max_steering], N)

        lbx_state = np.tile([-np.inf, -np.inf, -np.pi*4, -0.3], P+1)
        ubx_state = np.tile([ np.inf,  np.inf,  np.pi*4, 12.0], P+1)

        self.lbx = np.concatenate([lbx_ctrl, lbx_state])
        self.ubx = np.concatenate([ubx_ctrl, ubx_state])

        n_g = 4 + 4 * P
        self.lbg = np.zeros(n_g)
        self.ubg = np.zeros(n_g)

        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 300,
            'ipopt.tol': 1e-3,
            'ipopt.acceptable_tol': 1e-2,
            'ipopt.acceptable_constr_viol_tol': 1e-3,
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        self.u_prev = np.zeros((2, N))
        self.x_guess = np.zeros((4, P+1))

        self.get_logger().info(f"MPC ready  N={N}  P={P}")

    def control_loop(self):
        if not self.pose_received:
            return

        X_ref_np = self.build_ref_traj()

        # Warm-start: copy reference but override first state
        self.x_guess[:,1:] = X_ref_np[:,:-1]
        self.x_guess[:, 0] = np.array([self.current_x, self.current_y, self.current_yaw, self.current_v])

        x0 = np.concatenate([self.u_prev.ravel(order='F'), self.x_guess.ravel(order='F')])

        p = np.concatenate([
            [self.current_x, self.current_y, self.current_yaw, self.current_v],
            X_ref_np.ravel(order='F')
        ])

        try:
            sol = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)
        except Exception as e:
            self.get_logger().error(f"Solver failed: {str(e)}")
            return

        offset_u = 0
        offset_x = 2 * self.control_horizon

        U_opt = sol['x'][offset_u:offset_x].full().reshape(2, self.control_horizon)
        X_opt = sol['x'][offset_x:].full().reshape(4, self.prediction_horizon + 1)

        a_cmd     = float(U_opt[0, 0])
        delta_cmd = float(U_opt[1, 0])
        v_pred    = float(X_opt[3, 0])

        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.drive.speed = v_pred
        cmd.drive.steering_angle = delta_cmd
        self.drive_pub.publish(cmd)

        self.u_prev = np.hstack((U_opt[:, 1:], U_opt[:, [-1]]))

        # Debug log ~every second
        now = self.get_clock().now()
        if now.nanoseconds % 1_000_000_000 < 200_000_000:
            dist = np.hypot(self.current_x - X_ref_np[0,0], self.current_y - X_ref_np[1,0])
            self.get_logger().info(
                f"cmd v={v_pred:.2f} m/s   δ={np.degrees(delta_cmd):.1f}°   "
                f"dist_to_ref0={dist:.3f} m   pred_v0={float(X_opt[3,0]):.2f}"
            )

def main():
    rclpy.init()
    rclpy.spin(MPCController())
    rclpy.shutdown()

if __name__ == '__main__':
    main()