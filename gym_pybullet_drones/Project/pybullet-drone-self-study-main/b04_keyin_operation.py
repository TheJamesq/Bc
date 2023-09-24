import numpy as np
import cv2
import pybullet as p



from util.data_definition import DroneForcePIDCoefficients
from util.data_definition import DroneType, PhysicsType
from util.data_definition import DroneControlTarget
from blt_env.drone import DroneBltEnv
from control.drone_ctrl import DSLPIDControl
from util.external_input import KeyboardInputCtrl
from dev.bullet_cam import BulletCameraDevice, compute_view_matrix_from_cam_location
from util.data_logger import DroneDataLogger

if __name__ == "__main__":

    # Initialize the PyBullet simulation environment.
    init_xyzx = np.array([[.5, 0, 1], [-.5, 0, .5]])
    aggr_phy_steps = 5
    num_drone = 2        # Step the simulation
    urdf_file = './assets/drone_x_01.urdf'
    drone_type = DroneType.QUAD_X
    phy_mode = PhysicsType.PYB_DW

    blt_env = DroneBltEnv(
        num_drones=num_drone,
        urdf_path=urdf_file,
        d_type=drone_type,
        phy_mode=phy_mode,
        is_gui=True,
        aggr_phy_steps=aggr_phy_steps,
        init_xyzs=init_xyzx,
        is_real_time_sim=True,
    )

    # Initialize the camera.
    my_camera = BulletCameraDevice(
        res_w=640,
        res_h=480,
        z_near=0.01,
        z_far=10.0,
        fov_w=50,
    )

    # Initialize the simulation aviary.
    eps_sec = 30
    action_freq = blt_env.get_sim_freq() / blt_env.get_aggr_phy_steps()

    # Initialize the logger (optional).
    d_log = DroneDataLogger(
        num_drones=num_drone,
        logging_freq=int(action_freq),
        logging_duration=eps_sec,
    )

    # Initialize PID controllers.
    init_pid = DroneForcePIDCoefficients(
        P_for=np.array([.4, .4, 1.25]),
        I_for=np.array([.05, .05, .05]),
        D_for=np.array([.2, .2, .5]),
        P_tor=np.array([70000., 70000., 60000.]),
        I_tor=np.array([.0, .0, 500.]),
        D_tor=np.array([20000., 20000., 12000.]),
    )

    ctrls = [
        DSLPIDControl(
            env=blt_env,
            pid_coeff=init_pid,
        ) for _ in range(num_drone)
    ]

    # Run the simulation.
    ctrl_event_n_steps = aggr_phy_steps
    action = np.array([np.array([0, 0, 0, 0]) for i in range(num_drone)])

    # Initialize the keyboard controller.
    target_drone_id = 0
    key_ctrl = KeyboardInputCtrl(blt_env=blt_env, nth_drone=target_drone_id)

    for i in range(0, int(eps_sec * blt_env.get_sim_freq()), aggr_phy_steps):
        # Step the simulation
        kis = blt_env.step(action)

        # Compute control at the desired frequency
        if i % ctrl_event_n_steps == 0:
            # Compute control target from keyin.
            ctrl_target = key_ctrl.get_ctrl_target()

            for j in range(num_drone):
                ki = kis[j]
                if j != target_drone_id:
                    ctrl_target = DroneControlTarget(
                        pos=np.array(init_xyzx[j]),
                        vel=np.zeros(3),
                        rpy=np.zeros(3),
                    )

                action[j], _, _ = ctrls[j].compute_control_from_kinematics(
                    control_timestep=ctrl_event_n_steps * blt_env.get_sim_time_step(),
                    kin_state=ki,
                    ctrl_target=ctrl_target,
                )

            # Cam capture.init_xyzx
            cam_pos = kis[0].pos + np.array([0, 0, 0.04])
            cam_quat = kis[0].quat
            view_mat = compute_view_matrix_from_cam_location(cam_pos=cam_pos, cam_quat=cam_quat, )
            rgbImg, depthImg, segImg = my_camera.cam_capture(view_mat)

            segImg_8U = (segImg / segImg.max() * 255).astype(np.uint8)

            # Vytvořit tříkanálový obraz z segImg_8U
            segImg_color = cv2.cvtColor(segImg_8U, cv2.COLOR_GRAY2BGR)


            # Výpočet středového bodu obrazu (reprezentuje střed kamery)
            height, width = segImg_8U.shape
            centerX, centerY = width // 2, height // 2

            # Nakreslit středový bod kamery (modrou tečkou)
            cv2.circle(rgbImg, (centerX, centerY), 5, (255, 0, 0), -1) #segImg_color

            # Najít kontury
            contours, _ = cv2.findContours(segImg_8U, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Projít všemi nalezenými konturami
            for contour in contours:
                # Vypočítat obvod kontury
                perimeter = cv2.arcLength(contour, True)

                # Filtr kontur podle obvodu (můžete nastavit jiný limit)
                if perimeter > 50:
                    # Vypočítat středový bod kontury
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

                        # Nakreslit čáru mezi středovým bodem kamery a středovým bodem kontury (žlutou barvou)
                        cv2.line(rgbImg, (centerX, centerY), (cX, cY), (0, 255, 255), 2) #segImg_color

                        # Nakreslit středový bod kontury na obraz (červenou tečkou)
                        cv2.circle(rgbImg, (cX, cY), 5, (0, 0, 255), -1) #segImg_color

                        # Vypočítat obdélník kolem kontury
                        x, y, w, h = cv2.boundingRect(contour)

                        # Nakreslit obdélník kolem kontury (zelenou barvou)
                        cv2.rectangle(rgbImg, (x, y), (x + w, y + h), (0, 255, 0), 2) #segImg_color

            # Zobrazit aktualizovaný obraz
            cv2.imshow('Drone Tracking', rgbImg)
            cv2.waitKey(1)

        # Log the simulation (optional).
        rpms = blt_env.get_last_rpm_values()
        for j in range(num_drone):
            d_log.log(
                drone_id=j,
                time_stamp=i / (blt_env.get_sim_freq()),
                kin_state=kis[j],
                rpm_values=rpms[j],
            )

    # Close the environment.
    blt_env.close()

    # Plot the simulation results (optional).
    d_log.plot()
