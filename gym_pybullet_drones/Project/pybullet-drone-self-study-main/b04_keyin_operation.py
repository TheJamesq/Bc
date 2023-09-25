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
    init_xyzx = np.array([[0.3, .5, 1], [-.4, .5, 1]])
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
        z_far=100.0,
        fov_w=83,
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
    follower_id = 1
    key_ctrl = KeyboardInputCtrl(blt_env=blt_env, nth_drone=target_drone_id)

    for i in range(0, int(eps_sec * blt_env.get_sim_freq()), aggr_phy_steps):
        # Step the simulation
        kis = blt_env.step(action)

        # Compute control at the desired frequency
        if i % ctrl_event_n_steps == 0:
            # Compute control target from keyin.
            # Cam capture.init_xyzx
            cam_pos = kis[follower_id].pos + np.array([0, 0, -0.025])
            cam_quat = kis[follower_id].quat
            view_mat = compute_view_matrix_from_cam_location(cam_pos=cam_pos, cam_quat=cam_quat, )
            rgbImg, depthImg, segImg = my_camera.cam_capture(view_mat)


            # Vytvoření masky, kde jsou pixely s tímto ID nastaveny na nulu
            mask = segImg != 2

            # Aplikace masky na segmentovaný obraz
            filtered_segImg = segImg * mask

            if filtered_segImg.max() > 0:
                segImg_8U = (filtered_segImg / filtered_segImg.max() * 255).astype(np.uint8)
            else:
                segImg_8U = np.zeros_like(filtered_segImg, dtype=np.uint8)
            # Vytvořit tříkanálový obraz z segImg_8U
            segImg_color = cv2.cvtColor(segImg_8U, cv2.COLOR_GRAY2BGR)


            # Výpočet středového bodu obrazu (reprezentuje střed kamery)
            height, width = segImg_8U.shape
            centerX, centerY = width // 2, height // 2



            # Nakreslit středový bod kamery (modrou tečkou)
            #cv2.circle(rgbImg, (centerX, centerY), 5, (255, 0, 0), -1) #segImg_color

            contours, _ = cv2.findContours(segImg_8U, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            scale_factor = 30  # Nastavte podle potřeby
            current_speed = kis[target_drone_id].vel
            # Projít všemi nalezenými konturami
            for contour in contours:
                # Vypočítat obvod kontury
                perimeter = cv2.arcLength(contour, True)

                # Filtr kontur podle obvodu (můžete nastavit jiný limit)
                if perimeter > 20:
                    # Vypočítat středový bod kontury
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

                        end_point_x = int(cX)
                        end_point_y = int(cY - current_speed[2] * scale_factor)  # minus, protože v obrazovém formátu je směr nahoru negativní

                        # Vykreslení šipky pro vertikální rychlost
                        cv2.arrowedLine(rgbImg, (cX, cY), (end_point_x, end_point_y), (255, 255, 0), 2, tipLength=0.2)

                        # Nakreslit čáru mezi středovým bodem kamery a středovým bodem kontury (žlutou barvou)
                        cv2.line(rgbImg, (centerX, centerY), (cX, cY), (0, 255, 255), 2) #segImg_color

                        # Nakreslit středový bod kontury na obraz (červenou tečkou)
                        cv2.circle(rgbImg, (cX, cY), 2, (0, 0, 255), -1) #segImg_color

                        # Vypočítat obdélník kolem kontury
                        x, y, w, h = cv2.boundingRect(contour)

                        # Nakreslit obdélník kolem kontury (zelenou barvou)
                        cv2.rectangle(rgbImg, (x, y), (x + w, y + h), (0, 255, 0), 2) #segImg_color

            # Zobrazit aktualizovaný obraz
            cv2.imshow('Drone Tracking', rgbImg)
            cv2.waitKey(1)

            ctrl_target = key_ctrl.get_ctrl_target()


            for j in range(num_drone):
                ki = kis[j]
                '''if j != target_drone_id:
                    ctrl_target = DroneControlTarget(
                        pos=np.array(init_xyzx[j]),
                        vel=np.zeros(3),
                        rpy=np.zeros(3),
                    )'''
                if j == follower_id:
                    target_distance = 0.35


                    # Konstanty PI regulátoru
                    Kp_yaw = 0.0015
                    Ki_yaw = 0.00072

                    Kp_height =  1.26# 0.0032 for video
                    Ki_height =  0.95#0.0045 for video

                    Kp_distance_x = 0.55
                    Ki_distance_x = 0.082
                    Kd_distance_x  = 0.01

                    Kp_distance_y = 0.55
                    Ki_distance_y = 0.082
                    Kd_distance_y  = 0.01

                    # Inicializace akumulovaného chybového signálu
                    accumulated_error_yaw = 0
                    accumulated_error_height = 0
                    accumulated_error_distance_x = 0
                    accumulated_error_distance_y = 0


                    # Získání aktuálního úhlu natočení (yaw) drona v radiánech
                    current_yaw = ki.rpy[2]
                    current_height = ki.pos[2]
                    current_distance_x = ki.pos[0]
                    current_distance_y = ki.pos[1]

                    #prev
                    previous_error_distance_x = 0
                    previous_error_distance_y = 0

                    # Získání středového bodu leadera
                    y, x = np.where(segImg_8U == 255)
                    if len(x) > 0:
                        leader_center_x = np.mean(x)

                        # Výpočet chybového signálu (rozsdíl ve středových bodech)
                        error_yaw =  centerX - leader_center_x

                        # Akumulace chybového signálu
                        accumulated_error_yaw += error_yaw

                        # Aktualizace úhlu natočení (yaw) pomocí PI regulátoru
                        delta_yaw = Kp_yaw * error_yaw + Ki_yaw * accumulated_error_yaw
                        current_yaw += delta_yaw

                        # Normalizace úhlu yaw do rozsahu [-pi, pi]
                        current_yaw = (current_yaw + np.pi) % (2 * np.pi) - np.pi




                        #--------------------- height control

                        # Výpočet chybového signálu (rozsdíl ve středových bodech)
                        error_height =  (kis[target_drone_id].pos[2]) - kis[follower_id].pos[2]

                        # Akumulace chybového signálu
                        accumulated_error_height += error_height

                        # Aktualizace úhlu natočení (yaw) pomocí PI regulátoru
                        delta_height = Kp_height * error_height + Ki_height * accumulated_error_height
                        current_height += delta_height




                        #------------------- x distance
                        current_distance_x = (kis[target_drone_id].pos[0] - kis[follower_id].pos[0])
                        error_distance_x =  current_distance_x - target_distance

                        # Akumulace chybového signálu
                        accumulated_error_distance_x += error_distance_x

                        derivative_distance_x = error_distance_x - previous_error_distance_x
                        previous_error_distance_x = error_distance_x

                        # Výpočet akce pomocí PID regulátoru
                        delta_distance_x = Kp_distance_x * error_distance_x + Ki_distance_x * accumulated_error_distance_x + Kd_distance_x * derivative_distance_x

                        updated_distance_x = ki.pos[0] + delta_distance_x




                        # ------------------- y distance
                        current_distance_y = (kis[target_drone_id].pos[1] - kis[follower_id].pos[1])
                        error_distance_y =  current_distance_y - target_distance

                        # Akumulace chybového signálu
                        accumulated_error_distance_y += error_distance_y

                        derivative_distance_y = error_distance_y - previous_error_distance_y
                        previous_error_distance_y = error_distance_y

                        # Výpočet akce pomocí PID regulátoru
                        delta_distance_y = Kp_distance_y * error_distance_y + Ki_distance_y * accumulated_error_distance_y + Kd_distance_y * derivative_distance_y

                        updated_distance_y = ki.pos[1] + delta_distance_y


                        print(delta_distance_x, delta_distance_y)

                        # Nastavení nového úhlu natočení (yaw)
                        ctrl_target = DroneControlTarget(
                            pos=np.array([updated_distance_x, updated_distance_y, current_height]),  # Aktuální pozice drona
                            vel=np.array([0,0,0]),
                            rpy=np.array([0, 0, current_yaw])  # Aktualizovaný yaw
                        )
                    else:
                        # Pokud "leader" není vidět, můžete například zůstat na aktuálním místě
                        ctrl_target = DroneControlTarget(
                            pos=kis[1].pos,  # Aktuální pozice drona
                            vel=np.zeros(3),
                            rpy=np.zeros(3)
                        )





                action[j], _, _ = ctrls[j].compute_control_from_kinematics(
                    control_timestep=ctrl_event_n_steps * blt_env.get_sim_time_step(),
                    kin_state=ki,
                    ctrl_target=ctrl_target,
                )


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
