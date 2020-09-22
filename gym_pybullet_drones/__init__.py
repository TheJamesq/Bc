from gym.envs.registration import register

register(
    id='ctrl-aviary-v0',
    entry_point='gym_pybullet_drones.envs:CtrlAviary',
)

register(
    id='dyn-ctrl-aviary-v0',
    entry_point='gym_pybullet_drones.envs:DynCtrlAviary',
)

register(
    id='vision-ctrl-aviary-v0',
    entry_point='gym_pybullet_drones.envs:VisionCtrlAviary',
)

register(
    id='rl-takeoff-aviary-v0',
    entry_point='gym_pybullet_drones.envs:RLTakeoffAviary',
)

register(
    id='marl-flock-aviary-v0',
    entry_point='gym_pybullet_drones.envs:MARLFlockAviary',
)

register(
    id='norm-dyn-ctrl-aviary-v0',
    entry_point='gym_pybullet_drones.envs:NormDynCtrlAviary',
)

