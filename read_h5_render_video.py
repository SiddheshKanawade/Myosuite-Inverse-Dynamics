# read h5 file and render video with mujoco -> check if it's same as video
import time

import gymnasium as gym
import numpy as np
import pandas as pd
from myosuite.logger.grouped_datasets import Trace

inverse_name_map = {
    "Abs_t2": "Abs_t2",
    "Abs_t1": "Abs_t1",
    "Abs_r3": "Abs_r3",
    "pro_sup": "pro_sup",
    "deviation": "deviation",
    ""
    # Spine
    "L5_S1_Flex_Ext": "flex_extension",
    "L5_S1_Lat_Bending": "lat_bending",
    "L5_S1_axial_rotation": "axial_rotation",
    "L4_L5_Flex_Ext": "L4_L5_FE",
    "L4_L5_Lat_Bending": "L4_L5_LB",
    "L4_L5_axial_rotation": "L4_L5_AR",
    "L3_L4_Flex_Ext": "L3_L4_FE",
    "L3_L4_Lat_Bending": "L3_L4_LB",
    "L3_L4_axial_rotation": "L3_L4_AR",
    "L2_L3_Flex_Ext": "L2_L3_FE",
    "L2_L3_Lat_Bending": "L2_L3_LB",
    "L2_L3_axial_rotation": "L2_L3_AR",
    "L1_L2_Flex_Ext": "L1_L2_FE",
    "L1_L2_Lat_Bending": "L1_L2_LB",
    "L1_L2_axial_rotation": "L1_L2_AR",
    # # Clavicle / scapula
    "sternoclavicular_r2_r": "sternoclavicular_r2",
    "sternoclavicular_r3_r": "sternoclavicular_r3",
    "unrotscap_r3_r": "unrotscap_r3",
    "unrotscap_r2_r": "unrotscap_r2",
    "acromioclavicular_r2_r": "acromioclavicular_r2",
    "acromioclavicular_r3_r": "acromioclavicular_r3",
    "acromioclavicular_r1_r": "acromioclavicular_r1",
    # # Humerus / arm
    "unrothum_r1_r": "unrothum_r1",
    "unrothum_r3_r": "unrothum_r3",
    "unrothum_r2_r": "unrothum_r2",
    "elv_angle_r": "elv_angle",
    "shoulder_elv_r": "shoulder_elv",
    "shoulder1_r2_r": "shoulder1_r2",
    "shoulder_rot_r": "shoulder_rot",
    "elbow_flex_r": "elbow_flexion",
    # # Wrist & thumb
    "flexion_r": "flexion",
    "cmc_abduction_r": "cmc_abduction",
    "cmc_flexion_r": "cmc_flexion",
    "mp_flexion_r": "mp_flexion",
    "ip_flexion_r": "ip_flexion",
    # # Fingers
    "mcp2_flexion_r": "mcp2_flexion",
    "mcp2_abduction_r": "mcp2_abduction",
    "pm2_flexion_r": "pm2_flexion",
    "md2_flexion_r": "md2_flexion",
    "mcp3_flexion_r": "mcp3_flexion",
    "mcp3_abduction_r": "mcp3_abduction",
    "pm3_flexion_r": "pm3_flexion",
    "md3_flexion_r": "md3_flexion",
    "mcp4_flexion_r": "mcp4_flexion",
    "mcp4_abduction_r": "mcp4_abduction",
    "pm4_flexion_r": "pm4_flexion",
    "md4_flexion_r": "md4_flexion",
    "mcp5_flexion_r": "mcp5_flexion",
    "mcp5_abduction_r": "mcp5_abduction",
    "pm5_flexion_r": "pm5_flexion",
    "md5_flexion_r": "md5_flexion",
    # # Freejoints
    # "paddle_freejoint": "paddle_freejoint",
    # "pingpong_freejoint": "pingpong_freejoint",
}

name_map = {v: k for k, v in inverse_name_map.items()}


H5_FILE = "./data/trace_resized_trimmed_test_2_compressed.h5"
h5trajectory = Trace.load(H5_FILE)

motion_name = list(h5trajectory.trace.keys())[0]

timesteps = np.array(h5trajectory[motion_name]["time"])
horizon = timesteps.shape[0]

joint_dict = h5trajectory[motion_name]["qpos"]
data_root = h5trajectory[motion_name]["qpos"]["myoskeleton_root"]

data = {
    joint: np.array(values).flatten()
    for joint, values in joint_dict.items()
    if joint != "myoskeleton_root"
}

df = pd.DataFrame(data)


env = gym.make("myoChallengeTableTennisP1-v0")
env.reset()

env_joint_names = [
    env.sim.model.joint(i).name for i in range(env.sim.model.njnt) if i < 60
]
_subc = [inverse_name_map[c] for c in df.columns if c in inverse_name_map.keys()]
subc = [c for c in _subc if c in env_joint_names]

try:
    for t in range(300):
        # env.sim.data.qpos[0:7] = np.array(data_root).reshape(-1, 7)[t]
        env.sim.data.qpos[0:7] = np.zeros(7)
        for jn in subc:
            j_idx = env_joint_names.index(jn)
            adr = env.sim.model.jnt_qposadr[j_idx]
            env.unwrapped.sim.data.qpos[adr] = df[name_map[jn]].loc[t]
        env.unwrapped.sim.data.qvel[:] = 0
        env.unwrapped.sim.forward()
        env.unwrapped.mj_render()
        time.sleep(0.05)
finally:
    env.close()


### Code to render Mujoco video - to visualise

# data = {
#     joint: np.array(values).flatten()
#     for joint, values in joint_dict.items()
#     if joint != "myoskeleton_root"
# }

# df = pd.DataFrame(data)

# joint_names = [mj_model.joint(jn).name for jn in range(mj_model.njnt)]
# subc = [c for c in df.columns if c in joint_names]

# # ---- camera settings -- Adjust camera settings
# camera = mujoco.MjvCamera()
# camera.azimuth = 90
# camera.distance = 3
# camera.elevation = -45.0
# camera.lookat = [0, 0, 1.75]
# options_ref = mujoco.MjvOption()
# options_ref.flags[:] = 0
# options_ref.geomgroup[1:] = 0
# renderer_ref = mujoco.Renderer(mj_model)
# renderer_ref.scene.flags[:] = 0
# frames = []
# from tqdm import tqdm

# for t in tqdm(range(len(df)), desc="Rendering frames"):
#     mj_data.qpos[:7] = data_root[t]
#     for jn in subc:
#         mjc_j_idx = mj_model.joint(joint_names.index(jn)).qposadr
#         mj_data.qpos[mjc_j_idx] = df[jn].loc[t]

#     mujoco.mj_forward(mj_model, mj_data)
#     renderer_ref.update_scene(
#         mj_data, camera=camera
#     )  # , scene_option=options_ref)
#     frame = renderer_ref.render()
#     frames.append(frame)

# import os

# import skvideo.io

# os.makedirs("videos", exist_ok=True)
# output_name = "videos/playback_mot.mp4"
# skvideo.io.vwrite(
#     output_name, np.asarray(frames), outputdict={"-pix_fmt": "yuv420p"}
# )
