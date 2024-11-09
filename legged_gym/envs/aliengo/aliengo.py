# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
from isaacgym.torch_utils import *

from isaacgym import gymapi
from legged_gym.envs import LeggedRobot

from .mixed_terrains.aliengo_rough_config import AliengoRoughCfg


class AlienGoCameraMixin:
    def __init__(self, *args, **kwargs):
        self.follow_cam = None
        self.floating_cam = None
        super().__init__(*args, **kwargs)

    def init_aux_cameras(self, follow_cam=False, float_cam=False):
        if follow_cam:
            self.follow_cam, follow_trans = self.make_handle_trans(
                1920, 1080, 0, (1.0, -1.0, 0.0), (0.0, 0.0, 3 * 3.14 / 4)
            )
            body_handle = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], "base"
            )
            self.gym.attach_camera_to_body(
                self.follow_cam,  # camera_handle,
                self.envs[0],
                body_handle,
                follow_trans,
                gymapi.FOLLOW_POSITION,
            )

        if float_cam:
            self.floating_cam, _ = self.make_handle_trans(
                # 1280, 720, 0, (0, 0, 0), (0, 0, 0), hfov=50
                1920, 1080, 0, (0, 0, 0), (0, 0, 0)
            )
            camera_position = gymapi.Vec3(5, 5, 5)
            camera_target = gymapi.Vec3(0, 0, 0)
            self.gym.set_camera_location(
                self.floating_cam, self.envs[0], camera_position, camera_target
            )

    def make_handle_trans(self, width, height, env_idx, trans, rot, hfov=None):
        camera_props = gymapi.CameraProperties()
        camera_props.width = width
        camera_props.height = height
        camera_props.enable_tensors = True
        if hfov is not None:
            camera_props.horizontal_fov = hfov
        camera_handle = self.gym.create_camera_sensor(self.envs[env_idx], camera_props)
        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(*trans)
        local_transform.r = gymapi.Quat.from_euler_zyx(*rot)
        return camera_handle, local_transform


class Aliengo(AlienGoCameraMixin, LeggedRobot):
    cfg: AliengoRoughCfg

    def __init__(
        self, cfg, sim_params, physics_engine, sim_device, headless, record=False
    ):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.camera_handles = []

        print("ALIENGO INIT")
        # follow_cam, follow_trans = self.make_handle_trans(1920, 1080, 0, (1.0, -1.0, 0.0), (0.0, 0.0, 3*3.14/4))
        # follow_cam, follow_trans = self.make_handle_trans((1920, 1080), 0, (0.0, 0.0, 2.0), (0.0, 3.14/2, 0.0))
        # self.follow_cam = follow_cam
        # body_handle = self.gym.find_actor_rigid_body_handle(
        #     self.envs[0], self.actor_handles[0], "base"
        # )

        # self.gym.attach_camera_to_body(
        #     follow_cam,  # camera_handle,
        #     self.envs[0],
        #     body_handle,
        #     follow_trans,
        #     gymapi.FOLLOW_POSITION,
        # )

        self.init_aux_cameras(cfg.env.follow_cam, cfg.env.float_cam)

        if cfg.env.train_type == "lbc":
            # print("INITIALIZING 2 CAMERAS")
            for i in range(self.num_envs):
            for i in range(self.num_envs):
                
                res = cfg.env.camera_res
                cam1, trans1 = self.make_handle_trans(res[0], res[1], i, (0.35, 0.0, 0.0), (0.0, 3.14/6, 0))
                
                self.camera_handles.append(cam1)

                body_handle = self.gym.find_actor_rigid_body_handle(
                    self.envs[i], self.actor_handles[i], "base"
                )

                self.gym.attach_camera_to_body(
                    cam1,  # camera_handle,
                    self.envs[i],
                    body_handle,
                    trans1,
                    gymapi.FOLLOW_TRANSFORM,
                )


    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # Additionaly empty actuator network hidden states
        self.sea_hidden_state_per_env[:, env_ids] = 0.0
        self.sea_cell_state_per_env[:, env_ids] = 0.0

    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize actuator network hidden state tensors
        self.sea_input = torch.zeros(
            self.num_envs * self.num_actions,
            1,
            2,
            device=self.device,
            requires_grad=False,
        )
        self.sea_hidden_state = torch.zeros(
            2,
            self.num_envs * self.num_actions,
            8,
            device=self.device,
            requires_grad=False,
        )
        self.sea_cell_state = torch.zeros(
            2,
            self.num_envs * self.num_actions,
            8,
            device=self.device,
            requires_grad=False,
        )
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(
            2, self.num_envs, self.num_actions, 8
        )
        self.sea_cell_state_per_env = self.sea_cell_state.view(
            2, self.num_envs, self.num_actions, 8
        )

    def _compute_torques(self, actions):
        return super()._compute_torques(actions)
