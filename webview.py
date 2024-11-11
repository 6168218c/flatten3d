import time
import numpy as np
import torch
import torchvision
from gaussiansplatting.scene import Scene, GaussianModel
from gaussiansplatting.scene.cameras import Camera
from gaussiansplatting.scene.colmap_loader import rotmat2qvec
from gaussiansplatting.utils.general_utils import safe_state
from gaussiansplatting.utils.graphics_utils import getWorld2View2, getProjectionMatrix

import viser
import viser.transforms as tf
from dataclasses import dataclass, field
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.arguments import ArgumentParser, ModelParams, PipelineParams, get_combined_args

import numpy as np
import torch
import random

import math


def get_device():
    return torch.device(f"cuda")


from typing import Any
import torch
import numpy as np
import time
import viser
import viser.transforms as tf
from omegaconf import OmegaConf
from collections import deque

from gaussiansplatting.scene.cameras import Simple_Camera, C2W_Camera
from gaussiansplatting.gaussian_renderer import render


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def get_c2w(camera):
    c2w = np.zeros([4, 4], dtype=np.float32)
    c2w[:3, :3] = qvec2rotmat(camera.wxyz).T
    c2w[:3, 3] = camera.position
    c2w[3, 3] = 1.0

    c2w = torch.from_numpy(c2w).to("cuda")

    return c2w


class WebViewer:
    def __init__(self, scene:Scene, train_mode=False):
        self.device = "cuda:0"
        self.port = 8080

        self.scene = scene
        self.train_mode = train_mode

        self.render_times = deque(maxlen=3)
        self.server = viser.ViserServer(port=self.port)
        self.reset_view_button = self.server.gui.add_button("Reset View")

        self.toggle_axis = self.server.gui.add_checkbox(
            "Toggle Axis",
            initial_value=True,
        )

        self.need_update = False

        self.pause_training = False

        self.train_viewer_update_period_slider = self.server.gui.add_slider(
            "Update Period",
            min=1,
            max=100,
            step=1,
            initial_value=10,
            disabled=self.pause_training,
        )

        self.pause_training_button = self.server.gui.add_button("Pause Training")
        self.sh_order = self.server.gui.add_slider(
            "SH Order", min=1, max=4, step=1, initial_value=1
        )
        self.resolution_slider = self.server.gui.add_slider(
            "Resolution", min=384, max=4096, step=2, initial_value=1024
        )
        self.near_plane_slider = self.server.gui.add_slider(
            "Near", min=0.1, max=30, step=0.5, initial_value=0.1
        )
        self.far_plane_slider = self.server.gui.add_slider(
            "Far", min=30.0, max=1000.0, step=10.0, initial_value=1000.0
        )

        self.show_train_camera = self.server.gui.add_checkbox(
            "Show Train Camera", initial_value=False
        )

        self.fps = self.server.gui.add_text("FPS", initial_value="-1", disabled=True)

        self.axis = self.server.scene.add_frame("Axis", show_axes=True, axes_length=1000)

        self.time_bar = self.server.gui.add_slider(
            "Timestep", min=0, max=1000, step=1, initial_value=0, visible=False
        )

        self.renderer_output = self.server.gui.add_dropdown(
            "Renderer Output",
            [
                "render",
            ],
        )

        @self.renderer_output.on_update
        def _(_):
            self.need_update = True

        @self.show_train_camera.on_update
        def _(_):
            for frame in self.frames:
                frame.visible = self.show_train_camera.value
            self.need_update = True

        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True

        @self.near_plane_slider.on_update
        def _(_):
            self.need_update = True

        @self.far_plane_slider.on_update
        def _(_):
            self.need_update = True

        @self.pause_training_button.on_click
        def _(_):
            self.pause_training = not self.pause_training
            self.train_viewer_update_period_slider.disabled = not self.pause_training
            self.pause_training_button.name = (
                "Resume Training" if self.pause_training else "Pause Training"
            )

        @self.reset_view_button.on_click
        def _(_):
            self.need_update = True
            for client in self.server.get_clients().values():
                client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
                    [0.0, -1.0, 0.0]
                )

        @self.toggle_axis.on_update
        def _(_):
            self.need_update = True
            self.axis.show_axes = self.toggle_axis.value

        self.c2ws = []
        self.camera_infos = []
        self.render_cameras = None

        with torch.no_grad():
            self.frames = []
            random.seed(0)
            cams = self.scene.getTrainCameras()
            for i,cam in enumerate(cams):
                self.make_one_camera_pose_frame(cam,i)

        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            @client.camera.on_update
            def _(_):
                self.need_update = True

        self.debug_idx = 0


    def make_one_camera_pose_frame(self, cam: Camera, index):
        # wxyz = tf.SO3.from_matrix(cam.R.T).wxyz
        # position = -cam.R.T @ cam.T

        T_world_camera = tf.SE3.from_rotation_and_translation(tf.SO3.from_matrix(cam.R.T), cam.T).inverse()
        wxyz = T_world_camera.rotation().wxyz
        position = T_world_camera.translation()

        # breakpoint()
        frame = self.server.scene.add_frame(
            f"/colmap/frame_{cam.colmap_id}_{index}",
            wxyz=wxyz,
            position=position,
            axes_length=0.2,
            axes_radius=0.01,
            visible=False,
        )
        self.frames.append(frame)

        @frame.on_click
        def _(event: viser.GuiEvent):
            client = event.client
            assert client is not None
            T_world_current = tf.SE3.from_rotation_and_translation(tf.SO3(client.camera.wxyz), client.camera.position)

            T_world_target = tf.SE3.from_rotation_and_translation(
                tf.SO3(frame.wxyz), frame.position
            ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

            T_current_target = T_world_current.inverse() @ T_world_target

            for j in range(5):
                T_world_set = T_world_current @ tf.SE3.exp(T_current_target.log() * j / 4.0)

                with client.atomic():
                    client.camera.wxyz = T_world_set.rotation().wxyz
                    client.camera.position = T_world_set.translation()

                time.sleep(1.0 / 15.0)
            client.camera.look_at = frame.position

        if not hasattr(self, "begin_call"):

            def begin_trans(client):
                assert client is not None
                T_world_current = tf.SE3.from_rotation_and_translation(
                    tf.SO3(client.camera.wxyz), client.camera.position
                )

                T_world_target = tf.SE3.from_rotation_and_translation(
                    tf.SO3(frame.wxyz), frame.position
                ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

                T_current_target = T_world_current.inverse() @ T_world_target

                for j in range(5):
                    T_world_set = T_world_current @ tf.SE3.exp(T_current_target.log() * j / 4.0)

                    with client.atomic():
                        client.camera.wxyz = T_world_set.rotation().wxyz
                        client.camera.position = T_world_set.translation()
                client.camera.look_at = frame.position

            self.begin_call = begin_trans

    def get_kwargs(self):
        out = {}
        if hasattr(self, "time_bar"):
            out["timestep"] = self.time_bar.value
        if hasattr(self, "mask_thresh"):
            out["mask_thresh"] = self.mask_thresh.value
        if hasattr(self, "invert_mask"):
            out["invert_mask"] = self.invert_mask.value

        return out
    
    @property
    def camera(self):
        if len(list(self.server.get_clients().values())) == 0:
            return None
        if self.render_cameras is None:
            self.aspect = list(self.server.get_clients().values())[0].camera.aspect
            self.render_cameras = self.scene.getTrainCameras()
            self.begin_call(list(self.server.get_clients().values())[0])

        viser_cam = list(self.server.get_clients().values())[0].camera
        # viser_cam.up_direction = tf.SO3(viser_cam.wxyz) @ np.array([0.0, -1.0, 0.0])
        # viser_cam.look_at = viser_cam.position
        R = tf.SO3(viser_cam.wxyz).as_matrix()
        T = -R.T @ viser_cam.position
        # print(viser_cam.position)
        # T = viser_cam.position
        if self.render_cameras is None:
            fovy = viser_cam.fov
        else:
            fovy = self.render_cameras[0].FoVy

        fovx = 2 * math.atan(math.tan(fovy / 2) * self.aspect)
        # fovy = self.render_cameras[0].FoVy
        # fovx = self.render_cameras[0].FoVx
        # math.tan(self.render_cameras[0].FoVx / 2) / math.tan(self.render_cameras[0].FoVy / 2)
        # math.tan(fovx/2) / math.tan(fovy/2)

        # print(viser_cam.wxyz)

        # aspect = viser_cam.aspect
        width = int(self.resolution_slider.value)
        height = int(width / self.aspect)
        return Simple_Camera(0, R, T, fovx, fovy, height, width, "", 0)

    @torch.no_grad()
    def update(self, gaussian, pipe, background):
        if self.need_update:
            times = []
            for client in self.server.get_clients().values():
                camera = client.camera
                w = self.resolution_slider.value
                h = int(w / camera.aspect)
                # cam = Simple_Camera(0, )
                # cam = C2W_Camera(get_c2w(camera), camera.fov, h, w)
                cam = self.camera
                # c2w = torch.from_numpy(get_c2w(camera)).to(self.device)
                
                if cam is not None:
                    try:
                        start = time.time()
                        out = render(
                            cam,
                            gaussian,
                            pipe,
                            background,
                        )
                        self.renderer_output.options = list(out.keys())
                        out = (
                            out[self.renderer_output.value]
                            .detach()
                            .cpu()
                            .clamp(min=0.0, max=1.0)
                            .numpy()
                            * 255.0
                        ).astype(np.uint8)
                        end = time.time()
                        times.append(end - start)
                    except RuntimeError as e:
                        print(e)
                        continue
                    out = np.moveaxis(out.squeeze(), 0, -1)
                    client.set_background_image(out, format="jpeg")
                    del out

            self.render_times.append(np.mean(times))
            self.fps.value = f"{1.0 / np.mean(self.render_times):.3g}"

    def render_loop(self, gaussian, pipe, background):
        while True:
            try:
                self.update(gaussian, pipe, background)
                time.sleep(0.001)
            except KeyboardInterrupt:
                return


if __name__ == "__main__":
    import sys
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--ply_path", type=str, required=True, help="Path to the ply file")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    
    safe_state(args.quiet)
    
    def run_webview(
        dataset: ModelParams,
        ply_path: str,
        pipeline: PipelineParams,
    ):
        with torch.no_grad():
            gaussians = GaussianModel(dataset.sh_degree, 0, 0, 0)
            scene = Scene(dataset, gaussians, shuffle=False)
            gaussians.load_ply(ply_path)

            bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            
            webview = WebViewer(scene)
            webview.render_loop(gaussians, pipeline, background)
    
    run_webview(
        model.extract(args),
        args.ply_path,
        pipeline.extract(args),
    )
    