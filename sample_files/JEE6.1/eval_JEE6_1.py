"""
GS View Rendering and Evaluation
Contributors: Alexandre Zaghetto

- Addapted from  Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey Hu, Matthew Tancik, & Angjoo Kanazawa (2024). 
gsplat: An Open-Source Library for Gaussian Splatting. 

Description: 
    This script performs the rendering and evaluation of Gaussian Splat (GS) models, 
    particularly for 3D point cloud data. It allows for the conversion of input PLY files into GS-based splats, 
    renders them along a specified trajectory, and computes various performance metrics such as PSNR, SSIM, and LPIPS 
    for evaluating the quality of rendered views.

Key Features:
- Loads GS model data from PLY file.
- Renders views along different types of motion paths.
- Supports evaluation of rendered images using metrics like LPIPS, PSNR, and SSIM.
- Saves results, including rendered images and metrics.

Notes:

arXiv preprint arXiv:2409.06765 (https://arxiv.org/abs/2409.06765)
- Ensure that the required dependencies are installed.
- Adjust paths and dataset settings as needed.
"""

import json
import math
import os
import time
from collections import defaultdict
from argparse import ArgumentParser
from plyfile import PlyData

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_interpolated_path,
    generate_ellipse_path_z,
    generate_spiral_path,
)

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from gsplat.rendering import rasterization

# JEE 6.1
import sys
from pathlib import Path
# Get the absolute path of the gsplat folder relative to the current script
current_folder = Path(__file__).resolve().parent
gsplat_path = current_folder.parent.parent / 'gsplat'
# Insert the relative gsplat path into sys.path
sys.path.insert(0, str(gsplat_path))

# 
def parse_args():

    parser = ArgumentParser("Render GS Views")
    
    # Input and Output Paths
    parser.add_argument("--input_ply", type=str, help="Input GS model", 
                        default="./results/bonsai/point_cloud/iteration_30000/point_cloud.ply")
    parser.add_argument("--render_traj_path", type=str, help="Render trajectory path", 
                        default="interp")
    parser.add_argument("--colmap_data_dir", type=str, help="Path to colmap dataset", 
                        default="/media/zaghetto/SonyHD/DATA/gaussian_splatting/INRIA_COLMAP/bonsai/")
    parser.add_argument("--result_dir", type=str, help="Directory to save results", 
                        default="./results/debug")
        
    # Dataset and Sampling
    parser.add_argument("--data_factor", type=int, help="Downsample factor for the dataset", 
                        default=4)
    parser.add_argument("--test_every", type=int, help="Every N images there is a test image", 
                        default=8)

    # Scene and Camera Configuration
    parser.add_argument("--global_scale", type=int, help="Global scaler for scene size parameters", 
                        default=1)
    parser.add_argument("--normalize_world_space", type=bool, help="Normalize the world space", 
                        default=True)
    parser.add_argument("--camera_model", type=str, help="Camera model", 
                        default="pinhole") # ["pinhole", "ortho", "fisheye"]
    parser.add_argument("--near_plane", type=float, help="Near plane clipping distance", 
                        default=0.01)
    parser.add_argument("--far_plane", type=float, help="Far plane clipping distance", 
                        default=1e10)
        
    # Gassian Parameters
    parser.add_argument("--sh_degree", type=int, help="Degree of spherical harmonics", 
                        default=3)

    # Training Parameters
    parser.add_argument("--step", type=int, help="Train step", 
                        default=30000)

    # Evaluation and Metrics
    parser.add_argument("--lpips_net", type=str, help="LPIPS network", 
                        default="alex") # ["vgg", "alex"]
                                                              
    return parser.parse_args()

# Conversion function
def metrics_per_view(data, prefix="ours_30000"):    
    converted = {prefix: {}}
    for metric, values in data.items():
        converted[prefix][metric.upper()] = {
            f"{i:05d}.png": float(value) for i, value in enumerate(values)
        }
    return converted

def print_progress_bar(iteration, total, bar_length=40):
    """
    Prints a progress bar for tracking progress in loops.
    
    Args:
    - iteration (int): Current iteration of the loop.
    - total (int): Total number of iterations.
    - bar_length (int): Length of the progress bar (default is 40).
    """
    percent = (iteration / total)
    arrow = '=' * int(round(percent * bar_length) - 1)
    spaces = ' ' * (bar_length - len(arrow))
    progress = f"[{arrow}{spaces}] {percent * 100:.1f}%"
    
    # Print the progress bar, overwrite it at the same line
    sys.stdout.write(f"\r{progress} ")
    sys.stdout.flush()


def load_gsplat_ply(input_path, device):
    """
    Load and process a PLY file to extract geometric and feature information for GSplat processing.

    Args:
    - input_path (str): The file path to the input PLY file.
    - device (torch.device): The device (CPU or GPU) where the data tensors will be loaded.

    Returns:
    - gs (torch.Tensor): The GSplat data structure representing points and features.
    - sh_degree (int): The spherical harmonics degree used for the input features.
    """

    plydata = PlyData.read(input_path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])        

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    
    max_sh_degree = 3
    assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    
    features_dc = features_dc.reshape(features_dc.shape[0], 1, features_dc.shape[1])
    features_extra = features_extra.transpose(0, 2, 1)
                        
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
    xyz = torch.tensor(xyz, dtype=torch.float, device=device)
    features_dc = torch.tensor(features_dc, dtype=torch.float, device=device)
    features_extra = torch.tensor(features_extra, dtype=torch.float, device=device)
    colors = torch.cat([features_dc, features_extra], dim=-2)
            
    rots = torch.tensor(rots, dtype=torch.float, device=device)
    scales = torch.tensor(scales, dtype=torch.float, device=device)
    opacities = torch.tensor(opacities, dtype=torch.float, device=device).flatten()
    
    gs_params_dict = {
            "points": xyz,
            "colors": colors,
            "scales": scales,
            "quats": rots,
            "opacities": opacities
    }
                        
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)             
    gs = create_splats(gs_params_dict, device)
                    
    return gs, sh_degree
    

def create_splats(gs_params, device: str = "cuda"):
    """
    Creates a GSplat structgure as PyTorch tensors using the input 
    data for points, scales, quaternions, opacities, and spherical harmonics. The function returns 
    a dictionary of these parameters that can be optimized during training.

    Args:
    - gs_params (dict): A dictionary containing the GSplat data:
        - "points" (torch.Tensor): The 3D points for the splats (shape: [N, 3]).
        - "scales" (torch.Tensor): Scaling factors (shape: [N, 3]).
        - "quats" (torch.Tensor): Quaternion rotations (shape: [N, 4]).
        - "opacities" (torch.Tensor): Opacity values (shape: [N]).
        - "colors" (torch.Tensor): Colors encoded in spherical harmonics (shape: [N, (sh_degree + 1)^2, 3]).
    - device (str, optional): The device to place the tensors on (default is "cuda"). 

    Returns:
    - torch.nn.ParameterDict: A dictionary of PyTorch `Parameter` objects that hold the model parameters 
      such as points, scales, quaternions, opacities, and spherical harmonic coefficients. These are 
      set to the device specified and ready for optimization.
    """
    
    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(gs_params["points"]), 1.6e-4),
        ("scales", torch.nn.Parameter(gs_params["scales"]), 5e-3),
        ("quats", torch.nn.Parameter(gs_params["quats"]), 1e-3),
        ("opacities", torch.nn.Parameter(gs_params["opacities"]), 5e-2),
    ]
    params.append(("sh0", torch.nn.Parameter(gs_params["colors"][:, :1, :]), 2.5e-3))
    params.append(("shN", torch.nn.Parameter(gs_params["colors"][:, 1:, :]), 2.5e-3 / 20))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    return splats

def crop_to_macro_block(canvas, block_size=16):

    """
    Crops the input canvas (frame) to match the nearest macro block size, ensuring that 
    the resulting dimensions are divisible by the specified block size.

    Args:
    - canvas (torch.Tensor or np.ndarray): The input frame to be cropped.
    - block_size (int, optional): The size of the block to crop the image to (default is 16). 
                                  The final image dimensions will be adjusted to be multiples of this value.

    Returns:
    - torch.Tensor or np.ndarray: The cropped canvas with dimensions that are multiples of `block_size`.
    """
    
    height, width = canvas.shape[:2]
    new_width = width // block_size * block_size
    new_height = height // block_size * block_size
    canvas_cropped = canvas[:new_height, :new_width]
    return canvas_cropped

class Runner:
    """Engine for testing."""

    def __init__(self, cfg):
        
        self.cfg = cfg            
        self.device = "cuda"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.stats_dir = f"{cfg.result_dir}/metrics"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        
        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.colmap_data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )

        self.valset = Dataset(self.parser, split="val")
        
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
    
        # Model        
        self.splats = create_splats(self.initialize_gs(cfg.sh_degree), device=self.device)
        print("Model initialized.")

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

    def initialize_gs(self, sh_degree=3):
        """
        Initializes a GSplat data structure with default values for points, colors, scales, quaternions, 
        and opacities. This is typically used to create an empty or placeholder GSplat object that can 
        later be filled with actual data.
        """
            
        points    = torch.zeros((1, 3))
        colors    = torch.zeros((1, (sh_degree + 1) ** 2, 3))
        scales    = torch.zeros((1, 3))
        quats     = torch.zeros((1, 4))
        opacities = torch.zeros((1))
        
        gs_params_dict = {
            "points": points,
            "colors": colors,
            "scales": scales,
            "quats": quats,
            "opacities": opacities
        }
        
        return gs_params_dict

    def rasterize_splats(self, camtoworlds, Ks, width, height, masks=None, **kwargs):
        """
        This method rasterizes the splats and optionally applies a mask to the rendered colors.                 
        """
        
        splats = self.splats
        means, quats, scales, opacities = (
            splats["means"],  # [N, 3]
            splats["quats"],  # [N, 4]
            torch.exp(splats["scales"]),  # [N, 3]
            torch.sigmoid(splats["opacities"]),  # [N,]
        )
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)  # [N, K, 3]
                
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=False,
            absgrad=False,
            sparse_grad=False,
            rasterize_mode="classic",
            distributed=False,
            camera_model=self.cfg.camera_model,
            **kwargs,
        )
        
        if masks is not None:
            render_colors[~masks] = 0
            
        return render_colors, render_alphas, info
    
    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """
        This method performs evaluation of the model on a validation set. It computes metrics such as PSNR, SSIM, and LPIPS, 
        and saves the rendered and ground truth images at each step. The method also computes and reports the average time 
        taken for each rendering operation. Results are saved as JSON files for both overall statistics and per-view metrics.
        """
        
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        
        ellipse_time = 0
        metrics = defaultdict(list)
        
        for i, data in enumerate(valloader):
            print_progress_bar(i+1, len(valloader))
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            imageio.imwrite(
                f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                canvas,
            )
            
            os.makedirs(os.path.join(self.render_dir,'gt'), exist_ok=True)
            # write gt
            canvas = pixels.squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            imageio.imwrite(
                f"{self.render_dir}/gt/{i:04d}.png",
                canvas,
            )
            
            os.makedirs(os.path.join(self.render_dir,'render'), exist_ok=True)
             # write render
            canvas = colors.squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            imageio.imwrite(
                f"{self.render_dir}/render/{i:04d}.png",
                canvas,
            )
            
            pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]

            metrics["psnr"].append(self.psnr(colors_p, pixels_p))
            metrics["ssim"].append(self.ssim(colors_p, pixels_p))
            metrics["lpips"].append(self.lpips(colors_p, pixels_p))
        
        ellipse_time /= len(valloader)

        stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
        stats.update(
            {
                "ellipse_time": ellipse_time,
                "num_GS": len(self.splats["means"]),
            }
        )
        
        print(
            f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
            f"Time: {stats['ellipse_time']:.3f}s/image "
            f"Number of GS: {stats['num_GS']}"
        )
        
        # save average results as json
        with open(f"{self.stats_dir}/step{step:04d}_average_results.json", "w") as f:
            json.dump(stats, f)

        # save results per view as json
        stats_per_view = metrics_per_view(metrics, f"ours_{step:04d}")
        with open(f"{self.stats_dir}/step{step:04d}_per_view.json", "w") as f:
            json.dump(stats_per_view, f, indent=4)            

    @torch.no_grad()
    def render_trajectory(self, step: int):
        """        
        This method generates a trajectory path based on a specified path type (e.g., interpolation, ellipse, spiral), 
        then renders images for each point along the path. These images are saved as three separate videos:
        - A merged video containing both the RGB image and depth information.
        - A color-only video showing the RGB image.
        - A depth-only video showing the normalized depth information.
        """
        
        def prepare_video_writer(name, block_size=16):
            """Helper to initialize a video writer."""
            return imageio.get_writer(
                f"{video_dir}/{cfg.render_traj_path}_{step}_{name}.mp4", 
                fps=30, 
                macro_block_size=block_size
            )
        
        def append_frame(writer, frame, block_size=16):
            """Helper to process and append a frame to a writer."""
            frame = (frame * 255).astype(np.uint8)
            writer.append_data(crop_to_macro_block(frame, block_size))

        # Configuration and setup
        cfg = self.cfg
        device = self.device
        camtoworlds_all = self.parser.camtoworlds

        # Generate trajectory path
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(camtoworlds_all, 1)
        elif cfg.render_traj_path == "ellipse":
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, 
                height=camtoworlds_all[:, 2, 3].mean()
            )
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(f"Unsupported render trajectory type: {cfg.render_traj_path}")

        # Convert to homogeneous coordinates and prepare tensors
        camtoworlds_all = np.concatenate(
            [camtoworlds_all, np.tile([[[0.0, 0.0, 0.0, 1.0]]], (len(camtoworlds_all), 1, 1))],
            axis=1,
        )
        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # Create video writers
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer_merged = prepare_video_writer("merged")
        writer_depth = prepare_video_writer("depth")
        writer_color = prepare_video_writer("color")

        # Render and save frames
        for camtoworlds in tqdm.tqdm(camtoworlds_all, desc="Rendering trajectory"):
            camtoworlds = camtoworlds[None]  # Add batch dimension
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )

            colors = torch.clamp(renders[..., :3], 0.0, 1.0)
            depths = renders[..., 3:4]
            depths = ((depths - depths.min()) / (depths.max() - depths.min()))
            merged = torch.cat([colors, depths.repeat(1, 1, 1, 3)], dim=2).squeeze(0).cpu().numpy()

            append_frame(writer_color, colors.squeeze(0).cpu().numpy())
            append_frame(writer_depth, depths.squeeze(0).cpu().numpy())
            append_frame(writer_merged, merged)

        # Finalize and save videos
        writer_merged.close()
        writer_depth.close()
        writer_color.close()
        print(f"Videos saved to {video_dir}")


if __name__ == "__main__":
    """
    Main script for initializing and running the Gaussian Splatting Model for rendering.
    This script loads point cloud data from a PLY file, initializes the renderer, and performs
    the rendering process to evaluate the model.
    
    Usage example:
    python JEE6_1_eval.py --colmap_data_dir ./INRIA_COLMAP/bonsai/ --data_factor 4 --result_dir ./results/bonsai/evaluation
    """

    cfg = parse_args()
    
    # Initializa GS Model with ply data
    renderer = Runner(cfg)    
    gs_model, sh_degree = load_gsplat_ply(cfg.input_ply, "cuda")                
    for k in renderer.splats.keys():
        renderer.splats[k].data = gs_model[k]        
    print("Number of Gaussians:", len(renderer.splats["means"]))

    # Evaluate
    renderer.eval(step=cfg.step)
    renderer.render_trajectory(step=cfg.step)