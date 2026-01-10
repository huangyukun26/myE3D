import argparse
import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = '1'
import subprocess
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Set

import cv2
import numpy as np
import open3d as o3d
import rembg

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

import pymeshlab
from PIL import Image, ImageFilter
from omegaconf import OmegaConf
from einops import rearrange, repeat
from objloader import Obj

from Configs import MainConfig, ProjectionConfig, CameraAngle
from FLUX.flux_refine_depth_exp_pipe import initialize_flux_pipeline, run_flux_pipeline
from normal_predictors import MarigoldPredictor, MockPredictor
from Projection import HeadlessProjectionMapping, HeadlessBaker
from Utils.sam import remove_bg, sam_init
from Utils.etc import (
    copy_texture_files,
    copy_obj_file,
    parse_target_angles,
    uncrop_image,
    move_existing_files,
    rename_existing_files,
    flush,
    dilate_mask,
    close_mask,
    generate_optional_schedule,
    generate_optional_schedule_from_files,
    RefineType,
)
from Utils.make_video import VideoRenderer
from Utils.smooth import L0Smoothing

from Poisson.mesh import Mesh
from Projection.scripts.remove_seen_faces import remove_seen_faces
from Utils.etc import normalize_angle

from Poisson.normal_blending import load_and_blend_maps
from Poisson import (
    convert_obj_to_ply,
    convert_ply_to_obj,
    process_poisson,
    render_depth_prior, 
    run_bilateral_integration,
)

class MeshRefinementPipeline:
    """
    A class-based pipeline to perform 3D object refinement using various tools:
    - FLUX for RGB refinement
    - SAM for background removal
    - Poisson/fusion steps for mesh updates
    - Optional schedules and steps as in the original script.
    """

    def __init__(
        self,
        obj_name: str,
        conf_name: str,
        device_idx: int,
        render_video: bool,
        bake_mesh: bool,
        normal_predictor: Optional[str] = None,
        view_manifest: Optional[str] = None,
        view_mode: Optional[str] = None,
        device: str = "cuda",
    ):
        """                                           
        Initialize the pipeline with given object name, config name, and device.

        :param obj_name: Name of the object to process.
        :param conf_name: Name of the YAML configuration (without extension).
        :param device: Device to run inference on ("cuda" or "cpu").
        """
        self.obj_name     = obj_name
        self.conf_name    = conf_name
        self.render_video = render_video
        self.bake_mesh    = bake_mesh
        self.device       = device
        self.device_idx   = device_idx
        self.normal_predictor_choice = normal_predictor
        self.view_manifest = view_manifest
        self.view_mode = view_mode

        # Will be set once the config is loaded in `run()`.
        self.cfg = None
        self.normal_predictor = None
        self.flux_pipe = None
        self.sam_predictor = None

        # Directories (member variables)
        self.exp_in_dir: Path = None
        self.exp_out_dir: Path = None
        self.mesh_dir: Path = None
        self.texture_dir: Path = None
        self.old_texture_dir: Path = None
        self.new_texture_dir: Path = None
        self.ref_texture_dir: Path = None
        #self.init_zoom_texture_dir: Path = None
        self.remeshing_dir: Path = None
        self.video_dir: Path = None
        self.textured_mesh_dir: Path = None
        self.poisson_save_dir: Path = None
        self.partial_meshes_dir: Path = None
        self.bini_save_dir: Path = None
        self.normal_dir: Path = None
        self.mask_dir: Path = None
        self.video_frame_dir: Path = None
        self.view_obs_dir: Path = None

        # These will be set once the mesh is loaded
        self.obj_mesh = None
        self.current_obj_fp = None
        self.coarse_obj_path = None
        self.bini_surfaces = []

        # Track initialization stage, angles seen, etc.
        self.is_init = True
        self.seen_angle_list: List["MeshRefinementPipeline.ViewInfo"] = []

        # Ref cond for grid
        self.np_image = None
        self.np_control = None
        self.latest_coarse_normal = None

    @dataclass(frozen=True)
    class ViewInfo:
        """View descriptor for refinement input selection."""
        yaw: float
        pitch: float
        image_path: Optional[Path] = None
        camera_location: Optional[List[float]] = None
        radius: Optional[float] = None
        ortho_scale: Optional[float] = None
        resolution: Optional[int] = None
        camera_type: Optional[str] = None
        track_axis: Optional[str] = None
        up_axis: Optional[str] = None
        axis_forward: Optional[str] = None
        axis_up: Optional[str] = None

    @staticmethod
    def load_config(*yaml_files, cli_args=[]):
        yaml_confs = [OmegaConf.load(f) for f in yaml_files]
        cli_conf = OmegaConf.from_cli(cli_args)
        conf = OmegaConf.merge(*yaml_confs, cli_conf)
        OmegaConf.resolve(conf)
        return conf

    def prepare_normal_predictor(self):
        """Initialize the normal predictor with fallback to a mock implementation."""
        predictor_choice = self.cfg.normal_predictor
        if predictor_choice is not None:
            predictor_choice = predictor_choice.lower()

        if predictor_choice == "auto":
            predictor_choice = None

        if predictor_choice == "mock":
            return MockPredictor(coarse_normal_provider=lambda: self.latest_coarse_normal)

        try:
            return MarigoldPredictor(self.cfg, self.device)
        except Exception as exc:
            print(f"WARN: Failed to initialize MarigoldPredictor ({exc}). Falling back to MockPredictor.")
            return MockPredictor(coarse_normal_provider=lambda: self.latest_coarse_normal)

    def load_view_manifest(self, manifest_path: Path) -> List["MeshRefinementPipeline.ViewInfo"]:
        """Load view metadata from a preprocess manifest with validation."""
        if not manifest_path.exists():
            raise FileNotFoundError(f"View manifest not found: {manifest_path}")

        manifest = json.loads(manifest_path.read_text())
        required_top = {"object_name", "normalized_obj_path", "axis_forward", "axis_up", "views"}
        missing_top = required_top - manifest.keys()
        if missing_top:
            raise ValueError(f"View manifest missing required fields: {sorted(missing_top)}")

        views = manifest.get("views")
        if not isinstance(views, list):
            raise ValueError("View manifest 'views' must be a list.")

        required_view = {
            "image_path",
            "azimuth_deg",
            "elevation_deg",
            "camera_location",
            "radius",
            "camera_type",
            "ortho_scale",
            "resolution",
            "track_axis",
            "up_axis",
        }
        view_infos: List[MeshRefinementPipeline.ViewInfo] = []
        for idx, view in enumerate(views):
            missing_view = required_view - view.keys()
            if missing_view:
                raise ValueError(f"View manifest entry {idx} missing fields: {sorted(missing_view)}")

            image_path = Path(view["image_path"])
            if not image_path.is_absolute():
                image_path = manifest_path.parent / image_path
            if not image_path.exists():
                raise FileNotFoundError(f"View image not found: {image_path}")

            view_infos.append(
                MeshRefinementPipeline.ViewInfo(
                    yaw=float(view["azimuth_deg"]),
                    pitch=float(view["elevation_deg"]),
                    image_path=image_path,
                    camera_location=list(view["camera_location"]),
                    radius=float(view["radius"]),
                    ortho_scale=float(view["ortho_scale"]),
                    resolution=int(view["resolution"]),
                    camera_type=str(view["camera_type"]),
                    track_axis=str(view["track_axis"]),
                    up_axis=str(view["up_axis"]),
                    axis_forward=str(manifest["axis_forward"]),
                    axis_up=str(manifest["axis_up"]),
                )
            )

        return view_infos

    @staticmethod
    def render_planar_projection(
        cfg,
        obj_mesh,
        texture_dir,
        pitch,
        yaw,
        output_dir,
        res=1024,
        zoom=1.0,
        thresh=0.7,
        device_idx=None,
    ):
        """
        Static utility method that renders a mesh from a given pitch/yaw and
        returns the zoom factor, refinement type, etc.
        """
        renderer = HeadlessProjectionMapping(
            vertex_shader_path=cfg.projection.vertex_shader_path,
            normal_fragment_shader_path=cfg.projection.normal_fragment_shader_path,
            obj_mesh=obj_mesh,
            texture_dir=texture_dir,
            device_idx=device_idx,
        )

        # Initial render to calculate zoom
        image = renderer.render(pitch=180 + pitch, yaw=yaw, img_res=(res, res), zoom=1.0)
        image.save(output_dir / "before_zoom.png")
        normal_image = renderer.render_normal(pitch=180 + pitch, yaw=yaw, img_res=(res, res), zoom=1.0)
        normal_image.save(output_dir / f"normal_before_zoom_{yaw:.1f}_{pitch:.1f}.png")

        zoom = renderer.adjust_camera_zoom(np.array(image), res, desired_ratio=0.90)

        # Define common rendering arguments
        render_kwargs = {
            "pitch": 180 + pitch,
            "yaw": yaw,
            "img_res": (res, res),
            "zoom": zoom,
        }

        # List of rendering functions and corresponding filename prefixes
        render_funcs = [
            ("render", "rgba"),
            ("render_normal", "normal"),
            ("render_depth", "depth"),
            ("render_cosine", "cosine"),
            ("render_cam_cos", "cam_cos"),
        ]

        # Dictionary to store rendered images
        rendered_images = {}

        output_dir.mkdir(parents=True, exist_ok=True)

        postfix = f"{yaw:.1f}_{pitch:.1f}_{zoom:.15f}"

        def render_and_save_image(renderer, func_name, filename_prefix, output_dir, **kwargs):
            render_func = getattr(renderer, func_name)
            image = render_func(**kwargs)
            image.save(output_dir / f"{filename_prefix}_{postfix}.png")
            return image

        # Render, crop, and save images
        for func_name, filename_prefix in render_funcs:
            image = render_and_save_image(
                renderer, func_name, filename_prefix, output_dir, **render_kwargs
            )
            rendered_images[filename_prefix] = image

        # Get the cosine image for further processing
        cosine_image = rendered_images["cosine"]

        cosine_thresh_mask = renderer.calculate_low_cosine_similarity_mask(cosine_image, thresh)
        cosine_thresh_mask.save(
            output_dir / f"before_morpho_cos_thresh_mask_{postfix}_thresh_{thresh}.png"
        )

        fg_pil = rendered_images["normal"].split()[-1]
        
        cosine_thresh_mask = dilate_mask(
            np.array(cosine_thresh_mask), np.array(fg_pil)
        )

        cosine_thresh_mask.save(
            output_dir / f"cos_thresh_mask_{postfix}_thresh_{thresh}.png"
        )

        # Compute the ratio of the inpaint mask to the foreground object
        fg_mask = np.array(fg_pil) > 0
        inpaint_mask = np.array(cosine_thresh_mask) > 0

        np_cam_cos_map = np.array(rendered_images["cam_cos"]) / 255.
        cos_weighted_inpaint_area = np_cam_cos_map[inpaint_mask].sum()
        fg_area = np.count_nonzero(fg_mask)

        if fg_area == 0:
            ratio = 0.0
        else:
            ratio = cos_weighted_inpaint_area / fg_area

        # Determine the refinement type based on the ratio
        if ratio >= cfg.significant_thresh:
            refine_type = RefineType.SIGNIFICANT
        elif ratio > cfg.minor_thresh:
            refine_type = RefineType.MINOR
        elif ratio > cfg.negligible_thresh:
            refine_type = RefineType.NEGLIGIBLE
        else:
            refine_type = RefineType.SKIP

        print(f"Inpaint mask ratio: {ratio:.4f}, Refinement type: {refine_type.name}")

        del renderer

        return zoom, postfix, refine_type, ratio, inpaint_mask, rendered_images

    def save_view_observation(
        self,
        postfix: str,
        yaw: float,
        pitch: float,
        depth_map: np.ndarray,
        depth_mask: np.ndarray,
        mv_mat,
        image_path: Optional[Path] = None,
        camera_location: Optional[List[float]] = None,
        radius: Optional[float] = None,
        ortho_scale: Optional[float] = None,
        resolution: Optional[int] = None,
        axis_forward: Optional[str] = None,
        axis_up: Optional[str] = None,
        normal_path: Optional[Path] = None,
        mask_path: Optional[Path] = None,
    ) -> None:
        """Persist per-view geometry observations for MVCC."""
        view_obs_dir = self.view_obs_dir or (self.exp_out_dir / "view_obs")
        view_obs_dir.mkdir(parents=True, exist_ok=True)

        depth_path = view_obs_dir / f"depth_{postfix}.npy"
        depth_mask_path = view_obs_dir / f"depthmask_{postfix}.npy"
        np.save(depth_path, depth_map.astype(np.float32))
        np.save(depth_mask_path, depth_mask.astype(np.uint8))

        meta = {
            "postfix": postfix,
            "yaw": float(yaw),
            "pitch": float(pitch),
            "image_path": str(image_path) if image_path else None,
            "camera_type": "ORTHO",
            "ortho_scale": ortho_scale if ortho_scale is not None else 1.0,
            "resolution": resolution if resolution is not None else self.cfg.im_res,
            "camera_location": camera_location,
            "radius": radius,
            "axis_forward": axis_forward if axis_forward is not None else "Z",
            "axis_up": axis_up if axis_up is not None else "Y",
            "mv_mat": mv_mat.tolist() if hasattr(mv_mat, "tolist") else mv_mat,
            "timestamp": datetime.now().isoformat(),
        }
        if normal_path is not None:
            meta["normal_path"] = str(normal_path)
        if mask_path is not None:
            meta["mask_path"] = str(mask_path)

        meta_path = view_obs_dir / f"meta_{postfix}.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"Saved view_obs: {depth_path.name}, {meta_path.name}")

    @staticmethod
    def compute_refinement_ratio(cfg, renderer, pitch, yaw, res=1024, thresh=0.7):
        """
        Render just enough to compute a ratio of how much needs refinement
        (low cosine values) vs. the total foreground.
        """
        image = renderer.render(
            pitch=180 + pitch, yaw=yaw, img_res=(res, res), zoom=1.0
        )
        zoom = renderer.adjust_camera_zoom(np.array(image), res, desired_ratio=0.90)

        # Render necessary images for ratio calculation
        render_kwargs = {
            "pitch": 180 + pitch,
            "yaw": yaw,
            "img_res": (res, res),
            "zoom": zoom,
        }

        normal_image  = renderer.render_normal(**render_kwargs)
        cosine_image  = renderer.render_cosine(**render_kwargs)
        cam_cos_image = renderer.render_cam_cos(**render_kwargs)

        cosine_thresh_mask = renderer.calculate_low_cosine_similarity_mask(
            cosine_image, thresh
        )
        fg_pil = normal_image.split()[-1]
        cosine_thresh_mask = dilate_mask(
            np.array(cosine_thresh_mask), np.array(fg_pil)
        )

        # Compute the ratio of the inpaint mask to the foreground object
        fg_mask = np.array(fg_pil) > 0
        inpaint_mask = np.array(cosine_thresh_mask) > 0
        
        np_cam_cos_map = np.array(cam_cos_image) / 255.
        cos_weighted_inpaint_area = np_cam_cos_map[inpaint_mask].sum()
        fg_area = np.count_nonzero(fg_mask)
        
        if fg_area == 0:
            ratio = 0.0
        else:
            ratio = cos_weighted_inpaint_area / fg_area

        if ratio >= cfg.significant_thresh:
            refine_type = RefineType.SIGNIFICANT
        elif ratio > cfg.minor_thresh:
            refine_type = RefineType.MINOR
        elif ratio > cfg.negligible_thresh:
            refine_type = RefineType.NEGLIGIBLE
        else:
            refine_type = RefineType.SKIP

        print(f"Inpaint mask ratio: {ratio:.4f}, Refinement type: {refine_type.name}")
        return ratio, refine_type

    def process_angle(
        self,
        angle: "MeshRefinementPipeline.ViewInfo",
        next_or_prev_angle: "MeshRefinementPipeline.ViewInfo",
        is_main_schedule=False,
    ):
        """
        Perform the entire pipeline for a single angle (pitch, yaw).
        This method does NOT take the member variables as arguments;
        it uses them directly (self.cfg, self.obj_mesh, self.texture_dir, etc.).
        """
        process_angle_start = time.perf_counter()
        yaw = angle.yaw
        pitch = angle.pitch
        if angle.image_path:
            print(f"Using view image path: {angle.image_path}")

        np_yaw   = next_or_prev_angle.yaw
        np_pitch = next_or_prev_angle.pitch

        print(f"Processing angle: pitch={pitch}, yaw={yaw}")
        print(f"Next or Prev is : pitch={np_pitch}, yaw={np_yaw}")

        # Create output dir for this angle
        output_dir = self.exp_out_dir / f"{yaw:.1f}_{pitch:.1f}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1) Render target view & compute ratio
        zoom, postfix, refine_type, ratio, inpaint_mask, target_renders = self.render_planar_projection(
            self.cfg,
            self.obj_mesh,
            self.texture_dir,
            pitch,
            yaw,
            output_dir,
            res=self.cfg.im_res,
            thresh=self.cfg.cos_thresh,
            device_idx=self.device_idx,
        )
        self.latest_coarse_normal = target_renders.get("normal")

        # 1-1) Render target view & compute ratio
        if self.is_init:
            _, _, _, _, _, np_renders = self.render_planar_projection(
                self.cfg,
                self.obj_mesh,
                self.texture_dir,
                np_pitch,
                np_yaw,
                output_dir,
                res=self.cfg.im_res,
                thresh=self.cfg.cos_thresh,
            )
        
        inpaint_mask_pil = Image.fromarray((inpaint_mask * 255).astype(np.uint8))
        inpaint_mask_np  = np.array(inpaint_mask_pil).astype(bool)

        step1_end = time.perf_counter()
        print(f"[Process Angle] Rendering target view took {step1_end - process_angle_start:0.4f} seconds")

        if refine_type == RefineType.SKIP:
            # Nothing to do
            return refine_type
        
        if self.cfg.ablation_tex != True:
            # 2) FLUX refinement
            output_flux_dir = output_dir / "flux"
            output_flux_dir.mkdir(parents=True, exist_ok=True)

            with torch.no_grad():
                if self.is_init:
                    next_rgb = run_flux_pipeline(
                        self.cfg,
                        my_pipe=self.flux_pipe,
                        exp_in_dir=self.exp_in_dir,
                        input_dir=output_dir,
                        ref_dir=self.ref_texture_dir,
                        output_subdir=output_flux_dir,
                        postfix=postfix,
                        is_init=self.is_init,
                        refine_type=refine_type,
                        np_image=np_renders['rgba'],
                        np_control=np_renders['depth'],
                        )

                else:
                    next_rgb = run_flux_pipeline(
                        self.cfg,
                        my_pipe=self.flux_pipe,
                        exp_in_dir=self.exp_in_dir,
                        input_dir=output_dir,
                        ref_dir=self.ref_texture_dir,
                        output_subdir=output_flux_dir,
                        postfix=postfix,
                        is_init=self.is_init,
                        refine_type=refine_type,
                        np_image=self.np_image,
                        np_control=self.np_control,
                        )
                
            if not self.cfg.flux.use_grid:
                next_rgb = next_rgb.resize(
                    (self.cfg.im_res, self.cfg.im_res), Image.Resampling.BICUBIC
                )
            else:
                next_rgb.save(output_dir / f"before_crop_refined_{yaw:.1f}_{pitch:.1f}.png")
                nr_w, nr_h = next_rgb.size
                next_rgb = next_rgb.crop((nr_w // 2, 0, nr_w, nr_h))
                next_rgb = next_rgb.resize(
                    (self.cfg.im_res, self.cfg.im_res), Image.Resampling.BICUBIC
                )
        else:
            unprocessed_rgb = Image.open(output_dir / f'rgba_{postfix}.png').convert('RGBA')

            # Define Background Application
            bg_color = self.cfg.flux.bg_color
            bg_color_options = self.cfg.flux.bg_color_options

            apply_background = lambda img: Image.alpha_composite(
                Image.new("RGBA", img.size, tuple(bg_color_options[bg_color])),
                img
            )
            
            next_rgb = apply_background(unprocessed_rgb)


        if self.is_init:
            self.np_image   = next_rgb
            self.np_control = Image.open(output_dir / f'depth_{postfix}.png').convert('RGB')

        #uncropped_inpaint_mask_pil.save(output_dir / f"uncropped_inpaint_mask_{yaw:.1f}_{pitch:.1f}.png")
        next_rgb.save(output_dir / f"refined_{yaw:.1f}_{pitch:.1f}.png")
        next_rgb.save(self.new_texture_dir / f"refined_{postfix}.png")
        next_rgb.save(self.ref_texture_dir / f"refined_{yaw:.1f}_{pitch:.1f}.png")
        next_rgb.save(output_dir / f"refined_rgb_{postfix}.png")
        flush()

        step2_end = time.perf_counter()
        print(f"[Process Angle] FLUX took {step2_end - step1_end:0.4f} seconds")

        if self.cfg.ablation_tex == True:
            next_rgba = unprocessed_rgb
        else:
            # 3) Background removal (SAM)
            with torch.no_grad():
                #next_rgb = Image.open(output_dir / f"refined_{yaw:.1f}_{pitch:.1f}.png")
                next_rgb = Image.open(output_dir / f"refined_rgb_{postfix}.png")
                next_rgba = remove_bg(self.sam_predictor, next_rgb)

        # 5) Move old textures, save new one
        move_existing_files(
            self.texture_dir,
            self.exp_out_dir / "old_textures",
            yaw,
            pitch,
            epsilon=1e-3
        )

        next_rgba.save(self.texture_dir / f"refined_{postfix}.png")
        #print(f"Saved refined RGBA image to {refined_rgba_path}")

        step3_end = time.perf_counter()
        print(f"[Process Angle] SAM and etc took {step3_end - step2_end:0.4f} seconds")

        if refine_type != RefineType.SIGNIFICANT:
            next_rgba = Image.fromarray(np.array(next_rgba) * inpaint_mask_np[..., None])
            next_rgba.save(self.texture_dir / f"refined_{postfix}.png")

        # If ratio is negligible, skip normal & Poisson
        if refine_type == RefineType.NEGLIGIBLE or self.cfg.ablation_geo:
            return refine_type

        # 6) Normal Prediction (Marigold)
        with torch.no_grad():
            normal_path = self.remeshing_dir / "normals" / f"normals_{postfix}.png"
            normal_img, extra = self.normal_predictor(next_rgb.convert("RGB"))
            _ = extra
            normal_img.save(normal_path)
            print(f"Saved normal map to {normal_path}")

        flush()

        # 7) Save alpha mask
        alpha_mask = next_rgba.split()[-1]  # RGBA alpha
        mask_path = self.remeshing_dir / "masks" / f"mask_{postfix}.png"
        alpha_mask.save(mask_path)
        print(f"Saved alpha mask to {mask_path}")

        flush()

        step4_end = time.perf_counter()
        print(f"[Process Angle] Marigold and etc took {step4_end - step3_end:0.4f} seconds")

        # 8) Render depth prior
        depth_map, depth_mask, mv_mat = render_depth_prior(
            obj_path=self.current_obj_fp,
            im_res=self.cfg.im_res, 
            pitch=pitch, 
            yaw=yaw,
            zoom=zoom,
            )

        step5_end = time.perf_counter()
        print(f"[Process Angle] Depth prior rendering took {step5_end - step4_end:0.4f} seconds")

        self.save_view_observation(
            postfix=postfix,
            yaw=yaw,
            pitch=pitch,
            depth_map=depth_map,
            depth_mask=depth_mask,
            mv_mat=mv_mat,
            image_path=angle.image_path,
            camera_location=angle.camera_location,
            radius=angle.radius,
            ortho_scale=angle.ortho_scale,
            resolution=angle.resolution,
            axis_forward=angle.axis_forward,
            axis_up=angle.axis_up,
            normal_path=self.remeshing_dir / "normals" / f"normals_{postfix}.png",
            mask_path=self.remeshing_dir / "masks" / f"mask_{postfix}.png",
        )

        # 8-1) Prepare normal maps

        blended_normal_map = load_and_blend_maps(base_map_path=output_dir / f"normal_{postfix}.png",
                                                 detail_map_path=normal_path,
                                                 l0_smoother=L0Smoothing())
        blended_normal_map_path = self.remeshing_dir / "normals" / f"blended_normals_{postfix}.png"

        blended_normal_map.save(blended_normal_map_path)
        
        # 9) Normal & depth prior fusion
        bini_surface, wu_wv_mask = run_bilateral_integration(
            save_path=self.bini_save_dir,
            #normal_map=normal_map,
            normal_path=blended_normal_map_path,
            # coarse_normal_path,
            normal_mask=np.array(alpha_mask).astype(bool),
            depth_map=depth_map,
            depth_mask=depth_mask, # Rendered coarse prior
            mv_mat=mv_mat,
            yaw=yaw,
            pitch=pitch,
            zoom=zoom,
            im_res=self.cfg.im_res,
            inpaint_mask=inpaint_mask_np * np.array(alpha_mask).astype(bool), # Inpaint target mask && actual result fg mask
            depth_lambda=self.cfg.poisson.bini_params.depth_lambda,
            depth_lambda2=self.cfg.poisson.bini_params.depth_lambda2,
            k=self.cfg.poisson.bini_params.k,
            iters=self.cfg.poisson.bini_params.iters,
            tol=self.cfg.poisson.bini_params.tol,
            cgiter=self.cfg.poisson.bini_params.cgiter,
            cgtol=self.cfg.poisson.bini_params.cgtol
        )
        
        # 9-0) Apply bini mask to refined image.
        next_rgba_np = np.array(next_rgba)
        next_rgba_np[..., -1] = next_rgba_np[..., -1] * wu_wv_mask
        next_rgba_wu_wv_mask = Image.fromarray(next_rgba_np)
        next_rgba_wu_wv_mask.save(output_dir / f"refined_wu_wv_{postfix}.png")
        next_rgba_wu_wv_mask.save(self.texture_dir / f"refined_{postfix}.png")
        
        # 9-1) bini postprocess
        bini_mesh = bini_surface.extract_surface().triangulate()
        bini_faces_as_array = bini_mesh.faces.reshape((bini_mesh.n_faces, 4))[:, 1:]
        bini_points_as_array = bini_mesh.points

        # Create a PyMeshLab MeshSet
        ms = pymeshlab.MeshSet()
        pm_mesh = pymeshlab.Mesh(vertex_matrix=bini_points_as_array, 
                                 face_matrix=bini_faces_as_array)
        ms.add_mesh(pm_mesh, "bini_mesh")

        # Remove small islands
        ms.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=pymeshlab.PercentageValue(5.),
            removeunref=True
        )
        
        # 5. Retrieve final mesh data
        final_mesh     = ms.current_mesh()
        final_vertices = final_mesh.vertex_matrix()
        final_faces    = final_mesh.face_matrix()

        ms.save_current_mesh(
            str(self.bini_save_dir / f"bini_mesh_k_{self.cfg.poisson.bini_params.k}_lambda1_{self.cfg.poisson.bini_params.depth_lambda}_{yaw:.1f}_{pitch:.1f}.ply"),
        )

        bini_mesh = Mesh(
            device='cpu',
            v=torch.Tensor(final_vertices).float(),
            f=torch.Tensor(final_faces).int(),
            )

        # 10) Accumulate bini surfaces
        self.bini_surfaces.append(bini_mesh)

        step6_end = time.perf_counter()
        print(f"[Process Angle] Bini took {step6_end - step5_end:0.4f} seconds")

        # 11) Poisson Pipeline: OBJ -> PLY -> Poisson -> Updated OBJ
        if not self.current_obj_fp.with_suffix('.ply').exists():
            ply_path = convert_obj_to_ply(self.current_obj_fp)
        else:
            ply_path = self.current_obj_fp.with_suffix('.ply')
    
        # WARN: 
        ply_path = process_poisson(
            angle,
            ply_path,
            #[bini_mesh],
            self.bini_surfaces,
            self.texture_dir,
            mask_path,
            normal_path,
            self.poisson_save_dir,
            self.partial_meshes_dir,
            #self.bini_save_dir,
            self.seen_angle_list,
            poisson_bin_fp=self.cfg.poisson.bin_fp,
            im_res=self.cfg.im_res,
            poisson_depth=self.cfg.poisson.poisson_depth,
            seen_thresh=self.cfg.poisson.bini_params.seen_thresh
        )

        updated_obj_path = convert_ply_to_obj(ply_path)
        self.current_obj_fp   = updated_obj_path
        self.obj_mesh         = Obj.open(updated_obj_path)

        # After processing the first angle, set init=False
        self.is_init = False

        step7_end = time.perf_counter()
        print(f"[Process Angle] Poisson took {step7_end - step6_end:0.4f} seconds")
        print(f"[Process Angle] In total took {step7_end - process_angle_start:0.4f} seconds")

        return refine_type

    def run(self):
        """
        Main execution pipeline:
        1. Load configs
        2. Prepare Marigold, FLUX, SAM
        3. Create directories
        4. Copy input object and texture files
        5. Process angles: main camera schedule, optional schedule
        6. Render videos (Optional)
        7. Bake final textured mesh (Optional)
        """
        # 1) Load config
        start_time = time.perf_counter()
        cfg_path = f"./Configs/{self.conf_name}.yaml"
        cfg = self.load_config(cfg_path)
        schema = OmegaConf.structured(MainConfig)
        self.cfg = OmegaConf.merge(schema, cfg)
        self.cfg.obj_name = self.obj_name
        if self.normal_predictor_choice is not None:
            self.cfg.normal_predictor = self.normal_predictor_choice
        if self.view_manifest is not None:
            self.cfg.view_manifest = self.view_manifest
        if self.view_mode is not None:
            self.cfg.view_mode = self.view_mode

        # 2) Initialize pipelines
        self.normal_predictor = self.prepare_normal_predictor()
        self.flux_pipe = initialize_flux_pipeline(self.cfg, self.device)
        self.sam_predictor = sam_init(self.cfg.sam_ckpt_path)

        # 3) Prepare I/O directories as member variables
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y%m%d_%H%M%S")

        self.exp_in_dir = self.cfg.input_base_dir / self.cfg.dataset / self.cfg.obj_name
        self.exp_out_dir = self.cfg.output_base_dir / self.cfg.dataset / self.cfg.obj_name / formatted_time
        self.mesh_dir = self.exp_out_dir / "meshes"
        self.texture_dir = self.exp_out_dir / "textures"
        self.old_texture_dir = self.exp_out_dir / "old_textures"
        self.new_texture_dir = self.exp_out_dir / "new_textures"
        self.ref_texture_dir = self.exp_out_dir / "ref_textures"
        #self.init_zoom_texture_dir = self.exp_out_dir / "zoom_init_texture"
        self.remeshing_dir = self.exp_out_dir / "remesh"
        self.video_dir = self.exp_out_dir / "video"
        self.textured_mesh_dir = self.exp_out_dir / "output_textured_mesh"
        self.poisson_save_dir = self.exp_out_dir / "poisson_surfaces"
        self.partial_meshes_dir = self.exp_out_dir / "partial_meshes"
        self.bini_save_dir = self.exp_out_dir / "bini_surfaces"
        self.normal_dir = self.remeshing_dir / "normals"
        self.mask_dir = self.remeshing_dir / "masks"
        self.video_frame_dir = self.video_dir / "frames"
        self.view_obs_dir = self.exp_out_dir / "view_obs"

        # Create directories
        for directory in [
            self.mesh_dir,
            self.texture_dir,
            self.old_texture_dir,
            self.new_texture_dir,
            self.ref_texture_dir,
            #self.init_zoom_texture_dir,
            self.remeshing_dir,
            self.normal_dir,
            self.mask_dir,
            self.video_dir,
            self.textured_mesh_dir,
            self.poisson_save_dir,
            self.partial_meshes_dir,
            self.bini_save_dir,
            self.video_frame_dir,
            self.view_obs_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Ensured directory exists: {directory}")
        shutil.copy(cfg_path, self.exp_out_dir / "cfg.yaml")

        self.coarse_obj_path = copy_obj_file(
            self.exp_in_dir,
            self.mesh_dir,
            new_name="coarse.obj",
        )
        self.current_obj_fp = self.coarse_obj_path
        self.obj_mesh = Obj.open(self.coarse_obj_path)

        # 5) Process camera schedules
        view_mode = (self.cfg.view_mode or "auto").lower()
        view_manifest_path = Path(self.cfg.view_manifest) if self.cfg.view_manifest else None
        if view_mode not in {"auto", "manifest", "procedural"}:
            raise ValueError(f"Unsupported view_mode: {view_mode}")

        use_manifest = False
        if view_mode == "manifest":
            if view_manifest_path is None:
                raise ValueError("view_mode is 'manifest' but no view_manifest is provided.")
            use_manifest = True
        elif view_mode == "auto":
            use_manifest = view_manifest_path is not None
        elif view_mode == "procedural" and view_manifest_path is not None:
            print("WARN: view_mode is 'procedural'; ignoring provided view_manifest.")

        if use_manifest:
            view_infos = self.load_view_manifest(view_manifest_path)
            if not view_infos:
                raise ValueError("View manifest is empty.")
            print(f"Using manifest views: {len(view_infos)} views from {view_manifest_path}")
        else:
            camera_schedule = self.cfg.camera_schedule
            if not camera_schedule:
                raise ValueError("Camera schedule is empty. Please define camera angles in the configuration.")
            view_infos = [
                MeshRefinementPipeline.ViewInfo(yaw=entry["yaw"], pitch=entry["pitch"])
                for entry in camera_schedule
            ]

        # Build a set of angles from the main schedule
        camera_angles = set((view.yaw % 360, view.pitch) for view in view_infos)

        # 4) Copy textures and OBJ
        copy_texture_files(self.exp_in_dir, self.texture_dir, len(camera_angles))

        # Generate optional schedules
        schedules = generate_optional_schedule_from_files(self.texture_dir)
        train_schedule = schedules["train"]
        test_schedule = schedules["test"]

        # Additional angles to exclude
        excluded_angles: Set[Tuple[float, float]] = {
            (0.0, 89.9),
            (0.0, -89.9),
        }
        # Combine
        exclusion_set = camera_angles.union(excluded_angles)

        # Filter train schedule
        filtered_train_schedule_dicts = [
            entry for entry in train_schedule
            if (entry['yaw'] % 360, entry['pitch']) not in exclusion_set
        ]
        filtered_train_schedule: List[CameraAngle] = [
            CameraAngle(yaw=entry['yaw'], pitch=entry['pitch']) for entry in filtered_train_schedule_dicts
        ]

        prepare_end = time.perf_counter()
        print(f"Refinement preparation took {prepare_end - start_time:0.4f} seconds")

        # Main schedule angles
        for i, angle in enumerate(view_infos):
            # WARN: This assumes main schedule will have at least 2 views
            if i == 0:
                if len(view_infos) < 2:
                    raise ValueError("Main schedule must include at least 2 views.")
                self.next_or_prev_angle = view_infos[i + 1]
            rename_existing_files(self.texture_dir, angle.yaw, angle.pitch)
            cur_refine_type = self.process_angle(angle, self.next_or_prev_angle, is_main_schedule=True)
            if cur_refine_type != RefineType.SKIP:
                self.next_or_prev_angle = angle
                self.seen_angle_list.append(angle)

        main_end = time.perf_counter()
        print(f"Main refinement took {main_end - prepare_end:0.4f} seconds")

        # Optional schedule
        while filtered_train_schedule:
            ratios = []
            renderer = HeadlessProjectionMapping(
                vertex_shader_path=self.cfg.projection.vertex_shader_path,
                normal_fragment_shader_path=self.cfg.projection.normal_fragment_shader_path,
                obj_mesh=self.obj_mesh,
                texture_dir=self.texture_dir,
                device_idx=self.device_idx,
            )
            # Evaluate each remaining angle
            for angle in filtered_train_schedule:
                ratio, refine_type = self.compute_refinement_ratio(
                    self.cfg,
                    renderer,
                    angle.pitch,
                    angle.yaw,
                    res=self.cfg.im_res,
                    thresh=self.cfg.cos_thresh,
                )
                if refine_type != RefineType.SKIP:
                    ratios.append((ratio, angle))

            if not ratios:
                print("No more angles to refine in optional_schedule.")
                break

            # Select angle with the largest ratio
            ratios.sort(key=lambda x: x[0], reverse=True)
            max_ratio, max_angle = ratios[0]
            print(f"Processing optional angle: pitch={max_angle.pitch}, yaw={max_angle.yaw} with ratio: {max_ratio:.4f}")

            # Rename & process
            rename_existing_files(self.texture_dir, max_angle.yaw, max_angle.pitch, refine_type)
            self.seen_angle_list.append(max_angle)
            cur_refine_type = self.process_angle(
                MeshRefinementPipeline.ViewInfo(yaw=max_angle.yaw, pitch=max_angle.pitch),
                self.next_or_prev_angle,
                is_main_schedule=False,
            )

            # Remove processed angle
            filtered_train_schedule.remove(max_angle)
            self.next_or_prev_angle = MeshRefinementPipeline.ViewInfo(
                yaw=max_angle.yaw,
                pitch=max_angle.pitch,
            )

            del renderer

        mid_time = time.perf_counter()
        print(f"Optional Refinement took {mid_time - main_end:0.4f} seconds")
        print(f"Total Refinement took {mid_time - start_time:0.4f} seconds")

        # 6) Render final video (Optional)
        if self.render_video:
            video_renderer = VideoRenderer(
                vertex_shader_path=self.cfg.projection.vertex_shader_path,
                normal_fragment_shader_path=self.cfg.projection.normal_fragment_shader_path,
                obj_mesh=self.obj_mesh,
                texture_dir=self.texture_dir,
                output_dir=self.video_dir,
                video_frame_dir=self.video_frame_dir,
                res=1024,
            )
            postfixes = video_renderer.render_video_frames(
                num_frames=360,
                num_spirals=2,
                initial_pitch=89.9,
                final_pitch=-89.9,
                zoom=1.0 / 1.1
            )
            video_paths = video_renderer.create_videos(postfixes=postfixes, fps=30)

            end_time = time.perf_counter()
            print(f"Video rendering took {end_time - mid_time:0.4f} seconds")
            print("Videos saved:", video_paths)

        # 7) Bake final textured mesh (Optional)
        if self.bake_mesh:
            bake_start = time.perf_counter()

            # Create a PyMeshLab MeshSet
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(self.current_obj_fp))
            # Decimiate
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=100_000,
                targetperc=0.,
                preserveboundary=True,
                preservenormal=True,
                optimalplacement=True,
                qualitythr=1.0,
                autoclean=True
            )
            self.current_obj_fp = self.current_obj_fp.parent / f"final_deci_mesh.obj"
            ms.save_current_mesh(str(self.current_obj_fp))
            unwrapped_obj_path = HeadlessBaker.unwrap_mesh_with_xatlas(self.current_obj_fp, self.textured_mesh_dir)
            unwrapped_obj = Obj.open(unwrapped_obj_path)

            baker = HeadlessBaker(
                self.cfg.projection.bake_vertex_shader_path,
                unwrapped_obj,
                self.texture_dir,
                self.device_idx,
            )
            baked_image = baker.bake_texture(uv_texture_size=(1024 * 4, 1024 * 4))
            baked_image = baker.pad_uvs_cupy(baked_image)
            baked_image.save(self.textured_mesh_dir / "baked_texture_map.png")

            baker.assign_baked_texture_to_mesh(
                obj_path=unwrapped_obj_path,
                baked_texture_filename=self.textured_mesh_dir / "baked_texture_map.png",
            )

            bake_end = time.perf_counter()
            print(f"Baking took {bake_end - bake_start:0.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Mesh Refinement Pipeline.")
    parser.add_argument('--obj_name', type=str, required=True, help='Name of the object to process.')
    parser.add_argument('--conf_name', type=str, required=True, help='Name of the YAML config (without extension).')
    parser.add_argument('--device_idx', type=int, required=True, help='Bake into textured mesh at the end?')
    parser.add_argument('--render_video', action='store_true', help='Render video at the end?')
    parser.add_argument('--bake_mesh', action='store_true', help='Bake into textured mesh at the end?')
    parser.add_argument(
        '--normal_predictor',
        type=str,
        choices=['auto', 'marigold', 'mock'],
        default='auto',
        help='Normal predictor selection.',
    )
    parser.add_argument('--view_manifest', type=str, default=None, help='Path to view manifest JSON.')
    parser.add_argument(
        '--view_mode',
        type=str,
        choices=['auto', 'manifest', 'procedural'],
        default='auto',
        help='View selection mode.',
    )
    args = parser.parse_args()

    pipeline = MeshRefinementPipeline(
            obj_name=args.obj_name, 
            conf_name=args.conf_name,
            device_idx=args.device_idx,
            render_video=args.render_video,
            bake_mesh=args.bake_mesh,
            normal_predictor=args.normal_predictor,
            view_manifest=args.view_manifest,
            view_mode=args.view_mode,
            )
    pipeline.run()
