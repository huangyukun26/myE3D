from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Union

@dataclass
class CameraAngle:
    yaw: float
    pitch: float

@dataclass
class MyFluxPipeline:
    base_pipe: Any = None
    redux_pipe: Any = None

@dataclass
class BaseConfig:
    input_base_dir: Path 
    output_base_dir: Path
    sam_ckpt_path: Path
    dataset: str = "OmniObject3D"
    obj_name: str = "toy_animal"
    camera_schedule: List[dict] = field(default_factory=lambda: [
        {"yaw": 0.0, "pitch": 15.0},
        # Add more camera angles as needed
    ])
    optional_schedule: List[Dict[str, float]] = field(default_factory=list)  # To be populated
    test_schedule: List[Dict[str, float]] = field(default_factory=list)  # To be populated
    seed: int = 0
    im_res: int = 768
    cos_thresh: float = 0.5

    significant_thresh: float = 0.500
    minor_thresh: float = 0.050
    negligible_thresh: float = 0.010

    ablation_tex: bool = False
    ablation_geo: bool = False
    normal_predictor: Optional[str] = None

@dataclass
class FluxPipelineParameters:
    # Shared parameters for anchor and non_anchor
    strength: float
    guidance_scale: float
    low_freq_ratio: float
    replace_steps: float
    replace_limit: float
    stop_replace_steps: float
    replace_type: str
    steps: int
    gamma: float

@dataclass
class BiniParameters:
    depth_lambda: float
    depth_lambda2: float
    k: float
    iters: int
    tol: float
    cgiter: int
    cgtol: float
    seen_thresh: float

@dataclass
class FluxConfig:
    # Model Paths
    base_pipeline_path:  Optional[Union[str, Path]] = None
    depth_pipeline_path: Optional[Union[str, Path]] = None
    redux_pipeline_path: Optional[Union[str, Path]] = None

    use_im_prompt: bool = True
    use_simple_prompt: bool = True
    use_grid: bool = True
    ref_sample_batch: int = 8

    # Pipeline Parameters
    significant: FluxPipelineParameters = FluxPipelineParameters(
        strength=1.0,
        guidance_scale=3.5,
        low_freq_ratio=0.07,
        replace_steps=12,
        replace_limit=840,
        stop_replace_steps=25,
        replace_type='lf',
        steps=30,
        gamma=0.1
    )
    minor: FluxPipelineParameters = FluxPipelineParameters(
        strength=0.6,
        guidance_scale=3.5,
        low_freq_ratio=0.07,
        replace_steps=11,
        replace_limit=840,
        stop_replace_steps=30,
        replace_type='lf',
        steps=30,
        gamma=0.1
    )
    negligible: FluxPipelineParameters = FluxPipelineParameters(
        strength=0.3,
        guidance_scale=3.5,
        low_freq_ratio=0.065,
        replace_steps=0,
        replace_limit=840,
        stop_replace_steps=30,
        replace_type='all',
        steps=30,
        gamma=0.1
    )
    
    # Background Colors
    bg_color_options: Dict[str, List[int]] = field(default_factory=lambda: {
        "green": [0, 255, 0, 255],
        "gray":  [127, 127, 127, 255],
        "black": [0, 0, 0, 255],
        "white": [255, 255, 255, 255]
    })
    bg_color: str = "white"

    def __post_init__(self):
        if self.base_pipeline_path and isinstance(self.base_pipeline_path, str):
            self.base_pipeline_path = Path(self.base_pipeline_path)
        if self.depth_pipeline_path and isinstance(self.depth_pipeline_path, str):
            self.depth_pipeline_path = Path(self.depth_pipeline_path)
        if self.redux_pipeline_path and isinstance(self.redux_pipeline_path, str):
            self.redux_pipeline_path = Path(self.redux_pipeline_path)


@dataclass
class MonoConfig:
    checkpoint: str = "GonzaloMG/marigold-e2e-ft-depth"
    denoise_steps: int = 1
    ensemble_size: int = 1
    half_precision: bool = False
    timestep_spacing: str = "trailing"
    processing_res: int = 768
    output_processing_res: bool = False
    resample_method: str = "bilinear"
    color_map: str = "Spectral"
    seed: Optional[int] = None
    batch_size: int = 0
    apple_silicon: bool = False
    noise: str = "zeros"
    modality: str = "depth"

@dataclass
class ProjectionConfig:
    vertex_shader_path: str = "./Projection/shaders/vertex_shader.glsl"
    bake_vertex_shader_path: str = "./Projection/shaders/vertex_shader_uv.glsl"
    normal_fragment_shader_path: str = "./Projection/shaders/normal_fragment_shader.glsl"

@dataclass
class PoissonConfig:
    bin_fp: Optional[str] = None
    poisson_depth: int = 9
    bini_params: BiniParameters = BiniParameters(
        depth_lambda=5e-3,
        depth_lambda2=1e-1,
        k=0.1,
        iters=150,
        tol=1e-4,
        cgiter=5000,
        cgtol=1e-3,
        seen_thresh=0.5,
    )

@dataclass
class MainConfig(BaseConfig):
    flux:       FluxConfig       = field(default_factory=lambda: FluxConfig())
    mono:       MonoConfig       = field(default_factory=lambda: MonoConfig())
    projection: ProjectionConfig = field(default_factory=lambda: ProjectionConfig())
    poisson:    PoissonConfig    = field(default_factory=lambda: PoissonConfig())
