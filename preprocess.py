"""
This script loads a 3D model, normalizes its position and scale, sets up
lighting, and renders it from multiple camera angles. It supports lighting via
an environment map or by making materials emissive (baked lighting).
It can render images with filenames that encode the camera's azimuth and elevation.
"""

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

try:
    import bpy
    from mathutils import Matrix, Vector
    from bpy_extras.io_utils import axis_conversion
except ImportError:
    print("This script must be run within Blender's Python environment.")
    sys.exit(1)

_CONTEXT = bpy.context
_SCENE   = _CONTEXT.scene
_RENDER  = _SCENE.render

# A dictionary mapping file extensions to Blender's import functions.
IMPORT_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "obj":  bpy.ops.wm.obj_import,
    "glb":  bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd":  bpy.ops.import_scene.usd,
    "fbx":  bpy.ops.import_scene.fbx,
    "stl":  bpy.ops.import_mesh.stl,
    "ply":  bpy.ops.import_mesh.ply,
}

# Map for translating user-friendly arguments to Blender's enum identifiers.
AXIS_MAP = {
    'X': 'X', 'Y': 'Y', 'Z': 'Z',
    '-X': 'NEGATIVE_X', '-Y': 'NEGATIVE_Y', '-Z': 'NEGATIVE_Z'
}

@dataclass(frozen=True)
class RenderTask:
    """An immutable container for defining a single rendering job."""
    num_renders: int
    random_camera: bool

def reset_scene() -> None:
    """Resets the Blender scene to a clean state by removing all but essential objects."""
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            obj.select_set(True)
    if _CONTEXT.selected_objects:
        bpy.ops.object.delete()

    # Comprehensive cleanup of data-blocks to avoid inter-run contamination.
    for collection in [
        bpy.data.materials,
        bpy.data.textures,
        bpy.data.images,
        bpy.data.meshes,
        bpy.data.actions,
        bpy.data.armatures,
        bpy.data.node_groups,
    ]:
        while collection:
            collection.remove(collection[0])

def clear_all_animation_data() -> None:
    print("Clearing all animation data to ensure static geometry...")
    for obj in bpy.data.objects:
        obj.animation_data_clear()
    if _SCENE.animation_data:
        _SCENE.animation_data_clear()

def load_object(object_path: Path, axis_forward: str, axis_up: str) -> None:
    """Loads a model with robust, format-aware axis correction."""
    if not object_path.is_file():
        raise FileNotFoundError(f"Object file not found: {object_path}")

    file_extension = object_path.suffix.lstrip(".").lower()
    import_func = IMPORT_FUNCTIONS.get(file_extension)
    if not import_func:
        raise ValueError(f"Unsupported file extension: '{file_extension}'")

    kwargs = {}
    final_correction_matrix = Matrix.Identity(4)

    match file_extension:
        case "obj" | "fbx" | "stl":
            kwargs['forward_axis'] = AXIS_MAP[axis_forward]
            kwargs['up_axis'] = AXIS_MAP[axis_up]

        case "glb" | "gltf":
            kwargs["merge_vertices"] = True
            target_correction = axis_conversion(
                from_forward=axis_forward, from_up=axis_up,
                to_forward='Y', to_up='Z'
            ).to_4x4()
            # Undo Blender's implicit +90deg X-rotation for glTF imports.
            gltf_undo_matrix = Matrix.Rotation(math.radians(-90.0), 4, 'X')
            # Apply undo matrix first, then the target correction.
            final_correction_matrix = target_correction @ gltf_undo_matrix

        case _:  # PLY, USD, etc.
            final_correction_matrix = axis_conversion(
                from_forward=axis_forward, from_up=axis_up,
                to_forward='Y', to_up='Z'
            ).to_4x4()

    objects_before = set(bpy.context.scene.objects)
    import_func(filepath=str(object_path), **kwargs)

    if not final_correction_matrix.is_identity:
        new_objects = set(bpy.context.scene.objects) - objects_before
        new_root_objects = [o for o in new_objects if o.parent is None or o.parent not in new_objects]

        for obj in new_root_objects:
            obj.matrix_world = final_correction_matrix @ obj.matrix_world

def get_scene_aabb() -> Tuple[Vector, Vector]:
    """
    Computes the axis-aligned bounding box (AABB) of all mesh objects in the scene.

    Returns:
        A tuple containing the minimum and maximum corner vectors of the AABB.

    Raises:
        RuntimeError: If no mesh objects are found in the scene.
    """
    bbox_min = Vector((math.inf, math.inf, math.inf))
    bbox_max = Vector((-math.inf, -math.inf, -math.inf))
    found_mesh = False

    for obj in _SCENE.objects:
        if obj.type == "MESH":
            found_mesh = True
            for vertex in obj.data.vertices:
                global_vertex = obj.matrix_world @ vertex.co
                for i in range(3):
                    bbox_min[i] = min(bbox_min[i], global_vertex[i])
                    bbox_max[i] = max(bbox_max[i], global_vertex[i])

    if not found_mesh:
        raise RuntimeError("No mesh objects in scene to compute AABB for.")

    return bbox_min, bbox_max


def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    """Yields all root objects (objects with no parent) in the scene."""
    for obj in _SCENE.objects:
        if not obj.parent:
            yield obj


def normalize_scene() -> None:
    """Normalizes the scene to fit in a unit cube and applies type-specific rotations."""
    root_objects = [obj for obj in get_scene_root_objects() if obj.type != "CAMERA"]
    if not root_objects:
        print("Warning: No objects to normalize.")
        return

    transform_target = root_objects[0]
    if len(root_objects) > 1:
        parent_empty = bpy.data.objects.new("NormalizationParent", None)
        _SCENE.collection.objects.link(parent_empty)
        for obj in root_objects:
            obj.parent = parent_empty
        transform_target = parent_empty

    _CONTEXT.view_layer.update()

    try:
        bbox_min, bbox_max = get_scene_aabb()
    except RuntimeError:
        print("Warning: No meshes in scene. Skipping normalization.")
        return

    scale_factor = 1.0 / max((bbox_max - bbox_min).length, 1e-6)
    offset = -(bbox_min + bbox_max) / 2.0

    transform_target.scale = Vector((scale_factor, scale_factor, scale_factor))
    transform_target.location = offset * scale_factor
    _CONTEXT.view_layer.update()

def set_lighting(
    env_map_path: Optional[Path], is_baked: bool, use_emission_shader: bool
) -> None:
    """Configures the scene's lighting based on provided arguments."""
    if is_baked:
        set_emission_shader_from_vertex_color()
    elif use_emission_shader:
        set_emission_shader_from_texture()
    elif env_map_path:
        load_environment_map(env_map_path)
    else:
        print("Warning: No lighting specified. Scene may be dark.")


def load_environment_map(env_map_path: Path) -> None:
    """Loads an HDR environment map and sets it as the world background."""
    if not env_map_path.exists():
        print(f"Warning: Environment map not found at {env_map_path}. Skipping.")
        return

    world = _SCENE.world or bpy.data.worlds.new("World")
    _SCENE.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()

    bg_node = nodes.new(type="ShaderNodeBackground")
    env_tex_node = nodes.new(type="ShaderNodeTexEnvironment")
    output_node = nodes.new(type="ShaderNodeOutputWorld")

    env_tex_node.image = bpy.data.images.load(str(env_map_path))
    links = world.node_tree.links
    links.new(env_tex_node.outputs["Color"], bg_node.inputs["Color"])
    links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])


def _replace_material_with_emission(obj: bpy.types.Object, setup_nodes: Callable):
    """Helper to replace materials on an object with a new emission setup."""
    mat = bpy.data.materials.new(name=f"{obj.name}_Emission")
    mat.use_nodes = True
    setup_nodes(mat.node_tree)

    if obj.material_slots:
        for slot in obj.material_slots:
            slot.material = mat
    else:
        obj.data.materials.append(mat)


def set_emission_shader_from_texture() -> None:
    """Replaces all materials with an emission shader using the base texture."""
    for obj in _SCENE.objects:
        if obj.type != "MESH":
            continue

        source_image = None
        for mat_slot in obj.material_slots:
            if mat := mat_slot.material:
                if mat.use_nodes and (
                    img_node := next(
                        (n for n in mat.node_tree.nodes if n.type == "TEX_IMAGE"), None
                    )
                ) and img_node.image:
                    source_image = img_node.image
                    break
        if not source_image:
            continue

        def setup_texture_emission(node_tree):
            nodes, links = node_tree.nodes, node_tree.links
            nodes.clear()
            tex_node = nodes.new(type="ShaderNodeTexImage")
            tex_node.image = source_image
            emission = nodes.new(type="ShaderNodeEmission")
            output = nodes.new(type="ShaderNodeOutputMaterial")
            links.new(tex_node.outputs["Color"], emission.inputs["Color"])
            links.new(emission.outputs["Emission"], output.inputs["Surface"])

        _replace_material_with_emission(obj, setup_texture_emission)


def set_emission_shader_from_vertex_color() -> None:
    """Replaces all materials with an emission shader that uses vertex colors."""
    for obj in _SCENE.objects:
        if (
            obj.type != "MESH"
            or not obj.data.color_attributes
            or not (color_layer := obj.data.color_attributes.active)
        ):
            continue

        def setup_vcolor_emission(node_tree):
            nodes, links = node_tree.nodes, node_tree.links
            nodes.clear()
            attr_node = nodes.new(type="ShaderNodeAttribute")
            attr_node.attribute_name = color_layer.name
            emission = nodes.new(type="ShaderNodeEmission")
            output = nodes.new(type="ShaderNodeOutputMaterial")
            links.new(attr_node.outputs["Color"], emission.inputs["Color"])
            links.new(emission.outputs["Emission"], output.inputs["Surface"])

        _replace_material_with_emission(obj, setup_vcolor_emission)


def get_camera_positions(
    task: RenderTask, radius: float
) -> Generator[Tuple[Vector, float, float], None, None]:
    """
    Generates camera positions using direct spherical coordinate calculation
    that aligns with Blender's world coordinate system (Right-Handed, Z-up):

        +Z (Up)
        ^    
        |    / +Y (Away/Forward)
        |   /
        |  /
        | /
        +------------> +X (Right)

    This function calculates points on a sphere around the origin (0,0,0)
    in this coordinate system. The negative sign in the `y` calculation
    is what places the camera in front of the object (looking from the
    positive Y direction towards the origin) for a 0-degree azimuth.
    """
    if task.random_camera:
        for _ in range(task.num_renders):
            azimuth_rad = random.uniform(0, 2 * math.pi)
            elevation_rad = math.asin(random.uniform(-1, 1))

            x = radius * math.cos(elevation_rad) * math.sin(azimuth_rad)
            y = -radius * math.cos(elevation_rad) * math.cos(azimuth_rad)
            z = radius * math.sin(elevation_rad)

            yield Vector((x, y, z)), math.degrees(azimuth_rad), math.degrees(elevation_rad)

    else: # Orbit
        elevation_rad = 0.0
        for i in range(task.num_renders):
            azimuth_deg = (360.0 / task.num_renders) * i
            azimuth_rad = math.radians(azimuth_deg)

            x = radius * math.cos(elevation_rad) * math.sin(azimuth_rad)
            y = -radius * math.cos(elevation_rad) * math.cos(azimuth_rad)
            z = radius * math.sin(elevation_rad)

            yield Vector((x, y, z)), azimuth_deg, 0.0


def setup_camera_and_track(location: Vector) -> bpy.types.Object:
    """Positions and configures the scene camera, and makes it track the origin."""
    bpy.ops.object.camera_add(location=location)
    cam = _CONTEXT.active_object
    cam.name = "SceneCamera"
    _SCENE.camera = cam

    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 1.0

    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
    target = _CONTEXT.active_object
    target.name = "TrackTarget"

    constraint = cam.constraints.new(type="TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"
    return cam


def setup_render_settings(engine: str, resolution: int, samples: int) -> None:
    """Configures global render settings for Blender."""
    _RENDER.engine = engine
    _RENDER.image_settings.file_format = "PNG"
    _RENDER.image_settings.color_mode = "RGBA"
    _RENDER.resolution_x = resolution
    _RENDER.resolution_y = resolution
    _RENDER.film_transparent = True

    if engine == "CYCLES":
        _SCENE.cycles.device = "GPU"
        _SCENE.cycles.samples = samples
        _SCENE.cycles.use_denoising = True
        _SCENE.cycles.use_persistent_data = True
        try:
            prefs = _CONTEXT.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "CUDA"  # Or "OPTIX", "HIP", "METAL"
            prefs.get_devices()
            for device in prefs.devices:
                device.use = "CPU" not in device.name.upper()
        except Exception as e:
            print(f"Could not configure GPU devices, using default. Error: {e}")


def execute_render_pass(
    tasks: List[RenderTask], output_dir: Path, radius: float, resolution: int
) -> List[Dict[str, Any]]:
    """
    Executes the orthographic rendering pass for a list of tasks.

    Args:
        tasks: A list of RenderTask objects defining the renders.
        output_dir: The base directory for all output renders.
        radius: The camera distance from the origin.
    """
    if not tasks:
        return []

    print("\n--- Starting orthographic render pass ---")

    cam = setup_camera_and_track(Vector((0, 0, 0)))
    frame_mappings: Dict[int, Path] = {}
    view_records: List[Dict[str, Any]] = []
    current_frame = 1

    # This loop will typically only run once, but handles the list structure.
    for task in tasks:
        if task.num_renders == 0:
            continue

        if task.random_camera:
            task_output_dir = output_dir
            prefix = "train"
            print(f"  - Planning {task.num_renders} random keyframes.")
        else:  # Orbit
            task_output_dir = output_dir / "orbit"
            prefix = "orbit"
            print(f"  - Planning {task.num_renders} orbit keyframes.")

        task_output_dir.mkdir(parents=True, exist_ok=True)

        pos_generator = get_camera_positions(task, radius)
        for i, (location, azimuth, elevation) in enumerate(pos_generator):
            cam.location = location
            cam.keyframe_insert(data_path="location", frame=current_frame)
            if task.random_camera:
                filename = f"{prefix}_{azimuth:.1f}_{elevation:.1f}.png"
            else:
                filename = f"{prefix}_{i:03d}.png"
            frame_mappings[current_frame] = task_output_dir / filename
            view_records.append(
                {
                    "image_path": str(task_output_dir / filename),
                    "azimuth_deg": azimuth,
                    "elevation_deg": elevation,
                    "camera_location": [location.x, location.y, location.z],
                    "radius": radius,
                    "camera_type": "ORTHO",
                    "ortho_scale": 1.0,
                    "resolution": resolution,
                    "track_axis": "NEGATIVE_Z",
                    "up_axis": "Y",
                }
            )
            current_frame += 1

    if not frame_mappings:
        return []

    _SCENE.frame_start = 1
    _SCENE.frame_end = current_frame - 1

    total_frames = len(frame_mappings)
    print(f"Rendering {total_frames} frames via manual loop...")

    for frame in range(_SCENE.frame_start, _SCENE.frame_end + 1):
        _SCENE.frame_set(frame)
        _RENDER.filepath = str(frame_mappings[frame])
        print(f"Rendering frame {frame}/{total_frames} to {frame_mappings[frame].name}")
        bpy.ops.render.render(write_still=True)

    print("Manual rendering complete.")

    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects[cam.name].select_set(True)
    bpy.data.objects["TrackTarget"].select_set(True)
    bpy.ops.object.delete()
    return view_records


def write_views_manifest(
    output_dir: Path,
    view_records: List[Dict[str, Any]],
    object_name: str,
    normalized_obj_path: Optional[Path],
) -> None:
    """Write a sidecar JSON manifest describing each rendered view."""
    manifest_path = output_dir / "views_manifest.json"
    manifest = {
        "object_name": object_name,
        "normalized_obj_path": str(normalized_obj_path) if normalized_obj_path else "",
        "axis_forward": "Z",
        "axis_up": "Y",
        "views": view_records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved views manifest to {manifest_path}")


def main():
    """Main execution function to parse arguments and orchestrate the rendering process."""
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Render a 3D object from multiple angles.")
    parser.add_argument("--object_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    axis_choices = ["X", "Y", "Z", "-X", "-Y", "-Z"]
    parser.add_argument(
        "--axis_forward", type=str, default='-Z', choices=axis_choices,
        help="Forward axis of the source file. Default is '-Z'."
    )
    parser.add_argument(
        "--axis_up", type=str, default='Y', choices=axis_choices,
        help="Up axis of the source file. Default is 'Y'."
    )
    parser.add_argument("--env_map_path", type=Path, default=None)
    parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--num_renders", type=int, default=100)
    parser.add_argument("--use_emission_shader", action="store_true")
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--random_camera", action="store_true")
    parser.add_argument("--baked", action="store_true")
    args = parser.parse_args(argv)

    reset_scene()
    load_object(args.object_path, args.axis_forward, args.axis_up)
    set_lighting(args.env_map_path, args.baked, args.use_emission_shader)
    normalize_scene()

    bpy.ops.object.select_all(action="DESELECT")
    for obj in _SCENE.objects:
        if obj.type == "MESH":
            obj.select_set(True)
    if _CONTEXT.selected_objects:
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    clear_all_animation_data()

    setup_render_settings(args.engine, args.resolution, args.samples)

    tasks: List[RenderTask] = []
    if args.num_renders > 0:
        tasks.append(RenderTask(args.num_renders, args.random_camera))

    view_records = execute_render_pass(tasks, args.output_dir, args.radius, args.resolution)

    bpy.ops.object.select_all(action="DESELECT")
    for obj in _SCENE.objects:
        if obj.type == "MESH":
            obj.select_set(True)
    normalized_path = None
    if _CONTEXT.selected_objects and args.num_renders > 0:
        obj_name = args.object_path.stem
        normalized_path = args.output_dir / f"{obj_name}_normalized.obj"
        # WARN: Changing to coord. system of the refinement framework
        bpy.ops.export_scene.obj(
            filepath=str(normalized_path),
            use_selection=True,
            axis_forward='Z',
            axis_up='Y'
        )
        print(f"\nSaved final normalized model to {normalized_path}")
        if view_records:
            write_views_manifest(args.output_dir, view_records, obj_name, normalized_path)
    print("Rendering complete.")

if __name__ == "__main__":
    main()
