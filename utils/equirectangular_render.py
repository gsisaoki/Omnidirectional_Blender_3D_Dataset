import numpy as np
import open3d as o3d
import os
import json

# def equirectangular_renderer_from_mesh(ply_path, width=1600, height=800, origin=np.array([0.0, 0.0, 0.0]), rotation=None, scale=1.0, camera_center=None):
#     """
#     render depth map and normal map at arbitrary camera from 3D mesh

#     Args:
#     - ply_path: path to the mesh file (PLY format)
#     - width: number of pixels in the horizontal direction (-180〜+180degree)
#     - height: number of pixels in the vertical direction (-90〜+90degree)
#     - origin: center position of the camera that casts rays
#     - rotation: rotation matrix of the camera (3x3)
#     - scale: scale factor
#     Returns:
#     - depth: depth map rendered from the mesh
#     - normal: normal map rendered from the mesh
#     """

#     if not os.path.exists(ply_path):
#         raise FileNotFoundError(f"File not found: {ply_path}")
#     mesh = o3d.io.read_triangle_mesh(ply_path)
#     mesh.compute_vertex_normals()

#     scene = o3d.t.geometry.RaycastingScene()
#     _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

#     phi = np.linspace(-np.pi, np.pi, width)        # holizontal: -180°〜180°
#     theta = np.linspace(-np.pi/2, np.pi/2, height) # vertical: -90°〜90°
#     phi, theta = np.meshgrid(phi, theta)

#     x = np.cos(theta) * np.sin(phi)
#     y = np.sin(theta)
#     z = np.cos(theta) * np.cos(phi)
#     dirs = np.stack([x, y, z], axis=-1).reshape(-1, 3)

#     if rotation is not None:
#         rotation = np.array(rotation).reshape(3, 3)
#         dirs = dirs @ rotation.T

#     origins = np.broadcast_to(origin, dirs.shape)

#     rays = np.concatenate([origins, dirs], axis=-1).astype(np.float32)
#     rays = o3d.core.Tensor(rays)

#     ans = scene.cast_rays(rays)

#     depth = ans['t_hit'].numpy().reshape(height, width)
#     depth[depth == np.inf] = 0
#     depth = depth * scale

#     normal = ans['primitive_normals'].numpy().reshape(height, width, 3)
#     norm = np.linalg.norm(normal, axis=-1, keepdims=True)
#     norm[norm == 0] = 1e-8
#     normal = normal / norm
#     normal = np.clip(normal, -1.0, 1.0)
#     normal = (normal + 1) / 2
#     normal = np.clip(normal, 0, 1)
#     normal = np.flip(normal, axis=2)

#     return depth, normal

def equirectangular_renderer_from_mesh(ply_path, camera_path, width=1600, height=800, scale=1.0):
    """
    render depth map and normal map at arbitrary camera from 3D mesh

    Args:
    - ply_path: path to the mesh file (PLY format)
    - camera_path: path to the camera file (JSON format)
    - width: number of pixels in the horizontal direction (-180〜+180degree)
    - height: number of pixels in the vertical direction (-90〜+90degree)
    - scale: scale factor
    Returns:
    - depth: depth map rendered from the mesh
    - normal: normal map rendered from the mesh
    """

    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"File not found: {ply_path}")
    mesh = o3d.io.read_triangle_mesh(ply_path)
    mesh.compute_vertex_normals()

    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    phi = np.linspace(-np.pi, np.pi, width)        # holizontal: -180°〜180°
    theta = np.linspace(-np.pi/2, np.pi/2, height) # vertical: -90°〜90°
    phi, theta = np.meshgrid(phi, theta)

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta)
    z = np.cos(theta) * np.cos(phi)
    dirs = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    with open(camera_path, 'r') as f:
        cameras = json.load(f)
    camera_info = cameras[0]
    extrinsics = camera_info["extrinsics"]
    rotation = np.array(extrinsics["rotation"])
    translation = np.array(extrinsics["translation"])
    rotation = np.linalg.inv(rotation)
    translation = -rotation @ np.array(translation)

    rotation = np.array(rotation).reshape(3, 3)
    dirs = dirs @ rotation.T

    origins = np.broadcast_to(translation, dirs.shape)

    rays = np.concatenate([origins, dirs], axis=-1).astype(np.float32)
    rays = o3d.core.Tensor(rays)

    ans = scene.cast_rays(rays)

    depth = ans['t_hit'].numpy().reshape(height, width)
    depth[depth == np.inf] = 0
    depth = depth * scale

    normal = ans['primitive_normals'].numpy().reshape(height, width, 3)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    norm[norm == 0] = 1e-8
    normal = normal / norm
    normal = np.clip(normal, -1.0, 1.0)
    normal = (normal + 1) / 2
    normal = np.clip(normal, 0, 1)
    normal = np.flip(normal, axis=2)

    return depth, normal