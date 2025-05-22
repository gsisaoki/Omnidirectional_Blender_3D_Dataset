import bpy
import os
import json
import numpy as np
import argparse
import sys
import math
import mathutils
import time

class CubicSpline3D:
    def __init__(self, t, points):
        """
        t: (N,) array of parameters (must be increasing)
        points: (N, 3) array of 3D points
        """
        self.t = t
        self.points = points
        self.n = len(t) - 1  # number of segments
        self.coeffs = []  # to store coefficients for x, y, z

        for dim in range(3):
            self.coeffs.append(self._compute_spline_coeffs(points[:, dim]))

    def _compute_spline_coeffs(self, y):
        """
        Computes cubic spline coefficients for a single dimension
        """
        n = self.n
        h = np.diff(self.t)

        # Setup the system
        A = np.zeros((n+1, n+1))
        b = np.zeros(n+1)

        # Natural boundary conditions
        A[0, 0] = 1
        A[-1, -1] = 1

        for i in range(1, n):
            A[i, i-1] = h[i-1]
            A[i, i] = 2 * (h[i-1] + h[i])
            A[i, i+1] = h[i]
            b[i] = 3 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])

        # Solve for c coefficients
        c = np.linalg.solve(A, b)

        # Compute b and d coefficients
        b_coeffs = np.zeros(n)
        d_coeffs = np.zeros(n)
        a_coeffs = y[:-1]  # a_i = y_i

        for i in range(n):
            b_coeffs[i] = (y[i+1] - y[i]) / h[i] - h[i] * (2*c[i] + c[i+1]) / 3
            d_coeffs[i] = (c[i+1] - c[i]) / (3 * h[i])

        # Each segment: (a, b, c, d)
        return np.stack([a_coeffs, b_coeffs, c[:-1], d_coeffs], axis=1)

    def evaluate(self, t_query):
        """
        Evaluate the spline at parameter t_query
        """
        t_query = np.atleast_1d(t_query)
        result = np.zeros((len(t_query), 3))

        for i, tq in enumerate(t_query):
            # Find the right segment
            if tq <= self.t[0]:
                idx = 0
            elif tq >= self.t[-1]:
                idx = self.n - 1
            else:
                idx = np.searchsorted(self.t, tq) - 1

            dt = tq - self.t[idx]
            for dim in range(3):
                a, b, c, d = self.coeffs[dim][idx]
                result[i, dim] = a + b*dt + c*dt**2 + d*dt**3

        return result if len(t_query) > 1 else result[0]


def get_rotmat_from_locations(location1, location2):
    direction = location1 - location2
    z_world = np.array([0, 0, 1])

    obj_vector = direction - np.dot(z_world, direction) * z_world

    neg_y = np.array([0, -1, 0])

    cos_phi = np.dot(obj_vector, neg_y) / (np.linalg.norm(neg_y) * np.linalg.norm(obj_vector))
    cross = np.cross(neg_y, obj_vector)

    sin_phi = np.linalg.norm(cross) / (np.linalg.norm(neg_y) * np.linalg.norm(obj_vector))
    sin_phi_signed = np.sign(cross[2]) * sin_phi

    # atan2
    phi = np.arctan2(sin_phi_signed, cos_phi)
    print("phi", 360*phi/2/np.pi)
    
    theta = np.pi/2

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    
    R_y = np.array([
        [np.cos(phi), 0, np.sin(phi)],
        [0, 1, 0],
        [-np.sin(phi), 0, np.cos(phi)]
    ])

    rotmat = R_x @ R_y
    rotmat = mathutils.Matrix(rotmat)

    return rotmat

def get_egocentric_camera_locations(initial_camera_pos, num_views, num_turns=2, height_increment=0.5, radius=0.5):
    """
    Returns num_views+1 locations to decide the camera orientation for each location
    """

    num_keypoints = num_views + 1

    x0, y0, z0 = initial_camera_pos
    total_angle = 2 * np.pi * num_turns
    angles = np.linspace(0, total_angle, num_keypoints)

    z_values = np.linspace(z0, z0 + num_turns * height_increment, num_keypoints)
    x_values = x0 + radius * np.cos(angles)
    y_values = y0 + radius * np.sin(angles)
    locations = np.vstack((x_values, y_values, z_values)).T
    
    return locations

def convert_blender_to_opencv(rotation, location):
    """
    rotation: 3x3 matrix (camera to world)
    location: 3x1 vector (center of camera in world coordinate)
    
    return:
        rotation: 3x3 matrix (opencv world to camera): R
        translation: 3x1 vector (opencv world to camera): T= -R*C
    """
    blender2opencv = np.diag([1, -1, -1])

    C = np.array(location)
    
    R_c2w_blender = np.array(rotation)
    R_w2c_blender = R_c2w_blender.T
    T_w2c_blender = -1 * R_w2c_blender @ C # T = -R*C
    
    R_w2c_opencv = blender2opencv @ R_w2c_blender
    T_w2c_opencv = blender2opencv @ T_w2c_blender

    return R_w2c_opencv, T_w2c_opencv


def main(args):

    print("sys.executable\n", sys.executable)
    start = time.perf_counter()

    ### Initialization
    # Camera
    main_camera = bpy.data.cameras[args.camera_name]

    # Check attributes
    # print(dir(main_camera))


    main_camera.type = args.camera_type
    if args.camera_type == 'PANO':
        main_camera.panorama_type = args.panorama_type
    main_camera.clip_start = args.clip_start
    main_camera.clip_end = args.clip_end

    # Scene
    scene = bpy.context.scene
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.use_nodes = True # Activate 'use nodes'
    
    # Set the camera as the scene camera
    scene_camera = scene.objects[args.camera_object_name]
    scene.camera = scene_camera

    # Unit
    unit = bpy.types.UnitSettings
    unit.system = 'METRIC'
    unit.length = 'Meters'
    
    # Image settings
    scene.render.image_settings.file_format = args.rgb_format
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'
    scene.render.image_settings.compression = 0

    # rn = scene.node_tree.nodes.new('CompositorNodeRLayers')

    # if args.camera_type == 'PANO':
    if args.capture_type == 'Egocentric':
        print("\n##### Egocentric Trajectory #####\n")
        cam_locations = get_egocentric_camera_locations(args.egocentric_init_pos, 
                                                        args.num_images,
                                                        height_increment=args.height_increment,
                                                        radius=args.egocentric_radius)
    else:
        print("\n##### Non-Egocentric Trajectory #####\n")
        keypoints = np.array(args.keypoints).reshape(-1, 3)

        n= len(keypoints)
        num_points = args.num_images + 1
        t = np.zeros(n)
        for i in range(1, n):
            t[i] = t[i-1] + np.linalg.norm(keypoints[i] - keypoints[i-1])
        
        t_out = np.linspace(t[0], t[-1], num_points)
        spline = CubicSpline3D(t, keypoints)
        cam_locations = spline.evaluate(t_out)
    
    output_folder = os.path.join(os.path.expanduser(args.output_dir), args.capture_type)
    os.makedirs(output_folder, exist_ok=True)

    # Set the output folder for each data
    for data_folder in ["cameras", "images", "depths", "normals"]:
        os.makedirs(os.path.join(output_folder, data_folder), exist_ok=True)

    # Main Loop
    # Render images for num_imags times
    for id, location in enumerate(cam_locations[:-1]):

        ### Get camera rotation matrix
        rotmat_trajectory = get_rotmat_from_locations(cam_locations[id], cam_locations[id+1])

        ### Set RGB Info
        output_path = os.path.join(output_folder, "images", f"{id:05}_rgb.png")
        scene.render.filepath = output_path
        
        ### Set Camera
        scene_camera.matrix_basis = rotmat_trajectory.to_4x4()
        scene_camera.location = location
        scene_camera.data.dof.use_dof = False
        bpy.context.view_layer.update() # Update Transform
        
        ### Initialize node
        node_tree = scene.node_tree
        node_tree.nodes.clear()

        ### Convert data format
        camera_pos, camera_rot = scene_camera.matrix_world.decompose()[0:2] # c2w
        camera_rot = camera_rot.to_matrix()
        # Blender to OpenCV camera convention
        rotation, translation = convert_blender_to_opencv(camera_rot, camera_pos)

        ###
        # Meta data creation
        ###

        ### Create metadata file
        camera_data = [
            {
                'id': id,
                'width': scene.render.resolution_x,
                'height': scene.render.resolution_y,
                'intrinsics': {
                    'focal': scene.render.resolution_y, # fx = fy = resolution_y = resolution_x / 2 = focal length
                    'cx': scene.render.resolution_x / 2,
                    'cy': scene.render.resolution_y / 2,
                },
                'extrinsics': {
                    'rotation': rotation.tolist(),
                    'translation': translation.tolist(),
            },
            }
        ]

        ### Save camera data to JSON file
        json_path = os.path.join(output_folder, "cameras", f"{id:05}_cam.json")
        with open(json_path, 'w') as json_file:
            json.dump(camera_data, json_file, indent=4)
        print(f"Camera data saved to: {json_path}")

        ###
        # Scene Rendering
        ###

        ### Set Compositor
        rn = scene.node_tree.nodes.new('CompositorNodeRLayers')
        viewer = scene.node_tree.nodes.new('CompositorNodeViewer')
        composite = scene.node_tree.nodes.new('CompositorNodeComposite')
        scene.node_tree.links.new(rn.outputs['Image'], viewer.inputs['Image'])
        scene.node_tree.links.new(rn.outputs['Image'], composite.inputs['Image'])

        ### Set Depth Info
        depth = scene.node_tree.nodes.new('CompositorNodeOutputFile')
        depth.format.file_format = 'OPEN_EXR'
        depth.format.color_depth = '32'
        depth.format.color_mode = 'RGB'
        depth.format.exr_codec = 'ZIP'
        depth.base_path = os.path.join(output_folder, "depths")
        depth.file_slots[0].path = f"{id:05}_depth"
        # scene.node_tree.links.new(rn.outputs['Denoising Depth'], depth.inputs[0])
        scene.node_tree.links.new(rn.outputs['Depth'], depth.inputs[0])
        # scene.node_tree.links.new(rn.outputs[1], depth.inputs[0])

        ### Set Normal Info
        normal = scene.node_tree.nodes.new('CompositorNodeOutputFile')
        normal.format.file_format = 'OPEN_EXR'
        normal.format.color_depth = '32'
        normal.format.color_mode = 'RGB'
        normal.base_path = os.path.join(output_folder, "normals")
        normal.file_slots[0].path = f"{id:05}_normal"
        scene.node_tree.links.new(rn.outputs['Normal'], normal.inputs[0])
        # scene.node_tree.links.new(rn.outputs[2], normal.inputs[0])

        ### Run Rendering
        bpy.context.scene.cycles.samples = args.num_samples
        bpy.ops.render.render(write_still=True) # Render image
        print(f"Rendered image saved to: {output_path}")

    end = time.perf_counter()
    print("#####\nAll rendering process has done!\n#####")
    running_time = end - start
    seconds = running_time % 60
    minutes = (running_time % 3600) // 60
    hours = running_time // 3600
    print(f"Time: {int(hours)}h{int(minutes)}m{int(seconds)}s")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    ## Output settings
    parser.add_argument('--output_dir', type=str, default='~/Desktop/Dataset/OmniDepthBlender/debug')
    parser.add_argument('--num_images', type=int, default=1)
    
    ## Data format
    parser.add_argument('--rgb_format', type=str, default='PNG')
    parser.add_argument('--depth_format', type=str, default='OPEN_EXR')
    parser.add_argument('--normal_format', type=str, default='OPEN_EXR')
    
    ## Camera type and settings
    parser.add_argument('--camera_type', type=str, choices=['PERSP', 'ORTH', 'PANO'], default='PANO')
    parser.add_argument('--panorama_type', type=str, choices=['EQUIRECTANGULAR', 'EQUIANGULAR_CUBEMAP_FACE', 'MIRRORBALL', 'FISHEYE_EQUIDISTANT', 'FISHEYE_EQUISOLID', 'FISHEYE_LENS_POLYNOMIAL', 'CENTRAL_CYLINDRICAL'], default='EQUIRECTANGULAR')
    parser.add_argument('--camera_name', type=str, default='Camera')
    parser.add_argument('--camera_object_name', type=str, default='Camera')
    parser.add_argument('--width', type=int, default=1600)
    parser.add_argument('--height', type=int, default=800)
    parser.add_argument('--clip_start', type=float, default=0.0)
    parser.add_argument('--clip_end', type=float, default=100.0)

    ## Camera trajectory
    parser.add_argument('--capture_type', choices=['Egocentric', 'Non-Egocentric'], default='Egocentric')
    parser.add_argument('--egocentric_init_pos', nargs="*", type=float, default=[1.8, 5.7, 1.0], help="1st camera position in the world coordinate")
    parser.add_argument('--height_increment', type=float, default=0.5)
    parser.add_argument('--egocentric_radius', type=float, default=0.5)
    parser.add_argument('--keypoints', nargs='+', type=float, default=[1.4, 2.0, 1.1, 1.4, 7.8, 1.7, 2.8, 6.8, 1.6, 2.65, 2.15, 1.42], help="Keypoints for non-egocentric camera path")

    ## Rendering settings
    parser.add_argument('--num_samples', type=int, default=128)

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    main(args)