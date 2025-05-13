import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os, sys
from glob import glob
# from icecream import ic
# from scipy.spatial.transform import Rotation as Rot
# from scipy.spatial.transform import Slerp
import csv, math
from models.dataloader.dataset_utils import *

import OpenEXR
import Imath
import json
from struct import unpack

class Egocentric360:
    def __init__(self, conf):
        super(Egocentric360, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf
        self.data_dir = conf.get_string('data_dir')
        self.camera_outside_sphere = True

        self.fr_start = conf.get_int('fr_start')
        self.fr_end = conf.get_int('fr_end')
        self.fr_interval = conf.get_int('fr_interval')
        self.fr_scale = conf.get_float('fr_scale')
        self.world_scale = conf.get_float('world_scale')
        self.far_sphere_bound = conf.get_float('far_sphere_bound')
        
        self.dataset_name = conf.get_string('dataset_name')

        try:
            self.world_shift = conf.get_float('world_shift')
        except:
            self.world_shift = 0
        
        ### images
        frames = []
        
        
        if self.dataset_name == "ODB":
            image_paths = sorted(glob(os.path.join(self.data_dir, "images", '*.png')))
            for image_path in image_paths:
                frame = cv.imread(image_path)
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frames.append(
                    cv.resize(
                        frame, dsize=(0, 0),
                        fx=self.fr_scale, fy=self.fr_scale, interpolation=cv.INTER_AREA
                        )
                    )
                
        elif self.dataset_name == "mp3d":
            image_paths = sorted(glob(os.path.join(self.data_dir, "images", '*.png')))
            for image_path in image_paths:
                frame = cv.imread(image_path)
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frames.append(
                    cv.resize(
                        frame, dsize=(0, 0),
                        fx=self.fr_scale, fy=self.fr_scale, interpolation=cv.INTER_AREA
                        )
                    )
    
        ## Blender360 データセットの画像の読み込み部分
        elif self.dataset_name == "Blender360":
            # import pdb; pdb.set_trace()
            image_paths = sorted(glob(os.path.join(self.data_dir, "images", "*_rgb.png")))
            for image_path in image_paths:
                frame = cv.imread(image_path)
                frames.append(
                    cv.resize(
                        frame, dsize=(0, 0),
                        fx=self.fr_scale, fy=self.fr_scale, interpolation=cv.INTER_AREA
                        )
                    )
                
        else:
            cap = cv.VideoCapture(os.path.join(self.data_dir, "video.mp4"))
            for frame_id in range(self.fr_start, self.fr_end, self.fr_interval):
                if not cap.isOpened():
                    print("Error opening video stream or file")
                cap.set(cv.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                if frame is not None:
                    frames.append(
                        cv.resize(
                            frame, dsize=(0, 0),
                            fx=self.fr_scale, fy=self.fr_scale, interpolation=cv.INTER_AREA
                            )
                        )
            cap.release()
        
        self.n_images = len(frames) # N
        self.images_np = np.stack(frames) / 256.0
        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cuda()  # [n_images, H, W, 3]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W
        
        ### masks
        if os.path.exists(os.path.join(self.data_dir, "mask_img.png")):
            mask_np = cv.imread(os.path.join(self.data_dir, "mask_img.png"))[..., 0] / 256.0
        else:
            mask_np = np.ones((self.H, self.W), dtype=np.float32) * 255.0
        self.mask = torch.from_numpy(cv.resize(mask_np, dsize=(self.W, self.H)) > 0.5).cuda()   # Valid pixel
        
        ### depths
        os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

        # デプスの読み込み部分
        if self.dataset_name == "ODB":
            self.odb_scale = conf.get_float('dataset_scale')
            self.depths_lis = [os.path.join(self.data_dir, "depth", f"{frame_id:03d}_depth.exr") for frame_id in range(self.fr_start, self.fr_end, 1)]
            
            cm2m_scale = 100
            if self.data_dir == "/home/jaxa/shintaro/OmniGauSS/data/OmniDepthBlender/archiviz-flat":
                cm2m_scale = 1
            
            self.depths_np = np.stack([self.read_exr_depth(im_name, cm2m_scale) for im_name in self.depths_lis]) # デプスの読み込み
            self.depths = torch.from_numpy(self.depths_np.astype(np.float32)).cuda()
            
            # TODO: デプスマップの無効な部分をマスクする
            # TODO: OpenSfM の世界座標系と真値のカメラパラメータの世界座標系との間のスケールを事前に計算し，
            #       それを基に far の値を決めることで octree の作成に用いるデプスマップのマスクを決定する
            # barbershop: scale = 0.12889872026806104
            invalid_depth_msk = torch.logical_or((self.depths > 1e2), ~self.mask) # とりあえず適当に 100
            depths = self.depths / self.odb_scale # ODB データセットのスケールで割る
            depths[invalid_depth_msk] = 0.0
            print("depth min, max", torch.min(depths), torch.max(depths))
        
        elif self.dataset_name == "mp3d":
            self.mp3d_scale = conf.get_float('dataset_scale')
            self.depths_lis = glob(os.path.join(self.data_dir, "depth", "*.dpt"))
            
            self.depths_np = np.stack([self.read_dpt(im_name) for im_name in self.depths_lis]) # デプスの読み込み
            self.depths = torch.from_numpy(self.depths_np.astype(np.float32)).cuda()
            invalid_depth_msk = torch.logical_or((self.depths == 0), ~self.mask)
            depths = self.depths / self.mp3d_scale # mp3d データセットのスケールで割る
            depths[invalid_depth_msk] = 0.0
            print("depth min, max", torch.min(depths), torch.max(depths))
        
        elif self.dataset_name == "Blender360":
            # TODO: 名前の指定方法を修正する
            # import pdb; pdb.set_trace()
            # self.depths_lis = [os.path.join(self.data_dir, "depths", f"{frame_id:05d}_depth{frame_id:04d}.exr") for frame_id in range(self.fr_start, self.fr_end, 1)]
            self.depths_lis = sorted(glob(os.path.join(self.data_dir, "depths", "*.exr*")))
            # self.depths_np = np.stack([self.read_exr_depth(im_name, scale=1) for im_name in self.depths_lis])
            self.depths_np = np.stack([self.read_exr_depth_v2(im_name, scale=1) for im_name in self.depths_lis])
            self.depths = torch.from_numpy(self.depths_np.astype(np.float32)).cuda()
            invalid_depth_msk = torch.logical_or((self.depths > 1e2), ~self.mask) # 100 以上のデプスは外れ値として，0 のデプスを与える
            # NOTE: Blender360 の場合，真値のカメラパラメータを使うため，スケール変換は必要ないはず... 
            depths = self.depths
            depths[invalid_depth_msk] = 0.0
            print("depth min, max", torch.min(depths), torch.max(depths))
        
        else:
            self.depths_lis = [os.path.join(self.data_dir, "idepth", f"{frame_id}.exr") for frame_id in range(self.fr_start, self.fr_end, self.fr_interval)]
            print("Image resolution", self.H, self.W)

            self.depths_np = np.stack([cv.resize(cv.imread(im_name, cv.IMREAD_UNCHANGED), dsize=(self.W, self.H)) for im_name in self.depths_lis])
            self.idepths = torch.from_numpy(self.depths_np.astype(np.float32)).cuda()
            
            invalid_depth_msk = torch.logical_or((self.idepths < 1e-4), ~self.mask)
            depths = 1.0 / self.idepths
            self.idepths[invalid_depth_msk] = 0.0
            depths[invalid_depth_msk] = 0.0
            print("depth min, max", torch.max(depths), torch.min(depths))
        
        # =====
        # 可視化用
        # import matplotlib.pyplot as plt
        # import matplotlib.cm as cm
        # depths[invalid_depth_msk] = 0.01
        # depths[depths > 100] = 0.01
        # depth_image = depths[0].cpu().numpy()
        # # depth_image_colored = cm.turbo(depth_image / np.max(depth_image))
        # # depth_image_colored = cm.turbo(depth_image)
        # plt.imshow(depth_image, cmap="turbo")
        # plt.title("Depth Image with Turbo Colormap")
        # plt.colorbar()
        # plt.savefig("depth_image.png")
        # import pdb; pdb.set_trace()
        # =====
        
        # depths = torch.nan_to_num(depths, 0)


        self.depths = depths
        self.depths_mask = ~invalid_depth_msk

        ### Cameras
        # カメラパラメータの読み込み部分
        if self.dataset_name == 'ODB' or self.dataset_name == 'mp3d':
            reconstruction_file = os.path.join(self.data_dir, 'reconstruction.json')
            with open(reconstruction_file) as f:
                reconstruction = json.load(f)
                
                camera_poses = []
                for entry in reconstruction:
                    shots = entry.get("shots", {})
                    for image_name, attributes in shots.items():
                        translation = attributes.get("translation", []) # [1, 3]
                        angles = attributes.get("rotation", [])
                        qvec = self.angle_axis_to_quaternion(angles)
                        rotation = self.qvec2rotmat(qvec) # [3, 3]
                        mat = np.concatenate([rotation, np.array(translation).reshape(3, 1)], axis=1) # [3, 4]
                        mat = np.concatenate([mat, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0) # [4, 4]
                        mat = np.linalg.inv(mat)
                        # 画像名とtranslationをタプルでリストに追加
                        camera_poses.append((image_name, mat.astype(np.float32)))

                # 画像名とtranslationのリスト
                camera_poses = sorted(camera_poses, key=lambda x: x[0])
                traj = [mat for _, mat in camera_poses]
        
        elif self.dataset_name == 'Blender360':
            cameras_dir = os.path.join(self.data_dir, "cameras")
            
            traj = []
            for camera_file in sorted(os.listdir(cameras_dir)):
                if camera_file.endswith("_cam.json"):
                    with open(os.path.join(cameras_dir, camera_file)) as f:
                        camera_data = json.load(f)[0]
                        translation = camera_data["extrinsics"]["translation"] # w2c
                        rotation = camera_data["extrinsics"]["rotation"] # w2c
                        mat = np.eye(4)
                        mat[:3, :3] = rotation
                        mat[:3, 3] = translation
                        mat = np.linalg.inv(mat) # c2w
                
                traj.append(mat.astype(np.float32))
        
        else:
            traj = []
            with open(os.path.join(self.data_dir, "traj.csv")) as f:
                csv_reader = csv.reader(f, delimiter=" ")
                for row in csv_reader:
                    mat = np.array(row[1:], dtype=np.float32).reshape((4, 4))   # c2w [4, 4]
                    mat = np.linalg.inv(mat)                                    # w2c
                    traj.append(mat)

        traj_full = np.stack(traj, axis=0)[:, :3, -1] # [N, 3]
        self.cam_center = np.mean(traj_full, axis=0)
        traj_full = traj_full - self.cam_center
        self.near = np.max(np.sqrt(traj_full[..., 0]**2 + traj_full[..., 1]**2 + traj_full[..., 2]**2))
        self.far = self.near / math.tan(math.pi / self.W) if self.far_sphere_bound < 0 else self.far_sphere_bound
        
        # Select camera
        self.traj = np.stack([traj[num] for num in range(self.fr_start, self.fr_end, self.fr_interval)], axis=0)   # [N, 4, 4]
        self.cam_pos = self.traj[:, :3, -1] - self.cam_center[None, ...]    # Centeralize [N, 3]
        self.cam_pos[:, :2] += self.world_shift
        self.cam_rot = self.traj[:, :3, :3]
        self.pose_all = torch.tensor(self.cam_rot).cuda()

        # Set bounding sphere
        
        print("Bounding sphere", self.near, self.far)
        self.sphere_scale = self.far
        self.cam_pos /=  self.far
        self.depths /= self.far
        self.near /= self.far
        self.far /= self.far
        print("Bounding sphere after normalization", self.near, self.far)
        
        try:
            self.object_bbox_min = np.array(conf['obj_bbox_min'])
            self.object_bbox_max = np.array(conf['obj_bbox_max'])
        except:
            self.object_bbox_min = np.array([-1.01, -1.01, -1.01])
            self.object_bbox_max = np.array([1.01, 1.01, 1.01])
        
        print('Load data: End')


    def gen_rays_at(self, img_idx, resolution_level=1, debug=False):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        pixels_x = pixels_x.transpose(0, 1)
        pixels_y = pixels_y.transpose(0, 1)
        
        # w = self.W//l
        # h = self.H//l
        h = pixels_x.size(0)
        w = pixels_x.size(1)

        rays_o = torch.tensor(np.tile(self.cam_pos[img_idx:img_idx+1, ...], (w*h, 1))).cuda()
        
        rays_rot = torch.tensor(np.tile(self.cam_rot[img_idx:img_idx+1, ...], (w*h, 1, 1))).contiguous().cuda()  # [N, 3, 3]
        rays_v = pixel_to_rays_dir(pixels_x, pixels_y, self.H, self.W).reshape(-1, 3).contiguous().cuda()  # [N=H*W, 3]
        # rays_v = pixel_to_rays_dir(pixels_x, pixels_y, h, w).reshape(-1, 3).contiguous().cuda()  # [N=H*W, 3]
        rays_v = cam2world_rays(rays_v, rays_rot)
        
        rays_o = zaxis_front2top(rays_o)
        rays_v = zaxis_front2top(rays_v)
        rays_o = rays_o.reshape(h, w, 3)
        rays_v = rays_v.reshape(h, w, 3)

        rays_v = torch.nn.functional.normalize(rays_v, dim=-1)
        
        return rays_o, rays_v
        # depth = self.depths[img_idx]    
        # return rays_o, rays_v, depth
    
    def gen_discrete_rays_at(self, img_idx, resolution_level=1):
        # l = resolution_level
        tx = torch.arange(0, self.W - 1, resolution_level) #l)
        ty = torch.arange(0, self.H - 1, resolution_level) #l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty) # [W, H]
        pixels_x = pixels_x.transpose(0, 1)         # [H, W]
        pixels_y = pixels_y.transpose(0, 1)

        #mask = self.depths_mask[img_idx][(pixels_y, pixels_x)]      # [H, W]
        mask = torch.logical_and(
            self.depths_mask[img_idx][(pixels_y, pixels_x)],
            self.mask[(pixels_y, pixels_x)]
        )
        depth = self.depths[img_idx][(pixels_y, pixels_x)]
        
        h = pixels_x.size(0)
        w = pixels_x.size(1)

        rays_o = torch.tensor(np.tile(self.cam_pos[img_idx:img_idx+1, ...], (w*h, 1))).cuda()
        
        rays_rot = torch.tensor(np.tile(self.cam_rot[img_idx:img_idx+1, ...], (w*h, 1, 1))).contiguous().cuda()  # [N, 3, 3]
        # rays_v = pixel_to_rays_dir(pixels_x, pixels_y, h, w).reshape(-1, 3).contiguous().cuda()  # [N=H*W, 3]
        # rays_v = pixel_to_rays_dir(pixels_x, pixels_y, self.H, self.W).reshape(-1, 3).contiguous().cuda()  # [N=H*W, 3]
        rays_v = pixel_to_rays_dir(pixels_x, pixels_y, self.H, self.W).reshape(-1, 3).contiguous().cuda()  # [N=H*W, 3]
        rays_v = cam2world_rays(rays_v, rays_rot)
        
        rays_o = zaxis_front2top(rays_o)
        rays_v = zaxis_front2top(rays_v)
        rays_o = rays_o.reshape(h, w, 3)
        rays_v = rays_v.reshape(h, w, 3)

        rays_v = torch.nn.functional.normalize(rays_v, dim=-1)

        color = self.images[img_idx][(pixels_y, pixels_x)]
        
        return rays_o, rays_v, color, depth, mask


    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """

        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.mask[(pixels_y, pixels_x)]      # batch_size, 3
        depth = self.depths[img_idx][(pixels_y, pixels_x)]

        rays_o = torch.tensor(np.tile(self.cam_pos[img_idx:img_idx+1, ...], (batch_size, 1))).cuda()
        
        rays_rot = torch.tensor(np.tile(self.cam_rot[img_idx:img_idx+1, ...], (batch_size, 1, 1))).contiguous().cuda()  # [N, 3, 3]
        rays_v = pixel_to_rays_dir(pixels_x, pixels_y, self.H, self.W).reshape(-1, 3).contiguous().cuda()  # [N=H*W, 3]
        rays_v = cam2world_rays(rays_v, rays_rot)

        rays_o = zaxis_front2top(rays_o)
        rays_v = zaxis_front2top(rays_v)
        rays_v = torch.nn.functional.normalize(rays_v, dim=-1)

        return torch.cat([rays_o, rays_v, color.cuda(), mask[..., None], depth[..., None]], dim=-1)    # batch_size, 10 + 1


    def image_at(self, idx, resolution_level):
        return (cv.resize(self.images_np[idx]*256, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)
        # return (self.images_np[idx] * 256.0).clip(0, 255)
        
    #
    # Matterport3D データセット実験用
    #
    def read_dpt(self, dpt_file_path):
        """read depth map from *.dpt file.

        :param dpt_file_path: the dpt file path
        :type dpt_file_path: str
        :return: depth map data
        :rtype: numpy
        """
        TAG_FLOAT = 202021.25  # check for this when READING the file

        ext = os.path.splitext(dpt_file_path)[1]

        assert len(ext) > 0, ('readFlowFile: extension required in fname %s' % dpt_file_path)
        assert ext == '.dpt', exit('readFlowFile: fname %s should have extension ''.flo''' % dpt_file_path)

        fid = None
        try:
            fid = open(dpt_file_path, 'rb')
        except IOError:
            print('readFlowFile: could not open %s', dpt_file_path)

        tag = unpack('f', fid.read(4))[0]
        width = unpack('i', fid.read(4))[0]
        height = unpack('i', fid.read(4))[0]

        assert tag == TAG_FLOAT, ('readFlowFile(%s): wrong tag (possibly due to big-endian machine?)' % dpt_file_path)
        assert 0 < width and width < 100000, ('readFlowFile(%s): illegal width %d' % (dpt_file_path, width))
        assert 0 < height and height < 100000, ('readFlowFile(%s): illegal height %d' % (dpt_file_path, height))

        # arrange into matrix form
        depth_data = np.fromfile(fid, np.float32)
        depth_data = depth_data.reshape(height, width)

        fid.close()

        return depth_data

    #
    # ODB データセット実験用
    # --- addded by takama 2025/05/09 ---
    def read_exr_depth_v2(self, file_path, scale=1):
        exr_file = OpenEXR.InputFile(file_path)
        dw = exr_file.header()['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        channel = 'V'
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        depth_str = exr_file.channel(channel, pt)
        depth = np.frombuffer(depth_str, dtype=np.float32)
        depth = np.reshape(depth, (height, width))
        return depth
    # ------------------------------------

    def read_exr_depth(self, file_path, scale=100):
        exr_file = OpenEXR.InputFile(file_path)
        header = exr_file.header()
        dw = header['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        
        pixel_type = header['channels']['B'].type # 型判別用の変数
        
        if pixel_type == Imath.PixelType(Imath.PixelType.FLOAT):
            # FLOAT を使う場合
            print("The EXR file is stored in FLOAT format.")
            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT) # archiviz-flat を使った実行用
            try:
                depth_str = exr_file.channel('V', FLOAT)
            except:
                try:
                    depth_str = exr_file.channel('B', FLOAT)
                except:
                    try:
                        depth_str = exr_file.channel('G', FLOAT)
                    except:
                        try:
                            depth_str = exr_file.channel('R', FLOAT)
                        except:
                            raise ValueError("No valid depth channel found in the EXR file.")
            depth = np.frombuffer(depth_str, dtype=np.float32).reshape(size[1], size[0])
        
        elif pixel_type == Imath.PixelType(Imath.PixelType.HALF):
            # HALF を使う場合
            # Read the depth channel as 16-bit floats
            print("The EXR file is stored in HALF format.")
            HALF = Imath.PixelType(Imath.PixelType.HALF)
            depth_str = exr_file.channel('B', HALF)
            # Convert the binary string to a numpy array
            depth = np.frombuffer(depth_str, dtype=np.float16).reshape(size[1], size[0])
        
        else:
            print("The EXR file has an unsupported pixel type.")
        
        depth = depth*scale
        
        return depth
    
    def qvec2rotmat(self, qvec):
        return np.array([
            [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
            [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
            [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
    
    def angle_axis_to_quaternion(self, angle_axis: np.ndarray):
        angle = np.linalg.norm(angle_axis) # 回転角度を求める
        x = angle_axis[0] / angle # 回転軸の単位ベクトル
        y = angle_axis[1] / angle
        z = angle_axis[2] / angle

        qw = math.cos(angle / 2.0)
        qx = x * math.sqrt(1 - qw * qw)
        qy = y * math.sqrt(1 - qw * qw)
        qz = z * math.sqrt(1 - qw * qw)

        return np.array([qw, qx, qy, qz])
    