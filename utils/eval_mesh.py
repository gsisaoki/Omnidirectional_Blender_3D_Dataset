import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import trimesh

# from MonoSDF
def nn_correspondance(verts1, verts2):
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    distances = distances.reshape(-1)

    return distances

def evaluate(mesh_pred, mesh_trgt, threshold=.05, down_sample=.02):     # NOTE: threshold is [meter]，e.g., threshold=.05 means 0.05m(=5cm)
    pcd_trgt = o3d.geometry.PointCloud()
    pcd_pred = o3d.geometry.PointCloud()
    
    pcd_trgt.points = o3d.utility.Vector3dVector(mesh_trgt.vertices[:, :3])
    pcd_pred.points = o3d.utility.Vector3dVector(mesh_pred.vertices[:, :3])

    num_points_pred = len(pcd_pred.points)
    num_points_trgt = len(pcd_trgt.points)

    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    dist1 = nn_correspondance(verts_pred, verts_trgt)   # from verts_trgt to verts_pred
    dist2 = nn_correspondance(verts_trgt, verts_pred)   # from verts_pred to verts_trgt
    chamfer_distance = 0.5 * (np.mean(dist1) + np.mean(dist2))

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)

    metrics = {
        'Acc': np.mean(dist2),
        'Comp': np.mean(dist1),
        'CD': chamfer_distance,
        'Prec@{}'.format(threshold): precision,
        'Recal@{}'.format(threshold): recal,
        'F-score@{}'.format(threshold): fscore,
        '# of points before downsampling (pred)': num_points_pred,
        '# of points before downsampling (gt)': num_points_trgt,
        '# of points after downsampling (pred)': len(verts_pred),
        '# of points after downsampling (gt)': len(verts_trgt),
    }
    return metrics