import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.eval_depth import read_exr_depth_v2
from utils.eval_normal import read_exr_normal

def visualize_rgb_depth_normal(rgb, depth_map, normal_map, visualize_depth_max=8.0):
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 3, 1)
    rgb_ax = plt.gca()
    im = rgb_ax.imshow(rgb)
    plt.title('RGB')
    plt.axis('off')
    divider = make_axes_locatable(rgb_ax)

    plt.subplot(1, 3, 2)
    depth_ax = plt.gca()
    im = depth_ax.imshow(depth_map, cmap='plasma', vmin=0, vmax=visualize_depth_max)
    plt.title('Depth')
    plt.axis('off')
    divider = make_axes_locatable(depth_ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.subplot(1, 3, 3)
    normal_ax = plt.gca()
    im = normal_ax.imshow(normal_map)
    plt.title('Normal')
    plt.axis('off')
    divider = make_axes_locatable(normal_ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.show()
    plt.close()

def visualize_depth_normal(depth_map, normal_map, visualize_depth_max=8.0):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    depth_ax = plt.gca()
    im = depth_ax.imshow(depth_map, cmap='plasma', vmin=0, vmax=visualize_depth_max)
    plt.title('Depth')
    plt.axis('off')
    divider = make_axes_locatable(depth_ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.subplot(1, 2, 2)
    normal_ax = plt.gca()
    im = normal_ax.imshow(normal_map)
    plt.title('Normal')
    plt.axis('off')
    divider = make_axes_locatable(normal_ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()
    plt.close()

def visualize_depth(depth_map, visualize_depth_max=8.0, description="Depth"):
    plt.figure(figsize=(8, 4))
    if description is not None:
        plt.title(description)
    depth_ax = plt.gca()
    im = depth_ax.imshow(depth_map, cmap='plasma', vmin=0, vmax=visualize_depth_max)
    plt.axis('off')
    divider = make_axes_locatable(depth_ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()
    plt.close()

def visualize_normal(normal_map, description="Normal"):
    plt.figure(figsize=(8, 4))
    if description is not None:
        plt.title(description)
    normal_ax = plt.gca()
    im = normal_ax.imshow(normal_map)
    plt.title('Normal')
    plt.axis('off')
    divider = make_axes_locatable(normal_ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()
    plt.close()

def visualize_depth_comparison(depth_map1, depth_map2, visualize_depth_max=8.0, description1=None, description2=None):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    if description1 is not None:
        plt.title(description1)
    depth_ax = plt.gca()
    im = depth_ax.imshow(depth_map1, cmap='plasma', vmin=0, vmax=visualize_depth_max)
    plt.axis('off')
    divider = make_axes_locatable(depth_ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.subplot(1, 2, 2)
    if description2 is not None:
        plt.title(description2)
    depth_ax = plt.gca()
    im = depth_ax.imshow(depth_map2, cmap='plasma', vmin=0, vmax=visualize_depth_max)
    plt.axis('off')
    divider = make_axes_locatable(depth_ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()
    plt.close()

def visualize_normal_conparison(normal_map1, normal_map2, description1=None, description2=None):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    if description1 is not None:
        plt.title(description1)
    normal_ax = plt.gca()
    im = normal_ax.imshow(normal_map1)
    plt.axis('off')
    divider = make_axes_locatable(normal_ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.subplot(1, 2, 2)
    if description2 is not None:
        plt.title(description2)
    normal_ax = plt.gca()
    im = normal_ax.imshow(normal_map2)
    plt.axis('off')
    divider = make_axes_locatable(normal_ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()