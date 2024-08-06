import numpy as np
import matplotlib.pyplot as plt
from analisys import create_movie_from_images
import os
import glob
import cv2
import imageio


def parse_cube_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Skip first two comment lines
    metadata = lines[2].split()
    num_atoms = int(metadata[0])
    origin = list(map(float, metadata[1:]))

    # Read the lattice vectors
    lattice_vectors = []
    for i in range(3):
        lattice_vectors.append(list(map(float, lines[3 + i].split()[1:])))

    # Read atom information
    atoms = []
    for i in range(num_atoms):
        atoms.append(list(map(float, lines[6 + i].split()[1:])))

    # Read voxel data
    voxel_data = []
    for line in lines[6 + num_atoms:]:
        voxel_data.extend(map(float, line.split()))

    voxel_data = np.array(voxel_data)

    # Determine the grid dimensions
    grid_size = int(round(len(voxel_data) ** (1 / 3)))
    voxel_data = voxel_data.reshape((grid_size, grid_size, grid_size))

    return {
        'num_atoms': num_atoms,
        'origin': origin,
        'lattice_vectors': lattice_vectors,
        'atoms': atoms,
        'voxel_data': voxel_data
    }


def plot_3d_potential(data, output_file):
    voxel_data = data['voxel_data']
    lattice_vectors = np.array(data['lattice_vectors'])

    # Create a meshgrid for plotting
    grid_size = voxel_data.shape[0]
    x = np.linspace(0, grid_size - 1, grid_size)
    y = np.linspace(0, grid_size - 1, grid_size)
    z = np.linspace(0, grid_size - 1, grid_size)
    X, Y, Z = np.meshgrid(x, y, z)

    # Apply lattice vectors to get the real coordinates
    coords = np.dot(np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T, lattice_vectors)
    X, Y, Z = coords[:, 0], coords[:, 1], coords[:, 2]
    V = voxel_data.flatten()

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(X, Y, Z, c=V, cmap='viridis')
    plt.colorbar(sc, ax=ax, label='Potential')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Potential Distribution')

    plt.savefig(output_file)
    plt.close()


def create_movie(directory):
    images = []
    image_files = sorted(glob.glob(os.path.join(directory, '*.png')))

    for image_path in image_files:
        if os.path.isfile(image_path):
            images.append(imageio.imread(image_path))

    # Create a movie from the images
    output_file = os.path.join(directory, 'movie.mp4')
    if images:
        imageio.mimsave(output_file, images, fps=10)

# Example usage
cube_files_directory = '/Volumes/My Passport/data_ofer/I0_1000/output_iter'
image_output_directory = '/tmp/test1'
output_movie_file = '/tmp/test1/potential_movie.avi'

# Ensure output directories exist
os.makedirs(image_output_directory, exist_ok=True)

# Process each cube file and generate 3D plots
cube_files = sorted(glob.glob(os.path.join(cube_files_directory, '**', 'vxc.cube'), recursive=True))
for i, cube_file in enumerate(cube_files):
    data = parse_cube_file(cube_file)
    output_image = os.path.join(image_output_directory, f'frame_{i:03d}.png')
    plot_3d_potential(data, output_image)

create_movie(image_output_directory)
print("Movie generated successfully!")
