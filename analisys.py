import os
import json
import matplotlib.pyplot as plt
import imageio

def extract_and_plot_quality(dipole_dir, save=False):
    quality_data = {}
    for dir in sorted(os.listdir(dipole_dir))[1:]:
        defocus_dir = os.path.join(dipole_dir, dir)
        data_path = os.path.join(defocus_dir, 'measurement_data.json')
        if not os.path.isfile(data_path):
            continue
        with open(data_path, 'r') as f:
            data = json.load(f).get('measurement_data')
            dipole_len = data.get('dipole_len')
            defocus = data.get('defocus')
            quality_measures = data.get('quality_measures')
            if quality_measures == 0:
                quality_value = 0
            else:
                quality_value = quality_measures.get('quality_value')

            quality_data[defocus] = quality_value

    # Sort data by defocus distance
    sorted_quality_data = dict(sorted(quality_data.items()))
    defocus_um = [d * 1e6 for d in sorted_quality_data.keys()]
    # Plot the quality data
    plt.figure(figsize=(10, 6))
    plt.plot(defocus_um, list(sorted_quality_data.values()), marker='o')
    plt.title(f'Image Quality vs. Defocus Distance for Dipole Length {dipole_dir.split("/")[-1].split("_")[-1]}')
    plt.xlabel('Defocus Distance [um]')
    plt.ylabel('Image Quality')
    plt.grid(True)
    plot_path = f'{dipole_dir}/image_quality_plot.png'
    if save:
        plt.savefig(plot_path)
    plt.show()


def create_movie_from_images(dipole_dir):
    images = []

    for dir in sorted(os.listdir(dipole_dir))[1:]:
        defocus_dir = os.path.join(dipole_dir, dir)
        image_path = os.path.join(defocus_dir, 'image.png')
        if os.path.isfile(image_path):
            images.append(imageio.imread(image_path))

    # Create a movie from the images
    output_file = os.path.join(dipole_dir, 'movie.mp4')
    if images:
        imageio.mimsave(output_file, images, fps=10)  # Adjust fps as needed

if __name__ == '__main__':
    base_dir = '/Users/dadonofek/Documents/phase_imaging_db/v_1/dipole'
    for dir in sorted(os.listdir(base_dir))[1:]:
        dipole_dir = os.path.join(base_dir, dir)
        extract_and_plot_quality(dipole_dir, save=True)
        create_movie_from_images(dipole_dir)
