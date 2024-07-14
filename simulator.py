import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from analisys import extract_and_plot_quality, create_movie_from_images
import os
import json
import argparse


class TEST():

    def __init__(self, test_args):
        self.test_args = test_args
        self.db_path = test_args.get('db_path')
        self.sim_version = test_args.get('sim_version')
        self.sim_params = test_args.get('sim_params')
        self.save_results = test_args.get('save_results') == 'True'
        self.beam_params = test_args.get('beam_params')
        self.sample_params = test_args.get('sample_params')
        self.image_quality_weights = test_args.get('image_quality_weights')
        self.test_measurements = []
        self.grid_size = self.sim_params.get('grid_size')
        self.defocus_distances = test_args.get('iteration_params').get('defocuses')
        self.dipole_folder_path = None

        x = np.linspace(-self.sim_params.get('x_max'), self.sim_params.get('x_max'), self.sim_params.get('grid_size'))
        y = x
        self.X, self.Y = np.meshgrid(x, y)

    def run_test(self, user_args):
        dipole_len = user_args.dipole_len*1e-9
        self.dipole_folder_path = f'{self.db_path}/dipolelen_{dipole_len*1e9:.1f}[nm]'
        os.makedirs(self.dipole_folder_path, exist_ok=True)

        for defocus in self.defocus_distances:
            meas = MEASUREMENT(self.dipole_folder_path, defocus, dipole_len, self)
            self.test_measurements.append(meas)
            meas.run_measurement()
            print(f'measurements for dipole_len: {dipole_len*1e9:.2f}[nm], defocus: {defocus*1e6:.2f}[um] done')

    def run_post_test(self):
        self.analyze_results()

    def analyze_results(self):
        extract_and_plot_quality(self.dipole_folder_path, save=True)
        create_movie_from_images(self.dipole_folder_path)

    def make_sample(self):
        hole_radius = self.sample_params.get('hole_radius', 1e-7)
        sample_attenuation = self.sample_params.get('sample_attenuation', 0.3)
        sample_phase_gain = self.sample_params.get('sample_phase_gain', 0.3)

        hole_indices = np.sqrt(self.X ** 2 + self.Y ** 2) <= hole_radius
        carbon_params = (1 - sample_attenuation) * np.exp(1j * np.pi * sample_phase_gain)
        sample = np.where(hole_indices, 1, carbon_params)
        return sample

    def make_hexagonal_sample(self):
        hole_spacing = self.sample_params.get('hole_spacing', 1e-7)
        hole_centers_x = np.arange(self.X.min(), self.X.max(), hole_spacing)
        hole_centers_y = np.arange(self.Y.min(), self.Y.max(), hole_spacing * np.sqrt(3) / 2)

        # Create a boolean array for holes
        hole_indices = np.zeros_like(self.X, dtype=bool)

        for i, hx in enumerate(hole_centers_x):
            for j, hy in enumerate(hole_centers_y):
                # Shift every second row by half the hole spacing
                if j % 2 == 1:
                    hx_shifted = hx + hole_spacing / 2
                else:
                    hx_shifted = hx

                # Calculate distance from each point to the current hole center
                distances = np.sqrt((self.X - hx_shifted) ** 2 + (self.Y - hy) ** 2)
                # Update hole indices where the distance is less than or equal to the hole radius
                hole_indices = np.logical_or(hole_indices, distances <= self.sample_params.get('hole_radius', 5e-7))

        # Calculate carbon parameters (attenuation and phase gain)
        carbon_params = (1 - self.sample_params.get('sample_attenuation', 0.3)) * np.exp(
            1j * np.pi * self.sample_params.get('sample_phase_gain', 0.3))

        # Create the sample with holes and carbon material properties
        sample = np.where(hole_indices, 1, carbon_params)

        return sample

    def fresnel_propagator(self, R, wavelength):
        k = 2 * np.pi / wavelength
        return (-1j / (R * wavelength)) * np.exp(1j * k * (self.X ** 2 + self.Y ** 2) / (2 * R))

    def make_dipole_sample(self, dipole_len, res_folder):
        phase_scale_factor = 0.095
        phase = phase_scale_factor * np.pi * np.log(
            ((self.X + dipole_len / 2) ** 2 + self.Y ** 2) /
            ((self.X - dipole_len / 2) ** 2 + self.Y ** 2))
        self.plot(phase,
                  f'Phase Distribution of an electron-hole Dipole at d = {dipole_len*1e9} [nm])',
                  path=res_folder,
                  filename='sample_phase.png')

        return np.exp(1j * phase)

    def gaussian_beam(self, x, y, **kwargs):
        E0 = kwargs.get('E0', 1.0)
        w0 = kwargs.get('w0', 100e-9)
        wavelength = kwargs.get('wavelength', 0.0251e-9)
        center = kwargs.get('center', [0, 0])
        z = kwargs.get('z', 0)

        # Calculate parameters
        k = 2 * np.pi / wavelength
        z_R = np.pi * w0 ** 2 / wavelength
        w_z = w0 * np.sqrt(1 + (z / z_R) ** 2)
        R_z = z * (1 + (z_R / z) ** 2) if z != 0 else np.inf
        psi_z = np.arctan(z / z_R)

        # Adjust coordinates to center the beam
        x_centered = x - center[0]
        y_centered = y - center[1]

        # Amplitude part
        amplitude = (w0 / w_z) * np.exp(-(x_centered ** 2 + y_centered ** 2) / w_z ** 2)

        # Phase part
        phase = np.exp(-1j * (k * z + k * (x_centered ** 2 + y_centered ** 2) / (2 * R_z) - psi_z))

        # Electric field
        E = E0 * amplitude * phase
        return E

    import matplotlib.pyplot as plt

    def plot(self, I, title, path=None, filename=None, cmap='gray'):
        plt.figure(figsize=(8, 6))
        plt.imshow(I, extent=(self.X.min(), self.X.max(), self.Y.min(), self.Y.max()), cmap=cmap)
        plt.title(title)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.colorbar(label='Intensity')

        if self.save_results and path is not None:
            plt.savefig(f'{path}/{filename}')
        # plt.show()  # Uncomment if you want to display the plot
        plt.close()  # Close the plot to free up memory

    def complex_convolve2d_v2(self, f, g):
        # Add zero padding to avoid aliasing
        pad_x = f.shape[0] // 2
        pad_y = f.shape[1] // 2
        f_padded = np.pad(f, ((pad_x, pad_x), (pad_y, pad_y)), mode='constant')
        g_padded = np.pad(g, ((pad_x, pad_x), (pad_y, pad_y)), mode='constant')

        # Perform FFT
        F_fft = np.fft.fft2(np.fft.fftshift(f_padded))
        G_fft = np.fft.fft2(np.fft.fftshift(g_padded))

        # Convolve in frequency domain
        convolved_fft = F_fft * G_fft

        # Perform inverse FFT
        convolved = np.fft.ifft2(convolved_fft)

        # Shift and crop the result to original size
        convolved = np.fft.ifftshift(convolved)
        convolved = convolved[pad_x:-pad_x, pad_y:-pad_y]

        return convolved

    def take_photo(self, incident, sample, propogator, crop_factor=1.0):
        post_sample_beam = incident * sample
        beam_propagated = self.complex_convolve2d_v2(post_sample_beam, propogator)
        image = np.abs(beam_propagated) ** 2
        if crop_factor < 1:
            center = image.shape[0] // 2
            w = int(image.shape[0] * crop_factor / 2)
            image = image[center - w:center + w, center - w:center + w]
        return image

    def image_quality(self, image, path=None):
        """
        Evaluate the quality of the image based on periodicity and amplitude differences.
        """
        line_data = image[image.shape[0] // 2 + 10, :]

        # Smooth the line data
        smoothed_line_data = gaussian_filter1d(line_data, sigma=2)

        # Plot the smoothed values
        plt.figure(figsize=(10, 4))
        plt.plot(smoothed_line_data)
        plt.title('Middle Line Data')
        plt.xlabel('X-axis')
        plt.ylabel('Intensity')
        plt.grid(True)
        if self.save_results and path is not None:
            plt.savefig(f'{path}/line_plot.png')
        # plt.show()


        # Check if the data is periodic
        periodic, mean_period_length, mean_amplitude_diff, num_periods = self.is_periodic(smoothed_line_data)

        if not periodic:
            return 0

        # Calculate the quality value
        quality_value = mean_period_length * self.image_quality_weights.get('period_len')\
                        + mean_amplitude_diff * self.image_quality_weights.get('amplitude')\
                        + num_periods * self.image_quality_weights.get('num_periods')
        quality_measures = {
            'quality_value': round(quality_value, 3),
            'mean_period_length': mean_period_length,
            'mean_amplitude_diff': mean_amplitude_diff,
            'num_periods': num_periods
        }
        return quality_measures

    def is_periodic(self, data, min_peak_distance=15, min_num_periods=4):
        """
        Check if the data is periodic by finding peaks and evaluating their uniformity.
        """
        # Find all peaks
        peaks, _ = find_peaks(data)

        if len(peaks) < 2:
            return False, 0, 0, 0

        # Find the center of the data
        center = len(data) // 2

        peaks = [peak for peak in peaks if peak <= center] + [
            min([peak for peak in peaks if peak > center], default=center + 1)]

        # Sort peaks based on their distance from the center
        peaks = peaks[::-1]

        # Filter peaks to maintain minimum peak distance
        filtered_peaks = [peaks[0]]
        for peak in peaks[1:]:
            if abs(peak - filtered_peaks[-1]) >= min_peak_distance:
                filtered_peaks.append(peak)
            else:
                break

        # Ensure there are at least min_num_periods valid peaks
        if len(filtered_peaks) < min_num_periods:
            return False, 0, 0, 0

        peak_distances = np.abs(np.diff(filtered_peaks))

        # Find the local minima (lows) between the filtered peaks
        lows = []
        for i in range(len(filtered_peaks) - 1):
            segment = data[filtered_peaks[i + 1]:filtered_peaks[i]]
            if len(segment) > 0:
                lows.append(filtered_peaks[i + 1] + np.argmin(segment))

        # Calculate the mean amplitude difference between peaks and lows
        amplitude_differences = [
            data[filtered_peaks[i]] - data[lows[i]]
            for i in range(len(lows))
        ]

        period_length = np.mean(peak_distances)
        mean_amplitude_diff = np.mean(amplitude_differences)
        num_periods = len(peak_distances)
        return True, period_length, mean_amplitude_diff, num_periods


class MEASUREMENT():
    def __init__(self, dipole_folder_path, defocus, dipole_len, test):
        self.test = test
        self.defocus = defocus
        self.dipole_len = dipole_len
        self.incident_beam = None
        self.incident_beam_type = None
        self.result_folder = None
        self.propogator = None
        self.sample = None
        self.image = None
        self.quality_measures = None
        self.dipole_folder_path = dipole_folder_path

    def run_measurement(self):
        self.incident_beam = np.ones(shape=(self.test.grid_size, self.test.grid_size))
        self.incident_beam_type = 'plane_wave'
        self.result_folder = f'{self.dipole_folder_path}/defocus_{self.defocus*1e6:.1f}[um]'
        os.makedirs(self.result_folder, exist_ok=True)
        self.propogator = self.test.fresnel_propagator(self.defocus, self.test.beam_params.get('wavelength'))
        self.sample = self.test.make_dipole_sample(self.dipole_len, self.result_folder)
        self.image = self.test.take_photo(self.incident_beam, self.sample, self.propogator, crop_factor=0.5)
        self.image /= np.amax(self.image)  # normalyze intencity to 1
        self.test.plot(self.image,
                       title=f'image at defocus = {self.defocus * 1e6:.1f} [um], dipole_len = {self.dipole_len * 1e9:.1f} [nm]',
                       path=self.result_folder,
                       filename='image.png')

        self.quality_measures = self.test.image_quality(self.image,
                                                   path=self.result_folder)
        if self.test.save_results:
            self.save_to_results_directory()

        plt.close()

    def save_to_results_directory(self):
        measurement_data = {'metadata': {'sim_version': self.test.sim_version,
                                         'sim_params': self.test.sim_params,
                                         'beam_params': self.test.beam_params,
                                         'sample_params': self.test.sample_params,
                                         'image_quality_weights': self.test.image_quality_weights,
                                         },

                            'measurement_data': {'defocus': self.defocus,
                                                 'dipole_len': self.dipole_len,
                                                 'incident_beam_type': self.incident_beam_type,
                                                 'quality_measures': self.quality_measures
                                                 },
                            }
        with open(f'{self.result_folder}/measurement_data.json', 'w') as f:
            json.dump(measurement_data, f, indent=4)



if __name__ == '__main__':

    ############ Simulation Params ############
    args = {
        'db_path': '/Users/dadonofek/Documents/phase_imaging_db/v_1/dipole',
        'sim_version': 1,
        'save_results': 'True',
        'beam_params': {
            'E0': 1.0,
            'w0': 800e-9,
            'wavelength': 0.0251e-9,
            'defocus': 1e-4,
            'center': [2e-7, 0]
        },

        'image_quality_weights': {
            'period_len': 0.1,
            'amplitude': 100,
            'num_periods': 1

        },
        'sample_params': {},
        'sim_params': {'grid_size': 2000,
                       'x_max': 6e-7
                       },
        'iteration_params': {'defocuses': np.linspace(1, 100, 100)*1e-6
                             }
    }



    # dipole_lengths = np.linspace(1e-9, 10e-9, 10)
    # defocuses = np.linspace(1, 10, 10)*1e-5

    '''
    some insights:
    every pixel is about 1 angstrem- i need dipole to be at least 1 nm in length.
    a measure of the picture quality is intencity delta and fringes wavelength
    maybe generalyze the integral to calc phase gain
    '''
    ###########################################

    parser = argparse.ArgumentParser(description="Run TEST with specified dipole_len")
    parser.add_argument("--dipole_len", type=float, required=True, help="Dipole length to measure in nm")
    user_args = parser.parse_args()
    test = TEST(test_args=args)
    test.run_test(user_args)
    test.run_post_test()

