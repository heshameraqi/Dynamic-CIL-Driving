# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""CARLA sensors."""


import os
import numpy as np
import cv2
import scipy

from collections import namedtuple
from scipy import misc

try:
    import numpy
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed.')

from .transform import Transform, Translation, Rotation, Scale


# ==============================================================================
# -- Helpers -------------------------------------------------------------------
# ==============================================================================


Color = namedtuple('Color', 'r g b')
Color.__new__.__defaults__ = (0, 0, 0)


Point = namedtuple('Point', 'x y z color')
Point.__new__.__defaults__ = (0.0, 0.0, 0.0, None)


def _append_extension(filename, ext):
    return filename if filename.lower().endswith(ext.lower()) else filename + ext


# ==============================================================================
# -- Sensor --------------------------------------------------------------------
# ==============================================================================


class Sensor(object):
    """
    Base class for sensor descriptions. Used to add sensors to CarlaSettings.
    """

    def __init__(self, name, sensor_type):
        self.SensorName = name
        self.SensorType = sensor_type
        self.PositionX = 0.2
        self.PositionY = 0.0
        self.PositionZ = 1.3
        self.RotationPitch = 0.0
        self.RotationRoll = 0.0
        self.RotationYaw = 0.0

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError('sensor.Sensor: no key named %r' % key)
            setattr(self, key, value)

    def set_position(self, x, y, z):
        self.PositionX = x
        self.PositionY = y
        self.PositionZ = z

    def set_rotation(self, pitch, yaw, roll):
        self.RotationPitch = pitch
        self.RotationYaw = yaw
        self.RotationRoll = roll

    def get_transform(self):
        '''
        Returns the camera to [whatever the camera is attached to]
        transformation.
        '''
        return Transform(
            Translation(self.PositionX, self.PositionY, self.PositionZ),
            Rotation(self.RotationPitch, self.RotationYaw, self.RotationRoll))

    def get_unreal_transform(self):
        '''
        Returns the camera to [whatever the camera is attached to]
        transformation with the Unreal necessary corrections applied.

        @todo Do we need to expose this?
        '''
        to_unreal_transform = Transform(Rotation(roll=-90, yaw=90), Scale(x=-1))
        return self.get_transform() * to_unreal_transform


class Camera(Sensor):
    """
    Camera description. This class can be added to a CarlaSettings object to add
    a camera to the player vehicle.
    """

    def __init__(self, name, **kwargs):
        super(Camera, self).__init__(name, sensor_type="CAMERA")
        self.PostProcessing = 'SceneFinal'
        self.ImageSizeX = 720
        self.ImageSizeY = 512
        self.FOV = 90.0
        self.set(**kwargs)

    def set_image_size(self, pixels_x, pixels_y):
        '''Sets the image size in pixels'''
        self.ImageSizeX = pixels_x
        self.ImageSizeY = pixels_y


class Lidar(Sensor):
    """
    Lidar description. This class can be added to a CarlaSettings object to add
    a Lidar to the player vehicle.
    """

    def __init__(self, name, **kwargs):
        super(Lidar, self).__init__(name, sensor_type="LIDAR_RAY_CAST")
        self.Channels = 32
        self.Range = 50.0
        self.PointsPerSecond = 56000
        self.RotationFrequency = 10.0
        self.UpperFovLimit = 10.0
        self.LowerFovLimit = -30.0
        self.ShowDebugPoints = False
        self.set(**kwargs)


# ==============================================================================
# -- SensorData ----------------------------------------------------------------
# ==============================================================================


class SensorData(object):
    """Base class for sensor data returned from the server."""
    def __init__(self, frame_number):
        self.frame_number = frame_number


class Image(SensorData):
    """Data generated by a Camera."""

    def __init__(self, frame_number, width, height, image_type, fov, raw_data):
        super(Image, self).__init__(frame_number=frame_number)
        assert len(raw_data) == 4 * width * height
        self.width = width
        self.height = height
        self.type = image_type
        self.fov = fov
        self.raw_data = raw_data
        self._converted_data = None

    @property
    def data(self):
        """
        Lazy initialization for data property, stores converted data in its
        default format.
        """
        if self._converted_data is None:
            from . import image_converter

            if self.type == 'Depth':
                # self._converted_data = image_converter.depth_to_array(self)
                self._converted_data = image_converter.depth_to_logarithmic_grayscale(self)
            elif self.type == 'SemanticSegmentation':
                # self._converted_data = image_converter.labels_to_array(self)
                self._converted_data = image_converter.labels_to_cityscapes_palette(self)
            else:
                self._converted_data = image_converter.to_rgb_array(self)
        return self._converted_data

    def save_to_disk(self, filename, format='.png'):
        """Save this image to disk (requires PIL installed)."""
        filename = _append_extension(filename, format)

        try:
            from PIL import Image as PImage
        except ImportError:
            raise RuntimeError(
                'cannot import PIL, make sure pillow package is installed')

        image = PImage.frombytes(
            mode='RGBA',
            size=(self.width, self.height),
            data=self.raw_data,
            decoder_name='raw')
        image = scipy.misc.imresize(image, [88, 200])  # scipy resize is better anti-aliasing
        # image = image.resize((200, 88))  # TODO: image resized handcoded for now
        # color = image.split()  # TODO: why was needed?
        # image = PImage.merge("RGB", color[2::-1])  # TODO: why was needed?

        folder = os.path.dirname(filename)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        # TODO: If else is ignored for now, so Depth & SemanticSegmentation images are not stored
        '''
        if self.type == 'Depth' or self.type == 'SemanticSegmentation':
            cv2.imwrite(filename, self.data)
        else:
            image.save(filename, quality=100)
        '''
        if not (self.type == 'Depth' or self.type == 'SemanticSegmentation'):
            misc.imsave(filename, image[:,:,[2,1,0]])

class PointCloud(SensorData):
    """A list of points."""

    def __init__(self, frame_number, array, color_array=None):
        super(PointCloud, self).__init__(frame_number=frame_number)
        self._array = array
        self._color_array = color_array
        self._has_colors = color_array is not None

    @property
    def array(self):
        """The numpy array holding the point-cloud.

        3D points format for n elements:
        [ [X0,Y0,Z0],
          ...,
          [Xn,Yn,Zn] ]
        """
        return self._array

    @property
    def color_array(self):
        """The numpy array holding the colors corresponding to each point.
        It is None if there are no colors.

        Colors format for n elements:
        [ [R0,G0,B0],
          ...,
          [Rn,Gn,Bn] ]
        """
        return self._color_array

    def has_colors(self):
        """Return whether the points have color."""
        return self._has_colors

    def apply_transform(self, transformation):
        """Modify the PointCloud instance transforming its points"""
        self._array = transformation.transform_points(self._array)

    def save_to_disk(self, filename, format='.ply'):
        """Save this point-cloud to disk as PLY format."""
        filename = _append_extension(filename, format)

        def construct_ply_header():
            """Generates a PLY header given a total number of 3D points and
            coloring property if specified
            """
            points = len(self)  # Total point number
            header = ['ply',
                      'format ascii 1.0',
                      'element vertex {}',
                      'property float32 x',
                      'property float32 y',
                      'property float32 z',
                      'property uchar diffuse_red',
                      'property uchar diffuse_green',
                      'property uchar diffuse_blue',
                      'end_header']
            if not self._has_colors:
                return '\n'.join(header[0:6] + [header[-1]]).format(points)
            return '\n'.join(header).format(points)

        def cart2pol(x, y, z):
            xy = x ** 2 + y ** 2
            rho = np.sqrt(xy + z ** 2)
            theta = np.arctan2(y, x)
            phi = np.arctan2(z, np.sqrt(xy))  # np.arctan2 retruns from -np.pi to np.pi
            # make angles from 0 to 360
            theta_deg = (np.degrees(theta) + 360) % 360
            phi_deg = (np.degrees(phi) + 360) % 360
            return rho, theta_deg, phi_deg

        if not self._has_colors:
            ply = '\n'.join(['{:.2f} {:.2f} {:.2f}'.format(
                *p) for p in self._array.tolist()])
        else:
            points_3d = numpy.concatenate(
                (self._array, self._color_array), axis=1)
            ply = '\n'.join(['{:.2f} {:.2f} {:.2f} {:.0f} {:.0f} {:.0f}'
                             .format(*p) for p in points_3d.tolist()])

        # Create folder to save if does not exist.
        folder = os.path.dirname(filename)
        if not os.path.isdir(folder):
            os.makedirs(folder)

        # Open the file and save with the specific PLY format.
        with open(filename, 'w+') as ply_file:
            ply_file.write('\n'.join([construct_ply_header(), ply]))

        # LiDAR PGM (TODO: this should be the same as the LiDAR configs in data-configuration-name)
        unreflected_value = 50
        upper_fov = 10
        lower_fov = -30
        nbrLayers = 32
        horizontal_angle_step = 2  # in degrees, should be divisible by 180
        lidar_data = np.array(self._array)
        x = lidar_data[:, 0]
        y = lidar_data[:, 1]
        z = -lidar_data[:, 2]
        (rho, theta_deg, phi_deg) = cart2pol(x, y, z)  # theta of 90 means front-facing
        unique_phis = (np.arange(upper_fov, lower_fov, -(upper_fov - lower_fov) / nbrLayers) -
                       (upper_fov - lower_fov) / (2. * nbrLayers) + 360) % 360
        unique_thetas = np.arange(180, 360,
                                  horizontal_angle_step) + horizontal_angle_step / 2.  # only range front-facing
        _lidar_pgm_image = np.ones((len(unique_phis), len(unique_thetas))) * unreflected_value
        for i in range(_lidar_pgm_image.shape[0]):  # For each layer
            for j in range(_lidar_pgm_image.shape[1]):  # For each group of neighboring beams
                # TODO: I found that multiplying with 1.1 (a number slightly bigger than 1) prevents PGM artifacts, why?
                indices_phi = np.abs(phi_deg - unique_phis[i]) <= 1.1 * (upper_fov - lower_fov) / (2. * nbrLayers)
                indices_theta = np.abs(theta_deg - unique_thetas[j]) <= horizontal_angle_step / 2.
                rhos = rho[indices_phi & indices_theta]
                if len(rhos) > 0:
                    _lidar_pgm_image[i, j] = np.mean(rhos)
        _lidar_pgm_image = 255.0 * np.repeat(_lidar_pgm_image[:, :, np.newaxis], 3, axis=2) / unreflected_value

        visualize = False
        if visualize:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(_lidar_pgm_image/255.)
            plt.show(block=False)

        cv2.imwrite(filename.replace(".ply", ".png"), _lidar_pgm_image)

        # draw top view lidar image
        '''try:
            scale = 2.0
            lidar_img_size = (int(200*scale), int(200*scale), 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_data *= scale
            lidar_data += 100.0 * scale
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            top_lidar_image = lidar_img.swapaxes(0, 1)
            cv2.imwrite(filename.replace(".ply", "_top.png"), top_lidar_image)
        except:
            print('Generating top view lidar image failed.')'''

    def __len__(self):
        return len(self.array)

    def __getitem__(self, key):
        color = None if self._color_array is None else Color(
            *self._color_array[key])
        return Point(*self._array[key], color=color)

    def __iter__(self):
        class PointIterator(object):
            """Iterator class for PointCloud"""

            def __init__(self, point_cloud):
                self.point_cloud = point_cloud
                self.index = -1

            def __next__(self):
                self.index += 1
                if self.index >= len(self.point_cloud):
                    raise StopIteration
                return self.point_cloud[self.index]

            def next(self):
                return self.__next__()

        return PointIterator(self)

    def __str__(self):
        return str(self.array)


class LidarMeasurement(SensorData):
    """Data generated by a Lidar."""

    def __init__(self, frame_number, horizontal_angle, channels, point_count_by_channel, point_cloud):
        super(LidarMeasurement, self).__init__(frame_number=frame_number)
        assert numpy.sum(point_count_by_channel) == len(point_cloud.array)
        self.horizontal_angle = horizontal_angle
        self.channels = channels
        self.point_count_by_channel = point_count_by_channel
        self.point_cloud = point_cloud

    @property
    def data(self):
        """The numpy array holding the point-cloud.

        3D points format for n elements:
        [ [X0,Y0,Z0],
          ...,
          [Xn,Yn,Zn] ]
        """
        return self.point_cloud.array

    def save_to_disk(self, filename, format='.ply'):
        """Save point-cloud to disk as PLY format."""
        self.point_cloud.save_to_disk(filename, format)
