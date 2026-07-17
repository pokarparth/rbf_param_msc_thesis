import numpy as np
from simpeg.survey import BaseSurvey
from typing import Generator

class RectilinearRadiographPixel(object):
    """
    Defines a radiograph pixel. Includes implementation
    of sampling a random raypath direction within the pixel
    and computing solid angle differential for the pixel.
    """
    def __init__(
        self,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
    ) -> None:
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

    def get_random_ray(self) -> np.ndarray:
        """
        Sample a random raypath direction within the pixel.
        """
        tan_theta_x = self.x_bounds[0] + (self.x_bounds[1] - self.x_bounds[0])*np.random.rand()
        tan_theta_y = self.y_bounds[0] + (self.y_bounds[1] - self.y_bounds[0])*np.random.rand()
        rd = tan_theta_x, tan_theta_y
        return rd

    def solid_angle_differential(self, tan_theta_x: float, tan_theta_y: float) -> float:
        """
        Compute the solid angle differential for the pixel.
        """
        return 1.0 / (1.0 + tan_theta_x**2 + tan_theta_y**2)**(3/2)

    def get_volume(self) -> float:
        """
        Compute the volume of the pixel.

        Returns:
            float: volume of the pixel
        """
        return (self.x_bounds[1] - self.x_bounds[0]) * (self.y_bounds[1] - self.y_bounds[0])

class MuonSensor(object):
    """
    Toy muon tomography sensor class. Assume each muon measurement consists of the
    opacity (line integral of density along a raypath) for a single raypath. Each
    sensor contains measurements on a 2D cartesian tensor product grid of raypath
    directions. The raypath directions are specified by the tan(theta_x) and
    tan(theta_y) coordinates of the raypath direction, where theta_x and theta_y
    are the angles between vertical and the x and y components of the raypath
    direction, respectively.

    Constructor inputs:

    loc: np.ndarray -> location of the sensor as length 3 np.ndarray
    rgrid_x: np.ndarray -> 1D array of tan(theta_x) coordinates of raypath grid
    rgrid_y: np.ndarray -> 1D array of tan(theta_y) coordinates of raypath grid
    """
    def __init__(self,
        loc: np.ndarray,
        rgrid_x: np.ndarray,
        rgrid_y: np.ndarray
    ) -> None:
        self.loc = loc
        self.rgrid_x = rgrid_x
        self.rgrid_y = rgrid_y

    def pixels(self) -> Generator[RectilinearRadiographPixel, None, None]:
        """
        Generator for the pixels in the sensor's grid.
        """
        for ix in range(len(self.rgrid_x)-1):
            for iy in range(len(self.rgrid_y)-1):
                yield RectilinearRadiographPixel(
                    (self.rgrid_x[ix], self.rgrid_x[ix+1]),
                    (self.rgrid_y[iy], self.rgrid_y[iy+1]),
                )

# SimPEG requires a survey class. More useful for problems
# where fields and data are different.
class MuonSurvey(BaseSurvey):
    def __init__(self, nD: int) -> None:
        self._nD = nD

    @property
    def nD(self) -> int:
        return self._nD
    @nD.setter
    def nD(self, value) -> None:
        self._nD = value