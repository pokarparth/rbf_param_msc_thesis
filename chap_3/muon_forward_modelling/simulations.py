from collections import OrderedDict
from typing import Any, Dict, Optional, Union

from discretize import TensorMesh
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
import scipy.sparse as sparse
import simpeg.maps as maps
from simpeg.simulation import LinearSimulation, BaseSimulation

from . import MuonSensor, MuonSurvey

from .opacity import get_opacity_matrix_no_integration, get_opacity_matrix_monte_carlo_integration


class OpacitySimulation(LinearSimulation):
    """
    SimPEG simulation class for toy muon tomography problem.
    Performs linear forward modelling and sensitivity calculations
    on a SimPEG TensorMesh.

    Linear forward operator is constructed as a scipy CSR sparse matrix G. This
    class provides the methods required to interface with SimPEG inversion
    routines.

    Note that following SimPEG conventions, the forward modelling methods in this
    class compute G*f(m), where f is called a mapping function. In general, this
    allows the inversion to optimize over a transformed or parameterized model
    space, rather than working directly with the mesh cell density values.

    Constructor inputs:

    mesh: discretize.TensorMesh -> mesh for the problem
    sensors: dict -> dictionary of sensors for the problem
    model_map: SimPEG.maps -> mapping from model to physical parameters.
    """

    def __init__(
        self,
        mesh: TensorMesh,
        sensors: Dict[Any, MuonSensor],
        model_map: maps.IdentityMap = maps.IdentityMap(),  # For some reason, IdentityMap is the base class for all othe mappings
        **kwargs
    ) -> None:
        super().__init__(mesh=mesh, **kwargs)
        self.model_map = model_map
        self.mesh = mesh
        self.sensors = sensors

        # Store number of data for each sensor, for mapping
        # between vector of all data and dicts of data for each sensor.
        self.n_data = np.concatenate(
            (
                [0],
                np.cumsum(
                    [len(sensor.rgrid_x) * len(sensor.rgrid_y) for sensor in sensors.values()]
                ),
            )
        )

        # Get the forward operator
        self.G: sparse.csr_matrix = get_opacity_matrix_no_integration(mesh, sensors)

        nD = self.G.shape[0]
        self.survey = MuonSurvey(nD)

    def getJ(self, m, f=None):
        """
        Return the sensitivity matrix.
        """
        J = self.G @ self.model_map.deriv(m)
        return J

    # minimum functionalities for SimPEG simulation object
    def Jvec(self, m, v, f=None):
        """
        Return the sensitivity matrix multiplied by a vector.
        """
        return self.G @ self.model_map.deriv(m, v)

    def Jtvec(self, m, v, f=None):
        """
        Return the adjoint sens matrix multiplied by a vector.
        """
        return self.model_map.deriv(m).T @ (self.G.T @ v)

    def fields(self, m=None):
        """
        This method is required. For this problem, fields
        and data are the same.
        """
        return self.G @ self.model_map._transform(m)

    def get_data(self, m: np.ndarray) -> OrderedDict:
        """
        Compute the data for a given model m and split by sensor.
        This method is for convenience. SimPEG requires the data
        as a numpy array.
        """
        dall = self.fields(m)
        return OrderedDict(
            [
                (key, dall[self.n_data[i] : self.n_data[i + 1]])
                for i, key in enumerate(self.sensors.keys())
            ]
        )

    def dpred(self, m, f=None):
        """
        Return the predicted data for a given model.
        """
        return self.fields(m)

    def residual(self, m, dobs, f=None):
        r"""
        The data residual:

        .. math::

            \mu_\\text{data} = \mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs}

        :param numpy.ndarray m: geophysical model
        :param numpy.ndarray f: fields
        :rtype: numpy.ndarray
        :return: data residual
        """
        return self.dpred(m) - dobs


class MuonCountSimulation(BaseSimulation):
    """
    SimPEG simulation class for simplified muon tomography problem.
    Performs forward modelling and sensitivity calculations
    on a SimPEG TensorMesh.

    Opacity operator forward operator is constructed as a scipy CSR sparse matrix G. This
    class provides the methods required to interface with SimPEG inversion
    routines.

    Note that following SimPEG conventions, the forward modelling methods in this
    class compute G*f(m), where f is called a mapping function. In general, this
    allows the inversion to optimize over a transformed or parameterized model
    space, rather than working directly with the mesh cell density values.

    Constructor inputs:

    mesh: discretize.TensorMesh -> mesh for the problem
    sensors: dict -> dictionary of sensors for the problem
    exposure_time: float -> exposure time for the sensors in seconds.
    model_map: SimPEG.maps -> mapping from model to physical parameters.
    """

    def __init__(
        self,
        mesh: TensorMesh,
        sensors: Dict[Any, MuonSensor],
        exposure_time: float,
        n_rays_per_radiograph_pixel: int = 10,
        background_opacity: Optional[np.ndarray] = None,
        model_map: maps.IdentityMap = maps.IdentityMap(),  # For some reason, IdentityMap is the base class for all othe mappings
        use_precomputed_ray_directions: bool = False,
        ray_directions: Optional[np.ndarray] = None,
         survey=None,
        #**kwargs
    ) -> None:
        super().__init__(survey=survey)#mesh=mesh, **kwargs)
        self.model_map = model_map
        self.mesh = mesh
        self.exposure_time = exposure_time
        self.n_rays_per_radiograph_pixel = n_rays_per_radiograph_pixel
        self.sensors = sensors
        self.background_opacity = background_opacity
        if self.background_opacity is None:
            self.background_opacity = 0.0
        self.use_precomputed_ray_directions = use_precomputed_ray_directions
        self.ray_directions = ray_directions

        # Store number of data for each sensor, for mapping
        # between vector of all data and dicts of data for each sensor.
        self.n_data = np.concatenate(
            (
                [0],
                np.cumsum(
                    [
                        (len(sensor.rgrid_x) - 1) * (len(sensor.rgrid_y) - 1)
                        for sensor in sensors.values()
                    ]
                ),
            )
        )

        # Get the forward operator components
        self.n_radiograph_pixels = sum(
            [(len(sensor.rgrid_x) - 1) * (len(sensor.rgrid_y) - 1) for sensor in sensors.values()]
        )
        G, self.ray_directions, integration_weights = get_opacity_matrix_monte_carlo_integration(
            mesh,
            sensors,
            n_rays_per_pixel=n_rays_per_radiograph_pixel,
            use_precomputed_ray_directions=use_precomputed_ray_directions,
            ray_directions=ray_directions,
        )
        self._Opacity_mtx: sparse.csr_matrix = G
        self._Integration_mtx: sparse.csr_matrix = get_solid_angle_integration_matrix(
            integration_weights,
            self.n_radiograph_pixels,
            n_rays_per_radiograph_pixel,
        )
        self.zenith_angle = get_zenith_angle(self.ray_directions[:, 0], self.ray_directions[:, 1])
        self.Averaging_mtx = pixel_averaging_matrix(
            self.n_radiograph_pixels, n_rays_per_radiograph_pixel
        )

        @jax.jit
        def intensity_jvp(opacity: ArrayLike, vector: ArrayLike) -> jax.Array:
            return jax.jvp(
                lambda opacity: get_intensity(opacity, self.zenith_angle),
                (opacity,),
                (vector,),
            )[1]
        self.intensity_jvp = intensity_jvp

        @jax.jit
        def intensity_vjp(opacity: ArrayLike, vector: ArrayLike) -> jax.Array:
            vjp_fn = jax.vjp(
                lambda opacity: get_intensity(opacity, self.zenith_angle), opacity
            )[1]
            return vjp_fn(vector)[0]
        self.intensity_vjp = intensity_vjp

        nD = self._Integration_mtx.shape[0]
        self.survey = MuonSurvey(nD)


    def getJ(self, m, f=None):
        """
        Return the sensitivity matrix.
        """
        raise NotImplementedError
        # opacity = self.background_opacity + self._Opacity_mtx @ self.model_map._transform(m)
        # dOdm = self._Opacity_mtx @ self.model_map.deriv(m)
        # dIdO_fn = jax.grad(
        #     get_intensity,
        #     argnums=0,
        # )
        # dIdO = dIdO_fn(opacity, self.zenith_angle)
        # dIdO = sparse.diags(np.array(dIdO), format="csr")
        # J = self.exposure_time * self._Integration_mtx @ dIdO @ dOdm
        # return J

    # minimum functionalities for SimPEG simulation object
    def Jvec(self, m, v, f=None):
        """
        Return the sensitivity matrix multiplied by a vector.
        """
        d_opacity_dm = self._Opacity_mtx @ self.model_map.deriv(m, v)
        opacity = self.background_opacity + self._Opacity_mtx @ self.model_map._transform(m)
        return np.array(
            self.exposure_time
            * self._Integration_mtx
            @ np.array(self.intensity_jvp(opacity, d_opacity_dm))
        )

    def Jtvec(self, m, v, f=None):
        """
        Return the adjoint sens matrix multiplied by a vector.
        """
        opacity = self.background_opacity + self._Opacity_mtx @ self.model_map._transform(m)
        tmp = np.array(
            self.intensity_vjp(opacity, self.exposure_time * self._Integration_mtx.T @ v)
        )
        return np.array(self.model_map.deriv(m).T @ (self._Opacity_mtx.T @ tmp))
        # return self.model_map.deriv(m).T @ (self._G.T @ v)

    def fields(self, m=None) -> np.ndarray:
        """
        This method is required. For this problem, fields
        and data are the same.
        """
        dt = self.exposure_time
        return (
            dt
            * self._Integration_mtx
            @ get_intensity(
                self.background_opacity + self._Opacity_mtx @ self.model_map._transform(m),
                self.zenith_angle,
            )
        )

    def get_data(self, m: np.ndarray) -> np.ndarray:
        """
        Compute the data for a given model m. Keeping this method for consistency
        """
        return self.fields(m)

    def get_opacity(self, m: np.ndarray) -> OrderedDict:
        """
        Compute the average opacity in each pixel for a given model.
        """
        return self.Averaging_mtx @ (self._Opacity_mtx @ self.model_map._transform(m))
        # return OrderedDict(
        #     [
        #         (key, dall[self.n_data[i] : self.n_data[i + 1]])
        #         for i, key in enumerate(self.sensors.keys())
        #     ]
        # )

    def dpred(self, m, f=None) -> np.ndarray:
        """
        Return the predicted data for a given model.
        """
        return self.fields(m)

    def residual(self, m, dobs, f=None):
        r"""
        The data residual:

        .. math::

            \mu_\\text{data} = \mathbf{d}_\\text{pred} - \mathbf{d}_\\text{obs}

        :param numpy.ndarray m: geophysical model
        :param numpy.ndarray f: fields
        :rtype: numpy.ndarray
        :return: data residual
        """
        return self.dpred(m) - dobs


gram_per_cc = 1e6

@jax.jit
def get_intensity(opacity: ArrayLike, theta: ArrayLike) -> np.ndarray:
    hrho = opacity #/ gram_per_cc
    n = 1.850 + 4.650e-6 * hrho**1.670
    intensity = vertical_intensity(hrho)
    oblique_rays = theta != 0.0
    # intensity[oblique_rays] = intensity[oblique_rays] * np.abs(np.cos(theta[oblique_rays])) ** n
    intensity = jnp.where(oblique_rays, intensity * jnp.abs(jnp.cos(theta)) ** n, intensity)
    return intensity



def vertical_intensity(hrho: ArrayLike) -> jax.Array:
    return 1.740e6 / ((400.0 + hrho) * (11.0 + hrho) ** 1.530) * jnp.exp(-hrho * 8e-4)


def get_acceptance(tan_theta_x, tan_theta_y) -> Union[float, np.ndarray]:
    """
    Get acceptance of a ray angle given in rectilinear coordinates (tan_theta_x, tan_theta_y).
    """
    A = 1.0  # cross-sectional area in m^2
    th = 0.25  # thickness in m
    wx = 1.0  # width in m
    wy = 1.0  # length in m
    return np.maximum(0.0, (wx - th * abs(tan_theta_x)) * (wy - th * abs(tan_theta_y)) / (wx * wy))


def get_zenith_angle(tan_theta_x: np.ndarray, tan_theta_y: np.ndarray) -> np.ndarray:
    """
    Get the zenith angle of a sensor given by sensor_name.
    """
    return np.arctan(np.sqrt(tan_theta_x**2 + tan_theta_y**2))


def get_solid_angle_integration_matrix(
    integration_weights: np.ndarray,
    n_radiograph_pixels: float,
    n_rays_per_radiograph_pixel: float,
) -> sparse.csr_matrix:
    """
    Compute the integration matrix for the given ray directions and solid angle differentials.
    """
    n_rows_G = n_radiograph_pixels
    n_cols_G = n_rays_per_radiograph_pixel * n_radiograph_pixels
    nnz_total = int(n_rows_G * n_rays_per_radiograph_pixel)
    nzvals = np.zeros(nnz_total)
    colinds = np.zeros(nnz_total, dtype=int)
    rowptrs = np.zeros(n_rows_G + 1, dtype=int)

    nnz_counter = 0
    irow = 0
    # geometric_acceptance = get_acceptance(ray_directions[:, 0], ray_directions[:, 1])
    # integration_weights = (
    #     geometric_acceptance * integration_weights
    # )
    for i in range(n_rows_G):
        colinds_i = np.arange(
            i * n_rays_per_radiograph_pixel, (i + 1) * n_rays_per_radiograph_pixel
        )
        nnz_i = len(colinds_i)
        nzvals[nnz_counter : nnz_counter + nnz_i] = integration_weights[
            i * n_rays_per_radiograph_pixel : (i + 1) * n_rays_per_radiograph_pixel
        ]
        colinds[nnz_counter : nnz_counter + nnz_i] = colinds_i
        nnz_counter += nnz_i
        rowptrs[irow + 1] = nnz_counter
        irow += 1

    return sparse.csr_matrix((nzvals, colinds, rowptrs), shape=(n_rows_G, n_cols_G))


def pixel_averaging_matrix(
    n_radiograph_pixels: float,
    n_rays_per_radiograph_pixel: float,
) -> sparse.csr_matrix:
    """
    Compute the integration matrix for the given ray directions and solid angle differentials.
    """
    n_rows_G = n_radiograph_pixels
    n_cols_G = n_rays_per_radiograph_pixel * n_radiograph_pixels
    nnz_total = int(n_rows_G * n_rays_per_radiograph_pixel)
    nzvals = np.zeros(nnz_total)
    colinds = np.zeros(nnz_total, dtype=int)
    rowptrs = np.zeros(n_rows_G + 1, dtype=int)

    nnz_counter = 0
    irow = 0
    for i in range(n_rows_G):
        colinds_i = np.arange(
            i * n_rays_per_radiograph_pixel, (i + 1) * n_rays_per_radiograph_pixel
        )
        nnz_i = len(colinds_i)
        nzvals[nnz_counter : nnz_counter + nnz_i] = (
            np.ones(n_rays_per_radiograph_pixel) / n_rays_per_radiograph_pixel
        )
        colinds[nnz_counter : nnz_counter + nnz_i] = colinds_i
        nnz_counter += nnz_i
        rowptrs[irow + 1] = nnz_counter
        irow += 1

    return sparse.csr_matrix((nzvals, colinds, rowptrs), shape=(n_rows_G, n_cols_G))
