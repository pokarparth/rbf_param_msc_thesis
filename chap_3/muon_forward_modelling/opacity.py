import discretize
import numba
import numpy as np
from scipy import sparse
from typing import Dict, Any

from . import MuonSensor


def get_opacity_matrix_no_integration(
    mesh: discretize.TensorMesh, sensors: Dict[Any, MuonSensor]
) -> sparse.csr_matrix:
    """
    Compute the forward operator G for the given
    mesh and sensors.

    Constructs G as a CSR sparse matrix directly.
    """
    nx, ny, nz = mesh.shape_cells
    nrows_G = sum([len(sensor.rgrid_x) * len(sensor.rgrid_y) for sensor in sensors.values()])
    nnz_est = int(nrows_G * np.sqrt(nx * ny * nz))
    nzvals = np.zeros(nnz_est)
    colinds = np.zeros(nnz_est, dtype=int)
    rowptrs = np.zeros(nrows_G + 1, dtype=int)

    xgrid = mesh.x0[0] + np.r_[0.0, np.cumsum(mesh.h[0])]
    ygrid = mesh.x0[1] + np.r_[0.0, np.cumsum(mesh.h[1])]
    zgrid = mesh.x0[2] + np.r_[0.0, np.cumsum(mesh.h[2])]

    nnz = 0
    irow = 0
    for sensor in sensors.values():
        # Loop over the raypaths for this sensor
        ro = sensor.loc
        for ix in range(len(sensor.rgrid_x)):
            for iy in range(len(sensor.rgrid_y)):

                # Get ray direction
                rd = np.array([sensor.rgrid_x[ix], sensor.rgrid_y[iy], 1.0])
                rd = rd / np.linalg.norm(rd)

                # Get ray lengths through each cell in the full mesh
                # Cell edges in each dimension
                _, lvals = get_ray_intersection_pts(ro, rd, xgrid, ygrid, zgrid)

                # Build up G as a CSR sparse matrix directly
                lvals = lvals.flatten(order="F")  # Match SimPEG cell ordering
                colinds_i = np.argwhere(lvals > 0.0).flatten()
                nnz_i = len(colinds_i)
                if nnz + nnz_i > nnz_est:
                    nnz_est = int(1.5 * nnz_est)
                    nzvals = np.hstack((nzvals, np.zeros(nnz_est - nnz)))
                    colinds = np.hstack((colinds, np.zeros(nnz_est - nnz, dtype=int)))
                nzvals[nnz : nnz + nnz_i] = lvals[colinds_i]
                colinds[nnz : nnz + nnz_i] = colinds_i
                nnz += nnz_i
                rowptrs[irow + 1] = nnz
                irow += 1
    return sparse.csr_matrix((nzvals, colinds, rowptrs), shape=(nrows_G, nx * ny * nz))


def get_opacity_matrix_monte_carlo_integration(
    mesh: discretize.TensorMesh,
    sensors: Dict[Any, MuonSensor],
    n_rays_per_pixel: int,
    use_precomputed_ray_directions: bool = False,
    ray_directions: np.ndarray = None,
) -> sparse.csr_matrix:
    """
    Compute the forward operator G for the given
    mesh and sensors.

    Constructs G as a CSR sparse matrix directly.
    """
    nx, ny, nz = mesh.shape_cells
    n_pixels = sum([(len(sensor.rgrid_x)-1) * (len(sensor.rgrid_y)-1) for sensor in sensors.values()])
    n_rows_G = n_pixels * n_rays_per_pixel
    nnz_est = int(n_rows_G * np.sqrt(nx * ny * nz))
    nzvals = np.zeros(nnz_est)
    colinds = np.zeros(nnz_est, dtype=int)
    rowptrs = np.zeros(n_rows_G + 1, dtype=int)

    xgrid = mesh.x0[0] + np.r_[0.0, np.cumsum(mesh.h[0])]
    ygrid = mesh.x0[1] + np.r_[0.0, np.cumsum(mesh.h[1])]
    zgrid = mesh.x0[2] + np.r_[0.0, np.cumsum(mesh.h[2])]

    nnz = 0
    irow = 0
    if not use_precomputed_ray_directions:
        ray_directions = np.zeros((n_rows_G, 2))
    solid_angle_intgrtn_wgts = np.zeros(n_rows_G)
    for sensor in sensors.values():
        # Loop over the raypaths for this sensor
        ro = sensor.loc
        for pixel in sensor.pixels():
            intgrl_normalization_factor = pixel.get_volume() / n_rays_per_pixel
            for _ in range(n_rays_per_pixel):

                # Get ray direction
                if use_precomputed_ray_directions:
                    tan_theta_x, tan_theta_y = ray_directions[irow]
                else:
                    tan_theta_x, tan_theta_y = pixel.get_random_ray()
                    ray_directions[irow] = [tan_theta_x, tan_theta_y]
                rd = np.array([tan_theta_x, tan_theta_y, 1.0])
                rd = rd / np.linalg.norm(rd)
                solid_angle_intgrtn_wgts[irow] = pixel.solid_angle_differential(
                    tan_theta_x, tan_theta_y
                ) * intgrl_normalization_factor

                # Get ray lengths through each cell in the full mesh
                # Cell edges in each dimension

                _, lvals = get_ray_intersection_pts(ro, rd, xgrid, ygrid, zgrid)

                # Build up G as a CSR sparse matrix directly
                lvals = lvals.flatten(order="F")  # Match SimPEG cell ordering
                colinds_i = np.argwhere(lvals > 0.0).flatten()
                nnz_i = len(colinds_i)
                if nnz + nnz_i > nnz_est:
                    nnz_est = int(1.5 * nnz_est)
                    nzvals = np.hstack((nzvals, np.zeros(nnz_est - nnz)))
                    colinds = np.hstack((colinds, np.zeros(nnz_est - nnz, dtype=int)))
                nzvals[nnz : nnz + nnz_i] = lvals[colinds_i]
                colinds[nnz : nnz + nnz_i] = colinds_i
                nnz += nnz_i
                rowptrs[irow + 1] = nnz
                irow += 1
    G = sparse.csr_matrix((nzvals[:nnz], colinds[:nnz], rowptrs), shape=(n_rows_G, nx * ny * nz))
    return G, ray_directions, solid_angle_intgrtn_wgts


@numba.njit(error_model="numpy")
def get_ray_intersection_pts(
    ro: np.ndarray, rd: np.ndarray, xgrid: np.ndarray, ygrid: np.ndarray, zgrid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a ray and a grid, compute the intersection points of the ray with each
    cell in the grid and the length of the ray's path through each cell. Return
    numpy arrays holding the intersection points for each cell and the path lengths.
    """
    nx = len(xgrid) - 1
    ny = len(ygrid) - 1
    nz = len(zgrid) - 1
    intersections = np.full((nx, ny, nz, 2, 3), np.nan)
    lvals = np.zeros((nx, ny, nz))

    # Compute intersection of ray with domain boundaries
    # and from that which z slices of the mesh we have
    # to check for intersection with the ray.
    tminx, tmaxx = plane_col_intersections(ro, rd, 0, xgrid[0], xgrid[-1])
    tminy, tmaxy = plane_col_intersections(ro, rd, 1, ygrid[0], ygrid[-1])
    tminz, tmaxz = plane_col_intersections(ro, rd, 2, zgrid[0], zgrid[-1])
    tmax = min(tmaxx, tmaxy, tmaxz)
    z1 = ro[2]
    z2 = ro[2] + tmax * rd[2]
    izmin = max(np.searchsorted(zgrid, np.minimum(z1, z2)) - 1, 0)
    izmax = min(np.searchsorted(zgrid, np.maximum(z1, z2)), nz - 1)

    for iz in range(izmin, izmax + 1):  # Loop over columns
        # Get intersections with zmin and zmax planes
        tminz, tmaxz = plane_col_intersections(ro, rd, 2, zgrid[iz], zgrid[iz + 1])
        tminz = max(tminz, 0.0)

        # Bound the search for cells that we need to check for intersection
        # with ray.
        x1 = ro[0] + tminz * rd[0]
        x2 = ro[0] + tmaxz * rd[0]
        ixmin = max(np.searchsorted(xgrid, np.minimum(x1, x2)) - 1, 0)
        ixmax = min(np.searchsorted(xgrid, np.maximum(x1, x2)), nx - 1)
        y1 = ro[1] + tminz * rd[1]
        y2 = ro[1] + tmaxz * rd[1]
        iymin = max(np.searchsorted(ygrid, np.minimum(y1, y2)) - 1, 0)
        iymax = min(np.searchsorted(ygrid, np.maximum(y1, y2)), ny - 1)

        # Loop over block of cells
        for ix in range(ixmin, ixmax + 1):
            for iy in range(iymin, iymax + 1):
                # Compute intersections with these cells
                x1 = xgrid[ix]
                x2 = xgrid[ix + 1]
                y1 = ygrid[iy]
                y2 = ygrid[iy + 1]
                tminx, tmaxx = plane_col_intersections(ro, rd, 0, x1, x2)
                tminy, tmaxy = plane_col_intersections(ro, rd, 1, y1, y2)
                tmin = max(tminx, tminy, tminz)
                tmax = min(tmaxx, tmaxy, tmaxz)
                if (tmax >= tmin) and (tmax >= 0.0):
                    # intersected_cols[ix, iy, iz] = True
                    tmin = tmin if tmin >= 0.0 else 0.0
                    intersections[ix, iy, iz, 0, :] = ro + tmin * rd
                    intersections[ix, iy, iz, 1, :] = ro + tmax * rd
                    lvals[ix, iy, iz] = np.abs(tmax - tmin)
    return intersections, lvals


@numba.njit(error_model="numpy")
def plane_col_intersections(ro, rd, idim, box_dmin, box_dmax) -> tuple[float, float]:
    """
    Compute the intersection of a ray with axis-aligned planes.
    """
    # is ray parallel to planes?
    vd = rd[idim]  # np.dot(rd, nrm_box)
    tmin = -np.inf
    tmax = np.inf
    t0 = np.divide(box_dmin - ro[idim], vd)
    t1 = np.divide(box_dmax - ro[idim], vd)
    tmin = np.maximum(tmin, np.minimum(t0, t1))
    tmax = np.minimum(tmax, np.maximum(t0, t1))
    return tmin, tmax
