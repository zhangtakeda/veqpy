from __future__ import annotations

import numpy as np
from contourpy import contour_generator
from matplotlib.path import Path
from scipy.interpolate import RectBivariateSpline

from veqpy.model.equilibrium import (
    _interpolate_psin_to_rectilinear_grid,
    _ray_polygon_radii,
)


def _distance_to_closed_polygon(points: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    distance = np.full(points.shape[0], np.inf, dtype=np.float64)
    for start, end in zip(boundary, np.roll(boundary, -1, axis=0), strict=True):
        segment = end - start
        fraction = np.clip(
            ((points - start) @ segment) / float(np.dot(segment, segment)),
            0.0,
            1.0,
        )
        closest = start + fraction[:, None] * segment
        distance = np.minimum(distance, np.linalg.norm(points - closest, axis=1))
    return distance


def _outward_polygon_normals(boundary: np.ndarray, center: np.ndarray) -> np.ndarray:
    tangent = np.roll(boundary, -1, axis=0) - np.roll(boundary, 1, axis=0)
    normal = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    inward = np.sum(normal * (boundary - center), axis=1) < 0.0
    normal[inward] *= -1.0
    return normal


def test_chease_exterior_linearly_continues_the_last_flux_interval() -> None:
    R = np.linspace(2.0, 10.0, 65)
    Z = np.linspace(-5.0, 6.0, 69)
    R_grid, Z_grid = np.meshgrid(R, Z, indexing="ij")

    center = np.array([6.0, 0.4])
    theta = np.linspace(0.0, 2.0 * np.pi, 257, endpoint=False)
    boundary_R = center[0] + 2.2 * np.cos(theta)
    boundary_Z = center[1] + 3.5 * np.sin(theta)
    boundary = np.column_stack((boundary_R, boundary_Z))
    inside = Path(boundary, closed=True).contains_points(
        np.column_stack((R_grid.reshape(-1), Z_grid.reshape(-1)))
    ).reshape(R_grid.shape)

    rho = np.linspace(0.0, 1.0, 17)
    R_surfaces = center[0] + rho[:, None] * (boundary_R[None, :] - center[0])
    Z_surfaces = center[1] + rho[:, None] * (boundary_Z[None, :] - center[1])
    psin = np.square(rho)
    psi_axis = -2.0
    normalized_grids = []
    for psi_scale in (3.0, -3.0):
        psi_grid = _interpolate_psin_to_rectilinear_grid(
            R_surfaces,
            Z_surfaces,
            psin,
            psin,
            R_nodes=R,
            Z_nodes=Z,
            psi_axis=psi_axis,
            psi_scale=psi_scale,
            psi_outside=None,
        )
        normalized_grids.append((psi_grid - psi_axis) / psi_scale)

    # Reversing alpha2 reverses only physical value ordering, not normalized
    # flux surfaces or the CHEASE exterior continuation.
    np.testing.assert_allclose(normalized_grids[1], normalized_grids[0], atol=1.0e-14)
    exterior_psin = normalized_grids[0]
    assert np.all(exterior_psin[~inside] > 1.0)

    outside_points = np.column_stack((R_grid[~inside], Z_grid[~inside]))
    offsets = outside_points - center
    point_radii = np.linalg.norm(offsets, axis=1)
    directions = offsets / point_radii[:, None]
    boundary_radii = _ray_polygon_radii(center, directions, boundary)
    reference_radii = _ray_polygon_radii(
        center,
        directions,
        np.column_stack((R_surfaces[-2], Z_surfaces[-2])),
    )
    sigma = point_radii / boundary_radii
    reference_sigma = reference_radii / boundary_radii
    expected = 1.0 + (1.0 - psin[-2]) / (1.0 - reference_sigma) * (sigma - 1.0)
    np.testing.assert_allclose(exterior_psin[~inside], expected, rtol=2.0e-12, atol=2.0e-12)

    # psi_bound is the unique LCFS branch.  Nearby exterior levels are allowed
    # to be closed, but remain ordered, nested continuations of the given LCFS.
    generator = contour_generator(x=R, y=Z, z=exterior_psin.T, name="serial")
    cell_diagonal = float(np.hypot(R[1] - R[0], Z[1] - Z[0]))
    maximum_offsets = []
    for level in (1.0, 1.02, 1.05):
        contours = generator.lines(level)
        enclosing = [
            contour
            for contour in contours
            if contour.shape[0] >= 3 and Path(contour, closed=True).contains_point(center)
        ]
        assert len(enclosing) == 1
        contour = enclosing[0]
        assert np.linalg.norm(contour[0] - contour[-1]) <= cell_diagonal
        maximum_offsets.append(float(np.max(_distance_to_closed_polygon(contour, boundary))))
    assert maximum_offsets[0] <= cell_diagonal
    assert maximum_offsets[0] < maximum_offsets[1] < maximum_offsets[2]

    # Approximate ONETWO's modified CNTOUR update: Newton steps are taken along
    # grad(psi) from both sides of the boundary.  The linearly continued edge
    # gradient keeps both starts in the LCFS branch's local basin of attraction.
    spline = RectBivariateSpline(R, Z, exterior_psin, kx=3, ky=3, s=0.0)
    normals = _outward_polygon_normals(boundary, center)
    sampled_boundary = boundary[::4]
    sampled_normals = normals[::4]
    for offset in (-cell_diagonal, cell_diagonal):
        points = sampled_boundary + offset * sampled_normals
        for _ in range(20):
            value = spline.ev(points[:, 0], points[:, 1]) - 1.0
            derivative_R = spline.ev(points[:, 0], points[:, 1], dx=1)
            derivative_Z = spline.ev(points[:, 0], points[:, 1], dy=1)
            gradient_squared = np.square(derivative_R) + np.square(derivative_Z)
            assert np.all(gradient_squared > 1.0e-20)
            step = np.column_stack(
                (
                    value * derivative_R / gradient_squared,
                    value * derivative_Z / gradient_squared,
                )
            )
            step_length = np.linalg.norm(step, axis=1)
            step *= np.minimum(
                1.0,
                cell_diagonal / np.maximum(step_length, 1.0e-30),
            )[:, None]
            points -= step

        residual = np.abs(spline.ev(points[:, 0], points[:, 1]) - 1.0)
        distance = _distance_to_closed_polygon(points, boundary)
        assert np.max(residual) < 1.0e-10
        assert np.max(distance) <= cell_diagonal
