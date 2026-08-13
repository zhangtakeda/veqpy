from __future__ import annotations

import heapq

import numpy as np
from contourpy import contour_generator
from matplotlib.path import Path

from veqpy.model.equilibrium import _interpolate_psin_to_rectilinear_grid


def _point_is_on_box(point: np.ndarray, R: np.ndarray, Z: np.ndarray) -> bool:
    tolerance = 1.0e-10 * max(float(np.ptp(R)), float(np.ptp(Z)))
    return bool(
        abs(point[0] - R[0]) <= tolerance
        or abs(point[0] - R[-1]) <= tolerance
        or abs(point[1] - Z[0]) <= tolerance
        or abs(point[1] - Z[-1]) <= tolerance
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


def _minimum_sublevel_barrier(values: np.ndarray, inside: np.ndarray) -> float:
    """Return the lowest level joining the LCFS neighborhood to the box."""

    outside = ~inside
    adjacent_to_lcfs = np.zeros(inside.shape, dtype=bool)
    adjacent_to_lcfs[1:, :] |= inside[:-1, :]
    adjacent_to_lcfs[:-1, :] |= inside[1:, :]
    adjacent_to_lcfs[:, 1:] |= inside[:, :-1]
    adjacent_to_lcfs[:, :-1] |= inside[:, 1:]
    sources = outside & adjacent_to_lcfs

    box = np.zeros(inside.shape, dtype=bool)
    box[0, :] = True
    box[-1, :] = True
    box[:, 0] = True
    box[:, -1] = True
    targets = outside & box

    cost = np.full(values.shape, np.inf, dtype=np.float64)
    queue: list[tuple[float, int, int]] = []
    for i, j in np.argwhere(sources):
        cost[i, j] = float(values[i, j])
        heapq.heappush(queue, (cost[i, j], int(i), int(j)))

    while queue:
        current, i, j = heapq.heappop(queue)
        if current != cost[i, j]:
            continue
        if targets[i, j]:
            return current
        for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if not (0 <= ni < values.shape[0] and 0 <= nj < values.shape[1]):
                continue
            if not outside[ni, nj]:
                continue
            candidate = max(current, float(values[ni, nj]))
            if candidate < cost[ni, nj]:
                cost[ni, nj] = candidate
                heapq.heappush(queue, (candidate, ni, nj))

    raise AssertionError("No exterior path joins the LCFS neighborhood to the box")


def test_vacuum_exterior_confines_closed_contours_to_lcfs_collar() -> None:
    R = np.linspace(2.0, 10.0, 49)
    Z = np.linspace(-5.0, 6.0, 53)
    R_grid, Z_grid = np.meshgrid(R, Z, indexing="ij")

    theta = np.linspace(0.0, 2.0 * np.pi, 129, endpoint=False)
    boundary_R = (
        6.0
        + 2.2 * (1.0 + 0.16 * np.cos(theta)) * np.cos(theta)
        + 0.18 * np.cos(2.0 * theta)
    )
    boundary_Z = (
        0.4
        + 3.5 * (1.0 - 0.08 * np.sin(theta)) * np.sin(theta)
        + 0.25 * np.sin(2.0 * theta)
    )
    boundary = np.column_stack((boundary_R, boundary_Z))
    inside = Path(boundary, closed=True).contains_points(
        np.column_stack((R_grid.reshape(-1), Z_grid.reshape(-1)))
    ).reshape(R_grid.shape)

    rho = np.linspace(0.0, 1.0, 17)
    R_surfaces = 6.0 + rho[:, None] * (boundary_R[None, :] - 6.0)
    Z_surfaces = 0.4 + rho[:, None] * (boundary_Z[None, :] - 0.4)
    psi_axis = -2.0
    normalized_grids = []
    for psi_scale in (3.0, -3.0):
        psi_grid = _interpolate_psin_to_rectilinear_grid(
            R_surfaces,
            Z_surfaces,
            np.square(rho),
            np.square(rho),
            R_nodes=R,
            Z_nodes=Z,
            psi_axis=psi_axis,
            psi_scale=psi_scale,
            psi_outside=None,
        )
        normalized_grids.append((psi_grid - psi_axis) / psi_scale)

    # Reversing the physical flux span reverses only its value ordering.  The
    # normalized field, all contours, and the exterior topology remain fixed.
    np.testing.assert_allclose(normalized_grids[1], normalized_grids[0], rtol=1.0e-14, atol=1.0e-14)
    exterior_psin = normalized_grids[0]

    outside_values = exterior_psin[~inside]
    assert np.all(outside_values > 1.0)
    assert 0.09 < np.ptp(outside_values) < 0.11

    # A path whose normalized increment never exceeds ``barrier`` joins the
    # LCFS neighborhood to the box.  Any LCFS-surrounding contour above it must
    # cross that path and is therefore impossible.  This is the discrete
    # certificate for the far exterior, independent of visual contour sampling.
    barrier = _minimum_sublevel_barrier(exterior_psin - 1.0, inside)
    assert 0.0 < barrier < 1.0e-6

    adjacent_to_lcfs = np.zeros(inside.shape, dtype=bool)
    adjacent_to_lcfs[1:, :] |= inside[:-1, :]
    adjacent_to_lcfs[:-1, :] |= inside[1:, :]
    adjacent_to_lcfs[:, 1:] |= inside[:, :-1]
    adjacent_to_lcfs[:, :-1] |= inside[:, 1:]
    bulk = ~inside & ~adjacent_to_lcfs
    bulk[0, :] = False
    bulk[-1, :] = False
    bulk[:, 0] = False
    bulk[:, -1] = False
    dR = float(R[1] - R[0])
    dZ = float(Z[1] - Z[0])
    residual = np.zeros(exterior_psin.shape, dtype=np.float64)
    residual[1:-1, 1:-1] = (
        (exterior_psin[2:, 1:-1] - 2.0 * exterior_psin[1:-1, 1:-1] + exterior_psin[:-2, 1:-1])
        / dR**2
        - (exterior_psin[2:, 1:-1] - exterior_psin[:-2, 1:-1])
        / (2.0 * R_grid[1:-1, 1:-1] * dR)
        + (exterior_psin[1:-1, 2:] - 2.0 * exterior_psin[1:-1, 1:-1] + exterior_psin[1:-1, :-2])
        / dZ**2
    )
    assert np.max(np.abs(residual[bulk])) < 2.0e-12

    # Contour topology can change only when a level crosses a grid-node value.
    # Check one level in every open interval between exterior critical values.
    # The maximum principle excludes contractible closed islands.  Check every
    # discrete topology interval and require the remaining LCFS-surrounding
    # contours to stay in a narrow grid-scale collar.
    critical_values = np.unique(outside_values[outside_values > 1.0])
    interval_levels = critical_values[:-1] + 0.5 * np.diff(critical_values)
    interval_levels = interval_levels[
        (interval_levels > critical_values[:-1])
        & (interval_levels < critical_values[1:])
    ]
    levels = np.concatenate(([1.0, np.nextafter(1.0, np.inf)], interval_levels))
    generator = contour_generator(x=R, y=Z, z=exterior_psin.T, name="serial")

    contour_count = 0
    closed_count = 0
    open_count = 0
    cell_diagonal = float(np.hypot(R[1] - R[0], Z[1] - Z[0]))
    for level in levels:
        for contour in generator.lines(float(level)):
            if contour.shape[0] < 2:
                continue
            contour_count += 1
            open_on_box = _point_is_on_box(contour[0], R, Z) and _point_is_on_box(
                contour[-1], R, Z
            )
            if open_on_box:
                open_count += 1
            else:
                closed_count += 1
                assert level <= 1.0 + barrier
                assert np.max(_distance_to_closed_polygon(contour, boundary)) <= 4.0 * cell_diagonal
    assert contour_count > 0
    assert closed_count > 0
    assert open_count > 0

    lcfs_contours = generator.lines(1.0)
    assert len(lcfs_contours) == 1
    assert not _point_is_on_box(lcfs_contours[0][0], R, Z)
    assert not _point_is_on_box(lcfs_contours[0][-1], R, Z)
    assert Path(lcfs_contours[0], closed=True).contains_point((6.0, 0.4))
