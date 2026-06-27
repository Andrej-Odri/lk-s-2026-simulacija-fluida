import numpy as np
from time import perf_counter
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import factorized

from essential_functions import (
    IndexMap,
    Univerzalna_Advekcija,
    Univerzalna_Difuzija,
    add_vortex,
    initialize_blob,
    initialize_double_vortex,
    initialize_four_vortices,
    initialize_shear_layer,
    initialize_single_vortex,
    initialize_taylor_green,
)


PRESETS = {
    "1": "single_vortex",
    "2": "double_vortex",
    "3": "four_vortices",
    "4": "shear_layer",
    "5": "taylor_green",
    "6": "blob",
}


class FluidSimulation:
    def __init__(
        self,
        n=32,
        dt=0.01,
        h=0.1,
        viscosity=0.08,
        density=1.0,
        preset="shear_layer",
    ):
        self.n = n
        self.dt = dt
        self.h = h
        self.viscosity = viscosity
        self.density = density
        self.preset = preset
        self.frame = 0

        self.cell_type = np.ones((n, n), dtype=int)
        self.cell_type[0, :] = 0
        self.cell_type[-1, :] = 0
        self.cell_type[:, 0] = 0
        self.cell_type[:, -1] = 0

        self.index_map = IndexMap(self.cell_type)
        self.system_matrix = self._build_sparse_system_matrix()
        self.pressure_solver = factorized(self._build_anchored_pressure_matrix())
        self.fluid_mask = self.index_map != -1

        self.velocity_x = np.zeros((n, n + 1))
        self.velocity_y = np.zeros((n + 1, n))
        self.pressure = np.zeros((n, n))
        self.reset(preset)

    def reset(self, preset=None):
        if preset is not None:
            self.preset = preset

        if self.preset == "single_vortex":
            initialize_single_vortex(self.velocity_x, self.velocity_y, self.n, self.h)
        elif self.preset == "double_vortex":
            initialize_double_vortex(self.velocity_x, self.velocity_y, self.n, self.h)
        elif self.preset == "four_vortices":
            initialize_four_vortices(self.velocity_x, self.velocity_y, self.n, self.h)
        elif self.preset == "shear_layer":
            initialize_shear_layer(self.velocity_x, self.velocity_y, 8, self.n)
        elif self.preset == "taylor_green":
            initialize_taylor_green(self.velocity_x, self.velocity_y, self.n, self.h)
        elif self.preset == "blob":
            initialize_blob(self.velocity_x, self.velocity_y, self.n)
        else:
            raise ValueError(f"Unknown preset: {self.preset}")

        self.pressure.fill(0.0)
        self.frame = 0

    def step(self, substeps=1, profile=False):
        timings = {
            "rhs_ms": 0.0,
            "pressure_ms": 0.0,
            "projection_ms": 0.0,
            "advection_ms": 0.0,
            "diffusion_ms": 0.0,
            "walls_ms": 0.0,
            "total_ms": 0.0,
        }
        total_start = perf_counter()

        for _ in range(substeps):
            step_start = perf_counter()
            b_vector = self._pressure_rhs()
            timings["rhs_ms"] += (perf_counter() - step_start) * 1000.0

            step_start = perf_counter()
            self.pressure = self._solve_pressure(b_vector)
            timings["pressure_ms"] += (perf_counter() - step_start) * 1000.0

            step_start = perf_counter()
            self._project_pressure()
            timings["projection_ms"] += (perf_counter() - step_start) * 1000.0

            step_start = perf_counter()
            advected_x = Univerzalna_Advekcija(
                self.velocity_x,
                self.velocity_x,
                self.velocity_y,
                "x_ivica",
                self.dt,
                self.h,
            )
            advected_y = Univerzalna_Advekcija(
                self.velocity_y,
                self.velocity_x,
                self.velocity_y,
                "y_ivica",
                self.dt,
                self.h,
            )
            timings["advection_ms"] += (perf_counter() - step_start) * 1000.0

            step_start = perf_counter()
            self.velocity_x = Univerzalna_Difuzija(
                advected_x,
                self.viscosity,
                self.dt,
                self.h,
            )
            self.velocity_y = Univerzalna_Difuzija(
                advected_y,
                self.viscosity,
                self.dt,
                self.h,
            )
            timings["diffusion_ms"] += (perf_counter() - step_start) * 1000.0

            step_start = perf_counter()
            self._enforce_walls()
            timings["walls_ms"] += (perf_counter() - step_start) * 1000.0
            self.frame += 1

        timings["total_ms"] = (perf_counter() - total_start) * 1000.0
        if substeps > 1:
            for key in timings:
                timings[key] /= substeps

        if profile:
            return timings
        return None

    def _build_sparse_system_matrix(self):
        unknown_count = int(np.max(self.index_map) + 1)
        matrix = lil_matrix((unknown_count, unknown_count), dtype=float)

        rows, columns = self.index_map.shape
        for i in range(rows):
            for j in range(columns):
                row_index = int(self.index_map[i, j])
                if row_index == -1:
                    continue

                neighbor_count = 0
                for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                    if self.cell_type[ni, nj] == 1:
                        neighbor_count += 1
                        matrix[row_index, int(self.index_map[ni, nj])] = 1.0
                    elif self.cell_type[ni, nj] == 2:
                        neighbor_count += 1

                matrix[row_index, row_index] = -neighbor_count

        return matrix.tocsr()

    def _build_anchored_pressure_matrix(self):
        anchored = self.system_matrix.tolil(copy=True)
        anchored[0, :] = 0.0
        anchored[0, 0] = 1.0
        return anchored.tocsc()

    def _solve_pressure(self, b_vector):
        anchored_b = b_vector.copy()
        anchored_b[0] = 0.0
        pressure_vector = self.pressure_solver(anchored_b)

        pressure = np.zeros(self.cell_type.shape)
        pressure[self.fluid_mask] = pressure_vector[self.index_map[self.fluid_mask]]
        pressure[self.fluid_mask] -= np.mean(pressure[self.fluid_mask])
        return pressure

    def _pressure_rhs(self):
        unknown_count = int(np.max(self.index_map) + 1)
        b_vector = np.zeros(unknown_count)

        u_left = self.velocity_x[:, :-1].copy()
        u_right = self.velocity_x[:, 1:].copy()
        v_top = self.velocity_y[:-1, :].copy()
        v_bottom = self.velocity_y[1:, :].copy()

        u_left[:, 1:] = np.where(self.cell_type[:, :-1] == 0, 0.0, u_left[:, 1:])
        u_right[:, :-1] = np.where(self.cell_type[:, 1:] == 0, 0.0, u_right[:, :-1])
        v_top[1:, :] = np.where(self.cell_type[:-1, :] == 0, 0.0, v_top[1:, :])
        v_bottom[:-1, :] = np.where(self.cell_type[1:, :] == 0, 0.0, v_bottom[:-1, :])

        divergence = (u_right - u_left) + (v_bottom - v_top)
        b_vector[self.index_map[self.fluid_mask]] = (
            self.density * self.h / self.dt
        ) * divergence[self.fluid_mask]

        return b_vector

    def _project_pressure(self):
        pressure_scale = self.dt / (self.density * self.h)

        x_mask = (self.cell_type[:, 1:] != 0) & (self.cell_type[:, :-1] != 0)
        x_delta = pressure_scale * (self.pressure[:, 1:] - self.pressure[:, :-1])
        self.velocity_x[:, 1:self.n] = np.where(
            x_mask,
            self.velocity_x[:, 1:self.n] - x_delta,
            0.0,
        )

        y_mask = (self.cell_type[1:, :] != 0) & (self.cell_type[:-1, :] != 0)
        y_delta = pressure_scale * (self.pressure[1:, :] - self.pressure[:-1, :])
        self.velocity_y[1:self.n, :] = np.where(
            y_mask,
            self.velocity_y[1:self.n, :] - y_delta,
            0.0,
        )

    def _enforce_walls(self):
        self.velocity_x[:, 0] = 0.0
        self.velocity_x[:, -1] = 0.0
        self.velocity_x[0, :] = 0.0
        self.velocity_x[-1, :] = 0.0

        self.velocity_y[0, :] = 0.0
        self.velocity_y[-1, :] = 0.0
        self.velocity_y[:, 0] = 0.0
        self.velocity_y[:, -1] = 0.0

    def add_vortex_at_cell(self, x_cell, y_cell, radius=0.5, strength=24.0):
        cx = float(np.clip(x_cell, 1, self.n - 2)) * self.h
        cy = float(np.clip(y_cell, 1, self.n - 2)) * self.h
        add_vortex(self.velocity_x, self.velocity_y, cx, cy, radius, strength, self.h)

    def centered_velocity(self):
        u_center = (self.velocity_x[:, :-1] + self.velocity_x[:, 1:]) / 2.0
        v_center = (self.velocity_y[:-1, :] + self.velocity_y[1:, :]) / 2.0
        return u_center, v_center

    def speed(self):
        u_center, v_center = self.centered_velocity()
        return np.sqrt(u_center**2 + v_center**2)

    def metrics(self):
        values = self.accuracy_metrics()
        return values["divergence"], values["vorticity"]

    def accuracy_metrics(self):
        divergence = (
            self.velocity_x[1:-1, 2:self.n]
            - self.velocity_x[1:-1, 1 : self.n - 1]
            + self.velocity_y[2:self.n, 1:-1]
            - self.velocity_y[1 : self.n - 1, 1:-1]
        )

        vorticity = (
            (self.velocity_y[: self.n - 1, 1:self.n] - self.velocity_y[: self.n - 1, : self.n - 1])
            / self.h
            - (
                self.velocity_x[1:self.n, : self.n - 1]
                - self.velocity_x[: self.n - 1, : self.n - 1]
            )
            / self.h
        )

        u_center, v_center = self.centered_velocity()
        speed_squared = u_center**2 + v_center**2
        fluid_speed_squared = speed_squared[self.fluid_mask]
        max_speed = float(np.sqrt(np.max(fluid_speed_squared))) if fluid_speed_squared.size else 0.0
        kinetic_energy = 0.5 * self.density * float(np.sum(fluid_speed_squared)) * (self.h**2)

        return {
            "divergence": float(np.sum(np.abs(divergence))),
            "vorticity": float(np.sum(np.abs(vorticity))),
            "cfl": max_speed * self.dt / self.h,
            "max_speed": max_speed,
            "kinetic_energy": kinetic_energy,
        }
