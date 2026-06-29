import numpy as np
import cupy as cp
from cupyx.scipy import sparse as cp_sparse
from cupyx.scipy.sparse.linalg import cg as gpu_cg
from scipy.sparse import lil_matrix
from time import perf_counter


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
        n=128,
        dt=0.01,
        h=0.1,
        viscosity=0.08,
        density=1.0,
        preset="shear_layer",
        pressure_tol=1e-5,
        pressure_maxiter=None,
    ):
        self.n = n
        self.dt = dt
        self.h = h
        self.viscosity = viscosity
        self.density = density
        self.preset = preset
        self.pressure_tol = pressure_tol
        self.pressure_maxiter = pressure_maxiter
        self.frame = 0

        self.cell_type = cp.ones((n, n), dtype=cp.int32)
        self.cell_type[0, :] = 0
        self.cell_type[-1, :] = 0
        self.cell_type[:, 0] = 0
        self.cell_type[:, -1] = 0
        self.cell_type_cpu = cp.asnumpy(self.cell_type)

        self.index_map_cpu = self._build_index_map(self.cell_type_cpu)
        self.index_map = cp.asarray(self.index_map_cpu)
        self.fluid_mask_cpu = self.index_map_cpu != -1
        self.fluid_mask = self.index_map != -1
        self.unknown_count = int(np.max(self.index_map_cpu) + 1)
        self.pressure_matrix = self._build_gpu_pressure_matrix()

        self.velocity_x = cp.zeros((n, n + 1), dtype=cp.float64)
        self.velocity_y = cp.zeros((n + 1, n), dtype=cp.float64)
        self.pressure = cp.zeros((n, n), dtype=cp.float64)

        self.total_accuracy_sum = 0.0
        self.accuracy_frame_count = 0

        self.reset(preset)

    def _sync_gpu(self):
        cp.cuda.Stream.null.synchronize()

    def _elapsed_ms(self, start):
        self._sync_gpu()
        return (perf_counter() - start) * 1000.0

    def _build_index_map(self, cell_type):
        index_map = -np.ones(cell_type.shape, dtype=np.int32)
        fluid_mask = (cell_type != 0) & (cell_type != 2)
        index_map[fluid_mask] = np.arange(np.count_nonzero(fluid_mask), dtype=np.int32)
        return index_map

    def _build_gpu_pressure_matrix(self):
        matrix = lil_matrix((self.unknown_count, self.unknown_count), dtype=np.float64)

        rows, columns = self.index_map_cpu.shape
        for i in range(rows):
            for j in range(columns):
                row_index = int(self.index_map_cpu[i, j])
                if row_index == -1:
                    continue

                neighbor_count = 0
                for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                    if self.cell_type_cpu[ni, nj] == 1:
                        neighbor_count += 1
                        matrix[row_index, int(self.index_map_cpu[ni, nj])] = 1.0
                    elif self.cell_type_cpu[ni, nj] == 2:
                        neighbor_count += 1

                matrix[row_index, row_index] = -float(neighbor_count)

        anchored = matrix.tolil()
        anchored[0, :] = 0.0
        anchored[:, 0] = 0.0
        anchored[0, 0] = -1.0

        # cupyx.cg expects a positive-definite system, so solve (-A) p = (-b).
        return -cp_sparse.csr_matrix(anchored.tocsr())

    def reset(self, preset=None):
        if preset is not None:
            self.preset = preset

        if self.preset == "single_vortex":
            self.initialize_single_vortex(self.velocity_x, self.velocity_y, self.n, self.h)
        elif self.preset == "double_vortex":
            self.initialize_double_vortex(self.velocity_x, self.velocity_y, self.n, self.h)
        elif self.preset == "four_vortices":
            self.initialize_four_vortices(self.velocity_x, self.velocity_y, self.n, self.h)
        elif self.preset == "shear_layer":
            self.initialize_shear_layer(self.velocity_x, self.velocity_y, 8, self.n)
        elif self.preset == "taylor_green":
            self.initialize_taylor_green(self.velocity_x, self.velocity_y, self.n, self.h)
        elif self.preset == "blob":
            self.initialize_blob(self.velocity_x, self.velocity_y, self.n)
        else:
            raise ValueError(f"Unknown preset: {self.preset}")

        self.pressure.fill(0.0)
        self.total_accuracy_sum = 0.0
        self.accuracy_frame_count = 0
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
        self._sync_gpu()
        total_start = perf_counter()

        for _ in range(substeps):
            step_start = perf_counter()
            b_vector = self._pressure_rhs()
            timings["rhs_ms"] += self._elapsed_ms(step_start)

            step_start = perf_counter()
            self.pressure = self._solve_pressure(b_vector)
            timings["pressure_ms"] += self._elapsed_ms(step_start)

            step_start = perf_counter()
            self._project_pressure()
            timings["projection_ms"] += self._elapsed_ms(step_start)

            step_start = perf_counter()
            advected_x = self.advect(
                self.velocity_x,
                self.velocity_x,
                self.velocity_y,
                "x_edge",
                self.dt,
                self.h,
            )
            advected_y = self.advect(
                self.velocity_y,
                self.velocity_x,
                self.velocity_y,
                "y_edge",
                self.dt,
                self.h,
            )
            timings["advection_ms"] += self._elapsed_ms(step_start)

            step_start = perf_counter()
            self.velocity_x = self.diffuse(advected_x, self.viscosity, self.dt, self.h)
            self.velocity_y = self.diffuse(advected_y, self.viscosity, self.dt, self.h)
            timings["diffusion_ms"] += self._elapsed_ms(step_start)

            step_start = perf_counter()
            self._enforce_walls()
            current_acc = self.l2_velocity_norm()
            self.total_accuracy_sum += current_acc
            self.accuracy_frame_count += 1
            timings["walls_ms"] += self._elapsed_ms(step_start)
            self.frame += 1

        timings["total_ms"] = self._elapsed_ms(total_start)
        if substeps > 1:
            for key in timings:
                timings[key] /= substeps

        if profile:
            return timings
        return None

    def advect(self, field, velocity_x, velocity_y, position_type, dt=0.05, h=0.1):
        rows, columns = field.shape
        i_grid, j_grid = cp.indices(field.shape)

        if position_type == "x_edge":
            u_wind = field
            i_top = cp.clip(i_grid - 1, 0, velocity_y.shape[0] - 1)
            i_bot = cp.clip(i_grid, 0, velocity_y.shape[0] - 1)
            j_left = cp.clip(j_grid - 1, 0, velocity_y.shape[1] - 1)
            j_right = cp.clip(j_grid, 0, velocity_y.shape[1] - 1)
            v_wind = (
                velocity_y[i_top, j_left]
                + velocity_y[i_bot, j_left]
                + velocity_y[i_top, j_right]
                + velocity_y[i_bot, j_right]
            ) / 4.0
        elif position_type == "y_edge":
            v_wind = field
            i_top = cp.clip(i_grid - 1, 0, velocity_x.shape[0] - 1)
            i_bot = cp.clip(i_grid, 0, velocity_x.shape[0] - 1)
            j_left = cp.clip(j_grid - 1, 0, velocity_x.shape[1] - 1)
            j_right = cp.clip(j_grid, 0, velocity_x.shape[1] - 1)
            u_wind = (
                velocity_x[i_top, j_left]
                + velocity_x[i_bot, j_left]
                + velocity_x[i_top, j_right]
                + velocity_x[i_bot, j_right]
            ) / 4.0
        else:
            raise ValueError(f"Unknown position type: {position_type}")

        i_old = cp.clip(i_grid - (v_wind * dt / h), 0.0, float(rows - 1))
        j_old = cp.clip(j_grid - (u_wind * dt / h), 0.0, float(columns - 1))

        i0 = cp.floor(i_old).astype(cp.int32)
        j0 = cp.floor(j_old).astype(cp.int32)
        i1 = cp.minimum(rows - 1, i0 + 1)
        j1 = cp.minimum(columns - 1, j0 + 1)

        beta = i_old - i0
        alpha = j_old - j0

        return (
            field[i0, j0] * (1 - alpha) * (1 - beta)
            + field[i0, j1] * alpha * (1 - beta)
            + field[i1, j0] * (1 - alpha) * beta
            + field[i1, j1] * alpha * beta
        )

    def diffuse(self, field, viscosity, dt, h):
        new_field = cp.copy(field)
        laplacian = (
            field[:-2, 1:-1]
            + field[2:, 1:-1]
            + field[1:-1, :-2]
            + field[1:-1, 2:]
            - 4.0 * field[1:-1, 1:-1]
        ) / (h**2)
        new_field[1:-1, 1:-1] = field[1:-1, 1:-1] + viscosity * dt * laplacian
        return new_field

    def _pressure_rhs(self):
        b = cp.zeros(self.unknown_count, dtype=cp.float64)

        u_left = self.velocity_x[:, :-1].copy()
        u_right = self.velocity_x[:, 1:].copy()
        v_top = self.velocity_y[:-1, :].copy()
        v_bottom = self.velocity_y[1:, :].copy()

        u_left[:, 1:] = cp.where(self.cell_type[:, :-1] == 0, 0.0, u_left[:, 1:])
        u_right[:, :-1] = cp.where(self.cell_type[:, 1:] == 0, 0.0, u_right[:, :-1])
        v_top[1:, :] = cp.where(self.cell_type[:-1, :] == 0, 0.0, v_top[1:, :])
        v_bottom[:-1, :] = cp.where(self.cell_type[1:, :] == 0, 0.0, v_bottom[:-1, :])

        divergence = (u_right - u_left) + (v_bottom - v_top)
        b[self.index_map[self.fluid_mask]] = (self.density * self.h / self.dt) * divergence[
            self.fluid_mask
        ]
        return b

    def _solve_pressure(self, b_vector):
        anchored_b = b_vector.copy()
        anchored_b[0] = 0.0

        pressure_vector, info = gpu_cg(
            self.pressure_matrix,
            -anchored_b,
            tol=self.pressure_tol,
            maxiter=self.pressure_maxiter,
        )
        if int(info) != 0:
            raise RuntimeError(f"GPU pressure solve did not converge, cg info={int(info)}")

        pressure = cp.zeros(self.cell_type.shape, dtype=cp.float64)
        pressure[self.fluid_mask] = pressure_vector[self.index_map[self.fluid_mask]]
        pressure[self.fluid_mask] -= cp.mean(pressure[self.fluid_mask])
        return pressure

    def _project_pressure(self):
        pressure_scale = self.dt / (self.density * self.h)

        x_mask = (self.cell_type[:, 1:] != 0) & (self.cell_type[:, :-1] != 0)
        x_delta = pressure_scale * (self.pressure[:, 1:] - self.pressure[:, :-1])
        self.velocity_x[:, 1:self.n] = cp.where(
            x_mask,
            self.velocity_x[:, 1:self.n] - x_delta,
            0.0,
        )

        y_mask = (self.cell_type[1:, :] != 0) & (self.cell_type[:-1, :] != 0)
        y_delta = pressure_scale * (self.pressure[1:, :] - self.pressure[:-1, :])
        self.velocity_y[1:self.n, :] = cp.where(
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
        self.add_vortex(self.velocity_x, self.velocity_y, cx, cy, radius, strength, self.h)

    def add_vortex(self, velocity_x, velocity_y, cx, cy, radius, strength, h):
        eps = 1e-8

        i_x, j_x = cp.indices(velocity_x.shape)
        dx = j_x * h - cx
        dy = (i_x + 0.5) * h - cy
        r = cp.sqrt(dx * dx + dy * dy)
        mask = r < radius
        factor = cp.zeros_like(velocity_x, dtype=cp.float64)
        factor[mask] = (1.0 - cp.exp(-(r[mask] / radius) ** 2)) / (r[mask] + eps)
        velocity_x[mask] += -strength * dy[mask] * factor[mask]

        i_y, j_y = cp.indices(velocity_y.shape)
        dx = (j_y + 0.5) * h - cx
        dy = i_y * h - cy
        r = cp.sqrt(dx * dx + dy * dy)
        mask = r < radius
        factor = cp.zeros_like(velocity_y, dtype=cp.float64)
        factor[mask] = (1.0 - cp.exp(-(r[mask] / radius) ** 2)) / (r[mask] + eps)
        velocity_y[mask] += strength * dx[mask] * factor[mask]

    def initialize_blob(self, velocity_x, velocity_y, n):
        velocity_x.fill(0.0)
        velocity_y.fill(0.0)

        center_i = n // 2
        center_j = n // 4
        radius = 5

        i_x, j_x = cp.indices(velocity_x.shape)
        velocity_x[(i_x - center_i) ** 2 + (j_x - center_j) ** 2 < radius**2] = 10.0

        i_y, j_y = cp.indices(velocity_y.shape)
        velocity_y[(i_y - center_i) ** 2 + (j_y - center_j) ** 2 < radius**2] = 10.0

    def initialize_single_vortex(self, velocity_x, velocity_y, n, h):
        velocity_x.fill(0.0)
        velocity_y.fill(0.0)
        self.add_vortex(velocity_x, velocity_y, n * h * 0.5, n * h * 0.5, 0.8, 40.0, h)

    def initialize_double_vortex(self, velocity_x, velocity_y, n, h):
        velocity_x.fill(0.0)
        velocity_y.fill(0.0)
        self.add_vortex(velocity_x, velocity_y, n * h * 0.30, n * h * 0.50, 0.6, 40.0, h)
        self.add_vortex(velocity_x, velocity_y, n * h * 0.70, n * h * 0.50, 0.6, -40.0, h)

    def initialize_four_vortices(self, velocity_x, velocity_y, n, h):
        velocity_x.fill(0.0)
        velocity_y.fill(0.0)
        vortices = [
            (0.30, 0.30, 40.0),
            (0.70, 0.30, -40.0),
            (0.30, 0.70, -40.0),
            (0.70, 0.70, 40.0),
        ]
        for px, py, strength in vortices:
            self.add_vortex(velocity_x, velocity_y, n * h * px, n * h * py, 0.45, strength, h)

    def initialize_shear_layer(self, velocity_x, velocity_y, strength, n):
        velocity_x.fill(0.0)
        velocity_y.fill(0.0)
        velocity_x[: n // 2, :] = strength
        velocity_x[n // 2 :, :] = -strength

    def initialize_taylor_green(self, velocity_x, velocity_y, n, h):
        velocity_x.fill(0.0)
        velocity_y.fill(0.0)

        u0 = 20.0
        length = n * h

        i_x, j_x = cp.indices(velocity_x.shape)
        x = j_x * h
        y = (i_x + 0.5) * h
        velocity_x[:, :] = u0 * cp.sin(2 * cp.pi * x / length) * cp.cos(2 * cp.pi * y / length)

        i_y, j_y = cp.indices(velocity_y.shape)
        x = (j_y + 0.5) * h
        y = i_y * h
        velocity_y[:, :] = -u0 * cp.cos(2 * cp.pi * x / length) * cp.sin(2 * cp.pi * y / length)

    def centered_velocity(self):
        u_center = (self.velocity_x[:, :-1] + self.velocity_x[:, 1:]) / 2.0
        v_center = (self.velocity_y[:-1, :] + self.velocity_y[1:, :]) / 2.0
        return u_center, v_center

    def speed(self):
        u_center, v_center = self.centered_velocity()
        return cp.sqrt(u_center**2 + v_center**2)

    def curl_field(self):
        curl = cp.zeros((self.n, self.n), dtype=cp.float64)
        curl[: self.n - 1, : self.n - 1] = (
            (self.velocity_y[: self.n - 1, 1:self.n] - self.velocity_y[: self.n - 1, : self.n - 1])
            / self.h
            - (
                self.velocity_x[1:self.n, : self.n - 1]
                - self.velocity_x[: self.n - 1, : self.n - 1]
            )
            / self.h
        )
        return curl

    def l2_velocity_norm(self):
        return float(cp.asnumpy(cp.sqrt(cp.mean(self.velocity_x**2) + cp.mean(self.velocity_y**2))))

    def metrics(self):
        divergence = (
            self.velocity_x[1:-1, 2:self.n]
            - self.velocity_x[1:-1, 1 : self.n - 1]
            + self.velocity_y[2:self.n, 1:-1]
            - self.velocity_y[1 : self.n - 1, 1:-1]
        )
        vorticity = self.curl_field()

        divergence_sum = float(cp.asnumpy(cp.sum(cp.abs(divergence))))
        vorticity_sum = float(cp.asnumpy(cp.sum(cp.abs(vorticity))))
        return divergence_sum, vorticity_sum

    def curl_to_rgb(self, curl, curl_scale):
        normalized = cp.clip(curl / curl_scale, -1.0, 1.0)
        positive = cp.clip(normalized, 0.0, 1.0)
        negative = cp.clip(-normalized, 0.0, 1.0)

        rgb = cp.empty((*curl.shape, 3), dtype=cp.uint8)
        rgb[..., 0] = (35 + 220 * positive).astype(cp.uint8)
        rgb[..., 1] = (45 + 150 * (1.0 - cp.abs(normalized))).astype(cp.uint8)
        rgb[..., 2] = (55 + 200 * negative).astype(cp.uint8)
        return rgb

    def speed_to_rgb(self, speed, speed_scale):
        normalized = cp.clip(speed / speed_scale, 0.0, 1.0)

        rgb = cp.empty((*speed.shape, 3), dtype=cp.uint8)
        rgb[..., 0] = (255 * cp.minimum(normalized * 2.0, 1.0)).astype(cp.uint8)
        rgb[..., 1] = (255 * cp.clip((normalized - 0.3) * 2.0, 0.0, 1.0)).astype(cp.uint8)
        rgb[..., 2] = (255 * cp.clip((normalized - 0.7) * 3.0, 0.0, 1.0)).astype(cp.uint8)
        return rgb
