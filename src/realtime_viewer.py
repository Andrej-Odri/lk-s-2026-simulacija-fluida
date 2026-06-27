import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from realtime_simulation import FluidSimulation, PRESETS


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time 2D fluid simulation viewer.")
    parser.add_argument("--n", type=int, default=32, help="Grid size.")
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation time step.")
    parser.add_argument("--h", type=float, default=0.1, help="Cell size.")
    parser.add_argument("--viscosity", type=float, default=0.08, help="Fluid viscosity.")
    parser.add_argument(
        "--preset",
        default="shear_layer",
        choices=sorted(set(PRESETS.values())),
        help="Initial velocity field.",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=1,
        help="Simulation steps per rendered frame.",
    )
    parser.add_argument(
        "--draw-every",
        type=int,
        default=1,
        help="Render every N simulation ticks.",
    )
    parser.add_argument(
        "--quiver-stride",
        type=int,
        default=2,
        help="Draw every Nth velocity arrow. Lower is prettier, higher is faster.",
    )
    parser.add_argument(
        "--no-quiver",
        action="store_true",
        help="Hide velocity arrows for the fastest pressure-field display.",
    )
    parser.add_argument(
        "--metrics-every",
        type=int,
        default=10,
        help="Recompute divergence/vorticity every N rendered frames.",
    )
    parser.add_argument(
        "--blocking-draw",
        action="store_true",
        help="Use canvas.draw() so render timing measures the real Matplotlib draw cost.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    simulation = FluidSimulation(
        n=args.n,
        dt=args.dt,
        h=args.h,
        viscosity=args.viscosity,
        preset=args.preset,
    )

    quiver_stride = max(1, args.quiver_stride)
    draw_every = max(1, args.draw_every)
    metrics_every = max(1, args.metrics_every)
    x_grid, y_grid = np.meshgrid(
        np.arange(0, args.n, quiver_stride),
        np.arange(0, args.n, quiver_stride),
    )
    paused = {"value": False}
    last_time = {"value": time.perf_counter(), "fps": 0.0}
    counters = {"ticks": 0, "draws": 0}
    cached_metrics = {"divergence": 0.0, "vorticity": 0.0}
    timings = {
        "rhs_ms": 0.0,
        "pressure_ms": 0.0,
        "projection_ms": 0.0,
        "advection_ms": 0.0,
        "diffusion_ms": 0.0,
        "walls_ms": 0.0,
        "sim_total_ms": 0.0,
        "artists_ms": 0.0,
        "metrics_ms": 0.0,
        "draw_ms": 0.0,
        "frame_total_ms": 0.0,
    }

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title("Real-time Fluid Simulation")

    pressure_image = ax.imshow(
        simulation.pressure,
        cmap="turbo",
        origin="upper",
        extent=[-0.5, args.n - 0.5, args.n - 0.5, -0.5],
        vmin=-5.0,
        vmax=5.0,
    )

    u_center, v_center = simulation.centered_velocity()
    velocity_quiver = None
    if not args.no_quiver:
        speed = np.sqrt(u_center**2 + v_center**2)
        velocity_quiver = ax.quiver(
            x_grid,
            y_grid,
            u_center[::quiver_stride, ::quiver_stride],
            -v_center[::quiver_stride, ::quiver_stride],
            speed[::quiver_stride, ::quiver_stride],
            cmap="magma",
            scale=120,
            width=0.003,
        )

    title = ax.set_title("")
    timing_text = ax.text(
        0.015,
        0.985,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 5},
    )
    ax.set_xlim(-0.5, args.n - 0.5)
    ax.set_ylim(args.n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(pressure_image, ax=ax, label="Pressure")

    def redraw():
        frame_start = time.perf_counter()
        artists_start = time.perf_counter()
        pressure_image.set_array(simulation.pressure)
        if velocity_quiver is not None:
            u_center, v_center = simulation.centered_velocity()
            speed = np.sqrt(u_center**2 + v_center**2)
            velocity_quiver.set_UVC(
                u_center[::quiver_stride, ::quiver_stride],
                -v_center[::quiver_stride, ::quiver_stride],
                speed[::quiver_stride, ::quiver_stride],
            )
        timings["artists_ms"] = (time.perf_counter() - artists_start) * 1000.0

        now = time.perf_counter()
        elapsed = now - last_time["value"]
        if elapsed > 0:
            last_time["fps"] = 0.9 * last_time["fps"] + 0.1 * (1.0 / elapsed)
        last_time["value"] = now

        if counters["draws"] % metrics_every == 0:
            metrics_start = time.perf_counter()
            divergence, vorticity = simulation.metrics()
            cached_metrics["divergence"] = divergence
            cached_metrics["vorticity"] = vorticity
            timings["metrics_ms"] = (time.perf_counter() - metrics_start) * 1000.0

        status = "paused" if paused["value"] else "running"
        title.set_text(
            f"{simulation.preset} | frame {simulation.frame} | {status} | "
            f"fps {last_time['fps']:.1f} | div {cached_metrics['divergence']:.3f} | "
            f"vort {cached_metrics['vorticity']:.1f}"
        )
        timing_text.set_text(
            "update timing (ms)\n"
            f"rhs        {timings['rhs_ms']:6.2f}\n"
            f"pressure   {timings['pressure_ms']:6.2f}\n"
            f"project    {timings['projection_ms']:6.2f}\n"
            f"advect     {timings['advection_ms']:6.2f}\n"
            f"diffuse    {timings['diffusion_ms']:6.2f}\n"
            f"walls      {timings['walls_ms']:6.2f}\n"
            f"sim total  {timings['sim_total_ms']:6.2f}\n"
            f"artists    {timings['artists_ms']:6.2f}\n"
            f"metrics    {timings['metrics_ms']:6.2f}\n"
            f"draw       {timings['draw_ms']:6.2f}\n"
            f"frame      {timings['frame_total_ms']:6.2f}"
        )
        counters["draws"] += 1
        draw_start = time.perf_counter()
        if args.blocking_draw:
            fig.canvas.draw()
        else:
            fig.canvas.draw_idle()
        timings["draw_ms"] = (time.perf_counter() - draw_start) * 1000.0
        timings["frame_total_ms"] = (time.perf_counter() - frame_start) * 1000.0

    def on_key(event):
        if event.key == " ":
            paused["value"] = not paused["value"]
        elif event.key == "r":
            simulation.reset()
        elif event.key in PRESETS:
            simulation.reset(PRESETS[event.key])
        elif event.key == "escape":
            plt.close(fig)
        redraw()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        strength = -28.0 if event.button == 3 else 28.0
        simulation.add_vortex_at_cell(event.xdata, event.ydata, strength=strength)
        redraw()

    def on_timer():
        if plt.fignum_exists(fig.number):
            if not paused["value"]:
                step_timings = simulation.step(args.substeps, profile=True)
                timings["rhs_ms"] = step_timings["rhs_ms"]
                timings["pressure_ms"] = step_timings["pressure_ms"]
                timings["projection_ms"] = step_timings["projection_ms"]
                timings["advection_ms"] = step_timings["advection_ms"]
                timings["diffusion_ms"] = step_timings["diffusion_ms"]
                timings["walls_ms"] = step_timings["walls_ms"]
                timings["sim_total_ms"] = step_timings["total_ms"]
                counters["ticks"] += 1
                if counters["ticks"] % draw_every == 0:
                    redraw()
            timer.start()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)

    help_text = (
        "space: pause | r: reset | 1-6: presets | left/right click: add vortex | esc: close"
    )
    fig.text(0.5, 0.02, help_text, ha="center", va="bottom", fontsize=9)

    redraw()
    timer = fig.canvas.new_timer(interval=1)
    timer.add_callback(on_timer)
    timer.start()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
