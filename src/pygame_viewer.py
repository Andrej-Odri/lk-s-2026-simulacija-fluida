import argparse
import time

import numpy as np
import pygame

from realtime_simulation import FluidSimulation, PRESETS


def parse_args():
    parser = argparse.ArgumentParser(description="Fast Pygame viewer for the fluid simulation.")
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
    parser.add_argument("--substeps", type=int, default=1, help="Simulation steps per frame.")
    parser.add_argument("--size", type=int, default=768, help="Square simulation viewport size.")
    parser.add_argument(
        "--quiver-stride",
        type=int,
        default=2,
        help="Draw every Nth velocity arrow.",
    )
    parser.add_argument("--no-quiver", action="store_true", help="Hide velocity arrows.")
    parser.add_argument("--max-fps", type=int, default=0, help="0 means uncapped.")
    parser.add_argument("--frames", type=int, default=0, help="Quit after N frames. 0 means run forever.")
    parser.add_argument(
        "--metrics-every",
        type=int,
        default=5,
        help="Recompute divergence and vorticity every N frames.",
    )
    parser.add_argument(
        "--stream-strength",
        type=float,
        default=6.0,
        help="Per-frame vortex strength while a mouse button is held.",
    )
    parser.add_argument(
        "--stream-radius",
        type=float,
        default=0.45,
        help="Radius of the held mouse stream in simulation units.",
    )
    parser.add_argument(
        "--pressure-scale",
        type=float,
        default=5.0,
        help="Pressure magnitude mapped to full color intensity.",
    )
    parser.add_argument(
        "--curl-scale",
        type=float,
        default=80.0,
        help="Curl magnitude mapped to full color intensity.",
    )
    parser.add_argument(
        "--view",
        choices=["pressure", "curl"],
        default="pressure",
        help="Initial scalar field to display.",
    )
    return parser.parse_args()


def pressure_to_rgb(pressure, pressure_scale):
    normalized = np.clip(pressure / pressure_scale, -1.0, 1.0)
    positive = np.clip(normalized, 0.0, 1.0)
    negative = np.clip(-normalized, 0.0, 1.0)

    rgb = np.empty((*pressure.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (35 + 220 * positive).astype(np.uint8)
    rgb[..., 1] = (45 + 150 * (1.0 - np.abs(normalized))).astype(np.uint8)
    rgb[..., 2] = (55 + 200 * negative).astype(np.uint8)
    return rgb


def curl_to_rgb(curl, curl_scale):
    normalized = np.clip(curl / curl_scale, -1.0, 1.0)
    positive = np.clip(normalized, 0.0, 1.0)
    negative = np.clip(-normalized, 0.0, 1.0)
    magnitude = np.abs(normalized)

    rgb = np.empty((*curl.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (20 + 235 * positive).astype(np.uint8)
    rgb[..., 1] = (20 + 180 * (1.0 - magnitude)).astype(np.uint8)
    rgb[..., 2] = (20 + 235 * negative).astype(np.uint8)
    return rgb


def draw_velocity_arrows(surface, simulation, viewport_size, stride):
    u_center, v_center = simulation.centered_velocity()
    cell_size = viewport_size / simulation.n
    scale = cell_size * 0.13

    for i in range(1, simulation.n - 1, stride):
        y = (i + 0.5) * cell_size
        for j in range(1, simulation.n - 1, stride):
            x = (j + 0.5) * cell_size
            dx = float(u_center[i, j]) * scale
            dy = float(v_center[i, j]) * scale
            end = (x + dx, y + dy)
            pygame.draw.line(surface, (240, 245, 255), (x, y), end, 1)
            pygame.draw.circle(surface, (240, 245, 255), (int(end[0]), int(end[1])), 2)


def draw_text_lines(surface, font, lines, x, y, color=(235, 240, 245)):
    line_height = font.get_height() + 2
    width = max(font.size(line)[0] for line in lines) + 12
    height = len(lines) * line_height + 10
    panel = pygame.Surface((width, height), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 150))
    surface.blit(panel, (x, y))

    for index, line in enumerate(lines):
        rendered = font.render(line, True, color)
        surface.blit(rendered, (x + 6, y + 5 + index * line_height))


def main():
    args = parse_args()
    simulation = FluidSimulation(
        n=args.n,
        dt=args.dt,
        h=args.h,
        viscosity=args.viscosity,
        preset=args.preset,
    )

    pygame.init()
    pygame.display.set_caption("Real-time Fluid Simulation - Pygame")
    screen = pygame.display.set_mode((args.size, args.size))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 15)
    small_font = pygame.font.SysFont("consolas", 13)

    paused = False
    running = True
    view_mode = args.view
    fps = 0.0
    timings = {
        "events_ms": 0.0,
        "stream_ms": 0.0,
        "metrics_ms": 0.0,
        "sim_total_ms": 0.0,
        "rgb_ms": 0.0,
        "scale_ms": 0.0,
        "quiver_ms": 0.0,
        "text_ms": 0.0,
        "flip_ms": 0.0,
        "frame_ms": 0.0,
    }
    last_frame_time = time.perf_counter()
    last_step_timings = {
        "rhs_ms": 0.0,
        "pressure_ms": 0.0,
        "projection_ms": 0.0,
        "advection_ms": 0.0,
        "diffusion_ms": 0.0,
        "walls_ms": 0.0,
        "total_ms": 0.0,
    }
    metrics_every = max(1, args.metrics_every)
    metric_values = {
        "divergence": 0.0,
        "curl": 0.0,
        "vorticity": 0.0,
        "divergence_avg": 0.0,
        "curl_avg": 0.0,
        "vorticity_avg": 0.0,
        "cfl": 0.0,
        "max_speed": 0.0,
        "kinetic_energy": 0.0,
    }
    metric_cell_count = max(1, int(np.count_nonzero(simulation.fluid_mask)))

    while running:
        frame_start = time.perf_counter()

        event_start = time.perf_counter()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_c:
                    view_mode = "curl" if view_mode == "pressure" else "pressure"
                elif event.key == pygame.K_r:
                    simulation.reset()
                else:
                    key_name = pygame.key.name(event.key)
                    if key_name in PRESETS:
                        simulation.reset(PRESETS[key_name])
        timings["events_ms"] = (time.perf_counter() - event_start) * 1000.0

        stream_start = time.perf_counter()
        stream_strength = 0.0
        mouse_buttons = pygame.mouse.get_pressed(num_buttons=3)
        if mouse_buttons[0] or mouse_buttons[2]:
            x, y = pygame.mouse.get_pos()
            if 0 <= x < args.size and 0 <= y < args.size:
                stream_strength = args.stream_strength
                if mouse_buttons[2]:
                    stream_strength *= -1.0
                simulation.add_vortex_at_cell(
                    x / args.size * simulation.n,
                    y / args.size * simulation.n,
                    radius=args.stream_radius,
                    strength=stream_strength,
                )
        timings["stream_ms"] = (time.perf_counter() - stream_start) * 1000.0

        if not paused:
            last_step_timings = simulation.step(args.substeps, profile=True)
            timings["sim_total_ms"] = last_step_timings["total_ms"]

        if simulation.frame % metrics_every == 0:
            metrics_start = time.perf_counter()
            current_metrics = simulation.accuracy_metrics()
            metric_values["divergence"] = current_metrics["divergence"]
            metric_values["curl"] = current_metrics["curl"]
            metric_values["vorticity"] = current_metrics["vorticity"]
            metric_values["divergence_avg"] = current_metrics["divergence"] / metric_cell_count
            metric_values["curl_avg"] = current_metrics["curl"] / metric_cell_count
            metric_values["vorticity_avg"] = current_metrics["vorticity"] / metric_cell_count
            metric_values["cfl"] = current_metrics["cfl"]
            metric_values["max_speed"] = current_metrics["max_speed"]
            metric_values["kinetic_energy"] = current_metrics["kinetic_energy"]
            timings["metrics_ms"] = (time.perf_counter() - metrics_start) * 1000.0

        rgb_start = time.perf_counter()
        if view_mode == "curl":
            rgb = curl_to_rgb(simulation.curl_field(), args.curl_scale)
        else:
            rgb = pressure_to_rgb(simulation.pressure, args.pressure_scale)
        timings["rgb_ms"] = (time.perf_counter() - rgb_start) * 1000.0

        scale_start = time.perf_counter()
        field_surface = pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))
        field_surface = pygame.transform.scale(field_surface, (args.size, args.size))
        screen.blit(field_surface, (0, 0))
        timings["scale_ms"] = (time.perf_counter() - scale_start) * 1000.0

        quiver_start = time.perf_counter()
        if not args.no_quiver:
            draw_velocity_arrows(screen, simulation, args.size, max(1, args.quiver_stride))
        if stream_strength != 0.0:
            color = (255, 230, 120) if stream_strength > 0.0 else (125, 210, 255)
            pygame.draw.circle(screen, color, pygame.mouse.get_pos(), 12, 2)
            pygame.draw.circle(screen, color, pygame.mouse.get_pos(), 3)
        timings["quiver_ms"] = (time.perf_counter() - quiver_start) * 1000.0

        now = time.perf_counter()
        elapsed = now - last_frame_time
        if elapsed > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed)
        last_frame_time = now

        text_start = time.perf_counter()
        status = "paused" if paused else "running"
        lines = [
            f"{simulation.preset} | {view_mode} | frame {simulation.frame} | {status} | fps {fps:5.1f}",
            "simulation timing (ms)",
            f"rhs        {last_step_timings['rhs_ms']:6.2f}",
            f"pressure   {last_step_timings['pressure_ms']:6.2f}",
            f"project    {last_step_timings['projection_ms']:6.2f}",
            f"advect     {last_step_timings['advection_ms']:6.2f}",
            f"diffuse    {last_step_timings['diffusion_ms']:6.2f}",
            f"walls      {last_step_timings['walls_ms']:6.2f}",
            f"sim total  {timings['sim_total_ms']:6.2f}",
            "accuracy metrics",
            f"div total  {metric_values['divergence']:9.3f}",
            f"div avg    {metric_values['divergence_avg']:9.5f}",
            f"curl total {metric_values['curl']:9.3f}",
            f"curl avg   {metric_values['curl_avg']:9.5f}",
            f"vort total {metric_values['vorticity']:9.3f}",
            f"vort avg   {metric_values['vorticity_avg']:9.5f}",
            f"cfl        {metric_values['cfl']:9.5f}",
            f"max speed  {metric_values['max_speed']:9.3f}",
            f"kin energy {metric_values['kinetic_energy']:9.3f}",
            "render timing (ms)",
            f"events     {timings['events_ms']:6.2f}",
            f"stream     {timings['stream_ms']:6.2f}",
            f"metrics    {timings['metrics_ms']:6.2f}",
            f"rgb        {timings['rgb_ms']:6.2f}",
            f"scale      {timings['scale_ms']:6.2f}",
            f"quiver     {timings['quiver_ms']:6.2f}",
            f"text       {timings['text_ms']:6.2f}",
            f"flip       {timings['flip_ms']:6.2f}",
            f"frame      {timings['frame_ms']:6.2f}",
        ]
        draw_text_lines(screen, font, lines, 10, 10)
        help_lines = ["space pause | c pressure/curl | r reset | 1-6 presets | hold L/R stream | esc quit"]
        draw_text_lines(screen, small_font, help_lines, 10, args.size - 32)
        timings["text_ms"] = (time.perf_counter() - text_start) * 1000.0

        flip_start = time.perf_counter()
        pygame.display.flip()
        timings["flip_ms"] = (time.perf_counter() - flip_start) * 1000.0
        timings["frame_ms"] = (time.perf_counter() - frame_start) * 1000.0

        if args.max_fps > 0:
            clock.tick(args.max_fps)

        if args.frames > 0 and simulation.frame >= args.frames:
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
