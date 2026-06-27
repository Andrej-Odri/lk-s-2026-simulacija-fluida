import argparse
import csv
from pathlib import Path

from realtime_simulation import FluidSimulation, PRESETS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record how fluid simulation presets evolve over time."
    )
    parser.add_argument("--frames", type=int, default=300, help="Frames to record per preset.")
    parser.add_argument("--n", type=int, default=32, help="Grid size.")
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation time step.")
    parser.add_argument("--h", type=float, default=0.1, help="Cell size.")
    parser.add_argument("--viscosity", type=float, default=0.08, help="Fluid viscosity.")
    parser.add_argument("--density", type=float, default=1.0, help="Fluid density.")
    parser.add_argument("--substeps", type=int, default=1, help="Simulation steps per row.")
    parser.add_argument(
        "--output-dir",
        default="recordings",
        help="Directory for per-preset CSV files and summary.csv.",
    )
    parser.add_argument(
        "--preset",
        action="append",
        choices=sorted(set(PRESETS.values())),
        help="Preset to record. Repeat for multiple presets. Defaults to all presets.",
    )
    return parser.parse_args()


def unique_presets():
    seen = set()
    presets = []
    for preset in PRESETS.values():
        if preset not in seen:
            presets.append(preset)
            seen.add(preset)
    return presets


def record_preset(preset, args, output_dir):
    simulation = FluidSimulation(
        n=args.n,
        dt=args.dt,
        h=args.h,
        viscosity=args.viscosity,
        density=args.density,
        preset=preset,
    )

    csv_path = output_dir / f"{preset}.csv"
    fieldnames = [
        "frame",
        "divergence",
        "curl",
        "vorticity",
        "cfl",
        "max_speed",
        "kinetic_energy",
        "rhs_ms",
        "pressure_ms",
        "projection_ms",
        "advection_ms",
        "diffusion_ms",
        "walls_ms",
        "total_ms",
    ]

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for _ in range(args.frames):
            timings = simulation.step(args.substeps, profile=True)
            metrics = simulation.accuracy_metrics()
            writer.writerow(
                {
                    "frame": simulation.frame,
                    **metrics,
                    **timings,
                }
            )

    return {
        "preset": preset,
        "frames": args.frames,
        "final_frame": simulation.frame,
        **simulation.accuracy_metrics(),
        "csv_path": str(csv_path),
    }


def write_summary(summary_rows, output_dir):
    summary_path = output_dir / "summary.csv"
    fieldnames = [
        "preset",
        "frames",
        "final_frame",
        "divergence",
        "curl",
        "vorticity",
        "cfl",
        "max_speed",
        "kinetic_energy",
        "csv_path",
    ]

    with summary_path.open("w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    return summary_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    presets = args.preset if args.preset else unique_presets()
    summary_rows = []
    for preset in presets:
        print(f"Recording {preset}...")
        summary_rows.append(record_preset(preset, args, output_dir))

    summary_path = write_summary(summary_rows, output_dir)
    print(f"Done. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
