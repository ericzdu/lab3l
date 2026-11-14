"""Prepare an existing SAM 3D / other mesh export for the offline robot scene."""
import argparse
from photo_scene import prepare_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Local GLB/GLTF/OBJ/STL/mesh-PLY; NOT a photo or Gaussian splat")
    parser.add_argument("destination", help="New output folder; existing folders are never overwritten")
    parser.add_argument("--id", required=True, dest="object_id")
    parser.add_argument("--height", type=float, required=True, dest="height_m", help="Known or assumed object height in METRES")
    parser.add_argument("--up-axis", choices=("x", "y", "z"), required=True,
                        help="Original mesh up-axis; GLB is commonly y-up, but check your export")
    parser.add_argument("--position", type=float, nargs=3, default=(.55, .28, .60), metavar=("X", "Y", "Z"))
    parser.add_argument("--mass", type=float, default=.15, dest="mass_kg", help="Assumed mass in kg, not inferred from image")
    parser.add_argument("--fixed", action="store_true", help="Static fixture rather than movable object")
    parser.add_argument("--collision", choices=("box", "convex"), default="convex",
                        help="APPROXIMATION: convex fills cavities; box is even coarser")
    parser.add_argument("--provenance", choices=("external_mesh", "sam3d_export", "synthetic_test"), default="external_mesh")
    args = parser.parse_args()
    try:
        path = prepare_bundle(**vars(args))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Cannot prepare object: {exc}\n")
    print(f"Prepared bundle: {path}")
    print("This imports a mesh only. It does not run SAM 3D or infer scale, mass or friction.")
    print(f'Run: zsh ./run_photo_scene.command "{path}"')


if __name__ == "__main__":
    main()
