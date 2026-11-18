# Third-party 3D models

The `ycb_*` folders are derived from the **YCB Object and Model Set**
(Yale-CMU-Berkeley), licensed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

> B. Calli, A. Walsman, A. Singh, S. Srinivasa, P. Abbeel and A. M. Dollar,
> "Benchmarking in Manipulation Research: The YCB Object and Model Set and
> Benchmarking Protocols," IEEE Robotics and Automation Magazine, 2015.
> Models: https://www.ycbbenchmarks.com (data mirror:
> https://registry.opendata.aws/ycb-benchmarks)

| Folder | Original model | Changes |
|---|---|---|
| `ycb_apple` | `013_apple` (google_16k) | Converted to OBJ, texture downscaled to 1024 px, convex decomposition added |
| `ycb_plate_14cm` | `029_plate` (google_16k) | Uniformly scaled 26 cm → 14 cm, texture downscaled, convex decomposition added |
| `ycb_block_5cm` | `070-b_colored_wood_blocks` (google_16k) | Rotated 45° about vertical, uniformly scaled to a 50 mm cube, texture downscaled to 512 px, box collision |
| `ycb_block_7cm` | `070-b_colored_wood_blocks` (google_16k) | As above, scaled to a 70 mm cube |

Each folder's `object.json` records the source URL and the sha256 of the
original file. These are placeholders until meshes of the team's own objects
replace them.
