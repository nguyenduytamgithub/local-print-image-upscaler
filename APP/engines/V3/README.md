# V3 HIGH engine (internal)

This is the high-quality engine behind the unified `upscale.cmd` command in
the RESIZE root. Users do not need to run anything in this directory.

V3 runs three complementary x4 models over the full image:

- Swin2SR for color and large structures;
- HAT for curves and middle frequencies;
- Real-ESRGAN for fine detail.

Their frequency bands are fused uniformly. A requested scale other than x4
is produced once from the neural x4 master with Lanczos. Temporary component
outputs are automatically removed; retained masters and manifests are routed
to `APP/masters` and `APP/manifests` by the public launcher.
