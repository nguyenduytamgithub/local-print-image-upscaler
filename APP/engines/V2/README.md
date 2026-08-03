# V2 FAST engine (internal)

This folder contains the portable Real-ESRGAN NCNN/Vulkan engine used by the
single public command at the RESIZE root. It is intentionally kept inside
`APP`; users should run `upscale.cmd`, not files in this folder directly.

Pipeline: one Real-ESRGAN x4plus GPU pass, then one Lanczos resize when the
requested final scale is not x4. Model and third-party licenses stay beside
the engine for redistribution compliance.
