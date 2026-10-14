# TODO

- Package it as a Nix package and run it as a COSMIC user service from nixcfg. The service should write the stills with `--still` before starting, so the COSMIC background never lags behind a constant change. Waits until the user approves the look. The package passes `-C target-cpu=${config.mine.cpu.arch}` instead of `.cargo/config.toml`'s `native`, see `~/Projects/nixcfg/modules/nixos/cpu.nix`. Enabling it needs a `nixos-rebuild switch`, so read nixcfg's `AGENTS.md` first.
- Rotated screens: the buffer size comes from the output's mode, which is not swapped for a 90° transform.
- Presentation time: frames are timed by `Instant::now()` at the frame callback, not by when they reach the screen.
