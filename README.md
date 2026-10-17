# wallpaper

A live wallpaper for COSMIC: points drifting across a black background, joined by white shimmering lines. One background layer surface per screen, drawn with wgpu on Vulkan once per frame callback, so each screen runs at its own refresh rate and a covered screen gets about one frame per second.

The tuning constants are at the top of `src/main.rs`.

Build, with the Wayland headers from nixpkgs:

```bash
nix shell nixpkgs#pkg-config nixpkgs#wayland.dev --command bash -c \
  'PKG_CONFIG_PATH=$(nix eval --raw nixpkgs#wayland.dev)/lib/pkgconfig cargo build --release'
```

`nix/wallpaper.nix` is the Nix package. nixcfg builds it with its own `mine.cpu.arch` in place of `.cargo/config.toml`'s `native`, and runs it as a COSMIC user service that writes the stills first.

Run, with the Vulkan loader on the library path:

```bash
LD_LIBRARY_PATH=$(nix build --no-link --print-out-paths nixpkgs#vulkan-loader)/lib ./target/release/wallpaper
```

The animation steps by when each frame reaches the screen, not by when its frame callback arrived. The compositor reports that through `wp_presentation` feedback (COSMIC does), and each frame is timed one refresh period per commit made since the last one reported shown, and never earlier than now or than the frame before it. Without `wp_presentation` it falls back to the callback's time. The buffer is the output's mode with its sides swapped for a 90° or 270° transform.

Every 5 seconds it logs to stderr, per screen: the frame rate, how many frame callbacks came more than 1.5 refresh periods after the previous one, and the time spent in a draw call and waiting for a surface texture inside it.

`--still <dir>` writes the first frame each screen shows to `<dir>/<output>.png` and exits without drawing on the screens. The COSMIC background points at these files in `~/.local/share/wallpaper/`, through `backgrounds` and `output.<name>` in `~/.config/cosmic/com.system76.CosmicBackground/v1/`, so the screen looks the same while the live wallpaper is down, and the live one starts from that exact picture. Rerun it after changing a constant:

```bash
LD_LIBRARY_PATH=$(nix build --no-link --print-out-paths nixpkgs#vulkan-loader)/lib ./target/release/wallpaper --still ~/.local/share/wallpaper
```
