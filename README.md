# wallpaper

A live wallpaper for COSMIC: points drifting across a black background, joined by white shimmering lines. One background layer surface per screen, drawn with wgpu on Vulkan once per frame callback, so each screen runs at its own refresh rate and a covered screen gets about one frame per second.

The tuning constants are at the top of `src/main.rs`.

Build, with the Wayland headers from nixpkgs:

```bash
nix shell nixpkgs#pkg-config nixpkgs#wayland.dev --command bash -c \
  'PKG_CONFIG_PATH=$(nix eval --raw nixpkgs#wayland.dev)/lib/pkgconfig cargo build --release'
```

Run, with the Vulkan loader on the library path:

```bash
LD_LIBRARY_PATH=$(nix build --no-link --print-out-paths nixpkgs#vulkan-loader)/lib ./target/release/wallpaper
```

Every 5 seconds it logs each screen's frame rate to stderr.
