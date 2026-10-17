# The wallpaper as a Nix package, built by nixcfg against its own nixpkgs with
# its mine.cpu.arch, the way nixcfg builds Dikt.
#
# cpuArch replaces .cargo/config.toml's target-cpu=native, which stays out of
# src. Under nix, new hardware changes no input, so a "native" build would
# never be rebuilt for it (nixcfg's modules/nixos/cpu.nix).
{
  lib,
  rustPlatform,
  pkg-config,
  wayland,
  vulkan-loader,
  autoAddDriverRunpath,
  cpuArch ? null,
}:

let
  root = ../.;
in
rustPlatform.buildRustPackage {
  pname = "wallpaper";
  inherit ((lib.importTOML ../Cargo.toml).package) version;

  src = lib.fileset.toSource {
    inherit root;
    fileset = lib.fileset.unions [
      ../Cargo.toml
      ../Cargo.lock
      ../src
    ];
  };

  cargoLock.lockFile = ../Cargo.lock;

  nativeBuildInputs = [
    pkg-config
    # /run/opengl-driver/lib in the RUNPATH. The NVIDIA Vulkan driver itself
    # comes through the loader's ICD file, which names it by absolute path.
    autoAddDriverRunpath
  ];
  # wayland-backend's client_system feature links libwayland-client.
  buildInputs = [ wayland ];

  # [profile.release] in Cargo.toml already has opt-level 3, fat LTO and one
  # codegen unit, and buildRustPackage builds that profile.
  env = lib.optionalAttrs (cpuArch != null) {
    RUSTFLAGS = "-C target-cpu=${cpuArch}";
  };

  # ash, under wgpu, loads libvulkan.so.1 with dlopen, so the linker never
  # sees it. postFixup runs after fixup shrinks the RUNPATH to the NEEDED
  # libraries.
  postFixup = ''
    patchelf --add-rpath ${lib.makeLibraryPath [ vulkan-loader ]} $out/bin/wallpaper
  '';

  meta = {
    description = "Live wallpaper for COSMIC: drifting points joined by shimmering lines";
    mainProgram = "wallpaper";
    platforms = [ "x86_64-linux" ];
  };
}
