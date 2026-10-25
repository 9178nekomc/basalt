Basalt Native Classpath Layout
===========================
Native shared libraries are bundled under:
  /native/windows-x86_64/basalt_native.dll
  /native/linux-x86_64/libbasalt_native.so
  /native/linux-aarch64/libbasalt_native.so
  /native/macos-x86_64/libbasalt_native.dylib
  /native/macos-aarch64/libbasalt_native.dylib

At startup, basalt.core.NativeKernels automatically detects os.name and os.arch,
extracts the matching binary to a temporary directory, and loads it via System.load.
Run ./native-build.sh locally or trigger .github/workflows/ci-release.yml to build
all platform targets.
