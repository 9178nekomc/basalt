JSL Native Classpath Layout
===========================
Native shared libraries are bundled under:
  /native/windows-x86_64/jsl_native.dll
  /native/linux-x86_64/libjsl_native.so
  /native/linux-aarch64/libjsl_native.so
  /native/macos-x86_64/libjsl_native.dylib
  /native/macos-aarch64/libjsl_native.dylib

At startup, jsl.core.NativeKernels automatically detects os.name and os.arch,
extracts the matching binary to a temporary directory, and loads it via System.load.
Run ./native-build.sh locally or trigger .github/workflows/ci-release.yml to build
all platform targets.
