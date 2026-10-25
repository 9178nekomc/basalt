JSL native runtime layout
=========================

The Maven jar contains the prebuilt Windows x86_64 library:

    native/windows-x86_64/jsl_native.dll

Other platforms can be added without changing Java code:

    native/linux-x86_64/libjsl_native.so
    native/linux-aarch64/libjsl_native.so
    native/macos-x86_64/libjsl_native.dylib
    native/macos-aarch64/libjsl_native.dylib

Build them with ./native-build.sh on the target platform and copy the artifact
into the matching directory before `mvn package`. At runtime the loader checks
-Djsl.native.path, then the classpath resource, then java.library.path.
