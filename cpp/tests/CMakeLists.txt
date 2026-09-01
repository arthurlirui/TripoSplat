# Test executables for triposplat_cpp.
# Enabled via CMake option -DTRIPOSPLAT_BUILD_TESTS=ON.
#
# These are lightweight smoke tests that do NOT require the TorchScript weights
# (they exercise the pure-C++ utilities: quaternion transforms in GaussianIO,
# the PLY/SPLAT round-trip on a synthetic Gaussian). Numerical parity tests
# against the Python pipeline live in tests/python/ and are run separately.

add_executable(test_gaussian_io test_gaussian_io.cpp)
target_link_libraries(test_gaussian_io PRIVATE triposplat)
add_test(NAME gaussian_io COMMAND test_gaussian_io)
