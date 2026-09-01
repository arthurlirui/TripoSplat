#pragma once

// DLL export macro. When building triposplat as a shared library define
// TRIPOSPLAT_BUILD_DLL so TRIPOSPLAT_API marks symbols for export. Consumers
// linking against the built library get the import side automatically. When
// building/using as a static library TRIPOSPLAT_API expands to nothing.

#if defined(TRIPOSPLAT_BUILD_DLL)
  #if defined(_WIN32)
    #if defined(TRIPOSPLAT_EXPORTS)
      #define TRIPOSPLAT_API __declspec(dllexport)
    #else
      #define TRIPOSPLAT_API __declspec(dllimport)
    #endif
  #else
    #define TRIPOSPLAT_API __attribute__((visibility("default")))
  #endif
#else
  #define TRIPOSPLAT_API
#endif
