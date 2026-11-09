// MSVC names the AMD FidelityFX FSR 2 sources use (runtime/third_party/fsr2), for other compilers: __declspec, which
// clang accepts only with -fdeclspec, and the _countof and wcscpy_s of the Microsoft C runtime. Included by
// vk_fsr2.cpp before those sources, so the third-party files stay as AMD ships them.
#pragma once
#if !defined(_MSC_VER) && defined(__cplusplus)
#include <cstddef>
#include <cwchar>

#ifndef __declspec
#define __declspec(x)
#endif
#ifndef _countof
#define _countof(a) (sizeof(a) / sizeof((a)[0]))
#endif

template <std::size_t N> static inline int wcscpy_s(wchar_t (&dst)[N], const wchar_t *src) {
    std::wcsncpy(dst, src, N);
    dst[N - 1] = 0;
    return 0;
}
static inline int wcscpy_s(wchar_t *dst, std::size_t n, const wchar_t *src) {
    std::wcsncpy(dst, src, n);
    if (n) dst[n - 1] = 0;
    return 0;
}
#endif
