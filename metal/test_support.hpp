// Minimal checks for the Metal backend tests.
#pragma once

#include <cstdio>

inline int g_failures = 0;

#define CHECK(cond)                                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::printf("  FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                   \
    }                                                                 \
  } while (0)

#define TEST(name) std::printf("%s\n", name)
