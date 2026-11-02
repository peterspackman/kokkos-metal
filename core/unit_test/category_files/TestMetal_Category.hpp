// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_TEST_METAL_HPP
#define KOKKOS_TEST_METAL_HPP

#include <gtest/gtest.h>

#define TEST_CATEGORY metal
#define TEST_CATEGORY_NUMBER 10
#define TEST_CATEGORY_DEATH metal_DeathTest
#define TEST_EXECSPACE Kokkos::Experimental::Metal
#define TEST_CATEGORY_FIXTURE(name) metal_##name

#endif
