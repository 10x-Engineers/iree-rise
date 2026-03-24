// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <riscv_vector.h>

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/pack_riscv_64_internal.h"

void iree_uk_pack_tile_Xx1_x8_riscv_64_transpose(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 1);

  const iree_uk_int8_t* IREE_UK_RESTRICT in_tile_ptr_i8 = in_tile_ptr;
  iree_uk_int8_t* IREE_UK_RESTRICT out_tile_i8_ptr = out_tile_ptr;
  size_t vl = tile_size1;

  for (; outer_size1 > 0; --outer_size1) {
    __riscv_vse8_v_i8m1(out_tile_i8_ptr,
                        __riscv_vle8_v_i8m1(in_tile_ptr_i8, vl), vl);
    out_tile_i8_ptr += out_stride1;
    in_tile_ptr_i8 += tile_size1;
  }
}

void iree_uk_pack_tile_7xX_x8_riscv_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 7);

  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;
  size_t vl = tile_size1;

  for (; outer_size1 > 0; --outer_size1) {
    vint8m1_t row0 = __riscv_vle8_v_i8m1(in_ptr + 0 * in_stride0, vl);
    vint8m1_t row1 = __riscv_vle8_v_i8m1(in_ptr + 1 * in_stride0, vl);
    vint8m1_t row2 = __riscv_vle8_v_i8m1(in_ptr + 2 * in_stride0, vl);
    vint8m1_t row3 = __riscv_vle8_v_i8m1(in_ptr + 3 * in_stride0, vl);
    vint8m1_t row4 = __riscv_vle8_v_i8m1(in_ptr + 4 * in_stride0, vl);
    vint8m1_t row5 = __riscv_vle8_v_i8m1(in_ptr + 5 * in_stride0, vl);
    vint8m1_t row6 = __riscv_vle8_v_i8m1(in_ptr + 6 * in_stride0, vl);

    __riscv_vse8_v_i8m1(out_ptr + 0 * vl, row0, vl);
    __riscv_vse8_v_i8m1(out_ptr + 1 * vl, row1, vl);
    __riscv_vse8_v_i8m1(out_ptr + 2 * vl, row2, vl);
    __riscv_vse8_v_i8m1(out_ptr + 3 * vl, row3, vl);
    __riscv_vse8_v_i8m1(out_ptr + 4 * vl, row4, vl);
    __riscv_vse8_v_i8m1(out_ptr + 5 * vl, row5, vl);
    __riscv_vse8_v_i8m1(out_ptr + 6 * vl, row6, vl);

    out_ptr += out_stride1;
    in_ptr += tile_size1;
  }
}
