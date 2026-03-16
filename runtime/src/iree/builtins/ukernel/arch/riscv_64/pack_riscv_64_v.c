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
