// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <riscv_vector.h>

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/pack_riscv_64_internal.h"

static inline void iree_uk_riscv_64_copy_x32_contiguous(
    iree_uk_uint32_t* IREE_UK_RESTRICT out,
    const iree_uk_uint32_t* IREE_UK_RESTRICT in, iree_uk_index_t count) {
  iree_uk_index_t i = 0;
  while (i < count) {
    size_t vl = __riscv_vsetvl_e32m1((size_t)(count - i));
    vuint32m1_t v = __riscv_vle32_v_u32m1(in + i, vl);
    __riscv_vse32_v_u32m1(out + i, v, vl);
    i += (iree_uk_index_t)vl;
  }
}

static inline void iree_uk_riscv_64_copy_x16_contiguous(
    iree_uk_uint16_t* IREE_UK_RESTRICT out,
    const iree_uk_uint16_t* IREE_UK_RESTRICT in, iree_uk_index_t count) {
  iree_uk_index_t i = 0;
  while (i < count) {
    size_t vl = __riscv_vsetvl_e16m1((size_t)(count - i));
    vuint16m1_t v = __riscv_vle16_v_u16m1(in + i, vl);
    __riscv_vse16_v_u16m1(out + i, v, vl);
    i += (iree_uk_index_t)vl;
  }
}

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

void iree_uk_pack_tile_7x1_x8_riscv_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 7);
  IREE_UK_ASSERT(tile_size1 == 1);

  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;

  for (; outer_size1 > 0; --outer_size1) {
    vint8m1_t v = __riscv_vlse8_v_i8m1(in_ptr, in_stride0, 7);
    __riscv_vse8_v_i8m1(out_ptr, v, 7);
    out_ptr += out_stride1;
    in_ptr += 1;
  }
}

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

void iree_uk_pack_tile_7x1_x8_riscv_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 1);
  IREE_UK_ASSERT(tile_size0 == 7);
  IREE_UK_ASSERT(tile_size1 == 1);

  iree_uk_int8_t* IREE_UK_RESTRICT out_ptr = out_tile_ptr;
  const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr = in_tile_ptr;

  for (; outer_size1 > 0; --outer_size1) {
    vint8m1_t v = __riscv_vlse8_v_i8m1(in_ptr, in_stride0, 7);
    __riscv_vse8_v_i8m1(out_ptr, v, 7);
    out_ptr += out_stride1;
    in_ptr += 1;
  }
}

void iree_uk_pack_tile_7x1_x16_riscv_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 2);
  IREE_UK_ASSERT(tile_size0 == 7);
  IREE_UK_ASSERT(tile_size1 == 1);

  const iree_uk_uint16_t* IREE_UK_RESTRICT in_tile_u16_ptr = in_tile_ptr;
  iree_uk_uint16_t* IREE_UK_RESTRICT out_tile_u16_ptr = out_tile_ptr;
  ptrdiff_t in_stride0_bytes =
      (ptrdiff_t)(in_stride0 * (iree_uk_index_t)sizeof(*in_tile_u16_ptr));
  size_t vl = (size_t)tile_size0;

  for (; outer_size1 > 0; --outer_size1) {
    vuint16m1_t v =
        __riscv_vlse16_v_u16m1(in_tile_u16_ptr, in_stride0_bytes, vl);
    __riscv_vse16_v_u16m1(out_tile_u16_ptr, v, vl);
    out_tile_u16_ptr += out_stride1;
    in_tile_u16_ptr += tile_size1;
  }
}

void iree_uk_pack_tile_7x32_x16_riscv_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 2);
  IREE_UK_ASSERT(tile_size0 == 7);
  IREE_UK_ASSERT(tile_size1 == 32);

  const iree_uk_uint16_t* IREE_UK_RESTRICT in_tile_u16_ptr = in_tile_ptr;
  iree_uk_uint16_t* IREE_UK_RESTRICT out_tile_u16_ptr = out_tile_ptr;

  for (; outer_size1 > 0; --outer_size1) {
    for (iree_uk_index_t i0 = 0; i0 < tile_size0; ++i0) {
      const iree_uk_uint16_t* IREE_UK_RESTRICT src_row =
          in_tile_u16_ptr + i0 * in_stride0;
      iree_uk_uint16_t* IREE_UK_RESTRICT dst_row =
          out_tile_u16_ptr + i0 * tile_size1;
      iree_uk_riscv_64_copy_x16_contiguous(dst_row, src_row, tile_size1);
    }
    out_tile_u16_ptr += out_stride1;
    in_tile_u16_ptr += tile_size1;
  }
}

void iree_uk_pack_tile_32x1_x16_riscv_64_transpose(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  (void)in_stride0;
  IREE_UK_ASSERT(elem_size == 2);
  IREE_UK_ASSERT(tile_size0 == 1);
  IREE_UK_ASSERT(tile_size1 == 32);

  const iree_uk_uint16_t* IREE_UK_RESTRICT in_tile_u16_ptr = in_tile_ptr;
  iree_uk_uint16_t* IREE_UK_RESTRICT out_tile_u16_ptr = out_tile_ptr;

  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_riscv_64_copy_x16_contiguous(out_tile_u16_ptr, in_tile_u16_ptr,
                                         tile_size1);
    out_tile_u16_ptr += out_stride1;
    in_tile_u16_ptr += tile_size1;
  }
}

void iree_uk_pack_tile_7x1_x32_riscv_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 4);
  IREE_UK_ASSERT(tile_size0 == 7);
  IREE_UK_ASSERT(tile_size1 == 1);

  const iree_uk_uint32_t* IREE_UK_RESTRICT in_tile_u32_ptr = in_tile_ptr;
  iree_uk_uint32_t* IREE_UK_RESTRICT out_tile_u32_ptr = out_tile_ptr;
  ptrdiff_t in_stride0_bytes = (ptrdiff_t)(in_stride0 * (iree_uk_index_t)sizeof(*in_tile_u32_ptr));
  size_t vl = (size_t)tile_size0;

  for (; outer_size1 > 0; --outer_size1) {
    vuint32m1_t v = __riscv_vlse32_v_u32m1(in_tile_u32_ptr, in_stride0_bytes, vl);
    __riscv_vse32_v_u32m1(out_tile_u32_ptr, v, vl);
    out_tile_u32_ptr += out_stride1;
    in_tile_u32_ptr += tile_size1;
  }
}

void iree_uk_pack_tile_7x32_x32_riscv_64_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  IREE_UK_ASSERT(elem_size == 4);
  IREE_UK_ASSERT(tile_size0 == 7);
  IREE_UK_ASSERT(tile_size1 == 32);

  const iree_uk_uint32_t* IREE_UK_RESTRICT in_tile_u32_ptr = in_tile_ptr;
  iree_uk_uint32_t* IREE_UK_RESTRICT out_tile_u32_ptr = out_tile_ptr;

  for (; outer_size1 > 0; --outer_size1) {
    for (iree_uk_index_t i0 = 0; i0 < tile_size0; ++i0) {
      const iree_uk_uint32_t* IREE_UK_RESTRICT src_row = in_tile_u32_ptr + i0 * in_stride0;
      iree_uk_uint32_t* IREE_UK_RESTRICT dst_row = out_tile_u32_ptr + i0 * tile_size1;
      iree_uk_riscv_64_copy_x32_contiguous(dst_row, src_row, tile_size1);
    }
    out_tile_u32_ptr += out_stride1;
    in_tile_u32_ptr += tile_size1;
  }
}

void iree_uk_pack_tile_32x1_x32_riscv_64_transpose(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, iree_uk_index_t outer_size1,
    iree_uk_index_t out_stride1, iree_uk_index_t in_stride0,
    iree_uk_index_t elem_size, iree_uk_index_t tile_size0,
    iree_uk_index_t tile_size1) {
  (void)in_stride0;
  IREE_UK_ASSERT(elem_size == 4);
  IREE_UK_ASSERT(tile_size0 == 1);
  IREE_UK_ASSERT(tile_size1 == 32);

  const iree_uk_uint32_t* IREE_UK_RESTRICT in_tile_u32_ptr = in_tile_ptr;
  iree_uk_uint32_t* IREE_UK_RESTRICT out_tile_u32_ptr = out_tile_ptr;

  for (; outer_size1 > 0; --outer_size1) {
    iree_uk_riscv_64_copy_x32_contiguous(out_tile_u32_ptr, in_tile_u32_ptr,
                                         tile_size1);
    out_tile_u32_ptr += out_stride1;
    in_tile_u32_ptr += tile_size1;
  }
}
