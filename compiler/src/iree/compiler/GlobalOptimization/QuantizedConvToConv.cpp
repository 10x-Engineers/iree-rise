// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/GlobalOptimization/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_LINALGQUANTIZEDCONVTOCONVPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {

// Creates an empty copy matching the provided value.
Value emptyCopy(ImplicitLocOpBuilder &rewriter, Value value) {
  Type eTy = getElementTypeOrSelf(value.getType());
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(rewriter, rewriter.getLoc(), value);
  return tensor::EmptyOp::create(rewriter, mixedSizes, eTy);
}

// Creates an zero initialized tensor of given shape and type.
Value emptyZero(ImplicitLocOpBuilder &builder, RankedTensorType ty,
                llvm::SmallVector<Value> dyn) {
  Value empty =
      tensor::EmptyOp::create(builder, ty.getShape(), ty.getElementType(), dyn);

  TypedAttr attr = builder.getZeroAttr(ty.getElementType());
  Value cnst = arith::ConstantOp::create(builder, attr);
  return linalg::FillOp::create(builder, ValueRange{cnst}, ValueRange{empty})
      .result();
}

// Apply the multiply subtract corresponding with a zero-point adjustment
// broadcasting according to the affine map.
Value applyZeroPoint(ImplicitLocOpBuilder &builder, Value conv, Value sum,
                     Value zp, ArrayRef<int> affine_map) {
  auto context = builder.getContext();
  auto convTy = cast<RankedTensorType>(conv.getType());

  llvm::SmallVector<AffineExpr> sumExprs;
  for (auto i : affine_map) {
    sumExprs.push_back(builder.getAffineDimExpr(i));
  }

  SmallVector<utils::IteratorType> iterators(convTy.getRank(),
                                             utils::IteratorType::parallel);

  auto convMap = builder.getMultiDimIdentityMap(convTy.getRank());
  auto sumMap = AffineMap::get(convTy.getRank(), 0, sumExprs, context);

  SmallVector<AffineMap> affineMaps{convMap, sumMap, convMap};

  Value init = emptyCopy(builder, conv);
  return linalg::GenericOp::create(
             builder, init.getType(), ValueRange{conv, sum}, ValueRange{init},
             affineMaps, iterators,
             [=](OpBuilder &b, Location loc, ValueRange args) {
               Value mul = arith::MulIOp::create(b, loc, args[1], zp);
               Value sum = arith::SubIOp::create(b, loc, args[0], mul);
               linalg::YieldOp::create(b, loc, sum);
             })
      .getResult(0);
}

// Add the scalar value to the tensor.
Value addScalar(ImplicitLocOpBuilder &builder, Value value, Value scalar) {
  auto ty = cast<RankedTensorType>(value.getType());
  SmallVector<utils::IteratorType> iterators(ty.getRank(),
                                             utils::IteratorType::parallel);
  auto map = builder.getMultiDimIdentityMap(ty.getRank());
  Value init = emptyCopy(builder, value);
  return linalg::GenericOp::create(
             builder, init.getType(), ValueRange{value}, ValueRange{init},
             ArrayRef<AffineMap>{map, map}, iterators,
             [=](OpBuilder &b, Location loc, ValueRange args) {
               Value add = arith::AddIOp::create(b, loc, args[0], scalar);
               linalg::YieldOp::create(b, loc, add);
             })
      .getResult(0);
}

void GetDynamicDym(ImplicitLocOpBuilder &builder,
                   llvm::SmallVector<int64_t> &dims,
                   llvm::SmallVector<Value> &dynDims, Value value,
                   int64_t dim) {
  ShapedType ty = cast<ShapedType>(value.getType());
  dims.push_back(ty.getDimSize(dim));
  if (ty && ty.isDynamicDim(dim)) {
    dynDims.push_back(tensor::DimOp::create(builder, value, dim));
  }
}

Value multiplyDims(ImplicitLocOpBuilder &builder, Value value,
                   llvm::ArrayRef<int64_t> dims) {
  Value count = tensor::DimOp::create(builder, value, dims.front());

  for (auto d : dims.drop_front()) {
    Value dim = tensor::DimOp::create(builder, value, d);
    count = arith::MulIOp::create(builder, count, dim);
  }

  return count;
}

bool isZeroConstant(Value value) {
  IntegerAttr attr;
  return matchPattern(value, m_Constant(&attr)) && attr.getValue().isZero();
}

std::optional<int64_t> loopToOperandDim(AffineMap map, unsigned loop) {
  for (auto [pos, expr] : llvm::enumerate(map.getResults())) {
    if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
      if (dim.getPosition() == loop) {
        return pos;
      }
    }
  }
  return std::nullopt;
}

struct ConvLayout {
  int64_t batchOutputDim;
  int64_t channelOutputDim;
  int64_t channelFilterDim;
  std::optional<int64_t> inputChannelDim;
  SmallVector<int64_t> outImageOutputDims;
  SmallVector<int64_t> filterWindowDims;
  SmallVector<int64_t> spatialInputDims;
};

FailureOr<ConvLayout> getConvLayout(linalg::LinalgOp linalgOp) {
  FailureOr<linalg::ConvolutionDimensions> convDims =
      linalg::inferConvolutionDims(linalgOp);
  if (failed(convDims)) {
    return failure();
  }

  // Regular convs have a single output-channel and input-channel; depthwise
  // convs have a single shared depth dimension instead.
  bool isRegular = convDims->outputChannel.size() == 1 &&
                   convDims->inputChannel.size() == 1 &&
                   convDims->depth.empty();
  bool isDepthwise = convDims->outputChannel.empty() &&
                     convDims->inputChannel.empty() &&
                     convDims->depth.size() == 1;
  if (convDims->batch.size() != 1 || (!isRegular && !isDepthwise)) {
    return failure();
  }

  SmallVector<AffineMap> maps = linalgOp.getIndexingMapsArray();
  AffineMap inputMap = maps[0];
  AffineMap filterMap = maps[1];
  AffineMap outputMap = maps.back();
  unsigned channelLoop =
      isDepthwise ? convDims->depth[0] : convDims->outputChannel[0];

  std::optional<int64_t> batchOutputDim =
      loopToOperandDim(outputMap, convDims->batch[0]);
  std::optional<int64_t> channelOutputDim =
      loopToOperandDim(outputMap, channelLoop);
  std::optional<int64_t> channelFilterDim =
      loopToOperandDim(filterMap, channelLoop);
  if (!batchOutputDim || !channelOutputDim || !channelFilterDim) {
    return failure();
  }

  ConvLayout layout;
  layout.batchOutputDim = *batchOutputDim;
  layout.channelOutputDim = *channelOutputDim;
  layout.channelFilterDim = *channelFilterDim;

  if (isRegular) {
    std::optional<int64_t> icInputDim =
        loopToOperandDim(inputMap, convDims->inputChannel[0]);
    // The unit input-channel is later merged with the preceding dim, so it
    // cannot be the leading dimension.
    if (!icInputDim || *icInputDim == 0) {
      return failure();
    }
    layout.inputChannelDim = *icInputDim;
  }

  for (unsigned outImage : convDims->outputImage) {
    std::optional<int64_t> d = loopToOperandDim(outputMap, outImage);
    if (!d) {
      return failure();
    }
    layout.outImageOutputDims.push_back(*d);
  }
  for (unsigned filterLoop : convDims->filterLoop) {
    std::optional<int64_t> d = loopToOperandDim(filterMap, filterLoop);
    if (!d) {
      return failure();
    }
    layout.filterWindowDims.push_back(*d);
  }
  for (int64_t d = 0, e = inputMap.getNumResults(); d < e; ++d) {
    if (!isa<AffineDimExpr>(inputMap.getResult(d))) {
      layout.spatialInputDims.push_back(d);
    }
  }
  return layout;
}

// Pattern lowering a quantized convolution (conv_2d_nhwc_hwcf_q,
// conv_2d_nchw_fchw_q or depthwise_conv_2d_nhwc_hwc_q) to its non-quantized
// form.
//
// This is implementing the math explained in Section 2.3 of
// https://arxiv.org/abs/1712.05877.
template <typename QConvOpTy, typename ConvOpTy, typename PoolOpTy>
struct QuantizedConvToConv : public OpRewritePattern<QConvOpTy> {
  using OpRewritePattern<QConvOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(QConvOpTy op,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    ValueRange inputs = op.getInputs();
    Value input = inputs[0];
    Value filter = inputs[1];
    Value iZp = inputs[2];
    Value fZp = inputs[3];
    auto inputTy = cast<RankedTensorType>(input.getType());
    auto filterTy = cast<RankedTensorType>(filter.getType());
    auto resultTy = cast<ShapedType>(op.getType(0));
    auto accETy = resultTy.getElementType();
    int64_t inputRank = inputTy.getRank();

    auto strides = op.getStrides();
    auto dilations = op.getDilations();

    bool iZpIsZero = isZeroConstant(iZp);
    bool fZpIsZero = isZeroConstant(fZp);

    // Recover the layout only when a correction is needed, before creating IR.
    FailureOr<ConvLayout> layoutOr(failure());
    if (!iZpIsZero || !fZpIsZero) {
      layoutOr = getConvLayout(cast<linalg::LinalgOp>(op.getOperation()));
      if (failed(layoutOr)) {
        return rewriter.notifyMatchFailure(op,
                                           "unsupported convolution layout");
      }
    }

    // First implement the convolution without the zero point.
    Value newConv =
        ConvOpTy::create(builder, resultTy, ValueRange{input, filter},
                         op.getOutputs(), strides, dilations)
            .getResult(0);

    // If the zero point values are both zero we can just replace.
    if (iZpIsZero && fZpIsZero) {
      rewriter.replaceOp(op, newConv);
      return success();
    }
    const ConvLayout &layout = *layoutOr;

    // newConv -= iZp * filterSum   (broadcast over the output channel)
    if (!iZpIsZero) {
      SmallVector<bool> filterReduce(filterTy.getRank(), true);
      filterReduce[layout.channelFilterDim] = false;
      Value filterSum =
          sumReduceDimensionSubset(builder, filter, accETy, filterReduce);
      newConv = applyZeroPoint(builder, newConv, filterSum, iZp,
                               {static_cast<int>(layout.channelOutputDim)});
    }

    // newConv -= fZp * inputSum   (broadcast over batch + output image)
    if (!fZpIsZero) {
      // Regular convs reduce the input channel and re-materialize it as a unit
      // dimension before pooling; depthwise convs pool the input directly.
      bool reduceChannel = layout.inputChannelDim.has_value();
      Value poolInput = input;
      SmallVector<ReassociationExprs> reassociationMap;
      if (reduceChannel) {
        int64_t icDim = *layout.inputChannelDim;
        SmallVector<bool> inputReduce(inputRank, false);
        inputReduce[icDim] = true;
        Value reduced =
            sumReduceDimensionSubset(builder, input, accETy, inputReduce);

        reassociationMap.resize(inputRank - 1);
        for (int64_t d = 0, g = -1; d < inputRank; ++d) {
          if (d == icDim) {
            reassociationMap[g].push_back(builder.getAffineDimExpr(d));
          } else {
            reassociationMap[++g].push_back(builder.getAffineDimExpr(d));
          }
        }
        SmallVector<int64_t> expandShape(inputRank);
        for (int64_t d = 0; d < inputRank; ++d) {
          expandShape[d] = d == icDim ? 1 : inputTy.getDimSize(d);
        }
        poolInput = tensor::ExpandShapeOp::create(
            builder, RankedTensorType::get(expandShape, accETy), reduced,
            reassociationMap);
      }

      SmallVector<int64_t> poolDims;
      SmallVector<Value> poolDynDims;
      for (int64_t d = 0; d < inputRank; ++d) {
        bool isSpatial = llvm::is_contained(layout.spatialInputDims, d);
        GetDynamicDym(builder, poolDims, poolDynDims,
                      isSpatial ? newConv : poolInput, d);
      }
      auto poolTy = RankedTensorType::get(poolDims, accETy);
      Value poolTensor = emptyZero(builder, poolTy, poolDynDims);

      // Create the empty kernel defining the shape for the pooling operation.
      SmallVector<int64_t> kDims;
      SmallVector<Value> kDyn;
      for (int64_t windowDim : layout.filterWindowDims) {
        GetDynamicDym(builder, kDims, kDyn, filter, windowDim);
      }
      Value poolInit = tensor::EmptyOp::create(builder, kDims, accETy, kDyn);

      Value inputSum = PoolOpTy::create(builder, ArrayRef<Type>{poolTy},
                                        ValueRange{poolInput, poolInit},
                                        poolTensor, strides, dilations)
                           .getResult(0);

      // Collapse the re-materialized unit channel back out (regular convs).
      if (reduceChannel) {
        SmallVector<int64_t> collapseShape;
        for (int64_t d = 0; d < inputRank; ++d) {
          if (d != *layout.inputChannelDim) {
            collapseShape.push_back(poolDims[d]);
          }
        }
        inputSum = tensor::CollapseShapeOp::create(
            builder, RankedTensorType::get(collapseShape, accETy), inputSum,
            reassociationMap);
      }

      SmallVector<int> broadcast{static_cast<int>(layout.batchOutputDim)};
      for (int64_t d : layout.outImageOutputDims) {
        broadcast.push_back(static_cast<int>(d));
      }
      if (!reduceChannel) {
        broadcast.push_back(static_cast<int>(layout.channelOutputDim));
      }
      llvm::sort(broadcast);
      newConv = applyZeroPoint(builder, newConv, inputSum, fZp, broadcast);
    }

    // newConv += iZp * fZp * (number of reduced elements)
    if (!iZpIsZero && !fZpIsZero) {
      SmallVector<int64_t> countDims;
      for (int64_t d = 0, e = filterTy.getRank(); d < e; ++d) {
        if (d != layout.channelFilterDim) {
          countDims.push_back(d);
        }
      }
      Value count = multiplyDims(builder, filter, countDims);
      Value cast = arith::IndexCastOp::create(builder, accETy, count);
      Value ifZp = arith::MulIOp::create(builder, iZp, fZp);
      Value zpUpdate = arith::MulIOp::create(builder, ifZp, cast);
      newConv = addScalar(builder, newConv, zpUpdate);
    }

    rewriter.replaceOp(op, newConv);
    return success();
  }
};

/// Pass that lowers quantized_conv to conv.
class LinalgQuantizedConvToConvPass final
    : public impl::LinalgQuantizedConvToConvPassBase<
          LinalgQuantizedConvToConvPass> {
public:
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    RewritePatternSet patterns(context);
    linalg::populateSimplifyDepthwiseConvPatterns(patterns);
    patterns.add<
        QuantizedConvToConv<linalg::Conv2DNhwcHwcfQOp, linalg::Conv2DNhwcHwcfOp,
                            linalg::PoolingNhwcSumOp>,
        QuantizedConvToConv<linalg::Conv2DNchwFchwQOp, linalg::Conv2DNchwFchwOp,
                            linalg::PoolingNchwSumOp>,
        QuantizedConvToConv<linalg::DepthwiseConv2DNhwcHwcQOp,
                            linalg::DepthwiseConv2DNhwcHwcOp,
                            linalg::PoolingNhwcSumOp>>(context);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
