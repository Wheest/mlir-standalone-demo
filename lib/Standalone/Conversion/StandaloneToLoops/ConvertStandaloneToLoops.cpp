#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Standalone/ConversionMacros.h"
#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"
#include "Standalone/StandalonePasses.h"

#include <iostream>
#include <vector>

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::memref;
using namespace mlir::standalone;

namespace {
struct MyOpRewriter : public OpRewritePattern<MyOp> {
  using OpRewritePattern<MyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MyOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    IntegerType intType = rewriter.getIntegerType(64);
    FloatType floatType = rewriter.getF64Type();
    Value lhs = op.lhs();
    Value rhs = op.rhs();
    Value res = op.res();
    ArrayRef<int64_t> matShape = res.getType().cast<MemRefType>().getShape();

    // the loop
    MLIR_STND_BEGIN_AFFINE(loop, 0, matShape[0], 1, i)
    Value lhs_i = rewriter.create<AffineLoadOp>(loc, lhs, ArrayRef<Value>{i});
    MLIR_STND_END_AFFINE(loop)

    MemRefType my_type = MemRefType::get({5}, intType);
    Value my_alloc = rewriter.create<AllocOp>(loc, my_type);

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

// find the ops that can be converted
namespace mlir {
void populateStandaloneToLoopsConversionPatterns(MLIRContext *context,
                                                 RewritePatternSet &patterns) {
  patterns.insert<MyOpRewriter>(context);
}
} // namespace mlir
  //
namespace {
struct StandaloneToLoopsLoweringPass
    : public PassWrapper<StandaloneToLoopsLoweringPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, MemRefDialect, ArithmeticDialect>();
  }
  void runOnFunction() final;
  StringRef getArgument() const final { return "standalone-to-loops"; }
  StringRef getDescription() const final {
    return "lower Standalone dialect to loops";
  }
};
} // namespace

// lower all functions
void StandaloneToLoopsLoweringPass::runOnFunction() {
  ConversionTarget target(getContext());
  target.addLegalDialect<AffineDialect, scf::SCFDialect, ArithmeticDialect,
                         StandardOpsDialect>();
  RewritePatternSet patterns(&getContext());
  populateStandaloneToLoopsConversionPatterns(&getContext(), patterns);
  if (failed(
          applyPartialConversion(getFunction(), target, std::move(patterns))))
    signalPassFailure();
}

void Standalone::createLowerStandaloneToLoopsPass() {
  PassRegistration<StandaloneToLoopsLoweringPass>();
}
