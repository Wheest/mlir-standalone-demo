#pragma once

// make an affine for loop
#define MLIR_STND_BEGIN_AFFINE(loop_name, lb, ub, step, induction_var_name)    \
  AffineForOp loop_name = rewriter.create<AffineForOp>(loc, lb, ub, step);     \
  Value induction_var_name = loop_name.getInductionVar();                      \
  rewriter.setInsertionPointToStart(loop_name.getBody());

#define MLIR_STND_END_AFFINE(loop_name)                                        \
  rewriter.setInsertionPointAfter(loop_name);
