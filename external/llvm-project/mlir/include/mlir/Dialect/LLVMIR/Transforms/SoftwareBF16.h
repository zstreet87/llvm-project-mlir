//===- LegalizeForExport.h - Prepare for translation to LLVM IR -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_TRANSFORMS_SOFTWAREBF16_H
#define MLIR_DIALECT_LLVMIR_TRANSFORMS_SOFTWAREBF16_H

#include <memory>

namespace mlir {
class Operation;
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

namespace LLVM {

void doBF16(Operation *op);
void populateBF16(LLVMTypeConverter&, RewritePatternSet&);
/// Creates a pass that converts BF16 type
std::unique_ptr<Pass> createSoftwareBF16Pass();

} // namespace LLVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_TRANSFORMS_SOFTWAREBF16_H
