//===- LegalizeForExport.cpp - Prepare for translation to LLVM IR ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/Transforms/SoftwareBF16.h"
#include "PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

//#include "mlir/IR/Block.h"
//#include "mlir/IR/Builders.h"
//#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace mlir {
namespace LLVM {
/// Rewrites bf16 constants to their i16 equivalents
/// This is relying on the fact that the vector, i16, and bf16 types used in the
/// LLVM dialect are the standard ones and not weird custom wrappers
//struct BF16ConstCasting : OpRewritePattern<LLVM::ConstantOp> {
//  explicit BF16ConstCasting(MLIRContext *context) : OpRewritePattern(context) {}
struct  BF16ConstCasting : public ConvertOpToLLVMPattern<LLVM::ConstantOp> {
  using ConvertOpToLLVMPattern<ConstantOp>::ConvertOpToLLVMPattern;

  llvm::APInt toInt(llvm::APFloat value) const {
    assert(&value.getSemantics() == &llvm::APFloat::BFloat() &&
           "Must cast bf16 only");
    APInt ret = value.bitcastToAPInt();
    assert(ret.getBitWidth() == 16 && "bf16 conversion should make i16");
    return ret;
  }

  LogicalResult matchAndRewrite(LLVM::ConstantOp op, LLVM::ConstantOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // llvm::errs()<<"ConstCast\n";
    Attribute val = op.getValueAttr();
    Operation *rawOp = op.getOperation();
    Type bf16 = rewriter.getBF16Type();
    Type i16 = rewriter.getIntegerType(16);
    Type retType = typeConverter->convertType(op.getRes().getType());
    // Type retElemType = retType;
    Type retElemType = i16;
    // if (auto retTypeShaped = retType.dyn_cast<ShapedType>())
    //  retElemType = retTypeShaped.getElementType();

    if (auto valFloat = val.dyn_cast<mlir::FloatAttr>()) {
      if (valFloat.getType() != bf16)
        return failure();
      llvm::errs() << "ConstCast: ";
      op->dump();
      APInt newVal = toInt(valFloat.getValue());
      mlir::IntegerAttr tmp = rewriter.getIntegerAttr(i16, newVal);
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
          rawOp, retType, rewriter.getIntegerAttr(i16, newVal));
      llvm::errs() << "To ";
      tmp.dump();
      llvm::errs() << "\n";
      return success();
    }

    if (auto valDense = val.dyn_cast<mlir::DenseElementsAttr>()) {
      if (valDense.getElementType() != bf16)
        return failure();
      DenseElementsAttr newVal = valDense.bitcast(retElemType);
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(rawOp, retType, newVal);
      return success();
    }

    if (auto valSparse = val.dyn_cast<mlir::SparseElementsAttr>()) {
      if (valSparse.getElementType() != bf16)
        return failure();
      DenseElementsAttr values = valSparse.getValues();
      DenseElementsAttr newValues = values.bitcast(retElemType);
      auto newVal = SparseElementsAttr::get(retType.cast<ShapedType>(),
                                            valSparse.getIndices(), newValues);
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(rawOp, retType, newVal);
      return success();
    }
    // No match otherwise
    return failure();
  }
};

template <typename Op>
//struct BF16AsF32 : OpRewritePattern<Op> {
//  explicit BF16AsF32(MLIRContext *context) : OpRewritePattern<Op>(context) {}
struct  BF16AsF32 : public ConvertOpToLLVMPattern<Op> {
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "BF16AsF32: ";
    op->dump();
    Location loc = op.getLoc();
    Type resType = op.getResult().getType();

    // resType.dump();
    Type extType = rewriter.getF32Type();
    Type resElementType = resType;

    Type i16 = rewriter.getIntegerType(16);
    Type bf16 = rewriter.getBF16Type();
    if (auto resShaped = resType.dyn_cast<ShapedType>()) {
      extType = resShaped.clone(extType);
      resElementType = resShaped.getElementType();
      if (resElementType == bf16)
        resType = resShaped.clone(i16);
    }

    //if (resElementType != i16)
    if (resElementType != bf16)
      return failure();

    llvm::errs() << "convert it to: ";
    llvm::SmallVector<Value, 2> extended;
    //for (Value v : op.getOperands()) {
    for (Value v : adaptor.getOperands()) {
      extended.push_back(rewriter.create<LLVM::FPExtOp>(loc, extType, v)); // i16->f32
    }

    Op operation = rewriter.create<Op>(loc, extType, extended, op->getAttrs());
    rewriter.replaceOpWithNewOp<LLVM::FPTruncOp>(op, this->getTypeConverter()->convertType(resType),
                                                 operation.getResult());

    llvm::errs() << "To ";
    operation.dump();
    llvm::errs()<<"\n";
    return success();
  }
};
/*
template <typename Op>
struct UnaryBF16AsF32 : OpRewritePattern<Op> {
  explicit UnaryBF16AsF32(MLIRContext *context)
      : OpRewritePattern<Op>(context) {}

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resType = op.getResult().getType();
    Type extType = rewriter.getF32Type();
    Type resElementType = resType;

    if (auto resShaped = resType.dyn_cast<ShapedType>()) {
      extType = resShaped.clone(extType);
      resElementType = resShaped.getElementType();
    }
    Type i16 = rewriter.getIntegerType(16);
    if (resElementType != i16)
      return failure();

    op->dump();
    llvm::SmallVector<Value, 2> extended;
    // for (Value v : op.getOperand()) {
    extended.push_back(
        rewriter.create<LLVM::FPExtOp>(loc, extType, op.getOperand()));
    //}
    Op operation = rewriter.create<Op>(loc, extType, extended, op->getAttrs());
    rewriter.replaceOpWithNewOp<LLVM::FPTruncOp>(op, resType,
                                                 operation.getResult());
    operation.dump();
    llvm::errs()<<"\n";
    return success();
  }
};
*/
struct  SoftwareBF16Cmp : public ConvertOpToLLVMPattern<FCmpOp> {
  using ConvertOpToLLVMPattern<FCmpOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(LLVM::FCmpOp op, LLVM::FCmpOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "BF16Fcmp: ";
    op->dump();
    Location loc = op.getLoc();
    Type resType = op.getResult().getType();
    // resType.dump();
    // llvm::errs()<<"\n";
    Type extType = rewriter.getF32Type();
    Type resElementType = resType;

    if (auto resShaped = resType.dyn_cast<ShapedType>()) {
      extType = resShaped.clone(extType);
      resElementType = resShaped.getElementType();
    }
    Type i16 = rewriter.getIntegerType(16);

    //Type bf16 = rewriter.getBF16Type();
    auto operandType = adaptor.getLhs().getType();
    if (operandType != i16)
        return failure();

    llvm::errs() << "convert a BF16Cmp: ";
    op->dump();
    llvm::SmallVector<Value, 2> extended;
    for (Value v : adaptor.getOperands()) {
      extended.push_back(rewriter.create<LLVM::FPExtOp>(loc, extType, v));
    }

    // extType = resElementType;
    LLVM::FCmpOp operation =
        rewriter.create<LLVM::FCmpOp>(loc, resType, extended, op->getAttrs());
    llvm::errs() << "after create FCmpOp\n";
    operation.dump();
    rewriter.replaceOp(op.getOperation(), {operation});

    llvm::errs() << "To ";
    operation.dump();
    llvm::errs() << "\n";
    return success();
  }
};

Value getLlvmI32Const(Location loc, PatternRewriter &rewriter, Type type,
                      int32_t value) {
  Attribute ret = rewriter.getI32IntegerAttr(value);
  if (LLVM::isCompatibleVectorType(type))
    ret = SplatElementsAttr::get(type.cast<ShapedType>(), ret);
  return rewriter.create<LLVM::ConstantOp>(loc, type, ret);
}

/// Rewrites extension of bfloat as a bitshift. This is needed since the ROCDL
/// target doesn't support the bfloat type even though LLVM in general does.
struct  SoftwareBF16Ext : public ConvertOpToLLVMPattern<LLVM::FPExtOp> {
  using ConvertOpToLLVMPattern<FPExtOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(LLVM::FPExtOp op, LLVM::FPExtOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    //Type srcType = op.getArg().getType();
    Type srcType = adaptor.getArg().getType();
    Type destType = op.getResult().getType();
    Type srcElemType = srcType;
    if (auto shaped = srcType.dyn_cast<ShapedType>())
      srcElemType = shaped.getElementType();

    Type i16 = rewriter.getIntegerType(16);
    // Type bf16 = rewriter.getBF16Type();
    if (srcElemType != i16) // && srcElemType != bf16)
      return failure();

    Type extType = rewriter.getI32Type();
    if (auto srcShaped = srcType.dyn_cast<ShapedType>())
      extType = srcShaped.clone(extType);

    Type f32 = rewriter.getF32Type();
    if (auto destShaped = destType.dyn_cast<ShapedType>()) {
      if (destShaped.getElementType() != f32)
        return failure();
    } else if (destType != f32)
      return failure();

    llvm::errs()<<"Software FPExtOp: ";
    op.dump();
    Value extended = rewriter.create<LLVM::ZExtOp>(loc, extType, adaptor.getArg());
    Value shifted = rewriter.create<LLVM::ShlOp>(
        loc, extended, getLlvmI32Const(loc, rewriter, extType, 16));
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, destType, shifted);
    extended.dump();
    shifted.dump();

    llvm::errs()<<"\n";
    return success();
  }
};

/// Rewrites truncation to bfloat as a series of integer operations.
/// This is needed since the ROCDL target doesn't support the bfloat type,
/// even though LLVM in general does
struct  SoftwareBF16Trunc : public ConvertOpToLLVMPattern<LLVM::FPTruncOp> {
  using ConvertOpToLLVMPattern<FPTruncOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(LLVM::FPTruncOp op, LLVM::FPTruncOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Type srcType = op.getArg().getType();
    Type destType = op.getRes().getType();
    Type srcElemType = srcType;
    if (auto shaped = srcType.dyn_cast<ShapedType>())
      srcElemType = shaped.getElementType();

    Type f32 = rewriter.getF32Type();
    if (srcElemType != f32)
      return failure();

    Type bitcastType = rewriter.getI32Type();
    if (auto srcShaped = srcType.dyn_cast<ShapedType>())
      bitcastType = srcShaped.clone(bitcastType);

    Type bf16 = rewriter.getBF16Type();
    Type i16 = rewriter.getIntegerType(16);
    if (auto destShaped = destType.dyn_cast<ShapedType>()) {
      if (destShaped.getElementType() != i16 &&
          destShaped.getElementType() != bf16)
        return failure();
    } else if (destType != bf16 && destType != i16)
      return failure();
llvm::errs()<<"Software FPTruncOp: ";
    op.dump();

    // a = bitcast f32 value to i32
    // b = (a + 32767) << 16
    // c = ((a << 16) & 1)
    // d = b + c
    // truncate (d << 16) to i16 and return this i16
    Value bitcastop =
        rewriter.create<LLVM::BitcastOp>(loc, bitcastType, op.getArg());
    Value constantSixteen = getLlvmI32Const(loc, rewriter, bitcastType, 16);
    Value shiftValue = rewriter.create<LLVM::LShrOp>(
        loc, bitcastType, bitcastop, constantSixteen);

    Value constantOne = getLlvmI32Const(loc, rewriter, bitcastType, 1);
    Value andValue = rewriter.create<LLVM::AndOp>(loc, shiftValue, constantOne);

    Value constantBig = getLlvmI32Const(loc, rewriter, bitcastType, 32767);
    Value addBigValue =
        rewriter.create<LLVM::AddOp>(loc, bitcastop, constantBig);
    Value addValue = rewriter.create<LLVM::AddOp>(loc, andValue, addBigValue);

    Value shiftBeforeTruncValue = rewriter.create<LLVM::LShrOp>(
        loc, bitcastType, addValue, constantSixteen);
    Value truncValue =
        rewriter.create<LLVM::TruncOp>(loc, typeConverter->convertType(destType), shiftBeforeTruncValue);
    rewriter.replaceOp(op.getOperation(), {truncValue});
    truncValue.dump();

    llvm::errs()<<"\n";
    return success();
  }
};

struct  SoftwareBF16Load : public ConvertOpToLLVMPattern<LLVM::LoadOp> {
  using ConvertOpToLLVMPattern<LoadOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(LLVM::LoadOp op, LLVM::LoadOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    //Type i16 = rewriter.getIntegerType(16);
    Type bf16 = rewriter.getBF16Type();
    Type retType = op.getResult().getType();
    Type retElemType = retType;
    if (auto retTypeShaped = retType.dyn_cast<ShapedType>())
      retElemType = retTypeShaped.getElementType();
    if (retElemType != bf16)
      return failure();
    llvm::errs()<<"convert a BFload ";
    Value loadOp = rewriter.create<LLVM::LoadOp>(loc, typeConverter->convertType(retType), adaptor.getAddr()); //, op.getAlignment(), op.getVolatile_(), op.getNontemporal());

    loadOp.dump();
    llvm::errs()<<"\n";
    rewriter.replaceOp(op.getOperation(), {loadOp});
    return success();
  }
};

struct  SoftwareBF16Store : public ConvertOpToLLVMPattern<LLVM::StoreOp> {
  using ConvertOpToLLVMPattern<StoreOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(LLVM::StoreOp op, LLVM::StoreOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    //auto loc = op.getLoc();

    //uint64_t alignment = op.getAlignment().getValue();
  rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getValue(),
                                             adaptor.getAddr());//, alignment,
                                             //op.getVolatile_(), op.getNontemporal());
  }
};
struct  SoftwareBF16Call : public ConvertOpToLLVMPattern<LLVM::CallOp> {
  using ConvertOpToLLVMPattern<CallOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(LLVM::CallOp op, LLVM::CallOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

//op.getResult(0).getType().dump();
//llvm::errs()<<"\n";
    FlatSymbolRefAttr calleeName = op.getCalleeAttr();
    Operation *callee =
         SymbolTable::lookupNearestSymbolFrom(op, calleeName.getAttr());
    auto fn = dyn_cast<LLVMFuncOp>(callee);
    mlir::NamedAttribute attr = rewriter.getNamedAttr(
            "callee", mlir::SymbolRefAttr::get(fn));
    if (op.getNumResults() == 0) {
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(
          op, llvm::None, adaptor.getOperands(), llvm::ArrayRef<mlir::NamedAttribute>{attr});
      return success();
    }

    // Function returns a single result.
    auto dstType = typeConverter->convertType(op.getResult(0).getType());
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, dstType, adaptor.getOperands(), llvm::ArrayRef<mlir::NamedAttribute>{attr});
    return success();

    //Operation *callee =
    //     SymbolTable::lookupNearestSymbolFrom(op, calleeName.getAttr());
    //auto fn = dyn_cast<LLVMFuncOp>(callee);
    //auto callOp = rewriter.create<LLVM::CallOp>(loc, fn, adaptor.getOperands());
    //rewriter.replaceOp(op.getOperation(), {callOp});
    //rewriter.eraseOp(op);
    
    return success();
  }
};
/*
if (auto loadOp = dyn_cast<spirv::LoadOp>(op)) {
    auto dstType = typeConverter.convertType(loadOp.getType());
    if (!dstType)
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        loadOp, dstType, spirv::LoadOpAdaptor(operands).ptr(), alignment,
        isVolatile, isNonTemporal);
    return success();
  }
  auto storeOp = cast<spirv::StoreOp>(op);
  spirv::StoreOpAdaptor adaptor(operands);
  rewriter.replaceOpWithNewOp<LLVM::StoreOp>(storeOp, adaptor.value(),
                                             adaptor.ptr(), alignment,
                                             isVolatile, isNonTemporal);
  return success();*/

template <typename LLVMOp>
struct ConvertBF16ToI16 : public ConvertOpToLLVMPattern<LLVMOp> {
  using ConvertOpToLLVMPattern<LLVMOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(LLVMOp op, typename LLVMOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "Entering ConvertBF16ToI16: ";
    op.dump();
    llvm::errs()<<"\n";
    for ( auto v: adaptor.getOperands() )
      v.dump();
    //assert(!adaptor.getOperands().empty());
    return LLVM::detail::oneToOneRewrite(op, LLVMOp::getOperationName(),
                                         adaptor.getOperands(),
                                         *this->getTypeConverter(), rewriter);
  }
};

struct ConvertBF16Load : ConvertOpToLLVMPattern<LLVM::LoadOp> {
  using ConvertOpToLLVMPattern<LLVM::LoadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(LLVM::LoadOp op, LLVM::LoadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type retType = op.getResult().getType();
    Type retElemType = retType;
    if (auto retTypeShaped = retType.dyn_cast<ShapedType>())
      retElemType = retTypeShaped.getElementType();
    if (retElemType != rewriter.getBF16Type())
      return failure();
    llvm::errs()<<"convert a BFload to ";
    op.getAddr().dump();
    llvm::errs()<<"\n";
    Value loadOp = rewriter.create<LLVM::LoadOp>(loc, typeConverter->convertType(retType), op.getAddr());
    llvm::errs()<<"convert a BFload to ";
    loadOp.dump();
    llvm::errs()<<"\n";
    rewriter.replaceOp(op.getOperation(), {loadOp});
    return success();
  }
};

/*
struct ConvertBFConstCasting : ConvertOpToLLVMPattern<LLVM::ConstantOp> {
  explicit ConvertBFConstCasting : ConvertOpToLLVMPattern() {}
  LogicalResult
  matchAndRewrite(LLVM::ConstantOp op, typename LLVM::ConstantOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = op.getRes().getType();
    Type retElemType = retType; 
    if (auto retTypeShaped = retType.dyn_cast<ShapedType>())
      retElemType = retTypeShaped.getElementType();
    if (retElemType != rewriter.getBF16Type()) 
      return failure();
    return 
  }
}

void convertBF16Type(LLVMTypeConverter &converter,
                     RewritePatternSet &patterns) {
  patterns.add<ConvertBF16ToI16<LLVM::FAddOp>, ConvertBF16ToI16<LLVM::FMulOp>,
               ConvertBF16ToI16<LLVM::FSubOp>, ConvertBF16ToI16<LLVM::FDivOp>,
               ConvertBF16ToI16<LLVM::FCmpOp>, ConvertBF16ToI16<LLVM::FPExtOp>,
               ConvertBF16ToI16<LLVM::FPTruncOp>, ConvertBF16ToI16<LLVM::ConstantOp>>(converter);
               //ConvertBF16Load>(converter);
}

 */ 
} // namespace LLVM
} // namespace mlir

void mlir::LLVM::doBF16(Operation *op) {
  llvm::errs() << "SoftwareBF16pass!\n";
  return;
}

void mlir::LLVM::populateBF16(LLVMTypeConverter &converter,
                              RewritePatternSet &patterns) {
  llvm::errs() << "populateBF16\n";
  MLIRContext *ctx = converter.getDialect()->getContext();
  // AMD GPUs don't have a backend that understands bfloat, even though LLVM's
  // frontend does. Remove this if/when that changes. Note that adding
  // conversions after the default constructor runs gives them priority
  // over the defaults.
  Type llvmI16 = converter.convertType(IntegerType::get(ctx, 16));
  Type bf16 = mlir::BFloat16Type::get(ctx);
  converter.addConversion([llvmI16](mlir::BFloat16Type type) -> Type {
    llvm::errs() << "\ntype-convert a bf16\n";
    return llvmI16;
  });

/*
  converter.addConversion([llvmI16, bf16](LLVM::LLVMPointerType type) -> llvm::Optional<Type> {
    if ( type.getElementType() == bf16 )
      return LLVM::LLVMPointerType::get(llvmI16, type.getAddressSpace());
    return type;
  });
*/
  // Override for vector types since they get caught by isCompatibleType(),
  // which doesn't convert the element type
  converter.addConversion(
      [llvmI16, bf16](mlir::VectorType type) -> Optional<Type> {
        if (type.getElementType() == bf16 && type.getRank() == 1)
          return type.clone(llvmI16);
        return llvm::None; // continue search
      });


  converter.addConversion([llvmI16, bf16](LLVM::LLVMStructType type) -> Optional<Type> {
    SmallVector<Type> convertedElemTypes;
    convertedElemTypes.reserve(type.getBody().size());
    bool converted = false;
//llvm::errs()<<"customized converter: ";
//type.dump();
//llvm::errs()<<"\n";
    for (auto t : type.getBody()) {
      if ( t == bf16 )
      {
        convertedElemTypes.push_back(llvmI16);
        converted = true;
      }
      else if ( t.isa<LLVM::LLVMPointerType>() )
      {
//llvm::errs()<<"my pos1\n";
        LLVM::LLVMPointerType ptr_type = t.dyn_cast<LLVM::LLVMPointerType>();
        if ( ptr_type.getElementType() == bf16 ) {
//llvm::errs()<<"my pos2\n";
          Type i16ptr = LLVM::LLVMPointerType::get(llvmI16, ptr_type.getAddressSpace());
          convertedElemTypes.push_back(i16ptr);
          converted = true;
        }
      }
      else
        convertedElemTypes.push_back(t);
     }
     if (converted)
{
//llvm::errs()<<"my pos3\n";
       return LLVM::LLVMStructType::getLiteral(type.getContext(), convertedElemTypes, type.isPacked());
}
     return llvm::None; 
  });


  patterns.add<LLVM::BF16ConstCasting, LLVM::SoftwareBF16Trunc,
               LLVM::SoftwareBF16Ext>(converter);
  patterns.add<LLVM::BF16AsF32<LLVM::FAddOp>,
               LLVM::BF16AsF32<LLVM::FMulOp>,
               LLVM::BF16AsF32<LLVM::FMAOp>,
               LLVM::BF16AsF32<LLVM::FSubOp>, 
               LLVM::BF16AsF32<LLVM::FDivOp>>(converter);

  patterns.add<LLVM::SoftwareBF16Cmp>(converter);
  //patterns.add<LLVM::SoftwareBF16Cmp, LLVM::UnaryBF16AsF32<LLVM::FAbsOp>>(converter);
  patterns.add<LLVM::SoftwareBF16Load, LLVM::SoftwareBF16Store>(converter);
  patterns.add<LLVM::ConvertBF16ToI16<LLVM::GEPOp>, LLVM::ConvertBF16ToI16<LLVM::AllocaOp>,
               LLVM::ConvertBF16ToI16<LLVM::PtrToIntOp>, LLVM::ConvertBF16ToI16<LLVM::BitcastOp>, LLVM::ConvertBF16ToI16<LLVM::NullOp>,
                LLVM::ConvertBF16ToI16<LLVM::UndefOp>>(converter);
  patterns.add<LLVM::SoftwareBF16Call>(converter);
  patterns.add<LLVM::ConvertBF16ToI16<LLVM::InsertValueOp>, LLVM::ConvertBF16ToI16<LLVM::ExtractValueOp>>(converter); 
}

namespace {
struct SoftwareBF16Pass : public SoftwareBF16Base<SoftwareBF16Pass> {
  void runOnOperation() override {
    LLVM::doBF16(getOperation());
    auto m = getOperation();
    MLIRContext *ctx = m->getContext();
    LowerToLLVMOptions options(ctx);
    LLVMTypeConverter converter(ctx, options);
    RewritePatternSet bf16fixupPatterns(ctx);
    RewritePatternSet llvmPatterns(ctx);
    LLVMConversionTarget target(getContext());

    auto isLegalOperation = [&](Operation *op) {
llvm::errs()<<"isLegalOperation : ";
op->dump();
llvm::errs()<<"\n";
      return converter.isLegal(op);
    };

    auto isLegalFPExtOpOrFPTruncOp =  [&](Operation *op) {
      
      return converter.isLegal(op) && op->getOperands()[0].getType().isa<FloatType>()
                                   && op->getResults()[0].getType().isa<FloatType>();
    };

    auto isLegalUndefOp = [&](Operation *op) {
      //if (converter.isLegal(op->getResultTypes()))
      //  llvm::errs()<<"Legal UnderfOp\n";
      //else
      //  llvm::errs()<<"ilLegal UnderfOp\n";
      return converter.isLegal(op->getResultTypes()); 
    };

    auto isLegalFuncOp  = [&](Operation *op) {
      
    };

    target.addDynamicallyLegalDialect<::mlir::LLVM::LLVMDialect>(isLegalOperation);
    //target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
    target
        .addIllegalOp<LLVM::CosOp, LLVM::ExpOp, LLVM::Exp2Op, LLVM::FAbsOp,
                      LLVM::FCeilOp, LLVM::FFloorOp, LLVM::LogOp, LLVM::Log10Op,
                      LLVM::Log2Op, LLVM::PowOp, LLVM::SinOp, LLVM::SqrtOp>();

    target.addDynamicallyLegalOp<LLVM::FAddOp, LLVM::FDivOp, LLVM::FAddOp,
                                 LLVM::FMulOp, LLVM::FMAOp,
                                 LLVM::FAbsOp, LLVM::FCmpOp,
                                 LLVM::ConstantOp, LLVM::LoadOp, LLVM::StoreOp, LLVM::GEPOp, LLVM::AllocaOp,
                                 LLVM::PtrToIntOp, LLVM::BitcastOp, LLVM::InsertValueOp, LLVM::ExtractValueOp, LLVM::NullOp, LLVM::CallOp, LLVM::LLVMFuncOp>(isLegalOperation);

    target.addDynamicallyLegalOp<LLVM::FPExtOp, LLVM::FPTruncOp>(isLegalFPExtOpOrFPTruncOp);
    target.addDynamicallyLegalOp<LLVM::UndefOp>(isLegalUndefOp);

    //target.markUnknownOpDynamicallyLegal(isLegalOperation);

    //LLVM::convertBF16Type(converter, llvmPatterns);
    LLVM::populateBF16(converter, bf16fixupPatterns);
    if (failed(applyPartialConversion(m, target, std::move(bf16fixupPatterns))))
      signalPassFailure();
    llvm::errs()<<"\nDone applyPartialConversion\n";
    //if (failed(applyPatternsAndFoldGreedily(m, std::move(bf16fixupPatterns))))
    //  signalPassFailure();
    //llvm::errs()<<"\nDone applyPatternsAndFoldGreedily\n";
  }
};
} // namespace

std::unique_ptr<Pass> LLVM::createSoftwareBF16Pass() {
  return std::make_unique<SoftwareBF16Pass>();
}
