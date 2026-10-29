// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// The kokkos-metal compiler plugin, loaded into clang with
// -fpass-plugin=libKokkosMetalPlugin.dylib. At the start of the optimisation
// pipeline, for every kernel entry the Metal backend instantiated
// (Kokkos_Metal_Intrinsics.hpp):
//
//  1. the entry and everything it reaches is cloned into a device module,
//     inlined and optimised there, and translated to MSL;
//  2. in the host module the entry's body is dropped, and a static
//     constructor registers the MSL under the entry's key with the runtime.

#include "MSLEmitter.hpp"

#include <llvm/ADT/DenseSet.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DiagnosticInfo.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/MD5.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

using namespace llvm;

namespace {

constexpr const char* kAnnotation = "kokkos_metal_kernel";
// MetalRuntime::KernelFlags.
constexpr uint32_t kUnsupported = 1u << 2;

struct Entry {
  Function* fn;
  GlobalVariable* key;
};

std::vector<Entry> find_entries(Module& M) {
  std::vector<Entry> out;
  GlobalVariable* annotations = M.getGlobalVariable("llvm.global.annotations");
  if (!annotations || !annotations->hasInitializer()) return out;
  auto* list = dyn_cast<ConstantArray>(annotations->getInitializer());
  if (!list) return out;
  for (Value* op : list->operands()) {
    auto* item = dyn_cast<ConstantStruct>(op);
    if (!item || item->getNumOperands() < 2) continue;
    auto* fn = dyn_cast<Function>(item->getOperand(0)->stripPointerCasts());
    auto* text =
        dyn_cast<GlobalVariable>(item->getOperand(1)->stripPointerCasts());
    if (!fn || !text || !text->hasInitializer()) continue;
    auto* data = dyn_cast<ConstantDataArray>(text->getInitializer());
    if (!data || !data->isCString() || data->getAsCString() != kAnnotation)
      continue;
    if (fn->isDeclaration()) continue;
    GlobalVariable* key = nullptr;
    for (Instruction& inst : instructions(fn))
      if (auto* call = dyn_cast<CallInst>(&inst))
        if (Function* callee = call->getCalledFunction();
            callee && callee->getName() == "__kmetal_kernel_key")
          key = dyn_cast<GlobalVariable>(
              call->getArgOperand(0)->stripPointerCasts());
    if (!key) {
      M.getContext().emitError("kokkos-metal: kernel entry " +
                               demangle(fn->getName().str()) +
                               " does not name its key");
      continue;
    }
    out.push_back({fn, key});
  }
  return out;
}

// Functions and globals reachable from the entries: all the device module
// needs, so that a large translation unit is not cloned whole.
DenseSet<const GlobalValue*> reachable(const std::vector<Entry>& entries) {
  DenseSet<const GlobalValue*> seen;
  SmallVector<const Value*, 64> work;
  DenseSet<const Value*> visited;
  for (const Entry& e : entries) work.push_back(e.fn);
  while (!work.empty()) {
    const Value* v = work.pop_back_val();
    if (!visited.insert(v).second) continue;
    if (auto* gv = dyn_cast<GlobalValue>(v)) {
      seen.insert(gv);
      if (auto* fn = dyn_cast<Function>(gv)) {
        for (const Instruction& inst : instructions(fn))
          for (const Value* op : inst.operands())
            if (isa<Constant>(op)) work.push_back(op);
      } else if (auto* var = dyn_cast<GlobalVariable>(gv);
                 var && var->hasInitializer()) {
        work.push_back(var->getInitializer());
      }
    } else if (auto* c = dyn_cast<Constant>(v)) {
      for (const Value* op : c->operands()) work.push_back(op);
    }
  }
  return seen;
}

// Calls and invokes (in a try block) alike: an invoke becomes a call and a
// branch to its normal destination first.
CallInst* as_call(CallBase* cb) {
  if (auto* call = dyn_cast<CallInst>(cb)) return call;
  auto* inv = cast<InvokeInst>(cb);
  SmallVector<Value*, 4> args(inv->args());
  auto* call = CallInst::Create(inv->getFunctionType(), inv->getCalledOperand(),
                                args, "", inv->getIterator());
  call->setCallingConv(inv->getCallingConv());
  BranchInst::Create(inv->getNormalDest(), inv->getIterator());
  inv->getUnwindDest()->removePredecessor(inv->getParent());
  inv->replaceAllUsesWith(call);
  inv->eraseFromParent();
  return call;
}

void replace_calls(Module& M, StringRef name,
                   function_ref<void(CallInst*)> fn) {
  Function* f = M.getFunction(name);
  if (!f) return;
  for (User* u : make_early_inc_range(f->users()))
    if (auto* cb = dyn_cast<CallBase>(u); cb && cb->getCalledOperand() == f)
      fn(as_call(cb));
}

void fold_is_device(Module& M, bool device) {
  replace_calls(M, "__kmetal_is_device", [&](CallInst* call) {
    call->replaceAllUsesWith(ConstantInt::getBool(M.getContext(), device));
    call->eraseFromParent();
  });
}

// Every libm function off but the single-precision maths the emitter maps,
// so that no optimisation introduces a call the GPU cannot make
// (memset_pattern16, __sincosf_stret, ...).
TargetLibraryInfoImpl device_library_info(const Module& M) {
  TargetLibraryInfoImpl tli(M.getTargetTriple());
  tli.disableAllFunctions();
  for (LibFunc f :
       {LibFunc_sqrtf, LibFunc_expf, LibFunc_exp2f, LibFunc_logf, LibFunc_log2f,
        LibFunc_log10f, LibFunc_powf, LibFunc_sinf, LibFunc_cosf, LibFunc_tanf,
        LibFunc_fabsf, LibFunc_floorf, LibFunc_ceilf, LibFunc_truncf,
        LibFunc_roundf, LibFunc_rintf, LibFunc_fminf, LibFunc_fmaxf,
        LibFunc_copysignf, LibFunc_fmodf, LibFunc_tanhf, LibFunc_atan2f,
        // Double precision, emulated (MSLEmitter.cpp).
        LibFunc_sqrt, LibFunc_exp, LibFunc_exp2, LibFunc_log, LibFunc_log2,
        LibFunc_log10, LibFunc_pow, LibFunc_sin, LibFunc_cos, LibFunc_tan,
        LibFunc_fabs, LibFunc_floor, LibFunc_ceil, LibFunc_trunc, LibFunc_round,
        LibFunc_rint, LibFunc_fmin, LibFunc_fmax, LibFunc_copysign,
        LibFunc_fmod, LibFunc_tanh, LibFunc_atan2})
    tli.setAvailable(f);
  return tli;
}

void optimize(Module& D) {
  PipelineTuningOptions tuning;
  tuning.LoopVectorization = false;
  tuning.SLPVectorization  = false;
  PassBuilder pb(nullptr, tuning);
  LoopAnalysisManager lam;
  FunctionAnalysisManager fam;
  CGSCCAnalysisManager cgam;
  ModuleAnalysisManager mam;
  TargetLibraryInfoImpl tli = device_library_info(D);
  fam.registerPass([&] { return TargetLibraryAnalysis(tli); });
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);
  ModulePassManager mpm =
      pb.buildPerModuleDefaultPipeline(OptimizationLevel::O3);
  mpm.run(D, mam);
}

std::string kernel_name(const Function& entry) {
  MD5 hash;
  hash.update(entry.getName());
  MD5::MD5Result r;
  hash.final(r);
  return ("km_" + r.digest().str().substr(0, 16)).str();
}

struct Kernel {
  Entry entry;
  std::string name, source;
  uint32_t flags;  // MetalRuntime::KernelFlags
};

// Builds the device module and returns the MSL for every entry, or reports
// errors and returns nothing.
std::vector<Kernel> compile_device(Module& M,
                                   const std::vector<Entry>& entries) {
  auto keep = reachable(entries);
  ValueToValueMapTy vmap;
  std::unique_ptr<Module> D = CloneModule(
      M, vmap, [&](const GlobalValue* gv) { return keep.count(gv) > 0; });

  for (const char* name :
       {"llvm.global.annotations", "llvm.used", "llvm.compiler.used",
        "llvm.global_ctors", "llvm.global_dtors"})
    if (GlobalVariable* gv = D->getGlobalVariable(name)) gv->eraseFromParent();

  SmallPtrSet<Function*, 16> device_entries;
  for (const Entry& e : entries)
    device_entries.insert(cast<Function>(vmap[e.fn]));

  for (Function& fn : *D) {
    if (fn.isDeclaration()) continue;
    fn.removeFnAttr(Attribute::OptimizeNone);
    fn.removeFnAttr(Attribute::NoInline);
    if (device_entries.count(&fn)) {
      fn.setLinkage(GlobalValue::ExternalLinkage);
      fn.setVisibility(GlobalValue::DefaultVisibility);
      fn.addParamAttr(0, Attribute::NoAlias);
      fn.addParamAttr(0, Attribute::ReadOnly);
    } else {
      fn.setLinkage(GlobalValue::InternalLinkage);
      fn.addFnAttr(Attribute::AlwaysInline);
    }
  }
  for (GlobalVariable& gv : D->globals())
    if (!gv.isDeclaration()) gv.setLinkage(GlobalValue::InternalLinkage);

  fold_is_device(*D, true);
  replace_calls(*D, "__kmetal_kernel_key",
                [](CallInst* call) { call->eraseFromParent(); });

  // Everything is convergent, as in any GPU language: the host compiler did
  // not know that a call could reach a barrier or a shuffle.
  for (Function& fn : *D) {
    fn.addFnAttr(Attribute::Convergent);
    for (Instruction& inst : instructions(fn))
      if (auto* call = dyn_cast<CallBase>(&inst)) call->setConvergent();
  }

  optimize(*D);

  std::vector<Kernel> kernels;
  for (const Entry& e : entries) {
    auto* fn         = cast<Function>(vmap[e.fn]);
    std::string name = kernel_name(*e.fn);
    try {
      kmetal::KernelSource k = kmetal::emit_kernel(*fn, name);
      // The first line names the kernel for people: dumps, profiles.
      k.source = "// " + demangle(e.fn->getName().str()) + "\n" + k.source;
      kernels.push_back(
          {e, name, std::move(k.source),
           uint32_t(k.uses_threadgroup_memory) | uint32_t(k.uses_abort) << 1});
    } catch (const kmetal::Unsupported& err) {
      // Not an error yet: templates instantiate kernels that never run (a
      // View's copy kernel for a type the GPU cannot compute with). Launching
      // it is.
      std::string what = "kernel " + demangle(e.fn->getName().str()) +
                         " cannot run on the GPU: " + err.what();
      M.getContext().diagnose(
          DiagnosticInfoGeneric("kokkos-metal: " + what, DS_Warning));
      kernels.push_back({e, name, std::move(what), kUnsupported});
    }
  }
  return kernels;
}

// Unused local and inline functions left behind by dropping the entries'
// bodies; they may call __kmetal_* functions, which have no host definition.
void drop_dead_functions(Module& M) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (Function& fn : make_early_inc_range(M)) {
      if (fn.isDeclaration() || !fn.use_empty()) continue;
      if (!fn.hasLocalLinkage() && !fn.hasLinkOnceLinkage() &&
          !fn.hasAvailableExternallyLinkage())
        continue;
      fn.eraseFromParent();
      changed = true;
    }
  }
}

void trap_device_calls(Module& M);

void strip_host(Module& M, const std::vector<Entry>& entries) {
  LLVMContext& ctx = M.getContext();
  for (const Entry& e : entries) {
    e.fn->deleteBody();
    e.fn->setLinkage(GlobalValue::LinkOnceODRLinkage);
    BasicBlock* bb = BasicBlock::Create(ctx, "", e.fn);
    ReturnInst::Create(ctx, bb);
  }
  drop_dead_functions(M);
  trap_device_calls(M);
}

// Device-only code still referenced on the host (behind a folded
// __kmetal_is_device(), say) traps if it ever runs, rather than leaving
// undefined __kmetal_* symbols for the linker.
void trap_device_calls(Module& M) {
  for (Function& fn : M) {
    if (!fn.getName().starts_with("__kmetal_") || !fn.isDeclaration()) continue;
    for (User* u : make_early_inc_range(fn.users())) {
      auto* cb = dyn_cast<CallBase>(u);
      if (!cb || cb->getCalledOperand() != &fn) continue;
      CallInst* call = as_call(cb);
      IRBuilder<> b(call);
      b.CreateIntrinsic(Intrinsic::trap, {});
      if (!call->getType()->isVoidTy())
        call->replaceAllUsesWith(PoisonValue::get(call->getType()));
      call->eraseFromParent();
    }
  }
}

void add_registration(Module& M, const std::vector<Kernel>& kernels) {
  LLVMContext& ctx = M.getContext();
  auto* ptr        = PointerType::getUnqual(ctx);
  auto* i32        = Type::getInt32Ty(ctx);
  auto string      = [&](StringRef text, const Twine& name) {
    Constant* data = ConstantDataArray::getString(ctx, text);
    auto* gv       = new GlobalVariable(M, data->getType(), true,
                                             GlobalValue::PrivateLinkage, data, name);
    gv->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    return gv;
  };
  StructType* desc  = StructType::get(ctx, {ptr, ptr, ptr, ptr, i32});
  Constant* prelude = string(kmetal::prelude(), "__kmetal_prelude");
  std::vector<Constant*> items;
  for (const Kernel& k : kernels)
    items.push_back(ConstantStruct::get(
        desc,
        {k.entry.key, string(k.name, "__kmetal_name"), prelude,
         string(k.source, "__kmetal_source"), ConstantInt::get(i32, k.flags)}));
  auto* array_type = ArrayType::get(desc, items.size());
  auto* table      = new GlobalVariable(
      M, array_type, true, GlobalValue::PrivateLinkage,
      ConstantArray::get(array_type, items), "__kmetal_kernels");

  FunctionCallee reg = M.getOrInsertFunction(
      "kokkos_metal_register_kernels",
      FunctionType::get(Type::getVoidTy(ctx), {ptr, i32}, false));
  Function* ctor =
      Function::Create(FunctionType::get(Type::getVoidTy(ctx), false),
                       GlobalValue::InternalLinkage, "__kmetal_register", M);
  IRBuilder<> b(BasicBlock::Create(ctx, "", ctor));
  b.CreateCall(reg, {table, ConstantInt::get(i32, items.size())});
  b.CreateRetVoid();
  appendToGlobalCtors(M, ctor, 0);
}

struct KokkosMetalPass : PassInfoMixin<KokkosMetalPass> {
  PreservedAnalyses run(Module& M, ModuleAnalysisManager&) {
    std::vector<Entry> entries = find_entries(M);
    if (entries.empty()) {
      if (!M.getFunction("__kmetal_is_device")) return PreservedAnalyses::all();
      fold_is_device(M, false);
      trap_device_calls(M);
      return PreservedAnalyses::none();
    }
    std::vector<Kernel> kernels = compile_device(M, entries);
    fold_is_device(M, false);
    strip_host(M, entries);
    if (!kernels.empty()) add_registration(M, kernels);
    if (getenv("KOKKOS_METAL_PLUGIN_DUMP"))
      for (const Kernel& k : kernels)
        errs() << "// " << demangle(k.entry.fn->getName().str()) << "\n"
               << k.source << "\n";
    return PreservedAnalyses::none();
  }
  static bool isRequired() { return true; }
};

}  // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "KokkosMetal", "0.1", [](PassBuilder& pb) {
            pb.registerPipelineStartEPCallback(
                [](ModulePassManager& mpm, OptimizationLevel) {
                  mpm.addPass(KokkosMetalPass());
                });
          }};
}
