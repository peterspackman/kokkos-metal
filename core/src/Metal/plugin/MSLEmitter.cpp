// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// LLVM IR -> Metal Shading Language for one kernel.
//
// Every SSA value becomes a function-scope variable. MSL has no goto, so the
// control flow is rebuilt from the dominator tree following Ramsey, "Beyond
// Relooper" (ICFP 2022): a loop header opens `while (true)`, a merge node
// closes a `switch (0) { default: ... }` block that a forward branch leaves
// with `break`, and a branch past more than the innermost construct goes
// through the __jmp variable, which each construct checks on its way out.
//
// Pointers are untyped `uchar*` in one of Metal's address spaces, inferred
// over the function; memory is accessed through may_alias typedefs, since
// the IR's access types need not match any C++ type.

#include "MSLEmitter.hpp"

#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/GetElementPtrTypeIterator.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/Support/raw_ostream.h>

#include <cstring>
#include <map>
#include <set>

using namespace llvm;

namespace kmetal {
namespace {

enum class Space : uint8_t { Unknown, Device, Constant, Thread, Threadgroup };

const char* space_name(Space s) {
  switch (s) {
    case Space::Constant: return "constant";
    case Space::Thread: return "thread";
    case Space::Threadgroup: return "threadgroup";
    default: return "device";
  }
}

[[noreturn]] void unsupported(const Twine& what, const Value* where = nullptr) {
  std::string msg = what.str();
  if (const auto* I = dyn_cast_or_null<Instruction>(where)) {
    if (const DebugLoc& loc = I->getDebugLoc())
      msg += " (" + loc->getFilename().str() + ":" +
             std::to_string(loc->getLine()) + ")";
    raw_string_ostream os(msg);
    os << "\n  at:";
    I->print(os);
  }
  throw Unsupported(msg);
}

std::string demangled(StringRef name) { return demangle(name.str()); }

// Metal's default maths is its fast variant, wrong well beyond rounding
// (tanh(50) is NaN, tanh(1e-6) 1.3% off): the precise one where it exists.
std::string precise(const std::string& f) {
  static const std::set<std::string, std::less<>> names = {
      "sqrt",  "rsqrt", "sin",  "cos",   "tan",  "asin",  "acos",
      "atan",  "atan2", "sinh", "cosh",  "tanh", "asinh", "acosh",
      "atanh", "exp",   "exp2", "exp10", "log",  "log2",  "log10",
      "pow",   "fmod",  "cbrt", "ldexp"};
  return names.count(f) ? "precise::" + f : f;
}

// The unsigned MSL type an integer is held in, and its signed twin.
std::string int_type(unsigned bits, bool is_signed = false) {
  switch (bits) {
    case 1: return "bool";
    case 8: return is_signed ? "char" : "uchar";
    case 16: return is_signed ? "short" : "ushort";
    case 32: return is_signed ? "int" : "uint";
    case 64: return is_signed ? "long" : "ulong";
  }
  unsupported("integers of " + Twine(bits) + " bits");
}

std::string float_type(Type* t) {
  if (t->isFloatTy()) return "float";
  if (t->isHalfTy()) return "half";
  if (t->isBFloatTy()) return "bfloat";
  if (t->isDoubleTy()) return "ulong";  // bits; see the prelude's fp64
  unsupported("floating-point type");
}

// The may_alias typedef (see the prelude) memory of type t is accessed by.
std::string memory_type(Type* t) {
  if (t->isPointerTy()) return "km_u64";
  if (t->isIntegerTy(1)) return "km_bool";
  if (t->isIntegerTy()) {
    unsigned b = t->getIntegerBitWidth();
    if (b == 8 || b == 16 || b == 32 || b == 64)
      return "km_u" + std::to_string(b);
  }
  if (t->isFloatTy()) return "km_f32";
  if (t->isHalfTy()) return "km_f16";
  if (t->isBFloatTy()) return "km_bf16";
  if (t->isDoubleTy()) return "km_u64";
  std::string s;
  raw_string_ostream os(s);
  t->print(os);
  unsupported("memory access of type " + s);
}

class Emitter {
 public:
  explicit Emitter(Function& f)
      : F(f), DL(f.getParent()->getDataLayout()), DT(f) {}

  KernelSource run(const std::string& name) {
    if (F.arg_size() != 1) unsupported("kernel entry must take one argument");
    names[F.getArg(0)]  = "__arg0";
    spaces[F.getArg(0)] = Space::Constant;
    analyze_cfg();
    infer_spaces();
    declare_values();
    do_tree(&F.getEntryBlock());

    std::string out = globals_out;
    out += "kernel void " + name + "(constant uchar* __arg0 [[buffer(0)]]";
    static const std::pair<const char*, const char*> kBuiltins[] = {
        {"__tpig", "uint3 __tpig [[thread_position_in_grid]]"},
        {"__tptg", "uint3 __tptg [[thread_position_in_threadgroup]]"},
        {"__tgpig", "uint3 __tgpig [[threadgroup_position_in_grid]]"},
        {"__ntptg", "uint3 __ntptg [[threads_per_threadgroup]]"},
        {"__ntgpg", "uint3 __ntgpg [[threadgroups_per_grid]]"},
        {"__lane", "uint __lane [[thread_index_in_simdgroup]]"},
        {"__sgidx", "uint __sgidx [[simdgroup_index_in_threadgroup]]"},
        {"__sgsize", "uint __sgsize [[threads_per_simdgroup]]"},
    };
    for (auto& [id, decl] : kBuiltins)
      if (builtins.count(id)) out += std::string(",\n    ") + decl;
    if (uses_tg) out += ",\n    threadgroup uchar* __tgmem [[threadgroup(0)]]";
    if (uses_abort) out += ",\n    device atomic_uint* __abort [[buffer(1)]]";
    out += ") {\n";
    out += decls_out;
    if (uses_jmp) out += "  int __jmp = 0;\n";
    out += body_out;
    out += "}\n";
    return {out, uses_tg, uses_abort};
  }

 private:
  Function& F;
  const DataLayout& DL;
  DominatorTree DT;

  DenseMap<const BasicBlock*, unsigned> rpo;
  SmallPtrSet<const BasicBlock*, 16> loop_headers, merge_nodes;

  DenseMap<const Value*, Space> spaces;
  DenseMap<const Value*, std::string> names;
  DenseMap<const GlobalVariable*, std::string> global_names;
  std::set<std::string> builtins;
  bool uses_jmp = false, uses_tg = false, uses_abort = false;
  int next_value = 0;

  std::string globals_out, decls_out, body_out;
  int depth = 1;

  struct Frame {
    enum Kind { Loop, Block } kind;
    const BasicBlock* bb;
    int id;
    std::vector<int> pending;  // multi-level jumps that left through here
  };
  std::vector<Frame> frames;
  int next_frame = 1;

  void line(const std::string& s) {
    body_out.append(2 * depth, ' ');
    body_out += s;
    body_out += '\n';
  }

  // --- control flow -------------------------------------------------------

  void analyze_cfg() {
    ReversePostOrderTraversal<Function*> order(&F);
    unsigned i = 0;
    for (BasicBlock* bb : order) rpo[bb] = i++;
    for (BasicBlock* bb : order) {
      for (BasicBlock* succ : successors(bb)) {
        if (rpo[succ] > rpo[bb]) continue;
        if (!DT.dominates(succ, bb))
          unsupported("irreducible control flow", bb->getTerminator());
        loop_headers.insert(succ);
      }
      SmallPtrSet<const BasicBlock*, 4> forward;
      for (BasicBlock* pred : predecessors(bb))
        if (rpo.count(pred) && rpo[pred] < rpo[bb]) forward.insert(pred);
      if (forward.size() >= 2) merge_nodes.insert(bb);
    }
  }

  std::vector<const BasicBlock*> merge_children(const BasicBlock* x) {
    std::vector<const BasicBlock*> out;
    for (DomTreeNode* child : DT.getNode(x)->children())
      if (merge_nodes.count(child->getBlock()))
        out.push_back(child->getBlock());
    // The latest in reverse postorder is the outermost block.
    std::sort(out.begin(), out.end(),
              [&](const BasicBlock* a, const BasicBlock* b) {
                return rpo[a] > rpo[b];
              });
    return out;
  }

  void do_tree(const BasicBlock* x) {
    auto merges = merge_children(x);
    if (loop_headers.count(x)) {
      open(Frame::Loop, x, "while (true) {");
      node_within(x, merges, 0);
      close("}");
    } else {
      node_within(x, merges, 0);
    }
  }

  void node_within(const BasicBlock* x,
                   const std::vector<const BasicBlock*>& merges, size_t i) {
    if (i < merges.size()) {
      open(Frame::Block, merges[i], "switch (0) { default: {");
      node_within(x, merges, i + 1);
      close("}}");
      do_tree(merges[i]);
      return;
    }
    for (const Instruction& inst : *x)
      if (!isa<PHINode>(inst) && !inst.isTerminator()) emit(inst);
    emit_terminator(x->getTerminator());
  }

  void open(Frame::Kind kind, const BasicBlock* bb, const char* text) {
    line(text);
    ++depth;
    frames.push_back({kind, bb, next_frame++, {}});
  }

  void close(const char* text) {
    Frame f = std::move(frames.back());
    frames.pop_back();
    --depth;
    line(text);
    for (int id : f.pending) {
      line("if (__jmp == " + std::to_string(id) + ") {");
      ++depth;
      line("__jmp = 0;");
      jump(frame_index(id));
      --depth;
      line("}");
    }
  }

  int frame_index(int id) {
    for (int i = int(frames.size()) - 1; i >= 0; --i)
      if (frames[i].id == id) return i;
    unsupported("internal error: jump to a closed construct");
  }

  int innermost(bool loops_only) {
    for (int i = int(frames.size()) - 1; i >= 0; --i)
      if (!loops_only || frames[i].kind == Frame::Loop) return i;
    return -1;
  }

  // Continue the loop or leave the block of frames[target].
  void jump(int target) {
    bool loop = frames[target].kind == Frame::Loop;
    if (target == innermost(loop)) {
      line(loop ? "continue;" : "break;");
      return;
    }
    uses_jmp  = true;
    int inner = innermost(false);
    frames[inner].pending.push_back(frames[target].id);
    line("__jmp = " + std::to_string(frames[target].id) + ";");
    line("break;");
  }

  int find_frame(Frame::Kind kind, const BasicBlock* bb) {
    for (int i = int(frames.size()) - 1; i >= 0; --i)
      if (frames[i].kind == kind && frames[i].bb == bb) return i;
    unsupported("internal error: no enclosing construct for a branch");
  }

  void do_branch(const BasicBlock* from, const BasicBlock* to) {
    phi_copies(from, to);
    if (rpo[to] <= rpo[from])
      jump(find_frame(Frame::Loop, to));
    else if (merge_nodes.count(to))
      jump(find_frame(Frame::Block, to));
    else
      do_tree(to);
  }

  void phi_copies(const BasicBlock* from, const BasicBlock* to) {
    std::vector<std::pair<const PHINode*, std::string>> copies;
    bool reads_phi = false;
    for (const PHINode& phi : to->phis()) {
      const Value* v = phi.getIncomingValueForBlock(from);
      if (auto* p = dyn_cast<PHINode>(v); p && p->getParent() == to)
        reads_phi = true;
      copies.push_back({&phi, value(v)});
    }
    if (!reads_phi) {
      for (auto& [phi, v] : copies) line(names[phi] + " = " + v + ";");
      return;
    }
    // A parallel copy: read every incoming value before any phi is written.
    line("{");
    ++depth;
    for (size_t i = 0; i < copies.size(); ++i)
      line(type_of(copies[i].first) + " __t" + std::to_string(i) + " = " +
           copies[i].second + ";");
    for (size_t i = 0; i < copies.size(); ++i)
      line(names[copies[i].first] + " = __t" + std::to_string(i) + ";");
    --depth;
    line("}");
  }

  void emit_terminator(const Instruction* t) {
    const BasicBlock* bb = t->getParent();
    if (auto* br = dyn_cast<BranchInst>(t)) {
      if (br->isUnconditional() || br->getSuccessor(0) == br->getSuccessor(1)) {
        do_branch(bb, br->getSuccessor(0));
        return;
      }
      line("if (" + value(br->getCondition()) + ") {");
      ++depth;
      do_branch(bb, br->getSuccessor(0));
      --depth;
      line("} else {");
      ++depth;
      do_branch(bb, br->getSuccessor(1));
      --depth;
      line("}");
      return;
    }
    if (auto* sw = dyn_cast<SwitchInst>(t)) {
      std::map<const BasicBlock*, std::string> conds;
      std::vector<const BasicBlock*> order;
      std::string v = value(sw->getCondition());
      for (auto& c : sw->cases()) {
        const BasicBlock* dest = c.getCaseSuccessor();
        if (dest == sw->getDefaultDest()) continue;
        if (!conds.count(dest)) order.push_back(dest);
        std::string& cond = conds[dest];
        cond +=
            (cond.empty() ? "" : " || ") + v + " == " + value(c.getCaseValue());
      }
      for (const BasicBlock* dest : order) {
        line((dest == order.front() ? "if (" : "} else if (") + conds[dest] +
             ") {");
        ++depth;
        do_branch(bb, dest);
        --depth;
      }
      if (!order.empty()) line("} else {");
      ++depth;
      do_branch(bb, sw->getDefaultDest());
      --depth;
      if (!order.empty()) line("}");
      return;
    }
    if (isa<ReturnInst>(t) || isa<UnreachableInst>(t)) {
      line("return;");
      return;
    }
    unsupported("terminator", t);
  }

  // --- address spaces -----------------------------------------------------

  Space space_of(const Value* v) {
    auto it = spaces.find(v);
    if (it != spaces.end()) return it->second;
    if (isa<GlobalVariable>(v)) return Space::Constant;
    if (auto* ce = dyn_cast<ConstantExpr>(v)) {
      if (ce->getOpcode() == Instruction::IntToPtr) return Space::Device;
      if (ce->getNumOperands() > 0 &&
          ce->getOperand(0)->getType()->isPointerTy())
        return space_of(ce->getOperand(0));
    }
    return Space::Unknown;
  }

  static bool join(DenseMap<const Value*, Space>& map, const Value* key,
                   Space s, const Value* where) {
    if (s == Space::Unknown) return false;
    Space& cur = map[key];
    if (cur == s) return false;
    if (cur != Space::Unknown)
      unsupported(Twine("a pointer that may refer to both ") + space_name(cur) +
                      " and " + space_name(s) + " memory",
                  where);
    cur = s;
    return true;
  }

  Space pointee_pointer_space(const Value* ptr,
                              const DenseMap<const Value*, Space>& stored) {
    SmallVector<const Value*, 4> objects;
    getUnderlyingObjects(ptr, objects);
    Space s = Space::Unknown;
    for (const Value* o : objects) {
      Space here =
          Space::Device;  // pointers kept in memory are device addresses
      if (isa<AllocaInst>(o)) {
        auto it = stored.find(o);
        here    = it == stored.end() ? Space::Unknown : it->second;
      }
      if (here == Space::Unknown) continue;
      if (s != Space::Unknown && s != here)
        unsupported(
            "a pointer loaded from memory holding pointers to different spaces",
            ptr);
      s = here;
    }
    return s;
  }

  void infer_spaces() {
    DenseMap<const Value*, Space>
        stored;  // alloca -> space of pointers stored in it
    bool changed = true;
    while (changed) {
      changed = false;
      for (const Instruction& inst : instructions(F)) {
        if (auto* st = dyn_cast<StoreInst>(&inst);
            st && st->getValueOperand()->getType()->isPointerTy()) {
          SmallVector<const Value*, 4> objects;
          getUnderlyingObjects(st->getPointerOperand(), objects);
          for (const Value* o : objects)
            if (isa<AllocaInst>(o))
              changed |=
                  join(stored, o, space_of(st->getValueOperand()), &inst);
        }
        if (auto* mt = dyn_cast<MemTransferInst>(&inst)) {
          Space src = pointee_pointer_space(mt->getSource(), stored);
          SmallVector<const Value*, 4> objects;
          getUnderlyingObjects(mt->getDest(), objects);
          for (const Value* o : objects)
            if (isa<AllocaInst>(o)) changed |= join(stored, o, src, &inst);
        }
        if (inst.getType()->isPointerTy())
          changed |= join(spaces, &inst, compute_space(inst, stored), &inst);
      }
    }
    for (const Instruction& inst : instructions(F))
      if (inst.getType()->isPointerTy() && space_of(&inst) == Space::Unknown)
        spaces[&inst] = Space::Device;
  }

  Space compute_space(const Instruction& inst,
                      const DenseMap<const Value*, Space>& stored) {
    if (isa<AllocaInst>(inst)) return Space::Thread;
    if (isa<IntToPtrInst>(inst)) return Space::Device;
    if (auto* gep = dyn_cast<GetElementPtrInst>(&inst))
      return space_of(gep->getPointerOperand());
    if (isa<FreezeInst>(inst) || isa<CastInst>(inst))
      return space_of(inst.getOperand(0));
    if (auto* sel = dyn_cast<SelectInst>(&inst)) {
      Space a = space_of(sel->getTrueValue()),
            b = space_of(sel->getFalseValue());
      if (a != Space::Unknown && b != Space::Unknown && a != b)
        unsupported("a pointer that may refer to two address spaces", &inst);
      return a != Space::Unknown ? a : b;
    }
    if (auto* phi = dyn_cast<PHINode>(&inst)) {
      Space s = Space::Unknown;
      for (const Value* v : phi->incoming_values()) {
        Space here = space_of(v);
        if (here == Space::Unknown) continue;
        if (s != Space::Unknown && s != here)
          unsupported("a pointer that may refer to two address spaces", &inst);
        s = here;
      }
      return s;
    }
    if (auto* ld = dyn_cast<LoadInst>(&inst))
      return pointee_pointer_space(ld->getPointerOperand(), stored);
    if (auto* call = dyn_cast<CallInst>(&inst)) {
      if (Function* fn = call->getCalledFunction()) {
        if (fn->getName() == "__kmetal_threadgroup_memory")
          return Space::Threadgroup;
        if (fn->isIntrinsic() && call->arg_size() > 0)
          return space_of(call->getArgOperand(0));
      }
    }
    return Space::Device;
  }

  // --- values -------------------------------------------------------------

  std::string type_of(const Value* v) {
    Type* t = v->getType();
    if (t->isPointerTy())
      return std::string(space_name(space_of(v))) + " uchar*";
    return scalar_type(t);
  }

  std::string scalar_type(Type* t) {
    if (t->isIntegerTy()) return int_type(t->getIntegerBitWidth());
    if (t->isFloatingPointTy()) return float_type(t);
    std::string s;
    raw_string_ostream os(s);
    t->print(os);
    unsupported("values of type " + s);
  }

  void declare_values() {
    for (Instruction& inst : instructions(F)) {
      if (inst.getType()->isVoidTy()) continue;
      std::string name = "v" + std::to_string(next_value++);
      if (auto* a = dyn_cast<AllocaInst>(&inst)) {
        auto size = a->getAllocationSize(DL);
        if (!size || a->getParent() != &F.getEntryBlock())
          unsupported("a variable-sized stack allocation", &inst);
        uint64_t align = std::max<uint64_t>(a->getAlign().value(), 4);
        decls_out +=
            "  thread uchar " + name + "_mem[" +
            std::to_string(std::max<uint64_t>(size->getFixedValue(), 1)) +
            "] __attribute__((aligned(" + std::to_string(align) + ")));\n";
        names[a] = "((thread uchar*)" + name + "_mem)";
        continue;
      }
      names[&inst] = name;
      if (auto* st = dyn_cast<StructType>(inst.getType())) {
        for (unsigned i = 0; i < st->getNumElements(); ++i)
          decls_out += "  " + scalar_type(st->getElementType(i)) + " " + name +
                       "_" + std::to_string(i) + ";\n";
        continue;
      }
      decls_out += "  " + type_of(&inst) + " " + name + ";\n";
    }
  }

  std::string value(const Value* v) {
    auto it = names.find(v);
    if (it != names.end()) return it->second;
    if (auto* ci = dyn_cast<ConstantInt>(v)) {
      unsigned bits = ci->getBitWidth();
      if (bits == 1) return ci->isOne() ? "true" : "false";
      std::string n = std::to_string(ci->getZExtValue());
      if (bits == 32) return n + "u";
      if (bits == 64) return n + "ul";
      return "(" + int_type(bits) + ")" + n;
    }
    if (auto* cf = dyn_cast<ConstantFP>(v)) {
      Type* t        = cf->getType();
      std::string ft = float_type(t);
      APInt bits     = cf->getValueAPF().bitcastToAPInt();
      if (t->isDoubleTy()) return std::to_string(bits.getZExtValue()) + "ul";
      if (t->isFloatTy()) {
        float f = cf->getValueAPF().convertToFloat();
        if (f == (float)(int)f && std::abs(f) < 16777216.0f &&
            !(f == 0 && std::signbit(f)))
          return std::to_string((int)f) + ".0f";
        char buf[48];
        std::snprintf(buf, sizeof buf, "as_type<float>(0x%08xu)",
                      (unsigned)bits.getZExtValue());
        return buf;
      }
      char buf[48];
      std::snprintf(buf, sizeof buf, "as_type<%s>((ushort)0x%04x)", ft.c_str(),
                    (unsigned)bits.getZExtValue());
      return buf;
    }
    if (isa<ConstantPointerNull>(v)) return "0";
    if (isa<UndefValue>(v)) {  // and poison
      Type* t = v->getType();
      if (t->isPointerTy()) return "0";
      if (t->isIntegerTy(1)) return "false";
      return "(" + scalar_type(t) + ")0";
    }
    if (auto* gv = dyn_cast<GlobalVariable>(v))
      return "((constant uchar*)" + global(gv) + ")";
    if (auto* ce = dyn_cast<ConstantExpr>(v)) {
      switch (ce->getOpcode()) {
        case Instruction::GetElementPtr: return gep(cast<GEPOperator>(ce));
        case Instruction::PtrToInt:
          return "((" + int_type(ce->getType()->getIntegerBitWidth()) +
                 ")(ulong)" + value(ce->getOperand(0)) + ")";
        case Instruction::IntToPtr:
          return "((device uchar*)(ulong)" + value(ce->getOperand(0)) + ")";
        case Instruction::BitCast:
        case Instruction::AddrSpaceCast: return value(ce->getOperand(0));
        default: break;
      }
    }
    if (isa<Function>(v)) unsupported("function pointers in a kernel");
    std::string s;
    raw_string_ostream os(s);
    v->print(os);
    unsupported("constant " + s);
  }

  // A read-only global, copied into the kernel's source as a constant array.
  std::string global(const GlobalVariable* gv) {
    auto it = global_names.find(gv);
    if (it != global_names.end()) return it->second;
    if (!gv->isConstant() || !gv->hasInitializer())
      unsupported("the kernel uses the mutable global variable '" +
                  demangled(gv->getName()) +
                  "'; only constants can be read on the GPU");
    std::vector<uint8_t> bytes(DL.getTypeAllocSize(gv->getValueType()), 0);
    serialize(gv->getInitializer(), bytes.data(), gv);
    std::string name = "g" + std::to_string(global_names.size());
    globals_out += "constant uchar " + name + "[" +
                   std::to_string(std::max<size_t>(bytes.size(), 1)) +
                   "] __attribute__((aligned(16))) = {";
    for (size_t i = 0; i < bytes.size(); ++i)
      globals_out += (i ? "," : "") + std::to_string(bytes[i]);
    globals_out += "};\n";
    return global_names[gv] = name;
  }

  void serialize(const Constant* c, uint8_t* out, const GlobalVariable* gv) {
    if (isa<ConstantAggregateZero>(c) || isa<UndefValue>(c)) return;
    if (auto* ci = dyn_cast<ConstantInt>(c)) {
      uint64_t v = ci->getZExtValue();
      std::memcpy(out, &v, DL.getTypeStoreSize(c->getType()));
      return;
    }
    if (auto* cf = dyn_cast<ConstantFP>(c)) {
      uint64_t v = cf->getValueAPF().bitcastToAPInt().getZExtValue();
      std::memcpy(out, &v, DL.getTypeStoreSize(c->getType()));
      return;
    }
    if (auto* cds = dyn_cast<ConstantDataSequential>(c)) {
      StringRef raw = cds->getRawDataValues();
      std::memcpy(out, raw.data(), raw.size());
      return;
    }
    if (auto* cs = dyn_cast<ConstantStruct>(c)) {
      const StructLayout* layout = DL.getStructLayout(cs->getType());
      for (unsigned i = 0; i < cs->getNumOperands(); ++i)
        serialize(cs->getOperand(i), out + layout->getElementOffset(i), gv);
      return;
    }
    if (auto* ca = dyn_cast<ConstantArray>(c)) {
      uint64_t stride = DL.getTypeAllocSize(ca->getType()->getElementType());
      for (unsigned i = 0; i < ca->getNumOperands(); ++i)
        serialize(ca->getOperand(i), out + i * stride, gv);
      return;
    }
    unsupported("the initializer of constant '" + demangled(gv->getName()) +
                "' (pointers are not supported in GPU constants)");
  }

  std::string gep(const GEPOperator* g) {
    if (g->getType()->isVectorTy()) unsupported("vector getelementptr");
    std::string base = value(g->getPointerOperand());
    if (isa<ConstantPointerNull>(g->getPointerOperand()))
      base = std::string("((") + space_name(space_of(g)) + " uchar*)0)";
    int64_t offset = 0;
    std::string terms;
    for (auto it = gep_type_begin(g), end = gep_type_end(g); it != end; ++it) {
      const Value* idx = it.getOperand();
      if (StructType* st = it.getStructTypeOrNull()) {
        offset += DL.getStructLayout(st)->getElementOffset(
            cast<ConstantInt>(idx)->getZExtValue());
        continue;
      }
      int64_t stride = it.getSequentialElementStride(DL);
      if (auto* ci = dyn_cast<ConstantInt>(idx)) {
        offset += ci->getSExtValue() * stride;
        continue;
      }
      unsigned bits = idx->getType()->getIntegerBitWidth();
      std::string i = "(long)(" + int_type(bits, true) + ")" + value(idx);
      terms += " + " + (stride == 1 ? i : i + " * " + std::to_string(stride));
    }
    if (offset) terms += " + (long)" + std::to_string(offset);
    return terms.empty() ? base : "(" + base + terms + ")";
  }

  std::string signed_value(const Value* v) {
    return "(" + int_type(v->getType()->getIntegerBitWidth(), true) + ")" +
           value(v);
  }

  std::string memory(const Value* ptr, Type* t) {
    return std::string("(*(") + space_name(space_of(ptr)) + " " +
           memory_type(t) + "*)" + value(ptr) + ")";
  }

  // --- instructions -------------------------------------------------------

  void assign(const Instruction& inst, const std::string& expr) {
    line(names[&inst] + " = " + expr + ";");
  }

  void emit(const Instruction& inst) {
    if (isa<AllocaInst>(inst)) return;
    if (auto* bo = dyn_cast<BinaryOperator>(&inst))
      return assign(inst, binary(*bo));
    if (auto* uo = dyn_cast<UnaryOperator>(&inst);
        uo && uo->getOpcode() == Instruction::FNeg)
      return assign(inst, uo->getType()->isDoubleTy()
                              ? "km_dneg(" + value(uo->getOperand(0)) + ")"
                              : "-" + value(uo->getOperand(0)));
    if (auto* cmp = dyn_cast<ICmpInst>(&inst)) return assign(inst, icmp(*cmp));
    if (auto* cmp = dyn_cast<FCmpInst>(&inst)) return assign(inst, fcmp(*cmp));
    if (auto* cast = dyn_cast<CastInst>(&inst))
      return assign(inst, cast_expr(*cast));
    if (auto* g = dyn_cast<GetElementPtrInst>(&inst))
      return assign(inst, gep(cast<GEPOperator>(g)));
    if (auto* ld = dyn_cast<LoadInst>(&inst)) {
      if (ld->isAtomic()) return assign(inst, atomic_load(*ld));
      std::string m = memory(ld->getPointerOperand(), ld->getType());
      if (ld->getType()->isPointerTy()) m = "(" + type_of(ld) + ")" + m;
      return assign(inst, m);
    }
    if (auto* st = dyn_cast<StoreInst>(&inst)) {
      const Value* v  = st->getValueOperand();
      std::string val = value(v);
      if (v->getType()->isPointerTy()) val = "(ulong)" + val;
      if (st->isAtomic() && is_wide(v->getType())) {
        line(volatile_word(st->getPointerOperand()) + " = " + val + ";");
        return;
      }
      if (st->isAtomic()) {
        line(std::string("atomic_store_explicit((") +
             space_name(space_of(st->getPointerOperand())) + " " +
             atomic_type(v->getType(), st) + "*)" +
             value(st->getPointerOperand()) + ", " + val +
             ", memory_order_relaxed);");
        return;
      }
      line(memory(st->getPointerOperand(), v->getType()) + " = " + val + ";");
      return;
    }
    if (auto* sel = dyn_cast<SelectInst>(&inst))
      return assign(inst, value(sel->getCondition()) + " ? " +
                              value(sel->getTrueValue()) + " : " +
                              value(sel->getFalseValue()));
    if (isa<FreezeInst>(inst)) return assign(inst, value(inst.getOperand(0)));
    if (auto* call = dyn_cast<CallInst>(&inst)) return emit_call(*call);
    if (auto* rmw = dyn_cast<AtomicRMWInst>(&inst))
      return assign(inst, atomic_rmw(*rmw));
    if (auto* cx = dyn_cast<AtomicCmpXchgInst>(&inst)) return emit_cmpxchg(*cx);
    if (isa<FenceInst>(inst)) {
      line(
          "atomic_thread_fence(mem_flags::mem_device | "
          "mem_flags::mem_threadgroup, "
          "memory_order_seq_cst, thread_scope_device);");
      return;
    }
    if (auto* ev = dyn_cast<ExtractValueInst>(&inst)) {
      if (ev->getNumIndices() != 1 ||
          !isa<StructType>(ev->getAggregateOperand()->getType()) ||
          !names.count(ev->getAggregateOperand()))
        unsupported("extractvalue", &inst);
      return assign(inst, names[ev->getAggregateOperand()] + "_" +
                              std::to_string(ev->getIndices()[0]));
    }
    unsupported("instruction", &inst);
  }

  std::string binary(const BinaryOperator& bo) {
    std::string a = value(bo.getOperand(0)), b = value(bo.getOperand(1));
    Type* t = bo.getType();
    if (t->isVectorTy()) unsupported("vector arithmetic", &bo);
    if (t->isDoubleTy()) {
      switch (bo.getOpcode()) {
        case Instruction::FAdd: return "km_dadd(" + a + ", " + b + ")";
        case Instruction::FSub: return "km_dsub(" + a + ", " + b + ")";
        case Instruction::FMul: return "km_dmul(" + a + ", " + b + ")";
        case Instruction::FDiv: return "km_ddiv(" + a + ", " + b + ")";
        default: unsupported("this double-precision operation", &bo);
      }
    }
    switch (bo.getOpcode()) {
      case Instruction::FAdd: return a + " + " + b;
      case Instruction::FSub: return a + " - " + b;
      case Instruction::FMul: return a + " * " + b;
      case Instruction::FDiv: return a + " / " + b;
      case Instruction::FRem: return "fmod(" + a + ", " + b + ")";
      default: break;
    }
    unsigned bits = t->getIntegerBitWidth();
    std::string T = int_type(bits), S = int_type(bits, true);
    if (bits == 1) {
      switch (bo.getOpcode()) {
        case Instruction::And:
        case Instruction::Mul: return a + " && " + b;
        case Instruction::Or: return a + " || " + b;
        case Instruction::Xor:
        case Instruction::Add:
        case Instruction::Sub: return a + " != " + b;
        default: unsupported("i1 arithmetic", &bo);
      }
    }
    // uchar and ushort promote to int, where a product can overflow.
    bool narrow = bits < 32;
    auto wrap   = [&](const std::string& op) {
      return narrow ? "(" + T + ")((uint)" + a + " " + op + " (uint)" + b + ")"
                      : a + " " + op + " " + b;
    };
    switch (bo.getOpcode()) {
      case Instruction::Add: return wrap("+");
      case Instruction::Sub: return wrap("-");
      case Instruction::Mul: return wrap("*");
      case Instruction::Shl: return wrap("<<");
      case Instruction::And: return wrap("&");
      case Instruction::Or: return wrap("|");
      case Instruction::Xor: return wrap("^");
      case Instruction::UDiv: return a + " / " + b;
      case Instruction::URem: return a + " % " + b;
      case Instruction::LShr: return a + " >> " + b;
      case Instruction::SDiv:
        return "(" + T + ")((" + S + ")" + a + " / (" + S + ")" + b + ")";
      case Instruction::SRem:
        return "(" + T + ")((" + S + ")" + a + " % (" + S + ")" + b + ")";
      case Instruction::AShr:
        return "(" + T + ")((" + S + ")" + a + " >> " + b + ")";
      default: unsupported("binary operator", &bo);
    }
  }

  std::string icmp(const ICmpInst& cmp) {
    const Value* x = cmp.getOperand(0);
    if (x->getType()->isVectorTy()) unsupported("vector compare", &cmp);
    std::string a = value(x), b = value(cmp.getOperand(1));
    if (x->getType()->isPointerTy()) {
      a = "(ulong)" + a;
      b = "(ulong)" + b;
    } else if (cmp.isSigned()) {
      a = signed_value(x);
      b = signed_value(cmp.getOperand(1));
    }
    const char* op = nullptr;
    switch (cmp.getPredicate()) {
      case CmpInst::ICMP_EQ: op = "=="; break;
      case CmpInst::ICMP_NE: op = "!="; break;
      case CmpInst::ICMP_UGT:
      case CmpInst::ICMP_SGT: op = ">"; break;
      case CmpInst::ICMP_UGE:
      case CmpInst::ICMP_SGE: op = ">="; break;
      case CmpInst::ICMP_ULT:
      case CmpInst::ICMP_SLT: op = "<"; break;
      case CmpInst::ICMP_ULE:
      case CmpInst::ICMP_SLE: op = "<="; break;
      default: unsupported("icmp predicate", &cmp);
    }
    return a + " " + op + " " + b;
  }

  std::string fcmp(const FCmpInst& cmp) {
    if (cmp.getOperand(0)->getType()->isVectorTy())
      unsupported("vector compare", &cmp);
    std::string a = value(cmp.getOperand(0)), b = value(cmp.getOperand(1));
    const bool dbl = cmp.getOperand(0)->getType()->isDoubleTy();
    auto ordered   = [&](CmpInst::Predicate p) -> std::string {
      if (dbl) {
        switch (p) {
          case CmpInst::FCMP_OEQ: return "km_deq(" + a + ", " + b + ")";
          case CmpInst::FCMP_OGT: return "km_dlt(" + b + ", " + a + ")";
          case CmpInst::FCMP_OGE: return "km_dle(" + b + ", " + a + ")";
          case CmpInst::FCMP_OLT: return "km_dlt(" + a + ", " + b + ")";
          case CmpInst::FCMP_OLE: return "km_dle(" + a + ", " + b + ")";
          case CmpInst::FCMP_ONE:
            return "(km_dord(" + a + ", " + b + ") && !km_deq(" + a + ", " + b +
                   "))";
          case CmpInst::FCMP_ORD: return "km_dord(" + a + ", " + b + ")";
          default: return "";
        }
      }
      switch (p) {
        case CmpInst::FCMP_OEQ: return a + " == " + b;
        case CmpInst::FCMP_OGT: return a + " > " + b;
        case CmpInst::FCMP_OGE: return a + " >= " + b;
        case CmpInst::FCMP_OLT: return a + " < " + b;
        case CmpInst::FCMP_OLE: return a + " <= " + b;
        case CmpInst::FCMP_ONE:
          return "(" + a + " < " + b + " || " + a + " > " + b + ")";
        case CmpInst::FCMP_ORD:
          return "(!isnan(" + a + ") && !isnan(" + b + "))";
        default: return "";
      }
    };
    CmpInst::Predicate p = cmp.getPredicate();
    if (p == CmpInst::FCMP_TRUE) return "true";
    if (p == CmpInst::FCMP_FALSE) return "false";
    if (CmpInst::isOrdered(p)) return ordered(p);
    // An unordered predicate is the negation of the inverse ordered one.
    return "!(" + ordered(CmpInst::getInversePredicate(p)) + ")";
  }

  std::string cast_expr(const CastInst& c) {
    const Value* x = c.getOperand(0);
    Type *from = x->getType(), *to = c.getType();
    if (from->isVectorTy() || to->isVectorTy()) unsupported("vector cast", &c);
    std::string a = value(x);
    if (from->isDoubleTy() || to->isDoubleTy()) {
      const bool to_int = to->isIntegerTy();
      switch (c.getOpcode()) {
        case Instruction::FPExt: return "km_f2d((float)" + a + ")";
        case Instruction::FPTrunc:
          return "(" + float_type(to) + ")km_d2f(" + a + ")";
        case Instruction::FPToSI:
          return "(" + int_type(to->getIntegerBitWidth()) + ")km_d2i(" + a +
                 ")";
        case Instruction::FPToUI:
          return "(" + int_type(to->getIntegerBitWidth()) + ")km_d2u(" + a +
                 ")";
        case Instruction::SIToFP:
          if (from->isIntegerTy(1))
            return "(" + a + " ? 0xbff0000000000000ul : 0ul)";
          return "km_i2d((long)" + signed_value(x) + ")";
        case Instruction::UIToFP: return "km_u2d((ulong)" + a + ")";
        case Instruction::BitCast:
          if (to_int || from->isIntegerTy()) return a;  // both are ulong
          break;
        default: break;
      }
      unsupported("this double-precision conversion", &c);
    }
    switch (c.getOpcode()) {
      case Instruction::Trunc:
        if (to->isIntegerTy(1)) return "(bool)(" + a + " & 1)";
        return "(" + int_type(to->getIntegerBitWidth()) + ")" + a;
      case Instruction::ZExt:
        return "(" + int_type(to->getIntegerBitWidth()) + ")" + a;
      case Instruction::SExt: {
        std::string T = int_type(to->getIntegerBitWidth());
        if (from->isIntegerTy(1))
          return "(" + T + ")(-(" + int_type(to->getIntegerBitWidth(), true) +
                 ")" + a + ")";
        return "(" + T + ")(" + int_type(to->getIntegerBitWidth(), true) + ")" +
               signed_value(x);
      }
      case Instruction::FPToUI:
        return "(" + int_type(to->getIntegerBitWidth()) + ")" + a;
      case Instruction::FPToSI:
        return "(" + int_type(to->getIntegerBitWidth()) + ")(" +
               int_type(to->getIntegerBitWidth(), true) + ")" + a;
      case Instruction::UIToFP: return "(" + float_type(to) + ")" + a;
      case Instruction::SIToFP:
        if (from->isIntegerTy(1))
          return "(" + a + " ? (" + float_type(to) + ")-1 : (" +
                 float_type(to) + ")0)";
        return "(" + float_type(to) + ")" + signed_value(x);
      case Instruction::FPTrunc:
      case Instruction::FPExt: return "(" + float_type(to) + ")" + a;
      case Instruction::PtrToInt:
        return "(" + int_type(to->getIntegerBitWidth()) + ")(ulong)" + a;
      case Instruction::IntToPtr: return "(" + type_of(&c) + ")(ulong)" + a;
      case Instruction::BitCast:
        if (to->isPointerTy()) return a;
        return "as_type<" + scalar_type(to) + ">(" + a + ")";
      case Instruction::AddrSpaceCast: return a;
      default: unsupported("cast", &c);
    }
  }

  // --- atomics ------------------------------------------------------------

  std::string atomic_type(Type* t, const Instruction* where) {
    if (t->isIntegerTy(32)) return "atomic_uint";
    if (t->isFloatTy()) return "atomic_float";
    if (t->isIntegerTy(64)) return "atomic_ulong";
    unsupported(
        "atomics on this type (Metal has 32-bit atomics, and 64-bit min/max)",
        where);
  }

  std::string atomic_pointer(const Value* ptr, const std::string& type,
                             const Instruction* where) {
    Space s = space_of(ptr);
    if (s != Space::Device && s != Space::Threadgroup)
      unsupported(Twine("an atomic on ") + space_name(s) + " memory", where);
    return std::string("(") + space_name(s) + " " + type + "*)" + value(ptr);
  }

  static bool is_wide(Type* t) {
    return t->isIntegerTy(64) || t->isDoubleTy() || t->isPointerTy();
  }

  // A 64-bit word read or written whole, which needs no atomic.
  std::string volatile_word(const Value* ptr) {
    return std::string("(*(") + space_name(space_of(ptr)) +
           " volatile ulong*)" + value(ptr) + ")";
  }

  std::string atomic_load(const LoadInst& ld) {
    if (is_wide(ld.getType()))
      return ld.getType()->isPointerTy()
                 ? "(" + type_of(&ld) + ")" +
                       volatile_word(ld.getPointerOperand())
                 : volatile_word(ld.getPointerOperand());
    std::string at = atomic_type(ld.getType(), &ld);
    return "atomic_load_explicit(" +
           atomic_pointer(ld.getPointerOperand(), at, &ld) +
           ", memory_order_relaxed)";
  }

  std::string atomic_rmw(const AtomicRMWInst& rmw) {
    Type* t       = rmw.getType();
    std::string v = value(rmw.getValOperand());
    auto op       = rmw.getOperation();
    if (is_wide(t) && op != AtomicRMWInst::UMax && op != AtomicRMWInst::UMin)
      unsupported(
          "a 64-bit atomic other than unsigned min/max (Metal has no 64-bit "
          "atomic add or "
          "CAS; atomics on doubles and 64-bit integers do not run on Apple "
          "GPUs)",
          &rmw);
    std::string at = atomic_type(t, &rmw);
    if (op == AtomicRMWInst::Max || op == AtomicRMWInst::Min) at = "atomic_int";
    std::string p = atomic_pointer(rmw.getPointerOperand(), at, &rmw);
    auto call     = [&](const char* fn, const std::string& arg) {
      return std::string(fn) + "(" + p + ", " + arg + ", memory_order_relaxed)";
    };
    switch (op) {
      case AtomicRMWInst::Xchg: return call("atomic_exchange_explicit", v);
      case AtomicRMWInst::Add:
      case AtomicRMWInst::FAdd: return call("atomic_fetch_add_explicit", v);
      case AtomicRMWInst::Sub:
      case AtomicRMWInst::FSub: return call("atomic_fetch_sub_explicit", v);
      case AtomicRMWInst::And: return call("atomic_fetch_and_explicit", v);
      case AtomicRMWInst::Or: return call("atomic_fetch_or_explicit", v);
      case AtomicRMWInst::Xor: return call("atomic_fetch_xor_explicit", v);
      case AtomicRMWInst::UMax: return call("atomic_fetch_max_explicit", v);
      case AtomicRMWInst::UMin: return call("atomic_fetch_min_explicit", v);
      case AtomicRMWInst::Max:
        return "(uint)" + call("atomic_fetch_max_explicit", "(int)" + v);
      case AtomicRMWInst::Min:
        return "(uint)" + call("atomic_fetch_min_explicit", "(int)" + v);
      case AtomicRMWInst::FMax:
      case AtomicRMWInst::FMin: {
        std::string q =
            atomic_pointer(rmw.getPointerOperand(), "atomic_uint", &rmw);
        return std::string(op == AtomicRMWInst::FMax ? "km_atomic_fmax("
                                                     : "km_atomic_fmin(") +
               q + ", " + v + ")";
      }
      default: unsupported("atomicrmw operation", &rmw);
    }
  }

  void emit_cmpxchg(const AtomicCmpXchgInst& cx) {
    Type* t              = cx.getCompareOperand()->getType();
    const std::string& n = names[&cx];
    if (!t->isIntegerTy(32))
      unsupported(
          "compare-and-swap on anything but 32-bit words (Metal has no 64-bit "
          "CAS; "
          "atomics on doubles and 64-bit integers do not run on Apple GPUs)",
          &cx);
    std::string p = atomic_pointer(cx.getPointerOperand(), "atomic_uint", &cx);
    line(n + "_0 = km_cas(" + p + ", " + value(cx.getCompareOperand()) + ", " +
         value(cx.getNewValOperand()) + ", " + n + "_1);");
  }

  // --- calls --------------------------------------------------------------

  std::string arg(const CallInst& c, unsigned i) {
    return value(c.getArgOperand(i));
  }

  std::string builtin_component(const char* var, const CallInst& c) {
    builtins.insert(var);
    if (auto* ci = dyn_cast<ConstantInt>(c.getArgOperand(0))) {
      static const char* xyz[] = {".x", ".y", ".z"};
      return std::string(var) + xyz[ci->getZExtValue() % 3];
    }
    return std::string(var) + "[" + arg(c, 0) + "]";
  }

  void emit_call(const CallInst& c) {
    Function* fn = c.getCalledFunction();
    if (!fn)
      unsupported(
          "an indirect call (virtual functions and function pointers do not "
          "work in kernels)",
          &c);
    StringRef name = fn->getName();
    if (fn->isIntrinsic()) return emit_intrinsic(c, fn->getIntrinsicID());
    if (name.starts_with("__kmetal_")) return emit_kmetal(c, name);
    if (c.getType()->isDoubleTy() && emit_double(c, name)) return;
    if (std::string m = libm(name); !m.empty()) {
      std::string args;
      for (unsigned i = 0; i < c.arg_size(); ++i)
        args += (i ? ", " : "") + arg(c, i);
      return assign(c, m + "(" + args + ")");
    }
    unsupported("a call to '" + demangled(name) +
                    "', whose body is not visible here. Functions called on "
                    "the GPU must be "
                    "inline (KOKKOS_INLINE_FUNCTION) in the translation unit "
                    "of the kernel",
                &c);
  }

  static std::string libm(StringRef name) {
    static const std::map<std::string, std::string, std::less<>> table = {
        {"sqrtf", "sqrt"},
        {"cbrtf", "cbrt"},
        {"expf", "exp"},
        {"exp2f", "exp2"},
        {"exp10f", "exp10"},
        {"expm1f", "km_expm1"},
        {"logf", "log"},
        {"log2f", "log2"},
        {"log10f", "log10"},
        {"log1pf", "km_log1p"},
        {"powf", "pow"},
        {"sinf", "sin"},
        {"cosf", "cos"},
        {"tanf", "tan"},
        {"asinf", "asin"},
        {"acosf", "acos"},
        {"atanf", "atan"},
        {"atan2f", "atan2"},
        {"sinhf", "sinh"},
        {"coshf", "cosh"},
        {"tanhf", "tanh"},
        {"asinhf", "asinh"},
        {"acoshf", "acosh"},
        {"atanhf", "atanh"},
        {"fabsf", "fabs"},
        {"floorf", "floor"},
        {"ceilf", "ceil"},
        {"truncf", "trunc"},
        {"roundf", "round"},
        {"rintf", "rint"},
        {"nearbyintf", "rint"},
        {"fmodf", "fmod"},
        {"fminf", "fmin"},
        {"fmaxf", "fmax"},
        {"fmaf", "fma"},
        {"copysignf", "copysign"},
        {"hypotf", "km_hypot"},
        {"erff", "km_erf"},
        {"erfcf", "km_erfc"},
        {"fdimf", "fdim"},
        {"remainderf", "km_remainder"},
    };
    auto it = table.find(name);
    return it == table.end() ? std::string() : precise(it->second);
  }

  void emit_kmetal(const CallInst& c, StringRef name) {
    if (name == "__kmetal_thread_position_in_grid")
      return assign(c, builtin_component("__tpig", c));
    if (name == "__kmetal_thread_position_in_threadgroup")
      return assign(c, builtin_component("__tptg", c));
    if (name == "__kmetal_threadgroup_position_in_grid")
      return assign(c, builtin_component("__tgpig", c));
    if (name == "__kmetal_threads_per_threadgroup")
      return assign(c, builtin_component("__ntptg", c));
    if (name == "__kmetal_threadgroups_per_grid")
      return assign(c, builtin_component("__ntgpg", c));
    if (name == "__kmetal_thread_index_in_simdgroup")
      return builtins.insert("__lane"), assign(c, "__lane");
    if (name == "__kmetal_simdgroup_index_in_threadgroup")
      return builtins.insert("__sgidx"), assign(c, "__sgidx");
    if (name == "__kmetal_threads_per_simdgroup")
      return builtins.insert("__sgsize"), assign(c, "__sgsize");
    if (name == "__kmetal_threadgroup_barrier")
      return line(
          "threadgroup_barrier(mem_flags::mem_threadgroup | "
          "mem_flags::mem_device);");
    if (name == "__kmetal_simdgroup_barrier")
      return line(
          "simdgroup_barrier(mem_flags::mem_threadgroup | "
          "mem_flags::mem_device);");
    if (name == "__kmetal_threadgroup_memory")
      return uses_tg = true, assign(c, "__tgmem");
    if (name == "__kmetal_simd_shuffle")
      return assign(
          c, "simd_shuffle(" + arg(c, 0) + ", (ushort)" + arg(c, 1) + ")");
    if (name == "__kmetal_simd_shuffle_xor")
      return assign(
          c, "simd_shuffle_xor(" + arg(c, 0) + ", (ushort)" + arg(c, 1) + ")");
    if (name == "__kmetal_simd_shuffle_down")
      return assign(
          c, "simd_shuffle_down(" + arg(c, 0) + ", (ushort)" + arg(c, 1) + ")");
    if (name == "__kmetal_simd_shuffle_up")
      return assign(
          c, "simd_shuffle_up(" + arg(c, 0) + ", (ushort)" + arg(c, 1) + ")");
    if (name == "__kmetal_is_device") return assign(c, "true");
    if (name == "__kmetal_kernel_key") return;
    if (name == "__kmetal_abort") {
      uses_abort = true;
      line("atomic_store_explicit(__abort, 1u, memory_order_relaxed);");
      return line("return;");
    }
    unsupported("unknown Metal intrinsic " + name, &c);
  }

  // Double-precision maths: exact operations in software, the rest through
  // float (about 7 significant digits).
  bool emit_double(const CallInst& c, StringRef f) {
    static const std::map<std::string, std::string, std::less<>> exact = {
        {"sqrt", "km_dsqrt"},      {"fabs", "km_dabs"},
        {"floor", "km_dfloor"},    {"ceil", "km_dceil"},
        {"trunc", "km_dtrunc"},    {"round", "km_droundf"},
        {"rint", "km_drint"},      {"nearbyint", "km_drint"},
        {"roundeven", "km_drint"}, {"copysign", "km_dcopysign"},
        {"fmin", "km_dmin"},       {"minnum", "km_dmin"},
        {"minimum", "km_dmin"},    {"minimumnum", "km_dmin"},
        {"fmax", "km_dmax"},       {"maxnum", "km_dmax"},
        {"maximum", "km_dmax"},    {"maximumnum", "km_dmax"},
        {"fma", "km_dfma"},        {"fmuladd", "km_dfma"},
    };
    static const std::set<std::string, std::less<>> via_float = {
        "sin",   "cos",   "tan",   "asin",      "acos",  "atan",
        "atan2", "sinh",  "cosh",  "tanh",      "asinh", "acosh",
        "atanh", "exp",   "exp2",  "exp10",     "log",   "log2",
        "log10", "pow",   "cbrt",  "fmod",      "hypot", "erf",
        "erfc",  "expm1", "log1p", "remainder", "fdim"};
    std::string args;
    if (auto it = exact.find(f); it != exact.end()) {
      for (unsigned i = 0; i < c.arg_size(); ++i)
        args += (i ? ", " : "") + arg(c, i);
      assign(c, it->second + "(" + args + ")");
      return true;
    }
    if (via_float.count(f)) {
      std::string m = libm(f.str() + "f");
      for (unsigned i = 0; i < c.arg_size(); ++i)
        args += (i ? ", " : "") + ("km_d2f(" + arg(c, i) + ")");
      assign(c, "km_f2d(" + (m.empty() ? f.str() : m) + "(" + args + "))");
      return true;
    }
    if (f == "powi")
      return assign(c, "km_dpowi(" + arg(c, 0) + ", (int)" + arg(c, 1) + ")"),
             true;
    if (f == "ldexp")
      return assign(c, "km_dldexp(" + arg(c, 0) + ", (int)" + arg(c, 1) + ")"),
             true;
    return false;
  }

  void emit_intrinsic(const CallInst& c, Intrinsic::ID id) {
    if ((c.getType()->isDoubleTy() ||
         (c.arg_size() && c.getArgOperand(0)->getType()->isDoubleTy())) &&
        id != Intrinsic::is_fpclass) {
      // llvm.sqrt.f64 -> sqrt
      StringRef name = c.getCalledFunction()->getName().drop_front(5);
      name           = name.take_until([](char ch) { return ch == '.'; });
      if (emit_double(c, name)) return;
    }
    auto fn1 = [&](const char* f) {
      assign(c, precise(f) + "(" + arg(c, 0) + ")");
    };
    auto fn2 = [&](const char* f) {
      assign(c, precise(f) + "(" + arg(c, 0) + ", " + arg(c, 1) + ")");
    };
    Type* t = c.getType();
    switch (id) {
      // No-ops for the GPU.
      case Intrinsic::lifetime_start:
      case Intrinsic::lifetime_end:
      case Intrinsic::assume:
      case Intrinsic::dbg_declare:
      case Intrinsic::dbg_value:
      case Intrinsic::dbg_label:
      case Intrinsic::experimental_noalias_scope_decl:
      case Intrinsic::invariant_end:
      case Intrinsic::sideeffect:
      case Intrinsic::var_annotation:
      case Intrinsic::donothing:
      case Intrinsic::pseudoprobe: return;
      case Intrinsic::invariant_start: return;
      case Intrinsic::expect:
      case Intrinsic::expect_with_probability:
      case Intrinsic::ptr_annotation:
      case Intrinsic::launder_invariant_group:
      case Intrinsic::strip_invariant_group: return assign(c, arg(c, 0));
      case Intrinsic::objectsize:
        return assign(
            c, cast<ConstantInt>(c.getArgOperand(1))->isOne() ? "0ul" : "~0ul");
      case Intrinsic::trap:
      case Intrinsic::ubsantrap:
      case Intrinsic::debugtrap: return line("return;");

      case Intrinsic::memcpy:
      case Intrinsic::memcpy_inline:
      case Intrinsic::memmove:
        return line(std::string(id == Intrinsic::memmove ? "km_memmove("
                                                         : "km_memcpy(") +
                    arg(c, 0) + ", " + arg(c, 1) + ", (ulong)" + arg(c, 2) +
                    ");");
      case Intrinsic::memset:
      case Intrinsic::memset_inline:
        return line("km_memset(" + arg(c, 0) + ", " + arg(c, 1) + ", (ulong)" +
                    arg(c, 2) + ");");

      // Floating point.
      case Intrinsic::sqrt: return fn1("sqrt");
      case Intrinsic::sin: return fn1("sin");
      case Intrinsic::cos: return fn1("cos");
      case Intrinsic::tan: return fn1("tan");
      case Intrinsic::asin: return fn1("asin");
      case Intrinsic::acos: return fn1("acos");
      case Intrinsic::atan: return fn1("atan");
      case Intrinsic::atan2: return fn2("atan2");
      case Intrinsic::sinh: return fn1("sinh");
      case Intrinsic::cosh: return fn1("cosh");
      case Intrinsic::tanh: return fn1("tanh");
      case Intrinsic::exp: return fn1("exp");
      case Intrinsic::exp2: return fn1("exp2");
      case Intrinsic::exp10: return fn1("exp10");
      case Intrinsic::log: return fn1("log");
      case Intrinsic::log2: return fn1("log2");
      case Intrinsic::log10: return fn1("log10");
      case Intrinsic::pow: return fn2("pow");
      case Intrinsic::fabs: return fn1("fabs");
      case Intrinsic::floor: return fn1("floor");
      case Intrinsic::ceil: return fn1("ceil");
      case Intrinsic::trunc: return fn1("trunc");
      case Intrinsic::rint:
      case Intrinsic::nearbyint:
      case Intrinsic::roundeven: return fn1("rint");
      case Intrinsic::round: return fn1("round");
      case Intrinsic::copysign: return fn2("copysign");
      case Intrinsic::minnum:
      case Intrinsic::minimum:
      case Intrinsic::minimumnum: return fn2("fmin");
      case Intrinsic::maxnum:
      case Intrinsic::maximum:
      case Intrinsic::maximumnum: return fn2("fmax");
      case Intrinsic::fma:
      case Intrinsic::fmuladd:
        return assign(
            c, "fma(" + arg(c, 0) + ", " + arg(c, 1) + ", " + arg(c, 2) + ")");
      case Intrinsic::ldexp:
        return assign(c, "ldexp(" + arg(c, 0) + ", (int)" + arg(c, 1) + ")");
      case Intrinsic::powi:
        return assign(c, "precise::pow(" + arg(c, 0) + ", (" + scalar_type(t) +
                             ")(int)" + arg(c, 1) + ")");
      case Intrinsic::is_fpclass: {
        uint64_t mask = cast<ConstantInt>(c.getArgOperand(1))->getZExtValue();
        std::string x = arg(c, 0);
        if (c.getArgOperand(0)->getType()->isDoubleTy()) {
          if (mask == fcNan) return assign(c, "km_dnan(" + x + ")");
          if (mask == fcInf) return assign(c, "km_dinf(" + x + ")");
          if (mask == fcFinite) return assign(c, "km_dfinite(" + x + ")");
          if (mask == (fcAllFlags & ~fcNan))
            return assign(c, "!km_dnan(" + x + ")");
          if (mask == fcZero) return assign(c, "((" + x + " << 1) == 0)");
          unsupported("is_fpclass on double with mask " + Twine(mask), &c);
        }
        if (mask == fcNan) return assign(c, "isnan(" + x + ")");
        if (mask == fcInf) return assign(c, "isinf(" + x + ")");
        if (mask == fcFinite) return assign(c, "isfinite(" + x + ")");
        if (mask == (fcAllFlags & ~fcNan))
          return assign(c, "!isnan(" + x + ")");
        if (mask == fcPosInf)
          return assign(c, "(isinf(" + x + ") && " + x + " > 0)");
        if (mask == fcNegInf)
          return assign(c, "(isinf(" + x + ") && " + x + " < 0)");
        if (mask == fcZero) return assign(c, "(" + x + " == 0)");
        unsupported("is_fpclass with mask " + Twine(mask), &c);
      }

      // Integer.
      case Intrinsic::umin: return fn2("min");
      case Intrinsic::umax: return fn2("max");
      case Intrinsic::smin:
      case Intrinsic::smax:
        return assign(c, "(" + scalar_type(t) + ")" +
                             (id == Intrinsic::smin ? "min(" : "max(") +
                             signed_value(c.getArgOperand(0)) + ", " +
                             signed_value(c.getArgOperand(1)) + ")");
      case Intrinsic::abs:
        return assign(c, "(" + scalar_type(t) + ")abs(" +
                             signed_value(c.getArgOperand(0)) + ")");
      case Intrinsic::ctpop:
        return assign(c, "(" + scalar_type(t) + ")popcount(" + arg(c, 0) + ")");
      case Intrinsic::ctlz:
        return assign(c, "(" + scalar_type(t) + ")clz(" + arg(c, 0) + ")");
      case Intrinsic::cttz:
        return assign(c, "(" + scalar_type(t) + ")ctz(" + arg(c, 0) + ")");
      case Intrinsic::bitreverse: return fn1("reverse_bits");
      case Intrinsic::bswap: return fn1("km_bswap");
      case Intrinsic::fshl:
      case Intrinsic::fshr: {
        unsigned bits = t->getIntegerBitWidth();
        std::string a = arg(c, 0), b = arg(c, 1),
                    n = "(" + arg(c, 2) + " % " + std::to_string(bits) + ")";
        std::string T = scalar_type(t);
        // fshl: high half of (a:b) << n; fshr: low half of (a:b) >> n.
        if (id == Intrinsic::fshl)
          return assign(c, "(" + n + " == 0 ? " + a + " : (" + T + ")((" + a +
                               " << " + n + ") | (" + b + " >> (" +
                               std::to_string(bits) + " - " + n + "))))");
        return assign(c, "(" + n + " == 0 ? " + b + " : (" + T + ")((" + b +
                             " >> " + n + ") | (" + a + " << (" +
                             std::to_string(bits) + " - " + n + "))))");
      }
      case Intrinsic::uadd_sat: return fn2("addsat");
      case Intrinsic::usub_sat: return fn2("subsat");
      case Intrinsic::uadd_with_overflow:
      case Intrinsic::usub_with_overflow:
      case Intrinsic::sadd_with_overflow:
      case Intrinsic::ssub_with_overflow: {
        const std::string& n = names[&c];
        Type* et             = cast<StructType>(t)->getElementType(0);
        std::string T        = scalar_type(et),
                    S        = int_type(et->getIntegerBitWidth(), true);
        std::string a = arg(c, 0), b = arg(c, 1);
        bool add = id == Intrinsic::uadd_with_overflow ||
                   id == Intrinsic::sadd_with_overflow;
        line(n + "_0 = (" + T + ")(" + a + (add ? " + " : " - ") + b + ");");
        if (id == Intrinsic::uadd_with_overflow)
          line(n + "_1 = " + n + "_0 < " + a + ";");
        else if (id == Intrinsic::usub_with_overflow)
          line(n + "_1 = " + a + " < " + b + ";");
        else if (add)
          line(n + "_1 = ((" + S + ")" + b + " > 0) ? ((" + S + ")" + n +
               "_0 < (" + S + ")" + a + ") : ((" + S + ")" + n + "_0 > (" + S +
               ")" + a + ");");
        else
          line(n + "_1 = ((" + S + ")" + b + " > 0) ? ((" + S + ")" + n +
               "_0 > (" + S + ")" + a + ") : ((" + S + ")" + n + "_0 < (" + S +
               ")" + a + ");");
        return;
      }
      default: break;
    }
    unsupported("the intrinsic " + c.getCalledFunction()->getName(), &c);
  }
};

}  // namespace

KernelSource emit_kernel(Function& F, const std::string& name) {
  return Emitter(F).run(name);
}

const std::string& prelude() {
  static const std::string text = R"(#include <metal_stdlib>
using namespace metal;
typedef bool __attribute__((__may_alias__)) km_bool;
typedef uchar __attribute__((__may_alias__)) km_u8;
typedef ushort __attribute__((__may_alias__)) km_u16;
typedef uint __attribute__((__may_alias__)) km_u32;
typedef ulong __attribute__((__may_alias__)) km_u64;
typedef half __attribute__((__may_alias__)) km_f16;
typedef float __attribute__((__may_alias__)) km_f32;
typedef bfloat __attribute__((__may_alias__)) km_bf16;
template <typename D, typename S> inline void km_memcpy(D d, S s, ulong n) {
  for (ulong i = 0; i < n; ++i) d[i] = s[i];
}
template <typename D, typename S> inline void km_memmove(D d, S s, ulong n) {
  if ((ulong)d < (ulong)s) { for (ulong i = 0; i < n; ++i) d[i] = s[i]; }
  else { for (ulong i = n; i > 0; --i) d[i - 1] = s[i - 1]; }
}
template <typename D> inline void km_memset(D d, uchar v, ulong n) {
  for (ulong i = 0; i < n; ++i) d[i] = v;
}
// Strong compare-and-swap from Metal's weak one; returns the old value.
template <typename P> inline uint km_cas(P p, uint cmp, uint val, thread bool& ok) {
  for (;;) {
    uint e = cmp;
    if (atomic_compare_exchange_weak_explicit(p, &e, val, memory_order_relaxed, memory_order_relaxed)) {
      ok = true;
      return cmp;
    }
    if (e != cmp) { ok = false; return e; }
  }
}
template <typename P> inline float km_atomic_fmax(P p, float v) {
  uint old = atomic_load_explicit(p, memory_order_relaxed);
  while (!atomic_compare_exchange_weak_explicit(p, &old, as_type<uint>(fmax(as_type<float>(old), v)),
                                                memory_order_relaxed, memory_order_relaxed)) {}
  return as_type<float>(old);
}
template <typename P> inline float km_atomic_fmin(P p, float v) {
  uint old = atomic_load_explicit(p, memory_order_relaxed);
  while (!atomic_compare_exchange_weak_explicit(p, &old, as_type<uint>(fmin(as_type<float>(old), v)),
                                                memory_order_relaxed, memory_order_relaxed)) {}
  return as_type<float>(old);
}
inline ushort km_bswap(ushort x) { return (x >> 8) | (x << 8); }
inline uint km_bswap(uint x) { return (x >> 24) | ((x >> 8) & 0xff00u) | ((x << 8) & 0xff0000u) | (x << 24); }
inline ulong km_bswap(ulong x) { return ((ulong)km_bswap((uint)x) << 32) | km_bswap((uint)(x >> 32)); }
template <typename T> inline T km_expm1(T x) {
  return fabs(x) < T(1e-5) ? x + T(0.5) * x * x : precise::exp(x) - T(1);
}
template <typename T> inline T km_log1p(T x) {
  T u = T(1) + x;
  return u == T(1) ? x : precise::log(u) * x / (u - T(1));
}
template <typename T> inline T km_hypot(T x, T y) { return length(vec<T, 2>(x, y)); }
template <typename T> inline T km_remainder(T x, T y) { return x - y * rint(x / y); }
// erf from Abramowitz & Stegun 7.1.26 refined: |error| < 1.2e-7.
inline float km_erfc(float x) {
  float z = fabs(x), t = 1.0f / (1.0f + 0.5f * z);
  float r = t * precise::exp(-z * z - 1.26551223f + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f +
            t * (-0.18628806f + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f +
            t * (-0.82215223f + t * 0.17087277f)))))))));
  return x >= 0 ? r : 2.0f - r;
}
inline float km_erf(float x) { return 1.0f - km_erfc(x); }

// --- fp64 in software ----------------------------------------------------------
// Apple GPUs have no double precision. A double is held as its bit pattern in
// a ulong; arithmetic follows Berkeley SoftFloat 3 (round to nearest even,
// subnormals, no exception flags), so results match the CPU bit for bit.
// Transcendentals go through float. Metal has no 64-bit atomics but
// unsigned min and max, so atomics on doubles cannot be had.
constant ulong KM_DSIGN = 0x8000000000000000ul;
constant ulong KM_DFRAC = 0x000ffffffffffffful;
constant ulong KM_DINF = 0x7ff0000000000000ul;
constant ulong KM_DQNAN = 0x7ff8000000000000ul;
inline int km_dexpo(ulong a) { return (int)((a >> 52) & 0x7ff); }
inline bool km_dnan(ulong a) { return (a & ~KM_DSIGN) > KM_DINF; }
inline ulong km_dnan2(ulong a, ulong b) { return (km_dnan(a) ? a : b) | 0x0008000000000000ul; }
inline ulong km_srjam(ulong a, uint d) {
  return d < 63 ? (a >> d) | (ulong)((a << ((0u - d) & 63)) != 0) : (ulong)(a != 0);
}
// sig has its leading bit at 62 and 10 rounding bits; exp is the biased
// exponent less one, as the leading bit adds one when packed.
inline ulong km_dround(ulong sign, int exp, ulong sig) {
  uint rb = (uint)(sig & 0x3ff);
  if ((uint)exp >= 0x7fdu) {
    if (exp < 0) {
      sig = km_srjam(sig, (uint)(-exp));
      exp = 0;
      rb = (uint)(sig & 0x3ff);
    } else if (exp > 0x7fd || sig + 0x200 >= 0x8000000000000000ul) {
      return sign | KM_DINF;
    }
  }
  sig = (sig + 0x200) >> 10;
  if (rb == 0x200) sig &= ~1ul;
  if (sig == 0) exp = 0;
  return sign + ((ulong)exp << 52) + sig;
}
inline ulong km_dnormround(ulong sign, int exp, ulong sig) {
  int sh = (int)clz(sig) - 1;
  exp -= sh;
  if (sh >= 10 && (uint)exp < 0x7fdu) return sign + ((ulong)(sig ? exp : 0) << 52) + (sig << (sh - 10));
  return km_dround(sign, exp, sig << sh);
}
inline ulong km_daddmags(ulong a, ulong b, ulong sign) {
  int ea = km_dexpo(a), eb = km_dexpo(b), d = ea - eb, ez;
  ulong sa = a & KM_DFRAC, sb = b & KM_DFRAC, sz;
  if (d == 0) {
    if (ea == 0) return a + sb;
    if (ea == 0x7ff) return (sa | sb) ? km_dnan2(a, b) : a;
    ez = ea;
    sz = (0x0020000000000000ul + sa + sb) << 9;
  } else {
    sa <<= 9;
    sb <<= 9;
    if (d < 0) {
      if (eb == 0x7ff) return sb ? km_dnan2(a, b) : (sign | KM_DINF);
      ez = eb;
      sa = ea ? sa + 0x2000000000000000ul : sa << 1;
      sa = km_srjam(sa, (uint)(-d));
    } else {
      if (ea == 0x7ff) return sa ? km_dnan2(a, b) : a;
      ez = ea;
      sb = eb ? sb + 0x2000000000000000ul : sb << 1;
      sb = km_srjam(sb, (uint)d);
    }
    sz = 0x2000000000000000ul + sa + sb;
    if (sz < 0x4000000000000000ul) { --ez; sz <<= 1; }
  }
  return km_dround(sign, ez, sz);
}
inline ulong km_dsubmags(ulong a, ulong b, ulong sign) {
  int ea = km_dexpo(a), eb = km_dexpo(b), d = ea - eb, ez;
  ulong sa = a & KM_DFRAC, sb = b & KM_DFRAC, sz;
  if (d == 0) {
    if (ea == 0x7ff) return (sa | sb) ? km_dnan2(a, b) : KM_DQNAN;
    long diff = (long)(sa - sb);
    if (diff == 0) return 0;
    if (ea) --ea;
    if (diff < 0) { sign ^= KM_DSIGN; diff = -diff; }
    int sh = (int)clz((ulong)diff) - 11;
    ez = ea - sh;
    if (ez < 0) { sh = ea; ez = 0; }
    return sign + ((ulong)ez << 52) + ((ulong)diff << sh);
  }
  sa <<= 10;
  sb <<= 10;
  if (d < 0) {
    sign ^= KM_DSIGN;
    if (eb == 0x7ff) return sb ? km_dnan2(a, b) : (sign | KM_DINF);
    sa += ea ? 0x4000000000000000ul : sa;
    sa = km_srjam(sa, (uint)(-d));
    sb |= 0x4000000000000000ul;
    ez = eb;
    sz = sb - sa;
  } else {
    if (ea == 0x7ff) return sa ? km_dnan2(a, b) : a;
    sb += eb ? 0x4000000000000000ul : sb;
    sb = km_srjam(sb, (uint)d);
    sa |= 0x4000000000000000ul;
    ez = ea;
    sz = sa - sb;
  }
  return km_dnormround(sign, ez - 1, sz);
}
inline ulong km_dadd(ulong a, ulong b) {
  ulong s = a & KM_DSIGN;
  return s == (b & KM_DSIGN) ? km_daddmags(a, b, s) : km_dsubmags(a, b, s);
}
inline ulong km_dsub(ulong a, ulong b) {
  ulong s = a & KM_DSIGN;
  return s == (b & KM_DSIGN) ? km_dsubmags(a, b, s) : km_daddmags(a, b, s);
}
// A subnormal's significand normalised to bit 52, and its exponent.
inline void km_dnormsub(thread int& e, thread ulong& s) {
  int sh = (int)clz(s) - 11;
  e = 1 - sh;
  s <<= sh;
}
inline ulong km_dmul(ulong a, ulong b) {
  ulong sign = (a ^ b) & KM_DSIGN;
  int ea = km_dexpo(a), eb = km_dexpo(b);
  ulong sa = a & KM_DFRAC, sb = b & KM_DFRAC;
  if (ea == 0x7ff) {
    if (sa || (eb == 0x7ff && sb)) return km_dnan2(a, b);
    return ((ulong)eb | sb) ? (sign | KM_DINF) : KM_DQNAN;
  }
  if (eb == 0x7ff) {
    if (sb) return km_dnan2(a, b);
    return ((ulong)ea | sa) ? (sign | KM_DINF) : KM_DQNAN;
  }
  if (ea == 0) { if (sa == 0) return sign; km_dnormsub(ea, sa); }
  if (eb == 0) { if (sb == 0) return sign; km_dnormsub(eb, sb); }
  int ez = ea + eb - 0x3ff;
  sa = (sa | 0x0010000000000000ul) << 10;
  sb = (sb | 0x0010000000000000ul) << 11;
  ulong sz = mulhi(sa, sb) | (ulong)(sa * sb != 0);
  if (sz < 0x4000000000000000ul) { --ez; sz <<= 1; }
  return km_dround(sign, ez, sz);
}
inline ulong km_ddiv(ulong a, ulong b) {
  ulong sign = (a ^ b) & KM_DSIGN;
  int ea = km_dexpo(a), eb = km_dexpo(b);
  ulong sa = a & KM_DFRAC, sb = b & KM_DFRAC;
  if (ea == 0x7ff) {
    if (sa) return km_dnan2(a, b);
    if (eb == 0x7ff) return sb ? km_dnan2(a, b) : KM_DQNAN;
    return sign | KM_DINF;
  }
  if (eb == 0x7ff) return sb ? km_dnan2(a, b) : sign;
  if (eb == 0) {
    if (sb == 0) return ((ulong)ea | sa) ? (sign | KM_DINF) : KM_DQNAN;
    km_dnormsub(eb, sb);
  }
  if (ea == 0) { if (sa == 0) return sign; km_dnormsub(ea, sa); }
  int ez = ea - eb + 0x3fe;
  sa |= 0x0010000000000000ul;
  sb |= 0x0010000000000000ul;
  if (sa < sb) { --ez; sa <<= 1; }
  ulong q = 0, r = sa;
  for (int i = 0; i < 63; ++i) {
    q <<= 1;
    if (r >= sb) { r -= sb; q |= 1; }
    r <<= 1;
  }
  return km_dround(sign, ez, q | (ulong)(r != 0));
}
inline ulong km_dsqrt(ulong a) {
  ulong sign = a & KM_DSIGN;
  int ea = km_dexpo(a);
  ulong sa = a & KM_DFRAC;
  if (ea == 0x7ff) return sa ? km_dnan2(a, a) : (sign ? KM_DQNAN : a);
  if (sign) return ((ulong)ea | sa) ? KM_DQNAN : a;
  if (ea == 0) { if (sa == 0) return a; km_dnormsub(ea, sa); }
  int e = ea - 0x3ff;
  ulong m = sa | 0x0010000000000000ul;
  if (e & 1) { m <<= 1; e -= 1; }
  // Digit by digit, over the 128-bit m << 72: a root with its leading bit at 62.
  ulong nh = m << 8, nl = 0, root = 0, rh = 0, rl = 0;
  for (int i = 0; i < 64; ++i) {
    rh = (rh << 2) | (rl >> 62);
    rl = (rl << 2) | (nh >> 62);
    nh = (nh << 2) | (nl >> 62);
    nl <<= 2;
    root <<= 1;
    ulong th = root >> 63, tl = (root << 1) | 1;
    if (rh > th || (rh == th && rl >= tl)) {
      ulong borrow = (ulong)(rl < tl);
      rl -= tl;
      rh -= th + borrow;
      root |= 1;
    }
  }
  return km_dround(0, e / 2 + 0x3fe, root | (ulong)((rh | rl) != 0));
}
inline ulong km_dfma(ulong a, ulong b, ulong c) { return km_dadd(km_dmul(a, b), c); }
inline ulong km_dneg(ulong a) { return a ^ KM_DSIGN; }
inline ulong km_dabs(ulong a) { return a & ~KM_DSIGN; }
inline ulong km_dcopysign(ulong a, ulong b) { return (a & ~KM_DSIGN) | (b & KM_DSIGN); }
inline bool km_deq(ulong a, ulong b) { return !km_dnan(a) && !km_dnan(b) && (a == b || ((a | b) << 1) == 0); }
inline bool km_dlt(ulong a, ulong b) {
  if (km_dnan(a) || km_dnan(b)) return false;
  bool sa = (a >> 63) != 0, sb = (b >> 63) != 0;
  if (sa != sb) return sa && ((a | b) << 1) != 0;
  return a != b && (sa != (a < b));
}
inline bool km_dle(ulong a, ulong b) {
  if (km_dnan(a) || km_dnan(b)) return false;
  bool sa = (a >> 63) != 0, sb = (b >> 63) != 0;
  if (sa != sb) return sa || ((a | b) << 1) == 0;
  return a == b || (sa != (a < b));
}
inline bool km_dord(ulong a, ulong b) { return !km_dnan(a) && !km_dnan(b); }
inline ulong km_dmin(ulong a, ulong b) { return km_dnan(a) ? b : km_dnan(b) ? a : km_dlt(b, a) ? b : a; }
inline ulong km_dmax(ulong a, ulong b) { return km_dnan(a) ? b : km_dnan(b) ? a : km_dlt(a, b) ? b : a; }
inline ulong km_f2d(float f) {
  uint u = as_type<uint>(f);
  ulong sign = (ulong)(u >> 31) << 63;
  int e = (int)((u >> 23) & 0xff);
  uint fr = u & 0x7fffffu;
  if (e == 0xff) return sign | KM_DINF | ((ulong)fr << 29) | (fr ? 0x0008000000000000ul : 0ul);
  if (e == 0) {
    if (fr == 0) return sign;
    int sh = (int)clz(fr) - 8;
    e = 1 - sh;
    fr = (fr << sh) & 0x7fffffu;
  }
  return sign | ((ulong)(e + 0x380) << 52) | ((ulong)fr << 29);
}
inline uint km_fround(uint sign, int exp, uint sig) {
  uint rb = sig & 0x7f;
  if ((uint)exp >= 0xfdu) {
    if (exp < 0) {
      sig = exp > -31 ? (sig >> (uint)(-exp)) | (uint)((sig << (uint)(32 + exp)) != 0) : (uint)(sig != 0);
      exp = 0;
      rb = sig & 0x7f;
    } else if (exp > 0xfd || sig + 0x40 >= 0x80000000u) {
      return sign | 0x7f800000u;
    }
  }
  sig = (sig + 0x40) >> 7;
  if (rb == 0x40) sig &= ~1u;
  if (!sig) exp = 0;
  return sign + ((uint)exp << 23) + sig;
}
inline float km_d2f(ulong a) {
  uint sign = (uint)(a >> 32) & 0x80000000u;
  int e = km_dexpo(a);
  ulong fr = a & KM_DFRAC;
  if (e == 0x7ff) return as_type<float>(sign | 0x7f800000u | (fr ? 0x400000u | (uint)(fr >> 29) : 0u));
  uint f = (uint)km_srjam(fr, 22);
  if (((uint)e | f) == 0) return as_type<float>(sign);
  return as_type<float>(km_fround(sign, e - 0x381, f | 0x40000000u));
}
inline ulong km_i2d(long a) {
  ulong sign = (ulong)a & KM_DSIGN;
  if (((ulong)a << 1) == 0) return sign ? 0xc3e0000000000000ul : 0ul;
  return km_dnormround(sign, 0x43c, sign ? (ulong)(-a) : (ulong)a);
}
inline ulong km_u2d(ulong a) {
  if (a & KM_DSIGN) return km_dround(0, 0x43d, (a >> 1) | (a & 1));
  return km_dnormround(0, 0x43c, a);
}
inline long km_d2i(ulong a) {
  int sh = km_dexpo(a) - 0x3ff;
  if (sh < 0 || km_dnan(a)) return 0;
  if (sh > 62) return (a & KM_DSIGN) ? (long)KM_DSIGN : 0x7ffffffffffffffful;
  ulong sig = (a & KM_DFRAC) | 0x0010000000000000ul;
  ulong m = sh >= 52 ? sig << (sh - 52) : sig >> (52 - sh);
  return (a & KM_DSIGN) ? -(long)m : (long)m;
}
inline ulong km_d2u(ulong a) {
  int sh = km_dexpo(a) - 0x3ff;
  if (sh < 0 || (a & KM_DSIGN) || km_dnan(a)) return 0;
  if (sh > 63) return ~0ul;
  ulong sig = (a & KM_DFRAC) | 0x0010000000000000ul;
  return sh >= 52 ? sig << (sh - 52) : sig >> (52 - sh);
}
inline ulong km_dtrunc(ulong a) {
  int e = km_dexpo(a);
  if (e >= 0x433) return a;
  if (e < 0x3ff) return a & KM_DSIGN;
  return a & ~((1ul << (0x433 - e)) - 1);
}
inline ulong km_dfloor(ulong a) {
  ulong t = km_dtrunc(a);
  return ((a & KM_DSIGN) && t != a) ? km_dsub(t, 0x3ff0000000000000ul) : t;
}
inline ulong km_dceil(ulong a) {
  ulong t = km_dtrunc(a);
  return (!(a & KM_DSIGN) && t != a) ? km_dadd(t, 0x3ff0000000000000ul) : t;
}
// Halves away from zero.
inline ulong km_droundf(ulong a) {
  int e = km_dexpo(a);
  if (e >= 0x433) return a;
  if (e < 0x3fe) return a & KM_DSIGN;
  if (e == 0x3fe) return (a & KM_DSIGN) | 0x3ff0000000000000ul;
  ulong h = 1ul << (0x432 - e);
  return (a + h) & ~((h << 1) - 1);
}
// Halves to even.
inline ulong km_drint(ulong a) {
  int e = km_dexpo(a);
  if (e >= 0x433) return a;
  if (e < 0x3fe) return a & KM_DSIGN;
  if (e == 0x3fe) return (a & KM_DSIGN) | ((a & KM_DFRAC) ? 0x3ff0000000000000ul : 0ul);
  ulong one = 1ul << (0x433 - e), h = one >> 1, frac = a & (one - 1);
  a &= ~(one - 1);
  if (frac > h || (frac == h && (a & one))) a += one;
  return a;
}
inline ulong km_dpowi(ulong a, int n) {
  ulong r = 0x3ff0000000000000ul, b = a;
  uint m = n < 0 ? (uint)(-n) : (uint)n;
  for (; m; m >>= 1) {
    if (m & 1) r = km_dmul(r, b);
    b = km_dmul(b, b);
  }
  return n < 0 ? km_ddiv(0x3ff0000000000000ul, r) : r;
}
inline ulong km_dldexp(ulong a, int n) {
  int e = km_dexpo(a);
  if (e == 0x7ff || (a << 1) == 0) return a;
  // Two steps of at most 2^1000 each keep the factors normal.
  n = clamp(n, -2200, 2200);
  for (int k = 0; k < 3 && n != 0; ++k) {
    int s = clamp(n, -1000, 1000);
    a = km_dmul(a, (ulong)(s + 0x3ff) << 52);
    n -= s;
  }
  return a;
}
inline bool km_dinf(ulong a) { return (a << 1) == 0xffe0000000000000ul; }
inline bool km_dfinite(ulong a) { return km_dexpo(a) != 0x7ff; }
)";
  return text;
}

}  // namespace kmetal
