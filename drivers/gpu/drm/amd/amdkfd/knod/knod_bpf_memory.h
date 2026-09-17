/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef KNOD_BPF_MEMORY_H
#define KNOD_BPF_MEMORY_H

/* Shared verified-BPF facts. No hardware opcode or physical register here.
 * Callers provide path-conservative ownership and final CFG boundaries.
 */
/* Whole verified-BPF register effects. A 32-bit ALU result zeroes its high
 * half, so it defines the complete BPF register. Special helper/atomic and
 * unsupported instructions remain conservative barriers. */
#define KNOD_BPF_REGS_LIVE ((1U << 11) - 1)
struct knod_bpf_effect {
	unsigned short uses, defs;
	bool barrier, terminal;
};
static inline struct knod_bpf_effect knod_bpf_effect(const struct bpf_insn *i)
{
	struct knod_bpf_effect f = {.uses = KNOD_BPF_REGS_LIVE, .barrier = true};
	unsigned int cls = BPF_CLASS(i->code), op = BPF_OP(i->code);
	unsigned int mode = BPF_MODE(i->code);
	bool source = false, dest = false;

	switch (cls) {
	case BPF_ALU:
	case BPF_ALU64:
		switch (op) {
		case BPF_MOV:
			source = BPF_SRC(i->code) == BPF_X;
			break;
		case BPF_END:
		case BPF_NEG:
			dest = true;
			break;
		case BPF_ADD:
		case BPF_SUB:
		case BPF_MUL:
		case BPF_DIV:
		case BPF_MOD:
		case BPF_AND:
		case BPF_OR:
		case BPF_XOR:
		case BPF_LSH:
		case BPF_RSH:
		case BPF_ARSH:
			dest = true;
			source = BPF_SRC(i->code) == BPF_X;
			break;
		default:
			return f;
		}
		if (i->dst_reg > 10 || (source && i->src_reg > 10))
			return f;
		f.defs = 1U << i->dst_reg;
		f.uses = (dest ? f.defs : 0) | (source ? 1U << i->src_reg : 0);
		break;
	case BPF_LD:
		if (i->code != (BPF_LD | BPF_IMM | BPF_DW) || i->dst_reg > 10)
			return f;
		/* src_reg is a pseudo-immediate tag, never a register operand. The
		 * continuation word stays an unknown barrier if visited separately. */
		f.uses = 0;
		f.defs = 1U << i->dst_reg;
		break;
	case BPF_LDX:
		if ((mode != BPF_MEM && mode != BPF_MEMSX) || i->src_reg > 10 || i->dst_reg > 10)
			return f;
		f.uses = 1U << i->src_reg;
		f.defs = 1U << i->dst_reg;
		break;
	case BPF_ST:
	case BPF_STX:
		if (mode != BPF_MEM || i->dst_reg > 10 || (cls == BPF_STX && i->src_reg > 10))
			return f;
		f.uses = (1U << i->dst_reg) | (cls == BPF_STX ? 1U << i->src_reg : 0);
		break;
	case BPF_JMP:
	case BPF_JMP32:
		if (cls == BPF_JMP && op == BPF_EXIT) {
			f.uses = 1U << 0;
			f.terminal = true;
			break;
		}
		/* Report explicit branch operands, but liveness still stops at every
		 * successor boundary. Calls retain the default all-live effect. */
		switch (op) {
		case BPF_JA:
			f.uses = 0;
			break;
		case BPF_JEQ:
		case BPF_JGT:
		case BPF_JGE:
		case BPF_JSET:
		case BPF_JNE:
		case BPF_JSGT:
		case BPF_JSGE:
		case BPF_JLT:
		case BPF_JLE:
		case BPF_JSLT:
		case BPF_JSLE:
			source = BPF_SRC(i->code) == BPF_X;
			if (i->dst_reg > 10 || (source && i->src_reg > 10))
				return f;
			f.uses = (1U << i->dst_reg) | (source ? 1U << i->src_reg : 0);
			break;
		default:
			break;
		}
		return f;
	default:
		return f;
	}
	f.barrier = false;
	return f;
}
static inline unsigned short knod_bpf_live_before(const struct bpf_insn *i, unsigned short after)
{
	struct knod_bpf_effect f = knod_bpf_effect(i);

	if (f.barrier)
		return KNOD_BPF_REGS_LIVE;
	if (f.terminal)
		return f.uses;
	return (f.uses | (after & ~f.defs)) & KNOD_BPF_REGS_LIVE;
}
enum knod_memory_domain {
	KNOD_MEMORY_UNKNOWN,
	KNOD_MEMORY_PACKET,
	KNOD_MEMORY_PRIVATE_STACK,
	KNOD_MEMORY_SHARED_MAP,
};
struct knod_memory_fact {
	unsigned short uses, defs;
	int off;
	unsigned char domain, width, base, dst;
	bool barrier;
};
struct knod_memory_load_group {
	short off;
	unsigned char bytes, count, base;
	struct {
		unsigned char dst, off, width;
	} member[16];
};
static inline struct knod_memory_fact
knod_memory_fact(const struct bpf_insn *i, enum knod_memory_domain domain)
{
	struct knod_memory_fact f = {.domain = domain, .barrier = true};
	if (i->code != (BPF_LDX | BPF_MEM | BPF_B) &&
	    i->code != (BPF_LDX | BPF_MEM | BPF_H) &&
	    i->code != (BPF_LDX | BPF_MEM | BPF_W) &&
	    i->code != (BPF_LDX | BPF_MEM | BPF_DW))
		return f;
	if (i->src_reg > 10 || i->dst_reg > 9)
		return f;
	f.width = BPF_SIZE(i->code) == BPF_DW  ? 8
		  : BPF_SIZE(i->code) == BPF_W ? 4
		  : BPF_SIZE(i->code) == BPF_H ? 2
					       : 1;
	f.off = i->off;
	f.base = i->src_reg;
	f.dst = i->dst_reg;
	f.uses = 1U << f.base;
	f.defs = 1U << f.dst;
	f.barrier =
	    domain != KNOD_MEMORY_PACKET && domain != KNOD_MEMORY_PRIVATE_STACK;
	return f;
}
/* Exact contiguous nonoverlapping loads, in either address order. Delaying
 * all destination writes is legal only if no source depends on those writes.
 * Duplicate destinations are rejected: every original value is retained.
 */
static inline bool knod_memory_load_append(struct knod_memory_load_group *g,
					   const struct knod_memory_fact *f,
					   bool boundary)
{
	unsigned int i, defs = 0;
	int lo, hi;
	if (f->barrier || (g->count && boundary) || g->count == 16 ||
	    (f->uses & f->defs))
		return false;
	if (g->count && g->base != f->base)
		return false;
	for (i = 0; i < g->count; i++)
		defs |= 1U << g->member[i].dst;
	if (defs & (f->uses | f->defs))
		return false;
	lo = g->count ? g->off : f->off;
	hi = g->count ? g->off + g->bytes : f->off;
	if (g->count && f->off != hi && f->off + f->width != lo)
		return false;
	if (g->bytes + f->width > 16)
		return false;
	if (f->off < lo) {
		for (i = 0; i < g->count; i++)
			g->member[i].off += lo - f->off;
		lo = f->off;
	}
	g->member[g->count].dst = f->dst;
	g->member[g->count].off = f->off - lo;
	g->member[g->count++].width = f->width;
	g->off = lo;
	g->base = f->base;
	g->bytes += f->width;
	return true;
}
static inline unsigned int knod_memory_chunk_width(unsigned int bytes)
{
	return bytes >= 16   ? 16
	       : bytes >= 12 ? 12
	       : bytes >= 8  ? 8
	       : bytes >= 4  ? 4
	       : bytes >= 2  ? 2
			     : 1;
}
/* Compiler-only dependency facts. Backends separately certify physical hazards. */
#define KNOD_LOCAL_MAX 8
struct knod_local_effect {
	struct knod_bpf_effect reg;
	unsigned char space, width;
	bool read, write, exact, boundary, owned;
	unsigned int identity;
	int offset;
};

static inline bool knod_local_dependency(const struct knod_local_effect *a,
					 const struct knod_local_effect *b)
{
	if (a->boundary || b->boundary || a->owned || b->owned ||
	    a->reg.barrier || b->reg.barrier || a->reg.terminal || b->reg.terminal ||
	    (a->reg.defs & (b->reg.uses | b->reg.defs)) ||
	    (a->reg.uses & b->reg.defs))
		return true;
	if (!(a->read || a->write) || !(b->read || b->write))
		return false;
	/* Keep all ordinary reads in original order, including mutable maps. */
	if (!a->write && !b->write)
		return true;
	if (a->space == KNOD_MEMORY_UNKNOWN || b->space == KNOD_MEMORY_UNKNOWN)
		return true;
	if ((a->space == KNOD_MEMORY_PRIVATE_STACK) !=
	    (b->space == KNOD_MEMORY_PRIVATE_STACK))
		return false;
	if (a->space != b->space || !a->exact || !b->exact || !a->width || !b->width ||
	    a->identity != b->identity)
		return true;
	return !((long long)a->offset + a->width <= b->offset ||
		 (long long)b->offset + b->width <= a->offset);
}

/* Two reads may share a finish boundary while retaining their issue order. */
static inline bool knod_local_batch_reads(const struct knod_local_effect *a,
					 const struct knod_local_effect *b)
{
	return a->read && b->read && !a->write && !b->write &&
	       !a->boundary && !b->boundary && !a->owned && !b->owned &&
	       !a->reg.barrier && !b->reg.barrier &&
	       !a->reg.terminal && !b->reg.terminal &&
	       !(a->reg.defs & (b->reg.uses | b->reg.defs)) &&
	       !(a->reg.uses & b->reg.defs);
}

/* Stable ready-list scheduler. Scores are caller-supplied lowering costs;
 * ties preserve source order. No mutation of original BPF nodes or ownership.
 */
static inline bool knod_local_schedule(const struct knod_local_effect *nodes,
				      const unsigned char *scores,
				      unsigned int count, unsigned char *order)
{
	unsigned char predecessors[KNOD_LOCAL_MAX] = { 0 };
	unsigned int done = 0, i, j, step;

	if (!count || count > KNOD_LOCAL_MAX)
		return false;
	for (i = 0; i < count; i++)
		for (j = 0; j < i; j++)
			if (knod_local_dependency(&nodes[j], &nodes[i]))
				predecessors[i] |= 1U << j;
	for (step = 0; step < count; step++) {
		int best = -1;

		for (i = 0; i < count; i++)
			if (!(done & (1U << i)) && !(predecessors[i] & ~done) &&
			    (best < 0 || scores[i] > scores[best]))
				best = i;
		if (best < 0)
			return false;
		order[step] = best;
		done |= 1U << best;
	}
	return true;
}


#endif
