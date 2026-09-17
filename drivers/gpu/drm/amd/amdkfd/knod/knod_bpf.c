// SPDX-License-Identifier: GPL-2.0-or-later
/* Copyright (c) 2021 Taehee Yoo <ap420073@gmail.com>
 * Copyright (c) 2021 Hoyeon Lee <hoyeon.rhee@gmail.com>
 */

#include <linux/cpumask.h>
#include <linux/types.h>
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/delay.h>
#include <linux/sched.h>
#include <linux/workqueue.h>
#include <linux/file.h>
#include <linux/jhash.h>
#include <drm/ttm/ttm_tt.h>
#include <net/page_pool/helpers.h>
#include "kfd_priv.h"
#include "kfd_hsa.h"
#include "knod_bpf.h"
#include "kfd_migrate.h"
#include "kfd_events.h"
#include "kfd_device_queue_manager.h"
#include <linux/firmware.h>
#include <linux/reciprocal_div.h>
#include <linux/jhash.h>
#include <net/knod.h>
#include <net/netdev_rx_queue.h>

/* The prologue walks these structures, and a prologue built outside the kernel
 * has only the numbers knod_blob.h publishes to walk them with.  Nothing warns
 * when a field moves, so say here what those numbers are supposed to be.
 */
static_assert(offsetof(struct hsa_kernel_dispatch_packet, kernarg_address) ==
	      KNOD_BLOB_AQL_KERNARG);
static_assert(offsetof(struct knod_bpf_param, batch_shift) ==
	      KNOD_BLOB_PARAM_BATCH_SHIFT);
static_assert(offsetof(struct knod_bpf_param, wg_shift) ==
	      KNOD_BLOB_PARAM_WG_SHIFT);
static_assert(offsetof(struct knod_bpf_param, page_shift) ==
	      KNOD_BLOB_PARAM_PAGE_SHIFT);
static_assert(offsetof(struct knod_bpf_param, spsc_shift) ==
	      KNOD_BLOB_PARAM_SPSC_SHIFT);
static_assert(offsetof(struct knod_bpf_param, queues) ==
	      KNOD_BLOB_PARAM_QUEUES);
static_assert(offsetof(struct knod_bpf_param, pass_indices) ==
	      KNOD_BLOB_PARAM_PASS_INDICES);
static_assert(offsetof(struct knod_bpf_param, sub) ==
	      KNOD_BLOB_PARAM_SUB);
static_assert(offsetof(struct knod_bpf_queue_desc, count) ==
	      KNOD_BLOB_QUEUE_COUNT);
static_assert(offsetof(struct knod_bpf_queue_desc, ring_start) ==
	      KNOD_BLOB_QUEUE_RING_START);
static_assert(offsetof(struct knod_bpf_queue_desc, ring_mask) ==
	      KNOD_BLOB_QUEUE_RING_MASK);
static_assert(sizeof(struct knod_bpf_queue_desc) ==
	      KNOD_BLOB_QUEUE_SIZE);
static_assert(offsetof(struct spsc_bd, off) ==
	      KNOD_BLOB_BD_OFF);
static_assert(offsetof(struct spsc_bd, page_idx) ==
	      KNOD_BLOB_BD_PAGE_IDX);
/* The epilogue publishes the verdict, the bounds and the page in one
 * four-dword store from KNOD_BLOB_BD_ACT, so nothing the shader must not
 * write may sit inside that range.
 */
static_assert(offsetof(struct spsc_bd, pp) >= KNOD_BLOB_BD_ACT + 16);
/* The probe writes past spsc_bd and inside the slot it sits in.  Grow the one
 * or shrink the other and it would quietly write over a descriptor.
 */
static_assert(sizeof(struct spsc_bd) <= KNOD_BLOB_BD_PROBE);
static_assert(KNOD_BLOB_BD_PROBE + KNOD_PROBE_PARTS * sizeof(u32) <=
	      ALIGN(sizeof(struct spsc_bd), SPSC_ELEM_ALIGN));
static_assert(sizeof(struct knod_bpf_subparam_obj) ==
	      KNOD_BLOB_SUB_SIZE);

/*+--------+---------+------+-------+----+--+-----+------+------+--------+
 *| v0-v21 | v22-v57 |58-59 |v60-v61| 62 |63|64-65|66-67 |68-69 |v70-v127|
 *+--------+---------+------+-------+----+--+-----+------+------+--------+
 *|BPF REGS|TMP REGS | SLOT |CTX REG|WIDX|PI|DATA |D_END |PGBASE|  free  |
 *+--------+---------+------+-------+----+--+-----+------+------+--------+
 * SLOT through PGBASE are set in the prologue and read later, so nothing there
 * may be used as scratch.  TMP is the opposite: it holds nothing across the
 * program, which is what lets prebuilt routines spliced into it clobber the
 * lot.  v70-v127 is free since the packet cache was removed.
 *+---------+-----------+
 *| 128-130 | v131-v255 |
 *+---------+-----------+
 *| LDS WIN |   free    |
 *+---------+-----------+
 * The BPF stack lives in LDS: v128:129 are the two-register window into it and
 * v130 holds the lane's LDS base.
 */

/* Temp register map
 *+-------------+-------------+---------------+---------------+
 *|TREG0 - TREG2|TREG3 - TREG9|TREG10 - TREG16|     TREG17    |
 *+-------------+-------------+---------------+---------------+
 *| General Use | Key cache A |   Key in MAP  | JHASH Temp Reg|
 *+-------------+-------------+---------------+---------------+
 * Available Key cache size is 56.
 * So, key size of map can't be exceed 56B.
 */

#define KNOD_AMDGPU_VREG0_LO		0 /* v0 */
#define KNOD_AMDGPU_VREG0_HI		1
#define KNOD_AMDGPU_VREG1_LO		2
#define KNOD_AMDGPU_VREG1_HI		3
#define KNOD_AMDGPU_VREG2_LO		4
#define KNOD_AMDGPU_VREG2_HI		5
#define KNOD_AMDGPU_VREG3_LO		6
#define KNOD_AMDGPU_VREG3_HI		7
#define KNOD_AMDGPU_VREG4_LO		8
#define KNOD_AMDGPU_VREG4_HI		9
#define KNOD_AMDGPU_VREG5_LO		10
#define KNOD_AMDGPU_VREG5_HI		11
#define KNOD_AMDGPU_VREG6_LO		12
#define KNOD_AMDGPU_VREG6_HI		13
#define KNOD_AMDGPU_VREG7_LO		14
#define KNOD_AMDGPU_VREG7_HI		15
#define KNOD_AMDGPU_VREG8_LO		16
#define KNOD_AMDGPU_VREG8_HI		17
#define KNOD_AMDGPU_VREG9_LO		18
#define KNOD_AMDGPU_VREG9_HI		19
#define KNOD_AMDGPU_FRAME_POINTER_VREG_LO 20 /* v20 */
#define KNOD_AMDGPU_FRAME_POINTER_VREG_HI 21 /* v20 */

#define KNOD_AMDGPU_TMP_VREG0_LO	22
#define KNOD_AMDGPU_TMP_VREG0_HI	23
#define KNOD_AMDGPU_TMP_VREG1_LO	24
#define KNOD_AMDGPU_TMP_VREG1_HI	25
#define KNOD_AMDGPU_TMP_VREG2_LO	26
#define KNOD_AMDGPU_TMP_VREG2_HI	27
#define KNOD_AMDGPU_TMP_VREG3_LO	28
#define KNOD_AMDGPU_TMP_VREG3_HI	29
#define KNOD_AMDGPU_TMP_VREG4_LO	30
#define KNOD_AMDGPU_TMP_VREG4_HI	31
#define KNOD_AMDGPU_TMP_VREG5_LO	32
#define KNOD_AMDGPU_TMP_VREG5_HI	33
#define KNOD_AMDGPU_TMP_VREG6_LO	34
#define KNOD_AMDGPU_TMP_VREG6_HI	35
#define KNOD_AMDGPU_TMP_VREG7_LO	36
#define KNOD_AMDGPU_TMP_VREG7_HI	37
#define KNOD_AMDGPU_TMP_VREG8_LO	38
#define KNOD_AMDGPU_TMP_VREG8_HI	39
#define KNOD_AMDGPU_TMP_VREG9_LO	40
#define KNOD_AMDGPU_TMP_VREG9_HI	41
#define KNOD_AMDGPU_TMP_VREG10_LO	42
#define KNOD_AMDGPU_TMP_VREG10_HI	43
#define KNOD_AMDGPU_TMP_VREG11_LO	44
#define KNOD_AMDGPU_TMP_VREG11_HI	45
#define KNOD_AMDGPU_TMP_VREG12_LO	46
#define KNOD_AMDGPU_TMP_VREG12_HI	47
#define KNOD_AMDGPU_TMP_VREG13_LO	48
#define KNOD_AMDGPU_TMP_VREG13_HI	49
#define KNOD_AMDGPU_TMP_VREG14_LO	50
#define KNOD_AMDGPU_TMP_VREG14_HI	51
#define KNOD_AMDGPU_TMP_VREG15_LO	52
#define KNOD_AMDGPU_TMP_VREG15_HI	53
#define KNOD_AMDGPU_TMP_VREG16_LO	54
#define KNOD_AMDGPU_TMP_VREG16_HI	55
#define KNOD_AMDGPU_TMP_VREG17_LO	56
#define KNOD_AMDGPU_TMP_VREG17_HI	57
#define KNOD_AMDGPU_TMP_VREG_MAX	KNOD_AMDGPU_TMP_VREG17_HI
/*
 * slot_addr (spsc_bd GTT address) is set in the prologue and read in the
 * epilogue, so it sits above the scratch registers rather than inside them.
 * It used to share v62:v63 with IDX_VREG, which meant the backlog index had
 * to be copied out to a scratch register to survive - a value living across
 * the whole program in space nothing else could then rely on.
 */
#define KNOD_AMDGPU_SLOT_VREG_LO	58
#define KNOD_AMDGPU_SLOT_VREG_HI	59
#define KNOD_AMDGPU_CTX_VREG_LO		60
#define KNOD_AMDGPU_CTX_VREG_HI		61
#define KNOD_AMDGPU_IDX_VREG		62
/* The page the producer named, carried across the program so the epilogue can
 * hand it back unchanged - which it does only so the verdict, the bounds and
 * the page go out in one store rather than two.
 */
#define KNOD_AMDGPU_PAGE_IDX_VREG	63
/*
 * DATA/DATA_END VGPRs: hold packet gaddr and end address.
 * Set in prologue, read by BPF ctx->data / ctx->data_end accesses.
 * Replaces GTT round-trip (prologue store -> BPF load).
 */
#define KNOD_AMDGPU_DATA_VREG_LO	64
#define KNOD_AMDGPU_DATA_VREG_HI	65
#define KNOD_AMDGPU_DATA_END_VREG_LO	66
#define KNOD_AMDGPU_DATA_END_VREG_HI	67
#define KNOD_AMDGPU_PAGE_BASE_VREG_LO	68
#define KNOD_AMDGPU_PAGE_BASE_VREG_HI	69
/* The stack lives in LDS, so v131 upwards is free for whoever wants it - the
 * wave still declares all 256 either way.  These two are the pair the load and
 * store helpers work a slot through, kept at the bottom of that range so that
 * what is free stays one contiguous run.
 */
#define KNOD_AMDGPU_STACK_WIN_VREG0	128
#define KNOD_AMDGPU_STACK_WIN_VREG1	129
/* The lane's byte offset into the LDS stack, lane * 4, computed once. */
#define KNOD_AMDGPU_LDS_BASE_VREG	130

#define KNOD_AMDGPU_RDNA_LDS_VREG0 70
static_assert(KNOD_AMDGPU_RDNA_LDS_VREG0 > KNOD_AMDGPU_PAGE_BASE_VREG_HI);
static_assert(KNOD_AMDGPU_RDNA_LDS_VREG0 + 2 < 80);

static unsigned int knod_bpf_lds_vreg(const struct knod_bpf_priv *priv,
				    unsigned int reg)
{
	if (priv->isa_version == 10 || priv->isa_version == 11)
		return KNOD_AMDGPU_RDNA_LDS_VREG0 + reg - KNOD_AMDGPU_STACK_WIN_VREG0;
	return reg;
}

/* One VGPR holds four bytes of packet, so what the cache can hold is decided
 * by how many VGPRs sit between its base and the stack.
 */

#define KNOD_BPF_PROG_BUF_SIZE		32768

/* Index for r64.
 * r64[TREG64_0]
 */
#define TREG64_0			0
#define TREG64_1			1
#define TREG64_2			2
#define TREG64_3			3
#define KEY_IN_PKT_64			TREG64_3
#define TREG64_4			4
#define TREG64_5			5
#define TREG64_6			6
#define TREG64_7			7
#define TREG64_8			8
#define TREG64_9			9
#define TREG64_10			10
#define KEY_IN_MAP_64			TREG64_10
#define TREG64_11			11
#define TREG64_12			12
#define TREG64_13			13
#define TREG64_14			14
#define TREG64_15			15
#define TREG64_16			16
#define TREG64_17			17

#define MAX_MAP_KEY_SIZE		56

/* Index for r32.
 * r32[TREG32_0_LO]
 */
#define TREG32_0_LO			0
#define TREG32_0_HI			1
#define TREG32_1_LO			2
#define TREG32_1_HI			3
#define TREG32_2_LO			4
#define TREG32_2_HI			5
#define TREG32_3_LO			6
#define KEY_IN_PKT_32			TREG32_3_LO
#define TREG32_3_HI			7
#define TREG32_4_LO			8
#define TREG32_4_HI			9
#define TREG32_5_LO			10
#define TREG32_5_HI			11
#define TREG32_6_LO			12
#define TREG32_6_HI			13
#define TREG32_7_LO			14
#define TREG32_7_HI			15
#define TREG32_8_LO			16
#define TREG32_8_HI			17
#define TREG32_9_LO			18
#define TREG32_9_HI			19
#define TREG32_10_LO			20
#define KEY_IN_MAP_32			TREG32_10_LO
#define TREG32_10_HI			21
#define TREG32_11_LO			22
#define TREG32_11_HI			23
#define TREG32_12_LO			24
#define TREG32_12_HI			25
#define TREG32_13_LO			26
#define TREG32_13_HI			27
#define TREG32_14_LO			28
#define TREG32_14_HI			29
#define TREG32_15_LO			30
#define TREG32_15_HI			31
#define TREG32_16_LO			32
#define TREG32_16_HI			33
#define TREG32_17_LO			34
#define TREG32_17_HI			35
#define TREG32_MAX			TREG32_17_HI

/*
 * The SGPR map, one layout for every generation so a dump reads the same
 * whichever GPU produced it and a prebuilt routine needs no shim to name a
 * register.
 *
 *+-------+-------+-------+-------+-------+-----+-----+
 *| s0-s3 | s4-s5 | s6-s7 | s8-s9 |s10-s11| s12 | s13 |
 *+-------+-------+-------+-------+-------+-----+-----+
 *|  PSB  | DISP  | QUEUE | KARG  |DISP_ID| WGX | WGY |
 *+-------+-------+-------+-------+-------+-----+-----+
 * The hardware loads these from the dispatch packet before the wave starts,
 * so nothing may be assigned there.  WGY doubles as the queue id.
 *
 *+---------+---------+---------+---------+---------+---------+---------+---------+
 *| s14-s25 | s26-s27 | s28-s29 | s30-s31 | s32-s33 | s34-s45 | s46-s49 | s50-s95 |
 *+---------+---------+---------+---------+---------+---------+---------+---------+
 *| TMP 0-5 |  PARAM  | FP/DESC |  PROBE  |DONE MASK|BLOB XSAV|BLOB TMP |EXEC SAVE|
 *+---------+---------+---------+---------+---------+---------+---------+---------+
 * plus s96:97 INIT EXEC.  Implicit: VCC = s[106:107]  EXEC = s[126:127].
 *
 * user_sgpr_count=12: private_segment_buffer(4) + dispatch(2) + queue(2) +
 * kernarg(2) + dispatch_id(2).  flat_scratch_init is disabled - the stack is
 * in LDS and nothing touches scratch - so workgroup_id_x lands right after at
 * s12.
 *
 * TMP holds nothing across a BPF instruction.  DONE MASK and INIT EXEC hold
 * theirs across the whole program, and EXEC SAVE across whichever BPF-level
 * scope was given the pair - so a spliced routine gets a window of its own
 * rather than any of those.  FP is written once in the prologue and never
 * read; s[28:29] carries the map descriptor into a routine.
 */
#define KNOD_AMDGPU_PSB_SREG		0  /* s[0:3] private_segment_buffer */
#define KNOD_AMDGPU_DISPATCH_PTR_SREG	4  /* s[4:5] dispatch_ptr */
#define KNOD_AMDGPU_ARG_SREG		4  /* alias for dispatch_ptr */
#define KNOD_AMDGPU_QUEUE_PTR_SREG	6  /* s[6:7] queue_ptr */
#define KNOD_AMDGPU_KERNARG_PTR_SREG	8  /* s[8:9] kernarg_segment_ptr */
#define KNOD_AMDGPU_DISPATCH_ID_SREG	10 /* s[10:11] dispatch_id */
#define KNOD_AMDGPU_WORKGROUP_ID_X_SREG	12 /* s12 workgroup_id_x */
#define KNOD_AMDGPU_WORKGROUP_ID_Y_SREG	13 /* s13 workgroup_id_y = queue_id */
#define KNOD_AMDGPU_TMP_SREG0_LO	14
#define KNOD_AMDGPU_TMP_SREG0_HI	15
#define KNOD_AMDGPU_TMP_SREG1_LO	16
#define KNOD_AMDGPU_TMP_SREG1_HI	17
#define KNOD_AMDGPU_TMP_SREG2_LO	18
#define KNOD_AMDGPU_TMP_SREG2_HI	19
#define KNOD_AMDGPU_TMP_SREG3_LO	20
#define KNOD_AMDGPU_TMP_SREG3_HI	21
#define KNOD_AMDGPU_TMP_SREG4_LO	22
#define KNOD_AMDGPU_TMP_SREG4_HI	23
#define KNOD_AMDGPU_TMP_SREG5_LO	24
#define KNOD_AMDGPU_TMP_SREG5_HI	25
#define KNOD_AMDGPU_PARAM_SREG_LO	26 /* s26 */
#define KNOD_AMDGPU_PARAM_SREG_HI	27 /* s27 */
#define KNOD_AMDGPU_FRAME_POINTER_SREG	28 /* s28 */

/* Structurized CFG: EXEC mask save/restore SGPRs.
 * done_mask tracks lanes that have reached BPF_EXIT.
 * exec_save pairs store EXEC at branch points for restore at merge points.
 *
 * One layout for every generation, at the same indices, so that a dump reads
 * the same whichever GPU produced it and a prebuilt routine needs no shim to
 * name a register.  That means taking what the narrowest generation allows:
 * GFX9 addresses s[0:101] where GFX10 and GFX11 reach s[0:105], and GFX9
 * hardware corrupts s[32:33].  The pairs the others could have had go unused.
 */
/* Common SGPR special register indices (same on GFX9 and GFX10) */
#define AMDGCN_SREG_VCC_LO		106
#define AMDGCN_SREG_EXEC_LO		126
#define AMDGCN_SREG_INTEGER_0		128
#define AMDGCN_SREG_INTEGER_1		129

/* s[32:33] - must not overlap TMP_SREGs */
#define KNOD_AMDGPU_DONE_MASK_SREG	32
/* s36-s51 belongs to whatever routine is spliced in: its own EXEC saves and
 * its scratch scalars.  A BPF-level scope cannot be given one of those,
 * because a splice inside the scope would overwrite it.
 */
#define KNOD_AMDGPU_EXEC_SAVE_SREG_BASE	(KNOD_BLOB_SPLICE_TMP_SREG_END + 1)
#define KNOD_AMDGPU_EXEC_SAVE_SREG_MAX	95
#define KNOD_AMDGPU_INITIAL_EXEC_SREG	96 /* in-bounds EXEC snapshot */
#define KNOD_AMDGPU_MAX_EXEC_SAVE_PAIRS					\
	((KNOD_AMDGPU_EXEC_SAVE_SREG_MAX -				\
	  KNOD_AMDGPU_EXEC_SAVE_SREG_BASE + 1) / 2)

/* The window is declared whole whether a program fills it or not, so this is
 * the same for every program.  Occupancy does not notice: a shader that asks
 * for all 256 VGPRs already gets one wave per SIMD, which no SGPR count can
 * lower.
 */
#define KNOD_AMDGPU_SGPRS_USED		(KNOD_AMDGPU_INITIAL_EXEC_SREG + 2)

/* The cycle probe holds two values across the program: what the prologue
 * took, and when it ended.  s[30:31] is the pair nothing else claims.
 * Everything else the probe needs it works out inside the epilogue, where the
 * scratch scalars are free again.
 */
#define KNOD_AMDGPU_PROBE_SREG0		30
#define KNOD_AMDGPU_PROBE_SREG1		31

enum knod_probe_stage {
	KNOD_PROBE_PRO_START,
	KNOD_PROBE_PRO_END,
	KNOD_PROBE_EPI_START,
	KNOD_PROBE_EPI_END,
};

static u8 knod_bpf_gfx9_sgpr_granule(unsigned int sgprs_used)
{
	if (sgprs_used <= 16)
		return 0;

	return 2 * (DIV_ROUND_UP(sgprs_used, 16) - 1);
}

unsigned int knod_bpf_workgroups = KNOD_BPF_WORKGROUPS_DEFAULT;
MODULE_PARM_DESC(workgroups, "Workgroup size, multiple of 64, Min(64) Default/Max(256)");
module_param_named(workgroups, knod_bpf_workgroups, int, 0600);

unsigned int knod_bpf_expire = KNOD_BPF_EXPIRE_DEFAULT;
MODULE_PARM_DESC(queue_expire, "Queue expire time(ms), Min(1), Default(10), Max(1000)");
module_param_named(queue_expire, knod_bpf_expire, int, 0600);

/* Where the shader's time goes, in shader clocks, split three ways: the
 * prologue, the program, and the epilogue.  Each wave writes its own three
 * into the tail of its ring slot, which spsc_bd leaves free, and the host
 * histograms them - so the answer is per wave rather than an average of the
 * whole dispatch.
 *
 * Only the kernel emitter grows the probe, so it wants jit_engine=0, and only
 * where there is a counter to read: GFX9 has none.
 */
unsigned int knod_bpf_cycle_probe;
MODULE_PARM_DESC(cycle_probe, "Time the shader in three parts, 0=Off(Default)");
module_param_named(cycle_probe, knod_bpf_cycle_probe, int, 0600);

/* Where the routines that have a prebuilt form come from.  "kernel" emits them
 * as it always has, and is what runs when no blob is installed; "blob" splices
 * in what was built outside.  The two are meant to produce the same bytes, so
 * this exists to check that they do - and to fall back if they ever do not.
 */
unsigned int knod_bpf_jit_engine = 1;
MODULE_PARM_DESC(jit_engine, "0=kernel, 1=blob(Default)");
module_param_named(jit_engine, knod_bpf_jit_engine, int, 0600);

/* The BPF stack always lives in LDS, laid out slot-major and reached through
 * a two-register window.  Keeping it out of the register file leaves the whole
 * upper half free for something else to hold, and LDS is cacheable where the
 * packet buffer in VRAM is not.
 *
 * Either engine will do.  A blob routine reaches no further than v69, because
 * the stack has always lived above that and a routine standing on it would
 * have broken the default long ago.
 */

static void knod_lshlrev32(struct knod_bpf_priv *priv,
			   struct knod_insn_meta *meta,
			   struct amdgcn_param32 dst,
			   struct amdgcn_param32 src0,
			   struct amdgcn_param32 src1);
static void knod_and32(struct knod_bpf_priv *priv,
		      struct knod_insn_meta *meta,
		      struct amdgcn_param32 dst,
		      struct amdgcn_param32 src0,
		      struct amdgcn_param32 src1);

/* The lane's byte offset into its slot-major LDS stack: (lane & (wg - 1)) * 4,
 * computed once into KNOD_AMDGPU_LDS_BASE_VREG for the window helpers to add to.
 */
static void knod_bpf_emit_lds_base_init(struct knod_bpf_priv *priv,
					struct knod_insn_meta *meta)
{
	struct amdgcn_param32 base, idx, k;

	knod_vset32(&base, knod_bpf_lds_vreg(priv, KNOD_AMDGPU_LDS_BASE_VREG));
	knod_vset32(&idx, KNOD_AMDGPU_IDX_VREG);
	knod_iset32(&k, knod_bpf_workgroups - 1);
	knod_and32(priv, meta, base, k, idx);
	knod_iset32(&k, 2);
	knod_lshlrev32(priv, meta, base, k, base);
}

/* Whether a workgroup takes the whole WGP.  In CU mode its waves sit on one CU
 * and share that CU's cache; in WGP mode they spread over both and reach more
 * of the memory pipe, but each half then fetches its own copy of whatever the
 * other half already had.  RDNA only - GCN has no WGP.
 *
 * Off, because sharing the cache is worth more here than the extra pipe: 30
 * Mpps against 20 (RDNA2, 511 flows).  Kept so that stays a measurement rather
 * than a belief.
 */
unsigned int knod_bpf_wgp;
MODULE_PARM_DESC(wgp, "Spread a workgroup over the WGP 0=Off(Default), 1=On");
module_param_named(wgp, knod_bpf_wgp, int, 0600);

#define KNOD_EA(extack, msg)   NL_SET_ERR_MSG_MOD((extack), msg)

DEFINE_STATIC_KEY_FALSE(knod_stats_key);

static const u32 bl_bounds[KNOD_BL_BUCKETS - 1] = {
	16, 64, 256, 1024, 4096, 8192, 16384
};

static const char * const lat_labels[] = {
	"< 1us", "1-2us", "2-4us", "4-8us", "8-16us",
	"16-32us", "32-64us", "64-128us", "128-256us", ">= 256us",
};

static const char * const bl_labels[] = {
	"1-16", "17-64", "65-256", "257-1K",
	"1K-4K", "4K-8K", "8K-16K", ">= 16K",
};

static LIST_HEAD(priv_list);
/* r64[0..17] describe the scratch pairs; r64[19] describes CTX. */
struct amdgcn_param64 r64[20], sr64[6], p64[4], bpf_reg64[11];
struct amdgcn_param32 r32[36];
/* Its address is a sentinel: a cache pointer equal to it means the stack, which
 * the load and store helpers route through the LDS window instead of reading.
 */
struct amdgcn_param32 stack[1];

struct amdgcn_label {
	struct knod_insn_meta *meta;
	int insn_idx;
};

struct amdgcn_branch_fixup {
	struct amdgcn_label *target_label;
	struct knod_insn_meta *meta;
	int insn_idx;
};

struct knod_accel_xdp_ops accel_xdp_ops;

static int knod_prog_prepare_insns(struct knod_bpf_priv *priv,
				   struct knod_prog *knod_prog);
static int knod_bpf_emit_epilogue(struct knod_bpf_priv *priv,
				  struct knod_prog *knod_prog);
static int knod_bpf_worker(void *arg);
static void knod_bpf_drain_worker(struct knod_bpf_priv *priv);
static void knod_prog_free(struct knod_prog *knod_prog);
static void knod_setup_bpf_prog(struct bpf_prog *prog);

static void knod_bpf_gpu_mem_fence(struct knod_bpf_priv *priv)
{
	if (!priv)
		return;

	/* drain the WC store buffer before the GPU reads the map */
	wmb();
}

static unsigned int knod_bpf_active_rxq_count(struct net_device *netdev)
{
	unsigned int nr_rxq;

	if (!netdev)
		return 0;

	nr_rxq = READ_ONCE(netdev->real_num_rx_queues);
	if (!nr_rxq)
		nr_rxq = netdev->num_rx_queues;

	/* Also cap by CPU count: a percpu map keeps one instance per work and
	 * aggregates per CPU, so more works than CPUs has nowhere to report the
	 * excess.  A NIC can have far more rx queues (bnxt: 80+) than either.
	 */
	nr_rxq = min_t(unsigned int, nr_rxq, KNOD_SPSC_MAX);
	return min_t(unsigned int, nr_rxq, num_possible_cpus());
}

#define KNOD_SQ_SIGNAL_OFFSET ALIGN(sizeof(struct knod_bpf_param), 64)
#define KNOD_SQ_PARAM_BYTES (KNOD_SQ_SIGNAL_OFFSET + sizeof(struct amd_signal))

static struct amd_signal *knod_bpf_sqw_signal(struct knod_bpf_work_sq *sqw)
{
	return (void *)((char *)sqw->param->kaddr + KNOD_SQ_SIGNAL_OFFSET);
}

#include "knod_persistent.h"

static bool knod_bpf_persistent;
module_param_named(persistent, knod_bpf_persistent, bool, 0444);
MODULE_PARM_DESC(persistent, "Keep one resident workgroup per RX queue");

struct knod_persistent_mem {
	struct knod_persistent_control control;
	struct amd_signal terminal;
};
static_assert(sizeof(struct knod_persistent_slot) == KNOD_PERSIST_SLOT_BYTES);
static_assert(offsetof(struct knod_persistent_control, slots) == KNOD_PERSIST_SLOT_BASE);
static_assert(offsetof(struct knod_persistent_slot, done) == KNOD_PERSIST_DONE);
static_assert(sizeof(struct knod_persistent_mem) <= PAGE_SIZE);

static bool knod_bpf_sqw_done(struct knod_bpf_priv *priv,
			      struct knod_bpf_work_sq *sqw)
{
	struct amd_signal *signal = knod_bpf_sqw_signal(sqw);
	struct knod_persistent_mem *mem;
	struct knod_persistent_slot *slot;
	unsigned int i;

	if (!READ_ONCE(signal->value))
		return true;
	if (!knod_bpf_persistent)
		return false;
	mem = priv->persistent_mem->kaddr;
	slot = &mem->control.slots[sqw->persistent_slot];
	for (i = 0; i < priv->nr_works; i++)
		if (READ_ONCE(slot->done[i]) != sqw->persistent_sequence)
			return false;
	dma_rmb();
	WRITE_ONCE(signal->value, 0);
	return true;
}

static void knod_bpf_persistent_stop(struct knod_bpf_priv *priv)
{
	struct knod_persistent_mem *mem;
	unsigned long deadline;
	bool warned = false;

	if (!priv->persistent_running)
		return;
	might_sleep();
	mem = priv->persistent_mem->kaddr;
	/* Caller has drained every published chunk. Never discard a request. */
	WARN_ON_ONCE(priv->inflight_cnt);
	dma_wmb();
	WRITE_ONCE(mem->control.stop, 1);
	deadline = jiffies + msecs_to_jiffies(1000);
	while (READ_ONCE(mem->terminal.value)) {
		if (!warned && time_after(jiffies, deadline)) {
			pr_warn("knod: retaining resident shader backing pending terminal completion\n");
			warned = true;
		}
		usleep_range(100, 200);
	}
	dma_rmb();
	priv->persistent_running = false;
}

static void knod_bpf_fill_dispatch(struct knod_bpf_priv *priv,
				   struct knod_bpf_work_sq *sqw,
				   struct knod_dispatch_params *p)
{
	struct knod_bpf_param *param = sqw->param->kaddr;
	int idx = READ_ONCE(priv->active_idx);

	p->workgroup_size_x = knod_bpf_workgroups;
	p->grid_size_x = priv->batch_size;
	p->grid_size_y = param->nr_queues;
	p->private_segment_size = 0;
	p->group_segment_size = priv->lds_bytes[idx];
	p->kernel_object = (u64)priv->knod->kernels[idx]->gaddr;
	p->kernarg_address = sqw->param->gaddr;
	p->completion_signal = sqw->param->gaddr + KNOD_SQ_SIGNAL_OFFSET;
}

static void debug_kernel_descriptor(struct kernel_descriptor *kernel_code)
{
	knod_jit_dbg(" kernel_code->group_segment_fixed_size = %d\n",
		kernel_code->group_segment_fixed_size);
	knod_jit_dbg(" kernel_code->private_segment_fixed_size = %d\n",
		kernel_code->private_segment_fixed_size);
	knod_jit_dbg(" kernel_code->kernarg_size = %d\n",
		kernel_code->kernarg_size);
	knod_jit_dbg(" kernel_code->kernel_code_entry_byte_offset = %lld\n",
		kernel_code->kernel_code_entry_byte_offset);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc3.accum_offset = %d\n",
		kernel_code->compute_pgm_rsrc3.accum_offset);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc3.reserved0 = %d\n",
		kernel_code->compute_pgm_rsrc3.reserved0);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc3.tg_split = %d\n",
		kernel_code->compute_pgm_rsrc3.tg_split);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc3.reserved1 = %d\n",
		kernel_code->compute_pgm_rsrc3.reserved1);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc1.granulated_workitem_vgpr_count = %d\n",
		kernel_code->compute_pgm_rsrc1.granulated_workitem_vgpr_count);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc1.granulated_wavefront_sgpr_count = %d\n",
		kernel_code->compute_pgm_rsrc1.granulated_wavefront_sgpr_count);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc1.priority = %d\n",
		kernel_code->compute_pgm_rsrc1.priority);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc1.float_round_mode_32 = %d\n",
		kernel_code->compute_pgm_rsrc1.float_round_mode_32);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc1.float_round_mode_16_64 = %d\n",
		kernel_code->compute_pgm_rsrc1.float_round_mode_16_64);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc1.float_denorm_mode_32 = %d\n",
		kernel_code->compute_pgm_rsrc1.float_denorm_mode_32);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc1.float_denorm_mode_16_64 = %d\n",
		kernel_code->compute_pgm_rsrc1.float_denorm_mode_16_64);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc1.priv = %d\n",
		kernel_code->compute_pgm_rsrc1.priv);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc1.enable_dx10_clamp = %d\n",
		kernel_code->compute_pgm_rsrc1.enable_dx10_clamp);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc1.debug_mode = %d\n",
		kernel_code->compute_pgm_rsrc1.debug_mode);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc1.enable_ieee_mode = %d\n",
		kernel_code->compute_pgm_rsrc1.enable_ieee_mode);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc1.bulky = %d\n",
		kernel_code->compute_pgm_rsrc1.bulky);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc1.cdbg_user = %d\n",
		kernel_code->compute_pgm_rsrc1.cdbg_user);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc1.fp16_ovfl = %d\n",
		kernel_code->compute_pgm_rsrc1.fp16_ovfl);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc1.reserved0 = %d\n",
		kernel_code->compute_pgm_rsrc1.reserved0);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc1.wgp_mode = %d\n",
		kernel_code->compute_pgm_rsrc1.wgp_mode);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc1.mem_ordered = %d\n",
		kernel_code->compute_pgm_rsrc1.mem_ordered);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc1.fwd_progress = %d\n",
		kernel_code->compute_pgm_rsrc1.fwd_progress);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc2.enable_private_segment = %d\n",
		kernel_code->compute_pgm_rsrc2.enable_private_segment);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc2.user_sgpr_count = %d\n",
		kernel_code->compute_pgm_rsrc2.user_sgpr_count);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc2.enable_trap_handler = %d\n",
		kernel_code->compute_pgm_rsrc2.enable_trap_handler);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc2.enable_sgpr_workgroup_id_x = %d\n",
		kernel_code->compute_pgm_rsrc2.enable_sgpr_workgroup_id_x);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc2.enable_sgpr_workgroup_id_y = %d\n",
		kernel_code->compute_pgm_rsrc2.enable_sgpr_workgroup_id_y);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc2.enable_sgpr_workgroup_id_z = %d\n",
		kernel_code->compute_pgm_rsrc2.enable_sgpr_workgroup_id_z);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc2.enable_sgpr_workgroup_info = %d\n",
		kernel_code->compute_pgm_rsrc2.enable_sgpr_workgroup_info);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc2.enable_vgpr_workitem_id = %d\n",
		kernel_code->compute_pgm_rsrc2.enable_vgpr_workitem_id);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc2.enable_exception_address_watch = %d\n",
		kernel_code->compute_pgm_rsrc2.enable_exception_address_watch);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc2.enable_exception_memory = %d\n",
		kernel_code->compute_pgm_rsrc2.enable_exception_memory);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc2.granulated_lds_size = %d\n",
		kernel_code->compute_pgm_rsrc2.granulated_lds_size);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc2.enable_exception_ieee_754_fp_invalid_operation = %d\n",
		kernel_code->compute_pgm_rsrc2
			.enable_exception_ieee_754_fp_invalid_operation);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc2.enable_exception_fp_denormal_source = %d\n",
		kernel_code->compute_pgm_rsrc2
			.enable_exception_fp_denormal_source);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc2.enable_exception_ieee_754_fp_division_by_zero = %d\n",
		kernel_code->compute_pgm_rsrc2
			.enable_exception_ieee_754_fp_division_by_zero);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc2.enable_exception_ieee_754_fp_overflow = %d\n",
		kernel_code->compute_pgm_rsrc2
			.enable_exception_ieee_754_fp_overflow);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc2.enable_exception_ieee_754_fp_underflow = %d\n",
		kernel_code->compute_pgm_rsrc2
			.enable_exception_ieee_754_fp_underflow);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc2.enable_exception_ieee_754_fp_inexact = %d\n",
		kernel_code->compute_pgm_rsrc2
			.enable_exception_ieee_754_fp_inexact);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc2.enable_exception_int_divide_by_zero = %d\n",
		kernel_code->compute_pgm_rsrc2
			.enable_exception_int_divide_by_zero);
	knod_jit_dbg(" kernel_code->compute_pgm_rsrc2.reserved0 = %d\n",
		kernel_code->compute_pgm_rsrc2.reserved0);
	knod_jit_dbg(" kernel_code->code_properties.enable_sgpr_private_segment_buffer = %d\n",
		kernel_code->code_properties
			.enable_sgpr_private_segment_buffer);
	knod_jit_dbg(" kernel_code->code_properties.enable_sgpr_dispatch_ptr = %d\n",
		kernel_code->code_properties.enable_sgpr_dispatch_ptr);
	knod_jit_dbg(" kernel_code->code_properties.enable_sgpr_queue_ptr = %d\n",
		kernel_code->code_properties.enable_sgpr_queue_ptr);
	knod_jit_dbg(" kernel_code->code_properties.enable_sgpr_kernarg_segment_ptr = %d\n",
		kernel_code->code_properties.enable_sgpr_kernarg_segment_ptr);
	knod_jit_dbg(" kernel_code->code_properties.enable_sgpr_dispatch_id = %d\n",
		kernel_code->code_properties.enable_sgpr_dispatch_id);
	knod_jit_dbg(" kernel_code->code_properties.enable_sgpr_flat_scratch_init = %d\n",
		kernel_code->code_properties.enable_sgpr_flat_scratch_init);
	knod_jit_dbg(" kernel_code->code_properties.enable_sgpr_private_segment_size = %d\n",
		kernel_code->code_properties.enable_sgpr_private_segment_size);
	knod_jit_dbg(" kernel_code->code_properties.reserved0 = %d\n",
		kernel_code->code_properties.reserved0);
	knod_jit_dbg(" kernel_code->code_properties.enable_wavefront_size32 = %d\n",
		kernel_code->code_properties.enable_wavefront_size32);
	knod_jit_dbg(" kernel_code->code_properties.uses_dynamic_stack = %d\n",
		kernel_code->code_properties.uses_dynamic_stack);
	knod_jit_dbg(" kernel_code->code_properties.reserved1 = %d\n",
		kernel_code->code_properties.reserved1);
}

static void kfd_kernel_gfx9_init(struct knod *knod)
{
	struct kernel_descriptor *kernel_code = knod->kernels[0]->kaddr;

	kernel_code->group_segment_fixed_size = 0;
	kernel_code->private_segment_fixed_size = 0;
	kernel_code->kernarg_size = 64;
	kernel_code->kernel_code_entry_byte_offset = 1024;

	/* GFX10+ or GFX90A+ */
	kernel_code->compute_pgm_rsrc3.accum_offset = 0;
	kernel_code->compute_pgm_rsrc3.reserved0 = 0;
	kernel_code->compute_pgm_rsrc3.tg_split = 0;
	kernel_code->compute_pgm_rsrc3.reserved1 = 0;

	kernel_code->compute_pgm_rsrc1.granulated_workitem_vgpr_count =
		(256 / 4) - 1;
	kernel_code->compute_pgm_rsrc1.granulated_wavefront_sgpr_count =
		knod_bpf_gfx9_sgpr_granule(KNOD_AMDGPU_SGPRS_USED);
	kernel_code->compute_pgm_rsrc1.priority = 0;
	kernel_code->compute_pgm_rsrc1.float_round_mode_32 = 0;
	kernel_code->compute_pgm_rsrc1.float_round_mode_16_64 = 0;
	kernel_code->compute_pgm_rsrc1.float_denorm_mode_32 = 3;
	kernel_code->compute_pgm_rsrc1.float_denorm_mode_16_64 = 3;
	kernel_code->compute_pgm_rsrc1.priv = 0;
	kernel_code->compute_pgm_rsrc1.enable_dx10_clamp = 1;
	kernel_code->compute_pgm_rsrc1.debug_mode = 0;
	kernel_code->compute_pgm_rsrc1.enable_ieee_mode = 1;
	kernel_code->compute_pgm_rsrc1.bulky = 0;
	kernel_code->compute_pgm_rsrc1.cdbg_user = 0;
	kernel_code->compute_pgm_rsrc1.fp16_ovfl = 0;
	kernel_code->compute_pgm_rsrc1.reserved0 = 0;
	kernel_code->compute_pgm_rsrc1.wgp_mode = 0;
	kernel_code->compute_pgm_rsrc1.mem_ordered = 0;
	kernel_code->compute_pgm_rsrc1.fwd_progress = 0;

	kernel_code->compute_pgm_rsrc2.enable_private_segment = 0;
	kernel_code->compute_pgm_rsrc2.user_sgpr_count = 12; /* 4+2+2+2+2 */
	kernel_code->compute_pgm_rsrc2.enable_trap_handler = 0;
	kernel_code->compute_pgm_rsrc2.enable_sgpr_workgroup_id_x = 1;
	kernel_code->compute_pgm_rsrc2.enable_sgpr_workgroup_id_y = 1;
	kernel_code->compute_pgm_rsrc2.enable_sgpr_workgroup_id_z = 0;
	kernel_code->compute_pgm_rsrc2.enable_sgpr_workgroup_info = 0;
	kernel_code->compute_pgm_rsrc2.enable_vgpr_workitem_id = 0;
	kernel_code->compute_pgm_rsrc2.enable_exception_address_watch = 0;
	kernel_code->compute_pgm_rsrc2.enable_exception_memory = 0;
	kernel_code->compute_pgm_rsrc2.granulated_lds_size = 0;
	kernel_code->compute_pgm_rsrc2
		.enable_exception_ieee_754_fp_invalid_operation = 0;
	kernel_code->compute_pgm_rsrc2.enable_exception_fp_denormal_source = 0;
	kernel_code->compute_pgm_rsrc2
		.enable_exception_ieee_754_fp_division_by_zero = 0;
	kernel_code->compute_pgm_rsrc2
		.enable_exception_ieee_754_fp_overflow = 0;
	kernel_code->compute_pgm_rsrc2
		.enable_exception_ieee_754_fp_underflow = 0;
	kernel_code->compute_pgm_rsrc2.enable_exception_ieee_754_fp_inexact = 0;
	kernel_code->compute_pgm_rsrc2.enable_exception_int_divide_by_zero = 0;
	kernel_code->compute_pgm_rsrc2.reserved0 = 0;

	/*
	 * User SGPR layout - loaded in fixed order, disabled entries are
	 * skipped (not reserved).  The resulting SGPR map depends on which
	 * flags are enabled:
	 *
	 *   enable_sgpr_private_segment_buffer  -> 4 SGPRs  (s[0:3])
	 *   enable_sgpr_dispatch_ptr            -> 2 SGPRs  (s[4:5])
	 *   enable_sgpr_queue_ptr               -> 2 SGPRs
	 *   enable_sgpr_kernarg_segment_ptr     -> 2 SGPRs
	 *   enable_sgpr_dispatch_id             -> 2 SGPRs
	 *   enable_sgpr_flat_scratch_init       -> disabled (LDS stack)
	 *   enable_sgpr_private_segment_size    -> 1 SGPR
	 *
	 * System SGPRs (WorkgroupId etc.) follow immediately after the
	 * last user SGPR.  user_sgpr_count must match the total above.
	 */
	/* 4 SGPRs */
	kernel_code->code_properties.enable_sgpr_private_segment_buffer = 1;
	/* 2 SGPRs */
	kernel_code->code_properties.enable_sgpr_dispatch_ptr = 1;
	/* 2 SGPRs */
	kernel_code->code_properties.enable_sgpr_queue_ptr = 1;
	/* 2 SGPRs */
	kernel_code->code_properties.enable_sgpr_kernarg_segment_ptr = 1;
	/* 2 SGPRs */
	kernel_code->code_properties.enable_sgpr_dispatch_id = 1;
	/* 2 SGPRs */
	/* disabled -> the stack is in LDS, nothing touches scratch */
	kernel_code->code_properties.enable_sgpr_flat_scratch_init = 0;
	kernel_code->code_properties.enable_sgpr_private_segment_size = 0;
	/* total = 12 SGPRs, workgroup_id_x lands at s12 */
	kernel_code->code_properties.reserved0 = 0;
	/* GFX10+ */
	kernel_code->code_properties.enable_wavefront_size32 = 0;
	kernel_code->code_properties.uses_dynamic_stack = 0;
	kernel_code->code_properties.reserved1 = 0;

	debug_kernel_descriptor(kernel_code);
}

/* gfx10 and gfx11 want the same descriptor.  Every field that is per
 * generation - the VGPR granule, the reserved SGPR count, wave size,
 * mem_ordered - has the same value on both, which is what the IPsec
 * shader found when it was measured on each.
 */
static unsigned int knod_bpf_vgpr_reserve = 80;
module_param_named(vgpr_reserve, knod_bpf_vgpr_reserve, uint, 0444);
MODULE_PARM_DESC(vgpr_reserve, "Native VGPR allocation: 80, 128, or 256");

static void kfd_kernel_rdna_init(struct knod *knod)
{
	struct kernel_descriptor *kernel_code = knod->kernels[0]->kaddr;

	kernel_code->group_segment_fixed_size = 0;
	kernel_code->private_segment_fixed_size = 0;
	kernel_code->kernarg_size = 64;
	kernel_code->kernel_code_entry_byte_offset = 1024;

	/*
	 * User SGPR layout - loaded in fixed order, disabled entries are
	 * skipped (not reserved).  The resulting SGPR map depends on which
	 * flags are enabled:
	 *
	 *   enable_sgpr_private_segment_buffer  -> 4 SGPRs  (s[0:3])
	 *   enable_sgpr_dispatch_ptr            -> 2 SGPRs  (s[4:5])
	 *   enable_sgpr_queue_ptr               -> 2 SGPRs
	 *   enable_sgpr_kernarg_segment_ptr     -> 2 SGPRs
	 *   enable_sgpr_dispatch_id             -> 2 SGPRs
	 *   enable_sgpr_flat_scratch_init       -> disabled (LDS stack)
	 *   enable_sgpr_private_segment_size    -> 1 SGPR
	 *
	 * System SGPRs (WorkgroupId etc.) follow immediately after the
	 * last user SGPR.  user_sgpr_count must match the total above.
	 */
	/* 4 SGPRs */
	kernel_code->code_properties.enable_sgpr_private_segment_buffer = 1;
	/* 2 SGPRs */
	kernel_code->code_properties.enable_sgpr_dispatch_ptr = 1;
	/* 2 SGPRs */
	kernel_code->code_properties.enable_sgpr_queue_ptr = 1;
	/* 2 SGPRs */
	kernel_code->code_properties.enable_sgpr_kernarg_segment_ptr = 1;
	/* 2 SGPRs */
	kernel_code->code_properties.enable_sgpr_dispatch_id = 1;
	/* 2 SGPRs */
	/* disabled -> the stack is in LDS, nothing touches scratch */
	kernel_code->code_properties.enable_sgpr_flat_scratch_init = 0;
	kernel_code->code_properties.enable_sgpr_private_segment_size = 0;
	/* total = 12 SGPRs, workgroup_id_x lands at s12 */
	kernel_code->code_properties.reserved0 = 0;
	kernel_code->code_properties.enable_wavefront_size32 = 0;
	kernel_code->code_properties.uses_dynamic_stack = 0;
	kernel_code->code_properties.reserved1 = 0;

	kernel_code->compute_pgm_rsrc3.accum_offset = 0;
	kernel_code->compute_pgm_rsrc3.reserved0 = 0;
	kernel_code->compute_pgm_rsrc3.tg_split = 0;
	kernel_code->compute_pgm_rsrc3.reserved1 = 0;

	if (kernel_code->code_properties.enable_wavefront_size32 == 1)
		kernel_code->compute_pgm_rsrc1.granulated_workitem_vgpr_count =
			(knod_bpf_vgpr_reserve / 8) - 1;
	else
		kernel_code->compute_pgm_rsrc1.granulated_workitem_vgpr_count =
			(knod_bpf_vgpr_reserve / 4) - 1;
	kernel_code->compute_pgm_rsrc1.granulated_wavefront_sgpr_count = 0;
	kernel_code->compute_pgm_rsrc1.priority = 0;
	kernel_code->compute_pgm_rsrc1.float_round_mode_32 = 0;
	kernel_code->compute_pgm_rsrc1.float_round_mode_16_64 = 0;
	kernel_code->compute_pgm_rsrc1.float_denorm_mode_32 = 3;
	kernel_code->compute_pgm_rsrc1.float_denorm_mode_16_64 = 3;
	kernel_code->compute_pgm_rsrc1.priv = 0;
	kernel_code->compute_pgm_rsrc1.enable_dx10_clamp = 1;
	kernel_code->compute_pgm_rsrc1.debug_mode = 0;
	kernel_code->compute_pgm_rsrc1.enable_ieee_mode = 1;
	kernel_code->compute_pgm_rsrc1.bulky = 0;
	kernel_code->compute_pgm_rsrc1.cdbg_user = 0;
	kernel_code->compute_pgm_rsrc1.fp16_ovfl = 0;
	kernel_code->compute_pgm_rsrc1.reserved0 = 0;
	kernel_code->compute_pgm_rsrc1.wgp_mode = !!knod_bpf_wgp;
	kernel_code->compute_pgm_rsrc1.mem_ordered = 1;
	kernel_code->compute_pgm_rsrc1.fwd_progress = 0;

	kernel_code->compute_pgm_rsrc2.enable_private_segment = 0;
	kernel_code->compute_pgm_rsrc2.user_sgpr_count = 12; /* 4+2+2+2+2 */
	kernel_code->compute_pgm_rsrc2.enable_trap_handler = 0;
	kernel_code->compute_pgm_rsrc2.enable_sgpr_workgroup_id_x = 1;
	kernel_code->compute_pgm_rsrc2.enable_sgpr_workgroup_id_y = 1;
	kernel_code->compute_pgm_rsrc2.enable_sgpr_workgroup_id_z = 0;
	kernel_code->compute_pgm_rsrc2.enable_sgpr_workgroup_info = 0;
	kernel_code->compute_pgm_rsrc2.enable_vgpr_workitem_id = 1;
	kernel_code->compute_pgm_rsrc2.enable_exception_address_watch = 0;
	kernel_code->compute_pgm_rsrc2.enable_exception_memory = 0;
	kernel_code->compute_pgm_rsrc2.granulated_lds_size = 0;
	kernel_code->compute_pgm_rsrc2
		.enable_exception_ieee_754_fp_invalid_operation = 0;
	kernel_code->compute_pgm_rsrc2.enable_exception_fp_denormal_source = 0;
	kernel_code->compute_pgm_rsrc2
		.enable_exception_ieee_754_fp_division_by_zero = 0;
	kernel_code->compute_pgm_rsrc2
		.enable_exception_ieee_754_fp_overflow = 0;
	kernel_code->compute_pgm_rsrc2
		.enable_exception_ieee_754_fp_underflow = 0;
	kernel_code->compute_pgm_rsrc2.enable_exception_ieee_754_fp_inexact = 0;
	kernel_code->compute_pgm_rsrc2.enable_exception_int_divide_by_zero = 0;
	kernel_code->compute_pgm_rsrc2.reserved0 = 0;

	debug_kernel_descriptor(kernel_code);
}

static int kfd_kernel_init(struct knod *knod, struct knod_bpf_priv *priv)
{
	struct kernel_descriptor *kd;

	if (!knod->kernels[1])
		return -ENOMEM;

	/*
	 * Pass-through starts on slot 0; the first XDP prog attach stages into
	 * slot 1 and flips the active index there, ping-ponging on each
	 * install.
	 */
	priv->active_idx = 0;

	if (priv->isa_version == 9)
		kfd_kernel_gfx9_init(knod);
	else
		kfd_kernel_rdna_init(knod);

	/*
	 * Slot 1 must carry the same kernel-descriptor as slot 0 -- gfx init
	 * only touches slot 0, and slot 1's BO is otherwise uninitialised,
	 * which stalls the compute queue.  Copy the kd + pre-code region.
	 */
	kd = knod->kernels[0]->kaddr;
	memcpy(knod->kernels[1]->kaddr, knod->kernels[0]->kaddr,
	       kd->kernel_code_entry_byte_offset);
	knod_bpf_gpu_mem_fence(priv);

	return 0;
}

static struct knod_bpf_work_sq *
__knod_get_free_work_sq(struct knod_bpf_priv *priv)
{
	return list_first_entry_or_null(&priv->free_list_sqw,
					struct knod_bpf_work_sq, list);
}

/* Prepare a dispatch: peek SPSC rings and fill params, but do not submit.
 * Returns the prepared sqw (with backlogs > 0), or NULL if nothing to do.
 *
 * A single in-flight AQL queue means the worker never has to reserve SPSC
 * ranges ahead of the current dispatch. The SPSC acquired pointer is advanced
 * only after the GPU finishes the dispatch that consumed those entries.
 */
static struct knod_bpf_work_sq *knod_prepare_bpf(struct knod_bpf_priv *priv)
{
	int i, cnt, backlogs = 0;
	struct knod_dev *knodev = priv->knodev;
	struct knod_bpf_work_sq *sqw;
	struct knod_bpf_param *param;

	if (READ_ONCE(priv->installing_kernel))
		return NULL;

	if (!priv->pass_prog_buf && !READ_ONCE(priv->prog))
		return NULL;

	sqw = __knod_get_free_work_sq(priv);
	if (!sqw)
		return NULL;

	param = (struct knod_bpf_param *)sqw->param->kaddr;
	memset(sqw->queue_idx, 0, sizeof(sqw->queue_idx));

	/* 2D dispatch: queue_id = workgroup_id_y, tid = workitem within WG.
	 * The shader indexes sub[] by (queue_id * batch_size + tid) and reads
	 * the descriptor out of the SPSC pool itself, so all that is wanted
	 * here is how many each queue has.  No cumulative start_idx -- each
	 * queue's slot range is fixed by i.
	 */
	for (i = 0; i < priv->nr_works; i++) {
		unsigned int skip = 0, j;

		/* Stage past every in-flight dispatch's claim on this queue so
		 * the new sqw reads disjoint SPSC slots.  Peek self-limits: if
		 * the ring holds fewer entries past @skip, cnt shrinks (or 0).
		 */
		for (j = 0; j < priv->inflight_cnt; j++)
			skip += priv->inflight[j]->queue_idx[i];

		param->queues[i].count = 0;
		spsc_peek_count(&knodev->wpriv[i].spsc_bds, skip,
				priv->batch_size, &cnt);
		if (!cnt) {
			sqw->queue_idx[i] = 0;
			param->queues[i].count = 0;
			continue;
		}

		/* Fill queue descriptor for GPU direct SPSC read.
		 * ring_start is the absolute ring position where this sqw
		 * begins - shader reads slots[(ring_start + tid) & mask].
		 * Offset by skip to keep staged sqws disjoint.
		 */
		param->queues[i].pool_gaddr = knodev->wpriv[i].spsc_pool_gaddr;
		param->queues[i].base_gaddr = priv->queue_base_gaddr[i];
		param->queues[i].count = cnt;
		param->queues[i].rx_geometry = knodev->wpriv[i].rx_geometry;
		param->queues[i].ring_start =
			knodev->wpriv[i].spsc_bds.acquired + skip;
		param->queues[i].ring_mask =
			knodev->wpriv[i].spsc_bds.mask;

		backlogs += cnt;
		sqw->queue_idx[i] = cnt;
	}
	sqw->backlogs = backlogs;
	param->nr_backlogs = backlogs;
	param->nr_queues = priv->nr_works;
	/* From the ring rather than worked out again here: the shader walks the
	 * pool with this, so it has to be what the pool was laid out with.
	 */
	param->spsc_stride = spsc_elem_size(&knodev->wpriv[0].spsc_bds);
	param->batch_shift = ilog2(priv->batch_size);
	param->wg_shift = ilog2(knod_bpf_workgroups);
	param->page_shift = PAGE_SHIFT;
	param->spsc_shift = ilog2(param->spsc_stride);
	for (i = 0; i < priv->nr_works; i++) {
		param->pass_count[i] = 0;
		param->pass_meta_buf_gaddr[i] = priv->pass_meta_buf ?
			priv->pass_meta_buf->gaddr +
			(u64)i * priv->pass_pkts_per_queue *
			KNOD_PASS_SLOT_SIZE :
			0;
	}
	param->ktime_ns = ktime_get_ns();

	if (!sqw->backlogs)
		return NULL;

	list_del_init(&sqw->list);
	return sqw;
}

static void knod_bpf_publish_chunk(struct knod_bpf_priv *priv,
				   struct knod_bpf_work_sq *sqw)
{
	struct knod_persistent_mem *mem = priv->persistent_mem->kaddr;
	struct knod_persistent_slot *slot;
	struct knod_dispatch_params p;

	if (!priv->persistent_running) {
		memset(mem, 0, sizeof(*mem));
		mem->control.version = KNOD_PERSIST_VERSION;
		mem->terminal = *(struct amd_signal *)priv->knod->kaql[0].queue_signal->kaddr;
		mem->terminal.value = 1;
		priv->persistent_sequence = 0;
		priv->persistent_slot = 0;
	}
	/* 64-bit sequence reuse would require a drained restart. */

	sqw->persistent_sequence = ++priv->persistent_sequence;
	sqw->persistent_slot = priv->persistent_slot;
	slot = &mem->control.slots[priv->persistent_slot];
	priv->persistent_slot = (priv->persistent_slot + 1) % KNOD_PERSIST_SLOTS;
	WRITE_ONCE(slot->param, sqw->param->gaddr);
	wmb();
	WRITE_ONCE(slot->ready, sqw->persistent_sequence);
	dma_wmb();
	if (priv->persistent_running)
		return;
	knod_bpf_fill_dispatch(priv, sqw, &p);
	p.kernarg_address = priv->persistent_mem->gaddr;
	p.completion_signal = priv->persistent_mem->gaddr +
		offsetof(struct knod_persistent_mem, terminal);
	wmb();
	knod_setup_header(priv->knod, &p, 0);
	priv->persistent_running = true;
	priv->persistent_launches++;
}

/* Submit a prepared sqw: write AQL packet, ring doorbell, record stats. */
static void knod_submit_bpf(struct knod_bpf_priv *priv,
			     struct knod_bpf_work_sq *sqw)
{
	struct amd_signal *signal =
		knod_bpf_sqw_signal(sqw);
	struct knod_bpf_stats *stats = &priv->stats;
	struct knod_dispatch_params p;
	int i, bucket = KNOD_BL_BUCKETS - 1;

	/* Only the owner resets this signal after its previous completion. */
	WARN_ON_ONCE(READ_ONCE(signal->value));
	WRITE_ONCE(signal->value, 1);
	dma_wmb();
	sqw->expire = jiffies + msecs_to_jiffies(knod_bpf_expire);
	if (static_branch_unlikely(&knod_stats_key)) {
		sqw->dispatch_time = ktime_get();

		/* Rate is packets over the time packets were flowing, not over
		 * however long ago the counters were reset.
		 */
		if (!stats->first_dispatch_ns)
			stats->first_dispatch_ns =
				ktime_to_ns(sqw->dispatch_time);
		stats->last_dispatch_ns = ktime_to_ns(sqw->dispatch_time);

		stats->backlogs_total += sqw->backlogs;
		for (i = 0; i < KNOD_BL_BUCKETS - 1; i++) {
			if (sqw->backlogs <= bl_bounds[i]) {
				bucket = i;
				break;
			}
		}
		stats->backlogs_hist[bucket]++;
	}

	if (knod_bpf_persistent) {
		knod_bpf_publish_chunk(priv, sqw);
		return;
	}
	knod_bpf_fill_dispatch(priv, sqw, &p);
	/* publish dispatch params before the AQL packet becomes visible */
	wmb();
	knod_setup_header(priv->knod, &p, 0);
}

/* Phase 1: advance SPSC consumer pointers so next dispatch can peek
 * new entries.
 */
/* Count what ran this dispatch's packets, before the cursor moves past them.
 * A dispatch completes oldest first, so its slots start where the ring was
 * acquired to.
 *
 * Two numbers, and the second is the one worth having.  The histogram counts
 * packets per unit over the whole run, which says only which units the device
 * has: the hardware rotates workgroups around them, so given enough dispatches
 * every unit shows up however few a dispatch uses at once.  The distinct count
 * per dispatch is what says whether asking for more workgroups per queue
 * actually reaches more units.
 */
/* Gather what the cycle probe left in each slot's spare half.  The counter is
 * twenty bits, so a difference is only right modulo that - which is where the
 * mask comes from, and why a part that really did run longer than the counter
 * takes to wrap reads as a small number rather than a large one.
 */
static void knod_cycle_count(struct knod_bpf_priv *priv,
			     struct knod_bpf_work_sq *sqw)
{
	struct knod_dev *knodev = priv->knodev;
	struct spsc_ring *r;
	struct spsc_bd *bd;
	unsigned int k;
	int i, j;

	for (i = 0; i < priv->nr_works; i++) {
		if (sqw->queue_idx[i] < 1)
			continue;

		r = &knodev->wpriv[i].spsc_bds;
		for (k = 0; k < sqw->queue_idx[i]; k++) {
			const u32 *probe;

			bd = r->slots[(r->acquired + k) & r->mask];
			probe = (const u32 *)((const char *)bd +
					      KNOD_BLOB_BD_PROBE);
			for (j = 0; j < KNOD_PROBE_PARTS; j++) {
				u32 c = probe[j] & 0xfffff;

				priv->stats.cyc_total[j] += c;
				if (c > priv->stats.cyc_max[j])
					priv->stats.cyc_max[j] = c;
			}
			priv->stats.cyc_count++;
		}
	}
}

static void knod_hwid_count(struct knod_bpf_priv *priv,
			    struct knod_bpf_work_sq *sqw)
{
	DECLARE_BITMAP(seen, KNOD_HWID_SLOTS);
	struct knod_dev *knodev = priv->knodev;
	struct spsc_ring *r;
	struct spsc_bd *bd;
	unsigned int k, unit;
	int i;

	bitmap_zero(seen, KNOD_HWID_SLOTS);

	for (i = 0; i < priv->nr_works; i++) {
		if (sqw->queue_idx[i] < 1)
			continue;

		r = &knodev->wpriv[i].spsc_bds;
		for (k = 0; k < sqw->queue_idx[i]; k++) {
			bd = r->slots[(r->acquired + k) & r->mask];
			unit = knod_hwid_unit((u32)(bd->act >> 32),
					      priv->isa_version);
			priv->stats.hwid_hist[unit]++;
			__set_bit(unit, seen);
		}
	}

	priv->stats.hwid_units_total += bitmap_weight(seen, KNOD_HWID_SLOTS);
	priv->stats.hwid_dispatches++;
}

/* Issue the device->host copy for the PASS bds of a completed dispatch, from
 * the worker rather than the NIC NAPI, so delivery runs on its own thread.  The
 * completed window is read before spsc_acquire publishes it to the act handler.
 */
static void knod_bpf_d2h_pass(struct knod_bpf_priv *priv,
			      struct knod_bpf_work_sq *sqw)
{
	struct spsc_pass_bd pass[KNOD_DEFAULT_PASS_SLOTS];
	struct knod_dev *knodev = priv->knodev;
	struct spsc_ring *r;
	struct spsc_bd *bd;
	unsigned int k;
	int i, n;

	for (i = 0; i < priv->nr_works; i++) {
		if (sqw->queue_idx[i] < 1)
			continue;

		r = &knodev->wpriv[i].spsc_bds;
		n = 0;
		for (k = 0; k < sqw->queue_idx[i]; k++) {
			bd = r->slots[(r->acquired + k) & r->mask];
			if ((u32)bd->act != XDP_PASS)
				continue;
			pass[n].netmem = bd->netmem;
			pass[n].page_idx = bd->page_idx;
			pass[n].off = bd->off;
			pass[n].len = bd->len;
			if (++n == KNOD_DEFAULT_PASS_SLOTS) {
				knod_d2h_copy(knodev, i, pass, n);
				n = 0;
			}
		}
		if (n)
			knod_d2h_copy(knodev, i, pass, n);
	}
}

static void knod_complete_acquire(struct knod_bpf_priv *priv,
				  struct knod_bpf_work_sq *sqw)
{
	struct knod_dev *knodev = priv->knodev;
	int i;

	if (static_branch_unlikely(&knod_stats_key))
		knod_hwid_count(priv, sqw);

	/* Either engine can carry the probe - the kernel emitter when it is
	 * told to, a blob when it was built with it - so collect on both and
	 * let the counts say whether anything wrote them.
	 */
	if (knod_bpf_cycle_probe)
		knod_cycle_count(priv, sqw);

	knod_bpf_d2h_pass(priv, sqw);

	for (i = 0; i < priv->nr_works; i++) {
		if (sqw->queue_idx[i] >= 1) {
			spsc_acquire(&knodev->wpriv[i].spsc_bds, NULL,
				     sqw->queue_idx[i], NULL);
		}
	}
}

/* Phase 2: schedule NAPI and free sqw.  Can run after the next dispatch
 * has been submitted - napi_schedule overlaps with GPU execution.
 */
static void knod_complete_napi(struct knod_bpf_priv *priv,
			       struct knod_bpf_work_sq *sqw)
{
	struct knod_dev *knodev = priv->knodev;
	struct knod_bpf_stats *stats = &priv->stats;
	ktime_t start;
	int i;

	if (static_branch_unlikely(&knod_stats_key))
		start = ktime_get();

	for (i = 0; i < priv->nr_works; i++) {
		if (sqw->queue_idx[i] >= 1)
			knod_napi_kick(&knodev->wpriv[i]);
	}

	sqw->backlogs = 0;
	sqw->expire = 0;
	list_add_tail_rcu(&sqw->list, &priv->free_list_sqw);

	if (static_branch_unlikely(&knod_stats_key)) {
		u64 ns = ktime_to_ns(ktime_sub(ktime_get(), start));

		stats->decode_act_total_ns += ns;
		stats->decode_act_count++;
		if (ns > stats->decode_act_max_ns)
			stats->decode_act_max_ns = ns;
	}
}

/* Resident code changes must wait for terminal completion, not just a chunk.
 * The worker acknowledges this generation only after stopping the resident
 * shader. A hung GPU retains its code and maps rather than racing an upload.
 * Start and stop use the same mutex, including the no-worker path.
 */
static void knod_bpf_code_op_begin(struct knod_bpf_priv *priv)
{
	u64 request;

	mutex_lock(&priv->map_op_lock);
	if (!priv->worker_task)
		return;
	request = priv->map_op_request + 1;
	WRITE_ONCE(priv->map_op_quiesce, true);
	smp_store_release(&priv->map_op_request, request);
	wait_event(priv->map_op_wq,
		   smp_load_acquire(&priv->map_op_ack) == request);
}

static void knod_bpf_code_op_end(struct knod_bpf_priv *priv)
{
	knod_bpf_gpu_mem_fence(priv);
	smp_store_release(&priv->map_op_quiesce, false);
	wake_up(&priv->map_op_wq);
	mutex_unlock(&priv->map_op_lock);
}

/*
 * Install into the selected code slot and publish its index after the upload.
 * Resident execution must terminate before either slot can be reused.
 */
static void knod_bpf_install_kernel(struct knod_bpf_priv *priv,
				    const struct knod_prog *knod_prog,
				    const void *code, u32 size)
{
	struct kernel_descriptor *kd;
	struct knod *knod = priv->knod;
	struct knod_mem *slot;
	u32 entry_off;
	u32 image_len;
	int idx;

	if (!code || !size || !knod->kernels[1])
		return;

	if (knod_bpf_persistent)
		knod_bpf_code_op_begin(priv);

	/*
	 * Before the worker runs, install in place; once it is dispatching,
	 * stage into the inactive slot and flip the active index so the live
	 * pipeline never reads a half-written slot.
	 */
	if (!priv->start || !knod->worker)
		idx = priv->active_idx;
	else
		idx = priv->active_idx ^ 1;

	slot = knod->kernels[idx];
	kd = slot->kaddr;
	entry_off = kd->kernel_code_entry_byte_offset;
	if (WARN_ON(entry_off >= slot->size))
		goto out;
	if (WARN_ON(size > slot->size - entry_off))
		size = slot->size - entry_off;
	image_len = entry_off + size;

	memcpy(slot->kaddr + entry_off, code, size);
	if (image_len < slot->size) {
		u32 clear_end = min_t(u32, slot->size,
					  entry_off + KNOD_BPF_PROG_BUF_SIZE);

		if (image_len < clear_end)
			memset(slot->kaddr + image_len, 0,
			       clear_end - image_len);
	}
	/*
	 * kernels[] is write-combining VRAM.  smp_wmb() is only a compiler
	 * barrier on x86 and does NOT drain the WC buffers, so the GPU could
	 * fetch half-written code and spin.  wmb() (sfence) flushes WC to VRAM
	 * before we publish the new slot; the dispatch doorbell is ordered
	 * behind it.
	 */
	wmb();
	knod_bpf_gpu_mem_fence(priv);
	priv->lds_bytes[idx] = knod_prog->lds_bytes;
	WRITE_ONCE(priv->kernel_image_len[idx], image_len);

	if (idx != priv->active_idx)
		WRITE_ONCE(priv->active_idx, idx);
out:
	if (knod_bpf_persistent)
		knod_bpf_code_op_end(priv);
}

/*
 * Keep the just-built pass-kernel IR for the debugfs "insn" dump, so it can
 * show the pass-through kernel when no XDP prog is attached.  The machine code
 * already lives in the kernel slot; this only retains the meta list.  Rebuilt
 * on every start (old metas freed first), released in knod_priv_exit().
 */
static void knod_bpf_retain_pass_ir(struct knod_bpf_priv *priv,
				    struct knod_prog *src)
{
	struct knod_insn_meta *meta, *tmp;
	struct knod_prog *dst = priv->pass_knod_prog;

	if (!dst) {
		dst = kzalloc_obj(*dst, GFP_KERNEL);
		if (!dst)
			return;
		INIT_LIST_HEAD(&dst->pre_insns);
		INIT_LIST_HEAD(&dst->insns);
		INIT_LIST_HEAD(&dst->post_insns);
		priv->pass_knod_prog = dst;
	} else {
		list_for_each_entry_safe(meta, tmp, &dst->pre_insns, l) {
			list_del(&meta->l);
			kfree(meta);
		}
		list_for_each_entry_safe(meta, tmp, &dst->insns, l) {
			list_del(&meta->l);
			kfree(meta);
		}
		list_for_each_entry_safe(meta, tmp, &dst->post_insns, l) {
			list_del(&meta->l);
			kfree(meta);
		}
	}
	list_splice_init(&src->pre_insns, &dst->pre_insns);
	list_splice_init(&src->insns, &dst->insns);
	list_splice_init(&src->post_insns, &dst->post_insns);
}

/* Instruction prefetch runs past s_endpgm by up to three cachelines. */
#define KNOD_SHADER_PAD_DWORDS		64

static struct knod_insn_meta *knod_bpf_pad_shader(struct knod_bpf_priv *priv,
						  struct knod_insn_meta *meta,
						  struct list_head *insns)
{
	int j;

	if (priv->isa_version < 10)
		return meta;

	for (j = 0; j < KNOD_SHADER_PAD_DWORDS; j++) {
		if (meta->amdgpu_insns >= KNOD_META_INSNS) {
			meta = kzalloc_obj(*meta, GFP_KERNEL);
			if (!meta)
				return NULL;
			list_add_tail(&meta->l, insns);
		}
		knod_emit(priv, meta, s_code_end);
	}

	return meta;
}

/* Bytes a meta puts in the program: everything it emitted, plus a spliced
 * routine.  Everything that walks the metas to work out where something sits
 * goes through here, because a routine the JIT did not emit still takes up
 * room and a branch that ignored it would land short.
 */
static u32 knod_meta_bytes(const struct knod_insn_meta *meta)
{
	u32 n = meta->blob_size;
	u32 i;

	for (i = 0; i < meta->amdgpu_insns; i++)
		n += meta->amdgpu_insn[i].size;

	return n;
}

/* Write one meta at @ptr and return where the next one starts. */
static u8 *knod_meta_write(const struct knod_insn_meta *meta, u8 *ptr,
			   bool trace)
{
	const u32 *dw;
	u32 size;
	u32 i;

	/* One past the last, so a routine spliced after everything emitted -
	 * or into a meta that emitted nothing at all - still gets written.
	 */
	for (i = 0; i <= meta->amdgpu_insns; i++) {
		if (meta->blob_size && i == meta->blob_at) {
			memcpy(ptr, meta->blob, meta->blob_size);
			if (trace)
				knod_jit_dbg(" 0x%.8X\t<%u bytes spliced>\n",
					     meta->amdgpu_insn_idx,
					     meta->blob_size);
			ptr += meta->blob_size;
		}
		if (i == meta->amdgpu_insns)
			break;

		size = meta->amdgpu_insn[i].size;
		dw = (const u32 *)&meta->amdgpu_insn[i];
		memcpy(ptr, dw, size);
		ptr += size;

		if (!trace)
			continue;
		if (size == 4)
			knod_jit_dbg(" 0x%.8X\t%.8X\n",
				     meta->amdgpu_insn_idx, dw[0]);
		else if (size == 8)
			knod_jit_dbg(" 0x%.8X\t%.8X %.8X\n",
				     meta->amdgpu_insn_idx, dw[0], dw[1]);
		else if (size == 12)
			knod_jit_dbg(" 0x%.8X\t%.8X %.8X %.8X\n",
				     meta->amdgpu_insn_idx, dw[0], dw[1], dw[2]);
		else
			WARN_ON_ONCE(1);
	}

	return ptr;
}

/* Nothing about the pass kernel follows a program, so unlike a translated one
 * it is prebuilt whole rather than in pieces with the program's code between.
 */
static int knod_bpf_pass_kernel_insns(struct knod_bpf_priv *priv,
				      struct knod_prog *pass_prog)
{
	struct knod_insn_meta *meta;
	const u32 *blob;
	u32 size;

	blob = knod_blob_find(&priv->blob, KNOD_BLOB_PASS_KERNEL, 0, &size);
	if (!blob) {
		WARN_ON_ONCE(1);
		return -EOPNOTSUPP;
	}

	meta = kzalloc_obj(*meta, GFP_KERNEL);
	if (!meta)
		return -ENOMEM;

	meta->blob = blob;
	meta->blob_size = size;
	list_add_tail(&meta->l, &pass_prog->pre_insns);
	return 0;
}

static int knod_bpf_jit_pass_kernel(struct knod_bpf_priv *priv)
{
	struct knod_insn_meta *meta, *tmp;
	struct knod *knod = priv->knod;
	struct knod_prog pass_prog;
	struct list_head *lists[2];
	u32 total = 0;
	u8 *buf, *ptr;
	int li, err;

	memset(&pass_prog, 0, sizeof(pass_prog));
	INIT_LIST_HEAD(&pass_prog.pre_insns);
	INIT_LIST_HEAD(&pass_prog.insns);
	INIT_LIST_HEAD(&pass_prog.post_insns);
	pass_prog.knod = knod;
	pass_prog.lds_bytes = 0;
	pass_prog.knodev = priv->knodev;
	pass_prog.done_mask_sreg = KNOD_AMDGPU_DONE_MASK_SREG;
	pass_prog.exec_save_base = KNOD_AMDGPU_EXEC_SAVE_SREG_BASE;
	pass_prog.initial_exec_sreg = KNOD_AMDGPU_INITIAL_EXEC_SREG;

	err = knod_bpf_pass_kernel_insns(priv, &pass_prog);
	if (err)
		goto free_all;

	/* Linearize prologue + epilogue into pass_prog_buf */
	lists[0] = &pass_prog.pre_insns;
	lists[1] = &pass_prog.post_insns;

	for (li = 0; li < 2; li++) {
		list_for_each_entry(meta, lists[li], l)
			total += knod_meta_bytes(meta);
	}

	kfree(priv->pass_prog_buf);
	priv->pass_prog_buf = NULL;
	priv->pass_prog_size = 0;
	buf = kzalloc(total, GFP_KERNEL);
	if (!buf) {
		err = -ENOMEM;
		goto free_all;
	}

	ptr = buf;
	for (li = 0; li < 2; li++) {
		list_for_each_entry(meta, lists[li], l)
			ptr = knod_meta_write(meta, ptr, false);
	}

	priv->pass_prog_buf = buf;
	priv->pass_prog_size = total;

	knod_bpf_install_kernel(priv, &pass_prog, priv->pass_prog_buf,
				priv->pass_prog_size);
	/* Remember which slot now holds pass so detach can flip back to it. */
	priv->pass_idx = priv->active_idx;

	pr_info("knod_bpf: pass kernel %u bytes\n", priv->pass_prog_size);
	err = 0;
	/* Retain the IR (moves the lists out) before the cleanup below
	 * frees.
	 */
	knod_bpf_retain_pass_ir(priv, &pass_prog);

free_all:
	list_for_each_entry_safe(meta, tmp, &pass_prog.post_insns, l) {
		list_del_init(&meta->l);
		kfree(meta);
	}
	list_for_each_entry_safe(meta, tmp, &pass_prog.pre_insns, l) {
		list_del_init(&meta->l);
		kfree(meta);
	}
	return err;
}

static void knod_bpf_wait_sqw(struct knod_bpf_priv *priv,
			      struct knod_bpf_work_sq *sqw)
{
	unsigned long deadline = jiffies + msecs_to_jiffies(1000);
	bool warned = false;

	/* A timeout does not revoke GPU ownership of the backing storage. */
	while (!knod_bpf_sqw_done(priv, sqw)) {
		if (!warned && time_after(jiffies, deadline)) {
			pr_warn("knod: retaining incomplete GPU dispatch during stop\n");
			warned = true;
		}
		usleep_range(100, 200);
	}
	dma_rmb();
}

static void knod_bpf_drain_worker(struct knod_bpf_priv *priv)
{
	struct knod_bpf_work_sq *sqw;

	/* Completion can be out of order; SPSC retirement must stay FIFO. */
	while (priv->inflight_cnt) {
		sqw = priv->inflight[0];
		knod_bpf_wait_sqw(priv, sqw);
		if (--priv->inflight_cnt)
			memmove(priv->inflight, priv->inflight + 1,
				priv->inflight_cnt * sizeof(priv->inflight[0]));
		priv->inflight[priv->inflight_cnt] = NULL;
		knod_complete_acquire(priv, sqw);
		knod_complete_napi(priv, sqw);
	}
	knod_bpf_persistent_stop(priv);
	WRITE_ONCE(priv->dispatch_fault, false);
}

static void knod_bpf_drain(struct knod_bpf_priv *priv)
{
	knod_bpf_drain_worker(priv);
}

static void knod_bpf_stop_worker(struct knod_bpf_priv *priv)
{
	/* Exclude host mutations until the stopped worker's GPU work retires. */
	mutex_lock(&priv->map_op_lock);
	priv->start = 0;
	if (priv->worker_task) {
		kthread_stop(priv->worker_task);
		put_task_struct(priv->worker_task);
		priv->worker_task = NULL;
	}
	synchronize_net();
	knod_bpf_drain(priv);
	mutex_unlock(&priv->map_op_lock);
}

static void knod_bpf_configure_worker(struct knod_bpf_priv *priv)
{
	knod_bpf_stop_worker(priv);

	priv->inflight_cnt = 0;
}

static int knod_bpf_start_worker(struct knod_bpf_priv *priv)
{
	struct task_struct *p;

	/* A no-worker map operation must finish before submissions restart. */
	mutex_lock(&priv->map_op_lock);
	p = kthread_run(knod_bpf_worker, priv, "knod_%d_0",
			priv->knodev->accel->id);
	if (IS_ERR(p)) {
		mutex_unlock(&priv->map_op_lock);
		return PTR_ERR(p);
	}

	get_task_struct(p);
	priv->worker_task = p;
	mutex_unlock(&priv->map_op_lock);
	return 0;
}

/* One workgroup per queue, so a queue is one CU's worth of work and its
 * dispatch batch is one workgroup of packets - capped by the static descriptor
 * array and rounded down to a power of two, since the shader derives the flat
 * slot as queue_id << ilog2(batch_size) + local_idx.
 *
 * Fanning a queue out over several workgroups was tried and gave the CUs back
 * nothing; what the dispatch waited on was never the compute.  It also cannot
 * be done for a program with percpu maps, whose instances are counted one per
 * queue and would otherwise have several workgroups writing one of them.
 * Rather than a rule that holds for some programs, every program is shaped the
 * same way.
 */
static unsigned int knod_bpf_batch_size(struct knod_bpf_priv *priv)
{
	unsigned int max_flat = KNOD_BPF_BACKLOGS_MAX / priv->nr_works;
	unsigned int batch = min_t(unsigned int, knod_bpf_workgroups, max_flat);

	if (!batch)
		batch = knod_bpf_workgroups;
	return rounddown_pow_of_two(batch);
}

static void knod_bpf_start(struct knod_dev *knodev)
{
	struct knod_bpf_priv *priv =
		(struct knod_bpf_priv *)knodev->accel->xdp.priv;
	struct bpf_prog *prog;
	unsigned int active_rxq;
	int err;

	priv->start = 1;
	active_rxq = knod_bpf_active_rxq_count(knodev->netdev);
	if (active_rxq && active_rxq != priv->nr_works)
		pr_warn("knod_bpf: active rx queues changed from %d to %u; using initialized count\n",
			priv->nr_works, active_rxq);

	priv->batch_size = knod_bpf_batch_size(priv);

	knod_jit_dbg(" batch_size = %d\n", priv->batch_size);
	knod_bpf_configure_worker(priv);
	pr_info("knod_bpf: using single AQL queue, rx_works=%d active_rxq=%u batch_size=%d\n",
		priv->nr_works, active_rxq, priv->batch_size);

	if (knod_bpf_jit_pass_kernel(priv))
		pr_warn("knod_bpf: pass kernel JIT failed\n");

	prog = READ_ONCE(priv->prog);
	if (prog)
		knod_setup_bpf_prog(prog);

	priv->start = 1;
	err = knod_bpf_start_worker(priv);
	if (err) {
		pr_err("knod_bpf: start_worker failed: %d\n", err);
		priv->start = 0;
		return;
	}
}

static void knod_bpf_stop(struct knod_dev *knodev)
{
	struct knod_bpf_priv *priv =
		(struct knod_bpf_priv *)knodev->accel->xdp.priv;

	knod_bpf_stop_worker(priv);

	kfree(priv->pass_prog_buf);
	priv->pass_prog_buf = NULL;
	priv->pass_prog_size = 0;
}

/* Restore PASS on detach; resident execution may have reused both slots. */
static void knod_bpf_reload_pass(struct knod_dev *knodev)
{
	struct knod_bpf_priv *priv = knodev->accel->xdp.priv;

	if (!priv)
		return;
	if (knod_bpf_persistent) {
		/* Two code slots can have been reused since PASS was installed. */
		if (priv->pass_knod_prog && priv->pass_prog_buf) {
			knod_bpf_install_kernel(priv, priv->pass_knod_prog,
						priv->pass_prog_buf,
						priv->pass_prog_size);
			priv->pass_idx = READ_ONCE(priv->active_idx);
		}
	} else {
		WRITE_ONCE(priv->active_idx, priv->pass_idx);
	}
}

static void knod_setup_bpf_prog(struct bpf_prog *prog)
{
	struct knod_prog *knod_prog = prog->aux->offload->dev_priv;
	struct knod_dev *knodev = knod_prog->knodev;
	struct knod_insn_meta *meta, *tmp;
	struct knod_bpf_priv *priv;
	u32 total_bytes;
	u8 *kernel_ptr;

	priv = (struct knod_bpf_priv *)knodev->accel->xdp.priv;
	WRITE_ONCE(priv->installing_kernel, true);

	if (prog) {
		WRITE_ONCE(priv->prog, NULL);
		kernel_ptr = priv->prog_buf;
		memset(priv->prog_buf, 0, KNOD_BPF_PROG_BUF_SIZE);

		list_for_each_entry(meta, &priv->knod_prog->pre_insns, l)
			kernel_ptr = knod_meta_write(meta, kernel_ptr, true);

		list_for_each_entry(meta, &priv->knod_prog->insns, l)
			kernel_ptr = knod_meta_write(meta, kernel_ptr, true);

		list_for_each_entry(meta, &priv->knod_prog->post_insns, l)
			kernel_ptr = knod_meta_write(meta, kernel_ptr, true);
		total_bytes = kernel_ptr - (u8 *)priv->prog_buf;

		pr_debug("KNOD JIT: total binary size = %u bytes (limit %u)\n",
			 total_bytes, KNOD_BPF_PROG_BUF_SIZE);
		if (WARN_ON(total_bytes > KNOD_BPF_PROG_BUF_SIZE))
			total_bytes = KNOD_BPF_PROG_BUF_SIZE;
		knod_bpf_install_kernel(priv, knod_prog, priv->prog_buf,
					total_bytes);
		WRITE_ONCE(priv->prog, prog);
	} else {
		WRITE_ONCE(priv->prog, NULL);
		list_for_each_entry_safe(meta, tmp, &priv->knod_prog->pre_insns,
					 l) {
			list_del_init(&meta->l);
			kfree(meta);
		}

		list_for_each_entry_safe(meta, tmp, &priv->knod_prog->insns,
					 l) {
			list_del_init(&meta->l);
			kfree(meta);
		}

		list_for_each_entry_safe(meta, tmp,
					 &priv->knod_prog->post_insns, l) {
			list_del_init(&meta->l);
			kfree(meta);
		}

		/* bbs points into the metas just freed */
		kfree(priv->knod_prog->bbs);
		priv->knod_prog->bbs = NULL;
		priv->knod_prog->n_bbs = 0;

		if (priv->pass_prog_buf)
			knod_bpf_install_kernel(priv, priv->pass_knod_prog,
						priv->pass_prog_buf,
						priv->pass_prog_size);
	}
	WRITE_ONCE(priv->installing_kernel, false);
}

static int knod_bpf_map_hash_init_elem(struct knod_bpf_map *knod_map,
				       struct knod_bpf_map_obj *knod_map_obj)
{
	unsigned int *queue = (unsigned int *)knod_map->queue_mem->kaddr;
	unsigned int *bucket = (unsigned int *)&knod_map_obj->bucket[0];
	void *elems = knod_map->hash_elems_mem->kaddr;
	struct knod_bpf_hash_elem_obj *e;
	int i, elem_size;

	elem_size = knod_bpf_hash_elem_size(knod_map_obj->key_size,
					    knod_map_obj->value_size,
					    knod_map_obj->meta.hmeta.n_instances);
	knod_map_obj->meta.hmeta.elem_size = elem_size;

	for (i = 0; i < knod_map_obj->meta.hmeta.n_buckets; i++)
		bucket[i] = KNOD_BPF_HASH_NEXT_END;

	for (i = 0; i < knod_map_obj->max_entries; i++) {
		e = elems + (i * elem_size);
		e->next = KNOD_BPF_HASH_NEXT_END;
		queue[i] = i;
	}
	knod_map_obj->meta.hmeta.cur = knod_map_obj->max_entries;

	return 0;
}

static inline unsigned char *
knod_bpf_hash_elem_kv(struct knod_bpf_hash_elem_obj *e)
{
	return (unsigned char *)e + offsetof(struct knod_bpf_hash_elem_obj, kv);
}

static inline void *
knod_bpf_array_value_ptr(struct knod_bpf_map_obj *knod_map_obj,
			 unsigned int idx)
{
	return (unsigned char *)knod_map_obj +
	       offsetof(struct knod_bpf_map_obj, bucket) +
	       (size_t)idx * knod_map_obj->value_size;
}

/* Restate the map for a prebuilt routine, which knows this layout and none of
 * the kernel's own.  Everything a routine can reach is an offset from here, so
 * a blob carries no relocations.
 */
static void knod_bpf_map_fill_desc(struct knod_bpf_map *knod_map)
{
	const struct knod_bpf_map_obj *obj = knod_map->knod_map_obj;
	struct knod_blob_map_desc *desc = knod_map->desc;
	u64 obj_gaddr = knod_map->mem->gaddr;

	memset(desc, 0, sizeof(*desc));
	desc->key_size = obj->key_size;
	desc->value_size = obj->value_size;
	desc->max_entries = obj->max_entries;
	desc->bucket_gaddr = obj_gaddr +
			     offsetof(struct knod_bpf_map_obj, bucket);
	/* Where the values are, whatever kind of map this is.  An array keeps
	 * them in the map object itself and a hash in a BO of its own, and a
	 * routine is told the base rather than the difference.
	 */
	desc->elems_gaddr = desc->bucket_gaddr;

	if (obj->map_type == BPF_MAP_TYPE_HASH ||
	    obj->map_type == BPF_MAP_TYPE_PERCPU_HASH) {
		desc->elem_size = obj->meta.hmeta.elem_size;
		desc->elems_gaddr = (u64)obj->meta.hmeta.elems;
		desc->queue_gaddr = (u64)obj->meta.hmeta.q;
		desc->gc_list_gaddr = (u64)obj->meta.hmeta.gc_list;
		desc->gc_count_gaddr = obj_gaddr +
			offsetof(struct knod_bpf_map_obj, meta.hmeta.gc_count);
		desc->free_cur_gaddr = obj_gaddr +
			offsetof(struct knod_bpf_map_obj, meta.hmeta.cur);
		desc->n_buckets = obj->meta.hmeta.n_buckets;
		/* The locks follow the bucket heads in the same array. */
		desc->lock_offset = obj->meta.hmeta.n_buckets *
				    sizeof(unsigned int);
		desc->hashrnd = obj->meta.hmeta.hashrnd;
		/* Non-zero only for PERCPU_HASH: the per-instance value slot
		 * stride the blob adds workgroup_id_y * this to reach.
		 */
		desc->per_instance_size = obj->meta.hmeta.per_instance_size;
	} else {
		desc->per_instance_size = obj->meta.ameta.per_instance_size;
	}
}

static int __knod_bpf_map_alloc(struct knod_dev *knodev,
				struct bpf_offloaded_map *offmap)
{
	struct knod_bpf_priv *priv =
		(struct knod_bpf_priv *)knodev->accel->xdp.priv;
	struct knod_mem *mem, *queue_mem, *hash_elems_mem, *gc_mem;
	int order, size, queue_size, i, value_size, nents, err;
	int n_instances = 1;
	int flags = KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
		    KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC |
		    KFD_IOC_ALLOC_MEM_FLAGS_VRAM;
	struct knod_bpf_map_obj *knod_map_obj;
	struct knod *knod = priv->knod;
	struct knod_bpf_map *knod_map;
	unsigned int gc_size, desc_off;
	unsigned int *q;

	bool is_hash = offmap->map.map_type == BPF_MAP_TYPE_HASH ||
		       offmap->map.map_type == BPF_MAP_TYPE_PERCPU_HASH;
	bool is_percpu = offmap->map.map_type == BPF_MAP_TYPE_PERCPU_ARRAY ||
			 offmap->map.map_type == BPF_MAP_TYPE_PERCPU_HASH;

	if (is_hash) {
		value_size = sizeof(unsigned int);
		nents = roundup_pow_of_two(offmap->map.max_entries);
	} else {
		value_size = offmap->map.value_size;
		nents = offmap->map.max_entries;
	}

	/* A percpu map keeps one value instance per GPU workgroup so each CU
	 * updates its own copy - no cross-CU atomic contention.  Instances map
	 * 1:1 to the per-cpu buffer (workgroup_id_y indexes into it), one per
	 * queue, handed back through the per-cpu slot of the same number, so
	 * there has to be a slot for every queue.  An ordinary NIC gives out no
	 * more queues than there are cpus; some do.  For PERCPU_ARRAY the value
	 * instances live in the map-obj tail; for PERCPU_HASH they are extra
	 * value slots inside each hash element (see knod_bpf_hash_elem_size).
	 */
	if (is_percpu) {
		if (priv->nr_works > num_possible_cpus()) {
			pr_warn("knod_bpf: %d rx queues but %u cpus; a percpu map keeps one instance per queue and has nowhere to report the rest\n",
				priv->nr_works, num_possible_cpus());
			return -EOPNOTSUPP;
		}
		n_instances = num_possible_cpus();
	}

	/* Hash types put their per-instance values in the elems BO, not the
	 * map-obj tail, so n_instances multiplies elem_size (below), not this
	 * bucket-head region.
	 */
	size = sizeof(struct knod_bpf_map_obj) +
	       value_size * nents * (is_hash ? 1 : n_instances);
	if (is_hash)
		size += sizeof(unsigned int) * nents;
	desc_off = round_up(size, __alignof__(struct knod_blob_map_desc));
	size = desc_off + sizeof(struct knod_blob_map_desc);
	order = get_order(size);

	mem = knod_alloc_mem(knod, PAGE_SIZE << order, flags);
	if (IS_ERR(mem))
		return -ENOMEM;

	memset(mem->kaddr, 0, size);
	knod_map = kzalloc_obj(struct knod_bpf_map, GFP_KERNEL);
	if (!knod_map) {
		knod_free_mem(knod, mem);
		return -ENOMEM;
	}

	knod_map->mem = mem;
	knod_map->queue_mem = NULL;
	knod_map->hash_elems_mem = NULL;
	knod_map->offmap = offmap;
	knod_map->priv = priv;
	if (offmap->dev_priv)
		WARN_ON_ONCE(1);
	offmap->dev_priv = knod_map;

	knod_map->desc = mem->kaddr + desc_off;
	knod_map->desc_gaddr = mem->gaddr + desc_off;

	knod_map_obj = (struct knod_bpf_map_obj *)mem->kaddr;
	knod_map_obj->key_size = offmap->map.key_size;
	if (knod_map_obj->key_size > MAX_MAP_KEY_SIZE) {
		pr_warn("request key size is %d, but max key size is %d\n",
			knod_map_obj->key_size, MAX_MAP_KEY_SIZE);
		return -ENOMEM;
	}
	knod_map_obj->value_size = offmap->map.value_size;
	knod_map_obj->max_entries = nents;
	knod_map_obj->id = offmap->map.id;
	knod_map_obj->map_type = offmap->map.map_type;
	if (is_hash) {
		knod_map_obj->meta.hmeta.n_buckets = nents;
		if (offmap->map.map_flags & BPF_F_ZERO_SEED)
			knod_map_obj->meta.hmeta.hashrnd = 0;
		else
			knod_map_obj->meta.hmeta.hashrnd = get_random_u32();
		knod_map_obj->meta.hmeta.n_instances = n_instances;
		knod_map_obj->meta.hmeta.per_instance_size = is_percpu ?
			knod_bpf_hash_value_stride(knod_map_obj->value_size) : 0;
	} else {
		knod_map_obj->meta.ameta.per_instance_size = value_size * nents;
		knod_map_obj->meta.ameta.n_instances = n_instances;
	}
	knod_map->knod_map_obj = knod_map_obj;
	/* map->flags = ? */
	knod_jit_dbg(" map_id = %d\n", knod_map_obj->id);

	if (is_hash) {
		queue_size = sizeof(unsigned int) * nents;
		queue_size = PAGE_SIZE << get_order(queue_size);
		queue_mem = knod_alloc_mem(knod, queue_size, flags);
		if (IS_ERR(queue_mem)) {
			knod_free_mem(knod, mem);
			kfree(knod_map);
			return -ENOMEM;
		}

		memset(queue_mem->kaddr, 0, queue_mem->size);
		q = queue_mem->kaddr;
		for (i = 0; i < knod_map_obj->meta.hmeta.n_buckets; i++)
			q[i] = i;
		knod_map->queue_mem = queue_mem;
		knod_map_obj->meta.hmeta.q = (struct _queue *)queue_mem->gaddr;

		queue_size = knod_bpf_hash_elem_size(knod_map_obj->key_size,
						     knod_map_obj->value_size,
						     n_instances) *
			     knod_map_obj->max_entries;
		queue_size = PAGE_SIZE << get_order(queue_size);

		hash_elems_mem = knod_alloc_mem(knod, queue_size, flags);
		if (IS_ERR(hash_elems_mem)) {
			knod_free_mem(knod, queue_mem);
			knod_free_mem(knod, mem);
			kfree(knod_map);
			return -ENOMEM;
		}

		memset(hash_elems_mem->kaddr, 0, queue_size);
		knod_map->hash_elems_mem = hash_elems_mem;
		knod_map_obj->meta.hmeta.elems = (void *)hash_elems_mem->gaddr;
		knod_bpf_map_hash_init_elem(knod_map, knod_map_obj);

		/* GC list for GPU-side delete: elem_ids pending unlink */
		gc_size = sizeof(unsigned int) * nents;
		gc_size = PAGE_SIZE << get_order(gc_size);
		gc_mem = knod_alloc_mem(knod, gc_size, flags);
		if (IS_ERR(gc_mem)) {
			knod_free_mem(knod, hash_elems_mem);
			knod_free_mem(knod, queue_mem);
			knod_free_mem(knod, mem);
			kfree(knod_map);
			return -ENOMEM;
		}
		memset(gc_mem->kaddr, 0, gc_size);
		knod_map->gc_mem = gc_mem;
		knod_map_obj->meta.hmeta.gc_count = 0;
		knod_map_obj->meta.hmeta.gc_list = (void *)gc_mem->gaddr;
	}

	knod_bpf_map_fill_desc(knod_map);

	err = __knod_map_mem(knod, mem);
	if (err) {
		pr_err("knod_bpf: failed to GPU-map map BO\n");
		goto err_map;
	}
	if (is_hash) {
		err = __knod_map_mem(knod, queue_mem);
		if (err) {
			pr_err("knod_bpf: failed to GPU-map queue BO\n");
			goto err_map;
		}
		err = __knod_map_mem(knod, hash_elems_mem);
		if (err) {
			pr_err("knod_bpf: failed to GPU-map hash_elems BO\n");
			goto err_map;
		}
		err = __knod_map_mem(knod, knod_map->gc_mem);
		if (err) {
			pr_err("knod_bpf: failed to GPU-map gc BO\n");
			goto err_map;
		}
	}
	knod_bpf_gpu_mem_fence(priv);

	mutex_lock(&knodev->lock);
	list_add(&knod_map->list, &knodev->accel->xdp.bound_maps);
	mutex_unlock(&knodev->lock);
	return 0;

err_map:
	if (is_hash) {
		knod_free_mem(knod, knod_map->gc_mem);
		knod_free_mem(knod, hash_elems_mem);
		knod_free_mem(knod, queue_mem);
	}
	knod_free_mem(knod, mem);
	kfree(knod_map);
	return err;
}

static void knod_bpf_map_setup(struct bpf_prog *prog)
{
	struct knod_prog *knod_prog = prog->aux->offload->dev_priv;
	struct knod_dev *knodev = knod_prog->knodev;
	struct knod_bpf_map *knod_map;
	struct knod_bpf_map_obj *map;
	struct knod_mem *mem;

	mutex_lock(&knodev->lock);
	list_for_each_entry(knod_map, &knodev->accel->xdp.bound_maps, list) {
		mem = knod_map->mem;
		map = mem->kaddr;
		map->id = knod_map->offmap->map.id;
		map->map_type = knod_map->offmap->map.map_type;
		knod_jit_dbg(" id = %d type = %d\n", knod_map->offmap->map.id,
			knod_map->offmap->map.map_type);
	}
	mutex_unlock(&knodev->lock);
}

static struct knod_bpf_hash_elem_obj *
knod_bpf_map_hash_pop(struct knod_bpf_map *knod_map,
		      struct knod_bpf_map_obj *knod_map_obj)
{
	void *elems = knod_map->hash_elems_mem->kaddr;
	unsigned int *queue = (unsigned int *)knod_map->queue_mem->kaddr;
	struct knod_bpf_hash_elem_obj *e;
	int elem_id;

	/* The count is what is free and the queue is a stack of that many, so
	 * the element to take is the one below the top.  Signed, because the
	 * shader decrements before it knows whether there was anything left.
	 */
	if ((int)knod_map_obj->meta.hmeta.cur <= 0)
		return NULL;

	knod_map_obj->meta.hmeta.cur--;
	elem_id = queue[knod_map_obj->meta.hmeta.cur];
	knod_jit_dbg(" elem_id = 0x%x\n", elem_id);
	e = elems + (elem_id * knod_map_obj->meta.hmeta.elem_size);
	e->next = KNOD_BPF_HASH_NEXT_END;

	return e;
}

/* Copy the value(s) between a hash element and a userspace buffer.  A plain
 * HASH has one value slot after the key; a PERCPU_HASH has n_instances slots,
 * each value_stride bytes, and userspace lays its per-cpu values out with the
 * same stride.
 */
static void knod_bpf_hash_read_value(const struct knod_bpf_map_obj *o,
				     const struct knod_bpf_hash_elem_obj *e,
				     void *value)
{
	unsigned int stride = knod_bpf_hash_value_stride(o->value_size);
	unsigned int voff = knod_bpf_hash_value_off(o->key_size);
	unsigned int n = o->meta.hmeta.n_instances ? : 1;
	unsigned int i;

	for (i = 0; i < n; i++)
		memcpy((char *)value + i * stride,
		       (const char *)e + voff + i * stride, o->value_size);
}

static void knod_bpf_hash_write_value(const struct knod_bpf_map_obj *o,
				      struct knod_bpf_hash_elem_obj *e,
				      const void *value)
{
	unsigned int stride = knod_bpf_hash_value_stride(o->value_size);
	unsigned int voff = knod_bpf_hash_value_off(o->key_size);
	unsigned int n = o->meta.hmeta.n_instances ? : 1;
	unsigned int i;

	for (i = 0; i < n; i++)
		unsafe_memcpy((char *)e + voff + i * stride,
			      (const char *)value + i * stride, o->value_size,
			      "knod hash elems are variable-sized GPU map records");
}

/* A shader insert into a PERCPU_HASH writes only this instance's value slot, so
 * an element handed back to the free list has to leave with every slot zeroed;
 * otherwise the next key it holds inherits the previous key's per-cpu values in
 * the slots this queue never touches, and the summed readback is wrong.  No-op
 * for a plain hash, whose insert overwrites the one value in full.
 */
static void knod_bpf_hash_free_value(const struct knod_bpf_map_obj *o,
				     struct knod_bpf_hash_elem_obj *e)
{
	unsigned int stride = knod_bpf_hash_value_stride(o->value_size);
	unsigned int voff = knod_bpf_hash_value_off(o->key_size);
	unsigned int n = o->meta.hmeta.n_instances ? : 1;

	if (o->map_type != BPF_MAP_TYPE_PERCPU_HASH)
		return;

	memset((char *)e + voff, 0, stride * n);
	wmb();
}

static struct knod_bpf_hash_elem_obj *
knod_bpf_map_hash_alloc_elem(struct knod_bpf_map *knod_map,
			     struct knod_bpf_map_obj *knod_map_obj,
			     void *key, void *value)
{
	struct knod_bpf_hash_elem_obj *e;

	e = knod_bpf_map_hash_pop(knod_map, knod_map_obj);
	if (!e)
		return NULL;

	unsafe_memcpy(knod_bpf_hash_elem_kv(e), key, knod_map_obj->key_size,
		      "knod hash elems are variable-sized GPU map records");
	knod_bpf_hash_write_value(knod_map_obj, e, value);
	/* VRAM is ioremap_wc - drain new elem's next and kv stores before
	 * the caller publishes a pointer to this elem.
	 */
	wmb();
	return e;
}

static int knod_bpf_map_hash_lookup_elem(struct knod_bpf_map *knod_map,
					 struct knod_bpf_map_obj *knod_map_obj,
					 void *key,
					 void *value)
{
	void *elems = knod_map->hash_elems_mem->kaddr;
	unsigned int hash, elem_id, elem_size;
	struct knod_bpf_hash_elem_obj *e;
	unsigned int *bucket;

	hash = jhash((const void *)key, knod_map_obj->key_size,
		     knod_map_obj->meta.hmeta.hashrnd);
	knod_jit_dbg(" hash = %x\n", hash);
	hash = hash & (knod_map_obj->meta.hmeta.n_buckets - 1);
	knod_jit_dbg(" hash = %x\n", hash);
	bucket = (unsigned int *)&knod_map_obj->bucket[0];

	elem_id = bucket[hash];
	if (elem_id == KNOD_BPF_HASH_NEXT_END)
		return -ENOENT;

	elem_size = knod_map_obj->meta.hmeta.elem_size;

	e = elems + (elem_id * elem_size);
	while (1) {
		if (!(e->next & KNOD_BPF_HASH_NEXT_DELETED) &&
		    !memcmp(&e->kv[0], (const unsigned char *)key,
			    knod_map_obj->key_size)) {
			knod_bpf_hash_read_value(knod_map_obj, e, value);
			return 0;
		}
		unsigned int real_next = e->next & KNOD_BPF_HASH_NEXT_MASK;

		if (real_next == KNOD_BPF_HASH_NEXT_END)
			return -ENOENT;
		e = elems + (real_next * elem_size);
	}

	return -ENOENT;
}

static int knod_bpf_map_hash_update_elem(struct knod_bpf_map *knod_map,
					 struct knod_bpf_map_obj *knod_map_obj,
					 void *key,
					 void *value)
{
	void *elems = knod_map->hash_elems_mem->kaddr;
	unsigned int hash, elem_id, elem_size;
	struct knod_bpf_hash_elem_obj *e, *ne;
	unsigned int *bucket;

	hash = jhash((const void *)key, knod_map_obj->key_size,
		     knod_map_obj->meta.hmeta.hashrnd);
	hash = hash & (knod_map_obj->meta.hmeta.n_buckets - 1);
	bucket = (unsigned int *)&knod_map_obj->bucket[0];

	elem_size = knod_map_obj->meta.hmeta.elem_size;
	elem_id = bucket[hash];
	if (elem_id == KNOD_BPF_HASH_NEXT_END) {
		ne = knod_bpf_map_hash_alloc_elem(knod_map, knod_map_obj, key,
						  value);
		if (!ne)
			return -ENOMEM;
		bucket[hash] = ((void *)ne - (void *)elems) / elem_size;
		return 0;
	}

	e = elems + (elem_id * elem_size);
	while (1) {
		if (!(e->next & KNOD_BPF_HASH_NEXT_DELETED) &&
		    !memcmp(&e->kv[0], (const unsigned char *)key,
			    knod_map_obj->key_size)) {
			knod_bpf_hash_write_value(knod_map_obj, e, value);
			return 0;
		}
		unsigned int real_next = e->next & KNOD_BPF_HASH_NEXT_MASK;

		if (real_next == KNOD_BPF_HASH_NEXT_END) {
			ne = knod_bpf_map_hash_alloc_elem(knod_map,
							  knod_map_obj,
							  key, value);
			if (!ne)
				return -ENOMEM;
			e->next = (e->next & KNOD_BPF_HASH_NEXT_DELETED) |
				  (((void *)ne - (void *)elems) / elem_size);
			return 0;
		}
		e = elems + (real_next * elem_size);
	}

	return -ENOENT;
}

static int knod_bpf_map_hash_delete_elem(struct knod_bpf_map *knod_map,
					 struct knod_bpf_map_obj *knod_map_obj,
					 void *key)
{
	void *elems = knod_map->hash_elems_mem->kaddr;
	unsigned int *queue = knod_map->queue_mem->kaddr;
	unsigned int hash, elem_id, elem_size, cur;
	struct knod_bpf_hash_elem_obj *e, *pe;
	unsigned int *bucket;

	hash = jhash((const void *)key, knod_map_obj->key_size,
		     knod_map_obj->meta.hmeta.hashrnd);
	hash = hash & (knod_map_obj->meta.hmeta.n_buckets - 1);
	bucket = (unsigned int *)&knod_map_obj->bucket[0];

	elem_id = bucket[hash];
	if (elem_id == KNOD_BPF_HASH_NEXT_END)
		return -ENOENT;

	elem_size = knod_map_obj->meta.hmeta.elem_size;

	e = elems + (elem_id * elem_size);
	pe = e;
	while (1) {
		if (!(e->next & KNOD_BPF_HASH_NEXT_DELETED) &&
		    !memcmp(&e->kv[0], (const unsigned char *)key,
			    knod_map_obj->key_size)) {
			unsigned int e_next = e->next & KNOD_BPF_HASH_NEXT_MASK;
			unsigned int del_id = ((void *)e - elems) / elem_size;

			/* Unlink (GPU is paused - safe) */
			if (pe != e)
				pe->next = (pe->next &
					    KNOD_BPF_HASH_NEXT_DELETED) |
					   e_next;
			else
				bucket[hash] = e_next;

			e->next = KNOD_BPF_HASH_NEXT_END;
			knod_bpf_hash_free_value(knod_map_obj, e);

			/* Return elem to queue */
			cur = knod_map_obj->meta.hmeta.cur;
			queue[cur] = del_id;
			knod_map_obj->meta.hmeta.cur = cur + 1;
			return 0;
		}
		unsigned int real_next = e->next & KNOD_BPF_HASH_NEXT_MASK;

		if (real_next == KNOD_BPF_HASH_NEXT_END)
			return -ENOENT;
		pe = e;
		e = elems + (real_next * elem_size);
	}

	return -ENOENT;
}

static int knod_bpf_map_hash_get_first_key(struct bpf_offloaded_map *offmap,
					   void *nkey)
{
	struct knod_bpf_map *knod_map = (struct knod_bpf_map *)offmap->dev_priv;
	struct knod_bpf_map_obj *knod_map_obj;
	unsigned int *bucket, elem_size, i;
	struct knod_bpf_hash_elem_obj *e;
	void *elems;

	knod_map_obj = knod_map->knod_map_obj;
	bucket =  (unsigned int *)&knod_map_obj->bucket[0];
	elems = knod_map->hash_elems_mem->kaddr;

	elem_size = knod_map_obj->meta.hmeta.elem_size;

	for (i = 0; i < knod_map_obj->meta.hmeta.n_buckets; i++) {
		unsigned int eid;

		if (bucket[i] == KNOD_BPF_HASH_NEXT_END)
			continue;
		eid = bucket[i];
		while (eid != KNOD_BPF_HASH_NEXT_END) {
			e = elems + (eid * elem_size);
			if (!(e->next & KNOD_BPF_HASH_NEXT_DELETED)) {
				unsafe_memcpy(nkey, knod_bpf_hash_elem_kv(e),
					      knod_map_obj->key_size,
					      "knod hash elems are variable-sized GPU map records");
				return 0;
			}
			eid = e->next & KNOD_BPF_HASH_NEXT_MASK;
		}
	}

	return -ENOENT;
}

static int knod_bpf_map_hash_get_next_key(struct bpf_offloaded_map *offmap,
					  void *key, void *nkey)
{
	struct knod_bpf_map *knod_map = (struct knod_bpf_map *)offmap->dev_priv;
	struct knod_bpf_map_obj *knod_map_obj;
	unsigned int *bucket, elem_size, i;
	struct knod_bpf_hash_elem_obj *e;
	bool found = false;
	unsigned int hash;
	void *elems;

	knod_map_obj = knod_map->knod_map_obj;

	bucket =  (unsigned int *)&knod_map_obj->bucket[0];
	elems = knod_map->hash_elems_mem->kaddr;

	hash = jhash((const void *)key, knod_map_obj->key_size,
		     knod_map_obj->meta.hmeta.hashrnd);
	hash = hash & (knod_map_obj->meta.hmeta.n_buckets - 1);
	elem_size = knod_map_obj->meta.hmeta.elem_size;

	for (i = hash; i < knod_map_obj->meta.hmeta.n_buckets; i++) {
		unsigned int eid;

		if (bucket[i] == KNOD_BPF_HASH_NEXT_END)
			continue;

		eid = bucket[i];
		while (eid != KNOD_BPF_HASH_NEXT_END) {
			e = elems + (eid * elem_size);
			if (!(e->next & KNOD_BPF_HASH_NEXT_DELETED)) {
				if (found &&
				    memcmp(&e->kv[0],
					   (const unsigned char *)key,
					   knod_map_obj->key_size)) {
					unsafe_memcpy(nkey,
						      knod_bpf_hash_elem_kv(e),
						      knod_map_obj->key_size,
						      "knod hash elems are variable-sized GPU map records");
					return 0;
				}
				if (!memcmp(&e->kv[0],
					    (const unsigned char *)key,
					    knod_map_obj->key_size))
					found = true;
			}
			eid = e->next & KNOD_BPF_HASH_NEXT_MASK;
		}
	}

	return -ENOENT;
}

static void knod_bpf_map_free(struct knod_dev *knodev,
			      struct bpf_offloaded_map *offmap)
{
	struct knod_bpf_map *knod_map = offmap->dev_priv;
	struct knod_bpf_priv *priv = knodev->accel->xdp.priv;

	if (!knod_map)
		return;
	/*
	 * Defer the BO free: an in-flight prog dispatch may still reference
	 * this map's VRAM.  Move it from bound_maps onto dead_maps under
	 * knodev->lock (the lock that guards the add); the worker reaps it
	 * from there after its next completion, by which point the in-flight
	 * dispatch on the old slot has retired (clean atomic flip).
	 */
	mutex_lock(&knodev->lock);
	list_del(&knod_map->list);
	list_add(&knod_map->list, &priv->dead_maps);
	WRITE_ONCE(priv->maps_gc_pending, true);
	mutex_unlock(&knodev->lock);
	offmap->dev_priv = NULL;
}

static int __knod_bpf_map_lookup_elem(struct bpf_offloaded_map *offmap,
				      void *key, void *value)
{
	unsigned int idx = *(unsigned int *)key;
	struct knod_bpf_map_obj *knod_map_obj;
	struct knod_bpf_map *knod_map;
	void *bucket;
	u32 stride;
	int i;

	knod_map = (struct knod_bpf_map *)offmap->dev_priv;
	if (!knod_map || !knod_map->mem || !knod_map->mem->kaddr ||
	    (knod_map->hash_elems_mem && !knod_map->hash_elems_mem->kaddr)) {
		pr_err("knod_bpf: lookup on freed/invalid map (dev_priv=%p)\n",
		       offmap->dev_priv);
		return -ENODEV;
	}
	knod_map_obj = knod_map->knod_map_obj;

	if (knod_map_obj->map_type == BPF_MAP_TYPE_ARRAY) {
		if (*(unsigned int *)key >= knod_map_obj->max_entries)
			return -ENOENT;
		bucket = knod_bpf_array_value_ptr(knod_map_obj, idx);

		unsafe_memcpy(value, bucket, knod_map_obj->value_size,
			      "knod array values live in a variable-sized GPU map tail");
	} else if (knod_map_obj->map_type == BPF_MAP_TYPE_PERCPU_ARRAY) {
		if (idx >= knod_map_obj->max_entries)
			return -ENOENT;
		stride = round_up(knod_map_obj->value_size, 8);
		bucket = &knod_map_obj->bucket[0];
		bucket += (idx * knod_map_obj->value_size);
		for (i = 0; i < knod_map_obj->meta.ameta.n_instances; i++)
			unsafe_memcpy(value + i * stride,
				      bucket + i *
				      knod_map_obj->meta.ameta
				      .per_instance_size,
				      knod_map_obj->value_size,
				      "knod percpu array values live in a variable-sized GPU map tail");
	} else if (knod_map_obj->map_type == BPF_MAP_TYPE_HASH ||
		   knod_map_obj->map_type == BPF_MAP_TYPE_PERCPU_HASH) {
		return knod_bpf_map_hash_lookup_elem(knod_map, knod_map_obj,
						     key, value);
	}

	return 0;
}

#define KNOD_MAP_QUIESCE_MS	100

static int knod_bpf_map_op_begin(struct knod_bpf_priv *priv)
{
	u64 request;

	mutex_lock(&priv->map_op_lock);
	if (!priv->worker_task)
		return 0;

	/* Publish the request before the worker acknowledges this generation. */
	request = priv->map_op_request + 1;
	WRITE_ONCE(priv->map_op_quiesce, true);
	smp_store_release(&priv->map_op_request, request);
	/* Empty depth alone can race with an unpublished submission. */
	if (!wait_event_timeout(priv->map_op_wq,
				smp_load_acquire(&priv->map_op_ack) == request,
				msecs_to_jiffies(KNOD_MAP_QUIESCE_MS))) {
		WRITE_ONCE(priv->dispatch_fault, true);
		WRITE_ONCE(priv->map_op_quiesce, false);
		mutex_unlock(&priv->map_op_lock);
		return -ETIMEDOUT;
	}
	return 0;
}

static void knod_bpf_map_op_end(struct knod_bpf_priv *priv)
{
	/* Make host writes visible before allowing another GPU dispatch. */
	knod_bpf_gpu_mem_fence(priv);
	smp_store_release(&priv->map_op_quiesce, false);
	wake_up(&priv->map_op_wq);
	mutex_unlock(&priv->map_op_lock);
}

static int __knod_bpf_map_update_elem(struct bpf_offloaded_map *offmap,
				      void *key, void *value, u64 flags)
{
	struct knod_bpf_map *knod_map = (struct knod_bpf_map *)offmap->dev_priv;
	struct knod_bpf_map_obj *knod_map_obj;
	struct knod_bpf_priv *priv;
	unsigned int idx = *(unsigned int *)key;
	struct knod_dev *knodev;
	void *bucket;
	u32 stride;
	int i;

	if (!knod_map || !knod_map->mem || !knod_map->mem->kaddr)
		return -ENODEV;
	knod_map_obj = knod_map->knod_map_obj;
	priv = knod_map->priv;
	knodev = priv->knodev;
	if (knod_map_obj->map_type == BPF_MAP_TYPE_ARRAY) {
		if (idx >= knod_map_obj->max_entries)
			return -ENOENT;

		bucket = knod_bpf_array_value_ptr(knod_map_obj, idx);
		unsafe_memcpy(bucket, value, knod_map_obj->value_size,
			      "knod array values live in a variable-sized GPU map tail");
		knod_bpf_gpu_mem_fence(priv);
		return 0;
	} else if (knod_map_obj->map_type == BPF_MAP_TYPE_PERCPU_ARRAY) {
		if (idx >= knod_map_obj->max_entries)
			return -ENOENT;
		stride = round_up(knod_map_obj->value_size, 8);
		bucket = &knod_map_obj->bucket[0];
		bucket += (idx * knod_map_obj->value_size);
		for (i = 0; i < knod_map_obj->meta.ameta.n_instances; i++)
			unsafe_memcpy(bucket + i *
				      knod_map_obj->meta.ameta
				      .per_instance_size,
				      value + i * stride,
				      knod_map_obj->value_size,
				      "knod percpu array values live in a variable-sized GPU map tail");
		knod_bpf_gpu_mem_fence(priv);
		return 0;
	} else if (knod_map_obj->map_type == BPF_MAP_TYPE_HASH ||
		   knod_map_obj->map_type == BPF_MAP_TYPE_PERCPU_HASH) {
		int ret = -ENOENT;

		mutex_lock(&knodev->lock);
		list_for_each_entry(knod_map, &knodev->accel->xdp.bound_maps,
				    list) {
			if (knod_map->knod_map_obj == knod_map_obj) {
				mutex_unlock(&knodev->lock);
				ret = knod_bpf_map_op_begin(priv);
				if (ret)
					return ret;
				ret = knod_bpf_map_hash_update_elem(knod_map,
						knod_map_obj,
								    key, value);
				knod_bpf_map_op_end(priv);
				return ret;
			}
		}
		mutex_unlock(&knodev->lock);
	}

	return -ENOENT;
}

static int __knod_bpf_map_delete_elem(struct bpf_offloaded_map *offmap,
				      void *key)
{
	struct knod_bpf_map *knod_map = (struct knod_bpf_map *)offmap->dev_priv;
	struct knod_bpf_map_obj *knod_map_obj;
	struct knod_bpf_priv *priv;
	int ret;

	if (!knod_map || !knod_map->mem || !knod_map->mem->kaddr)
		return -ENODEV;
	knod_map_obj = knod_map->knod_map_obj;
	priv = knod_map->priv;

	if (knod_map_obj->map_type == BPF_MAP_TYPE_ARRAY ||
	    knod_map_obj->map_type == BPF_MAP_TYPE_PERCPU_ARRAY)
		return 0;
	else if (knod_map_obj->map_type == BPF_MAP_TYPE_HASH ||
		 knod_map_obj->map_type == BPF_MAP_TYPE_PERCPU_HASH) {
		ret = knod_bpf_map_op_begin(priv);
		if (ret)
			return ret;
		ret = knod_bpf_map_hash_delete_elem(knod_map, knod_map_obj,
						    key);
		knod_bpf_map_op_end(priv);
		return ret;
	}

	return -ENOENT;
}

static void knod_bpf_map_gc_process(struct knod_bpf_map *knod_map)
{
	struct knod_bpf_map_obj *knod_map_obj = knod_map->knod_map_obj;
	unsigned int *gc_list = knod_map->gc_mem->kaddr;
	unsigned int *queue = knod_map->queue_mem->kaddr;
	void *elems = knod_map->hash_elems_mem->kaddr;
	unsigned int *bucket = (unsigned int *)&knod_map_obj->bucket[0];
	unsigned int elem_size = knod_map_obj->meta.hmeta.elem_size;
	unsigned int gc_count, cur, i;

	gc_count = READ_ONCE(knod_map_obj->meta.hmeta.gc_count);
	if (!gc_count)
		return;

	for (i = 0; i < gc_count; i++) {
		unsigned int del_id = gc_list[i];
		struct knod_bpf_hash_elem_obj *del_elem =
			elems + (del_id * elem_size);
		unsigned int hash, eid;
		struct knod_bpf_hash_elem_obj *e, *pe;

		hash = jhash(&del_elem->kv[0], knod_map_obj->key_size,
			     knod_map_obj->meta.hmeta.hashrnd);
		hash = hash & (knod_map_obj->meta.hmeta.n_buckets - 1);

		eid = bucket[hash];
		pe = NULL;
		while (eid != KNOD_BPF_HASH_NEXT_END) {
			e = elems + (eid * elem_size);
			if (e == del_elem) {
				unsigned int next = e->next &
						   KNOD_BPF_HASH_NEXT_MASK;
				if (pe)
					pe->next =
						(pe->next &
						 KNOD_BPF_HASH_NEXT_DELETED) |
						next;
				else
					bucket[hash] = next;

				e->next = KNOD_BPF_HASH_NEXT_END;
				knod_bpf_hash_free_value(knod_map_obj, e);

				cur = knod_map_obj->meta.hmeta.cur;
				queue[cur] = del_id;
				knod_map_obj->meta.hmeta.cur = cur + 1;
				break;
			}
			pe = e;
			eid = e->next & KNOD_BPF_HASH_NEXT_MASK;
		}
	}

	WRITE_ONCE(knod_map_obj->meta.hmeta.gc_count, 0);
}

/*
 * Per-loop map maintenance, run from the worker loop head (outside any
 * rcu_read_lock_bh, since knod_free_mem() may sleep).  All bound_maps access
 * is serialized under knodev->lock -- the same lock map_alloc/map_free use:
 * GC live HASH maps, then reap maps that detach moved onto dead_maps.  The
 * caller holds map_op_lock with the dispatch pipe empty. Pending maintenance
 * suppresses new submissions until this function and its write fence finish.
 */
#define KNOD_BPF_MAPS_TICK_INTERVAL 65536

static void knod_bpf_maps_tick(struct knod_bpf_priv *priv)
{
	struct knod_dev *knodev = priv->knodev;
	struct knod_bpf_map *knod_map, *tmp;
	LIST_HEAD(reap);

	mutex_lock(&knodev->lock);
	list_for_each_entry(knod_map, &knodev->accel->xdp.bound_maps, list) {
		if (knod_map->knod_map_obj->map_type == BPF_MAP_TYPE_HASH ||
		    knod_map->knod_map_obj->map_type == BPF_MAP_TYPE_PERCPU_HASH)
			knod_bpf_map_gc_process(knod_map);
	}
	list_splice_init(&priv->dead_maps, &reap);
	/* A later map_free republishes its request under this same lock. */
	WRITE_ONCE(priv->maps_gc_pending, false);
	mutex_unlock(&knodev->lock);

	list_for_each_entry_safe(knod_map, tmp, &reap, list) {
		if (knod_map->gc_mem)
			knod_free_mem(priv->knod, knod_map->gc_mem);
		if (knod_map->queue_mem)
			knod_free_mem(priv->knod, knod_map->queue_mem);
		if (knod_map->hash_elems_mem)
			knod_free_mem(priv->knod, knod_map->hash_elems_mem);
		if (knod_map->mem)
			knod_free_mem(priv->knod, knod_map->mem);
		kfree(knod_map);
	}
}

/* Completion mode: 0 = event (default, sleep on the AQL signal interrupt),
 * 1 = poll (busy-spin the signal value).  Selectable via debugfs.
 */
static bool knod_bpf_poll_mode;

/* Max spacing (microseconds) between dispatch-ahead submissions.  Once a
 * dispatch is in flight the worker waits up to this long before submitting
 * the next so it batches the packets arriving meanwhile, letting inflight
 * grow >= 2 without degenerating into one-packet dispatches.  This is a
 * ceiling only: an empty pipe submits at once to keep the GPU fed, and a
 * completed dispatch is always retired without waiting.  To actually build
 * depth the value must be below the GPU execution time of a dispatch.
 * 0 disables spacing (submit as soon as the ring has anything).
 */
static u32 knod_bpf_dispatch_delay_us = 20;

/* Retry the signal this often while blocked, so that missing a completion
 * interrupt costs one dispatch rather than the whole expire budget.
 */
#define KNOD_BPF_WAIT_MS	1

static void knod_bpf_wait_event(struct knod_bpf_priv *priv)
{
	struct kfd_event_data events = {
		.event_id = priv->knod->aql_event[0].id,
	};
	u32 timeout_ms = KNOD_BPF_WAIT_MS;
	u32 wait_result;

	knod_wait_on_events(priv->knod->process, 1, &events, true,
			    &timeout_ms, &wait_result);
}

static bool knod_bpf_submit_work(struct knod_bpf_priv *priv)
{
	struct knod_bpf_work_sq *sqw;
	struct knod_bpf_stats *stats = &priv->stats;
	ktime_t dispatch_start;

	if (READ_ONCE(priv->dispatch_fault))
		return false;
	if (knod_bpf_persistent && priv->persistent_running &&
	    priv->persistent_sequence == U64_MAX)
		return false;

	if (priv->inflight_cnt >= KNOD_BPF_INFLIGHT)
		return false;

	/* Pace dispatch-ahead so the next dispatch batches the packets that
	 * arrive during this window instead of firing one-packet dispatches.
	 * An empty pipe skips the wait so the GPU is never left idle.
	 */
	if (priv->inflight_cnt &&
	    ktime_before(ktime_get(), priv->next_dispatch_time))
		return false;

	if (static_branch_unlikely(&knod_stats_key))
		dispatch_start = ktime_get();

	sqw = knod_prepare_bpf(priv);
	if (!sqw)
		return false;

	if (static_branch_unlikely(&knod_stats_key)) {
		u64 dns = ktime_to_ns(ktime_sub(ktime_get(),
						dispatch_start));

		stats->dispatch_total_ns += dns;
		stats->dispatch_count++;
		if (dns > stats->dispatch_max_ns)
			stats->dispatch_max_ns = dns;
	}

	knod_submit_bpf(priv, sqw);
	priv->inflight[priv->inflight_cnt++] = sqw;
	priv->next_dispatch_time =
		ktime_add_us(ktime_get(),
			     READ_ONCE(knod_bpf_dispatch_delay_us));
	return true;
}

static void knod_bpf_record_completion(struct knod_bpf_priv *priv,
				       struct knod_bpf_work_sq *sqw)
{
	struct knod_bpf_stats *stats = &priv->stats;
	u64 ns;
	int bucket;

	if (!static_branch_unlikely(&knod_stats_key))
		return;

	ns = ktime_to_ns(ktime_sub(ktime_get(), sqw->dispatch_time));
	stats->completion_total_ns += ns;
	stats->completion_count++;

	if (ns > stats->completion_max_ns)
		stats->completion_max_ns = ns;

	if (ns < 1000)
		bucket = 0;
	else
		bucket = min(ilog2(ns / 1000) + 1,
			     KNOD_LAT_BUCKETS - 1);
	stats->completion_hist[bucket]++;
}

static bool knod_bpf_poll_complete(struct knod_bpf_priv *priv,
				   struct knod_bpf_work_sq *sqw)
{
	if (!sqw)
		return false;

	if (knod_bpf_sqw_done(priv, sqw)) {
		dma_rmb();
		knod_bpf_record_completion(priv, sqw);
		return true;
	}

	if (time_after(jiffies, sqw->expire) && !READ_ONCE(priv->dispatch_fault)) {
		priv->stats.expire_count++;
		WRITE_ONCE(priv->dispatch_fault, true);
		pr_warn("knod: dispatch timed out; retaining GPU-owned buffers\n");
	}

	return false;
}

static void knod_bpf_schedule_pending_napi(struct knod_bpf_priv *priv)
{
	struct knod_dev *knodev = priv->knodev;
	int qi;

	for (qi = 0; qi < priv->nr_works; qi++) {
		if (spsc_pending(&knodev->wpriv[qi].spsc_bds) &&
		    knodev->wpriv[qi].napi)
			napi_schedule(knodev->wpriv[qi].napi);
	}
}

static int knod_bpf_worker(void *arg)
{
	struct knod_bpf_priv *priv = arg;
	struct knod_bpf_work_sq *sqw;
	bool progressed;
	bool quiesce;
	u64 map_request;

	while (!kthread_should_stop()) {
		if (kthread_should_park()) {
			knod_bpf_drain_worker(priv);
			kthread_parkme();
			continue;
		}

		/* Advance the maintenance cadence even while the pipe stays full. */
		if (!(++priv->maps_tick_skip & (KNOD_BPF_MAPS_TICK_INTERVAL - 1)))
			WRITE_ONCE(priv->maps_gc_pending, true);

		/* Reclaim map elements only after all GPU users have completed,
		 * and exclude host map mutations while processing their free lists.
		 */
		if (READ_ONCE(priv->maps_gc_pending) && !priv->inflight_cnt &&
		    !READ_ONCE(priv->map_op_quiesce) &&
		    mutex_trylock(&priv->map_op_lock)) {
			knod_bpf_persistent_stop(priv);
			knod_bpf_maps_tick(priv);
			knod_bpf_gpu_mem_fence(priv);
			mutex_unlock(&priv->map_op_lock);
		}

		progressed = false;
		quiesce = smp_load_acquire(&priv->map_op_quiesce);
		map_request = quiesce ? smp_load_acquire(&priv->map_op_request) : 0;

		rcu_read_lock_bh();
		/* Retire in SPSC order, checking each dispatch's own signal. */
		while (priv->inflight_cnt &&
		       knod_bpf_poll_complete(priv, priv->inflight[0])) {
			sqw = priv->inflight[0];
			if (--priv->inflight_cnt)
				memmove(priv->inflight, priv->inflight + 1,
					priv->inflight_cnt *
					sizeof(priv->inflight[0]));
			priv->inflight[priv->inflight_cnt] = NULL;
			knod_complete_acquire(priv, sqw);
			knod_complete_napi(priv, sqw);
			progressed = true;
		}

		if (!priv->inflight_cnt)
			WRITE_ONCE(priv->dispatch_fault, false);

		/* Keep the pipe full: dispatch ahead up to KNOD_BPF_INFLIGHT.
		 * Staging self-limits, so this stops once the ring is drained.
		 */
		if (unlikely(quiesce)) {
			/* Acknowledge outside RCU after any resident shader stops. */
		} else if (!READ_ONCE(priv->maps_gc_pending)) {
			while (knod_bpf_submit_work(priv))
				progressed = true;
		}
		rcu_read_unlock_bh();

		/* The terminal wait may sleep. No RCU read lock may cross it. */
		if (quiesce && !priv->inflight_cnt) {
			knod_bpf_persistent_stop(priv);
			smp_store_release(&priv->map_op_ack, map_request);
			wake_up(&priv->map_op_wq);
		} else if (knod_bpf_persistent && !priv->inflight_cnt &&
			   priv->persistent_running && priv->persistent_sequence == U64_MAX) {
			knod_bpf_persistent_stop(priv);
		}

		if (!priv->inflight_cnt) {
			knod_bpf_schedule_pending_napi(priv);
			usleep_range(100, 200);
		} else if (!progressed) {
			/* Block on the event only when there is nothing else to
			 * do with the time: not while a drain is waiting on the
			 * pipe, and not while the pacing window is still open
			 * and the next submit is due.
			 */
			if (quiesce || knod_bpf_poll_mode || knod_bpf_persistent)
				cpu_relax();
			else if (priv->inflight_cnt < KNOD_BPF_INFLIGHT &&
				 ktime_before(ktime_get(), priv->next_dispatch_time))
				cpu_relax();
			else
				knod_bpf_wait_event(priv);
		}
	}

	return 0;
}

static void knod_bpf_sq_init(struct knod_bpf_priv *priv)
{
	struct knod_bpf_work_sq *sqw;
	int i;

	priv->worker_task = NULL;
	priv->inflight_cnt = 0;
	INIT_LIST_HEAD(&priv->free_list_sqw);

	for (i = 0; i < 32; i++) {
		sqw = kvzalloc_obj(struct knod_bpf_work_sq, GFP_KERNEL);
		if (!sqw)
			continue;

		sqw->param = knod_alloc_mem(priv->knod,
					    KNOD_SQ_PARAM_BYTES,
					    KFD_IOC_ALLOC_MEM_FLAGS_GTT |
					    KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
					    KFD_IOC_ALLOC_MEM_FLAGS_COHERENT);
		if (IS_ERR_OR_NULL(sqw->param)) {
			kvfree(sqw);
			continue;
		}
		memset(sqw->param->kaddr, 0, KNOD_SQ_PARAM_BYTES);
		*knod_bpf_sqw_signal(sqw) = *(struct amd_signal *)
			priv->knod->kaql[0].queue_signal->kaddr;
		knod_bpf_sqw_signal(sqw)->value = 0;
		INIT_LIST_HEAD(&sqw->list);
		list_add(&sqw->list, &priv->free_list_sqw);
		sqw->backlogs = 0;
	}
}

static void knod_bpf_free_sqw(struct knod_bpf_priv *priv,
			      struct knod_bpf_work_sq *sqw)
{
	if (!sqw)
		return;

	knod_free_mem(priv->knod, sqw->param);
	kvfree(sqw);
}

static void knod_bpf_free_sqw_list(struct knod_bpf_priv *priv,
				   struct list_head *head)
{
	struct knod_bpf_work_sq *sqw, *tmp;

	list_for_each_entry_safe(sqw, tmp, head, list) {
		list_del(&sqw->list);
		knod_bpf_free_sqw(priv, sqw);
	}
}

static void knod_bpf_sq_exit(struct knod_bpf_priv *priv)
{
	if (!priv->knod)
		return;

	knod_bpf_stop_worker(priv);

	knod_bpf_free_sqw_list(priv, &priv->free_list_sqw);
	if (priv->persistent_mem) {
		knod_free_mem(priv->knod, priv->persistent_mem);
		priv->persistent_mem = NULL;
	}
	priv->inflight_cnt = 0;
}

static void knod_priv_exit(struct knod_bpf_priv *priv)
{
	struct knod_dev *knodev = priv->knodev;
	struct knod_bpf_map *knod_map, *tmp;
	LIST_HEAD(reap);

	knod_bpf_sq_exit(priv);

	/*
	 * The dispatch worker is not stopped until the next feature registers
	 * its own worker, so it may still be running knod_bpf_maps_tick() here.
	 * Serialize under knodev->lock and splice both lists to a local one:
	 * whichever side splices first frees them, the other sees them empty.
	 * Free outside the lock since knod_free_mem() may sleep.
	 */
	mutex_lock(&knodev->lock);
	list_splice_init(&knodev->accel->xdp.bound_maps, &reap);
	list_splice_init(&priv->dead_maps, &reap);
	mutex_unlock(&knodev->lock);

	list_for_each_entry_safe(knod_map, tmp, &reap, list) {
		if (knod_map->gc_mem)
			knod_free_mem(priv->knod, knod_map->gc_mem);
		if (knod_map->queue_mem)
			knod_free_mem(priv->knod, knod_map->queue_mem);
		if (knod_map->hash_elems_mem)
			knod_free_mem(priv->knod, knod_map->hash_elems_mem);
		if (knod_map->mem)
			knod_free_mem(priv->knod, knod_map->mem);
		kfree(knod_map);
	}

	kfree(priv->prog_buf);
	kfree(priv->pass_prog_buf);
	if (priv->pass_knod_prog) {
		knod_prog_free(priv->pass_knod_prog);
		priv->pass_knod_prog = NULL;
	}
	/* kernels[] are owned by knod (freed in knod_release_ctx), not here */
	if (priv->pass_meta_buf)
		knod_free_mem(priv->knod, priv->pass_meta_buf);
}

static int knod_priv_init(struct knod_bpf_priv *priv)
{
	struct knod_dev *knodev = priv->knodev;
	int pass_meta_buf_size;
	int index;

	priv->prog = NULL;
	INIT_LIST_HEAD(&priv->free_list_sqw);
	priv->worker_task = NULL;
	priv->inflight_cnt = 0;
	priv->dispatch_fault = false;
	priv->map_op_request = 0;
	priv->map_op_ack = 0;
	priv->map_op_quiesce = false;
	priv->maps_gc_pending = false;
	priv->maps_tick_skip = 0;
	mutex_init(&priv->map_op_lock);
	init_waitqueue_head(&priv->map_op_wq);
	INIT_LIST_HEAD(&priv->dead_maps);
	priv->maps_tick_skip = 0;

	priv->nr_works = knod_bpf_active_rxq_count(knodev->netdev);
	if (!priv->nr_works) {
		pr_warn("knod_bpf: no active RX queues for %s\n",
			knodev->netdev ? knodev->netdev->name : "<null>");
		return -EINVAL;
	}

	if (knod_bpf_persistent &&
	    (priv->isa_version != 10 || knod_bpf_jit_engine != 1 ||
	     knod_bpf_wgp || knod_bpf_workgroups != 256 ||
	     knod_bpf_cycle_probe || priv->nr_works > priv->knod->cu_count))
		return -EOPNOTSUPP;
	if (knod_bpf_persistent && !priv->knod->control_mem_coherent)
		return -EOPNOTSUPP;
	priv->prog_buf = kzalloc(KNOD_BPF_PROG_BUF_SIZE, GFP_KERNEL);
	if (!priv->prog_buf)
		return -ENOMEM;

	for (index = 0; index < priv->nr_works; index++)
		priv->queue_base_gaddr[index] = priv->knod->buf[index]->gaddr;

	/* Per-queue PASS slot count; sizes the shader pass_meta_buf below.
	 * At most one PASS packet per dispatched slot, i.e. batch_size.
	 */
	priv->pass_pkts_per_queue = knod_bpf_batch_size(priv);

	/* Allocate GTT buffer for per-queue shader PASS copy */
	pass_meta_buf_size = priv->nr_works * priv->pass_pkts_per_queue *
			KNOD_PASS_SLOT_SIZE;
	priv->pass_meta_buf = knod_alloc_mem(priv->knod, pass_meta_buf_size,
					KFD_IOC_ALLOC_MEM_FLAGS_GTT |
					KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
					KFD_IOC_ALLOC_MEM_FLAGS_COHERENT);
	if (IS_ERR(priv->pass_meta_buf)) {
		pr_warn("KNOD: failed to allocate pass_meta_buf\n");
		priv->pass_meta_buf = NULL;
		knod_priv_exit(priv);
		return -ENOMEM;
	}
	pr_debug("KNOD: pass_meta_buf gaddr=0x%llx..0x%llx size=%d nr_q=%d pass_pkts_per_queue=%u\n",
		 priv->pass_meta_buf->gaddr,
		 priv->pass_meta_buf->gaddr + pass_meta_buf_size,
		 priv->pass_meta_buf->size, priv->nr_works,
		 priv->pass_pkts_per_queue);

	/* GPU->host delivery pages come from the framework per-queue page_pool
	 * (knodev->wpriv[q].pass_pool): the producer allocs from it and the
	 * NAPI drain recycles, so no per-feature delivery BO is allocated here.
	 */

	if (knod_bpf_persistent) {
		priv->persistent_mem = knod_alloc_mem(priv->knod, PAGE_SIZE,
			KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
			KFD_IOC_ALLOC_MEM_FLAGS_COHERENT);
		if (IS_ERR_OR_NULL(priv->persistent_mem)) {
			priv->persistent_mem = NULL;
			knod_priv_exit(priv);
			return -ENOMEM;
		}
	}
	knod_bpf_sq_init(priv);
	if (list_empty(&priv->free_list_sqw)) {
		knod_priv_exit(priv);
		return -ENOMEM;
	}

	return 0;
}

static struct knod_bpf_priv *__knod_accel_xdp_init(struct knod_accel *accel,
						   struct knod_dev *knodev)
{
	struct knod *knod = (struct knod *)knodev->accel->priv;
	struct knod_bpf_priv *priv;
	int err;

	if (knod_bpf_vgpr_reserve != 80 && knod_bpf_vgpr_reserve != 128 &&
	    knod_bpf_vgpr_reserve != 256)
		return ERR_PTR(-EINVAL);

	priv = kzalloc_obj(struct knod_bpf_priv, GFP_KERNEL);
	if (!priv)
		return ERR_PTR(-ENOMEM);

	/* Every routine now comes from the blob; without it there is nothing
	 * to splice, so a missing or ABI-mismatched blob fails the attach
	 * rather than deferring to a program that then cannot be built.
	 */
	err = knod_blob_load(knod, &priv->blob,
			     knod_bpf_persistent ? "bpf-persistent" : "bpf");
	if (err) {
		kfree(priv);
		return ERR_PTR(err);
	}

	/* The prologue and both epilogue halves wrap every program, so a blob
	 * missing any of them cannot build one - refuse the attach now rather
	 * than at the first program.  A key-sized routine (a map op) is the
	 * program's own business and is checked when it is JITed.
	 */
	if (!knod_blob_find(&priv->blob, KNOD_BLOB_PROLOGUE, 0, NULL) ||
	    !knod_blob_find(&priv->blob, KNOD_BLOB_EPILOGUE, 0, NULL)) {
		pr_warn("knod_bpf: blob is missing a prologue or epilogue\n");
		knod_blob_free(&priv->blob);
		kfree(priv);
		return ERR_PTR(-EINVAL);
	}

	INIT_LIST_HEAD(&priv->list);
	if (knod_bpf_workgroups < KNOD_BPF_WORKGROUPS_MIN ||
	    knod_bpf_workgroups > KNOD_BPF_WORKGROUPS_MAX)
		knod_bpf_workgroups = KNOD_BPF_WORKGROUPS_DEFAULT;
	/* Round to a power of two so the prologue can reach a lane's slot with
	 * a shift.  A multiply would need the count itself, which is a module
	 * parameter and so cannot be an immediate in a prebuilt shader.
	 */
	else if (!is_power_of_2(knod_bpf_workgroups))
		knod_bpf_workgroups = 1U << ilog2(knod_bpf_workgroups);

	if (knod_bpf_expire < KNOD_BPF_EXPIRE_MIN ||
	    knod_bpf_expire > KNOD_BPF_EXPIRE_MAX)
		knod_bpf_expire = KNOD_BPF_EXPIRE_DEFAULT;
	pr_debug("workgroup size %d\n", knod_bpf_workgroups);
	pr_debug("expire time = %dms\n", knod_bpf_expire);

	INIT_LIST_HEAD(&accel->xdp.bound_maps);
	accel->flags |= KNOD_FLAGS_XDP;
	accel->xdp.priv = priv;
	list_add(&priv->list, &priv_list);

	priv->knod = knod;
	priv->accel = accel;
	priv->knodev = knodev;
	priv->dev = knodev->netdev;

	priv->isa_version = knod->isa_version;
	knod->coherent_control_required = knod_bpf_persistent;

	/*
	 * Only permanent per-attach state is set up here; the GPU compute
	 * buffers (knod_priv_init/kfd_kernel_init) are allocated by
	 * ->activate() when the BPF feature is selected.
	 */

	return priv;
}

/* Feature select: allocate the BPF GPU compute resources. */
static int knod_bpf_activate(struct knod_dev *knodev)
{
	struct knod_accel *accel = knodev->accel;
	struct knod_bpf_priv *priv = accel->xdp.priv;
	struct knod *knod = accel->priv;

	/*
	 * Past gfx11 the emitters would warn and drop every instruction
	 * while the kernel descriptor went out unwritten, so the dispatch
	 * would run whatever was in that VRAM.  Refuse rather than hang.
	 */
	if (priv->isa_version < 9 || priv->isa_version > 11) {
		pr_warn("knod_bpf: XDP offload needs gfx9 to gfx11, this GPU is gfx%d\n",
			priv->isa_version);
		return -EOPNOTSUPP;
	}

	/*
	 * Pin the module while BPF is the selected feature: the core calls
	 * into these ops, so it must not be unloaded until feature->none.
	 * (No-op when built in - THIS_MODULE is NULL.)
	 */
	if (!try_module_get(THIS_MODULE))
		return -ENODEV;

	if (knod_priv_init(priv)) {
		WARN_ON_ONCE(1);
		module_put(THIS_MODULE);
		return -EINVAL;
	}
	if (kfd_kernel_init(knod, priv)) {
		knod_priv_exit(priv);
		module_put(THIS_MODULE);
		return -ENOMEM;
	}

	priv->start = 0;
	return 0;
}

/* Feature deselect: free the BPF GPU compute resources. */
static void knod_bpf_deactivate(struct knod_dev *knodev)
{
	struct knod_bpf_priv *priv = knodev->accel->xdp.priv;

	knod_priv_exit(priv);
	module_put(THIS_MODULE);
}

/* True while a user XDP prog or offloaded map is still bound to this accel. */
static bool knod_bpf_busy(struct knod_dev *knodev)
{
	struct knod_accel *accel = knodev->accel;
	struct knod_bpf_priv *priv = accel->xdp.priv;

	if (!priv)
		return false;
	return READ_ONCE(priv->prog) || !list_empty(&accel->xdp.bound_maps);
}

static void __knod_accel_xdp_exit(struct knod_accel *accel,
				  struct knod_bpf_priv *priv)
{
	/* GPU compute buffers are freed by ->deactivate(); free the rest. */
	memset(&accel->xdp, 0, sizeof(struct knod_accel_xdp));
	accel->flags &= ~KNOD_FLAGS_XDP;
	list_del(&priv->list);
	knod_blob_free(&priv->blob);
	kfree(priv);
}

static struct knod_insn_meta *knod_bpf_goto_meta(struct knod_prog *knod_prog,
						 struct knod_insn_meta *meta,
						 unsigned int insn_idx)
{
	unsigned int forward, backward, i;

	backward = meta->bpf_insn_idx - insn_idx;
	forward = insn_idx - meta->bpf_insn_idx;

	if (min(forward, backward) > knod_prog->n_insns - insn_idx - 1) {
		backward = knod_prog->n_insns - insn_idx - 1;
		meta = knod_prog_last_meta(knod_prog);
	}
	if (min(forward, backward) > insn_idx && backward > insn_idx) {
		forward = insn_idx;
		meta = knod_prog_first_meta(knod_prog);
	}

	if (forward < backward)
		for (i = 0; i < forward; i++)
			meta = knod_meta_next(meta);
	else
		for (i = 0; i < backward; i++)
			meta = knod_meta_prev(meta);

	return meta;
}

static int knod_bpf_native_stack_offset(const struct bpf_reg_state *reg,
				       int frame, int *offset)
{
	s64 off = (s64)reg->var_off.value;

	if (reg->type != PTR_TO_STACK || reg->frameno != frame ||
	    !tnum_is_const(reg->var_off) || off < -512 || off > 0)
		return -EOPNOTSUPP;
	*offset = off;
	return 0;
}

static int knod_bpf_native_stack_arg(struct knod_bpf_reg_state *saved,
				     const struct bpf_reg_state *reg, int frame)
{
	int off, err;

	err = knod_bpf_native_stack_offset(reg, frame, &off);
	if (err)
		return err;
	if (saved->reg.type != NOT_INIT &&
	    (saved->reg.type != PTR_TO_STACK || saved->stack_off != off ||
	     saved->reg.frameno != reg->frameno))
		return -EOPNOTSUPP;
	saved->reg = *reg;
	saved->stack_off = off;
	return 0;
}

static int knod_bpf_check_stack_access(struct knod_prog *knod_prog,
				       struct knod_insn_meta *meta,
				       const struct bpf_reg_state *reg,
				       struct bpf_verifier_env *env)
{
	int off, err;

	err = knod_bpf_native_stack_offset(reg, env->cur_state->curframe, &off);
	if (err)
		return err;
	/* Equal alignment is insufficient: the emitted LDS offset is fixed. */
	if (meta->ptr.type != NOT_INIT &&
	    (meta->ptr.type != PTR_TO_STACK ||
	     meta->ptr.var_off.value != reg->var_off.value ||
	     meta->ptr.frameno != reg->frameno))
		return -EOPNOTSUPP;
	if (BPF_CLASS(meta->insn.code) == BPF_LDX)
		meta->sreg.stack_off = off;
	else
		meta->dreg.stack_off = off;
	knod_prog->max_stack_off = min(knod_prog->max_stack_off,
				      off + meta->insn.off);
	return 0;
}

static struct knod_insn_meta *
knod_bpf_lookup_prev_meta_by_dreg(struct knod_prog *knod_prog,
				  struct knod_insn_meta *meta,
				  int dreg_id)
{
	list_for_each_entry_continue_reverse(meta, &knod_prog->insns, l) {
		if (!is_mbpf_alu(meta) &&
		    !is_mbpf_load(meta) &&
		    !is_mbpf_store(meta))
			continue;
		if (meta->insn.dst_reg == dreg_id)
			return meta;
	}

	return NULL;
}

static int knod_bpf_check_ptr(struct knod_prog *knod_prog,
			      struct knod_insn_meta *meta,
			      struct bpf_verifier_env *env, u8 reg_no)
{
	const struct bpf_reg_state *reg = cur_regs(env) + reg_no;
	int err;

	if (reg->type != PTR_TO_CTX &&
	    reg->type != PTR_TO_STACK &&
	    reg->type != PTR_TO_MAP_VALUE &&
	    reg->type != PTR_TO_PACKET) {
		knod_jit_dbg(" unsupported ptr type: %d\n", reg->type);
		return -EINVAL;
	}

	if (reg->type == PTR_TO_STACK) {
		err = knod_bpf_check_stack_access(knod_prog, meta, reg, env);
		if (err)
			return err;
	}

	if (meta->ptr.type != NOT_INIT && meta->ptr.type != reg->type) {
		knod_jit_dbg(" ptr type changed for instruction %d -> %d\n",
			meta->ptr.type,
			reg->type);
		return -EINVAL;
	}

	meta->ptr = *reg;

	return 0;
}

static int knod_bpf_update_ptr_off(struct knod_prog *knod_prog,
				   struct knod_insn_meta *meta,
				   struct bpf_verifier_env *env)
{
	struct knod_bpf_reg_state *sreg = &meta->sreg;
	struct knod_bpf_reg_state *dreg = &meta->dreg;
	struct knod_insn_meta *prev_meta;

	if (is_mbpf_load(meta)) {
		if (sreg->reg.type == PTR_TO_PACKET) {
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.src_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			sreg->packet_off = prev_meta->dreg.packet_off;
		}
	} else if (is_mbpf_store(meta)) {
		if (dreg->reg.type == PTR_TO_PACKET) {
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.dst_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			dreg->packet_off = prev_meta->dreg.packet_off;
		}
	}

	return 0;
}

/* A percpu value has one instance per queue, and a queue is one workgroup: 256
 * lanes reach what a CPU reaches alone.
 *
 * Assigning to it is still right - the memory system picks a winner, which is
 * what the last write on a CPU comes to as well.  Reading it, adding to it and
 * writing that back is not: all 256 read the same value and all but one result
 * is thrown away.  On a CPU that sequence needs no atomic, so it is exactly
 * what a program written for one will do.
 *
 * Where it is an add, the three instructions become one atomic: the load turns
 * into an atomic add that hands back what was there before, the add works on
 * that as it always did, and the store has nothing left to do.  The register
 * ends up holding the same value it would have on a CPU, so nothing downstream
 * has to know.
 *
 * Anything else that reads and writes back is turned away, rather than left to
 * run and report a number that is quietly too small.  So is an add whose
 * amount is not settled by the time the load happens, since that is where the
 * atomic now goes.  Only the shape a compiler emits for += and ++ is matched -
 * a longer chain between the load and the store will get through, which is
 * worth knowing when a count still looks low.
 */
/* Whether a stack slot holds something every lane agrees on.
 *
 * A store of a literal does; a store of a register does when the verifier has
 * proved the register could only be one value.  Anything else is treated as
 * differing per lane, which is never wrong here, only slower.
 */
static bool knod_bpf_stack_is_const(struct knod_prog *knod_prog,
				    struct knod_insn_meta *meta, int stack_off)
{
	struct knod_insn_meta *m = meta;

	list_for_each_entry_continue_reverse(m, &knod_prog->insns, l) {
		if (!is_mbpf_store(m) && mbpf_class(m) != BPF_ST)
			continue;
		if (m->dreg.stack_off + m->insn.off != stack_off)
			continue;
		if (mbpf_class(m) == BPF_ST)
			return true;
		return tnum_is_const(m->sreg.reg.var_off);
	}
	return false;
}

/* Whether every lane of a wave reaches the same element of a percpu map, which
 * is what lets the wave send one atomic between them.
 *
 * The instance is picked by the queue and a workgroup is one queue, so what is
 * left to differ is the key.  A key the program wrote as a constant is the same
 * in every lane; one it worked out from the packet is not.
 *
 * Only the plainest shape is taken: the value pointer is still in r0, and the
 * lookup that put it there was handed a constant.  Anything else falls back to
 * an atomic per lane.
 */
static bool knod_bpf_percpu_addr_uniform(struct knod_prog *knod_prog,
					 struct knod_insn_meta *meta)
{
	struct knod_insn_meta *m = meta;

	if (meta->insn.dst_reg != BPF_REG_0)
		return false;

	list_for_each_entry_continue_reverse(m, &knod_prog->insns, l) {
		if (is_mbpf_map_call(m))
			return knod_bpf_stack_is_const(knod_prog, m,
						       m->kreg.stack_off);
		/* Anything else that lands in r0 breaks the trail. */
		if (mbpf_class(m) == BPF_JMP && BPF_OP(m->insn.code) == BPF_CALL)
			return false;
		if ((is_mbpf_alu(m) || is_mbpf_load(m)) &&
		    m->insn.dst_reg == BPF_REG_0)
			return false;
	}
	return false;
}

/* Whether @load reads the same place @store writes. */
static bool knod_bpf_same_place(const struct knod_insn_meta *load,
				const struct knod_insn_meta *store)
{
	return load && is_mbpf_load(load) &&
	       load->insn.src_reg == store->insn.dst_reg &&
	       load->insn.off == store->insn.off;
}

static int knod_bpf_check_percpu_store(struct knod_prog *knod_prog,
				       struct knod_insn_meta *meta)
{
	struct knod_insn_meta *alu, *load;
	const char *why;

	meta->percpu_rmw_add = NULL;
	meta->percpu_rmw_swapped = false;
	meta->percpu_rmw_uniform = false;
	alu = knod_bpf_lookup_prev_meta_by_dreg(knod_prog, meta,
						meta->insn.src_reg);
	if (!alu || !is_mbpf_alu(alu))
		return 0;

	/* An add takes its operands either way round, and a compiler will use
	 * both: what came out of the map can be what the add starts from, or
	 * what it adds on.  Only the first was looked for, so the second went
	 * out as a plain load and store and lost all but one lane of it.
	 */
	load = knod_bpf_lookup_prev_meta_by_dreg(knod_prog, alu,
						 alu->insn.dst_reg);
	if (!knod_bpf_same_place(load, meta)) {
		load = knod_bpf_lookup_prev_meta_by_dreg(knod_prog, alu,
							 alu->insn.src_reg);
		if (!knod_bpf_same_place(load, meta))
			return 0;
		meta->percpu_rmw_swapped = true;
	}

	if (BPF_OP(alu->insn.code) != BPF_ADD) {
		why = "only an add has an atomic form here; write it with __sync_fetch_and_add()";
		goto reject;
	}

	/* There is no atomic narrower than a dword, and a 64-bit one hangs
	 * GFX9 against VRAM.
	 */
	if (BPF_SIZE(meta->insn.code) != BPF_W &&
	    BPF_SIZE(meta->insn.code) != BPF_DW) {
		why = "no atomic is narrower than a dword; widen the value to __u32";
		goto reject;
	}
	if (BPF_SIZE(meta->insn.code) == BPF_DW &&
	    knod_prog->knod->isa_version == 9) {
		/* Saying "use an atomic" here would send them at a wall: the
		 * explicit form is refused too, this part has no 64-bit atomic
		 * at all.
		 */
		why = "this GPU has no 64-bit atomic, so no form of a 64-bit counter offloads; make it __u32";
		goto reject;
	}

	/* A DW repair must preserve the full ALU64 addition, not an ALU32
	 * truncation/zero-extension followed by a wider write.
	 */
	if (BPF_SIZE(load->insn.code) != BPF_SIZE(meta->insn.code) ||
	    (BPF_SIZE(meta->insn.code) == BPF_DW &&
	     mbpf_class(alu) != BPF_ALU64) ||
	    (meta->percpu_rmw_swapped && BPF_SRC(alu->insn.code) != BPF_X) ||
	    (BPF_SRC(alu->insn.code) == BPF_X &&
	     alu->insn.src_reg == alu->insn.dst_reg)) {
		why = "counter load/add/store widths or source versions do not match";
		goto reject;
	}
	{
		struct knod_insn_meta *m = load;
		bool after_add = false;

		/* Keep the repaired RMW in one straight-line value-version
		 * region. Atomics (including FETCH), helpers and partial CFG
		 * joins are not covered by the ALU/load-only clobber ledger.
		 */
		list_for_each_entry_continue(m, &knod_prog->insns, l) {
			if (m->is_merge_point || (m->flags & FLAG_INSN_IS_JUMP_DST)) {
				why = "counter RMW crosses a control-flow join";
				goto reject;
			}
			if (m == meta)
				break;
			if (!(is_mbpf_alu(m) || is_mbpf_load(m)) ||
			    m->insn.dst_reg == meta->insn.dst_reg ||
			    (after_add && BPF_SRC(alu->insn.code) == BPF_X &&
			     m->insn.dst_reg == alu->insn.src_reg)) {
				why = "counter source or address is changed before its store";
				goto reject;
			}
			if (m == alu)
				after_add = true;
		}
		if (m != meta || !after_add) {
			why = "counter RMW does not have a straight-line ADD";
			goto reject;
		}
	}

	{
		struct knod_insn_meta *m;

		/* Verifier callbacks precede CFG annotation. Check raw targets
		 * as well, rather than treating unset merge flags as a proof.
		 */
		list_for_each_entry(m, &knod_prog->insns, l) {
			int cls = mbpf_class(m), op = BPF_OP(m->insn.code);
			s64 target;

			if ((cls != BPF_JMP && cls != BPF_JMP32) ||
			    op == BPF_CALL || op == BPF_EXIT)
				continue;
			target = (s64)m->bpf_insn_idx + 1 +
				 (cls == BPF_JMP32 && op == BPF_JA ?
				  m->insn.imm : m->insn.off);
			if (target > load->bpf_insn_idx && target <= meta->bpf_insn_idx) {
				why = "counter RMW has an alternate incoming path";
				goto reject;
			}
		}
	}

	meta->percpu_rmw_add = alu;
	meta->percpu_rmw_uniform =
		knod_bpf_percpu_addr_uniform(knod_prog, meta);
	return 0;

reject:
	pr_warn("knod_bpf: percpu value at +%d is read and written back (bpf insn %d) where %u lanes would do it at once and lose all but one: %s\n",
		meta->insn.off, meta->bpf_insn_idx, knod_bpf_workgroups, why);
	return -EOPNOTSUPP;
}

static int knod_bpf_check_store(struct knod_prog *knod_prog,
				struct knod_insn_meta *meta,
				struct bpf_verifier_env *env)
{
	const struct bpf_reg_state *reg = cur_regs(env) + meta->insn.dst_reg;

	if (reg->type == PTR_TO_CTX) {
		if (knod_prog->type == BPF_PROG_TYPE_XDP) {
			/* XDP ctx accesses must be 4B in size */
			switch (meta->insn.off) {
			case offsetof(struct xdp_md, rx_queue_index):
				knod_jit_dbg(" queue selection not supported by FW\n");
				return -EOPNOTSUPP;
			}
		}
		knod_jit_dbg(" unsupported store to context field\n");
		return -EOPNOTSUPP;
	}

	if (reg->type == PTR_TO_MAP_VALUE && reg->map_ptr &&
	    (reg->map_ptr->map_type == BPF_MAP_TYPE_PERCPU_ARRAY ||
	     reg->map_ptr->map_type == BPF_MAP_TYPE_PERCPU_HASH)) {
		int err = knod_bpf_check_percpu_store(knod_prog, meta);

		if (err)
			return err;
	}

	return knod_bpf_check_ptr(knod_prog, meta, env, meta->insn.dst_reg);
}

/* NOTE:
 * knod_bpf_lookup_prev_meta_by_dreg(), src_reg vs dst_reg ???????/
 */
static int knod_bpf_check_alu(struct knod_prog *knod_prog,
			      struct knod_insn_meta *meta,
			      struct bpf_verifier_env *env)
{
	const struct bpf_reg_state *sreg = cur_regs(env) + meta->insn.src_reg;
	const struct bpf_reg_state *dreg = cur_regs(env) + meta->insn.dst_reg;
	struct knod_bpf_reg_state *ksreg = &meta->sreg;
	struct knod_bpf_reg_state *kdreg = &meta->dreg;
	struct knod_insn_meta *prev_meta;
	int imm;

	meta->umin_src = min(meta->umin_src, reg_umin(sreg));
	meta->umax_src = max(meta->umax_src, reg_umax(sreg));
	meta->umin_dst = min(meta->umin_dst, reg_umin(dreg));
	meta->umax_dst = max(meta->umax_dst, reg_umax(dreg));

	/* AMDGPU doesn't have divide instructions, we support divide by
	 * constant through reciprocal multiplication. Given NFP support
	 * multiplication no bigger than u32, we'd require divisor and dividend
	 * no bigger than that as well.
	 *
	 * Also eBPF doesn't support signed divide and has enforced this on C
	 * language level by failing compilation. However LLVM assembler hasn't
	 * enforced this, so it is possible for negative constant to leak in as
	 * a BPF_K operand through assembly code, we reject such cases as well.
	 */
	if (is_mbpf_div(meta)) {
		if (meta->umax_dst > U32_MAX) {
			knod_jit_dbg(" dividend is not within u32 value range\n");
			return -EINVAL;
		}
		if (mbpf_src(meta) == BPF_X) {
			if (meta->umin_src != meta->umax_src) {
				knod_jit_dbg(" divisor is not constant\n");
				return -EINVAL;
			}
			if (meta->umax_src > U32_MAX) {
				knod_jit_dbg(" divisor is not within u32 value range\n");
				return -EINVAL;
			}
		}
		if (mbpf_src(meta) == BPF_K && meta->insn.imm < 0) {
			knod_jit_dbg(" divide by negative constant is not supported\n");
			return -EINVAL;
		}
	}

	/* A move copies a stack pointer whole, so its offset is wherever the
	 * source was last set - the frame pointer itself is offset zero.  Keyed
	 * on the source: the hook sees the state before the move, when the
	 * destination is not a pointer yet and the block below is skipped.
	 */
	if ((meta->insn.code == (BPF_ALU | BPF_MOV | BPF_X) ||
	     meta->insn.code == (BPF_ALU64 | BPF_MOV | BPF_X)) &&
	    sreg->type == PTR_TO_STACK) {
		if (meta->insn.src_reg == BPF_REG_FP) {
			kdreg->stack_off = 0;
		} else {
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.src_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			kdreg->stack_off = prev_meta->dreg.stack_off;
		}
	}

	if (dreg->type == PTR_TO_STACK) {
		imm = meta->insn.imm;

		switch (meta->insn.code) {
		/* ALU
		 * If a destination register contains a pointer of STACK,
		 * offset should not be minus.
		 */
		case BPF_ALU | BPF_MOV | BPF_X:
		case BPF_ALU64 | BPF_MOV | BPF_X:
			//r[d] = r[s]; handled above from the source
			break;
		case BPF_ALU | BPF_MOV | BPF_K:
		case BPF_ALU64 | BPF_MOV | BPF_K:
			//r[d] = imm;
			kdreg->stack_off = ksreg->stack_off;
			break;
		case BPF_ALU | BPF_XOR | BPF_X:
		case BPF_ALU64 | BPF_XOR | BPF_X:
			//r[d] ^= r[s];
			knod_jit_dbg(" PTR_TO_STACK with BPF_X is not supported\n");
			return -EINVAL;
		case BPF_ALU | BPF_XOR | BPF_K:
		case BPF_ALU64 | BPF_XOR | BPF_K:
			//r[d] ^= imm;
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.dst_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			kdreg->stack_off = prev_meta->dreg.stack_off ^ imm;
			break;
		case BPF_ALU | BPF_MOD | BPF_X:
		case BPF_ALU64 | BPF_MOD | BPF_X:
			//r[d] %= r[s];
			knod_jit_dbg(" PTR_TO_STACK with BPF_X is not supported\n");
			return -EINVAL;
		case BPF_ALU | BPF_MOD | BPF_K:
		case BPF_ALU64 | BPF_MOD | BPF_K:
			//r[d] %= imm;
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.dst_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			kdreg->stack_off = prev_meta->dreg.stack_off % imm;
			break;
		case BPF_ALU | BPF_AND | BPF_X:
		case BPF_ALU64 | BPF_AND | BPF_X:
			//r[d] &= r[s];
			knod_jit_dbg(" PTR_TO_STACK with BPF_X is not supported\n");
			return -EINVAL;
		case BPF_ALU | BPF_AND | BPF_K:
		case BPF_ALU64 | BPF_AND | BPF_K:
			//r[d] &= imm;
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.dst_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			kdreg->stack_off = prev_meta->dreg.stack_off & imm;
			break;
		case BPF_ALU | BPF_OR | BPF_X:
		case BPF_ALU64 | BPF_OR | BPF_X:
			//r[d] |= r[s];
			knod_jit_dbg(" PTR_TO_STACK with BPF_X is not supported\n");
			return -EINVAL;
		case BPF_ALU | BPF_OR | BPF_K:
		case BPF_ALU64 | BPF_OR | BPF_K:
			//r[d] |= imm;
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.dst_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			kdreg->stack_off = prev_meta->dreg.stack_off | imm;
			break;
		case BPF_ALU | BPF_ADD | BPF_X:
		case BPF_ALU64 | BPF_ADD | BPF_X:
			//r[d] += r[s];
			knod_jit_dbg(" PTR_TO_STACK with BPF_X is not supported\n");
			return -EINVAL;
		case BPF_ALU | BPF_ADD | BPF_K:
		case BPF_ALU64 | BPF_ADD | BPF_K:
			//r[d] += imm;
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.dst_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			kdreg->stack_off = prev_meta->dreg.stack_off + imm;
			break;
		case BPF_ALU | BPF_SUB | BPF_X:
		case BPF_ALU64 | BPF_SUB | BPF_X:
			//r[d] -= r[s];
			knod_jit_dbg(" PTR_TO_STACK with BPF_X is not supported\n");
			return -EINVAL;
		case BPF_ALU | BPF_SUB | BPF_K:
		case BPF_ALU64 | BPF_SUB | BPF_K:
			//r[d] -= imm;
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.dst_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			kdreg->stack_off = prev_meta->dreg.stack_off - imm;
			break;
		case BPF_ALU | BPF_MUL | BPF_X:
		case BPF_ALU64 | BPF_MUL | BPF_X:
			//r[d] *= r[s];
			knod_jit_dbg(" PTR_TO_STACK with BPF_X is not supported\n");
			return -EINVAL;
		case BPF_ALU | BPF_MUL | BPF_K:
		case BPF_ALU64 | BPF_MUL | BPF_K:
			//r[d] *= imm;
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.dst_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			kdreg->stack_off = prev_meta->dreg.stack_off * imm;
			break;
		case BPF_ALU | BPF_DIV | BPF_X:
		case BPF_ALU64 | BPF_DIV | BPF_X:
			//r[d] /= r[s];
			knod_jit_dbg(" PTR_TO_STACK with BPF_X is not supported\n");
			return -EINVAL;
		case BPF_ALU | BPF_DIV | BPF_K:
		case BPF_ALU64 | BPF_DIV | BPF_K:
			//r[d] /= imm;
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.dst_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			kdreg->stack_off = prev_meta->dreg.stack_off / imm;
			break;
		case BPF_ALU | BPF_NEG:
		case BPF_ALU64 | BPF_NEG:
			//r[d] = -r[d];
			break;
		case BPF_ALU | BPF_LSH | BPF_X:
		case BPF_ALU64 | BPF_LSH | BPF_X:
			//r[d] <<= r[s];
			knod_jit_dbg(" PTR_TO_STACK with BPF_X is not supported\n");
			return -EINVAL;
		case BPF_ALU | BPF_LSH | BPF_K:
		case BPF_ALU64 | BPF_LSH | BPF_K:
			//r[d] <<= imm;
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.dst_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			kdreg->stack_off = prev_meta->dreg.stack_off << imm;
			break;
		case BPF_ALU | BPF_RSH | BPF_X:
		case BPF_ALU64 | BPF_RSH | BPF_X:
			//r[d] >>= r[s];
			knod_jit_dbg(" PTR_TO_STACK with BPF_X is not supported\n");
			return -EINVAL;
		case BPF_ALU | BPF_RSH | BPF_K:
		case BPF_ALU64 | BPF_RSH | BPF_K:
			//r[d] >>= imm;
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.dst_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			kdreg->stack_off = prev_meta->dreg.stack_off >> imm;
			break;
		case BPF_ALU | BPF_ARSH | BPF_X:
		case BPF_ALU64 | BPF_ARSH | BPF_X:
			//r[d] >>= r[s];
			knod_jit_dbg(" PTR_TO_STACK with BPF_X is not supported\n");
			return -EINVAL;
		case BPF_ALU | BPF_ARSH | BPF_K:
		case BPF_ALU64 | BPF_ARSH | BPF_K:
			//r[d] >>= imm;
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.dst_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			kdreg->stack_off = prev_meta->dreg.stack_off >> imm;
			break;
		}
		knod_jit_dbg(" %d: dreg->stack_off = %d\n", meta->bpf_insn_idx,
			kdreg->stack_off);
	}

	if (dreg->type == PTR_TO_PACKET) {
		imm = meta->insn.imm;

		switch (meta->insn.code) {
		/* ALU
		 * If a destination register contains a pointer of STACK,
		 * offset should not be minus.
		 */
		case BPF_ALU | BPF_MOV | BPF_X:
		case BPF_ALU64 | BPF_MOV | BPF_X:
			//r[d] = r[s];
			kdreg->packet_off = ksreg->packet_off;
			break;
		case BPF_ALU | BPF_MOV | BPF_K:
		case BPF_ALU64 | BPF_MOV | BPF_K:
			//r[d] = imm;
			kdreg->packet_off = ksreg->packet_off;
			break;
		case BPF_ALU | BPF_XOR | BPF_X:
		case BPF_ALU64 | BPF_XOR | BPF_X:
			//r[d] ^= r[s];
			knod_jit_dbg(" PTR_TO_STACK with BPF_X is not supported\n");
			return -EINVAL;
		case BPF_ALU | BPF_XOR | BPF_K:
		case BPF_ALU64 | BPF_XOR | BPF_K:
			//r[d] ^= imm;
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.dst_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			kdreg->packet_off = prev_meta->dreg.packet_off ^ imm;
			break;
		case BPF_ALU | BPF_MOD | BPF_X:
		case BPF_ALU64 | BPF_MOD | BPF_X:
			//r[d] %= r[s];
			knod_jit_dbg(" PTR_TO_STACK with BPF_X is not supported\n");
			return -EINVAL;
		case BPF_ALU | BPF_MOD | BPF_K:
		case BPF_ALU64 | BPF_MOD | BPF_K:
			//r[d] %= imm;
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.dst_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			kdreg->packet_off = prev_meta->dreg.packet_off % imm;
			break;
		case BPF_ALU | BPF_AND | BPF_X:
		case BPF_ALU64 | BPF_AND | BPF_X:
			//r[d] &= r[s];
			knod_jit_dbg(" PTR_TO_STACK with BPF_X is not supported\n");
			return -EINVAL;
		case BPF_ALU | BPF_AND | BPF_K:
		case BPF_ALU64 | BPF_AND | BPF_K:
			//r[d] &= imm;
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.dst_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			kdreg->packet_off = prev_meta->dreg.packet_off & imm;
			break;
		case BPF_ALU | BPF_OR | BPF_X:
		case BPF_ALU64 | BPF_OR | BPF_X:
			//r[d] |= r[s];
			knod_jit_dbg(" PTR_TO_STACK with BPF_X is not supported\n");
			return -EINVAL;
		case BPF_ALU | BPF_OR | BPF_K:
		case BPF_ALU64 | BPF_OR | BPF_K:
			//r[d] |= imm;
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.dst_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			kdreg->packet_off = prev_meta->dreg.packet_off | imm;
			break;
		case BPF_ALU | BPF_ADD | BPF_X:
		case BPF_ALU64 | BPF_ADD | BPF_X:
			//r[d] += r[s];
			knod_jit_dbg(" PTR_TO_STACK with BPF_X is not supported\n");
			return -EINVAL;
		case BPF_ALU | BPF_ADD | BPF_K:
		case BPF_ALU64 | BPF_ADD | BPF_K:
			//r[d] += imm;
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.dst_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			kdreg->packet_off = prev_meta->dreg.packet_off + imm;
			break;
		case BPF_ALU | BPF_SUB | BPF_X:
		case BPF_ALU64 | BPF_SUB | BPF_X:
			//r[d] -= r[s];
			knod_jit_dbg(" PTR_TO_STACK with BPF_X is not supported\n");
			return -EINVAL;
		case BPF_ALU | BPF_SUB | BPF_K:
		case BPF_ALU64 | BPF_SUB | BPF_K:
			//r[d] -= imm;
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.dst_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			kdreg->packet_off = prev_meta->dreg.packet_off - imm;
			break;
		case BPF_ALU | BPF_MUL | BPF_X:
		case BPF_ALU64 | BPF_MUL | BPF_X:
			//r[d] *= r[s];
			knod_jit_dbg(" PTR_TO_STACK with BPF_X is not supported\n");
			return -EINVAL;
		case BPF_ALU | BPF_MUL | BPF_K:
		case BPF_ALU64 | BPF_MUL | BPF_K:
			//r[d] *= imm;
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.dst_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			kdreg->packet_off = prev_meta->dreg.packet_off * imm;
			break;
		case BPF_ALU | BPF_DIV | BPF_X:
		case BPF_ALU64 | BPF_DIV | BPF_X:
			//r[d] /= r[s];
			knod_jit_dbg(" PTR_TO_STACK with BPF_X is not supported\n");
			return -EINVAL;
		case BPF_ALU | BPF_DIV | BPF_K:
		case BPF_ALU64 | BPF_DIV | BPF_K:
			//r[d] /= imm;
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.dst_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			kdreg->packet_off = prev_meta->dreg.packet_off / imm;
			break;
		case BPF_ALU | BPF_NEG:
		case BPF_ALU64 | BPF_NEG:
			//r[d] = -r[d];
			break;
		case BPF_ALU | BPF_LSH | BPF_X:
		case BPF_ALU64 | BPF_LSH | BPF_X:
			//r[d] <<= r[s];
			knod_jit_dbg(" PTR_TO_STACK with BPF_X is not supported\n");
			return -EINVAL;
		case BPF_ALU | BPF_LSH | BPF_K:
		case BPF_ALU64 | BPF_LSH | BPF_K:
			//r[d] <<= imm;
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.dst_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			kdreg->packet_off = prev_meta->dreg.packet_off << imm;
			break;
		case BPF_ALU | BPF_RSH | BPF_X:
		case BPF_ALU64 | BPF_RSH | BPF_X:
			//r[d] >>= r[s];
			knod_jit_dbg(" PTR_TO_STACK with BPF_X is not supported\n");
			return -EINVAL;
		case BPF_ALU | BPF_RSH | BPF_K:
		case BPF_ALU64 | BPF_RSH | BPF_K:
			//r[d] >>= imm;
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.dst_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			kdreg->packet_off = prev_meta->dreg.packet_off >> imm;
			break;
		case BPF_ALU | BPF_ARSH | BPF_X:
		case BPF_ALU64 | BPF_ARSH | BPF_X:
			//r[d] >>= r[s];
			knod_jit_dbg(" PTR_TO_STACK with BPF_X is not supported\n");
			return -EINVAL;
		case BPF_ALU | BPF_ARSH | BPF_K:
		case BPF_ALU64 | BPF_ARSH | BPF_K:
			//r[d] >>= imm;
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(
				knod_prog, meta, meta->insn.dst_reg);
			if (!prev_meta) {
				knod_jit_dbg(" Invalid\n");
				return -EINVAL;
			}
			kdreg->packet_off = prev_meta->dreg.packet_off >> imm;
			break;
		}
		knod_jit_dbg(" %d: dreg->packet_off = %d\n", meta->bpf_insn_idx,
			kdreg->packet_off);
	}
	return 0;
}

static int knod_bpf_check_global_offset(unsigned int isa, unsigned int engine,
					 const struct bpf_insn *insn,
					 enum bpf_reg_type type)
{
	unsigned int cls = BPF_CLASS(insn->code);

	if (isa != 10 || engine != 1 || BPF_MODE(insn->code) != BPF_MEM ||
	    (cls != BPF_LDX && cls != BPF_ST && cls != BPF_STX) ||
	    (type != PTR_TO_PACKET && type != PTR_TO_MAP_VALUE))
		return 0;
	if (insn->off < -2048 || insn->off > 2047)
		return -EOPNOTSUPP;
	return 0;
}

static int knod_bpf_verify_insn(struct bpf_verifier_env *env,
				int insn_idx, int prev_insn)
{
	struct knod_prog *knod_prog = env->prog->aux->offload->dev_priv;
	const struct bpf_reg_state *sreg, *dreg, *kreg, *vreg;
	struct knod_insn_meta *meta = knod_prog->meta;
	struct knod_insn_meta *prev_meta;
	int err = 0;

	meta = knod_bpf_goto_meta(knod_prog, meta, insn_idx);
	sreg = cur_regs(env) + meta->insn.src_reg;
	dreg = cur_regs(env) + meta->insn.dst_reg;
	knod_prog->meta = meta;
	meta->sreg.reg = *sreg;
	meta->dreg.reg = *dreg;

	err = knod_bpf_update_ptr_off(knod_prog, meta, env);
	if (err)
		goto out;

	if (meta->insn.src_reg >= MAX_BPF_REG ||
			meta->insn.dst_reg >= MAX_BPF_REG) {
		knod_jit_dbg(" program uses extended registers - jit hardening?\n");
		err = -EINVAL;
		goto out;
	}

	err = knod_bpf_check_global_offset(knod_prog->knod->isa_version,
					  knod_bpf_jit_engine, &meta->insn,
					  BPF_CLASS(meta->insn.code) == BPF_LDX ?
					  sreg->type : dreg->type);
	if (err)
		goto out;

	if (is_mbpf_load(meta)) {
		err = knod_bpf_check_ptr(knod_prog, meta, env,
					 meta->insn.src_reg);
		goto out;
	}
	/* Immediate stores need the same pointer provenance as STX. */
	if (BPF_CLASS(meta->insn.code) == BPF_ST &&
	    BPF_MODE(meta->insn.code) == BPF_MEM &&
	    (dreg->type == PTR_TO_PACKET || dreg->type == PTR_TO_STACK)) {
		err = knod_bpf_check_ptr(knod_prog, meta, env,
					 meta->insn.dst_reg);
		goto out;
	}
	if (is_mbpf_store(meta)) {
		err = knod_bpf_check_store(knod_prog, meta, env);
		goto out;
	}

	if (is_mbpf_map_call(meta)) {
		kreg = cur_regs(env) + 2;
		if (kreg->type == PTR_TO_STACK) {
			err = knod_bpf_native_stack_arg(&meta->kreg, kreg,
						 env->cur_state->curframe);
			if (err)
				goto out;
		} else {
			if (meta->kreg.reg.type == PTR_TO_STACK) {
				err = -EOPNOTSUPP;
				goto out;
			}
			meta->kreg.reg = *kreg;
			prev_meta = knod_bpf_lookup_prev_meta_by_dreg(knod_prog,
							      meta, 2);
			if (!prev_meta) {
				err = -EINVAL;
				goto out;
			}
			if (base_type(kreg->type) == PTR_TO_PACKET)
				meta->kreg.packet_off = prev_meta->dreg.packet_off;
		}
		if (knod_prog->max_stack_off > meta->kreg.stack_off)
			knod_prog->max_stack_off = meta->kreg.stack_off;

		/* bpf_map_update_elem: track r3 (value pointer). */
		if (meta->insn.imm == 2) {
			vreg = cur_regs(env) + 3;
			if (vreg->type == PTR_TO_STACK) {
				err = knod_bpf_native_stack_arg(&meta->vreg, vreg,
							 env->cur_state->curframe);
				if (err)
					goto out;
			} else {
				if (meta->vreg.reg.type == PTR_TO_STACK) {
					err = -EOPNOTSUPP;
					goto out;
				}
				meta->vreg.reg = *vreg;
				prev_meta = knod_bpf_lookup_prev_meta_by_dreg(knod_prog,
								      meta, 3);
				if (!prev_meta) {
					err = -EINVAL;
					goto out;
				}
				if (base_type(vreg->type) == PTR_TO_PACKET)
					meta->vreg.packet_off = prev_meta->dreg.packet_off;
			}
			if (knod_prog->max_stack_off > meta->vreg.stack_off)
				knod_prog->max_stack_off = meta->vreg.stack_off;
		}
	}

	if (is_mbpf_alu(meta))
		err = knod_bpf_check_alu(knod_prog, meta, env);

	/* less stack offset is bigger */
	if (knod_prog->max_stack_off > meta->sreg.stack_off)
		knod_prog->max_stack_off = meta->sreg.stack_off;
	if (knod_prog->max_stack_off > meta->dreg.stack_off)
		knod_prog->max_stack_off = meta->dreg.stack_off;
	/* A load or store reaches insn.off past its register, so the register
	 * alone understates the depth by exactly that - which is why this was
	 * computed for years and read by nothing: it was never quite right.
	 */
	if (BPF_CLASS(meta->insn.code) == BPF_LDX &&
	    meta->ptr.type == PTR_TO_STACK &&
	    knod_prog->max_stack_off > meta->sreg.stack_off + meta->insn.off)
		knod_prog->max_stack_off = meta->sreg.stack_off + meta->insn.off;
	if ((BPF_CLASS(meta->insn.code) == BPF_STX ||
	     BPF_CLASS(meta->insn.code) == BPF_ST) &&
	    meta->ptr.type == PTR_TO_STACK &&
	    knod_prog->max_stack_off > meta->dreg.stack_off + meta->insn.off)
		knod_prog->max_stack_off = meta->dreg.stack_off + meta->insn.off;

out:
	if (err)
		pr_warn("knod_bpf: verifier rejected bpf insn %d (code 0x%02x off %d imm %d): %d\n",
			insn_idx, meta->insn.code, meta->insn.off,
			meta->insn.imm, err);
	return err;
}

static int knod_bpf_finalize(struct bpf_verifier_env *env)
{
	return 0;
}

static int knod_bpf_offload(struct knod_dev *knodev,
			    struct bpf_prog *prog, bool oldprog)
{
	struct knod_bpf_priv *priv = knodev->accel->xdp.priv;

	WARN(!!knod_dev_offloaded(knodev) != oldprog,
	     "bad offload state, expected offload %sto be active",
	     oldprog ? "" : "not ");

	WRITE_ONCE(priv->prog, prog);
	knod_dev_offload(knodev, prog);

	/*
	 * Uninstalling the prog: reload the pass kernel now, while the prog's
	 * maps are still valid, so the worker stops dispatching prog code that
	 * is about to reference freed maps.
	 */
	if (!prog)
		knod_bpf_reload_pass(knodev);

	return 0;
}

static int knod_bpf_xdp_offload_prog(struct knod_dev *knodev,
				     struct netdev_bpf *bpf)
{
	if (!knod_dev_active(knodev) && !bpf->prog)
		return 0;

	if (!knod_dev_active(knodev) && bpf->prog &&
	    knodev->accel->xdp.bpf_offloaded) {
		return -EBUSY;
	}

	return knod_bpf_offload(knodev, bpf->prog, knod_dev_active(knodev));
}

static int knod_bpf_xdp_set_prog(struct knod_dev *knodev,
				 struct netdev_bpf *bpf)
{
	int err;

	if (bpf->command == XDP_SETUP_PROG_HW) {
		err = knod_bpf_xdp_offload_prog(knodev, bpf);
		if (err)
			return err;
	}

	xdp_attachment_setup(&knodev->accel->xdp.xdp_hw, bpf);

	return 0;
}

/* Keep the memory accesses a map emitter just made out of the CU's own cache.
 *
 * A map that is not percpu is reached by every workgroup, and a workgroup is a
 * CU with a cache of its own that nothing invalidates until the dispatch ends.
 * One CU inserting into a hash table and another looking the same key up in the
 * same dispatch will not find it: the reader answers from a line it read before
 * the write.
 *
 * RDNA holds another cache between the two, shared by the CUs of a shader
 * array, so both have to be stepped past - GLC for the one in the CU, DLC for
 * the one in the array.  GCN has only the first and no bit for the second.
 *
 * Only loads.  A store already reaches L2 whatever the bits say (RDNA2 8.1.10),
 * and DLC on one means bypass L2 instead of missing a cache above it - the
 * write goes to memory and the next reader, looking in L2, does not see it.
 * On an atomic, GLC changes whether it returns anything at all.
 *
 * A percpu map does not need any of this - its instance belongs to one queue,
 * so one workgroup, and the system-scope fence at the dispatch boundary carries
 * it from there.  Leaving those in the cache is most of why they are quick.
 *
 * Letting the array cache answer instead of L2 - GLC without DLC, which the ISA
 * allows and calls coherent for stores - was tried and changed the rate by
 * nothing at all, so the weaker guarantee buys nothing and is not taken.
 */
static void knod_map_bypass_l0(struct knod_bpf_priv *priv,
			       struct knod_insn_meta *meta, u32 first)
{
	u32 i;

	for (i = first; i < meta->amdgpu_insns; i++) {
		struct amdgcn_insn *insn = &meta->amdgpu_insn[i];

		if (insn->type != AMDGCN_INSN_TYPE_FLAT)
			continue;
		if (priv->isa_version == 11) {
			if (insn->gfx11.flat.op >= GFX11_GLOBAL_STORE_B8)
				continue;
			insn->gfx11.flat.glc = 1;
			insn->gfx11.flat.dlc = 1;
		} else if (priv->isa_version == 10) {
			if (insn->gfx10.flat.op >= GFX10_GLOBAL_STORE_BYTE)
				continue;
			insn->gfx10.flat.glc = 1;
			insn->gfx10.flat.dlc = 1;
		} else if (priv->isa_version == 9) {
			if (insn->gfx9.flat.op >= GFX9_GLOBAL_STORE_BYTE)
				continue;
			insn->gfx9.flat.glc = 1;
		} else {
			WARN_ON_ONCE(1);
		}
	}
}

static void knod_wait_vmcnt(struct knod_bpf_priv *priv,
			   struct knod_insn_meta *meta)
{
	knod_emit(priv, meta, s_waitcnt_vmcnt);
}

static int knod_prog_prepare_insns(struct knod_bpf_priv *priv,
				   struct knod_prog *knod_prog)
{
	struct knod_insn_meta *meta;

	meta = kzalloc_obj(*meta, GFP_KERNEL);
	if (!meta)
		return -ENOMEM;

	meta->amdgpu_insn_idx = 0;

	meta->blob = knod_blob_find(&priv->blob, KNOD_BLOB_PROLOGUE, 0,
				    &meta->blob_size);
	if (!meta->blob) {
		WARN_ON_ONCE(1);
		kfree(meta);
		return -EOPNOTSUPP;
	}

	/* blob_at is zero, so what this emits lands after the routine - the LDS
	 * stack base, which the prologue itself does not set up.
	 */
	knod_bpf_emit_lds_base_init(priv, meta);
	list_add_tail(&meta->l, &knod_prog->pre_insns);
	return 0;
}

static int knod_prog_prepare(struct knod_bpf_priv *priv,
			     struct knod_prog *knod_prog,
			     const struct bpf_insn *prog,
			     unsigned int cnt)
{
	struct knod_insn_meta *meta;
	unsigned int i;

	knod_vset64(&r64[0], KNOD_AMDGPU_TMP_VREG0_LO);
	knod_vset64(&r64[1], KNOD_AMDGPU_TMP_VREG1_LO);
	knod_vset64(&r64[2], KNOD_AMDGPU_TMP_VREG2_LO);
	knod_vset64(&r64[3], KNOD_AMDGPU_TMP_VREG3_LO);
	knod_vset64(&r64[4], KNOD_AMDGPU_TMP_VREG4_LO);
	knod_vset64(&r64[5], KNOD_AMDGPU_TMP_VREG5_LO);
	knod_vset64(&r64[6], KNOD_AMDGPU_TMP_VREG6_LO);
	knod_vset64(&r64[7], KNOD_AMDGPU_TMP_VREG7_LO);
	knod_vset64(&r64[8], KNOD_AMDGPU_TMP_VREG8_LO);
	knod_vset64(&r64[9], KNOD_AMDGPU_TMP_VREG9_LO);
	knod_vset64(&r64[10], KNOD_AMDGPU_TMP_VREG10_LO);
	knod_vset64(&r64[11], KNOD_AMDGPU_TMP_VREG11_LO);
	knod_vset64(&r64[12], KNOD_AMDGPU_TMP_VREG12_LO);
	knod_vset64(&r64[13], KNOD_AMDGPU_TMP_VREG13_LO);
	knod_vset64(&r64[14], KNOD_AMDGPU_TMP_VREG14_LO);
	knod_vset64(&r64[15], KNOD_AMDGPU_TMP_VREG15_LO);
	knod_vset64(&r64[16], KNOD_AMDGPU_TMP_VREG16_LO);
	knod_vset64(&r64[17], KNOD_AMDGPU_TMP_VREG17_LO);
	knod_vset64(&r64[19], KNOD_AMDGPU_CTX_VREG_LO);

	knod_sset64(&sr64[0], KNOD_AMDGPU_TMP_SREG0_LO);
	knod_sset64(&sr64[1], KNOD_AMDGPU_TMP_SREG1_LO);
	knod_sset64(&sr64[2], KNOD_AMDGPU_TMP_SREG2_LO);
	knod_sset64(&sr64[3], KNOD_AMDGPU_TMP_SREG3_LO);
	knod_sset64(&sr64[4], KNOD_AMDGPU_TMP_SREG4_LO);
	knod_sset64(&sr64[5], KNOD_AMDGPU_TMP_SREG5_LO);

	knod_vset64(&bpf_reg64[0], KNOD_AMDGPU_VREG0_LO);
	knod_vset64(&bpf_reg64[1], KNOD_AMDGPU_VREG1_LO);
	knod_vset64(&bpf_reg64[2], KNOD_AMDGPU_VREG2_LO);
	knod_vset64(&bpf_reg64[3], KNOD_AMDGPU_VREG3_LO);
	knod_vset64(&bpf_reg64[4], KNOD_AMDGPU_VREG4_LO);
	knod_vset64(&bpf_reg64[5], KNOD_AMDGPU_VREG5_LO);
	knod_vset64(&bpf_reg64[6], KNOD_AMDGPU_VREG6_LO);
	knod_vset64(&bpf_reg64[7], KNOD_AMDGPU_VREG7_LO);
	knod_vset64(&bpf_reg64[8], KNOD_AMDGPU_VREG8_LO);
	knod_vset64(&bpf_reg64[9], KNOD_AMDGPU_VREG9_LO);
	knod_vset64(&bpf_reg64[10], KNOD_AMDGPU_FRAME_POINTER_VREG_LO);

	knod_vset32(&r32[0], KNOD_AMDGPU_TMP_VREG0_LO);
	for (i = 1; i < 36; i++)
		knod_vset32(&r32[i], r32[i - 1].v + 1);

	for (i = 0; i < cnt; i++) {
		meta = kzalloc_obj(*meta, GFP_KERNEL);
		if (!meta)
			return -ENOMEM;

		meta->insn = prog[i];
		meta->bpf_insn_idx = i;

		list_add_tail(&meta->l, &knod_prog->insns);
	}
	knod_prog->n_insns = cnt;

	return 0;
}

static void knod_prog_free(struct knod_prog *knod_prog)
{
	struct knod_insn_meta *meta, *tmp;

	//kfree(knod_prog->subprog);

	list_for_each_entry_safe(meta, tmp, &knod_prog->pre_insns, l) {
		list_del(&meta->l);
		kfree(meta);
	}
	list_for_each_entry_safe(meta, tmp, &knod_prog->insns, l) {
		list_del(&meta->l);
		kfree(meta);
	}
	list_for_each_entry_safe(meta, tmp, &knod_prog->post_insns, l) {
		list_del(&meta->l);
		kfree(meta);
	}
	kfree(knod_prog);
}

static int knod_bpf_verifier_prep(struct bpf_prog *prog)
{
	struct knod_prog *knod_prog;
	struct knod_bpf_priv *priv;
	int err;

	knod_prog = kzalloc_obj(struct knod_prog, GFP_KERNEL);
	if (!knod_prog)
		return -ENOMEM;

	INIT_LIST_HEAD(&knod_prog->insns);
	INIT_LIST_HEAD(&knod_prog->pre_insns);
	INIT_LIST_HEAD(&knod_prog->post_insns);
	prog->aux->offload->dev_priv = knod_prog;
	priv = bpf_offload_dev_priv(prog->aux->offload->offdev);
	knod_prog->knodev = priv->knodev;
	WRITE_ONCE(priv->knod_prog, knod_prog);
	knod_prog->knod = priv->knod;
	knod_prog->insn_idx = 0;

	knod_prog->done_mask_sreg = KNOD_AMDGPU_DONE_MASK_SREG;
	knod_prog->exec_save_base = KNOD_AMDGPU_EXEC_SAVE_SREG_BASE;
	knod_prog->initial_exec_sreg = KNOD_AMDGPU_INITIAL_EXEC_SREG;

	err = knod_prog_prepare(priv, knod_prog, prog->insnsi, prog->len);
	if (err)
		goto err_free;

	knod_prog->meta = knod_prog_first_meta(knod_prog);

	return 0;

err_free:
	knod_prog_free(knod_prog);

	return err;
}

static struct knod_insn_meta *knod_bpf_lookup_meta(struct knod_prog *knod_prog,
						   short idx)
{
	struct knod_insn_meta *meta;

	list_for_each_entry(meta, &knod_prog->insns, l) {
		if (meta->amdgpu_insn_idx == AMDGPU_INSN_SKIP)
			continue;
		if (meta->bpf_insn_idx == idx)
			return meta;
	}

	return NULL;
}

static void knod_mov64_imm(struct knod_bpf_priv *priv,
			  struct knod_insn_meta *meta,
			  int d, u64 imm64)
{
	struct amdgcn_param32 param[2];

	knod_vset32(&param[0], d);
	knod_iset32(&param[1], imm64 & ~0U);
	knod_emit(priv, meta, v_mov_b32_e32, param[0], param[1]);
	knod_vset32(&param[0], d + 1);
	knod_iset32(&param[1], imm64 >> 32);
	knod_emit(priv, meta, v_mov_b32_e32, param[0], param[1]);
}

static void knod_mov32(struct knod_bpf_priv *priv,
		      struct knod_insn_meta *meta,
		      struct amdgcn_param32 dst,
		      struct amdgcn_param32 src)
{
	knod_emit(priv, meta, v_mov_b32_e32, dst, src);
}

static void knod_mov64(struct knod_bpf_priv *priv,
		      struct knod_insn_meta *meta,
		      struct amdgcn_param64 dst,
		      struct amdgcn_param64 src)
{
	knod_mov32(priv, meta, dst.lo, src.lo);
	knod_mov32(priv, meta, dst.hi, src.hi);
}

static void knod_add64(struct knod_bpf_priv *priv,
		      struct knod_insn_meta *meta,
		      struct amdgcn_param64 dst,
		      struct amdgcn_param64 src0,
		      struct amdgcn_param64 src1)
{
	knod_emit(priv, meta, v_add_co_u32, dst.lo, src0.lo, src1.lo);
	knod_emit(priv, meta, v_add_co_ci_u32_e32, dst.hi, src0.hi,
		  src1.hi);
}

/* No carry out/in */
static void knod_add32(struct knod_bpf_priv *priv,
		      struct knod_insn_meta *meta,
		      struct amdgcn_param32 dst,
		      struct amdgcn_param32 src0,
		      struct amdgcn_param32 src1)
{
	knod_emit(priv, meta, v_add_u32, dst, src0, src1);
}

static void knod_xor32(struct knod_bpf_priv *priv,
		      struct knod_insn_meta *meta,
		      struct amdgcn_param32 dst,
		      struct amdgcn_param32 src0,
		      struct amdgcn_param32 src1)
{
	knod_emit(priv, meta, v_xor_b32_e32, dst, src0, src1);
}

static void knod_bfe32(struct knod_bpf_priv *priv,
			   struct knod_insn_meta *meta,
			   struct amdgcn_param32 dst,
			   struct amdgcn_param32 src0,
			   struct amdgcn_param32 src1,
			   struct amdgcn_param32 src2)
{
	knod_emit(priv, meta, v_bfe_u32, dst, src0, src1, src2);
}

static void knod_bfi32(struct knod_bpf_priv *priv,
			   struct knod_insn_meta *meta,
			   struct amdgcn_param32 dst,
			   struct amdgcn_param32 src0,
			   struct amdgcn_param32 src1,
			   struct amdgcn_param32 src2)
{
	knod_emit(priv, meta, v_bfi_b32, dst, src0, src1, src2);
}

static void knod_lshrrev32(struct knod_bpf_priv *priv,
			   struct knod_insn_meta *meta,
			   struct amdgcn_param32 dst,
			   struct amdgcn_param32 src0,
			   struct amdgcn_param32 src1)
{
	knod_emit(priv, meta, v_lshrrev_b32, dst, src0, src1);
}

static void knod_lshrrev64(struct knod_bpf_priv *priv,
			   struct knod_insn_meta *meta,
			   struct amdgcn_param64 dst,
			   struct amdgcn_param64 src0,
			   struct amdgcn_param64 src1)
{
	knod_emit(priv, meta, v_lshrrev_b64, dst, src0, src1);
}

static void knod_ashrrev32(struct knod_bpf_priv *priv,
			   struct knod_insn_meta *meta,
			   struct amdgcn_param32 dst,
			   struct amdgcn_param32 src0,
			   struct amdgcn_param32 src1)
{
	knod_emit(priv, meta, v_ashrrev_i32, dst, src0, src1);
}

static void knod_ashrrev64(struct knod_bpf_priv *priv,
			   struct knod_insn_meta *meta,
			   struct amdgcn_param64 dst,
			   struct amdgcn_param64 src0,
			   struct amdgcn_param64 src1)
{
	knod_emit(priv, meta, v_ashrrev_i64, dst, src0, src1);
}

static void knod_lshlrev32(struct knod_bpf_priv *priv,
			   struct knod_insn_meta *meta,
			   struct amdgcn_param32 dst,
			   struct amdgcn_param32 src0,
			   struct amdgcn_param32 src1)
{
	knod_emit(priv, meta, v_lshlrev_b32, dst, src0, src1);
}

static void knod_lshlrev64(struct knod_bpf_priv *priv,
			   struct knod_insn_meta *meta,
			   struct amdgcn_param64 dst,
			   struct amdgcn_param64 src0,
			   struct amdgcn_param64 src1)
{
	knod_emit(priv, meta, v_lshlrev_b64, dst, src0, src1);
}

/* No carry out/in */
static void knod_sub32(struct knod_bpf_priv *priv,
		      struct knod_insn_meta *meta,
		      struct amdgcn_param32 dst,
		      struct amdgcn_param32 src0,
		      struct amdgcn_param32 src1)
{
	knod_emit(priv, meta, v_sub_u32, dst, src0, src1);
}

static void knod_and32(struct knod_bpf_priv *priv,
		      struct knod_insn_meta *meta,
		      struct amdgcn_param32 dst,
		      struct amdgcn_param32 src0,
		      struct amdgcn_param32 src1)
{
	knod_emit(priv, meta, v_and_b32_e32, dst, src0, src1);
}

static void knod_and64(struct knod_bpf_priv *priv,
		      struct knod_insn_meta *meta,
		      struct amdgcn_param64 dst,
		      struct amdgcn_param64 src0,
		      struct amdgcn_param64 src1)
{
	knod_and32(priv, meta, dst.lo, src0.lo, src1.lo);
	knod_and32(priv, meta, dst.hi, src0.hi, src1.hi);
}

static void knod_or32(struct knod_bpf_priv *priv,
		      struct knod_insn_meta *meta,
		      struct amdgcn_param32 dst,
		      struct amdgcn_param32 src0,
		      struct amdgcn_param32 src1)
{
	knod_emit(priv, meta, v_or_b32_e32, dst, src0, src1);
}

static void knod_sub64(struct knod_bpf_priv *priv,
		      struct knod_insn_meta *meta,
		      struct amdgcn_param64 dst,
		      struct amdgcn_param64 src0,
		      struct amdgcn_param64 src1)
{
	knod_emit(priv, meta, v_sub_co_u32, dst.lo, src0.lo, src1.lo);
	knod_emit(priv, meta, v_sub_co_ci_u32_e32, dst.hi, src0.hi,
		  src1.hi);
}

static void knod_mul_lo32(struct knod_bpf_priv *priv,
			 struct knod_insn_meta *meta,
			 struct amdgcn_param32 dst,
			 struct amdgcn_param32 src1,
			 struct amdgcn_param32 src2)
{
	knod_emit(priv, meta, v_mul_lo_u32, dst, src1, src2);
}

static void knod_mul_hi32(struct knod_bpf_priv *priv,
			 struct knod_insn_meta *meta,
			 struct amdgcn_param32 dst,
			 struct amdgcn_param32 src1,
			 struct amdgcn_param32 src2)
{
	knod_emit(priv, meta, v_mul_hi_u32, dst, src1, src2);
}

static void knod_mul64(struct knod_bpf_priv *priv,
		      struct knod_insn_meta *meta,
		      struct amdgcn_param64 dst,
		      struct amdgcn_param64 src1,
		      struct amdgcn_param64 src2,
		      struct amdgcn_param64 tmp)
{
	/*
	 * v_mul_lo_u32 v1, v2, v1
	 * v_mul_hi_u32 v5, v2, v0
	 * v_mul_lo_u32 v3, v3, v0
	 * v_mul_lo_u32 v0, v2, v0
	 * v_add_u32_e32 v1, v5, v1
	 * v_add_u32_e32 v1, v1, v3
	 *
	 * v[0:1] = src1, dst
	 * v[2:3] = src2
	 * v5 = tmp
	 */

	/* v_mul_lo_u32 v1, v2, v1 */
	knod_mul_lo32(priv, meta, src2.hi, src1.lo, src2.hi);
	/* v_mul_hi_u32 v5, v2, v0 */
	knod_mul_hi32(priv, meta, tmp.lo, src1.lo, src2.lo);
	/* v_mul_lo_u32 v3, v3, v0 */
	knod_mul_lo32(priv, meta, src1.hi, src1.hi, src2.lo);
	/* v_mul_lo_u32 v0, v2, v0 */
	knod_mul_lo32(priv, meta, src1.lo, src1.lo, src2.lo);
	/* v_add_u32_e32 v1, v5, v1 */
	knod_add32(priv, meta, src2.lo, tmp.lo, src2.hi);
	/* v_add_u32_e32 v1, v1, v3 */
	knod_add32(priv, meta, src1.hi, src2.lo, src1.hi);
	knod_mov32(priv, meta, dst.lo, src1.lo);
	knod_mov32(priv, meta, dst.hi, src1.hi);
}

static void knod_div(struct knod_bpf_priv *priv,
		    struct knod_insn_meta *meta,
		    struct amdgcn_param64 dst,
		    struct amdgcn_param64 imm,
		    struct amdgcn_param64 tmp_reg0,
		    struct amdgcn_param64 tmp_reg1,
		    struct amdgcn_param64 tmp_reg2,
		    struct amdgcn_param64 tmp_reg3)
{
	struct reciprocal_value_adv rvalue;
	struct amdgcn_param64 p64[4];
	u8 pre_shift, exp;

	WARN_ON((imm.lo.type != AMDGCN_PARAM_TYPE_INTEGER_0) &&
		     (imm.lo.type != AMDGCN_PARAM_TYPE_LITERAL_CONST));
	WARN_ON((imm.hi.type != AMDGCN_PARAM_TYPE_INTEGER_0) &&
		     (imm.hi.type != AMDGCN_PARAM_TYPE_LITERAL_CONST));
	knod_iset64(&p64[0], 0);
	knod_iset64(&p64[1], 0);
	knod_iset64(&p64[2], 0);
	knod_iset64(&p64[3], 0);
	/*
	 * dst := imm
	 * n := dst_reg
	 */
	if (imm.imm > U32_MAX) {
		knod_mov64(priv, meta, dst, p64[0]);
		return;
	}

	if (imm.imm >= 1U << 31) {
		/* result = n >= dst; */
		knod_mov64(priv, meta, tmp_reg0, imm);
		knod_emit(priv, meta, v_cmp_ge_u64, dst, tmp_reg0);
		return;
	}

	rvalue = reciprocal_value_adv(imm.lo.v, 32);
	exp = rvalue.exp;
	if (rvalue.is_wide_m && !(imm.lo.v & 1)) {
		pre_shift = fls(imm.lo.v & -imm.lo.v) - 1;
		rvalue = reciprocal_value_adv(imm.lo.v >> pre_shift,
					      32 - pre_shift);
	} else {
		pre_shift = 0;
	}

	if (imm.lo.v == 1U << exp) {
		knod_iset64(&p64[0], exp);
		/* n = n >> exp */
		knod_lshrrev64(priv, meta, dst, p64[0], dst);
		return;
	} else if (rvalue.is_wide_m) {
		/*
		 * pre_shift must be zero when reached here.
		 * t = (n * rvalue.m) >> 32;
		 * result = n - t;
		 * result >>= 1;
		 * result += t;
		 * result >>= rvalue.sh - 1;
		 */

		/*
		 * n := VREG0
		 * t := VREG1
		 * rvalue.m := VREG2
		 * tmp := VREG3
		 */

		/* n := TMP_VREG0 */
		knod_mov64(priv, meta, tmp_reg0, dst);

		knod_iset64(&p64[0], rvalue.m);
		/* rvalue.m := TMP_VREG2 */
		knod_mov64(priv, meta, tmp_reg2, p64[0]);

		/* t = n * rvalue.m; */
		knod_mul64(priv, meta,
			   tmp_reg1, /* t */
			   tmp_reg0, /* n */
			   tmp_reg2, /* rvalue.m */
			   tmp_reg3); /* tmp */

		/* t >>= 32; */
		knod_iset64(&p64[0], 0);
		knod_mov32(priv, meta, tmp_reg1.lo, tmp_reg1.hi);
		knod_mov32(priv, meta, tmp_reg1.hi, p64[0].lo);

		/* result = n - t */
		knod_sub64(priv, meta, dst, dst, tmp_reg1);

		/* result >>= 1 */
		knod_iset64(&p64[0], 1);
		knod_lshrrev64(priv, meta, dst, p64[0], dst);

		/* result += t; */
		knod_add64(priv, meta,
			   dst,
			   dst, /* result */
			   tmp_reg1); /* t */

		/* result >>= rvalue.sh - 1; */
		knod_iset64(&p64[0], rvalue.sh - 1);
		WARN_ON(rvalue.sh - 1 > 31);
		knod_lshrrev64(priv, meta, dst, p64[0], dst);
		return;
	}

	/*
	 * if (pre_shift)
	 *   result = n >> pre_shift;
	 * result = ((u64)result * rvalue.m) >> 32;
	 * result >>= rvalue.sh;
	 */

	/*
	 * n := VREG0
	 * <NONE> := VREG1
	 * rvalue.m := VREG2
	 * tmp := VREG3
	 * result := dst * 2
	 */

	/* n := TMP_VREG0 */
	knod_mov64(priv, meta, tmp_reg0, dst);

	/* rvalue.m := TMP_VREG2 */
	knod_iset64(&p64[0], rvalue.m);
	knod_mov64(priv, meta, tmp_reg2, p64[0]);

	if (pre_shift) {
		/* result = n >> pre_shift; */
		knod_iset64(&p64[0], pre_shift);
		knod_lshrrev64(priv, meta, dst, p64[0],
			       tmp_reg0); /* n */
	} else {
		/* tmp = 0 */
		knod_iset64(&p64[0], 0);
		knod_mov64(priv, meta, tmp_reg0, p64[0]);
	}

	/* result = result * rvalue.m; */
	knod_mul64(priv, meta,
		   dst, /* result */
		   dst, /* result */
		   tmp_reg2, /* rvalue.m */
		   tmp_reg3); /* tmp */

	/* result >>= (32 + rvalue.sh); */
	knod_iset64(&p64[0], 32 + rvalue.sh);
	knod_lshrrev64(priv, meta, dst, p64[0], dst);
}

static void knod_mod(struct knod_bpf_priv *priv,
		    struct knod_insn_meta *meta,
		    struct amdgcn_param64 dst,
		    struct amdgcn_param64 imm,
		    struct amdgcn_param64 tmp_reg0,
		    struct amdgcn_param64 tmp_reg1,
		    struct amdgcn_param64 tmp_reg2,
		    struct amdgcn_param64 tmp_reg3,
		    struct amdgcn_param64 tmp_reg4)
{
	WARN_ON((imm.lo.type != AMDGCN_PARAM_TYPE_INTEGER_0) &&
		     (imm.lo.type != AMDGCN_PARAM_TYPE_LITERAL_CONST));
	WARN_ON((imm.hi.type != AMDGCN_PARAM_TYPE_INTEGER_0) &&
		     (imm.hi.type != AMDGCN_PARAM_TYPE_LITERAL_CONST));
	/* q := tmp_reg0 */
	knod_mov64(priv, meta, tmp_reg0, dst);
	/* q = n / imm */
	knod_div(priv, meta, tmp_reg0, imm,
		     tmp_reg1, tmp_reg2, tmp_reg3, tmp_reg4);

	/* tmp_reg1 := imm_reg */
	knod_mov64(priv, meta, tmp_reg1, imm);

	/* imm * q := tmp_reg3 */
	knod_mul64(priv, meta,
		   tmp_reg3, /* imm * q */
		   tmp_reg0, /* q */
		   tmp_reg1, /* imm_reg */
		   tmp_reg2); /* tmp */

	knod_sub64(priv, meta, dst, dst, tmp_reg3);
}

/*
 * Fast constant modulo on the 32-bit value in @dst.lo for divisors of a
 * special form, avoiding knod_mod's reciprocal divide + 64-bit multiply:
 *   2^k     -> dst & (2^k-1)                     (mask)
 *   2^k + 1 -> lo - hi (+C if lo<hi)             (Fermat: 2^k = -1 mod C)
 *   2^k - 1 -> lo + hi (-C while >=C)            (Mersenne: 2^k = 1 mod C)
 * lo/hi are the low/high k-bit halves.  One fold is exact for a 32-bit
 * dividend when 2^k covers the high half (true for e.g. 65537 = 2^16+1,
 * kondor's per-packet `hash % RING_SIZE`).  Returns false for other
 * divisors (caller falls back to knod_mod).  Scratch: r64[0], r64[1].
 */
static bool knod_mod_k32(struct knod_bpf_priv *priv,
			 struct knod_insn_meta *meta,
			 struct amdgcn_param64 dst, u32 imm)
{
	struct amdgcn_param32 p;
	int i;

	if (is_power_of_2(imm)) {
		knod_iset32(&p, imm - 1);
		knod_and32(priv, meta, dst.lo, p, dst.lo);
	} else if (is_power_of_2(imm - 1) && (imm - 1) >= (1u << 16)) {
		knod_iset32(&p, imm - 2);			/* mask 2^k-1 */
		knod_and32(priv, meta, r64[0].lo, p, dst.lo);	/* lo */
		knod_iset32(&p, ilog2(imm - 1));		/* k */
		knod_emit(priv, meta, v_lshrrev_b32, r64[1].lo, p, dst.lo);
		/* lo-hi */
		knod_sub32(priv, meta, dst.lo, r64[0].lo, r64[1].lo);
		knod_emit(priv, meta, v_cmp_lt_u32, r64[0].lo, r64[1].lo);
		knod_iset32(&p, imm);
		knod_add32(priv, meta, r64[1].lo, p, dst.lo);	/* +C */
		knod_emit(priv, meta, v_cndmask_b32_e32, dst.lo, dst.lo,
			  r64[1].lo);
	} else if (is_power_of_2(imm + 1) && (imm + 1) >= (1u << 16)) {
		knod_iset32(&p, imm);				/* mask 2^k-1 */
		knod_and32(priv, meta, r64[0].lo, p, dst.lo);	/* lo */
		knod_iset32(&p, ilog2(imm + 1));		/* k */
		knod_emit(priv, meta, v_lshrrev_b32, r64[1].lo, p, dst.lo);
		/* lo+hi */
		knod_add32(priv, meta, dst.lo, r64[0].lo, r64[1].lo);
		knod_iset32(&p, imm);
		knod_mov32(priv, meta, r64[0].lo, p);		/* C in VGPR */
		for (i = 0; i < 2; i++) {			/* r < 2C */
			knod_emit(priv, meta, v_cmp_le_u32, r64[0].lo, dst.lo);
			knod_sub32(priv, meta, r64[1].lo, dst.lo, r64[0].lo);
			knod_emit(priv, meta, v_cndmask_b32_e32, dst.lo,
				  dst.lo, r64[1].lo);
		}
	} else {
		return false;
	}

	knod_iset32(&p, 0);
	knod_mov32(priv, meta, dst.hi, p);
	return true;
}

static u64 knod_bpf_map_gaddr(struct knod_bpf_priv *priv, int id)
{
	struct knod_dev *knodev = priv->knodev;
	struct knod_bpf_map *knod_map;
	struct knod_mem *mem;

	mutex_lock(&knodev->lock);
	list_for_each_entry(knod_map, &knodev->accel->xdp.bound_maps, list) {
		if (knod_map->offmap->map.id == id) {
			mem = knod_map->mem;
			mutex_unlock(&knodev->lock);
			return (u64)mem->gaddr;
		}
	}
	mutex_unlock(&knodev->lock);

	return 0;
}

static struct knod_bpf_map *knod_bpf_map_find(struct knod_bpf_priv *priv,
					      int id)
{
	struct knod_dev *knodev = priv->knodev;
	struct knod_bpf_map *knod_map;

	mutex_lock(&knodev->lock);
	list_for_each_entry(knod_map, &knodev->accel->xdp.bound_maps, list) {
		if (knod_map->offmap->map.id == id) {
			mutex_unlock(&knodev->lock);
			return knod_map;
		}
	}
	mutex_unlock(&knodev->lock);

	return NULL;
}

static u64 knod_bpf_get_map_gaddr(struct knod_bpf_priv *priv,
				  struct knod_insn_meta *meta1,
				  struct knod_insn_meta *meta2)
{
	struct bpf_map *map;

	map = (void *)(unsigned long)((u32)meta1->insn.imm |
			(u64)meta2->insn.imm << 32);

	return knod_bpf_map_gaddr(priv, map->id);
}

static int knod_bpf_get_map_id(struct knod_bpf_priv *priv,
			       struct knod_insn_meta *meta1,
			       struct knod_insn_meta *meta2)
{
	struct bpf_map *map;

	map = (void *)(unsigned long)((u32)meta1->insn.imm |
			(u64)meta2->insn.imm << 32);

	return map->id;
}

/* Derive immutable RX bounds from the pre-adjust data pointer and queue geometry.
 * Do not change PAGE_BASE: the epilogue needs it to return a physical offset.
 * The queue dword was padding in the blob ABI and is ignored by the blob.
 * Geometry zero preserves the legacy page bounds for other RX providers.
 * Uses s16/s18:s19 and v28-v33; callers keep their saved data and predicates in v22-v27.
 */
static void knod_bpf_packet_bound(struct knod_bpf_priv *priv,
				  struct knod_insn_meta *meta,
				  struct amdgcn_param64 dst, bool end)
{
	struct amdgcn_param64 addr, param, page;
	struct amdgcn_param32 geometry, off, stride, extra, queue, imm;
	struct amdgcn_param32 soff, scalar_geometry, source;

	knod_vset64(&addr, 28);
	knod_sset64(&param, KNOD_AMDGPU_PARAM_SREG_LO);
	knod_vset64(&page, KNOD_AMDGPU_PAGE_BASE_VREG_LO);
	knod_vset32(&geometry, 30);
	knod_vset32(&off, 31);
	knod_vset32(&stride, 32);
	knod_vset32(&extra, 33);
	knod_sset32(&queue, KNOD_BLOB_PRO_WG_Y_SREG);
	knod_sset32(&soff, 16);
	knod_sset32(&scalar_geometry, 18);
	knod_vset32(&source, end ? KNOD_AMDGPU_DATA_VREG_LO :
				 KNOD_AMDGPU_TMP_VREG0_LO);

	/* Geometry is uniform within this workgroup. No packet VMEM load is
	 * needed: the caller saved the pre-adjust pointer before changing it.
	 */
	knod_iset32(&imm, 5);
	knod_emit(priv, meta, s_lshl_b32, soff, queue, imm);
	knod_emit(priv, meta, s_load_dwordx2_soff, scalar_geometry, param.lo,
		  offsetof(struct knod_bpf_param, queues) +
		  offsetof(struct knod_bpf_queue_desc, rx_geometry), 16);
	knod_emit(priv, meta, s_waitcnt_lgkmcnt);
	knod_mov32(priv, meta, geometry, scalar_geometry);
	knod_emit(priv, meta, v_sub_co_u32, off, source, page.lo);

	knod_iset32(&imm, 16);
	knod_emit(priv, meta, v_lshrrev_b32, stride, imm, geometry);
	knod_iset32(&imm, 0);
	knod_emit(priv, meta, v_cmp_eq_u32, imm, geometry);
	knod_iset32(&imm, PAGE_SIZE);
	knod_mov32(priv, meta, extra, imm);
	knod_emit(priv, meta, v_cndmask_b32_e32, stride, stride, extra);
	/* The current data pointer remains inside its original RX slot. */
	knod_iset32(&imm, 0xffff);
	knod_emit(priv, meta, v_and_b32_e32, off, imm, off);
	knod_emit(priv, meta, v_and_b32_e32, extra, imm, geometry);
	if (end) {
		knod_iset32(&imm, SKB_DATA_ALIGN(sizeof(struct skb_shared_info)));
		knod_mov32(priv, meta, extra, imm);
		knod_iset32(&imm, 0);
		knod_emit(priv, meta, v_cmp_eq_u32, imm, geometry);
		knod_mov32(priv, meta, addr.lo, imm);
		knod_emit(priv, meta, v_cndmask_b32_e32, extra, extra, addr.lo);
		knod_emit(priv, meta, v_sub_co_u32, extra, stride, extra);
	}
	/* Power-of-two slot size was validated when the RX queue opened. */
	knod_iset32(&imm, 0);
	knod_emit(priv, meta, v_sub_co_u32, stride, imm, stride);
	knod_emit(priv, meta, v_and_b32_e32, off, stride, off);
	knod_emit(priv, meta, v_add_co_u32, off, extra, off);
	knod_emit(priv, meta, v_add_co_u32, dst.lo, off, page.lo);
	knod_emit(priv, meta, v_add_co_ci_u32_e32, dst.hi, imm, page.hi);
}

/*
 * knod_bpf_xdp_adjust_head - JIT bpf_xdp_adjust_head (helper 44).
 *
 * R2 = delta (signed 32-bit).  Adjusts DATA_VREG by delta.
 * Bounds: slot start + RX padding <= DATA_VREG <= DATA_END_VREG - ETH_HLEN.
 * Each bound is checked with its own VOPC, but VCC is captured into
 * VGPRs via v_cndmask (VALU) rather than SGPRs via s_mov_b64 (SALU).
 * VALU reads VCC correctly after VOPC; only SALU suffers the GFX10
 * dual-VOPC stale-read hazard.
 * page_base is reloaded on demand from param + spsc_bd.
 * On failure, DATA_VREG is restored and R0 = -EINVAL.
 * On success, R0 = 0.
 *
 * Clobbers: TMP_VREG0 (v22:v23), TMP_VREG1 (v24:v25), TMP_VREG2 (v26:v27),
 *           TMP_SREG0 (s16), TMP_SREG2 (s20:s21).
 */
static void knod_bpf_xdp_adjust_head(struct knod_bpf_priv *priv,
				    struct knod_insn_meta *meta)
{
	struct amdgcn_param32 ub_lo, ub_hi, dend_lo, dend_hi, sext_dst;
	struct amdgcn_param32 shift_amt;
	struct amdgcn_param32 tmp0_lo, tmp0_hi, data_lo, data_hi, fail_lo;
	struct amdgcn_param32 fail_hi;
	struct amdgcn_param64 data_vreg, pbase_vreg, ub;
	struct amdgcn_param32 r0_lo, r0_hi, imm, delta;

	knod_vset64(&data_vreg, KNOD_AMDGPU_DATA_VREG_LO);

	knod_vset32(&tmp0_lo, KNOD_AMDGPU_TMP_VREG0_LO);
	knod_vset32(&tmp0_hi, KNOD_AMDGPU_TMP_VREG0_HI);
	knod_vset32(&data_lo, KNOD_AMDGPU_DATA_VREG_LO);
	knod_vset32(&data_hi, KNOD_AMDGPU_DATA_VREG_HI);
	knod_vset32(&r0_lo, KNOD_AMDGPU_VREG0_LO);
	knod_vset32(&r0_hi, KNOD_AMDGPU_VREG0_HI);
	knod_vset32(&delta, bpf_reg64[2].lo.v);
	knod_vset32(&fail_lo, KNOD_AMDGPU_TMP_VREG2_LO);
	knod_vset32(&fail_hi, KNOD_AMDGPU_TMP_VREG2_HI);

	/* 1. Save original DATA_VREG -> TMP_VREG0 */
	knod_mov32(priv, meta, tmp0_lo, data_lo);
	knod_mov32(priv, meta, tmp0_hi, data_hi);

	/* 2. DATA_VREG += delta (R2.lo, sign-extended to 64-bit) */
	knod_emit(priv, meta, v_add_co_u32, data_lo, delta, data_lo);

	knod_vset32(&sext_dst, KNOD_AMDGPU_TMP_VREG1_LO);
	knod_iset32(&shift_amt, 31);
	knod_emit(priv, meta, v_ashrrev_i32, sext_dst, shift_amt, delta);

	knod_emit(priv, meta, v_add_co_ci_u32_e32, data_hi, sext_dst,
		  data_hi);

	/* 3. Lower bound: DATA_VREG < page_base -> VCC = fail */
	knod_vset64(&pbase_vreg, KNOD_AMDGPU_TMP_VREG1_LO);
	knod_bpf_packet_bound(priv, meta, pbase_vreg, false);
	knod_emit(priv, meta, v_cmp_lt_u64, data_vreg, pbase_vreg);

	/*
	 * Capture VCC -> VGPR via v_cndmask (VALU reads VCC correctly,
	 * unlike SALU which suffers the dual-VOPC stale-read hazard).
	 */
	knod_iset32(&imm, 1);
	knod_mov32(priv, meta, fail_hi, imm);
	knod_iset32(&imm, 0);
	knod_emit(priv, meta, v_cndmask_b32_e32, fail_lo, imm, fail_hi);

	/* 5. Upper bound: DATA_VREG > DATA_END_VREG - ETH_HLEN */
	knod_vset32(&ub_lo, KNOD_AMDGPU_TMP_VREG1_LO);
	knod_vset32(&ub_hi, KNOD_AMDGPU_TMP_VREG1_HI);
	knod_vset32(&dend_lo, KNOD_AMDGPU_DATA_END_VREG_LO);
	knod_vset32(&dend_hi, KNOD_AMDGPU_DATA_END_VREG_HI);

	knod_iset32(&imm, ETH_HLEN);
	/*
	 * v_sub_co_u32 is VOP2 on GFX9, whose vsrc1 must be a VGPR (a literal
	 * there reads v0). Subtraction is not commutative, so materialise
	 * ETH_HLEN into a scratch VGPR (ub_hi, overwritten by the high half
	 * below) and use it as src1 instead of an immediate.
	 */
	knod_mov32(priv, meta, ub_hi, imm);
	knod_emit(priv, meta, v_sub_co_u32, ub_lo, dend_lo, ub_hi);
	knod_iset32(&imm, 0);
	knod_mov32(priv, meta, delta, imm);
	knod_emit(priv, meta, v_sub_co_ci_u32_e32, ub_hi, dend_hi, delta);

	knod_vset64(&ub, KNOD_AMDGPU_TMP_VREG1_LO);
	knod_emit(priv, meta, v_cmp_gt_u64, data_vreg, ub);

	/* Capture upper_fail via v_cndmask, combine, convert to VCC */
	knod_iset32(&imm, 0);
	knod_emit(priv, meta, v_cndmask_b32_e32, fail_hi, imm, fail_hi);

	knod_emit(priv, meta, v_or_b32_e32, fail_lo, fail_lo, fail_hi);

	knod_emit(priv, meta, v_cmp_lt_u32, imm, fail_lo);

	/* 6. Conditional restore: VCC=1(fail) -> original,
	 *    VCC=0(pass) -> adjusted
	 */
	knod_emit(priv, meta, v_cndmask_b32_e32, data_lo, data_lo,
		  tmp0_lo);
	knod_emit(priv, meta, v_cndmask_b32_e32, data_hi, data_hi,
		  tmp0_hi);

	/* 7. R0 = VCC ? -EINVAL : 0 */
	knod_iset32(&imm, -EINVAL);
	knod_mov32(priv, meta, tmp0_lo, imm);
	knod_iset32(&imm, 0);
	knod_emit(priv, meta, v_cndmask_b32_e32, r0_lo, imm, tmp0_lo);

	knod_iset32(&imm, -1);
	knod_mov32(priv, meta, tmp0_hi, imm);
	knod_iset32(&imm, 0);
	knod_emit(priv, meta, v_cndmask_b32_e32, r0_hi, imm, tmp0_hi);
}

/*
 * knod_bpf_xdp_adjust_tail - JIT bpf_xdp_adjust_tail (helper 65).
 *
 * R2 = delta (signed 32-bit).  Adjusts DATA_END_VREG by delta.
 * Bounds: DATA_VREG + ETH_HLEN <= DATA_END_VREG <= usable slot end.
 * Each bound is checked with its own VOPC, but VCC is captured into
 * VGPRs via v_cndmask (VALU) rather than SGPRs via s_mov_b64 (SALU).
 * VALU reads VCC correctly after VOPC; only SALU suffers the GFX10
 * dual-VOPC stale-read hazard.
 * page_base is reloaded on demand from param + spsc_bd.
 * On failure, DATA_END_VREG is restored and R0 = -EINVAL.
 * On success, R0 = 0.
 *
 * Clobbers: TMP_VREG0 (v22:v23), TMP_VREG1 (v24:v25), TMP_VREG2 (v26:v27),
 *           TMP_SREG0 (s16), TMP_SREG2 (s20:s21).
 */
static void knod_bpf_xdp_adjust_tail(struct knod_bpf_priv *priv,
				    struct knod_insn_meta *meta)
{
	struct amdgcn_param32 tmp0_lo, tmp0_hi, dend_lo, dend_hi, fail_lo;
	struct amdgcn_param32 fail_hi;
	struct amdgcn_param32 lb_lo, lb_hi, d_lo, d_hi, sext_dst, shift_amt;
	struct amdgcn_param32 r0_lo, r0_hi;
	struct amdgcn_param32 imm, delta;
	struct amdgcn_param64 dend_vreg, lb;

	knod_vset32(&tmp0_lo, KNOD_AMDGPU_TMP_VREG0_LO);
	knod_vset32(&tmp0_hi, KNOD_AMDGPU_TMP_VREG0_HI);
	knod_vset32(&dend_lo, KNOD_AMDGPU_DATA_END_VREG_LO);
	knod_vset32(&dend_hi, KNOD_AMDGPU_DATA_END_VREG_HI);
	knod_vset32(&r0_lo, KNOD_AMDGPU_VREG0_LO);
	knod_vset32(&r0_hi, KNOD_AMDGPU_VREG0_HI);
	knod_vset32(&delta, bpf_reg64[2].lo.v);
	knod_vset32(&fail_lo, KNOD_AMDGPU_TMP_VREG2_LO);
	knod_vset32(&fail_hi, KNOD_AMDGPU_TMP_VREG2_HI);
	knod_vset64(&dend_vreg, KNOD_AMDGPU_DATA_END_VREG_LO);

	/* 1. Save original DATA_END_VREG -> TMP_VREG0 */
	knod_mov32(priv, meta, tmp0_lo, dend_lo);
	knod_mov32(priv, meta, tmp0_hi, dend_hi);

	/* 2. DATA_END_VREG += delta (R2.lo, sign-extended) */
	knod_emit(priv, meta, v_add_co_u32, dend_lo, delta, dend_lo);

	knod_vset32(&sext_dst, KNOD_AMDGPU_TMP_VREG1_LO);
	knod_iset32(&shift_amt, 31);
	knod_emit(priv, meta, v_ashrrev_i32, sext_dst, shift_amt, delta);

	knod_emit(priv, meta, v_add_co_ci_u32_e32, dend_hi, sext_dst,
		  dend_hi);

	/* 3. Lower bound: lb = DATA + ETH_HLEN -> TMP_VREG1 */
	knod_vset32(&lb_lo, KNOD_AMDGPU_TMP_VREG1_LO);
	knod_vset32(&lb_hi, KNOD_AMDGPU_TMP_VREG1_HI);
	knod_vset32(&d_lo, KNOD_AMDGPU_DATA_VREG_LO);
	knod_vset32(&d_hi, KNOD_AMDGPU_DATA_VREG_HI);

	knod_iset32(&imm, ETH_HLEN);
	knod_emit(priv, meta, v_add_co_u32, lb_lo, imm, d_lo);
	knod_iset32(&imm, 0);
	knod_emit(priv, meta, v_add_co_ci_u32_e32, lb_hi, imm, d_hi);

	/* VOPC#1: DATA_END < lb -> VCC = lower_fail */
	knod_vset64(&lb, KNOD_AMDGPU_TMP_VREG1_LO);
	knod_emit(priv, meta, v_cmp_lt_u64, dend_vreg, lb);

	/*
	 * Capture VCC -> VGPR via v_cndmask (VALU reads VCC correctly,
	 * unlike SALU which suffers the GFX10 dual-VOPC stale-read hazard).
	 */
	knod_iset32(&imm, 1);
	knod_mov32(priv, meta, fail_hi, imm);
	knod_iset32(&imm, 0);
	knod_emit(priv, meta, v_cndmask_b32_e32, fail_lo, imm, fail_hi);

	/* Exclude the next slot and reserved skb tail storage. */
	knod_bpf_packet_bound(priv, meta, lb, true);

	/* VOPC#2: DATA_END > ub -> VCC = upper_fail */
	knod_emit(priv, meta, v_cmp_gt_u64, dend_vreg, lb);

	/* Capture upper_fail via v_cndmask, combine, convert to VCC */
	knod_iset32(&imm, 0);
	knod_emit(priv, meta, v_cndmask_b32_e32, fail_hi, imm, fail_hi);

	knod_emit(priv, meta, v_or_b32_e32, fail_lo, fail_lo, fail_hi);

	knod_emit(priv, meta, v_cmp_lt_u32, imm, fail_lo);

	/* 5. Conditional restore: VCC=1(fail) -> original,
	 *    VCC=0(pass) -> adjusted
	 */
	knod_emit(priv, meta, v_cndmask_b32_e32, dend_lo, dend_lo,
		  tmp0_lo);
	knod_emit(priv, meta, v_cndmask_b32_e32, dend_hi, dend_hi,
		  tmp0_hi);

	/* 6. R0 = VCC ? -EINVAL : 0 */
	knod_iset32(&imm, -EINVAL);
	knod_mov32(priv, meta, tmp0_lo, imm);
	knod_iset32(&imm, 0);
	knod_emit(priv, meta, v_cndmask_b32_e32, r0_lo, imm, tmp0_lo);

	knod_iset32(&imm, -1);
	knod_mov32(priv, meta, tmp0_hi, imm);
	knod_iset32(&imm, 0);
	knod_emit(priv, meta, v_cndmask_b32_e32, r0_hi, imm, tmp0_hi);
}

/* @size bytes of @cache at @off, zero-extended into one 32-bit register.
 *
 * Three bytes have no load of their own.  Where they fit inside a dword the
 * bitfield extract takes them; where they straddle one, the dword load brings
 * the byte past the end along and it is masked off.  That byte matters: a key
 * is hashed a dword at a time and the host hashed only the key, so anything
 * carried in past its end lands the two on different buckets.
 */
/* Both accessors below reach exactly two consecutive dwords, cache[off / 4]
 * and the one after it, so a two-register window is all it takes to run them
 * unchanged against a stack that lives in scratch.  Returning the window
 * biased by the dword index is what lets the bodies keep indexing absolutely.
 */
/* Where a stack byte lives in LDS.  The stack is the top max_stack_off bytes of
 * the 512, laid out slot-major - every lane's dword N side by side - so the
 * offset is the slot's distance from the bottom of what is used, times the
 * workgroup.  The size check at finalisation keeps this under sixteen bits.
 */
static u16 knod_bpf_lds_off(struct knod_bpf_priv *priv,
			    struct knod_insn_meta *meta, int off)
{
	if (off < priv->lds_stack_base)
		pr_warn("knod_bpf: stack byte %d is below the tracked base %d (max_stack_off %d) at bpf insn %d code 0x%02x off %d\n",
			off, priv->lds_stack_base,
			priv->knod_prog ? priv->knod_prog->max_stack_off : -1,
			meta->bpf_insn_idx, meta->insn.code, meta->insn.off);
	return (off - priv->lds_stack_base) * knod_bpf_workgroups;
}





static bool knod_bpf_lds_pair_offsets(struct knod_bpf_priv *priv, int off,
				      u16 *off0, u16 *off1)
{
	u64 first, second;

	if (priv->isa_version != 10 || !knod_bpf_workgroups ||
	    knod_bpf_workgroups > KNOD_BPF_WORKGROUPS_MAX || (knod_bpf_workgroups & 63) ||
	    priv->lds_stack_base < 0 || off < priv->lds_stack_base ||
	    off > 504 || (off & 3))
		return false;
	first = (u64)(off - priv->lds_stack_base) * knod_bpf_workgroups;
	second = first + 4 * knod_bpf_workgroups;
	if ((first & 255) || (second & 255) || second > 65280 ||
	    second + 4 * knod_bpf_workgroups > 65536)
		return false;
	*off0 = first >> 8;
	*off1 = second >> 8;
	return true;
}

static bool knod_bpf_issue_stack_words(struct knod_bpf_priv *priv,
                                      struct knod_insn_meta *meta, int reg,
                                      int arg, int len)
{
	const struct knod_bpf_reg_state *state;
	struct amdgcn_param32 base, dst;
	int off, i;

	if (priv->isa_version != 10 || meta->jit_engine != 1 ||
	    arg != 2 || reg != KEY_IN_PKT_64 ||
	    len < 8 || len > 56 || (len & 3))
		return false;
	state = &meta->kreg;
	if (state->reg.type != PTR_TO_STACK)
		return false;
	off = 512 + state->stack_off;
	if (off < 0 || off > 512 - len || (off & 3) ||
	    priv->lds_stack_base < 0 || off < priv->lds_stack_base ||
	    !knod_bpf_workgroups || knod_bpf_workgroups > KNOD_BPF_WORKGROUPS_MAX ||
	    (u64)(off + len - priv->lds_stack_base) *
		knod_bpf_workgroups > 65536)
		return false;
	knod_vset32(&base, knod_bpf_lds_vreg(priv, KNOD_AMDGPU_LDS_BASE_VREG));
	for (i = 0; i < len / 4; i++) {
		u16 off0, off1;

		dst = i & 1 ? r64[reg + i / 2].hi : r64[reg + i / 2].lo;
		if (!(i & 1) && i + 1 < len / 4 &&
		    r64[reg + i / 2].hi.v == dst.v + 1 &&
		    knod_bpf_lds_pair_offsets(priv, off + i * 4, &off0, &off1)) {
			knod_emit(priv, meta, ds_read2st64_b32, dst, base, off0, off1);
			i++;
			continue;
		}
		knod_emit(priv, meta, ds_read_b32, dst, base,
			  knod_bpf_lds_off(priv, meta, off + i * 4));
	}
	return true;
}

static void knod_bpf_finish_stack_words(struct knod_bpf_priv *priv,
                                        struct knod_insn_meta *meta, int reg, int len)
{
	struct amdgcn_param32 zero;

	knod_emit(priv, meta, s_waitcnt_lgkmcnt);
	if (len & 4) {
		knod_iset32(&zero, 0);
		knod_mov32(priv, meta, r64[reg + len / 8].hi, zero);
	}
}

static bool knod_bpf_stage_stack_words(struct knod_bpf_priv *priv,
                                      struct knod_insn_meta *meta, int reg,
                                      int arg, int len)
{
	if (!knod_bpf_issue_stack_words(priv, meta, reg, arg, len))
		return false;
	knod_bpf_finish_stack_words(priv, meta, reg, len);
	return true;
}

static struct amdgcn_param32 *knod_bpf_stack_win(struct knod_bpf_priv *priv,
						 struct knod_insn_meta *meta,
						 struct amdgcn_param32 *win,
						 int off, bool load)
{
	knod_vset32(&win[0], knod_bpf_lds_vreg(priv, KNOD_AMDGPU_STACK_WIN_VREG0));
	knod_vset32(&win[1], knod_bpf_lds_vreg(priv, KNOD_AMDGPU_STACK_WIN_VREG1));

	if (load) {
		struct amdgcn_param32 base;

		knod_vset32(&base, knod_bpf_lds_vreg(priv, KNOD_AMDGPU_LDS_BASE_VREG));
		knod_emit(priv, meta, ds_read_b32, win[0], base,
			  knod_bpf_lds_off(priv, meta, off & ~3));
		knod_emit(priv, meta, ds_read_b32, win[1], base,
			  knod_bpf_lds_off(priv, meta, (off & ~3) + 4));
		knod_emit(priv, meta, s_waitcnt_lgkmcnt);
	}

	return win - (off / 4);
}

static void knod_bpf_stack_win_flush(struct knod_bpf_priv *priv,
				     struct knod_insn_meta *meta,
				     struct amdgcn_param32 *win, int off)
{
	struct amdgcn_param32 base;

	knod_vset32(&base, knod_bpf_lds_vreg(priv, KNOD_AMDGPU_LDS_BASE_VREG));
	knod_emit(priv, meta, ds_write_b32, base, win[0],
		  knod_bpf_lds_off(priv, meta, off & ~3));
	knod_emit(priv, meta, ds_write_b32, base, win[1],
		  knod_bpf_lds_off(priv, meta, (off & ~3) + 4));
}

static void __knod_bpf_load_size32(struct knod_bpf_priv *priv,
				   struct knod_insn_meta *meta,
				   struct amdgcn_param32 dst,
				   struct amdgcn_param32 *cache,
				   int size, int off);

static void knod_bpf_load_size32(struct knod_bpf_priv *priv,
				 struct knod_insn_meta *meta,
				 struct amdgcn_param32 dst,
				 struct amdgcn_param32 *cache,
				 int size, int off)
{
	struct amdgcn_param32 win[2];

	/* Subword slots are packed within each lane's dword. Odd halfwords
	 * retain the existing reconstruction path; never cross into a lane.
	 */
	if (cache == stack && priv->isa_version == 10 &&
	    (size == 1 || (size == 2 && !(off & 1)))) {
		struct amdgcn_param32 base;
		u16 addr = knod_bpf_lds_off(priv, meta, off & ~3) + (off & 3);

		knod_vset32(&base, knod_bpf_lds_vreg(priv, KNOD_AMDGPU_LDS_BASE_VREG));
		if (size == 1)
			knod_emit(priv, meta, ds_read_u8, dst, base, addr);
		else
			knod_emit(priv, meta, ds_read_u16, dst, base, addr);
		knod_emit(priv, meta, s_waitcnt_lgkmcnt);
		return;
	}

	/* A complete aligned word needs neither a second LDS slot nor a
	 * window copy. Keep the load completion boundary before its consumer.
	 */
	if (cache == stack && size == 4 && !(off & 3)) {
		struct amdgcn_param32 base;

		knod_vset32(&base, knod_bpf_lds_vreg(priv, KNOD_AMDGPU_LDS_BASE_VREG));
		knod_emit(priv, meta, ds_read_b32, dst, base,
			  knod_bpf_lds_off(priv, meta, off));
		knod_emit(priv, meta, s_waitcnt_lgkmcnt);
		return;
	}
	if (cache == stack)
		cache = knod_bpf_stack_win(priv, meta, win, off, true);

	__knod_bpf_load_size32(priv, meta, dst, cache, size, off);
}

static void __knod_bpf_load_size32(struct knod_bpf_priv *priv,
				 struct knod_insn_meta *meta,
				 struct amdgcn_param32 dst,
				 /* packet or stack */
				 struct amdgcn_param32 *cache,
				 int size, int off)
{
	struct amdgcn_param32 p32[2];

	if (size == 3 && (off % 4) <= 1) {
		knod_iset32(&p32[0], (off % 4) * 8);
		knod_iset32(&p32[1], 24);
		knod_bfe32(priv, meta, dst, cache[off / 4], p32[0], p32[1]);
		return;
	}

	switch (size) {
	case 3:
	case sizeof(unsigned int):
		if ((off % 4) == 0) {
			knod_mov32(priv, meta, dst, cache[off / 4]);
		} else if ((off % 4) == 1) {
			knod_iset32(&p32[0], 8);
			knod_lshrrev32(priv, meta, r32[0], p32[0],
					   cache[off / 4]);
			knod_iset32(&p32[0], 24);
			knod_lshlrev32(priv, meta, dst, p32[0],
					   cache[(off / 4) + 1]);
			knod_or32(priv, meta, dst, dst, r32[0]);
		} else if ((off % 4) == 2) {
			knod_iset32(&p32[0], 16);
			knod_lshrrev32(priv, meta, r32[0], p32[0],
					   cache[off / 4]);
			knod_lshlrev32(priv, meta, dst, p32[0],
					   cache[(off / 4) + 1]);
			knod_or32(priv, meta, dst, dst, r32[0]);
		} else {
			knod_iset32(&p32[0], 24);
			knod_lshrrev32(priv, meta, r32[0], p32[0],
					   cache[off / 4]);
			knod_iset32(&p32[0], 8);
			knod_lshlrev32(priv, meta, dst, p32[0],
					   cache[(off / 4) + 1]);
			knod_or32(priv, meta, dst, dst, r32[0]);
		}
		if (size == 3) {
			knod_iset32(&p32[0], 0xffffff);
			knod_and32(priv, meta, dst, dst, p32[0]);
		}
		break;
	case sizeof(unsigned short):
		if ((off % 4) == 3) {
			knod_iset32(&p32[0], 24);
			knod_iset32(&p32[1], 8);
			knod_bfe32(priv, meta, r64[0].lo, cache[off / 4],
				       p32[0], p32[1]);
			knod_iset32(&p32[0], 0);
			knod_bfe32(priv, meta, r64[0].hi,
				       cache[(off / 4) + 1], p32[0], p32[1]);
			/* dst = (r64[0].hi << 8) | r64[0].lo. */
			knod_emit(priv, meta, v_lshl_or_b32, dst,
				  r64[0].hi, p32[1], r64[0].lo);
		} else {
			if (!(off % 4))
				knod_iset32(&p32[0], 0);
			else if ((off % 4) == 1)
				knod_iset32(&p32[0], 8);
			else if ((off % 4) == 2)
				knod_iset32(&p32[0], 16);
			knod_iset32(&p32[1], 16);
			knod_bfe32(priv, meta, dst, cache[off / 4],
				       p32[0], p32[1]);
		}
		break;
	case sizeof(unsigned char):
		if ((off % 4) == 0)
			knod_iset32(&p32[0], 0);
		else if ((off % 4) == 1)
			knod_iset32(&p32[0], 8);
		else if ((off % 4) == 2)
			knod_iset32(&p32[0], 16);
		else
			knod_iset32(&p32[0], 24);
		knod_iset32(&p32[1], 8);
		knod_bfe32(priv, meta, dst, cache[off / 4], p32[0],
			       p32[1]);
		break;
	default:
		WARN_ON_ONCE(1);
		break;
	}
}

static void knod_bpf_load_size(struct knod_bpf_priv *priv,
			      struct knod_insn_meta *meta,
			      struct amdgcn_param64 *dst,
			      /* packet or stack */
			      struct amdgcn_param32 *cache,
			      int size, int off)
{
	struct amdgcn_param32 p32;

	knod_jit_dbg(" %d: off = %d off_4 = %d size = %d\n", meta->bpf_insn_idx,
		off, off%4, size);

	if (cache == stack && size == 8 && !(off & 3)) {
		struct amdgcn_param32 base;
		u16 off0, off1;

		/* Slot-major LDS: adjacent BPF words are WG*4 bytes apart,
		 * not a contiguous ds_read_b64 address.
		 */
		knod_vset32(&base, knod_bpf_lds_vreg(priv, KNOD_AMDGPU_LDS_BASE_VREG));
		if (dst->lo.type == AMDGCN_PARAM_TYPE_VGPR &&
		    dst->hi.type == AMDGCN_PARAM_TYPE_VGPR &&
		    dst->lo.v < 255 && dst->hi.v == dst->lo.v + 1 &&
		    knod_bpf_lds_pair_offsets(priv, off, &off0, &off1)) {
			knod_emit(priv, meta, ds_read2st64_b32, dst->lo, base,
				  off0, off1);
		} else {
			knod_emit(priv, meta, ds_read_b32, dst->lo, base,
				  knod_bpf_lds_off(priv, meta, off));
			knod_emit(priv, meta, ds_read_b32, dst->hi, base,
				  knod_bpf_lds_off(priv, meta, off + 4));
		}
		knod_emit(priv, meta, s_waitcnt_lgkmcnt);
		return;
	}
	if (size == sizeof(unsigned long)) {
		knod_bpf_load_size32(priv, meta, dst->lo, cache, 4, off);
		knod_bpf_load_size32(priv, meta, dst->hi, cache, 4, off + 4);
		return;
	}

	knod_bpf_load_size32(priv, meta, dst->lo, cache, size, off);
	knod_iset32(&p32, 0);
	knod_mov32(priv, meta, dst->hi, p32);
}

/* GFX10/11 queues enable unaligned accesses before code is submitted. */
static void knod_bpf_store_packet_imm(struct knod_bpf_priv *priv,
				  struct knod_insn_meta *meta,
				  struct amdgcn_param64 value,
				  struct amdgcn_param64 base, int off, int size)
{
	struct amdgcn_param64 data;

	knod_vset64(&data, KNOD_AMDGPU_TMP_VREG0_LO);
	knod_mov64(priv, meta, data, value);
	if (size == 2)
		knod_emit(priv, meta, global_store_short, data.lo, base.lo, off);
	else if (size == 4)
		knod_emit(priv, meta, global_store_dword, data.lo, base.lo, off);
	else
		knod_emit(priv, meta, global_store_dwordx2, data.lo, base.lo, off);
}

#define LABEL_NEXT	8
#define LABEL_OUT	9
static void knod_bpf_ktime_get_ns(struct knod_bpf_priv *priv,
				 struct knod_insn_meta *meta)
{
	struct amdgcn_param32 p[2];

	knod_sset32(&p[0], KNOD_AMDGPU_TMP_SREG0_LO);
	knod_sset32(&p[1], KNOD_AMDGPU_PARAM_SREG_LO);
	knod_emit(priv, meta, s_load_dwordx2, p[0], p[1],
		  offsetof(struct knod_bpf_param, ktime_ns));

	knod_emit(priv, meta, s_waitcnt_lgkmcnt);

	knod_sset32(&p[0], KNOD_AMDGPU_TMP_SREG0_LO);
	knod_mov32(priv, meta, bpf_reg64[0].lo, p[0]);
	knod_sset32(&p[0], KNOD_AMDGPU_TMP_SREG0_HI);
	knod_mov32(priv, meta, bpf_reg64[0].hi, p[0]);
}

/* Neither argument has to be moved into place: the JIT's fourth temporary pair
 * is the base of a routine's scratch window, and its eleventh is where the
 * value goes.
 */
static_assert(KNOD_AMDGPU_TMP_VREG0_LO + KEY_IN_PKT_64 * 2 ==
	      KNOD_BLOB_SPLICE_KEY_VREG);
static_assert(KNOD_AMDGPU_TMP_VREG0_LO + KEY_IN_MAP_64 * 2 ==
	      KNOD_BLOB_SPLICE_VAL_VREG);

enum knod_blob_op {
	KNOD_BLOB_OP_LOOKUP,
	KNOD_BLOB_OP_UPDATE,
	KNOD_BLOB_OP_DELETE,
};

/* Helper arguments are pointers, not necessarily addresses in the BPF stack.
 * Keep stack and packet reads in their software caches; map values use the
 * runtime BPF address, including any offset already added by the program.
 * R5 is caller-clobbered and is not used by the map emitters.
 */
static void knod_bpf_load_arg32(struct knod_bpf_priv *priv,
				struct knod_insn_meta *meta,
				struct amdgcn_param32 dst,
				int arg, int off, int len)
{
	const struct knod_bpf_reg_state *state =
		arg == 2 ? &meta->kreg : &meta->vreg;
	const struct bpf_reg_state *reg = &state->reg;
	struct amdgcn_param32 tmp = bpf_reg64[5].lo, shift;
	const struct bpf_map *map = reg->map_ptr;
	u32 first = meta->amdgpu_insns;
	bool aligned = false;
	int i;

	if (base_type(reg->type) == PTR_TO_STACK) {
		knod_bpf_load_size32(priv, meta, dst, stack, len,
				     512 + state->stack_off + off);
		return;
	}

	/* GFX10 dword loads require an aligned effective address.  Array
	 * elements and HASH values need their layout included in that proof.
	 * For other addresses use byte loads, never read beyond the argument.
	 */
	if (base_type(reg->type) == PTR_TO_MAP_VALUE && map &&
	    tnum_is_const(reg->var_off)) {
		u64 bias = reg->var_off.value + off;

		if (map->map_type == BPF_MAP_TYPE_HASH ||
		    map->map_type == BPF_MAP_TYPE_PERCPU_HASH)
			aligned = !((knod_bpf_hash_value_off(map->key_size) +
				     bias) & 3);
		else if (map->map_type == BPF_MAP_TYPE_ARRAY ||
			 map->map_type == BPF_MAP_TYPE_PERCPU_ARRAY)
			aligned = !(map->value_size & 3) && !(bias & 3);
	}
	if (aligned && len == 4) {
		knod_emit(priv, meta, global_load_dword, dst,
			  bpf_reg64[arg].lo, off);
		knod_wait_vmcnt(priv, meta);
	} else {
		knod_emit(priv, meta, global_load_ubyte, dst,
			  bpf_reg64[arg].lo, off);
		knod_wait_vmcnt(priv, meta);
		for (i = 1; i < len; i++) {
			knod_emit(priv, meta, global_load_ubyte, tmp,
				  bpf_reg64[arg].lo, off + i);
			knod_wait_vmcnt(priv, meta);
			knod_iset32(&shift, i * 8);
			knod_lshlrev32(priv, meta, tmp, shift, tmp);
			knod_or32(priv, meta, dst, dst, tmp);
		}
	}
	knod_map_bypass_l0(priv, meta, first);
}

static void knod_bpf_load_arg(struct knod_bpf_priv *priv,
			      struct knod_insn_meta *meta,
			      struct amdgcn_param64 *dst,
			      int arg, int off, int len)
{
	struct amdgcn_param32 zero;

	knod_bpf_load_arg32(priv, meta, dst->lo, arg, off, min(len, 4));
	if (len > 4)
		knod_bpf_load_arg32(priv, meta, dst->hi, arg, off + 4, len - 4);
	else {
		knod_iset32(&zero, 0);
		knod_mov32(priv, meta, dst->hi, zero);
	}
}

/* Gather exactly @len bytes into the routine's consecutive argument pairs. */
static void knod_bpf_stage_arg(struct knod_bpf_priv *priv,
			       struct knod_insn_meta *meta, int reg,
			       int arg, int len)
{
	int off = 0, n;

	if (knod_bpf_stage_stack_words(priv, meta, reg, arg, len))
		return;
	while (len > 0) {
		n = min(len, 8);
		knod_bpf_load_arg(priv, meta, &r64[reg++], arg, off, n);
		off += n;
		len -= n;
	}
}

/* Which routine does this, if the blob has one.  A value that is not a whole
 * number of dwords has no routine: an array packs its elements value_size
 * apart, so the tail would have to be cut out of a register chosen at run
 * time, which a prebuilt routine cannot do.
 */
static bool knod_bpf_map_blob_kind(const struct knod_bpf_map_obj *obj,
				   enum knod_blob_op op, u32 *kind, u32 *chunks)
{
	static const u32 by_type_op[4][3] = {
		[0] = { KNOD_BLOB_LOOKUP_ARRAY, KNOD_BLOB_UPDATE_ARRAY,
			KNOD_BLOB_DELETE_ARRAY },
		[1] = { KNOD_BLOB_LOOKUP_PERCPU_ARRAY,
			KNOD_BLOB_UPDATE_PERCPU_ARRAY,
			KNOD_BLOB_DELETE_PERCPU_ARRAY },
		[2] = { KNOD_BLOB_LOOKUP_HASH, KNOD_BLOB_UPDATE_HASH,
			KNOD_BLOB_DELETE_HASH },
		/* Delete is element-level (values do not matter), so it reuses
		 * the plain-hash routine.  Update writes only this instance's
		 * value slot; the other slots stay zero because the host clears an
		 * element's value region before returning it to the free list.
		 */
		[3] = { KNOD_BLOB_LOOKUP_PERCPU_HASH,
			KNOD_BLOB_UPDATE_PERCPU_HASH,
			KNOD_BLOB_DELETE_HASH },
	};
	unsigned int row;

	switch (obj->map_type) {
	case BPF_MAP_TYPE_ARRAY:
		row = 0;
		*chunks = 0;
		break;
	case BPF_MAP_TYPE_PERCPU_ARRAY:
		row = 1;
		*chunks = 0;
		break;
	case BPF_MAP_TYPE_HASH:
		row = 2;
		*chunks = DIV_ROUND_UP(obj->key_size, 4);
		break;
	case BPF_MAP_TYPE_PERCPU_HASH:
		row = 3;
		*chunks = DIV_ROUND_UP(obj->key_size, 4);
		break;
	default:
		return false;
	}

	if (op != KNOD_BLOB_OP_LOOKUP &&
	    (obj->value_size % 4 ||
	     DIV_ROUND_UP(obj->value_size, 4) > KNOD_BLOB_VALUE_CHUNKS_MAX))
		return false;

	*kind = by_type_op[row][op];

	return true;
}

/* Hand the operation to a prebuilt routine if there is one for this map.  All
 * the JIT puts around it is the arguments; how the map is searched and written
 * stops being its business, and the result is already in r0.
 */
static bool knod_bpf_array_lookup_const(struct knod_bpf_priv *priv,
				       struct knod_insn_meta *meta,
				       struct knod_bpf_map *map,
				       enum knod_blob_op op)
{
	const struct knod_bpf_map_obj *obj = map->knod_map_obj;
	const struct knod_blob_map_desc *desc = map->desc;
	struct amdgcn_param32 tmp, key, size, stride, carry, imm, wg;
	struct amdgcn_param64 base, ret;
	bool percpu;

	if (priv->isa_version != 10 || op != KNOD_BLOB_OP_LOOKUP ||
	    obj->key_size != 4 ||
	    (obj->map_type != BPF_MAP_TYPE_ARRAY &&
	     obj->map_type != BPF_MAP_TYPE_PERCPU_ARRAY))
		return false;
	if (obj->map_type == BPF_MAP_TYPE_PERCPU_ARRAY &&
	    desc->per_instance_size > U32_MAX)
		return false;
	if (WARN_ON_ONCE(meta->blob))
		return false;

	percpu = obj->map_type == BPF_MAP_TYPE_PERCPU_ARRAY;
	/* The descriptor already supplies immutable native address operands.
	 * Fold only a proven in-range key; retain the existing path on overflow.
	 * The BPF NULL branch and surrounding EXEC structure remain unchanged.
	 */
	if (meta->jit_engine == 1 && meta->array_key_proven &&
	    meta->array_key_literal < desc->max_entries) {
		u64 offset = (u64)meta->array_key_literal * desc->value_size;

		if (offset <= U64_MAX - desc->elems_gaddr) {
			u64 address = desc->elems_gaddr + offset;

			knod_vset64(&ret, KNOD_BLOB_SPLICE_R0_VREG);
			knod_iset32(&imm, (u32)address);
			knod_mov32(priv, meta, ret.lo, imm);
			knod_iset32(&imm, address >> 32);
			knod_mov32(priv, meta, ret.hi, imm);
			if (percpu) {
				knod_vset32(&tmp, KNOD_BLOB_SPLICE_TMP_VREG + 2);
				knod_sset32(&stride, KNOD_BLOB_SPLICE_TMP_SREG + 1);
				knod_sset32(&carry, KNOD_BLOB_SPLICE_TMP_SREG + 2);
				knod_sset32(&wg, KNOD_BLOB_PRO_WG_Y_SREG);
				knod_iset32(&imm, desc->per_instance_size);
				knod_emit(priv, meta, s_mov_b32, stride, imm);
				knod_mov32(priv, meta, tmp, wg);
				knod_emit(priv, meta, v_mad_u64_u32, ret, carry,
					  stride, tmp, ret);
			}
			return true;
		}
	}
	/* Only a CFG-proven literal replaces this private stack key read. */
	if (meta->jit_engine == 1 && meta->array_key_proven) {
		knod_vset32(&key, KNOD_BLOB_SPLICE_KEY_VREG);
		knod_iset32(&imm, meta->array_key_literal);
		knod_mov32(priv, meta, key, imm);
		knod_vset32(&key, KNOD_BLOB_SPLICE_KEY_VREG + 1);
		knod_iset32(&imm, 0);
		knod_mov32(priv, meta, key, imm);
	} else {
		knod_bpf_stage_arg(priv, meta, KEY_IN_PKT_64, 2, obj->key_size);
	}
	knod_vset64(&base, KNOD_BLOB_SPLICE_TMP_VREG);
	knod_vset64(&ret, KNOD_BLOB_SPLICE_R0_VREG);
	knod_vset32(&tmp, KNOD_BLOB_SPLICE_TMP_VREG + 2);
	knod_vset32(&key, KNOD_BLOB_SPLICE_KEY_VREG);
	knod_sset32(&size, KNOD_BLOB_SPLICE_TMP_SREG);
	knod_sset32(&stride, KNOD_BLOB_SPLICE_TMP_SREG + 1);
	knod_sset32(&carry, KNOD_BLOB_SPLICE_TMP_SREG + 2);
	knod_sset32(&wg, KNOD_BLOB_PRO_WG_Y_SREG);
	knod_iset32(&imm, 0);
	knod_mov32(priv, meta, ret.lo, imm);
	knod_mov32(priv, meta, ret.hi, imm);
	knod_iset32(&imm, desc->max_entries);
	knod_mov32(priv, meta, tmp, imm);
	knod_emit(priv, meta, v_cmp_ge_u32, key, tmp);
	knod_emit(priv, meta, s_and_b64, KNOD_BLOB_EXEC_SAVE_SREG,
		  AMDGCN_SREG_EXEC_LO, AMDGCN_SREG_VCC_LO);
	knod_emit(priv, meta, s_andn2_b64, AMDGCN_SREG_EXEC_LO,
		  AMDGCN_SREG_EXEC_LO, AMDGCN_SREG_VCC_LO);
	/*
	 * No memory access, election or loop follows. Empty EXEC is safe: SALU
	 * only materializes constants; masked VALU does not touch inactive lanes.
	 */
	knod_iset32(&imm, (u32)desc->elems_gaddr);
	knod_mov32(priv, meta, base.lo, imm);
	knod_iset32(&imm, desc->elems_gaddr >> 32);
	knod_mov32(priv, meta, base.hi, imm);
	knod_iset32(&imm, desc->value_size);
	knod_emit(priv, meta, s_mov_b32, size, imm);
	if (percpu) {
		knod_iset32(&imm, desc->per_instance_size);
		knod_emit(priv, meta, s_mov_b32, stride, imm);
		knod_mov32(priv, meta, tmp, wg);
		knod_emit(priv, meta, v_mad_u64_u32, base, carry,
			  stride, tmp, base);
	}
	knod_emit(priv, meta, v_mad_u64_u32, ret, carry, size, key, base);
	knod_emit(priv, meta, s_or_b64, AMDGCN_SREG_EXEC_LO,
		  AMDGCN_SREG_EXEC_LO, KNOD_BLOB_EXEC_SAVE_SREG);
	return true;
}

static void knod_bpf_hash_preload(struct knod_bpf_priv *priv,
				struct knod_insn_meta *meta,
				const struct knod_blob_map_desc *desc)
{
	struct amdgcn_param32 dst, imm;
	const u32 vreg[] = { KNOD_BLOB_HASH_PRE_ELEMS_VREG,
			    KNOD_BLOB_HASH_PRE_ELEMS_VREG + 1,
			    KNOD_BLOB_HASH_PRE_BUCKET_VREG,
			    KNOD_BLOB_HASH_PRE_BUCKET_VREG + 1 };
	const u32 value[] = { (u32)desc->elems_gaddr, desc->elems_gaddr >> 32,
			     (u32)desc->bucket_gaddr, desc->bucket_gaddr >> 32 };
	const u32 sreg[] = { KNOD_BLOB_HASH_PRE_SEED_SREG,
			    KNOD_BLOB_HASH_PRE_MASK_SREG,
			    KNOD_BLOB_HASH_PRE_STRIDE_SREG };
	const u32 scalar[] = { desc->key_size + desc->hashrnd,
			      desc->n_buckets - 1, desc->elem_size };
	unsigned int i;

	for (i = 0; i < ARRAY_SIZE(vreg); i++) {
		knod_vset32(&dst, vreg[i]);
		knod_iset32(&imm, value[i]);
		knod_mov32(priv, meta, dst, imm);
	}
	for (i = 0; i < ARRAY_SIZE(sreg); i++) {
		knod_sset32(&dst, sreg[i]);
		knod_iset32(&imm, scalar[i]);
		knod_emit(priv, meta, s_mov_b32, dst, imm);
	}
}

static bool knod_bpf_map_op_blob(struct knod_bpf_priv *priv,
				 struct knod_insn_meta *meta,
				 struct knod_bpf_map *knod_map,
				 enum knod_blob_op op)
{
	const struct knod_bpf_map_obj *obj = knod_map->knod_map_obj;
	struct amdgcn_param32 p32[2];
	u32 kind, chunks, size;
	const u32 *code;
	bool preloaded = false, key_pending = false;

	/* A meta holds one spliced routine, because it records one place to
	 * put it.  One BPF call is one meta, so this should not come up.
	 */
	if (WARN_ON_ONCE(meta->blob))
		return false;

	if (!knod_bpf_map_blob_kind(obj, op, &kind, &chunks))
		return false;

	/* Optional entry selection is transactional: absence retains the old ABI. */
	code = NULL;
	if ((priv->isa_version == 10 || priv->isa_version == 11) &&
	    kind == KNOD_BLOB_LOOKUP_HASH && knod_map->desc->n_buckets &&
	    is_power_of_2(knod_map->desc->n_buckets) &&
	    knod_map->desc->elem_size) {
		code = knod_blob_find(&priv->blob,
				      KNOD_BLOB_LOOKUP_HASH_PRELOADED, chunks, &size);
		preloaded = !!code;
	}
	if (!code)
		code = knod_blob_find(&priv->blob, kind, chunks, &size);
	if (!code) {
		pr_warn_once("knod_bpf: blob has no %s for a %u-dword key; emitting it\n",
			     knod_blob_kind_name(kind), chunks);
		return false;
	}

	/* Only this lookup owns an issued key until the descriptor MOVs finish.
	 * No UPDATE argument or fallback caller receives an asynchronous key.
	 */
	if (priv->isa_version == 10 && meta->jit_engine == 1 &&
	    preloaded &&
	    op == KNOD_BLOB_OP_LOOKUP)
		key_pending = knod_bpf_issue_stack_words(priv, meta,
				KEY_IN_PKT_64, 2, obj->key_size);
	if (!key_pending)
		knod_bpf_stage_arg(priv, meta, KEY_IN_PKT_64, 2,
				   obj->key_size);
	if (op == KNOD_BLOB_OP_UPDATE)
		knod_bpf_stage_arg(priv, meta, KEY_IN_MAP_64,
				   3, obj->value_size);

	if (preloaded) {
		knod_bpf_hash_preload(priv, meta, knod_map->desc);
	} else {
		knod_sset32(&p32[0], KNOD_BLOB_SPLICE_DESC_SREG);
		knod_iset32(&p32[1], knod_map->desc_gaddr & ~0U);
		knod_emit(priv, meta, s_mov_b32, p32[0], p32[1]);
		knod_sset32(&p32[0], KNOD_BLOB_SPLICE_DESC_SREG + 1);
		knod_iset32(&p32[1], knod_map->desc_gaddr >> 32);
		knod_emit(priv, meta, s_mov_b32, p32[0], p32[1]);
	}

	if (key_pending)
		knod_bpf_finish_stack_words(priv, meta, KEY_IN_PKT_64,
				    obj->key_size);

	/* Scalar instructions are not masked, so a routine entered with no live
	 * lane still runs - and one that elects a lane with mbcnt, or spins on
	 * a lock, does not come back out.  The contract puts this guard on the
	 * caller; the branch clears exactly the routine.
	 */
	emit_s_cbranch_execz(priv->isa_version,
			     &meta->amdgpu_insn[meta->amdgpu_insns], size / 4);
	meta->amdgpu_insns++;

	/* A routine standing in for a helper leaves its result in r0, so there
	 * is nothing to emit after it.
	 */
	meta->blob = code;
	meta->blob_size = size;
	meta->blob_at = meta->amdgpu_insns;

	return true;
}

/* True if a prebuilt routine took the operation and the emitter below it can
 * be skipped.
 */
static bool knod_bpf_map_op(struct knod_bpf_priv *priv,
			    struct knod_insn_meta *meta, int map_id,
			    enum knod_blob_op op)
{
	struct knod_bpf_map *knod_map = knod_bpf_map_find(priv, map_id);

	if (!knod_map) {
		WARN_ON_ONCE(1);
		return false;
	}

	if (knod_bpf_array_lookup_const(priv, meta, knod_map, op))
		return true;
	return knod_bpf_map_op_blob(priv, meta, knod_map, op);
}

/* A percpu value has one instance per queue, and a queue is one workgroup, so
 * the queue id names the instance.  The same step a lookup takes, for the
 * helpers that reach a value without going through one.
 *
 * r64[1] holds the bucket base and r64[3] is free between the bounds check and
 * the address it is about to be used for.
 */
/* The store of a percpu read-modify-write, as one atomic.
 *
 * A percpu value has one instance per queue, and a queue is one workgroup, so
 * the 256 lanes of it all reach the same copy.  Read it, add, write it back and
 * they each read the same number and one of the results survives; the atomic is
 * what makes every lane's addition land.
 *
 * Counting the lanes and sending their total once is cheaper, but needs them
 * to be adding the same thing to the same place.  Where the key was a constant
 * every lane reaches the same element and that holds; where the program looked
 * one up per packet it does not, and a single atomic would put the whole wave's
 * worth on whichever element the lane that sent it had - which still sums to
 * the right total, and took a per-VIP breakdown to notice.
 */
static unsigned knod_bpf_atomic_forward_words(struct knod_insn_meta *meta, unsigned branch,
					      unsigned end)
{
	unsigned i, words = 0;

	for (i = branch + 1; i < end; i++)
		words += meta->amdgpu_insn[i].size / 4;
	return words;
}

static void knod_bpf_percpu_delta(struct knod_bpf_priv *priv, struct knod_insn_meta *meta,
				  struct amdgcn_param64 delta)
{
	const struct knod_insn_meta *alu = meta->percpu_rmw_add;
	bool dw = BPF_SIZE(meta->insn.code) == BPF_DW;
	struct amdgcn_param32 imm;

	if (meta->percpu_delta_direct) {
		knod_mov32(priv, meta, delta.lo, bpf_reg64[alu->insn.dst_reg].lo);
		knod_mov32(priv, meta, delta.hi, bpf_reg64[alu->insn.dst_reg].hi);
	} else if (meta->percpu_rmw_swapped) {
		if (dw)
			knod_sub64(priv, meta, delta, bpf_reg64[alu->insn.dst_reg],
				   bpf_reg64[alu->insn.src_reg]);
		else
			knod_sub32(priv, meta, delta.lo, bpf_reg64[alu->insn.dst_reg].lo,
				   bpf_reg64[alu->insn.src_reg].lo);
	} else if (BPF_SRC(alu->insn.code) == BPF_K) {
		knod_iset32(&imm, alu->insn.imm);
		knod_mov32(priv, meta, delta.lo, imm);
		if (dw) {
			knod_iset32(&imm, alu->insn.imm < 0 ? ~0U : 0);
			knod_mov32(priv, meta, delta.hi, imm);
		}
	} else {
		knod_mov32(priv, meta, delta.lo, bpf_reg64[alu->insn.src_reg].lo);
		if (dw)
			knod_mov32(priv, meta, delta.hi, bpf_reg64[alu->insn.src_reg].hi);
	}
}

static void knod_bpf_percpu_atomic(struct knod_bpf_priv *priv, struct knod_insn_meta *meta,
				   struct amdgcn_param64 delta)
{
	if (BPF_SIZE(meta->insn.code) == BPF_DW)
		knod_emit(priv, meta, global_atomic_add_x2, delta.lo,
			  bpf_reg64[meta->insn.dst_reg].lo, delta.lo, meta->insn.off, 0);
	else
		knod_emit(priv, meta, global_atomic_add, delta.lo, bpf_reg64[meta->insn.dst_reg].lo,
			  delta.lo, meta->insn.off, 0);
}

static void knod_bpf_percpu_fold(struct knod_bpf_priv *priv, struct knod_insn_meta *meta,
				 struct amdgcn_param64 delta, bool one)
{
	struct amdgcn_param32 count, lane, high, zero, exec_lo, exec_hi;

	knod_sset32(&count, KNOD_AMDGPU_TMP_SREG0_LO);
	knod_sset32(&exec_lo, AMDGCN_SREG_EXEC_LO);
	knod_sset32(&exec_hi, AMDGCN_SREG_EXEC_LO + 1);
	knod_vset32(&lane, KNOD_AMDGPU_TMP_VREG1_LO);
	knod_vset32(&high, KNOD_AMDGPU_TMP_VREG1_HI);
	knod_iset32(&zero, 0);
	knod_emit(priv, meta, s_bcnt1_i32_b64, KNOD_AMDGPU_TMP_SREG0_LO, AMDGCN_SREG_EXEC_LO);
	if (one) {
		knod_mov32(priv, meta, delta.lo, count);
		knod_mov32(priv, meta, delta.hi, zero);
	} else {
		if (BPF_SIZE(meta->insn.code) == BPF_DW) {
			knod_emit(priv, meta, v_mul_hi_u32, high, count, delta.lo);
			knod_emit(priv, meta, v_mul_lo_u32, delta.hi, count, delta.hi);
			knod_emit(priv, meta, v_add_u32, delta.hi, delta.hi, high);
		}
		knod_emit(priv, meta, v_mul_lo_u32, delta.lo, count, delta.lo);
	}
	knod_emit(priv, meta, v_mbcnt_lo_u32_b32, lane, exec_lo, zero);
	if (64 == 64)
		knod_emit(priv, meta, v_mbcnt_hi_u32_b32, lane, exec_hi, lane);
	knod_emit(priv, meta, v_cmp_eq_u32, zero, lane);
	knod_emit(priv, meta, s_and_saveexec_b64, KNOD_AMDGPU_TMP_SREG0_LO, AMDGCN_SREG_VCC_LO);
	knod_bpf_percpu_atomic(priv, meta, delta);
	knod_emit(priv, meta, s_mov_b64, AMDGCN_SREG_EXEC_LO, KNOD_AMDGPU_TMP_SREG0_LO);
}

static void knod_bpf_emit_percpu_add(struct knod_bpf_priv *priv, struct knod_insn_meta *meta)
{
	struct amdgcn_param64 delta, first;
	struct amdgcn_param32 scalar;
	unsigned empty, address_diff = 0, different = 0, done, fallback, end;
	bool immediate = BPF_SRC(meta->percpu_rmw_add->insn.code) == BPF_K;

	knod_vset64(&delta, KNOD_AMDGPU_TMP_VREG0_LO);
	/* The address proof applies to the current EXEC at either width;
	 * a constant +1 needs only its active-lane count as the full-width delta.
	 */
	if (priv->isa_version == 10 && meta->jit_engine == 1 &&
	    meta->percpu_rmw_uniform && meta->percpu_addr_proven &&
	    !meta->percpu_rmw_swapped && !meta->percpu_delta_direct &&
	    BPF_SIZE(meta->insn.code) == BPF_DW &&
	    meta->percpu_rmw_add->insn.code == (BPF_ALU64 | BPF_ADD | BPF_K) &&
	    meta->percpu_rmw_add->insn.imm == 1) {
		empty = meta->amdgpu_insns;
		knod_emit(priv, meta, s_cbranch_execz, 0);
		knod_bpf_percpu_fold(priv, meta, delta, true);
		emit_s_cbranch_execz(priv->isa_version, &meta->amdgpu_insn[empty],
				     knod_bpf_atomic_forward_words(meta, empty,
							   meta->amdgpu_insns));
		return;
	}
	knod_bpf_percpu_delta(priv, meta, delta);
	if (!meta->percpu_rmw_uniform) {
		knod_bpf_percpu_atomic(priv, meta, delta);
		return;
	}
	if (priv->isa_version != 10) {
		knod_bpf_percpu_atomic(priv, meta, delta);
		return;
	}

	/* readfirstlane requires a live lane. Do not change EXEC for the test;
	 * any differing active amount selects the unchanged per-lane path.
	 */
	empty = meta->amdgpu_insns;
	knod_emit(priv, meta, s_cbranch_execz, 0);
	knod_vset64(&first, KNOD_AMDGPU_TMP_VREG1_LO);
	knod_sset32(&scalar, KNOD_BLOB_SPLICE_TMP_SREG);
	if (!meta->percpu_addr_proven) {
		/* The verifier-era uniform flag is only a hint. Prove the actual
		 * complete address at runtime, including conditional constant keys.
		 */
		knod_emit(priv, meta, v_readfirstlane_b32, KNOD_BLOB_SPLICE_TMP_SREG,
			  bpf_reg64[meta->insn.dst_reg].lo.v);
		knod_mov32(priv, meta, first.lo, scalar);
		knod_emit(priv, meta, v_readfirstlane_b32, KNOD_BLOB_SPLICE_TMP_SREG,
			  bpf_reg64[meta->insn.dst_reg].hi.v);
		knod_mov32(priv, meta, first.hi, scalar);
		knod_emit(priv, meta, v_cmp_eq_u64, bpf_reg64[meta->insn.dst_reg].lo, first.lo);
		knod_emit(priv, meta, s_andn2_b64, AMDGCN_SREG_VCC_LO, AMDGCN_SREG_EXEC_LO,
			  AMDGCN_SREG_VCC_LO);
		address_diff = meta->amdgpu_insns;
		knod_emit(priv, meta, s_cbranch_vccnz, 0);
	}
	if (!immediate) {
		knod_emit(priv, meta, v_readfirstlane_b32, KNOD_BLOB_SPLICE_TMP_SREG,
			  KNOD_AMDGPU_TMP_VREG0_LO);
		knod_mov32(priv, meta, first.lo, scalar);
		if (BPF_SIZE(meta->insn.code) == BPF_DW) {
			knod_emit(priv, meta, v_readfirstlane_b32, KNOD_BLOB_SPLICE_TMP_SREG,
				  KNOD_AMDGPU_TMP_VREG0_HI);
			knod_mov32(priv, meta, first.hi, scalar);
			knod_emit(priv, meta, v_cmp_eq_u64, delta.lo, first.lo);
		} else {
			knod_emit(priv, meta, v_cmp_eq_u32, delta.lo, first.lo);
		}
		knod_emit(priv, meta, s_andn2_b64, AMDGCN_SREG_VCC_LO, AMDGCN_SREG_EXEC_LO,
			  AMDGCN_SREG_VCC_LO);
		different = meta->amdgpu_insns;
		knod_emit(priv, meta, s_cbranch_vccnz, 0);
	}
	knod_bpf_percpu_fold(priv, meta, delta, false);
	done = meta->amdgpu_insns;
	knod_emit(priv, meta, s_branch, 0);
	fallback = meta->amdgpu_insns;
	knod_bpf_percpu_atomic(priv, meta, delta);
	end = meta->amdgpu_insns;
	emit_s_cbranch_execz(priv->isa_version, &meta->amdgpu_insn[empty],
			     knod_bpf_atomic_forward_words(meta, empty, end));
	if (!meta->percpu_addr_proven)
		emit_s_cbranch_vccnz(priv->isa_version, &meta->amdgpu_insn[address_diff],
				    knod_bpf_atomic_forward_words(meta, address_diff, fallback));
	if (!immediate)
		emit_s_cbranch_vccnz(priv->isa_version, &meta->amdgpu_insn[different],
				    knod_bpf_atomic_forward_words(meta, different, fallback));
	emit_s_branch(priv->isa_version, &meta->amdgpu_insn[done],
		      knod_bpf_atomic_forward_words(meta, done, end));
}

static void __knod_bpf_store_cache_size(struct knod_bpf_priv *priv,
					struct knod_insn_meta *meta,
					struct amdgcn_param64 *src,
					struct amdgcn_param32 *cache,
					int size, int off);

static void knod_bpf_store_cache_size(struct knod_bpf_priv *priv,
				      struct knod_insn_meta *meta,
				      struct amdgcn_param64 *src,
				      struct amdgcn_param32 *cache,
				      int size, int off)
{
	struct amdgcn_param32 win[2];

	if (cache != stack) {
		__knod_bpf_store_cache_size(priv, meta, src, cache, size, off);
		return;
	}

	if (priv->isa_version == 10 &&
	    (size == 1 || (size == 2 && !(off & 1)))) {
		struct amdgcn_param32 base;
		u16 addr = knod_bpf_lds_off(priv, meta, off & ~3) + (off & 3);

		knod_vset32(&base, knod_bpf_lds_vreg(priv, KNOD_AMDGPU_LDS_BASE_VREG));
		knod_vset32(&win[0], knod_bpf_lds_vreg(priv, KNOD_AMDGPU_STACK_WIN_VREG0));
		knod_mov32(priv, meta, win[0], src->lo);
		if (size == 1)
			knod_emit(priv, meta, ds_write_b8, base, win[0], addr);
		else
			knod_emit(priv, meta, ds_write_b16, base, win[0], addr);
		return;
	}

	if (!(off & 3) && (size == 4 || size == 8)) {
		struct amdgcn_param32 base;
		u16 off0, off1;

		/* Native LDS reads finish before returning. Ordinary GFX10
		 * LDS writes do not retain their source VGPRs until completion.
		 * Keep the established wait contract for other generations.
		 */
		if (priv->isa_version != 10)
			knod_emit(priv, meta, s_waitcnt_lgkmcnt);
		knod_vset32(&base, knod_bpf_lds_vreg(priv, KNOD_AMDGPU_LDS_BASE_VREG));
		knod_vset32(&win[0], knod_bpf_lds_vreg(priv, KNOD_AMDGPU_STACK_WIN_VREG0));
		knod_vset32(&win[1], knod_bpf_lds_vreg(priv, KNOD_AMDGPU_STACK_WIN_VREG1));
		knod_mov32(priv, meta, win[0], src->lo);
		if (size == 8)
			knod_mov32(priv, meta, win[1], src->hi);
		if (size == 8 &&
		    knod_bpf_lds_pair_offsets(priv, off, &off0, &off1)) {
			knod_emit(priv, meta, ds_write2st64_b32, base,
				  win[0], win[1], off0, off1);
		} else {
			knod_emit(priv, meta, ds_write_b32, base, win[0],
				  knod_bpf_lds_off(priv, meta, off));
			if (size == 8)
				knod_emit(priv, meta, ds_write_b32, base, win[1],
					  knod_bpf_lds_off(priv, meta, off + 4));
		}
		if (priv->isa_version != 10)
			knod_emit(priv, meta, s_waitcnt_lgkmcnt);
		return;
	}

	/* Read-modify-write even when the body writes only one of the pair:
	 * the sub-dword cases merge into what is already there.
	 */
	cache = knod_bpf_stack_win(priv, meta, win, off, true);
	__knod_bpf_store_cache_size(priv, meta, src, cache, size, off);
	knod_bpf_stack_win_flush(priv, meta, win, off);
}

static void __knod_bpf_store_cache_size(struct knod_bpf_priv *priv,
				     struct knod_insn_meta *meta,
				     struct amdgcn_param64 *src,
				     /* packet or stack */
				     struct amdgcn_param32 *cache,
				     int size, int off)
{
	struct amdgcn_param32 p32[2];

	knod_jit_dbg(" %d: off = %d off_4 = %d size = %d\n", meta->bpf_insn_idx,
		off, off%4, size);
	WARN_ON(knod_param_is_literal(src->lo) ||
		knod_param_is_literal(src->hi));
	switch (size) {
	case sizeof(unsigned long):
		if ((off % 4) == 0) {
			knod_mov32(priv, meta,
				       cache[off / 4],
				       src->lo);
			knod_mov32(priv, meta,
				       cache[(off / 4) + 1],
				       src->hi);
		} else if ((off % 4) == 1) {
			WARN_ON_ONCE(1);
		} else if ((off % 4) == 2) {
			WARN_ON_ONCE(1);
		} else {
			WARN_ON_ONCE(1);
		}
		break;
	case sizeof(unsigned int):
		if ((off % 4) == 0) {
			knod_mov32(priv, meta,
				       cache[off / 4],
				       src->lo);
		} else if ((off % 4) == 1) {
			knod_iset64(&p64[0], 8);
			knod_lshlrev64(priv, meta, r64[0], p64[0], *src);

			knod_iset32(&p32[0], 0xffffff00);
			knod_mov32(priv, meta, r32[2], p32[0]);
			knod_bfi32(priv, meta, cache[off / 4],
				       r32[2], r64[0].lo, cache[off / 4]);
			knod_iset32(&p32[0], 0x000000ff);
			knod_mov32(priv, meta, r32[2], p32[0]);
			knod_bfi32(priv, meta, cache[(off / 4) + 1], r32[2],
				       r64[0].hi, cache[(off / 4) + 1]);
		} else if ((off % 4) == 2) {
			knod_iset64(&p64[0], 16);
			knod_lshlrev64(priv, meta, r64[0], p64[0], *src);

			knod_iset32(&p32[0], 0xffff0000);
			knod_mov32(priv, meta, r32[2], p32[0]);
			knod_bfi32(priv, meta, cache[off / 4], r32[2],
				       r64[0].lo, cache[off / 4]);
			knod_iset32(&p32[0], 0x0000ffff);
			knod_mov32(priv, meta, r32[2], p32[0]);
			knod_bfi32(priv, meta, cache[(off / 4) + 1], r32[2],
				       r64[0].hi, cache[(off / 4) + 1]);
		} else {
			knod_iset64(&p64[0], 24);
			knod_lshlrev64(priv, meta, r64[0], p64[0], *src);

			knod_iset32(&p32[0], 0xffff0000);
			knod_mov32(priv, meta, r32[2], p32[0]);
			knod_bfi32(priv, meta, cache[off / 4], r32[2],
				       r64[0].lo, cache[off / 4]);
			knod_iset32(&p32[0], 0x00ffffff);
			knod_mov32(priv, meta, r32[2], p32[0]);
			knod_bfi32(priv, meta, cache[(off / 4) + 1], r32[2],
				       r64[0].hi, cache[(off / 4) + 1]);
		}
		break;
	case sizeof(unsigned short):
		if ((off % 4) == 0) {
			knod_iset32(&p32[0], 0x0000ffff);
			knod_mov32(priv, meta, r32[2], p32[0]);
			knod_bfi32(priv, meta, cache[off / 4], r32[2],
				       src->lo, cache[off / 4]);
		} else if ((off % 4) == 1) {
			knod_iset32(&p32[0], 8);
			knod_lshlrev32(priv, meta, r32[0], p32[0],
					   src->lo);
			knod_iset32(&p32[0], 0x00ffff00);
			knod_mov32(priv, meta, r32[2], p32[0]);
			knod_bfi32(priv, meta, cache[off / 4], r32[2],
				       r32[0], cache[off / 4]);
		} else if ((off % 4) == 2) {
			knod_iset32(&p32[0], 16);
			knod_lshlrev32(priv, meta, r32[0], p32[0],
					   src->lo);
			knod_iset32(&p32[0], 0xffff0000);
			knod_mov32(priv, meta, r32[2], p32[0]);
			knod_bfi32(priv, meta, cache[off / 4], r32[2],
				       r32[0], cache[off / 4]);
		} else {
			knod_iset64(&p64[0], 24);
			knod_lshlrev64(priv, meta, r64[0], p64[0], *src);

			knod_iset32(&p32[0], 0xff000000);
			knod_mov32(priv, meta, r32[2], p32[0]);
			knod_bfi32(priv, meta, cache[off / 4], r32[2],
				       r64[0].lo, cache[off / 4]);
			knod_iset32(&p32[0], 0x000000ff);
			knod_mov32(priv, meta, r32[2], p32[0]);
			knod_bfi32(priv, meta, cache[(off / 4) + 1], r32[2],
				       r64[0].hi, cache[(off / 4) + 1]);
		}
		break;
	case sizeof(unsigned char):
		if ((off % 4) == 0) {
			knod_iset32(&p32[0], 0x000000ff);
			knod_mov32(priv, meta, r32[2], p32[0]);
			knod_bfi32(priv, meta, cache[off / 4], r32[2],
				       src->lo, cache[off / 4]);
			return;
		} else if ((off % 4) == 1) {
			knod_iset32(&p32[0], 8);
			knod_lshlrev32(priv, meta, r32[0], p32[0],
					   src->lo);

			knod_iset32(&p32[0], 0x0000ff00);
		} else if ((off % 4) == 2) {
			knod_iset32(&p32[0], 16);
			knod_lshlrev32(priv, meta, r32[0], p32[0],
					   src->lo);

			knod_iset32(&p32[0], 0x00ff0000);
		} else {
			knod_iset32(&p32[0], 24);
			knod_lshlrev32(priv, meta, r32[0], p32[0],
					   src->lo);
			knod_iset32(&p32[0], 0xff000000);
		}

		knod_mov32(priv, meta, r32[2], p32[0]);
		knod_bfi32(priv, meta, cache[off / 4], r32[2], r32[0],
			       cache[off / 4]);
		break;
	default:
		WARN_ON_ONCE(1);
	}
}

static bool knod_meta_is_exit(const struct knod_insn_meta *meta);
static bool knod_bpf_is_retval_move_to_r0(const struct knod_insn_meta *meta);

/*
 * knod_bpf_emit_branch_tail - Emit EXEC mask manipulation after v_cmp for
 * structurized per-lane branching. Replaces the old s_cbranch_vccnz/vccz.
 *
 * For FORWARD_SKIP:
 *   Save jumping lanes -> narrow EXEC -> s_cbranch_execz
 *   (skip if no active lanes)
 *
 * For DIRECT_EXIT:
 *   Compute exit lanes -> update done_mask -> remove from EXEC (no branch)
 *
 * Emits the required EXEC mask manipulation in-place.
 */
static void knod_bpf_emit_direct_exit_retval(struct knod_bpf_priv *priv,
					     struct knod_insn_meta *emit_meta,
					     struct knod_insn_meta *target)
{
	struct amdgcn_param64 dst, src;
	s64 imm;

	if (!target || knod_meta_is_exit(target))
		return;

	if (WARN_ON_ONCE(!knod_bpf_is_retval_move_to_r0(target)))
		return;

	knod_vset64(&dst, KNOD_AMDGPU_VREG0_LO);

	switch (target->insn.code) {
	case BPF_ALU | BPF_MOV | BPF_X:
	case BPF_ALU64 | BPF_MOV | BPF_X:
		knod_vset64(&src, target->insn.src_reg * 2);
		knod_mov64(priv, emit_meta, dst, src);
		break;
	case BPF_ALU | BPF_MOV | BPF_K:
		imm = (u32)target->insn.imm;
		knod_iset64(&src, imm);
		knod_mov64(priv, emit_meta, dst, src);
		break;
	case BPF_ALU64 | BPF_MOV | BPF_K:
		imm = (s64)(s32)target->insn.imm;
		knod_iset64(&src, imm);
		knod_mov64(priv, emit_meta, dst, src);
		break;
	default:
		WARN_ON_ONCE(1);
		break;
	}
}

static void knod_bpf_emit_branch_tail(struct knod_bpf_priv *priv,
				      struct knod_insn_meta *meta,
				      struct knod_prog *knod_prog,
				      short off)
{
	switch (meta->branch_type) {
	case KNOD_BR_FORWARD_SKIP:
		if (meta->jump_neg_op) {
			/* JNE: VCC=0 -> jump, VCC=1 -> fall-through.
			 * Save jump lanes (VCC=0): s[n] = exec & ~vcc
			 */
			knod_emit(priv, meta, s_andn2_b64,
				  meta->exec_save_sreg,
				  AMDGCN_SREG_EXEC_LO,
				  AMDGCN_SREG_VCC_LO);

			/* Keep fall-through (VCC=1): exec = exec & vcc */
			knod_emit(priv, meta, s_and_b64,
				  AMDGCN_SREG_EXEC_LO,
				  AMDGCN_SREG_EXEC_LO,
				  AMDGCN_SREG_VCC_LO);
		} else {
			/* Normal: VCC=1 -> jump, VCC=0 -> fall-through.
			 * Save jump lanes (VCC=1): s[n] = exec & vcc
			 */
			knod_emit(priv, meta, s_and_b64,
				  meta->exec_save_sreg,
				  AMDGCN_SREG_EXEC_LO,
				  AMDGCN_SREG_VCC_LO);

			/* Keep fall-through (VCC=0): exec = exec & ~vcc */
			knod_emit(priv, meta, s_andn2_b64,
				  AMDGCN_SREG_EXEC_LO,
				  AMDGCN_SREG_EXEC_LO,
				  AMDGCN_SREG_VCC_LO);
		}

		/*
		 * No GPU branch.  After the RPO reorder, branch scopes
		 * interleave, so the jumping lanes must flow through every
		 * following block under the EXEC mask and rejoin at their merge
		 * point.  An s_cbranch_execz skipping ahead to the merge would
		 * jump over other scopes' merge points and strand their saved
		 * lanes (EXEC never restored -> act=0).
		 */
		break;

	case KNOD_BR_DIRECT_EXIT:
		if (meta->jump_neg_op) {
			/* JNE: VCC=0 -> exit. exit_lanes = exec & ~vcc */
			knod_emit(priv, meta, s_andn2_b64,
				  KNOD_AMDGPU_TMP_SREG0_LO,
				  AMDGCN_SREG_EXEC_LO,
				  AMDGCN_SREG_VCC_LO);
		} else {
			/* Normal: VCC=1 -> exit. exit_lanes = exec & vcc */
			knod_emit(priv, meta, s_and_b64,
				  KNOD_AMDGPU_TMP_SREG0_LO,
				  AMDGCN_SREG_EXEC_LO,
				  AMDGCN_SREG_VCC_LO);
		}

		/* Keep the lanes that did not take the exit path. */
		knod_emit(priv, meta, s_andn2_b64,
			  KNOD_AMDGPU_TMP_SREG1_LO,
			  AMDGCN_SREG_EXEC_LO,
			  KNOD_AMDGPU_TMP_SREG0_LO);

		/* Replay a shared "r0 = action; exit" target under the
		 * exiting lanes before marking them done.  Otherwise a direct
		 * branch to the common exit can publish stale r0 scratch state.
		 */
		knod_emit(priv, meta, s_mov_b64, AMDGCN_SREG_EXEC_LO,
			  KNOD_AMDGPU_TMP_SREG0_LO);
		knod_bpf_emit_direct_exit_retval(priv, meta, meta->merge_point);

		/* done_mask |= exit_lanes */
		knod_emit(priv, meta, s_or_b64,
			  knod_prog->done_mask_sreg,
			  knod_prog->done_mask_sreg,
			  AMDGCN_SREG_EXEC_LO);

		/* Continue with the non-exit lanes. */
		knod_emit(priv, meta, s_mov_b64, AMDGCN_SREG_EXEC_LO,
			  KNOD_AMDGPU_TMP_SREG1_LO);

		/* No branch - fall through with reduced EXEC.
		 * No fixup needed.
		 */
		meta->jmp_dst = NULL;
		break;

	default:
		WARN_ON_ONCE(1);
		break;
	}
}

/*
 * --- Basic-block CFG analysis (foundation for block reordering) ---
 *
 * The emitter is a linear SIMT machine: instructions run in list order under
 * an EXEC mask.  A *forward* jump is realized by masking off the jumping
 * lanes and restoring them at the merge point.  A *backward* jump has no such
 * realization unless it is a loop (real GPU branch + EXEC convergence, not yet
 * implemented).
 *
 * LLVM tail-sharing and block placement routinely emit jumps that are
 * backward in BPF byte order but are NOT loops - e.g. a UDP bounds check that
 * jumps back to a shared XDP_PASS tail.  Classifying those as "exit" (the
 * jmp_off < 0 heuristic in knod_bpf_analyze_cfg) silently miscompiles them:
 * the jumping lanes exit carrying whatever R0 happened to hold instead of
 * flowing to the real target.
 *
 * The fix is to classify by control-flow, not byte order:
 *   1. partition the instruction stream into basic blocks,
 *   2. build the control-flow graph (successor edges),
 *   3. DFS for a reverse-postorder (RPO) and detect back-edges,
 *   4. no back-edges (a DAG)  -> reorder blocks into RPO so every edge points
 *      forward, then classify by linear position,
 *   5. a real loop is present -> bail (-EOPNOTSUPP) until loop emission lands.
 *
 * Loop emission (step 5) is not implemented yet, so programs containing a
 * loop are rejected with -EOPNOTSUPP.
 */
struct knod_bb {
	struct knod_insn_meta *leader;	/* first instruction of the block */
	struct knod_insn_meta *last;	/* last instruction of the block */
	/* successors: [0] not-taken, [1] taken */
	struct knod_bb *succ[2];
	int n_succ;
	/* reverse-postorder rank, -1 if unreachable */
	int rpo;
	/* DFS color: 0 white, 1 gray, 2 black */
	int dfs;
	bool loop_header;		/* target of a back-edge */
	/* scratch: member of the loop being walked */
	bool in_loop;
	/* immediate dominator (self for entry) */
	struct knod_bb *idom;
};

static bool knod_meta_is_exit(const struct knod_insn_meta *meta)
{
	u8 code = meta->insn.code;

	return code == (BPF_JMP | BPF_EXIT) || code == (BPF_JMP32 | BPF_EXIT);
}

static struct knod_insn_meta *
knod_bpf_next_meta(struct knod_prog *knod_prog, struct knod_insn_meta *meta)
{
	if (!meta || list_is_last(&meta->l, &knod_prog->insns))
		return NULL;

	return list_next_entry(meta, l);
}

static bool knod_bpf_is_retval_move_to_r0(const struct knod_insn_meta *meta)
{
	u8 code;

	if (!meta || meta->insn.dst_reg != BPF_REG_0)
		return false;

	code = meta->insn.code;
	return code == (BPF_ALU | BPF_MOV | BPF_X) ||
	       code == (BPF_ALU64 | BPF_MOV | BPF_X) ||
	       code == (BPF_ALU | BPF_MOV | BPF_K) ||
	       code == (BPF_ALU64 | BPF_MOV | BPF_K);
}

static bool knod_bpf_is_direct_exit_target(struct knod_prog *knod_prog,
					   struct knod_insn_meta *target)
{
	if (knod_meta_is_exit(target))
		return true;

	if (!knod_bpf_is_retval_move_to_r0(target))
		return false;

	return knod_meta_is_exit(knod_bpf_next_meta(knod_prog, target));
}

static bool knod_meta_is_ja(const struct knod_insn_meta *meta)
{
	u8 code = meta->insn.code;

	return code == (BPF_JMP | BPF_JA | BPF_K) ||
	       code == (BPF_JMP32 | BPF_JA | BPF_K);
}

/* A block ends after a terminator; the next instruction starts a new block. */
static bool knod_meta_is_terminator(const struct knod_insn_meta *meta)
{
	return is_mbpf_cond_jump(meta) || knod_meta_is_ja(meta) ||
	       knod_meta_is_exit(meta);
}

/* Target instruction index of a conditional jump or BPF_JA. */
static short knod_meta_jump_target_idx(const struct knod_insn_meta *meta)
{
	if (meta->insn.code == (BPF_JMP32 | BPF_JA | BPF_K))
		return meta->bpf_insn_idx + meta->insn.imm + 1;
	return meta->bpf_insn_idx + meta->insn.off + 1;
}

static struct knod_bb *knod_bb_of_leader(struct knod_bb *bbs, int n_bbs,
					 const struct knod_insn_meta *meta)
{
	int i;

	for (i = 0; i < n_bbs; i++)
		if (bbs[i].leader == meta)
			return &bbs[i];
	return NULL;
}

/* Resolve the block a conditional jump / BPF_JA at @jmp transfers to. */
static struct knod_bb *knod_bb_jump_target(struct knod_prog *knod_prog,
					   struct knod_bb *bbs, int n_bbs,
					   const struct knod_insn_meta *jmp)
{
	struct knod_insn_meta *tgt;

	tgt = knod_bpf_lookup_meta(knod_prog, knod_meta_jump_target_idx(jmp));
	return tgt ? knod_bb_of_leader(bbs, n_bbs, tgt) : NULL;
}

/*
 * Partition knod_prog->insns into basic blocks.  A leader is the first
 * instruction, any jump target, or the instruction after a terminator.
 * Returns the block count or a negative errno; @bbs holds >= n_insns blocks.
 */
static int knod_bpf_build_bbs(struct knod_prog *knod_prog, struct knod_bb *bbs)
{
	struct knod_insn_meta *meta, *tgt;
	struct knod_bb *cur = NULL;
	int n_bbs = 0;
	short tgt_idx;

	/* Pass A: mark every jump target as a leader. */
	list_for_each_entry(meta, &knod_prog->insns, l)
		meta->flags &= ~FLAG_INSN_IS_JUMP_DST;

	list_for_each_entry(meta, &knod_prog->insns, l) {
		if (!is_mbpf_cond_jump(meta) && !knod_meta_is_ja(meta))
			continue;
		tgt_idx = knod_meta_jump_target_idx(meta);
		tgt = knod_bpf_lookup_meta(knod_prog, tgt_idx);
		if (!tgt) {
			pr_warn("knod_cfg: bpf#%d jump target %d unresolved\n",
				meta->bpf_insn_idx, tgt_idx);
			return -EINVAL;
		}
		tgt->flags |= FLAG_INSN_IS_JUMP_DST;
	}

	/* Pass B: cut the list into blocks. */
	list_for_each_entry(meta, &knod_prog->insns, l) {
		if (!cur || (meta->flags & FLAG_INSN_IS_JUMP_DST)) {
			cur = &bbs[n_bbs++];
			cur->leader = meta;
			cur->n_succ = 0;
		}
		cur->last = meta;

		if (knod_meta_is_terminator(meta))
			/* next instruction starts a new block */
			cur = NULL;
	}

	return n_bbs;
}

/* Build successor edges for every block from its terminator. */
static int knod_bpf_build_edges(struct knod_prog *knod_prog,
				struct knod_bb *bbs, int n_bbs)
{
	struct knod_bb *bb, *fall, *tgt_bb;
	struct knod_insn_meta *last;
	int i;

	for (i = 0; i < n_bbs; i++) {
		bb = &bbs[i];
		last = bb->last;
		bb->n_succ = 0;

		if (knod_meta_is_exit(last))
			continue;			/* no successors */

		/* Successor in list order: block led by the next
		 * instruction.
		 */
		fall = NULL;
		if (!list_is_last(&last->l, &knod_prog->insns))
			fall = knod_bb_of_leader(bbs, n_bbs,
						 list_next_entry(last, l));

		if (is_mbpf_cond_jump(last)) {
			tgt_bb = knod_bb_jump_target(knod_prog, bbs, n_bbs,
						     last);
			if (!fall || !tgt_bb)
				return -EINVAL;
			bb->succ[bb->n_succ++] = fall;		/* not taken */
			bb->succ[bb->n_succ++] = tgt_bb;	/* taken */
		} else if (knod_meta_is_ja(last)) {
			tgt_bb = knod_bb_jump_target(knod_prog, bbs, n_bbs,
						     last);
			if (!tgt_bb)
				return -EINVAL;
			bb->succ[bb->n_succ++] = tgt_bb;
		} else {
			if (!fall)			/* fell off the end */
				return -EINVAL;
			bb->succ[bb->n_succ++] = fall;
		}
	}

	return 0;
}

/*
 * Iterative DFS from the entry block.  Computes a reverse-postorder rank for
 * every reachable block and flags back-edge targets as loop headers.  Returns
 * the number of back-edges in *n_back, or a negative errno.
 */
static int knod_bpf_compute_rpo(struct knod_bb *bbs, int n_bbs,
				struct knod_bb *entry, int *n_back)
{
	struct knod_bb **stack;
	int *cursor;
	int top = 0, post = 0, nb = 0, i;

	for (i = 0; i < n_bbs; i++) {
		bbs[i].dfs = 0;
		bbs[i].rpo = -1;
		bbs[i].loop_header = false;
	}

	stack = kcalloc(n_bbs, sizeof(*stack), GFP_KERNEL);
	cursor = kcalloc(n_bbs, sizeof(*cursor), GFP_KERNEL);
	if (!stack || !cursor) {
		kfree(stack);
		kfree(cursor);
		return -ENOMEM;
	}

	entry->dfs = 1;
	stack[top] = entry;
	cursor[top] = 0;
	top++;

	while (top > 0) {
		struct knod_bb *bb = stack[top - 1];

		if (cursor[top - 1] < bb->n_succ) {
			struct knod_bb *s = bb->succ[cursor[top - 1]++];

			if (s->dfs == 0) {		/* tree edge */
				s->dfs = 1;
				stack[top] = s;
				cursor[top] = 0;
				top++;
			} else if (s->dfs == 1) {	/* gray -> back-edge */
				s->loop_header = true;
				nb++;
			}
			/* s->dfs == 2 -> forward/cross edge, nothing to do */
		} else {
			/* finished: postorder */
			bb->dfs = 2;
			bb->rpo = post++;
			top--;
		}
	}

	/* postorder -> reverse-postorder rank */
	for (i = 0; i < n_bbs; i++)
		if (bbs[i].rpo >= 0)
			bbs[i].rpo = post - 1 - bbs[i].rpo;

	kfree(stack);
	kfree(cursor);
	*n_back = nb;
	return 0;
}

/*
 * Cooper-Harvey-Kennedy dominator intersect: walk the two fingers up the idom
 * chain (toward the entry, which has the lowest RPO) until they meet.
 */
static struct knod_bb *knod_dom_intersect(struct knod_bb *a, struct knod_bb *b)
{
	while (a != b) {
		while (a->rpo > b->rpo)
			a = a->idom;
		while (b->rpo > a->rpo)
			b = b->idom;
	}
	return a;
}

/*
 * Compute the immediate dominator of every reachable block (Cooper, Harvey,
 * Kennedy, "A Simple, Fast Dominance Algorithm").  Iterates over RPO to a
 * fixpoint; bb->idom is the block's immediate dominator, the entry dominating
 * itself.  Requires bb->rpo from knod_bpf_compute_rpo.
 */
static int knod_bpf_compute_dom(struct knod_bb *bbs, int n_bbs,
				struct knod_bb *entry)
{
	struct knod_bb **order;
	int i, k, n_order = 0;
	bool changed;

	order = kcalloc(n_bbs, sizeof(*order), GFP_KERNEL);
	if (!order)
		return -ENOMEM;

	for (i = 0; i < n_bbs; i++) {
		bbs[i].idom = NULL;
		if (bbs[i].rpo >= 0) {
			order[bbs[i].rpo] = &bbs[i];
			n_order++;
		}
	}
	entry->idom = entry;

	do {
		changed = false;

		/* process every reachable block but the entry, in RPO order */
		for (k = 1; k < n_order; k++) {
			struct knod_bb *n = order[k];
			struct knod_bb *new_idom = NULL;
			int b, s;

			/* intersect over already-processed predecessors */
			for (b = 0; b < n_bbs; b++) {
				for (s = 0; s < bbs[b].n_succ; s++) {
					if (bbs[b].succ[s] != n || !bbs[b].idom)
						continue;
					new_idom = new_idom ?
						knod_dom_intersect(&bbs[b],
								   new_idom) :
						&bbs[b];
				}
			}

			if (new_idom && n->idom != new_idom) {
				n->idom = new_idom;
				changed = true;
			}
		}
	} while (changed);

	kfree(order);
	return 0;
}

/* Does block @a dominate block @b?  Walk @b up the idom chain to the entry. */
static bool knod_dom_dominates(struct knod_bb *a, struct knod_bb *b)
{
	for (;;) {
		if (b == a)
			return true;
		if (b->idom == b)	/* reached the entry */
			return false;
		b = b->idom;
	}
}

/*
 * Mark the natural loop body of back-edge @latch->@hdr in bb->in_loop: the
 * header plus every block that reaches the latch without passing through the
 * header, found by walking predecessors back from the latch.  @stack is
 * caller-provided scratch of at least @n_bbs entries.
 */
static void knod_loop_mark_body(struct knod_bb *bbs, int n_bbs,
				struct knod_bb *latch, struct knod_bb *hdr,
				struct knod_bb **stack)
{
	int b, sp, k, top = 0;

	for (k = 0; k < n_bbs; k++)
		bbs[k].in_loop = false;

	hdr->in_loop = true;
	if (latch != hdr) {
		latch->in_loop = true;
		stack[top++] = latch;
	}

	while (top > 0) {
		struct knod_bb *d = stack[--top];

		for (b = 0; b < n_bbs; b++) {
			if (bbs[b].in_loop)
				continue;
			for (sp = 0; sp < bbs[b].n_succ; sp++) {
				if (bbs[b].succ[sp] != d)
					continue;
				bbs[b].in_loop = true;
				stack[top++] = &bbs[b];
				break;
			}
		}
	}
}

/*
 * Detect natural loops from the dominator tree and report their structure.
 *
 * A back-edge is an edge u->v whose target v dominates its source u - v is
 * the loop header, u the latch.  Its natural loop body is the header plus the
 * blocks that reach the latch without passing through the header; an exit edge
 * leaves a body block for a non-body block.
 *
 * Loops are still rejected by the reorder (-EOPNOTSUPP); this only reports what
 * was found (to dmesg, since a rejected program never attaches so /bpf/cfg is
 * unavailable) so the detection can be verified before emission is built.
 */
static int knod_bpf_detect_loops(struct knod_bb *bbs, int n_bbs)
{
	struct knod_bb **stack;
	int u, s, k, n_be = 0;

	stack = kcalloc(n_bbs, sizeof(*stack), GFP_KERNEL);
	if (!stack)
		return -ENOMEM;

	for (u = 0; u < n_bbs; u++) {
		for (s = 0; s < bbs[u].n_succ; s++) {
			struct knod_bb *hdr = bbs[u].succ[s];
			int body = 0, exits = 0, sp;

			if (!knod_dom_dominates(hdr, &bbs[u]))
				continue;	/* not a back-edge */
			n_be++;

			knod_loop_mark_body(bbs, n_bbs, &bbs[u], hdr, stack);

			for (k = 0; k < n_bbs; k++) {
				if (!bbs[k].in_loop)
					continue;
				body++;
				for (sp = 0; sp < bbs[k].n_succ; sp++)
					if (!bbs[k].succ[sp]->in_loop)
						exits++;
			}

			pr_info("knod_loop: back-edge bpf#%d -> bpf#%d (latch->header) body=%d exits=%d\n",
				bbs[u].leader->bpf_insn_idx,
				hdr->leader->bpf_insn_idx, body, exits);
		}
	}

	kfree(stack);

	if (n_be)
		pr_info("knod_loop: %d back-edge(s) - %s\n", n_be,
			n_be == 1 ? "single loop (simple-shape candidate)" :
				    "nested/multiple loops (complex)");
	return 0;
}

/*
 * Block that lanes fall into in list order when the terminator is not taken:
 * the not-taken successor of a conditional jump, or the sole successor of a
 * block that ended only because the next instruction was a leader.  BPF_JA and
 * EXIT have no such successor (control leaves explicitly).
 */
static struct knod_bb *knod_bb_fall_succ(struct knod_bb *bb)
{
	if (knod_meta_is_exit(bb->last) || knod_meta_is_ja(bb->last))
		return NULL;
	return bb->n_succ ? bb->succ[0] : NULL;
}

/*
 * Reorder the instruction list into reverse-postorder so every control-flow
 * edge points forward, and splice in a synthetic BPF_JA wherever a block's
 * not-taken successor no longer follows it in list order.  After this the
 * emitter's forward-only machinery (FORWARD_SKIP / FORWARD_GOTO) handles the
 * whole program - including the backward-in-byte-order, non-loop jumps that
 * the old jmp_off < 0 heuristic miscompiled.
 *
 * Loops (back-edges) are rejected with -EOPNOTSUPP until loop emission lands.
 */
static int knod_bpf_reorder_rpo(struct knod_prog *knod_prog,
				struct knod_bb *bbs, int n_bbs, int n_back)
{
	struct knod_insn_meta *m, *nx, *sj;
	struct knod_bb **order;
	int n_order = 0, r, i, k, idx = 0;
	LIST_HEAD(new_list);

	if (n_back) {
		pr_warn("knod_cfg: %d loop back-edge(s) - block reorder cannot lower loops yet (-EOPNOTSUPP)\n",
			n_back);
		return -EOPNOTSUPP;
	}

	order = kcalloc(n_bbs, sizeof(*order), GFP_KERNEL);
	if (!order)
		return -ENOMEM;

	/* Reachable blocks in RPO, then any unreachable ones so no instruction
	 * is dropped from the list.
	 */
	for (r = 0; r < n_bbs; r++)
		for (i = 0; i < n_bbs; i++)
			if (bbs[i].rpo == r) {
				order[n_order++] = &bbs[i];
				break;
			}
	for (i = 0; i < n_bbs; i++)
		if (bbs[i].rpo < 0)
			order[n_order++] = &bbs[i];

	for (k = 0; k < n_order; k++) {
		struct knod_bb *bb = order[k];
		struct knod_bb *next = (k + 1 < n_order) ? order[k + 1] : NULL;
		struct knod_bb *fall;

		m = bb->leader;
		while (true) {
			nx = (m == bb->last) ? NULL : knod_meta_next(m);
			list_move_tail(&m->l, &new_list);
			if (m == bb->last)
				break;
			m = nx;
		}

		fall = knod_bb_fall_succ(bb);
		if (!fall || (next && next->leader == fall->leader))
			continue;

		/* Not-taken successor no longer adjacent: route it
		 * explicitly.
		 */
		sj = kzalloc_obj(*sj, GFP_KERNEL);
		if (!sj) {
			list_splice(&new_list, &knod_prog->insns);
			kfree(order);
			return -ENOMEM;
		}
		sj->insn.code = BPF_JMP | BPF_JA | BPF_K;
		/* synthetic, never a jump target */
		sj->bpf_insn_idx = -1;
		/* consumed by classify_linear */
		sj->jmp_dst = fall->leader;
		INIT_LIST_HEAD(&sj->l);
		list_add_tail(&sj->l, &new_list);
	}

	list_splice(&new_list, &knod_prog->insns);

	list_for_each_entry(m, &knod_prog->insns, l)
		m->linear_idx = idx++;

	kfree(order);
	return 0;
}

/*
 * Classify branches by linear position after the RPO reorder.  Every edge is
 * now forward, so a conditional jump is FORWARD_SKIP (or DIRECT_EXIT when it
 * targets the exit), and every BPF_JA - real or synthetic - is FORWARD_GOTO
 * (or DIRECT_EXIT).
 */
static int knod_bpf_classify_linear(struct knod_prog *knod_prog)
{
	struct knod_insn_meta *meta, *target;
	short ti;

	list_for_each_entry(meta, &knod_prog->insns, l) {
		if (is_mbpf_cond_jump(meta)) {
			meta->jump_neg_op = (mbpf_op(meta) == BPF_JNE);
			ti = knod_meta_jump_target_idx(meta);
			target = knod_bpf_lookup_meta(knod_prog, ti);
		} else if (knod_meta_is_ja(meta)) {
			/* synthetic JA carries its destination in jmp_dst;
			 * a real BPF_JA is resolved from its offset.
			 */
			if (meta->jmp_dst) {
				target = meta->jmp_dst;
			} else {
				ti = knod_meta_jump_target_idx(meta);
				target = knod_bpf_lookup_meta(knod_prog, ti);
			}
		} else {
			continue;
		}

		if (!target) {
			pr_err("knod_cfg: bpf#%d unresolved branch target\n",
			       meta->bpf_insn_idx);
			return -EINVAL;
		}

		if (target->linear_idx <= meta->linear_idx)
			pr_warn("knod_cfg: bpf#%d -> #%d still backward after reorder (linear %d -> %d)\n",
				meta->bpf_insn_idx, target->bpf_insn_idx,
				meta->linear_idx, target->linear_idx);

		if (knod_bpf_is_direct_exit_target(knod_prog, target)) {
			meta->branch_type = KNOD_BR_DIRECT_EXIT;
			meta->merge_point = target;
			continue;
		}

		meta->branch_type = is_mbpf_cond_jump(meta) ?
			KNOD_BR_FORWARD_SKIP : KNOD_BR_FORWARD_GOTO;
		meta->merge_point = target;
		target->is_merge_point = true;
	}

	return 0;
}

/*
 * Build the basic-block CFG, compute RPO, reorder the instruction list into
 * RPO and insert synthetic jumps.  Returns 0, or a negative errno (a loop
 * yields -EOPNOTSUPP).
 */
static int knod_bpf_build_cfg(struct knod_prog *knod_prog)
{
	struct knod_insn_meta *meta;
	int n_insns = 0, n_bbs, n_back = 0, ret;
	struct knod_bb *bbs;

	list_for_each_entry(meta, &knod_prog->insns, l)
		n_insns++;
	if (!n_insns)
		return 0;

	bbs = kcalloc(n_insns, sizeof(*bbs), GFP_KERNEL);
	if (!bbs)
		return -ENOMEM;

	n_bbs = knod_bpf_build_bbs(knod_prog, bbs);
	if (n_bbs < 0) {
		ret = n_bbs;
		goto out_free;
	}

	ret = knod_bpf_build_edges(knod_prog, bbs, n_bbs);
	if (ret)
		goto out_free;

	ret = knod_bpf_compute_rpo(bbs, n_bbs, &bbs[0], &n_back);
	if (ret)
		goto out_free;

	ret = knod_bpf_compute_dom(bbs, n_bbs, &bbs[0]);
	if (ret)
		goto out_free;

	if (n_back) {
		ret = knod_bpf_detect_loops(bbs, n_bbs);
		if (ret)
			goto out_free;
	}

	/* Hand the block array to the prog for the /bpf/cfg view (freed at
	 * teardown); kept even if the reorder below rejects a loop, so the
	 * rejection can be inspected.
	 */
	kfree(knod_prog->bbs);
	knod_prog->bbs = bbs;
	knod_prog->n_bbs = n_bbs;
	knod_prog->n_back = n_back;

	return knod_bpf_reorder_rpo(knod_prog, bbs, n_bbs, n_back);

out_free:
	kfree(bbs);
	return ret;
}

/*
 * Assign exec_save SGPR pairs to the forward branches, recycling a pair once
 * its merge point has been passed.  The peak concurrent live count is the
 * actual SGPR requirement - usually far less than the total branch count.
 */
static int knod_bpf_alloc_exec_sregs(struct knod_prog *knod_prog)
{
	struct {
		u8 sreg;
		struct knod_insn_meta *merge;
	} live[KNOD_AMDGPU_MAX_EXEC_SAVE_PAIRS];
	u8 free_stack[KNOD_AMDGPU_MAX_EXEC_SAVE_PAIRS];
	int max_pairs, n_live, peak, j;
	struct knod_insn_meta *meta;
	int free_top;

	max_pairs = (KNOD_AMDGPU_EXEC_SAVE_SREG_MAX -
		     knod_prog->exec_save_base + 1) / 2;

	for (free_top = 0; free_top < max_pairs; free_top++)
		free_stack[free_top] = knod_prog->exec_save_base +
			(max_pairs - 1 - free_top) * 2;

	n_live = 0;
	peak = 0;

	list_for_each_entry(meta, &knod_prog->insns, l) {
		/* Reclaim pairs from scopes that merge at this insn */
		for (j = n_live - 1; j >= 0; j--) {
			if (live[j].merge == meta) {
				free_stack[free_top++] = live[j].sreg;
				live[j] = live[--n_live];
			}
		}

		if (meta->branch_type != KNOD_BR_FORWARD_SKIP &&
		    meta->branch_type != KNOD_BR_FORWARD_GOTO)
			continue;

		if (free_top == 0) {
			pr_err("knod_cfg: exec_save exhausted, peak %d concurrent scopes (max %d)\n",
			       peak, max_pairs);
			return -ENOSPC;
		}

		meta->exec_save_sreg = free_stack[--free_top];
		live[n_live].sreg = meta->exec_save_sreg;
		live[n_live].merge = meta->merge_point;
		n_live++;

		if (n_live > peak)
			peak = n_live;
	}

	knod_prog->exec_save_pairs_used = peak;
	pr_debug("knod_cfg: done, peak %d concurrent scopes (total fwd jumps: %d+%d)\n",
		 peak, peak, n_live);
	return 0;
}

/*
 * knod_bpf_analyze_cfg - Classify branches and allocate SGPRs for
 * structurized CFG.
 *
 * Runs before instruction emission. For each conditional branch:
 *   - Backward jump or jump to EXIT -> DIRECT_EXIT (no SGPR needed)
 *   - Forward jump to non-EXIT -> FORWARD_SKIP, allocate SGPR pair
 *
 * The "save jumping lanes" pattern handles crossing scopes correctly:
 *   branch: s_and_b64 s[n], exec, vcc; s_andn2_b64 exec, exec, vcc
 *   merge:  s_or_b64 exec, exec, s[n]
 *
 * For JNE (jump_neg_op): VCC=0 -> jump, so lanes are swapped.
 */
static int knod_bpf_analyze_cfg(struct knod_prog *knod_prog)
{
	int ret;

	/* Build the basic-block CFG, reorder the instruction list into RPO so
	 * every branch is forward (inserting synthetic jumps where a not-taken
	 * successor would no longer be adjacent), then classify each branch by
	 * linear position.  A loop in the program is rejected (-EOPNOTSUPP).
	 */
	ret = knod_bpf_build_cfg(knod_prog);
	if (ret)
		return ret;
	ret = knod_bpf_classify_linear(knod_prog);
	if (ret)
		return ret;

	return knod_bpf_alloc_exec_sregs(knod_prog);
}

/* What every program ends with: publish a verdict for each lane the dispatch
 * covered, then hand the ones that said PASS to the host.  The pass kernel ends
 * the same way and calls this too - it differs only in what it publishes, which
 * it has already put in place before getting here.
 */
static int knod_bpf_emit_epilogue(struct knod_bpf_priv *priv,
				  struct knod_prog *knod_prog)
{
	struct knod_insn_meta *meta;

	/* Fallthrough EXIT: publish a verdict for every in-bounds lane.
	 * Lanes that did not reach BPF_EXIT are forced to XDP_DROP below.
	 */
	meta = kzalloc_obj(*meta, GFP_KERNEL);
	if (!meta)
		return -ENOMEM;
	list_add_tail(&meta->l, &knod_prog->post_insns);

	meta->blob = knod_blob_find(&priv->blob, KNOD_BLOB_EPILOGUE, 0,
				    &meta->blob_size);
	if (!meta->blob) {
		WARN_ON_ONCE(1);
		return -EOPNOTSUPP;
	}

	if (!knod_bpf_pad_shader(priv, meta, &knod_prog->post_insns))
		return -ENOMEM;

	return 0;
}

/* A bounded straight-line suffix is enough to prove a copy's final temporary
 * dead. Unknown successors/CFG/EXEC boundaries start all-live; an ordinary
 * overwrite before that boundary can still kill the old value. EXIT needs
 * only R0. This deliberately does not solve inter-block fixed points. */
static unsigned short knod_bpf_live_after_group(struct knod_prog *prog,
						struct knod_insn_meta *first, unsigned int count)
{
	struct knod_insn_meta *window[64], *meta = first, *next;
	unsigned short live = KNOD_BPF_REGS_LIVE;
	unsigned int n = 0, i;

	for (i = 1; i < count; i++) {
		if (list_is_last(&meta->l, &prog->insns))
			return live;
		meta = list_next_entry(meta, l);
	}
	while (n < ARRAY_SIZE(window) && !list_is_last(&meta->l, &prog->insns)) {
		struct knod_bpf_effect f;

		next = list_next_entry(meta, l);
		if (next->is_merge_point || (next->flags & FLAG_INSN_IS_JUMP_DST) ||
		    next->subprog_idx != first->subprog_idx || next->bpf_insn_idx < 0 ||
		    next->bpf_insn_idx != meta->bpf_insn_idx + 1)
			break;
		window[n++] = next;
		f = knod_bpf_effect(&next->insn);
		if (f.barrier || f.terminal)
			break;
		meta = next;
	}
	while (n)
		live = knod_bpf_live_before(&window[--n]->insn, live);
	return live;
}
/* The existing per-CPU rewrite consumes the increment, not the old load.
 * Drop an adjacent dead +1 or distinct-register ADD. Keep the atomic and all live
 * BPF values; uncertain successors remain all-live in the bounded proof.
 */
static bool knod_bpf_percpu_dead_owned(struct knod_insn_meta *m)
{
	return m->wide_read.owned || m->map_widen.owned || m->read_batch.owned || m->load_pair.owned || m->memory_group_owned || m->loads.count || m->copy.bytes ||
		m->store.bytes || m->packet_region.count || m->conststore.bytes ||
		m->conststore.elide || m->store_hoist.emit ||
		m->store_hoist.elide || m->sink.member;
}

/* Incoming EXEC edges retain their own values; only outgoing barriers
 * stop this current-lane dead-result proof. */
static bool knod_bpf_percpu_result_dead(struct knod_prog *prog,
				      struct knod_insn_meta *store, int reg)
{
	struct knod_insn_meta *m = store, *next;
	unsigned int n;

	for (n = 0; n < 64 && !list_is_last(&m->l, &prog->insns); n++) {
		struct knod_bpf_effect f;

		next = list_next_entry(m, l);
		if (next->subprog_idx != store->subprog_idx ||
		    next->bpf_insn_idx < 0 ||
		    next->bpf_insn_idx != m->bpf_insn_idx + 1)
			return false;
		f = knod_bpf_effect(&next->insn);
		if (f.barrier || (f.uses & (1U << reg)))
			return false;
		if (f.terminal || (f.defs & (1U << reg)))
			return true;
		m = next;
	}
	return false;
}

static noinline bool knod_bpf_forward_dead(struct knod_prog *prog,
				 struct knod_insn_meta *store, int reg);

static void knod_bpf_analyze_percpu_dead(struct knod_bpf_priv *priv,
				       struct knod_prog *prog)
{
	struct knod_insn_meta *load, *alu, *store, *prep;
	int r;

	list_for_each_entry(load, &prog->insns, l)
		load->percpu_dead = false;
	if (priv->isa_version != 10 && priv->isa_version != 11)
		return;
	list_for_each_entry(load, &prog->insns, l) {
		if (load->insn.code != (BPF_LDX | BPF_MEM | BPF_DW) ||
		    load->ptr.type != PTR_TO_MAP_VALUE ||
		    load->bpf_insn_idx < 0 ||
		    list_is_last(&load->l, &prog->insns))
			continue;
		alu = list_next_entry(load, l);
		prep = NULL;
		/* Preserve this exact private LDS read and its completed result.
		 * It prepares the distinct ADD source, never the map address. */
		if (alu->insn.code == (BPF_LDX | BPF_MEM | BPF_DW) &&
		    alu->ptr.type == PTR_TO_STACK && alu->insn.src_reg == 10 &&
		    alu->insn.dst_reg <= 9 &&
		    alu->insn.dst_reg != load->insn.dst_reg &&
		    alu->insn.dst_reg != load->insn.src_reg &&
		    !((alu->sreg.stack_off + alu->insn.off) & 3) &&
		    !alu->is_merge_point && !(alu->flags & FLAG_INSN_IS_JUMP_DST) &&
		    alu->subprog_idx == load->subprog_idx &&
		    alu->bpf_insn_idx == load->bpf_insn_idx + 1 &&
		    !knod_bpf_percpu_dead_owned(alu) &&
		    !list_is_last(&alu->l, &prog->insns)) {
			prep = alu;
			alu = list_next_entry(alu, l);
		}
		if (list_is_last(&alu->l, &prog->insns))
			continue;
		store = list_next_entry(alu, l);
		r = load->insn.dst_reg;
		if (r > 9 || r == load->insn.src_reg ||
		    !((alu->insn.code == (BPF_ALU64 | BPF_ADD | BPF_K) &&
		       alu->insn.imm == 1) ||
		      (alu->insn.code == (BPF_ALU64 | BPF_ADD | BPF_X) &&
		       alu->insn.src_reg <= 9 && alu->insn.src_reg != r)) ||
		    alu->insn.dst_reg != r ||
		    store->insn.code != (BPF_STX | BPF_MEM | BPF_DW) ||
		    store->insn.src_reg != r ||
		    store->insn.dst_reg != load->insn.src_reg ||
		    store->insn.off != load->insn.off ||
		    store->percpu_rmw_add != alu || store->percpu_rmw_swapped ||
		    alu->bpf_insn_idx != load->bpf_insn_idx + 1 + !!prep ||
		    (prep && (alu->insn.code != (BPF_ALU64 | BPF_ADD | BPF_X) ||
			      alu->insn.src_reg != prep->insn.dst_reg)) ||
		    store->bpf_insn_idx != alu->bpf_insn_idx + 1 ||
		    alu->subprog_idx != load->subprog_idx ||
		    store->subprog_idx != load->subprog_idx ||
		    alu->is_merge_point || store->is_merge_point ||
		    ((alu->flags | store->flags) & FLAG_INSN_IS_JUMP_DST) ||
		    knod_bpf_percpu_dead_owned(load) ||
		    knod_bpf_percpu_dead_owned(alu) ||
		    knod_bpf_percpu_dead_owned(store) ||
		    !(knod_bpf_percpu_result_dead(prog, store, r) ||
		      (priv->isa_version == 10 && prog->jit_engine == 1 &&
		       true &&
		       knod_bpf_forward_dead(prog, store, r))))
			continue;
		load->percpu_dead = true;
		alu->percpu_dead = true;
	}
}

/* Original BPF indices are stable across native RPO ordering. */
static struct knod_insn_meta *knod_bpf_unique_index(struct knod_prog *prog,
                                                  int index, int subprog)
{
	struct knod_insn_meta *m, *found = NULL;

	list_for_each_entry(m, &prog->insns, l) {
		if (m->bpf_insn_idx != index)
			continue;
		if (found || m->subprog_idx != subprog)
			return NULL;
		found = m;
	}
	return found;
}

/* Only definitions retained by the selected native lowering kill a value. */
static bool knod_bpf_retained_definition(struct knod_insn_meta *m)
{
	if (m->percpu_dead || m->sr.owner || m->map_lds_head ||
	    m->map_lds_tail || m->map_region.region || m->map_region.producer)
		return false;
	/* A selected wide-read projection writes this original destination. */
	if (m->wide_read.owned)
		return m->wide_read.load && BPF_CLASS(m->insn.code) == BPF_LDX &&
			BPF_MODE(m->insn.code) == BPF_MEM && !m->map_widen.owned &&
			!m->read_batch.owned && !m->load_pair.owned &&
			!m->memory_group_owned && !m->loads.count &&
			!m->copy.bytes && !m->store.bytes && !m->packet_region.count &&
			!m->conststore.bytes && !m->conststore.elide &&
			!m->store_hoist.emit && !m->store_hoist.elide && !m->sink.member;
	return !knod_bpf_percpu_dead_owned(m);
}

/* All forward successors must kill the old register before observing it.
 * Unknown instructions, calls, cycles and a bounded-work overflow fall back.
 */
static noinline bool knod_bpf_forward_dead(struct knod_prog *prog,
                                 struct knod_insn_meta *store, int reg)
{
	int pending[128], seen[64], nr = 1, count = 0, i;

	if (store->bpf_insn_idx < 0 || store->bpf_insn_idx > INT_MAX - 32769)
		return false;
	pending[0] = store->bpf_insn_idx + 1;
	while (nr) {
		struct knod_insn_meta *m, *high;
		struct knod_bpf_effect effect;
		int index = pending[--nr], next;
		unsigned int cls, op;

		for (i = 0; i < count; i++)
			if (seen[i] == index)
				break;
		if (i < count)
			continue;
		if (count == ARRAY_SIZE(seen))
			return false;
		seen[count++] = index;
		m = knod_bpf_unique_index(prog, index, store->subprog_idx);
		if (!m || index <= store->bpf_insn_idx || index > INT_MAX - 32769 ||
		    (m->flags & FLAG_INSN_IS_SUBPROG_START))
			return false;
		next = index + 1;
		cls = BPF_CLASS(m->insn.code);
		op = BPF_OP(m->insn.code);
		if (cls == BPF_JMP || cls == BPF_JMP32) {
			if (m->insn.code == (BPF_JMP | BPF_EXIT)) {
				if (!reg)
					return false;
				continue;
			}
			if (op == BPF_CALL || m->insn.off < 0 ||
			    (cls == BPF_JMP32 && op == BPF_JA))
				return false;
			switch (op) {
			case BPF_JA: break;
			case BPF_JEQ: case BPF_JNE: case BPF_JGT: case BPF_JGE:
			case BPF_JLT: case BPF_JLE: case BPF_JSET:
			case BPF_JSGT: case BPF_JSGE: case BPF_JSLT: case BPF_JSLE:
				if (m->insn.dst_reg == reg ||
				    (BPF_SRC(m->insn.code) == BPF_X && m->insn.src_reg == reg))
					return false;
				if (nr >= ARRAY_SIZE(pending))
					return false;
				pending[nr++] = next;
				break;
			default: return false;
			}
			next += m->insn.off;
		} else {
			effect = knod_bpf_effect(&m->insn);
			if (effect.barrier || (effect.uses & (1U << reg)))
				return false;
			if (m->insn.code == (BPF_LD | BPF_IMM | BPF_DW)) {
				high = knod_bpf_unique_index(prog, next, store->subprog_idx);
				if (!high || high->insn.code || high->insn.dst_reg ||
				    high->insn.src_reg || high->insn.off ||
				    high->is_merge_point || (high->flags & FLAG_INSN_IS_JUMP_DST))
					return false;
				next++;
			}
			if (effect.defs & (1U << reg)) {
				if (!knod_bpf_retained_definition(m))
					return false;
				continue;
			}
		}
		if (nr >= ARRAY_SIZE(pending))
			return false;
		pending[nr++] = next;
	}
	return true;
}

static bool knod_bpf_swapped_owned(struct knod_insn_meta *m)
{
	return knod_bpf_percpu_dead_owned(m) || m->percpu_dead || m->sr.owner ||
		m->map_lds_head || m->map_lds_tail ||
		m->map_region.region || m->map_region.producer;
}

static void knod_bpf_analyze_swapped_dead(struct knod_bpf_priv *priv,
                                         struct knod_prog *prog)
{
	struct knod_insn_meta *load, *alu, *store;

	list_for_each_entry(store, &prog->insns, l)
		store->percpu_delta_direct = false;
	if (priv->isa_version != 10 || prog->jit_engine != 1)
		return;
	list_for_each_entry(load, &prog->insns, l) {
		if (load->insn.code != (BPF_LDX | BPF_MEM | BPF_DW) ||
		    load->ptr.type != PTR_TO_MAP_VALUE || load->insn.dst_reg > 9 ||
		    load->insn.src_reg > 9 || load->insn.dst_reg == load->insn.src_reg ||
		    load->bpf_insn_idx < 0 || load->bpf_insn_idx > INT_MAX - 32771 ||
		    list_is_last(&load->l, &prog->insns))
			continue;
		alu = list_next_entry(load, l);
		if (list_is_last(&alu->l, &prog->insns))
			continue;
		store = list_next_entry(alu, l);
		if (alu->insn.code != (BPF_ALU64 | BPF_ADD | BPF_X) ||
		    alu->insn.src_reg != load->insn.dst_reg || alu->insn.dst_reg > 9 ||
		    alu->insn.dst_reg == alu->insn.src_reg ||
		    alu->insn.dst_reg == load->insn.src_reg ||
		    store->insn.code != (BPF_STX | BPF_MEM | BPF_DW) ||
		    store->insn.src_reg != alu->insn.dst_reg ||
		    store->insn.dst_reg != load->insn.src_reg ||
		    store->insn.off != load->insn.off ||
		    !store->percpu_rmw_swapped || store->percpu_rmw_add != alu ||
		    alu->bpf_insn_idx != load->bpf_insn_idx + 1 ||
		    store->bpf_insn_idx != alu->bpf_insn_idx + 1 ||
		    alu->subprog_idx != load->subprog_idx || store->subprog_idx != load->subprog_idx ||
		    alu->is_merge_point || store->is_merge_point ||
		    ((alu->flags | store->flags) & (FLAG_INSN_IS_JUMP_DST | FLAG_INSN_IS_SUBPROG_START)) ||
		    knod_bpf_swapped_owned(load) || knod_bpf_swapped_owned(alu) || knod_bpf_swapped_owned(store) ||
		    !knod_bpf_forward_dead(prog, store, load->insn.dst_reg) ||
		    !knod_bpf_forward_dead(prog, store, alu->insn.dst_reg))
			continue;
		load->percpu_dead = true;
		alu->percpu_dead = true;
		store->percpu_delta_direct = true;
	}
}

/* A retained AND with a nonnegative immediate clears the upper dword.
 * Keep this local to equality predicates: no arithmetic range inference.
 */
static bool knod_bpf_emit_known_zero_cmp(struct knod_bpf_priv *priv,
					 struct knod_prog *prog,
					 struct knod_insn_meta *meta)
{
	struct knod_insn_meta *prev;
	struct amdgcn_param32 value, imm;

	if (priv->isa_version != 10 || meta->jit_engine != 1 ||
	    (meta->insn.code != (BPF_JMP | BPF_JEQ | BPF_K) &&
	     meta->insn.code != (BPF_JMP | BPF_JNE | BPF_K)) ||
	    meta->insn.imm < 0 || meta->insn.imm > 64 ||
	    meta->insn.dst_reg > 9 || meta->bpf_insn_idx <= 0 ||
	    meta->is_merge_point ||
	    (meta->flags & (FLAG_INSN_IS_JUMP_DST | FLAG_INSN_IS_SUBPROG_START)) ||
	    knod_bpf_swapped_owned(meta) || list_is_first(&meta->l, &prog->insns))
		return false;
	prev = list_prev_entry(meta, l);
	if (prev->bpf_insn_idx != meta->bpf_insn_idx - 1 ||
	    prev->subprog_idx != meta->subprog_idx ||
	    prev->insn.code != (BPF_ALU64 | BPF_AND | BPF_K) ||
	    prev->insn.dst_reg != meta->insn.dst_reg || prev->insn.imm < 0 ||
	    prev->dreg.reg.type != SCALAR_VALUE ||
	    prev->is_merge_point ||
	    (prev->flags & (FLAG_INSN_IS_JUMP_DST | FLAG_INSN_IS_SUBPROG_START)) ||
	    knod_bpf_swapped_owned(prev))
		return false;
	knod_iset32(&imm, meta->insn.imm);
	knod_vset32(&value, meta->insn.dst_reg * 2);
	knod_emit(priv, meta, v_cmp_eq_u32, imm, value);
	return true;
}

/* Analyze the final emission order, after CFG restructuring. Interior jump
 * targets and EXEC merges terminate a group, including synthetic boundaries.
 * The unchanged common base proves aliasing from offsets alone. Packet
 * ownership excludes concurrent writers; shared map values are not eligible.
 */
static void knod_bpf_analyze_copy(struct knod_prog *prog,
				  struct knod_insn_meta *first)
{
	struct knod_insn_meta *ld = first, *st;
	int offsets[16], lo = SHRT_MAX, hi = SHRT_MIN;
	int base = first->insn.src_reg, tmp = first->insn.dst_reg;
	int delta = 0, best = 0, start = 0, last = 0;
	int i, j;

	if (first->insn.code != (BPF_LDX | BPF_MEM | BPF_B) ||
	    first->ptr.type != PTR_TO_PACKET || tmp == base)
		return;

	for (i = 0; i < ARRAY_SIZE(offsets); i++) {
		if (ld->insn.code != (BPF_LDX | BPF_MEM | BPF_B) ||
		    ld->ptr.type != PTR_TO_PACKET ||
		    ld->subprog_idx != first->subprog_idx ||
		    ld->insn.src_reg != base || ld->insn.dst_reg != tmp ||
		    ld->bpf_insn_idx != first->bpf_insn_idx + 2 * i ||
		    (i && (ld->is_merge_point ||
			   (ld->flags & FLAG_INSN_IS_JUMP_DST))) ||
		    list_is_last(&ld->l, &prog->insns))
			break;
		st = list_next_entry(ld, l);
		if (st->insn.code != (BPF_STX | BPF_MEM | BPF_B) ||
		    st->ptr.type != PTR_TO_PACKET ||
		    st->insn.dst_reg != base || st->insn.src_reg != tmp ||
		    st->bpf_insn_idx != ld->bpf_insn_idx + 1 ||
		    st->is_merge_point || (st->flags & FLAG_INSN_IS_JUMP_DST) ||
		    st->percpu_rmw_add ||
		    st->subprog_idx != first->subprog_idx)
			break;
		if (!i)
			delta = (int)st->insn.off - ld->insn.off;
		if ((int)st->insn.off - ld->insn.off != delta)
			break;
		for (j = 0; j < i; j++)
			if (offsets[j] == ld->insn.off)
				return;
		offsets[i] = ld->insn.off;
		lo = min(lo, offsets[i]);
		hi = max(hi, offsets[i]);
		/* The union must be contiguous, with no overlapping source/dest.
		 * Preserve original order between groups; never widen across gaps.
		 */
		if (hi - lo == i && abs(delta) >= i + 1) {
			best = i + 1;
			start = lo;
			last = offsets[i];
		}
		if (list_is_last(&st->l, &prog->insns))
			break;
		ld = list_next_entry(st, l);
	}
	if (best < 2)
		return;

	first->copy = (struct knod_bpf_copy) {
		.src_off = start,
		.dst_off = start + delta,
		.bytes = best,
		.last_byte = last - start,
		.base_reg = base,
		.value_reg = tmp,
		.result_dead = !(knod_bpf_live_after_group(prog, first, 2 * best) &
				 (1U << tmp)),
	};
}

/* Restrict forwarding to exact, private packet accesses. The proof is a BPF
 * value, not a target register or a cached packet window.
 */
static int knod_bpf_packet_width(const struct knod_insn_meta *meta)
{
	if (meta->ptr.type != PTR_TO_PACKET ||
	    BPF_MODE(meta->insn.code) != BPF_MEM)
		return 0;
	switch (BPF_SIZE(meta->insn.code)) {
	case BPF_B:
		return 1;
	case BPF_H:
		return 2;
	case BPF_W:
		return 4;
	case BPF_DW:
		return 8;
	default:
		return 0;
	}
}

static void knod_bpf_analyze_packet_imm(struct knod_prog *prog,
					struct knod_insn_meta *store)
{
	struct knod_insn_meta *meta, *prev;
	int width = knod_bpf_packet_width(store);
	int base = store->insn.dst_reg, off = store->insn.off;
	unsigned int value, count;

	if (!width || width > 4 || store->percpu_rmw_add)
		return;
	if (BPF_CLASS(store->insn.code) == BPF_ST) {
		value = store->insn.imm;
	} else if (BPF_CLASS(store->insn.code) == BPF_STX) {
		if (store->l.prev == &prog->insns || store->is_merge_point ||
		    (store->flags & FLAG_INSN_IS_JUMP_DST))
			return;
		prev = list_prev_entry(store, l);
		if ((prev->insn.code != (BPF_ALU | BPF_MOV | BPF_K) &&
		     prev->insn.code != (BPF_ALU64 | BPF_MOV | BPF_K)) ||
		    prev->insn.dst_reg != store->insn.src_reg ||
		    prev->bpf_insn_idx + 1 != store->bpf_insn_idx ||
		    prev->subprog_idx != store->subprog_idx)
			return;
		value = prev->insn.imm;
	} else {
		return;
	}
	if (width < 4)
		value &= (1U << (8 * width)) - 1;

	/* A bounded local scan keeps compilation linear. Unknown writes may
	 * alias this packet even through a different BPF base, so stop there.
	 * Never carry a value across a helper, control transfer or EXEC merge.
	 */
	prev = store;
	for (count = 0; count < 32; count++, prev = meta) {
		int size;

		if (list_is_last(&prev->l, &prog->insns))
			break;
		meta = list_next_entry(prev, l);
		if (meta->bpf_insn_idx != prev->bpf_insn_idx + 1 ||
		    meta->subprog_idx != store->subprog_idx ||
		    meta->is_merge_point || (meta->flags & FLAG_INSN_IS_JUMP_DST))
			break;

		switch (BPF_CLASS(meta->insn.code)) {
		case BPF_ALU:
		case BPF_ALU64:
			if (meta->insn.dst_reg == base)
				return;
			break;
		case BPF_LDX:
			if (BPF_MODE(meta->insn.code) != BPF_MEM)
				return;
			size = knod_bpf_packet_width(meta);
			if (meta->insn.src_reg == base && size == width &&
			    meta->insn.off == off) {
				meta->packet_imm = value;
				meta->packet_imm_valid = true;
			}
			if (meta->insn.dst_reg == base)
				return;
			break;
		case BPF_ST:
		case BPF_STX:
			size = knod_bpf_packet_width(meta);
			if (!size || meta->percpu_rmw_add ||
			    meta->insn.dst_reg != base ||
			    (meta->insn.off < off + width && off < meta->insn.off + size))
				return;
			break;
		default:
			return;
		}
	}
}

/* Only packet-owned bytes may be combined. No intervening load, register
 * write, helper or control-flow entry is moved across the store group.
 */
static void knod_bpf_analyze_store(struct knod_prog *prog,
				   struct knod_insn_meta *first)
{
	struct knod_insn_meta *meta = first;
	int offsets[16], regs[16], lo = SHRT_MAX, hi = SHRT_MIN;
	int base = first->insn.dst_reg, best = 0, start = 0, i, j;

	for (i = 0; i < ARRAY_SIZE(offsets); i++) {
		if (meta->insn.code != (BPF_STX | BPF_MEM | BPF_B) ||
		    meta->ptr.type != PTR_TO_PACKET || meta->percpu_rmw_add ||
		    meta->insn.dst_reg != base ||
		    meta->subprog_idx != first->subprog_idx ||
		    meta->bpf_insn_idx != first->bpf_insn_idx + i ||
		    (i && (meta->is_merge_point ||
			   (meta->flags & FLAG_INSN_IS_JUMP_DST))))
			break;
		for (j = 0; j < i; j++)
			if (offsets[j] == meta->insn.off)
				goto done;
		offsets[i] = meta->insn.off;
		regs[i] = meta->insn.src_reg;
		lo = min(lo, offsets[i]);
		hi = max(hi, offsets[i]);
		if (hi - lo == i) {
			best = i + 1;
			start = lo;
		}
		if (list_is_last(&meta->l, &prog->insns))
			break;
		meta = list_next_entry(meta, l);
	}
 done:
	if (best < 2)
		return;
	first->store.bytes = best;
	first->store.off = start;
	first->store.base_reg = base;
	for (i = 0; i < best; i++)
		first->store.value_reg[offsets[i] - start] = regs[i];
}

static void knod_bpf_analyze_loads(struct knod_prog *prog,
				   struct knod_insn_meta *first)
{
	struct knod_insn_meta *m = first, *prev = NULL;
	struct knod_memory_load_group group = {};
	for (;;) {
		struct knod_memory_fact f =
		    knod_memory_fact(&m->insn, m->ptr.type == PTR_TO_PACKET
						   ? KNOD_MEMORY_PACKET
						   : KNOD_MEMORY_UNKNOWN);
		bool boundary =
		    m->is_merge_point || (m->flags & FLAG_INSN_IS_JUMP_DST);
		if (prev && (m->bpf_insn_idx != prev->bpf_insn_idx + 1 ||
			     m->subprog_idx != first->subprog_idx))
			break;
		if (!knod_memory_load_append(&group, &f, boundary))
			break;
		if (list_is_last(&m->l, &prog->insns))
			break;
		prev = m;
		m = list_next_entry(m, l);
	}
	if (group.count > 1)
		first->loads = group;
}
static void knod_bpf_analyze_memory(struct knod_prog *prog)
{
	struct knod_insn_meta *meta;
	unsigned int remaining = 0;

	list_for_each_entry(meta, &prog->insns, l) {
		memset(&meta->copy, 0, sizeof(meta->copy));
		memset(&meta->store, 0, sizeof(meta->store));
		meta->packet_imm_valid = false;
		memset(&meta->loads, 0, sizeof(meta->loads));
		if (remaining) {
			remaining--;
			continue;
		}
		knod_bpf_analyze_loads(prog, meta);
		if (meta->loads.count) {
			remaining = meta->loads.count - 1;
			continue;
		}
		knod_bpf_analyze_copy(prog, meta);
		if (meta->copy.bytes) {
			remaining = 2 * meta->copy.bytes - 1;
		} else {
			knod_bpf_analyze_store(prog, meta);
			if (meta->store.bytes)
				remaining = meta->store.bytes - 1;
		}
	}
	list_for_each_entry(meta, &prog->insns, l)
		knod_bpf_analyze_packet_imm(prog, meta);
}

static unsigned int knod_bpf_region_chunks(unsigned int bytes)
{
	unsigned int n = 0;

	while (bytes) {
		bytes -= knod_memory_chunk_width(bytes);
		n++;
	}
	return n;
}

/* This small region accepts only byte memory transfers and immediate MOVs.
 * Byte versions are resolved at compile time; no register cache survives the
 * fused emission. Inputs and outputs must be exact disjoint packet ranges.
 */
static void knod_bpf_plan_packet_region(struct knod_prog *prog,
		struct knod_insn_meta *first, int limit)
{
	struct knod_bpf_packet_region plan = {}, best = {};
	struct knod_region_value value[10];
	struct knod_insn_meta *meta = first, *prev = NULL;
	unsigned int out_mask = 0, in_mask = 0, old_stores = 0;
	unsigned int constants = 0, cover = 0, step, r;
	int base = -1;

	for (r = 0; r < 10; r++)
		value[r] = (struct knod_region_value) {
			.kind = KNOD_REGION_ENTRY, .reg = r,
		};
	for (step = 0; step < 32; step++) {
		const struct bpf_insn *i = &meta->insn;
		bool load = i->code == (BPF_LDX | BPF_MEM | BPF_B);
		bool store = i->code == (BPF_STX | BPF_MEM | BPF_B) ||
			     i->code == (BPF_ST | BPF_MEM | BPF_B);
		bool mov = i->code == (BPF_ALU | BPF_MOV | BPF_K) ||
			   i->code == (BPF_ALU64 | BPF_MOV | BPF_K);
		unsigned int span;
		int pos, width;

		if ((!load && !store && !mov) || meta->percpu_rmw_add ||
		    meta->bpf_insn_idx < 0 || i->dst_reg > 9 ||
		    (prev && (meta->is_merge_point ||
			(meta->flags & FLAG_INSN_IS_JUMP_DST) ||
			meta->subprog_idx != first->subprog_idx ||
			meta->bpf_insn_idx != prev->bpf_insn_idx + 1)))
			break;
		/* Existing groups must be wholly enclosed, not cut in half. */
		if (!cover) {
			span = meta->loads.count ? meta->loads.count :
				meta->copy.bytes ? 2 * meta->copy.bytes : meta->store.bytes;
			cover = span ? span : 1;
			if (meta->copy.bytes)
				old_stores += knod_bpf_region_chunks(meta->copy.bytes);
			else if (meta->store.bytes)
				old_stores += knod_bpf_region_chunks(meta->store.bytes);
			else if (store)
				old_stores++;
		}
		cover--;
		if (mov) {
			if (i->off || i->dst_reg == base)
				break;
			value[i->dst_reg] = (struct knod_region_value) {
				.kind = KNOD_REGION_IMM,
				.imm = BPF_CLASS(i->code) == BPF_ALU ?
					(u64)(u32)i->imm : (u64)(s64)i->imm,
			};
			plan.changed |= 1U << i->dst_reg;
		} else {
			int addr = load ? i->src_reg : i->dst_reg;

			if (addr > 9 || meta->ptr.type != PTR_TO_PACKET ||
			    (plan.changed & (1U << addr)) ||
			    (base >= 0 && addr != base) ||
			    i->off < -limit || i->off >= limit ||
			    (load && (i->imm || i->dst_reg == addr)))
				break;
			base = addr;
			plan.base = base;
			if (load) {
				if (!in_mask)
					plan.input_off = i->off;
				pos = i->off - plan.input_off;
				if (pos < 0) {
					if (-pos + plan.input_bytes > 8)
						break;
					in_mask <<= -pos;
					plan.input_bytes += -pos;
					plan.input_off = i->off;
					pos = 0;
				}
				if (pos >= 8)
					break;
				in_mask |= 1U << pos;
				plan.input_bytes = max_t(unsigned int, plan.input_bytes, pos + 1);
				value[i->dst_reg] = (struct knod_region_value) {
					.kind = KNOD_REGION_PACKET, .off = i->off,
				};
				plan.changed |= 1U << i->dst_reg;
			} else {
				struct knod_region_value v;

				if (BPF_CLASS(i->code) == BPF_ST)
					v = (struct knod_region_value) {
						.kind = KNOD_REGION_IMM, .imm = (u8)i->imm,
					};
				else {
					if (i->src_reg > 9 || i->imm)
						break;
					v = value[i->src_reg];
				}
				if (!out_mask)
					plan.off = i->off;
				pos = i->off - plan.off;
				if (pos < 0) {
					if (-pos + plan.bytes > 16)
						break;
					memmove(plan.output - pos, plan.output,
						plan.bytes * sizeof(plan.output[0]));
					out_mask <<= -pos;
					plan.bytes += -pos;
					plan.off = i->off;
					pos = 0;
				}
				if (pos >= 16 || (out_mask & (1U << pos)))
					break;
				out_mask |= 1U << pos;
				plan.output[pos] = v;
				plan.bytes = max_t(unsigned int, plan.bytes, pos + 1);
				constants += v.kind == KNOD_REGION_IMM;
			}
		}
		/* No moved load can observe an output store, even one later in
		 * the region. Holes must not be filled by speculative reads.
		 */
		if (in_mask && out_mask &&
		    plan.input_off < plan.off + plan.bytes &&
		    plan.off < plan.input_off + plan.input_bytes)
			break;
		width = knod_bpf_region_chunks(plan.bytes);
		if (!cover && plan.bytes >= 8 && in_mask &&
		    out_mask == (1U << plan.bytes) - 1 &&
		    in_mask == (1U << plan.input_bytes) - 1 &&
		    plan.input_off + plan.input_bytes <= limit &&
		    plan.off + plan.bytes <= limit &&
		    old_stores > constants + width) {
			/* Subtract every constant store as a conservative allowance
			 * for the following constant-store pass's possible savings.
			 */
			plan.count = step + 1;
			memcpy(plan.final, value, sizeof(value));
			best = plan;
		}
		prev = meta;
		if (list_is_last(&meta->l, &prog->insns))
			break;
		meta = list_next_entry(meta, l);
	}
	if (!best.count)
		return;
	meta = first;
	for (step = 0; step < best.count; step++) {
		memset(&meta->loads, 0, sizeof(meta->loads));
		memset(&meta->copy, 0, sizeof(meta->copy));
		memset(&meta->store, 0, sizeof(meta->store));
		meta->packet_imm_valid = false;
		if (step + 1 < best.count)
			meta = list_next_entry(meta, l);
	}
	first->packet_region = best;
}

static void knod_bpf_analyze_packet_regions(struct knod_bpf_priv *priv,
					  struct knod_prog *prog)
{
	struct knod_insn_meta *meta;
	unsigned int remaining = 0, span;
	int limit;

	list_for_each_entry(meta, &prog->insns, l)
		memset(&meta->packet_region, 0, sizeof(meta->packet_region));
	if (priv->isa_version != 10 && priv->isa_version != 11)
		return;
	limit = priv->isa_version == 10 ? 2048 : 4096;
	list_for_each_entry(meta, &prog->insns, l) {
		if (remaining) {
			remaining--;
			continue;
		}
		span = meta->loads.count ? meta->loads.count :
			meta->copy.bytes ? 2 * meta->copy.bytes : meta->store.bytes;
		knod_bpf_plan_packet_region(prog, meta, limit);
		remaining = meta->packet_region.count ? meta->packet_region.count - 1 :
			span ? span - 1 : 0;
	}
}

static void knod_bpf_region_byte(struct knod_bpf_priv *priv,
		struct knod_insn_meta *meta, const struct knod_bpf_packet_region *p,
		const struct knod_region_value *v, struct amdgcn_param32 dst)
{
	struct amdgcn_param32 src, shift, bits;

	if (v->kind == KNOD_REGION_IMM) {
		knod_iset32(&src, (u8)v->imm);
		knod_mov32(priv, meta, dst, src);
		return;
	}
	if (v->kind == KNOD_REGION_ENTRY) {
		src = bpf_reg64[v->reg].lo;
		knod_iset32(&shift, 0);
	} else {
		int pos = v->off - p->input_off;

		knod_vset32(&src, KNOD_AMDGPU_TMP_VREG2_LO + pos / 4);
		knod_iset32(&shift, (pos % 4) * 8);
	}
	knod_iset32(&bits, 8);
	knod_bfe32(priv, meta, dst, src, shift, bits);
}

/* Task-local target-independent prototype. No AMD instruction/register IDs. */





static unsigned sr_width(unsigned bytes)
{
	return bytes >= 16   ? 16
	       : bytes >= 12 ? 12
	       : bytes >= 8  ? 8
	       : bytes >= 4  ? 4
	       : bytes >= 2  ? 2
			     : 1;
}
static void sr_set(u64 b[2], unsigned i) { b[i / 64] |= 1ULL << (i % 64); }
static bool sr_has(const u64 b[2], unsigned i) { return (b[i / 64] >> (i % 64)) & 1; }
/* Caller supplies a bounded DAG of semantic selected forms. Every surviving
 * path must reach last; no backward edge, early exit, or interior entry.
 * Scratch is caller-owned to avoid an oversized kernel stack frame.
 */

static bool sr_plan_region(const struct sr_node *nodes, unsigned n, const struct sr_store *stores,
			   unsigned count, struct sr_scratch *scratch, struct sr_plan *out)
{
	struct sr_plan *p = &scratch->draft;
	unsigned i, j, k, slots = 0, end, lo = 65535, hi = 0, flushes = 0;
	memset(out, 0, sizeof(*out));
	memset(scratch, 0, sizeof(*scratch));
	if (!n || n > SR_NODES || count < 2 || count > SR_STORES)
		return false;
	for (i = 0; i < count; i++) {
		const struct sr_store *s = &stores[i];
		if (s->node >= n || !s->width || s->width > 16 || s->width != sr_width(s->width) ||
		    s->base > 9 || s->offset < 0 ||
		    (i && (s->node < stores[i - 1].node || s->base != stores[0].base)))
			return false;
		if (i && s->node == stores[i - 1].node && s->ordinal != stores[i - 1].ordinal + 1)
			return false;
		if ((!i || s->node != stores[i - 1].node) && s->ordinal)
			return false;
		if ((unsigned)s->offset < lo)
			lo = s->offset;
		end = s->offset + s->width;
		if (end > hi)
			hi = end;
		slots += (s->width + 3) / 4;
	}
	if (hi - lo > SR_BYTES || slots > SR_WORDS || stores[count - 1].node != n - 1)
		return false;
	scratch->reached[0] = true;
	sr_set(scratch->dom[0], 0);
	for (i = 0; i < n; i++) {
		const struct sr_node *q = &nodes[i];
		if (!scratch->reached[i] || q->forbidden || q->base_changed ||
		    q->unresolved_overlap_read || (i && q->outside_entry) || q->group_last < i ||
		    q->group_last >= n)
			return false;
		if (i == n - 1) {
			if (q->successor_count)
				return false;
			continue;
		}
		if (!q->successor_count || q->successor_count > 2)
			return false;
		for (j = 0; j < q->successor_count; j++) {
			k = q->successor[j];
			if (k <= i || k >= n)
				return false;
			if (!scratch->reached[k]) {
				memcpy(scratch->dom[k], scratch->dom[i], 16);
				scratch->reached[k] = true;
			} else {
				scratch->dom[k][0] &= scratch->dom[i][0];
				scratch->dom[k][1] &= scratch->dom[i][1];
			}
			sr_set(scratch->dom[k], k);
		}
	}
	sr_set(scratch->post[n - 1], n - 1);
	for (i = n - 1; i-- > 0;) {
		memcpy(scratch->post[i], scratch->post[nodes[i].successor[0]], 16);
		for (j = 1; j < nodes[i].successor_count; j++)
			for (k = 0; k < 2; k++)
				scratch->post[i][k] &= scratch->post[nodes[i].successor[j]][k];
		sr_set(scratch->post[i], i);
	}
	memset(p->byte, 255, sizeof(p->byte));
	slots = 0;
	for (i = 0; i < count; i++) {
		const struct sr_store *s = &stores[i];
		if (!sr_has(scratch->dom[n - 1], s->node) || !sr_has(scratch->post[s->node], n - 1))
			return false;
		p->capture[i] = (struct sr_capture){*s, slots};
		for (j = 0; j < s->width; j++) {
			struct sr_byte *b = &p->byte[s->offset - lo + j];
			if (b->slot != 255)
				return false;
			*b = (struct sr_byte){slots + j / 4, (j % 4) * 8};
		}
		slots += (s->width + 3) / 4;
	}
	for (i = 0; i < hi - lo; i++)
		if (p->byte[i].slot == 255)
			return false;
	for (i = 0; i < hi - lo; i += sr_width(hi - lo - i))
		flushes++;
	if (flushes >= count)
		return false;
	p->count = count;
	p->slots = slots;
	p->bytes = hi - lo;
	p->offset = lo;
	p->base = stores[0].base;
	p->flush_node = n - 1;
	p->flush_count = flushes;
	*out = *p;
	return true;
}
/* Native backend only. Semantic planner sees abstract slots, never VGPRs. */
#define KNOD_SR_CAPTURE_BASE (KNOD_AMDGPU_TMP_VREG3_HI + 1)
static void knod_bpf_sr_emit_normal(struct knod_bpf_priv *priv, struct knod_insn_meta *m, int width,
				    struct amdgcn_param32 src, int base, int off)
{
	switch (width) {
	case 16:
		knod_emit(priv, m, global_store_dwordx4, src, bpf_reg64[base].lo, off);
		break;
	case 12:
		knod_emit(priv, m, global_store_dwordx3, src, bpf_reg64[base].lo, off);
		break;
	case 8:
		knod_emit(priv, m, global_store_dwordx2, src, bpf_reg64[base].lo, off);
		break;
	case 4:
		knod_emit(priv, m, global_store_dword, src, bpf_reg64[base].lo, off);
		break;
	case 2:
		knod_emit(priv, m, global_store_short, src, bpf_reg64[base].lo, off);
		break;
	default:
		knod_emit(priv, m, global_store_byte, src, bpf_reg64[base].lo, off);
		break;
	}
}
/* A full output word may join one source suffix to another's prefix. */
static bool knod_bpf_sr_align_word(struct knod_bpf_priv *priv,
				 struct knod_insn_meta *m, const struct sr_plan *p,
				 int pos, struct amdgcn_param32 dst)
{
	const struct sr_byte *b;
	struct amdgcn_param32 low, high, shift;
	int first, i, low_reg, high_reg;

	if (priv->isa_version != 10 || m->jit_engine != 1 ||
	    pos < 0 || p->bytes < 4 || pos > p->bytes - 4 ||
	    !p->slots || p->slots > SR_WORDS)
		return false;
	b = &p->byte[pos];
	if (b[0].shift != 8 && b[0].shift != 16 && b[0].shift != 24)
		return false;
	first = 4 - b[0].shift / 8;
	if (b[0].slot >= p->slots || b[first].slot >= p->slots)
		return false;
	for (i = 0; i < 4; i++) {
		int slot = i < first ? b[0].slot : b[first].slot;
		int bits = i < first ? b[0].shift + i * 8 : (i - first) * 8;

		if (b[i].slot != slot || b[i].shift != bits)
			return false;
	}
	low_reg = KNOD_SR_CAPTURE_BASE + b[0].slot;
	high_reg = KNOD_SR_CAPTURE_BASE + b[first].slot;
	if (dst.type != AMDGCN_PARAM_TYPE_VGPR ||
	    dst.v == low_reg || dst.v == high_reg)
		return false;
	knod_vset32(&low, low_reg);
	knod_vset32(&high, high_reg);
	knod_iset32(&shift, b[0].shift);
	knod_emit(priv, m, v_alignbit_b32, dst, high, low, shift);
	return true;
}

static void knod_bpf_sr_flush(struct knod_bpf_priv *priv, struct knod_insn_meta *m,
			      const struct sr_plan *p)
{
	struct amdgcn_param32 dst, src, tmp, shift, bits;
	int off, width, j, run;
	knod_vset32(&tmp, KNOD_AMDGPU_TMP_VREG3_HI);
	for (off = 0; off < p->bytes; off += width) {
		width = sr_width(p->bytes - off);
		for (j = 0; j < width; j += run) {
			const struct sr_byte *b = &p->byte[off + j];
			run = 1;
			while (j + run < width && (j % 4) + run < 4 && b->shift + run * 8 < 32 &&
			       p->byte[off + j + run].slot == b->slot &&
			       p->byte[off + j + run].shift == b->shift + run * 8)
				run++;
			knod_vset32(&dst, KNOD_AMDGPU_TMP_VREG0_LO + j / 4);
			if (!(j % 4) && j + 4 <= width &&
			    knod_bpf_sr_align_word(priv, m, p, off + j, dst)) {
				run = 4;
				continue;
			}
			knod_vset32(&src, KNOD_SR_CAPTURE_BASE + b->slot);
			if (run == 4)
				knod_mov32(priv, m, dst, src);
			else {
				knod_iset32(&shift, b->shift);
				knod_iset32(&bits, run * 8);
				knod_bfe32(priv, m, j % 4 ? tmp : dst, src, shift, bits);
				if (j % 4) {
					knod_iset32(&shift, (j % 4) * 8);
					knod_emit(priv, m, v_lshl_or_b32, dst, tmp, shift, dst);
				}
			}
		}
		knod_vset32(&src, KNOD_AMDGPU_TMP_VREG0_LO);
		knod_bpf_sr_emit_normal(priv, m, width, src, p->base, p->offset + off);
	}
}
/* Source comes from the existing packer exactly when its original store
 * would issue. A descriptor mismatch is a compile failure, never partial
 * fallback after earlier captures. Caller must discard failed generation.
 */
static bool knod_bpf_sr_store(struct knod_bpf_priv *priv, struct knod_insn_meta *m,
			      const struct sr_plan *plan, unsigned descriptor, int width,
			      struct amdgcn_param32 src, int base, int off)
{
	const struct sr_capture *c;
	struct amdgcn_param32 dst, word;
	int j;
	if (!plan) {
		knod_bpf_sr_emit_normal(priv, m, width, src, base, off);
		return true;
	}
	if (descriptor >= plan->count || src.type != AMDGCN_PARAM_TYPE_VGPR || src.v < 0 ||
	    src.v + (width + 3) / 4 > KNOD_SR_CAPTURE_BASE)
		return false;
	c = &plan->capture[descriptor];
	if (c->store.width != width || c->store.base != base || c->store.offset != off)
		return false;
	for (j = 0; j < (width + 3) / 4; j++) {
		knod_vset32(&dst, KNOD_SR_CAPTURE_BASE + c->first_slot + j);
		knod_vset32(&word, src.v + j);
		knod_mov32(priv, m, dst, word);
	}
	if (descriptor + 1 == plan->count)
		knod_bpf_sr_flush(priv, m, plan);
	return true;
}
/* Bounded selected-form adapter. No decoded AMD instructions participate. */
struct knod_sr_workspace {
	struct sr_node nodes[SR_NODES];
	struct sr_store stores[SR_STORES];
	struct knod_insn_meta *meta[SR_NODES];
	struct sr_scratch scratch;
	struct sr_plan plan;
};
static bool knod_sr_middle(struct knod_insn_meta *m)
{
	const struct bpf_insn *i = &m->insn;
	int cls = BPF_CLASS(i->code), op = BPF_OP(i->code);
	if (m->sink.member || m->store_hoist.elide || m->conststore.elide || m->packet_imm_valid)
		return true;
	if (cls == BPF_ALU64 && (op == BPF_ADD || op == BPF_AND || op == BPF_XOR || op == BPF_RSH))
		return true;
	if ((cls == BPF_ALU || cls == BPF_ALU64) && op == BPF_MOV && !i->off)
		return true;
	if (cls == BPF_ALU && op == BPF_END && (i->imm == 16 || i->imm == 32 || i->imm == 64))
		return true;
	if (i->code == (BPF_LD | BPF_IMM | BPF_DW) && !i->src_reg)
		return true;
	if (i->code == (BPF_JMP | BPF_JEQ | BPF_K) && i->off > 0 &&
	    m->branch_type == KNOD_BR_FORWARD_SKIP)
		return true;
	if (i->code == (BPF_LDX | BPF_MEM | BPF_DW) && m->ptr.type == PTR_TO_STACK &&
	    !((m->sreg.stack_off + i->off) & 3))
		return true;
	if (i->code == (BPF_LDX | BPF_MEM | BPF_B) && m->ptr.type == PTR_TO_STACK)
		return true;
	return i->code == (BPF_LDX | BPF_MEM | BPF_W) && m->ptr.type == PTR_TO_MAP_VALUE;
}
static bool knod_sr_append(struct knod_sr_workspace *w, unsigned *count, unsigned node, int base,
			   int off, unsigned bytes)
{
	unsigned j, width, ordinal = 0;
	if (off < 0 || !bytes || off + bytes > 2048)
		return false;
	for (j = 0; j < bytes; j += width) {
		width = sr_width(bytes - j);
		if (*count == SR_STORES)
			return false;
		w->stores[(*count)++] = (struct sr_store){node, width, base, ordinal++, off + j};
	}
	return true;
}
/* Fully prepare the slice in private workspace; do not set live meta flags
 * until the DAG, alias, selected-form and capacity checks all succeed. */
static bool knod_sr_try(struct knod_prog *prog, struct knod_insn_meta *first,
			struct knod_sr_workspace *w)
{
	struct knod_insn_meta *m = first, *other;
	unsigned n = 0, count = 0, opaque = 0, i, j;
	int base = first->packet_region.base;
	if (!first->packet_region.count || first->sr.owner)
		return false;
	memset(w, 0, sizeof(*w));
	while (n < SR_NODES) {
		struct sr_node *node = &w->nodes[n];
		const struct bpf_insn *in = &m->insn;
		struct knod_bpf_effect effect = knod_bpf_effect(in);
		bool output = false,
		     extension = n && w->meta[n - 1]->insn.code == (BPF_LD | BPF_IMM | BPF_DW);
		if (m->sr.owner || m->subprog_idx != first->subprog_idx || m->bpf_insn_idx < 0 ||
		    (n && m->bpf_insn_idx != w->meta[n - 1]->bpf_insn_idx + 1) ||
		    m->map_widen.owned || m->wide_read.owned || m->read_batch.owned || m->load_pair.owned || m->percpu_dead || m->percpu_rmw_add)
			return false;
		w->meta[n] = m;
		node->group_last = n;
		if (!extension && (effect.defs & (1U << base)))
			return false;
		if (opaque) {
			opaque--;
		} else if (m->packet_region.count) {
			if (n || m->packet_region.base != base ||
			    m->packet_region.count > SR_NODES - n)
				return false;
			opaque = m->packet_region.count - 1;
			node->group_last = n + opaque;
			if (!knod_sr_append(w, &count, n, base, m->packet_region.off,
					    m->packet_region.bytes))
				return false;
			output = true;
		} else if (m->sink.bytes && !m->sink.forward) {
			if (m->sink.base != base ||
			    !knod_sr_append(w, &count, n, base, m->sink.off, m->sink.bytes))
				return false;
			output = true;
		} else if (m->store_hoist.emit) {
			if (m->store_hoist.base != base ||
			    !knod_sr_append(w, &count, n, base, m->store_hoist.off, 8))
				return false;
			output = true;
		} else if (m->conststore.bytes) {
			if (m->conststore.base != base ||
			    !knod_sr_append(w, &count, n, base, m->conststore.off,
					    m->conststore.bytes))
				return false;
			output = true;
		} else if (m->memory_group_owned || m->loads.count || m->copy.bytes ||
			   m->store.bytes)
			return false;
		else if (BPF_CLASS(in->code) == BPF_STX && BPF_MODE(in->code) == BPF_MEM &&
			 m->ptr.type == PTR_TO_PACKET && !m->sink.member && !m->store_hoist.elide &&
			 !m->conststore.elide) {
			if (in->dst_reg != base ||
			    !knod_sr_append(w, &count, n, base, in->off, knod_bpf_packet_width(m)))
				return false;
			output = true;
		} else if (extension) {
			if (in->code)
				return false;
		} else if (!knod_sr_middle(m))
			return false;
		/* Raw group members are represented by the selected entry emitter. */
		if (n)
			w->nodes[n - 1].successor_count = w->nodes[n - 1].successor_count ?: 1;
		if (n && !w->nodes[n - 1].successor[0])
			w->nodes[n - 1].successor[0] = n;
		if (output && !opaque && count > 1) {
			bool valid = true;
			/* Resolve raw forward edges and detect all outside incoming jumps. */
			for (i = 0; i <= n; i++) {
				struct knod_insn_meta *q = w->meta[i];
				w->nodes[i].successor_count = i == n ? 0 : 1;
				w->nodes[i].successor[0] = i + 1;
				if (q->insn.code == (BPF_JMP | BPF_JEQ | BPF_K)) {
					int target = q->bpf_insn_idx + 1 + q->insn.off;
					for (j = i + 1;
					     j <= n && w->meta[j]->bpf_insn_idx != target; j++)
						;
					if (j > n || q->merge_point != w->meta[j]) {
						valid = false;
						break;
					}
					w->nodes[i].successor_count = 2;
					w->nodes[i].successor[1] = j;
				}
			}
			list_for_each_entry(other, &prog->insns, l)
			{
				int cls = BPF_CLASS(other->insn.code),
				    op = BPF_OP(other->insn.code);
				s64 target;
				if (cls != BPF_JMP && cls != BPF_JMP32)
					continue;
				if (op == BPF_CALL || op == BPF_EXIT)
					continue;
				if (other->bpf_insn_idx >= first->bpf_insn_idx &&
				    other->bpf_insn_idx <= m->bpf_insn_idx)
					continue;
				/* Synthetic jumps have no raw BPF displacement. */
				if (other->bpf_insn_idx < 0) {
					if (!other->merge_point) {
						valid = false;
						break;
					}
					target = other->merge_point->bpf_insn_idx;
				} else {
					/* JA32 carries its displacement in imm, not off. */
					target = (s64)other->bpf_insn_idx + 1 +
						 (cls == BPF_JMP32 && op == BPF_JA ?
						  other->insn.imm : other->insn.off);
				}
				if (target > first->bpf_insn_idx && target <= m->bpf_insn_idx) {
					valid = false;
					break;
				}
			}
			if (valid && sr_plan_region(w->nodes, n + 1, w->stores, count, &w->scratch,
						    &w->plan)) {
				first->sr.plan = w->plan;
				for (i = 0; i < count; i++) {
					struct knod_insn_meta *q = w->meta[w->stores[i].node];
					if (!q->sr.count)
						q->sr.first = i;
					q->sr.owner = first;
					q->sr.count++;
				}
				return true;
			}
		}
		if (list_is_last(&m->l, &prog->insns))
			break;
		m = list_next_entry(m, l);
		n++;
	}
	return false;
}
static void knod_bpf_analyze_store_regions(struct knod_bpf_priv *priv, struct knod_prog *prog)
{
	struct knod_insn_meta *m;
	struct knod_sr_workspace *w;
	list_for_each_entry(m, &prog->insns, l) memset(&m->sr, 0, sizeof(m->sr));
	if (priv->isa_version != 10)
		return;
	w = kzalloc(sizeof(*w), GFP_KERNEL);
	if (!w)
		return;
	list_for_each_entry(m, &prog->insns, l) knod_sr_try(prog, m, w);
	kfree(w);
}
static void knod_bpf_packet_store(struct knod_bpf_priv *priv, struct knod_insn_meta *m, int width,
				  struct amdgcn_param32 src, int base, int off)
{
	const struct sr_plan *p = m->sr.owner ? &m->sr.owner->sr.plan : NULL;
	if (!knod_bpf_sr_store(priv, m, p, m->sr.first + m->sr.cursor, width, src, base, off))
		m->sr.error = true;
	if (p)
		m->sr.cursor++;
}

static void knod_bpf_emit_region_inputs(struct knod_bpf_priv *priv,
				       struct knod_insn_meta *first,
				       const struct knod_bpf_packet_region *p)
{
	struct amdgcn_param32 data;
	unsigned int off, width, input_chunk = 0;

	/* Input bytes occupy v26/v27, packed output v22..25, scratch v29.
	 * Full contiguous input range is at most eight bytes, without holes.
	 */
	for (off = 0; off < p->input_bytes; off += width) {
		width = knod_memory_chunk_width(p->input_bytes - off);
		knod_vset32(&data, input_chunk < 2 ?
			KNOD_AMDGPU_TMP_VREG2_LO + input_chunk :
			KNOD_AMDGPU_TMP_VREG3_HI);
		input_chunk++;
		if (width == 8)
			knod_emit(priv, first, global_load_dwordx2, data, bpf_reg64[p->base].lo, p->input_off + off);
		else if (width == 4)
			knod_emit(priv, first, global_load_dword, data, bpf_reg64[p->base].lo, p->input_off + off);
		else if (width == 2)
			knod_emit(priv, first, global_load_ushort, data, bpf_reg64[p->base].lo, p->input_off + off);
		else
			knod_emit(priv, first, global_load_ubyte, data, bpf_reg64[p->base].lo, p->input_off + off);
	}
}

static unsigned int knod_bpf_emit_packet_region(struct knod_bpf_priv *priv,
					       struct knod_insn_meta *first)
{
	const struct knod_bpf_packet_region *p = &first->packet_region;
	struct amdgcn_param32 data, byte, shift, zero;
	struct amdgcn_param64 imm;
	unsigned int off, width, n;
	u8 packed[4] = {};

	if (!p->count)
		return 0;
	if (first->map_region.producer) {
		if (!first->map_region.staged ||
		    first->map_region.producer->map_region.region != first) {
			first->map_region.error = true;
			return p->count;
		}
	} else {
		knod_bpf_emit_region_inputs(priv, first, p);
	}
	/* Pack each output word's available prefix while packet reads run.
	 * ENTRY and IMM values do not depend on those reads. Stop each word
	 * at its first packet byte, retaining the partial word for completion.
	 * A seven-byte input uses v29 for its pending byte tail: that aliases
	 * the packing scratch, so retain the original schedule in that case.
	 * No BPF register changes or packet stores occur before the wait.
	 */
	if (priv->isa_version == 10 && first->jit_engine == 1 &&
	    p->input_bytes != 7) {
		knod_vset32(&byte, KNOD_AMDGPU_TMP_VREG3_HI);
		for (n = 0; n < p->bytes; n += 4) {
			knod_vset32(&data, KNOD_AMDGPU_TMP_VREG0_LO + n / 4);
			for (off = n; off < p->bytes && off < n + 4; off++) {
				if (p->output[off].kind != KNOD_REGION_ENTRY &&
				    p->output[off].kind != KNOD_REGION_IMM)
					break;
				if (!(off & 3)) {
					knod_bpf_region_byte(priv, first, p, &p->output[off], data);
				} else {
					knod_bpf_region_byte(priv, first, p, &p->output[off], byte);
					knod_iset32(&shift, (off & 3) * 8);
					knod_emit(priv, first, v_lshl_or_b32,
						  data, byte, shift, data);
				}
				packed[n / 4]++;
			}
		}
	}
	if (!first->map_region.producer)
		knod_wait_vmcnt(priv, first);
	if (p->input_bytes == 3 || p->input_bytes == 7) {
		/* A 2+1 tail uses distinct read destinations until completion. */
		knod_vset32(&data, KNOD_AMDGPU_TMP_VREG2_LO +
			(p->input_bytes == 7));
		knod_vset32(&byte, p->input_bytes == 3 ?
			KNOD_AMDGPU_TMP_VREG2_LO + 1 : KNOD_AMDGPU_TMP_VREG3_HI);
		knod_iset32(&shift, 16);
		knod_lshlrev32(priv, first, byte, shift, byte);
		knod_or32(priv, first, data, data, byte);
	}
	knod_iset32(&zero, 0);
	knod_vset32(&byte, KNOD_AMDGPU_TMP_VREG3_HI);
	for (off = 0; off < p->bytes; off++) {
		if ((off & 3) < packed[off / 4])
			continue;
		knod_vset32(&data, KNOD_AMDGPU_TMP_VREG0_LO + off / 4);
		if (!(off & 3)) {
			knod_bpf_region_byte(priv, first, p, &p->output[off], data);
		} else {
			knod_bpf_region_byte(priv, first, p, &p->output[off], byte);
			knod_iset32(&shift, (off & 3) * 8);
			knod_emit(priv, first, v_lshl_or_b32,
				  data, byte, shift, data);
		}
	}
	/* Every entry-register input was packed before any BPF register is
	 * changed. Final packet-loaded bytes come from protected input words.
	 */
	for (n = 0; n < 10; n++) {
		if (!(p->changed & (1U << n)))
			continue;
		if (p->final[n].kind == KNOD_REGION_IMM) {
			knod_iset64(&imm, p->final[n].imm);
			knod_mov64(priv, first, bpf_reg64[n], imm);
		} else {
			knod_bpf_region_byte(priv, first, p, &p->final[n], bpf_reg64[n].lo);
			knod_mov32(priv, first, bpf_reg64[n].hi, zero);
		}
	}
	for (off = 0; off < p->bytes; off += width) {
		width = knod_memory_chunk_width(p->bytes - off);
		knod_vset32(&data, KNOD_AMDGPU_TMP_VREG0_LO + off / 4);
		if (off & 3) {
			knod_iset32(&shift, (off & 3) * 8);
			knod_lshrrev32(priv, first, byte, shift, data);
			data = byte;
		}
		knod_bpf_packet_store(priv, first, width, data, p->base, p->off + off);
	}
	return p->count;
}

/* Pure BPF proof. max_off is a target capability, not an opcode. Stores are
 * delayed only across MOVs and other disjoint constant packet stores. */
static void knod_bpf_conststore_group(struct knod_prog *prog,
		struct knod_insn_meta *first, int max_off)
{
	struct knod_insn_meta *meta = first, *prev = NULL, *stores[16];
	struct knod_bpf_conststore plan = {}, best = {};
	u64 values[11] = {};
	unsigned int known = 0, n = 0, best_n = 0, step;
	int base = -1;

	for (step = 0; step < 32; step++) {
		const struct bpf_insn *i = &meta->insn;
		unsigned int cls = BPF_CLASS(i->code);
		u64 value;
		int width, off, j, count, left;

		if (meta->memory_group_owned || meta->conststore.elide ||
		    meta->conststore.bytes || meta->bpf_insn_idx < 0 ||
		    (prev && (meta->is_merge_point ||
			(meta->flags & FLAG_INSN_IS_JUMP_DST) ||
			meta->subprog_idx != first->subprog_idx ||
			meta->bpf_insn_idx != prev->bpf_insn_idx + 1)))
			break;
		if (i->code == (BPF_ALU | BPF_MOV | BPF_K) ||
		    i->code == (BPF_ALU64 | BPF_MOV | BPF_K) ||
		    i->code == (BPF_ALU | BPF_MOV | BPF_X) ||
		    i->code == (BPF_ALU64 | BPF_MOV | BPF_X)) {
			if (i->off || i->dst_reg > 9 || i->dst_reg == base)
				break;
			if (BPF_SRC(i->code) == BPF_K) {
				value = (u64)(s64)i->imm;
			} else {
				if (i->src_reg > 10)
					break;
				if (!(known & (1U << i->src_reg))) {
					known &= ~(1U << i->dst_reg);
					goto next;
				}
				value = values[i->src_reg];
			}
			values[i->dst_reg] = cls == BPF_ALU ? (u32)value : value;
			known |= 1U << i->dst_reg;
			goto next;
		}
		width = knod_bpf_packet_width(meta);
		if (!width || (cls != BPF_ST && cls != BPF_STX) ||
		    meta->percpu_rmw_add || i->dst_reg > 9 || n == 16)
			break;
		if (cls == BPF_ST)
			value = (u64)(s64)i->imm;
		else {
			if (i->src_reg > 10 || !(known & (1U << i->src_reg)))
				break;
			value = values[i->src_reg];
		}
		if (base < 0) {
			base = i->dst_reg;
			plan.base = base;
			plan.off = i->off;
		}
		if (base != i->dst_reg || plan.bytes + width > 16)
			break;
		off = i->off;
		if (plan.bytes && off != plan.off + plan.bytes &&
		    off + width != plan.off)
			break;
		if (off < plan.off) {
			memmove(plan.data + width, plan.data, plan.bytes);
			plan.off = off;
		}
		for (j = 0; j < width; j++)
			plan.data[off - plan.off + j] = value >> (8 * j);
		plan.bytes += width;
		stores[n++] = meta;
		for (count = 0, left = plan.bytes; left; count++)
			left -= knod_memory_chunk_width(left);
		if ((unsigned int)count < n && plan.off >= -max_off &&
		    plan.off + plan.bytes - 1 < max_off) {
			best = plan;
			best_n = n;
		}
next:
		if (list_is_last(&meta->l, &prog->insns))
			break;
		prev = meta;
		meta = list_next_entry(meta, l);
	}
	if (!best_n)
		return;
	for (n = 0; n + 1 < best_n; n++)
		stores[n]->conststore.elide = true;
	stores[best_n - 1]->conststore = best;
}

static void knod_bpf_analyze_conststores(struct knod_bpf_priv *priv,
		struct knod_prog *prog)
{
	struct knod_insn_meta *meta;
	unsigned int remaining = 0;
	int limit;

	list_for_each_entry(meta, &prog->insns, l) {
		memset(&meta->conststore, 0, sizeof(meta->conststore));
		if (!remaining)
			remaining = meta->packet_region.count ? meta->packet_region.count :
				meta->loads.count ? meta->loads.count :
				meta->copy.bytes ? 2 * meta->copy.bytes : meta->store.bytes;
		meta->memory_group_owned = remaining != 0;
		if (remaining)
			remaining--;
	}
	if (priv->isa_version != 10 && priv->isa_version != 11)
		return;
	limit = priv->isa_version == 10 ? 2048 : 4096;
	list_for_each_entry(meta, &prog->insns, l)
		knod_bpf_conststore_group(prog, meta, limit);
}

static void knod_bpf_emit_conststore(struct knod_bpf_priv *priv,
		struct knod_insn_meta *meta)
{
	const struct knod_bpf_conststore *p = &meta->conststore;
	struct amdgcn_param32 data, imm;
	int off, width, j, k, reg = KNOD_AMDGPU_TMP_VREG0_LO;

	/* Distinct temporary words remain valid through every store. */
	for (off = 0; off < p->bytes; off += width) {
		width = knod_memory_chunk_width(p->bytes - off);
		for (j = 0; j < width; j += 4) {
			u32 value = 0;
			for (k = 0; k < 4 && j + k < width; k++)
				value |= (u32)p->data[off + j + k] << (8 * k);
			knod_vset32(&data, reg + j / 4);
			knod_iset32(&imm, value);
			knod_mov32(priv, meta, data, imm);
		}
		knod_vset32(&data, reg);
		knod_bpf_packet_store(priv, meta, width, data, p->base, p->off + off);
		reg += (width + 3) / 4;
	}
}

/* Move only a later adjacent dword store to an earlier dword store. Every
 * other BPF instruction remains in place; no register value is kept cached.
 */
static bool knod_bpf_hoist_owned(const struct knod_insn_meta *m)
{
	return m->memory_group_owned || m->conststore.bytes ||
		m->conststore.elide || m->store_hoist.emit || m->store_hoist.elide;
}

static void knod_bpf_analyze_store_hoist_group(struct knod_prog *prog,
		struct knod_insn_meta *first, int limit)
{
	struct knod_insn_meta *meta = first, *prev;
	struct { int off, width; } accesses[32];
	unsigned int defs = 0, n = 0, step;
	int base = first->insn.dst_reg, start = first->insn.off;

	if (knod_bpf_hoist_owned(first) || first->percpu_rmw_add ||
	    first->insn.code != (BPF_STX | BPF_MEM | BPF_W) ||
	    first->ptr.type != PTR_TO_PACKET || base > 9 ||
	    first->insn.src_reg > 10 || first->bpf_insn_idx < 0)
		return;
	for (step = 0; step < 32; step++) {
		const struct bpf_insn *i;
		struct knod_bpf_effect effect;
		int width, lo, j;

		if (list_is_last(&meta->l, &prog->insns))
			return;
		prev = meta;
		meta = list_next_entry(meta, l);
		i = &meta->insn;
		if (knod_bpf_hoist_owned(meta) || meta->percpu_rmw_add ||
		    meta->is_merge_point || (meta->flags & FLAG_INSN_IS_JUMP_DST) ||
		    meta->subprog_idx != first->subprog_idx ||
		    meta->bpf_insn_idx != prev->bpf_insn_idx + 1)
			return;
		effect = knod_bpf_effect(i);
		if (effect.barrier || effect.terminal ||
		    (effect.defs & (1U << base)))
			return;
		width = knod_bpf_packet_width(meta);
		if (i->code == (BPF_STX | BPF_MEM | BPF_W) &&
		    width == 4 && i->dst_reg == base && i->src_reg <= 10 &&
		    !(defs & (1U << i->src_reg)) &&
		    (i->off == start + 4 || i->off + 4 == start)) {
			lo = min(start, (int)i->off);
			if (lo < -limit || lo + 7 >= limit)
				return;
			for (j = 0; (unsigned int)j < n; j++)
				if (i->off < accesses[j].off + accesses[j].width &&
				    accesses[j].off < i->off + 4)
					return;
			first->store_hoist.off = lo;
			first->store_hoist.base = base;
			first->store_hoist.reg[0] = start == lo ?
				first->insn.src_reg : i->src_reg;
			first->store_hoist.reg[1] = start == lo ?
				i->src_reg : first->insn.src_reg;
			first->store_hoist.emit = true;
			meta->store_hoist.elide = true;
			return;
		}
		switch (BPF_CLASS(i->code)) {
		case BPF_ALU:
		case BPF_ALU64:
			break;
		case BPF_LDX:
			if (BPF_MODE(i->code) != BPF_MEM)
				return;
			if (meta->ptr.type == PTR_TO_STACK)
				break;
			if (!width || i->src_reg != base)
				return;
			accesses[n].off = i->off;
			accesses[n++].width = width;
			break;
		case BPF_ST:
		case BPF_STX:
			if (!width || i->dst_reg != base)
				return;
			accesses[n].off = i->off;
			accesses[n++].width = width;
			break;
		default:
			return;
		}
		defs |= effect.defs;
	}
}

static void knod_bpf_analyze_store_hoists(struct knod_bpf_priv *priv,
		struct knod_prog *prog)
{
	struct knod_insn_meta *meta;
	int limit;

	list_for_each_entry(meta, &prog->insns, l)
		memset(&meta->store_hoist, 0, sizeof(meta->store_hoist));
	if (priv->isa_version != 10 && priv->isa_version != 11)
		return;
	limit = priv->isa_version == 10 ? 2048 : 4096;
	list_for_each_entry(meta, &prog->insns, l)
		knod_bpf_analyze_store_hoist_group(prog, meta, limit);
}

static void knod_bpf_emit_store_hoist(struct knod_bpf_priv *priv,
		struct knod_insn_meta *meta)
{
	struct amdgcn_param32 lo, hi;

	knod_vset32(&lo, KNOD_AMDGPU_TMP_VREG0_LO);
	knod_vset32(&hi, KNOD_AMDGPU_TMP_VREG0_HI);
	knod_mov32(priv, meta, lo, bpf_reg64[meta->store_hoist.reg[0]].lo);
	knod_mov32(priv, meta, hi, bpf_reg64[meta->store_hoist.reg[1]].lo);
	knod_bpf_packet_store(priv, meta, 8, lo, meta->store_hoist.base, meta->store_hoist.off);
}

/* Exact selected-emission ledger; unknown forms terminate the region. */
static bool knod_bpf_sink_transparent(struct knod_insn_meta *m)
{
	const struct bpf_insn *i = &m->insn;

	if (m->percpu_rmw_add || m->loads.count || m->copy.bytes ||
	    m->packet_region.count ||
	    m->conststore.bytes || m->conststore.elide || m->store_hoist.elide)
		return false;
	if (m->store_hoist.emit)
		return true; /* Existing emitter writes only v22/v23. */
	if (m->memory_group_owned || m->store.bytes)
		return false;
	if (!i->off && (i->code == (BPF_ALU | BPF_MOV | BPF_K) ||
	    i->code == (BPF_ALU64 | BPF_MOV | BPF_K) ||
	    i->code == (BPF_ALU | BPF_MOV | BPF_X) ||
	    i->code == (BPF_ALU64 | BPF_MOV | BPF_X)))
		return true;
	if (i->code == (BPF_ALU64 | BPF_ADD | BPF_K))
		return true; /* r64[0] = v22/v23. */
	if (i->code == (BPF_ALU | BPF_END | BPF_TO_BE) &&
	    (i->imm == 16 || i->imm == 32))
		return true; /* BPF destination and scalar selector only. */
	return (i->code == (BPF_LDX | BPF_MEM | BPF_W) ||
		i->code == (BPF_LDX | BPF_MEM | BPF_DW)) &&
		m->ptr.type == PTR_TO_STACK &&
		!((m->sreg.stack_off + i->off) & 3);
}

static bool knod_bpf_sink_store(struct knod_insn_meta *m)
{
	return !m->percpu_rmw_add && !m->store_hoist.emit &&
		!m->store_hoist.elide && !m->conststore.bytes &&
		!m->conststore.elide && !m->packet_region.count &&
		!m->copy.bytes && !m->loads.count &&
		m->ptr.type == PTR_TO_PACKET &&
		(m->insn.code == (BPF_STX | BPF_MEM | BPF_B) ||
		 m->insn.code == (BPF_STX | BPF_MEM | BPF_H) ||
		 m->insn.code == (BPF_STX | BPF_MEM | BPF_W));
}

/* BPF versions and byte provenance are independent of physical registers.
 * The accepted target ledger reserves at most two words, v26/v27, later.
 */
static void knod_bpf_sink_group(struct knod_prog *prog,
		struct knod_insn_meta *first, int limit)
{
	struct knod_insn_meta *m = first, *prev = NULL, *stores[16];
	unsigned short version[11] = {}, stored_version[16];
	int offsets[16], widths[16], regs[16], costs = 0, group_left = 0;
	int n = 0, step, base = first->insn.dst_reg, lo = SHRT_MAX, hi = SHRT_MIN;
	int disjoint_off[32], disjoint_n = 0;

	if (!knod_bpf_sink_store(first) || first->sink.member || base > 9)
		return;
	for (step = 0; step < 32; step++) {
		struct knod_bpf_effect effect;
		int width, j, k, covered = 0, chunks = 0, left;
		struct knod_bpf_sink plan = {};
		int capture_reg[2], capture_version[2], capture_at[2], captures = 0;

		if (m->sink.member || m->bpf_insn_idx < 0 ||
		    (prev && (m->is_merge_point || (m->flags & FLAG_INSN_IS_JUMP_DST) ||
			m->subprog_idx != first->subprog_idx ||
			m->bpf_insn_idx != prev->bpf_insn_idx + 1)))
			return;
		effect = knod_bpf_effect(&m->insn);
		if (effect.barrier || effect.terminal || (effect.defs & (1U << base)))
			return;
		if (knod_bpf_sink_store(m)) {
			if (m->insn.dst_reg != base || n == 16 || m->insn.src_reg > 10)
				return;
			if (m->store.bytes)
				group_left = m->store.bytes;
			if (m->memory_group_owned && !group_left)
				return; /* Never enter an existing group's interior. */
			if (!m->memory_group_owned || m->store.bytes)
				costs++;
			if (group_left)
				group_left--;
			width = knod_bpf_packet_width(m);
			for (j = 0; j < n; j++)
				if (m->insn.off < offsets[j] + widths[j] &&
				    offsets[j] < m->insn.off + width)
					return;
			stores[n] = m;
			offsets[n] = m->insn.off;
			widths[n] = width;
			regs[n] = m->insn.src_reg;
			stored_version[n++] = version[m->insn.src_reg];
			lo = min(lo, (int)m->insn.off);
			hi = max(hi, (int)m->insn.off + width);
			if (hi - lo > 16)
				return;
		} else {
			if (group_left || !knod_bpf_sink_transparent(m))
				return;
			if (m->store_hoist.emit) {
				if (m->store_hoist.base != base)
					return;
				disjoint_off[disjoint_n++] = m->store_hoist.off;
			}
			for (j = 0; j < 11; j++)
				if (effect.defs & (1U << j))
					version[j]++;
		}
		for (j = 0; j < disjoint_n; j++)
			if (disjoint_off[j] < hi && lo < disjoint_off[j] + 8)
				return;
		for (j = 0; j < n; j++)
			covered += widths[j];
		for (left = hi - lo; left > 0; chunks++)
			left -= knod_memory_chunk_width(left);
		/* Commit the first profitable complete group. No future speculation. */
		if (n > 1 && !group_left && covered == hi - lo && chunks < costs &&
		    lo >= -limit && hi - 1 < limit && knod_bpf_sink_store(m)) {
			plan.off = lo;
			plan.bytes = hi - lo;
			plan.base = base;
			for (j = 0; j < n; j++) {
				int source = regs[j];
				bool saved = stored_version[j] != version[regs[j]];
				if (saved) {
					for (k = 0; k < captures; k++)
						if (capture_reg[k] == regs[j] &&
						    capture_version[k] == stored_version[j])
							break;
					if (k == captures) {
						if (captures == 2)
							return;
						capture_reg[k] = regs[j];
						capture_version[k] = stored_version[j];
						capture_at[k] = j;
						captures++;
					}
					source = k;
				}
				for (k = 0; k < widths[j]; k++) {
					int b = offsets[j] - lo + k;
					plan.value[b].source = source;
					plan.value[b].saved = saved;
					plan.value[b].shift = 8 * k;
				}
			}
			for (j = 0; j < n; j++) {
				stores[j]->sink.member = true;
				memset(&stores[j]->store, 0, sizeof(stores[j]->store));
				stores[j]->memory_group_owned = false;
			}
			plan.member = true;
			m->sink = plan;
			for (j = 0; j < captures; j++) {
				stores[capture_at[j]]->sink.capture = j + 1;
				stores[capture_at[j]]->sink.capture_reg = capture_reg[j];
			}
			return;
		}
		if (list_is_last(&m->l, &prog->insns))
			return;
		prev = m;
		m = list_next_entry(m, l);
	}
}

/* Forward only the immediately following scalar read. Captured versions
 * survive the sink emitter; no intervening target instruction is admitted.
 */
static void knod_bpf_sink_forward(struct knod_prog *prog,
		struct knod_insn_meta *m)
{
	struct knod_insn_meta *n;
	int width, delta, j;

	if (!m->sink.bytes || m->sink.forward || list_is_last(&m->l, &prog->insns))
		return;
	n = list_next_entry(m, l);
	if (n->sink.member || n->memory_group_owned || n->packet_imm_valid ||
	    n->percpu_rmw_add || n->loads.count || n->copy.bytes ||
	    n->packet_region.count || n->conststore.bytes || n->conststore.elide ||
	    n->store_hoist.emit || n->store_hoist.elide || n->store.bytes ||
	    n->is_merge_point || (n->flags & FLAG_INSN_IS_JUMP_DST) ||
	    n->subprog_idx != m->subprog_idx || n->bpf_insn_idx != m->bpf_insn_idx + 1 ||
	    n->ptr.type != PTR_TO_PACKET || n->insn.src_reg != m->sink.base ||
	    n->insn.dst_reg > 9)
		return;
	switch (n->insn.code) {
	case BPF_LDX | BPF_MEM | BPF_B: width = 1; break;
	case BPF_LDX | BPF_MEM | BPF_H: width = 2; break;
	case BPF_LDX | BPF_MEM | BPF_W: width = 4; break;
	default: return;
	}
	delta = n->insn.off - m->sink.off;
	if (delta < 0 || delta + width > m->sink.bytes)
		return;
	n->sink.member = n->sink.forward = true;
	n->sink.bytes = width;
	for (j = 0; j < width; j++)
		n->sink.value[j] = m->sink.value[delta + j];
}

static void knod_bpf_analyze_sinks(struct knod_bpf_priv *priv, struct knod_prog *prog)
{
	struct knod_insn_meta *m;
	list_for_each_entry(m, &prog->insns, l)
		memset(&m->sink, 0, sizeof(m->sink));
	if (priv->isa_version != 10)
		return; /* Audited native selected forms on the active GFX10 path. */
	list_for_each_entry(m, &prog->insns, l)
		knod_bpf_sink_group(prog, m, 2048);
	list_for_each_entry(m, &prog->insns, l)
		knod_bpf_sink_forward(prog, m);
}

static void knod_bpf_emit_sink(struct knod_bpf_priv *priv, struct knod_insn_meta *m)
{
	struct amdgcn_param32 dst, src, shift, bits, tmp;
	int off, width, j;

	if (m->sink.capture) {
		knod_vset32(&dst, KNOD_AMDGPU_TMP_VREG2_LO + m->sink.capture - 1);
		knod_mov32(priv, m, dst, bpf_reg64[m->sink.capture_reg].lo);
	}
	knod_iset32(&bits, 8);
	knod_vset32(&tmp, KNOD_AMDGPU_TMP_VREG3_HI);
	for (off = 0; off < m->sink.bytes; off += width) {
		width = knod_memory_chunk_width(m->sink.bytes - off);
		for (j = 0; j < width;) {
			const struct knod_bpf_sink_value *v = &m->sink.value[off + j];
			int run = 1;

			/* Coalesce byte provenance only within one output word and
			 * one original 32-bit source version. Source registers never
			 * overlap the v22..25 output packing bank or v29 scratch.
			 */
			while (j + run < width && (j % 4) + run < 4 &&
			       v->shift + run * 8 < 32) {
				const struct knod_bpf_sink_value *next =
					&m->sink.value[off + j + run];

				if (next->saved != v->saved || next->source != v->source ||
				    next->shift != v->shift + run * 8)
					break;
				run++;
			}
			knod_vset32(&dst, KNOD_AMDGPU_TMP_VREG0_LO + j / 4);
			if (v->saved)
				knod_vset32(&src, KNOD_AMDGPU_TMP_VREG2_LO + v->source);
			else
				src = bpf_reg64[v->source].lo;
			if (run == 4) {
				knod_mov32(priv, m, dst, src);
			} else {
				knod_iset32(&shift, v->shift);
				knod_iset32(&bits, run * 8);
				knod_bfe32(priv, m, j % 4 ? tmp : dst, src, shift, bits);
				if (j % 4) {
					knod_iset32(&shift, (j % 4) * 8);
					knod_emit(priv, m, v_lshl_or_b32, dst, tmp, shift, dst);
				}
			}
			j += run;
		}
		knod_vset32(&dst, KNOD_AMDGPU_TMP_VREG0_LO);
		if (m->sink.forward) {
			knod_mov32(priv, m, bpf_reg64[m->insn.dst_reg].lo, dst);
			knod_iset32(&tmp, 0);
			knod_mov32(priv, m, bpf_reg64[m->insn.dst_reg].hi, tmp);
			return;
		}
		knod_bpf_packet_store(priv, m, width, dst, m->sink.base, m->sink.off + off);
	}
}

/* Consume the target-independent proof. Unsupported targets keep every
 * original instruction. The caller skips members only after successful emit.
 */
static unsigned int knod_bpf_emit_copy(struct knod_bpf_priv *priv,
				      struct knod_insn_meta *first)
{
	const struct knod_bpf_copy *copy = &first->copy;
	struct amdgcn_param32 data, word, shift, bits, zero;
	int base = copy->base_reg, tmp = copy->value_reg;
	int best = copy->bytes, start = copy->src_off;
	int delta = copy->dst_off - start, last = start + copy->last_byte;
	struct { int off, width, reg; } chunks[3];
	int off, width, pos, offset_limit, n = 0, i;
	int next_reg = KNOD_AMDGPU_TMP_VREG0_LO;

	if (!best || (priv->isa_version != 10 && priv->isa_version != 11))
		return 0;
	/* GLOBAL immediate offsets are narrower than BPF's s16. Do not
	 * truncate an address; retain ordinary lowering outside this range.
	 */
	offset_limit = priv->isa_version == 10 ? 2048 : 4096;
	if (start < -offset_limit || start + best - 1 >= offset_limit ||
	    copy->dst_off < -offset_limit ||
	    copy->dst_off + best - 1 >= offset_limit)
		return 0;

	/* Each load owns separate temporary VGPRs until its store. At most
	 * five are needed: 15 bytes split as 12+2+1 uses 3+1+1 registers
	 * (v22..v26), all inside the ABI temporary area. No helper intervenes.
	 * The proof covers the whole group, so loads can precede all stores.
	 * GFX10/11 queues enable UNALIGNED before any program is submitted.
	 */
	knod_iset32(&zero, 0);
	knod_iset32(&bits, 8);
	for (off = 0; off < best; off += width) {
		width = best - off >= 16 ? 16 : best - off >= 12 ? 12 :
			best - off >= 8 ? 8 : 1 << ilog2(best - off);
		chunks[n].off = off;
		chunks[n].width = width;
		chunks[n++].reg = next_reg;
		knod_vset32(&data, next_reg);
		next_reg += (width + 3) / 4;
		switch (width) {
		case 16:
			knod_emit(priv, first, global_load_dwordx4, data,
				  bpf_reg64[base].lo, start + off);
			break;
		case 12:
			knod_emit(priv, first, global_load_dwordx3, data,
				  bpf_reg64[base].lo, start + off);
			break;
		case 8:
			knod_emit(priv, first, global_load_dwordx2, data,
				  bpf_reg64[base].lo, start + off);
			break;
		case 4:
			knod_emit(priv, first, global_load_dword, data,
				  bpf_reg64[base].lo, start + off);
			break;
		case 2:
			knod_emit(priv, first, global_load_ushort, data,
				  bpf_reg64[base].lo, start + off);
			break;
		default:
			knod_emit(priv, first, global_load_ubyte, data,
				  bpf_reg64[base].lo, start + off);
		}
	}
	/* Every following consumer sees completed loads. Keep group boundaries
	 * intact; the next group may depend on stores from this one.
	 */
	knod_wait_vmcnt(priv, first);
	for (i = 0; i < n; i++) {
		off = chunks[i].off;
		width = chunks[i].width;
		knod_vset32(&data, chunks[i].reg);
		/* A byte load leaves its zero-extended value in the BPF temporary,
		 * even when that register remains live after the copy sequence.
		 */
		pos = last - start - off;
		if (pos >= 0 && pos < width) {
			knod_vset32(&word, data.v + pos / 4);
			knod_iset32(&shift, (pos % 4) * 8);
			knod_bfe32(priv, first, bpf_reg64[tmp].lo, word,
				   shift, bits);
			knod_mov32(priv, first, bpf_reg64[tmp].hi, zero);
		}
		switch (width) {
		case 16:
			knod_emit(priv, first, global_store_dwordx4, data,
				  bpf_reg64[base].lo, start + delta + off);
			break;
		case 12:
			knod_emit(priv, first, global_store_dwordx3, data,
				  bpf_reg64[base].lo, start + delta + off);
			break;
		case 8:
			knod_emit(priv, first, global_store_dwordx2, data,
				  bpf_reg64[base].lo, start + delta + off);
			break;
		case 4:
			knod_emit(priv, first, global_store_dword, data,
				  bpf_reg64[base].lo, start + delta + off);
			break;
		case 2:
			knod_emit(priv, first, global_store_short, data,
				  bpf_reg64[base].lo, start + delta + off);
			break;
		default:
			knod_emit(priv, first, global_store_byte, data,
				  bpf_reg64[base].lo, start + delta + off);
		}
	}
	return 2 * best;
}

/* Pack each source's low byte without modifying any BPF register. Distinct
 * temporary words stay intact through all stores; no load wait is needed.
 */
static unsigned int knod_bpf_emit_store(struct knod_bpf_priv *priv,
				       struct knod_insn_meta *first)
{
	const struct knod_bpf_store *store = &first->store;
	struct amdgcn_param32 data, byte, zero, bits, shift;
	struct { int off, width, reg; } chunks[3];
	int off, width, j, i, n = 0, limit;
	int next_reg = KNOD_AMDGPU_TMP_VREG0_LO;

	if (!store->bytes ||
	    (priv->isa_version != 10 && priv->isa_version != 11))
		return 0;
	limit = priv->isa_version == 10 ? 2048 : 4096;
	if (store->off < -limit || store->off + store->bytes - 1 >= limit)
		return 0;
	knod_vset32(&byte, KNOD_AMDGPU_TMP_VREG3_HI);
	knod_iset32(&zero, 0);
	knod_iset32(&bits, 8);
	for (off = 0; off < store->bytes; off += width) {
		width = store->bytes - off >= 16 ? 16 :
			store->bytes - off >= 12 ? 12 :
			store->bytes - off >= 8 ? 8 :
			1 << ilog2(store->bytes - off);
		chunks[n].off = off;
		chunks[n].width = width;
		chunks[n++].reg = next_reg;
		for (j = 0; j < width; j++) {
			knod_vset32(&data, next_reg + j / 4);
			if (!(j % 4)) {
				knod_bfe32(priv, first, data,
					   bpf_reg64[store->value_reg[off + j]].lo,
					   zero, bits);
			} else {
				knod_bfe32(priv, first, byte,
					   bpf_reg64[store->value_reg[off + j]].lo,
					   zero, bits);
				knod_iset32(&shift, (j % 4) * 8);
				knod_emit(priv, first, v_lshl_or_b32,
					  data, byte, shift, data);
			}
		}
		next_reg += (width + 3) / 4;
	}
	/* Up to five packed words (15B = 12+2+1) use v22..v26;
	 * the byte temporary v29 is disjoint and inside the existing ABI.
	 */
	for (i = 0; i < n; i++) {
		knod_vset32(&data, chunks[i].reg);
		off = store->off + chunks[i].off;
		switch (chunks[i].width) {
		case 16:
			knod_emit(priv, first, global_store_dwordx4, data,
				  bpf_reg64[store->base_reg].lo, off);
			break;
		case 12:
			knod_emit(priv, first, global_store_dwordx3, data,
				  bpf_reg64[store->base_reg].lo, off);
			break;
		case 8:
			knod_emit(priv, first, global_store_dwordx2, data,
				  bpf_reg64[store->base_reg].lo, off);
			break;
		case 4:
			knod_emit(priv, first, global_store_dword, data,
				  bpf_reg64[store->base_reg].lo, off);
			break;
		case 2:
			knod_emit(priv, first, global_store_short, data,
				  bpf_reg64[store->base_reg].lo, off);
			break;
		default:
			knod_emit(priv, first, global_store_byte, data,
				  bpf_reg64[store->base_reg].lo, off);
		}
	}
	return store->bytes;
}

static unsigned int knod_bpf_emit_loads(struct knod_bpf_priv *priv,
					struct knod_insn_meta *first)
{
	const struct knod_memory_load_group *g = &first->loads;
	struct amdgcn_param32 data, word, tmp, shift, bits, zero, dst;
	unsigned char byte_reg[16], byte_shift[16];
	unsigned int off, width, j, i, k, take, pos, out;
	int next = KNOD_AMDGPU_TMP_VREG0_LO, limit;
	if (!g->count || (priv->isa_version != 10 && priv->isa_version != 11))
		return 0;
	limit = priv->isa_version == 10 ? 2048 : 4096;
	if (g->off < -limit || g->off + g->bytes - 1 >= limit)
		return 0;
	/* At most five load words (15B = 12+2+1), plus scatter v29. */
	for (off = 0; off < g->bytes; off += width) {
		width = knod_memory_chunk_width(g->bytes - off);
		knod_vset32(&data, next);
		for (j = 0; j < width; j++) {
			byte_reg[off + j] = next + j / 4;
			byte_shift[off + j] = (j % 4) * 8;
		}
		switch (width) {
		case 16:
			knod_emit(priv, first, global_load_dwordx4, data,
				  bpf_reg64[g->base].lo, g->off + off);
			break;
		case 12:
			knod_emit(priv, first, global_load_dwordx3, data,
				  bpf_reg64[g->base].lo, g->off + off);
			break;
		case 8:
			knod_emit(priv, first, global_load_dwordx2, data,
				  bpf_reg64[g->base].lo, g->off + off);
			break;
		case 4:
			knod_emit(priv, first, global_load_dword, data,
				  bpf_reg64[g->base].lo, g->off + off);
			break;
		case 2:
			knod_emit(priv, first, global_load_ushort, data,
				  bpf_reg64[g->base].lo, g->off + off);
			break;
		case 1:
			knod_emit(priv, first, global_load_ubyte, data,
				  bpf_reg64[g->base].lo, g->off + off);
			break;
		}
		next += (width + 3) / 4;
	}
	knod_wait_vmcnt(priv, first);
	knod_iset32(&zero, 0);
	knod_vset32(&tmp, KNOD_AMDGPU_TMP_VREG3_HI);
	for (i = 0; i < g->count; i++) {
		for (k = 0; k < g->member[i].width; k += 4) {
			dst = k ? bpf_reg64[g->member[i].dst].hi
				: bpf_reg64[g->member[i].dst].lo;
			out = 0;
			for (j = k; j < g->member[i].width && j < k + 4;
			     j += take) {
				pos = g->member[i].off + j;
				take = min_t(
				    unsigned int, 4 - byte_shift[pos] / 8,
				    min_t(unsigned int, g->member[i].width - j,
					  k + 4 - j));
				/* A short final chunk may occupy fewer bytes
				 * than its VGPR. */
				while (take > 1 && byte_reg[pos + take - 1] !=
						       byte_reg[pos])
					take--;
				knod_vset32(&word, byte_reg[pos]);
				if (take == 4) {
					knod_mov32(priv, first, dst, word);
				} else {
					knod_iset32(&shift, byte_shift[pos]);
					knod_iset32(&bits, take * 8);
					knod_bfe32(priv, first, out ? tmp : dst,
						   word, shift, bits);
					if (out) {
						knod_iset32(&shift, out * 8);
						knod_emit(priv, first,
							  v_lshl_or_b32, dst,
							  tmp, shift, dst);
					}
				}
				out += take;
			}
		}
		if (g->member[i].width < 8)
			knod_mov32(priv, first, bpf_reg64[g->member[i].dst].hi,
				   zero);
	}
	return g->count;
}
/* Exact three-instruction pair: no speculative bytes or intervening CFG. */
static bool knod_bpf_load_pair_free(struct knod_insn_meta *m)
{
	return !m->map_region.region && !m->map_region.producer &&
	       !m->map_lds_head && !m->map_lds_tail && !m->wide_read.owned && !m->map_widen.owned && !m->read_batch.owned && !m->percpu_dead && !m->load_pair.owned && !m->sink.member && !m->memory_group_owned &&
		!m->packet_imm_valid && !m->percpu_rmw_add && !m->loads.count &&
		!m->copy.bytes && !m->store.bytes && !m->packet_region.count &&
		!m->conststore.bytes && !m->conststore.elide &&
		!m->store_hoist.emit && !m->store_hoist.elide;
}

/* Two independent loads retain issue order and meet at the LDS tail wait. */
static bool knod_bpf_map_lds_pair(struct knod_insn_meta *a,
				  struct knod_insn_meta *b)
{
	int width, off;
	struct knod_local_effect effects[2] = {0};

	if (a->jit_engine != 1 || b->jit_engine != 1 ||
	    a->map_lds_head || a->map_lds_tail ||
	    b->map_lds_head || b->map_lds_tail ||
	    !knod_bpf_load_pair_free(a) || !knod_bpf_load_pair_free(b) ||
	    a->sr.owner || b->sr.owner || a->is_merge_point || b->is_merge_point ||
	    ((a->flags | b->flags) & FLAG_INSN_IS_JUMP_DST) ||
	    a->bpf_insn_idx < 0 || b->bpf_insn_idx != a->bpf_insn_idx + 1 ||
	    a->subprog_idx != b->subprog_idx ||
	    a->insn.code != (BPF_LDX | BPF_MEM | BPF_W) ||
	    a->ptr.type != PTR_TO_MAP_VALUE || a->insn.imm || b->insn.imm ||
	    a->insn.dst_reg > 9 || a->insn.src_reg > 9 ||
	    b->insn.dst_reg > 9 || b->insn.src_reg > 10 ||
	    b->ptr.type != PTR_TO_STACK)
		return false;
	switch (b->insn.code) {
	case BPF_LDX | BPF_MEM | BPF_B: width = 1; break;
	case BPF_LDX | BPF_MEM | BPF_H: width = 2; break;
	case BPF_LDX | BPF_MEM | BPF_W: width = 4; break;
	case BPF_LDX | BPF_MEM | BPF_DW: width = 8; break;
	default: return false;
	}
	off = b->sreg.stack_off + b->insn.off;
	if (off < -512 || off > -width || (off & (width - 1)))
		return false;
	effects[0].reg = knod_bpf_effect(&a->insn);
	effects[1].reg = knod_bpf_effect(&b->insn);
	effects[0].read = effects[1].read = true;
	effects[0].space = KNOD_MEMORY_SHARED_MAP;
	effects[1].space = KNOD_MEMORY_PRIVATE_STACK;
	if (!knod_local_batch_reads(&effects[0], &effects[1]))
		return false;

	a->map_lds_head = true;
	b->map_lds_tail = true;
	return true;
}

static void knod_bpf_analyze_map_lds(struct knod_bpf_priv *priv,
				     struct knod_prog *prog)
{
	struct knod_insn_meta *m;

	list_for_each_entry(m, &prog->insns, l) {
		m->map_lds_head = false;
		m->map_lds_tail = false;
	}
	if (priv->isa_version != 10)
		return;
	list_for_each_entry(m, &prog->insns, l) {
		if (!list_is_last(&m->l, &prog->insns))
			knod_bpf_map_lds_pair(m, list_next_entry(m, l));
	}
}

/* Consecutive ordinary map byte loads only. Exact bytes, no speculative gap. */
static void knod_bpf_analyze_map_widen(struct knod_bpf_priv *priv,
				      struct knod_prog *prog)
{
	struct knod_insn_meta *first, *m, *nodes[8];
	unsigned int n, i;

	list_for_each_entry(m, &prog->insns, l)
		memset(&m->map_widen, 0, sizeof(m->map_widen));
	if (priv->isa_version != 10)
		return;
	list_for_each_entry(first, &prog->insns, l) {
		int base = first->insn.src_reg;

		m = first;
		for (n = 0; n < ARRAY_SIZE(nodes); n++) {
			if (!knod_bpf_load_pair_free(m) || m->map_widen.owned ||
			    m->insn.code != (BPF_LDX | BPF_MEM | BPF_B) ||
			    m->insn.imm || m->ptr.type != PTR_TO_MAP_VALUE ||
			    base > 9 || m->insn.src_reg != base ||
			    m->insn.dst_reg > 9 || first->bpf_insn_idx < 0 ||
			    m->bpf_insn_idx != first->bpf_insn_idx + n ||
			    m->subprog_idx != first->subprog_idx ||
			    m->insn.off != first->insn.off + (int)n ||
			    m->insn.off < -2048 || m->insn.off >= 2048 ||
			    (n && (m->is_merge_point ||
				   (m->flags & FLAG_INSN_IS_JUMP_DST))))
				break;
			nodes[n] = m;
			if (m->insn.dst_reg == base ||
			    list_is_last(&m->l, &prog->insns)) {
				n++;
				break;
			}
			m = list_next_entry(m, l);
		}
		if (n < 2)
			continue;
		first->map_widen.count = n;
		for (i = 0; i < n; i++) {
			first->map_widen.dst[i] = nodes[i]->insn.dst_reg;
			nodes[i]->map_widen.owned = true;
		}
	}
}

static void knod_bpf_emit_map_widen(struct knod_bpf_priv *priv,
				   struct knod_insn_meta *first)
{
	struct amdgcn_param32 data, shift, bits, zero;
	unsigned int i = 0, width, slot = 0, j, n = first->map_widen.count;
	u8 source[8], shifts[8];

	if (!n)
		return;
	if (first->map_region.region &&
	    (first->map_region.region->map_region.producer != first ||
	     first->map_region.region->map_region.staged)) {
		first->map_region.error = true;
		return;
	}
	while (i < n) {
		width = n - i >= 4 ? 4 : n - i >= 2 ? 2 : 1;
		/* Keep every exact tail in its own staging word. */
		for (j = 0; j < width; j++) {
			source[i + j] = slot;
			shifts[i + j] = 8 * j;
		}
		knod_vset32(&data, KNOD_AMDGPU_TMP_VREG0_LO + slot++);
		switch (width) {
		case 4:
			knod_emit(priv, first, global_load_dword, data,
				  bpf_reg64[first->insn.src_reg].lo, first->insn.off + i);
			break;
		case 2:
			knod_emit(priv, first, global_load_ushort, data,
				  bpf_reg64[first->insn.src_reg].lo, first->insn.off + i);
			break;
		default:
			knod_emit(priv, first, global_load_ubyte, data,
				  bpf_reg64[first->insn.src_reg].lo, first->insn.off + i);
			break;
		}
		i += width;
	}
	if (first->map_region.region)
		knod_bpf_emit_region_inputs(priv, first,
			&first->map_region.region->packet_region);
	knod_wait_vmcnt(priv, first);
	if (first->map_region.region)
		first->map_region.region->map_region.staged = true;
	knod_iset32(&bits, 8);
	knod_iset32(&zero, 0);
	for (i = 0; i < n; i++) {
		knod_vset32(&data, KNOD_AMDGPU_TMP_VREG0_LO + source[i]);
		knod_iset32(&shift, shifts[i]);
		knod_bfe32(priv, first, bpf_reg64[first->map_widen.dst[i]].lo,
			   data, shift, bits);
		knod_mov32(priv, first, bpf_reg64[first->map_widen.dst[i]].hi, zero);
	}
}

/* Compose selected groups only after every owner has been assigned. */
static bool knod_bpf_map_region_other_owner(struct knod_insn_meta *m, bool packet_owner)
{
	return m->map_lds_head || m->map_lds_tail || m->wide_read.owned || m->read_batch.owned || m->load_pair.owned ||
	       m->percpu_dead || m->percpu_rmw_add ||
	       (m->memory_group_owned && !packet_owner) ||
	       m->loads.count || m->copy.bytes || m->store.bytes ||
	       m->packet_imm_valid || m->conststore.bytes || m->conststore.elide ||
	       m->store_hoist.emit || m->store_hoist.elide || m->sink.member;
}

static void knod_bpf_analyze_map_regions(struct knod_bpf_priv *priv,
				       struct knod_prog *prog)
{
	struct knod_insn_meta *first, *m, *region;
	unsigned int i, defs, count;

	list_for_each_entry(m, &prog->insns, l)
		memset(&m->map_region, 0, sizeof(m->map_region));
	if (priv->isa_version != 10 || prog->jit_engine != 1)
		return;
	list_for_each_entry(first, &prog->insns, l) {
		count = first->map_widen.count;
		if (count < 2 || count > 8 || !first->map_widen.owned ||
		    first->bpf_insn_idx < 0 || first->jit_engine != 1 ||
		    first->insn.src_reg > 9)
			continue;
		defs = 0;
		m = first;
		for (i = 0; i < count; i++) {
			if (!m->map_widen.owned || m->packet_region.count ||
			    knod_bpf_map_region_other_owner(m, false) || m->sr.owner ||
			    m->jit_engine != 1 || (m->flags & FLAG_INSN_IS_SUBPROG_START) ||
			    m->subprog_idx != first->subprog_idx ||
			    m->bpf_insn_idx != first->bpf_insn_idx + (int)i ||
			    (i && (m->is_merge_point || (m->flags & FLAG_INSN_IS_JUMP_DST))) ||
			    m->insn.code != (BPF_LDX | BPF_MEM | BPF_B) ||
			    m->ptr.type != PTR_TO_MAP_VALUE || m->insn.imm ||
			    m->insn.dst_reg > 9 || m->insn.src_reg != first->insn.src_reg ||
			    (i + 1 < count && m->insn.dst_reg == m->insn.src_reg) ||
			    m->insn.off != first->insn.off + (int)i ||
			    first->map_widen.dst[i] != m->insn.dst_reg ||
			    list_is_last(&m->l, &prog->insns))
				break;
			defs |= 1U << m->insn.dst_reg;
			m = list_next_entry(m, l);
		}
		if (i != count)
			continue;
		region = m;
		if (!region->packet_region.count || region->packet_region.base > 9 ||
		    !region->packet_region.input_bytes || region->packet_region.input_bytes > 8 ||
		    region->packet_region.input_bytes == 7 ||
		    !region->packet_region.bytes || region->packet_region.bytes > 16 ||
		    region->jit_engine != 1 || region->is_merge_point ||
		    (region->flags & (FLAG_INSN_IS_JUMP_DST | FLAG_INSN_IS_SUBPROG_START)) ||
		    region->subprog_idx != first->subprog_idx ||
		    region->bpf_insn_idx != first->bpf_insn_idx + (int)count ||
		    (defs & (1U << region->packet_region.base)) ||
		    region->map_widen.owned || knod_bpf_map_region_other_owner(region, true) ||
		    (region->sr.owner && region->sr.owner != region) ||
		    first->map_region.region || region->map_region.producer)
			continue;
		/* Commit both directions only after the complete proof. */
		first->map_region.region = region;
		region->map_region.producer = first;
	}
}

#define KNOD_READ_BATCH_BASE (KNOD_AMDGPU_TMP_VREG3_HI + 1)

static unsigned int knod_bpf_read_batch_width(const struct bpf_insn *insn,
					     bool load)
{
	unsigned int cls = load ? BPF_LDX : BPF_STX;

	if (insn->code == (cls | BPF_MEM | BPF_B))
		return 1;
	if (insn->code == (cls | BPF_MEM | BPF_H))
		return 2;
	if (insn->code == (cls | BPF_MEM | BPF_W))
		return 4;
	return 0;
}

/* Only unconditional exact packet reads and private aligned LDS stores.
 * Publish ownership after all checks; a rejected prefix changes nothing.
 */
static void knod_bpf_read_batch_group(struct knod_prog *prog,
				      struct knod_insn_meta *first)
{
	struct knod_insn_meta *nodes[16], *loads[8], *m = first;
	unsigned int n = 0, count = 0, last = 0, i, width;
	int base = first->insn.src_reg, off;

	if (base > 9 || first->bpf_insn_idx < 0 ||
	    first->ptr.type != PTR_TO_PACKET ||
	    !knod_bpf_read_batch_width(&first->insn, true))
		return;
	while (n < ARRAY_SIZE(nodes) && count < ARRAY_SIZE(loads)) {
		if (!knod_bpf_load_pair_free(m) ||
		    m->subprog_idx != first->subprog_idx ||
		    m->bpf_insn_idx != first->bpf_insn_idx + n ||
		    (n && (m->is_merge_point ||
			   (m->flags & FLAG_INSN_IS_JUMP_DST))))
			break;
		width = knod_bpf_read_batch_width(&m->insn, true);
		if (width) {
			if (m->ptr.type != PTR_TO_PACKET ||
			    m->insn.src_reg != base || m->insn.dst_reg > 9 ||
			    m->insn.dst_reg == base || m->insn.off < -2048 ||
			    m->insn.off + (int)width > 2048)
				break;
			loads[count++] = m;
			last = n;
		} else {
			width = knod_bpf_read_batch_width(&m->insn, false);
			off = m->dreg.stack_off + m->insn.off;
			if (!width || m->ptr.type != PTR_TO_STACK ||
			    m->insn.dst_reg != BPF_REG_FP || m->insn.src_reg > 10 ||
			    off < -512 || off > -(int)width || (off & (width - 1)))
				break;
		}
		nodes[n++] = m;
		if (list_is_last(&m->l, &prog->insns))
			break;
		m = list_next_entry(m, l);
	}
	if (count < 3)
		return;
	for (i = 0; i <= last; i++)
		nodes[i]->read_batch.owned = true;
	for (i = 0; i < count; i++) {
		loads[i]->read_batch.load = true;
		loads[i]->read_batch.slot = i;
		first->read_batch.off[i] = loads[i]->insn.off;
		first->read_batch.width[i] =
			knod_bpf_read_batch_width(&loads[i]->insn, true);
	}
	/* Exact adjacent dwords only: no holes, no new address residue. */
	for (i = 0; i + 1 < count; i++) {
		if (first->read_batch.width[i] == 4 &&
		    first->read_batch.width[i + 1] == 4 &&
		    first->read_batch.off[i + 1] == first->read_batch.off[i] + 4) {
			first->read_batch.width[i] = 8;
			first->read_batch.width[++i] = 0;
		}
	}
	first->read_batch.first = true;
	first->read_batch.base = base;
	first->read_batch.count = count;
}

static void knod_bpf_analyze_read_batches(struct knod_bpf_priv *priv,
					 struct knod_prog *prog)
{
	struct knod_insn_meta *m;

	if (priv->isa_version != 10)
		return;
	list_for_each_entry(m, &prog->insns, l)
		knod_bpf_read_batch_group(prog, m);
}

static void knod_bpf_emit_read_batch(struct knod_bpf_priv *priv,
				    struct knod_insn_meta *m)
{
	struct amdgcn_param32 data, zero;
	unsigned int i;

	if (m->read_batch.first) {
		for (i = 0; i < m->read_batch.count; i++) {
			knod_vset32(&data, KNOD_READ_BATCH_BASE + i);
			switch (m->read_batch.width[i]) {
			case 1:
				knod_emit(priv, m, global_load_ubyte, data,
					  bpf_reg64[m->read_batch.base].lo,
					  m->read_batch.off[i]);
				break;
			case 2:
				knod_emit(priv, m, global_load_ushort, data,
					  bpf_reg64[m->read_batch.base].lo,
					  m->read_batch.off[i]);
				break;
			case 4:
				knod_emit(priv, m, global_load_dword, data,
					  bpf_reg64[m->read_batch.base].lo,
					  m->read_batch.off[i]);
				break;
			case 8:
				knod_emit(priv, m, global_load_dwordx2, data,
					  bpf_reg64[m->read_batch.base].lo,
					  m->read_batch.off[i]);
				break;
			}
		}
		knod_wait_vmcnt(priv, m);
	}
	knod_vset32(&data, KNOD_READ_BATCH_BASE + m->read_batch.slot);
	knod_mov32(priv, m, bpf_reg64[m->insn.dst_reg].lo, data);
	knod_iset32(&zero, 0);
	knod_mov32(priv, m, bpf_reg64[m->insn.dst_reg].hi, zero);
}

#define KNOD_WR_PREFIX 128
#define KNOD_WR_NODES 24
#define KNOD_WR_CAPTURE (KNOD_AMDGPU_TMP_VREG3_HI + 1)

struct knod_wr_expr { s16 off; u8 kind; };
struct knod_wr_workspace {
	struct knod_insn_meta *prefix[KNOD_WR_PREFIX];
	struct knod_wr_expr in[KNOD_WR_PREFIX][11];
	bool reached[KNOD_WR_PREFIX];
};

static void knod_wr_join(struct knod_wr_workspace *w, unsigned int to,
			 const struct knod_wr_expr *state)
{
	unsigned int r;

	if (!w->reached[to]) {
		memcpy(w->in[to], state, sizeof(w->in[to]));
		w->reached[to] = true;
		return;
	}
	for (r = 0; r < 11; r++)
		if (w->in[to][r].kind != state[r].kind ||
		    w->in[to][r].off != state[r].off)
			w->in[to][r] = (struct knod_wr_expr){};
}

/* Forward CFG intersection, not the verifier's last-visit range snapshot.
 * kind 1 is this XDP packet's data; kind 2 is its data_end; kind 3 is ctx.
 */
static bool knod_wr_certificate(struct knod_prog *prog,
				struct knod_insn_meta *bound,
				struct knod_wr_workspace *w,
				struct knod_wr_expr *state)
{
	struct knod_insn_meta *m;
	unsigned int count = 0, i, j, r, linear = 0;
	int next, target, off;

	memset(w, 0, sizeof(*w));
	list_for_each_entry(m, &prog->insns, l) {
		if (count == KNOD_WR_PREFIX || m->subprog_idx ||
		    m->bpf_insn_idx < 0)
			return false;
		if ((!count && m->bpf_insn_idx) ||
		    (count && m->bpf_insn_idx !=
		     w->prefix[count - 1]->bpf_insn_idx + 1))
			return false;
		if (count && w->prefix[count - 1]->insn.code ==
		    (BPF_LD | BPF_DW | BPF_IMM)) {
			if (m->insn.code || m->insn.dst_reg || m->insn.src_reg ||
			    m->insn.off || m->is_merge_point ||
			    (m->flags & FLAG_INSN_IS_JUMP_DST))
				return false;
		} else if (!m->insn.code) {
			return false;
		}
		w->prefix[count++] = m;
		if (m == bound)
			break;
	}
	if (!count || w->prefix[count - 1] != bound)
		return false;
	/* No out-of-prefix predecessor may bypass the analyzed entry paths. */
	list_for_each_entry(m, &prog->insns, l) {
		if (m->linear_idx != linear++)
			return false;
		if (m->linear_idx <= bound->linear_idx)
			continue;
		if ((BPF_CLASS(m->insn.code) == BPF_JMP ||
		     BPF_CLASS(m->insn.code) == BPF_JMP32) &&
		    BPF_OP(m->insn.code) != BPF_CALL &&
		    BPF_OP(m->insn.code) != BPF_EXIT && m->merge_point &&
		    m->merge_point->linear_idx <= bound->linear_idx)
			return false;
		if (m->bpf_insn_idx < 0) {
			/* classify_linear resolves synthetic JA jmp_dst to merge_point. */
			if (m->insn.code != (BPF_JMP | BPF_JA) || !m->jmp_dst ||
			    m->jmp_dst != m->merge_point)
				return false;
			continue;
		}
		if (m->insn.code == (BPF_JMP32 | BPF_JA))
			return false;
		if ((BPF_CLASS(m->insn.code) == BPF_JMP ||
		     BPF_CLASS(m->insn.code) == BPF_JMP32) &&
		    BPF_OP(m->insn.code) != BPF_CALL &&
		    BPF_OP(m->insn.code) != BPF_EXIT &&
		    m->bpf_insn_idx + 1 + m->insn.off <= bound->bpf_insn_idx)
			return false;
	}
	w->reached[0] = true;
	w->in[0][1].kind = 3; /* XDP entry r1 is the context. */
	for (i = 0; i < count; i++) {
		const struct bpf_insn *in;
		u8 cls, op, d, s;

		if (!w->reached[i])
			continue;
		m = w->prefix[i];
		memcpy(state, w->in[i], sizeof(w->in[i]));
		if (m == bound)
			return true;
		if (i && w->prefix[i - 1]->insn.code ==
		    (BPF_LD | BPF_DW | BPF_IMM)) {
			/* prepare/RPO retain the high slot; it is not an instruction. */
			if (i + 1 < count)
				knod_wr_join(w, i + 1, state);
			continue;
		}
		in = &m->insn;
		cls = BPF_CLASS(in->code);
		op = BPF_OP(in->code);
		d = in->dst_reg;
		s = in->src_reg;
		if (d > 10 || s > 10 ||
		    (cls == BPF_JMP32 && op == BPF_JA))
			return false;
		/* FETCH/CMPXCHG atomics can write src/r0; no affine proof here. */
		if ((cls == BPF_ST || cls == BPF_STX) &&
		    BPF_MODE(in->code) != BPF_MEM)
			return false;
		if ((cls == BPF_JMP || cls == BPF_JMP32) && op == BPF_CALL &&
		    in->code != (BPF_JMP | BPF_CALL))
			return false;
		if (in->code == (BPF_LDX | BPF_MEM | BPF_W) &&
		    m->ptr.type == PTR_TO_CTX && state[s].kind == 3 && !state[s].off &&
		    tnum_is_const(m->ptr.var_off) && !m->ptr.var_off.value &&
		    (in->off == 0 || in->off == 4)) {
			state[d] = (struct knod_wr_expr){ .kind = in->off ? 2 : 1 };
		} else if (in->code == (BPF_ALU64 | BPF_MOV | BPF_X) && !in->off) {
			state[d] = state[s];
		} else if (in->code == (BPF_ALU64 | BPF_ADD | BPF_K) &&
			   state[d].kind && in->imm >= -2048 && in->imm <= 2048) {
			off = state[d].off + in->imm;
			state[d] = off >= -2048 && off <= 2048 ?
				(struct knod_wr_expr){ off, state[d].kind } :
				(struct knod_wr_expr){};
		} else if (cls == BPF_LDX || cls == BPF_LD ||
			   cls == BPF_ALU || cls == BPF_ALU64) {
			state[d] = (struct knod_wr_expr){};
		} else if (in->code == (BPF_JMP | BPF_CALL)) {
			if (in->src_reg || in->imm != BPF_FUNC_map_lookup_elem)
				return false;
			for (r = 0; r <= 5; r++)
				state[r] = (struct knod_wr_expr){};
		}
		if (cls == BPF_JMP || cls == BPF_JMP32) {
			if (op == BPF_EXIT)
				continue;
			if (op != BPF_CALL) {
				target = m->bpf_insn_idx + 1 + in->off;
				if (!m->merge_point || m->merge_point->bpf_insn_idx != target)
					return false;
				if (target <= m->bpf_insn_idx)
					return false;
				if (target <= bound->bpf_insn_idx) {
					for (j = i + 1; j < count; j++)
						if (w->prefix[j]->bpf_insn_idx == target)
							break;
					if (j == count || (j && w->prefix[j - 1]->insn.code ==
					    (BPF_LD | BPF_DW | BPF_IMM)))
						return false;
					knod_wr_join(w, j, state);
				}
				if (op == BPF_JA)
					continue;
			}
		}
		next = i + 1;
		if (next < count)
			knod_wr_join(w, next, state);
	}
	return false;
}

static void knod_wr_group(struct knod_prog *prog, struct knod_insn_meta *bound,
			  struct knod_wr_workspace *w)
{
	struct knod_wr_expr state[11];
	struct knod_insn_meta *nodes[KNOD_WR_NODES], *loads[8], *m, *first;
	int offsets[8], lo = 2048, hi = 0, base = -1, limit, off;
	unsigned int n = 0, count = 0, last = 0, i, width;

	if (bound->insn.dst_reg > 10 || bound->insn.src_reg > 10 ||
	    bound->insn.code != (BPF_JMP | BPF_JGT | BPF_X) ||
	    bound->branch_type != KNOD_BR_FORWARD_SKIP || bound->jump_neg_op ||
	    !bound->merge_point || list_is_last(&bound->l, &prog->insns) ||
	    !knod_wr_certificate(prog, bound, w, state))
		return;
	if (state[bound->insn.dst_reg].kind != 1 ||
	    state[bound->insn.src_reg].kind != 2 ||
	    state[bound->insn.src_reg].off)
		return;
	limit = state[bound->insn.dst_reg].off;
	if (limit <= 0)
		return;
	for (i = 0; i < 10; i++)
		if (state[i].kind == 1 && !state[i].off) {
			base = i;
			break;
		}
	if (base < 0)
		return;
	first = m = list_next_entry(bound, l);
	if (!knod_bpf_read_batch_width(&first->insn, true))
		return;
	while (n < KNOD_WR_NODES && count < ARRAY_SIZE(loads)) {
		u8 code = m->insn.code;

		if (!knod_bpf_load_pair_free(m) || m->is_merge_point ||
		    (m->flags & FLAG_INSN_IS_JUMP_DST) ||
		    m->subprog_idx != bound->subprog_idx ||
		    m->bpf_insn_idx != bound->bpf_insn_idx + n + 1)
			break;
		width = knod_bpf_read_batch_width(&m->insn, true);
		if (width) {
			if (m->ptr.type != PTR_TO_PACKET || m->insn.src_reg > 9 ||
			    m->insn.dst_reg > 9 || m->insn.dst_reg == base ||
			    state[m->insn.src_reg].kind != 1)
				break;
			off = state[m->insn.src_reg].off + m->insn.off;
			if (off < 0 || off + (int)width > limit)
				return;
			offsets[count] = off;
			loads[count++] = m;
			lo = min(lo, off);
			hi = max(hi, off + (int)width);
			state[m->insn.dst_reg] = (struct knod_wr_expr){};
			last = n;
		} else if (code == (BPF_ALU64 | BPF_AND | BPF_K)) {
			if (m->insn.dst_reg > 9 || m->insn.dst_reg == base)
				break;
			state[m->insn.dst_reg] = (struct knod_wr_expr){};
		} else if (code == (BPF_JMP | BPF_JNE | BPF_K) ||
			   code == (BPF_JMP | BPF_JEQ | BPF_K)) {
			if (m->branch_type != KNOD_BR_FORWARD_SKIP ||
			    m->jump_neg_op != (BPF_OP(code) == BPF_JNE) ||
			    !m->merge_point)
				break;
		} else {
			width = knod_bpf_read_batch_width(&m->insn, false);
			off = m->dreg.stack_off + m->insn.off;
			if (!width || m->ptr.type != PTR_TO_STACK ||
			    m->insn.dst_reg != BPF_REG_FP || m->insn.src_reg > 10 ||
			    off < -512 || off > -(int)width || (off & (width - 1)))
				break;
		}
		nodes[n++] = m;
		if (list_is_last(&m->l, &prog->insns))
			break;
		m = list_next_entry(m, l);
	}
	if (count < 3 || hi - lo > 32 || (hi - lo) % 4 ||
	    bound->merge_point->linear_idx <= nodes[last]->linear_idx)
		return;
	for (i = 0; i <= last; i++) {
		m = nodes[i];
		if ((m->insn.code == (BPF_JMP | BPF_JNE | BPF_K) ||
		     m->insn.code == (BPF_JMP | BPF_JEQ | BPF_K)) &&
		    m->merge_point->linear_idx <= nodes[last]->linear_idx)
			return;
	}
	/* Do not rely solely on cached merge/destination flags. */
	list_for_each_entry(m, &prog->insns, l) {
		if ((BPF_CLASS(m->insn.code) == BPF_JMP ||
		     BPF_CLASS(m->insn.code) == BPF_JMP32) &&
		    BPF_OP(m->insn.code) != BPF_CALL &&
		    BPF_OP(m->insn.code) != BPF_EXIT && m->merge_point &&
		    m->merge_point->linear_idx >= first->linear_idx &&
		    m->merge_point->linear_idx <= nodes[last]->linear_idx)
			return;
	}
	for (i = 0; i < count; i++) {
		width = knod_bpf_read_batch_width(&loads[i]->insn, true);
		if ((offsets[i] - lo) % 4 + width > 4)
			return;
	}
	for (i = 0; i <= last; i++)
		nodes[i]->wide_read.owned = true;
	for (i = 0; i < count; i++) {
		loads[i]->wide_read.load = true;
		loads[i]->wide_read.half_phase = (lo % 4 == 2 && hi - lo <= 28);
		loads[i]->wide_read.offset = offsets[i] - lo;
		loads[i]->wide_read.width =
			knod_bpf_read_batch_width(&loads[i]->insn, true);
	}
	first->wide_read.first = true;
	first->wide_read.base = base;
	first->wide_read.off = lo;
	first->wide_read.bytes = hi - lo;
}

static void knod_bpf_analyze_wide_reads(struct knod_bpf_priv *priv,
				       struct knod_prog *prog)
{
	struct knod_wr_workspace *w;
	struct knod_insn_meta *m;

	if (priv->isa_version != 10 || !prog->wide_read_xdp)
		return;
	w = kzalloc(sizeof(*w), GFP_KERNEL);
	if (!w)
		return;
	list_for_each_entry(m, &prog->insns, l)
		knod_wr_group(prog, m, w);
	kfree(w);
}

static void knod_bpf_emit_wide_read(struct knod_bpf_priv *priv,
				   struct knod_insn_meta *m)
{
	struct amdgcn_param32 data, shift, mask, zero, next;
	unsigned int i, width, offset = m->wide_read.offset;
	bool half = m->wide_read.half_phase;

	if (m->wide_read.first) {
		/* Capture origin is lo-2, but those two padding bytes are never read. */
		if (half) {
			knod_vset32(&data, KNOD_WR_CAPTURE);
			knod_emit(priv, m, global_load_ushort, data,
				  bpf_reg64[m->wide_read.base].lo, m->wide_read.off);
			knod_vset32(&data, KNOD_WR_CAPTURE + m->wide_read.bytes / 4);
			knod_emit(priv, m, global_load_ushort, data,
				  bpf_reg64[m->wide_read.base].lo,
				  m->wide_read.off + m->wide_read.bytes - 2);
		}
		for (i = half ? 2 : 0; i < m->wide_read.bytes - (half ? 2 : 0); i += width) {
			width = min(16U, m->wide_read.bytes - (half ? 2 : 0) - i);
			knod_vset32(&data, KNOD_WR_CAPTURE + (i + (half ? 2 : 0)) / 4);
			switch (width) {
			case 16:
				knod_emit(priv, m, global_load_dwordx4, data,
					  bpf_reg64[m->wide_read.base].lo, m->wide_read.off + i);
				break;
			case 12:
				knod_emit(priv, m, global_load_dwordx3, data,
					  bpf_reg64[m->wide_read.base].lo, m->wide_read.off + i);
				break;
			case 8:
				knod_emit(priv, m, global_load_dwordx2, data,
					  bpf_reg64[m->wide_read.base].lo, m->wide_read.off + i);
				break;
			case 4:
				knod_emit(priv, m, global_load_dword, data,
					  bpf_reg64[m->wide_read.base].lo, m->wide_read.off + i);
				break;
			}
		}
		knod_wait_vmcnt(priv, m);
		if (half) {
			knod_vset32(&data, KNOD_WR_CAPTURE);
			knod_iset32(&shift, 16);
			knod_emit(priv, m, v_lshlrev_b32, data, shift, data);
		}
	}
	if (half)
		offset += 2;
	knod_vset32(&data, KNOD_WR_CAPTURE + offset / 4);
	if (offset % 4) {
		knod_iset32(&shift, (offset % 4) * 8);
		if (offset % 4 + m->wide_read.width > 4) {
			knod_vset32(&next, KNOD_WR_CAPTURE + offset / 4 + 1);
			knod_emit(priv, m, v_alignbit_b32, bpf_reg64[m->insn.dst_reg].lo,
				  next, data, shift);
		} else {
			knod_emit(priv, m, v_lshrrev_b32, bpf_reg64[m->insn.dst_reg].lo,
				  shift, data);
		}
		data = bpf_reg64[m->insn.dst_reg].lo;
	}
	if (m->wide_read.width < 4) {
		knod_iset32(&mask, (1U << (m->wide_read.width * 8)) - 1);
		knod_emit(priv, m, v_and_b32_e32, bpf_reg64[m->insn.dst_reg].lo,
			  mask, data);
	} else if (!(offset % 4)) {
		knod_mov32(priv, m, bpf_reg64[m->insn.dst_reg].lo, data);
	}
	knod_iset32(&zero, 0);
	knod_mov32(priv, m, bpf_reg64[m->insn.dst_reg].hi, zero);
}


/* The common DAG describes BPF effects; this adapter certifies only native
 * direct W loads and plain MOV encodings. No native temporary stays live.
 */
static bool knod_bpf_local_node(struct knod_insn_meta *m,
			       struct knod_local_effect *f)
{
	unsigned int cls = BPF_CLASS(m->insn.code);

	if (m->jit_engine != 1 || m->subprog_idx || m->bpf_insn_idx < 0 ||
	    m->is_merge_point || m->sr.owner || m->merge_read.role ||
	    m->local_wait_member || !knod_bpf_load_pair_free(m) ||
	    (m->flags & (FLAG_INSN_IS_JUMP_DST | FLAG_INSN_IS_SUBPROG_START)))
		return false;
	*f = (struct knod_local_effect){.reg = knod_bpf_effect(&m->insn)};
	if (f->reg.barrier || f->reg.terminal || m->insn.dst_reg > 9)
		return false;
	if (m->insn.code == (BPF_LDX | BPF_MEM | BPF_W)) {
		if (!m->insn.imm && m->insn.src_reg <= 10 &&
		    m->ptr.type == PTR_TO_STACK) {
			s64 off = (s64)m->sreg.stack_off + m->insn.off;

			if (off < -512 || off > -4 || (off & 3))
				return false;
			f->read = true;
			f->space = KNOD_MEMORY_PRIVATE_STACK;
			f->width = 4;
			return true;
		}
		if (m->insn.imm || m->insn.src_reg > 9 ||
		    (m->ptr.type != PTR_TO_PACKET && m->ptr.type != PTR_TO_MAP_VALUE))
			return false;
		f->read = true;
		f->space = m->ptr.type == PTR_TO_PACKET ?
			KNOD_MEMORY_PACKET : KNOD_MEMORY_SHARED_MAP;
		f->width = 4;
		return true;
	}
	return (cls == BPF_ALU || cls == BPF_ALU64) &&
	       (BPF_OP(m->insn.code) == BPF_MOV ||
		BPF_OP(m->insn.code) == BPF_AND ||
		BPF_OP(m->insn.code) == BPF_OR ||
		BPF_OP(m->insn.code) == BPF_XOR) && !m->insn.off &&
	       (BPF_SRC(m->insn.code) != BPF_X || m->insn.src_reg <= 10);
}

/* Bounded local issue/finish scheduling: prioritize independent reads while
 * retaining memory issue order and every BPF register dependency. Emit the
 * selected ready order at the head and complete pending results before exit.
 */
static void knod_bpf_analyze_local_waits(struct knod_bpf_priv *priv,
				       struct knod_prog *prog)
{
	struct knod_insn_meta *head, *m, *nodes[KNOD_LOCAL_MAX];
	struct knod_local_effect facts[KNOD_LOCAL_MAX];
	unsigned char order[KNOD_LOCAL_MAX], scores[KNOD_LOCAL_MAX] = {0};
	unsigned int n, last, reads, i;

	if (priv->isa_version != 10 || prog->jit_engine != 1)
		return;
	list_for_each_entry(head, &prog->insns, l) {
		m = head;
		reads = last = n = 0;
		while (n < KNOD_LOCAL_MAX) {
			if (!knod_bpf_local_node(m, &facts[n]) ||
			    head->bpf_insn_idx > INT_MAX - KNOD_LOCAL_MAX ||
			    head->linear_idx > INT_MAX - KNOD_LOCAL_MAX ||
			    m->bpf_insn_idx != head->bpf_insn_idx + n ||
			    m->linear_idx != head->linear_idx + n ||
			    (!n && !facts[n].read))
				break;
			nodes[n] = m;
			scores[n] = facts[n].read ? 1 : 0;
			if (facts[n].read) {
				reads++;
				last = n;
			}
			n++;
			if (list_is_last(&m->l, &prog->insns))
				break;
			m = list_next_entry(m, l);
		}
		if (reads < 2 || !knod_local_schedule(facts, scores, last + 1, order))
			continue;
		for (i = 0; i <= last; i++)
			nodes[i]->local_wait_member = true;
		head->local_issue_count = last + 1;
		for (i = 0; i <= last; i++)
			head->local_issue_order[i] = order[i];
	}
}


static void knod_bpf_emit_local_bitwise(struct knod_bpf_priv *priv,
				      struct knod_insn_meta *head,
				      const struct bpf_insn *insn)
{
	struct amdgcn_param32 value, dst;
	unsigned int high, op = BPF_OP(insn->code);
	bool wide = BPF_CLASS(insn->code) == BPF_ALU64;

	for (high = 0; high < (wide ? 2 : 1); high++) {
		dst = high ? bpf_reg64[insn->dst_reg].hi :
			bpf_reg64[insn->dst_reg].lo;
		if (BPF_SRC(insn->code) == BPF_X) {
			value = high ? bpf_reg64[insn->src_reg].hi :
				bpf_reg64[insn->src_reg].lo;
		} else {
			if (high && op == BPF_AND && insn->imm < 0)
				continue;
			if (high && op == BPF_OR && insn->imm >= 0)
				continue;
			knod_iset32(&value, high ? (insn->imm < 0 ? -1 : 0) :
				    insn->imm);
			if (high && op != BPF_XOR) {
				knod_mov32(priv, head, dst, value);
				continue;
			}
		}
		switch (op) {
		case BPF_AND:
			knod_and32(priv, head, dst, value, dst);
			break;
		case BPF_OR:
			knod_or32(priv, head, dst, value, dst);
			break;
		case BPF_XOR:
			knod_xor32(priv, head, dst, value, dst);
			break;
		}
	}
	if (!wide) {
		knod_iset32(&value, 0);
		knod_mov32(priv, head, bpf_reg64[insn->dst_reg].hi, value);
	}
}

/* Audited native direct-load/MOV/bitwise/LDS lowering. The common plan contains
 * only BPF positions; no physical register IDs or native opcodes live there.
 */
static void knod_bpf_emit_local_ready(struct knod_bpf_priv *priv,
				      struct knod_insn_meta *head)
{
	struct knod_insn_meta *nodes[KNOD_LOCAL_MAX], *m = head;
	struct amdgcn_param32 value;
	unsigned int i, pending = 0, pending_lds = 0;
	bool has_lds = false;

	if (!head->local_issue_count)
		return;
	for (i = 0; i < head->local_issue_count; i++) {
		nodes[i] = m;
		has_lds |= m->insn.code == (BPF_LDX | BPF_MEM | BPF_W) &&
			m->ptr.type == PTR_TO_STACK;
		m = list_next_entry(m, l);
	}
	/* Keep pre-existing private-stack writes complete at region entry. */
	if (has_lds)
		knod_emit(priv, head, s_waitcnt_lgkmcnt);
	for (i = 0; i < head->local_issue_count; i++) {
		struct knod_bpf_effect effect;
		unsigned int dst;

		m = nodes[head->local_issue_order[i]];
		dst = m->insn.dst_reg;
		effect = knod_bpf_effect(&m->insn);
		if (pending & (effect.uses | effect.defs)) {
			knod_wait_vmcnt(priv, head);
			pending = 0;
		}
		if (pending_lds & (effect.uses | effect.defs)) {
			knod_emit(priv, head, s_waitcnt_lgkmcnt);
			pending_lds = 0;
		}
		if (BPF_CLASS(m->insn.code) == BPF_LDX &&
		    m->ptr.type == PTR_TO_STACK) {
			struct amdgcn_param32 base;
			int off = 512 + m->sreg.stack_off + m->insn.off;

			knod_vset32(&base, knod_bpf_lds_vreg(priv,
						       KNOD_AMDGPU_LDS_BASE_VREG));
			knod_emit(priv, head, ds_read_b32, bpf_reg64[dst].lo,
				  base, knod_bpf_lds_off(priv, m, off));
			knod_iset32(&value, 0);
			knod_mov32(priv, head, bpf_reg64[dst].hi, value);
			pending_lds |= effect.defs;
			continue;
		}
		if (BPF_CLASS(m->insn.code) == BPF_LDX) {
			knod_emit(priv, head, global_load_dword, bpf_reg64[dst].lo,
				  bpf_reg64[m->insn.src_reg].lo, m->insn.off);
			knod_iset32(&value, 0);
			knod_mov32(priv, head, bpf_reg64[dst].hi, value);
			pending |= effect.defs;
			continue;
		}
		if (BPF_OP(m->insn.code) != BPF_MOV) {
			knod_bpf_emit_local_bitwise(priv, head, &m->insn);
			continue;
		}
		if (BPF_SRC(m->insn.code) == BPF_X)
			value = bpf_reg64[m->insn.src_reg].lo;
		else
			knod_iset32(&value, m->insn.imm);
		knod_mov32(priv, head, bpf_reg64[dst].lo, value);
		if (BPF_CLASS(m->insn.code) == BPF_ALU)
			knod_iset32(&value, 0);
		else if (BPF_SRC(m->insn.code) == BPF_X)
			value = bpf_reg64[m->insn.src_reg].hi;
		else
			knod_iset32(&value, m->insn.imm < 0 ? -1 : 0);
		knod_mov32(priv, head, bpf_reg64[dst].hi, value);
	}
	if (pending)
		knod_wait_vmcnt(priv, head);
	if (pending_lds)
		knod_emit(priv, head, s_waitcnt_lgkmcnt);
}

/* One bounded pair of predecessor chains, certified on the final CFG. */
#define KNOD_MP_PREFIX 128
#define KNOD_MP_PATH 24
struct knod_mp_state {
	struct knod_wr_expr reg[11];
	s16 bound;
	bool reached;
};
struct knod_mp_workspace {
	struct knod_insn_meta *node[KNOD_MP_PREFIX];
	struct knod_mp_state in[KNOD_MP_PREFIX];
	short edge[KNOD_MP_PREFIX][2];
	unsigned char color[KNOD_MP_PREFIX], depth[KNOD_MP_PREFIX];
	unsigned count, producers, visits;
	struct knod_insn_meta *producer[2];
};
static void knod_mp_join(struct knod_mp_state *d, const struct knod_mp_state *s)
{
	unsigned r;
	if (!d->reached) {
		*d = *s;
		return;
	}
	for (r = 0; r < 11; r++)
		if (d->reg[r].kind != s->reg[r].kind ||
		    d->reg[r].off != s->reg[r].off)
			d->reg[r] = (struct knod_wr_expr){};
	if (s->bound < d->bound)
		d->bound = s->bound;
}
static bool knod_mp_certificate(struct knod_prog *prog,
				struct knod_insn_meta *tail,
				struct knod_mp_workspace *w)
{
	struct knod_insn_meta *m;
	unsigned i, j, count = 0, linear = 0;
	memset(w, 0, sizeof(*w));
	list_for_each_entry(m, &prog->insns, l)
	{
		if (m->linear_idx != linear++ || m->subprog_idx)
			return false;
		if (count == KNOD_MP_PREFIX)
			return false;
		w->node[count++] = m;
		if (m == tail)
			break;
	}
	if (!count || w->node[count - 1] != tail)
		return false;
	w->count = count;
	/* Classified final graph only; include synthetic routing and reject
	 * backward entry. */
	list_for_each_entry(m, &prog->insns, l)
	{
		unsigned cls = BPF_CLASS(m->insn.code),
			 op = BPF_OP(m->insn.code);
		if ((cls == BPF_JMP || cls == BPF_JMP32) && op != BPF_CALL &&
		    op != BPF_EXIT) {
			if (!m->merge_point ||
			    m->merge_point->linear_idx <= m->linear_idx)
				return false;
			if (m->bpf_insn_idx < 0 &&
			    (m->insn.code != (BPF_JMP | BPF_JA) ||
			     m->jmp_dst != m->merge_point))
				return false;
			if (m->linear_idx >= count &&
			    m->merge_point->linear_idx < count)
				return false;
		}
	}
	for (i = 0; i < count; i++) {
		const struct bpf_insn *b = &w->node[i]->insn;
		unsigned cls = BPF_CLASS(b->code), op = BPF_OP(b->code);
		w->edge[i][0] = i + 1 < count ? (short)(i + 1) : -1;
		w->edge[i][1] = -1;
		if ((cls == BPF_JMP || cls == BPF_JMP32) && op == BPF_EXIT)
			w->edge[i][0] = -1;
		else if ((cls == BPF_JMP || cls == BPF_JMP32) &&
			 op != BPF_CALL) {
			struct knod_insn_meta *t = w->node[i]->merge_point;
			if (w->node[i]->branch_type != KNOD_BR_FORWARD_SKIP &&
			    w->node[i]->branch_type != KNOD_BR_FORWARD_GOTO &&
			    w->node[i]->branch_type != KNOD_BR_DIRECT_EXIT)
				return false;
			if (w->node[i]->branch_type != KNOD_BR_DIRECT_EXIT &&
			    (!t->is_merge_point ||
			     (op == BPF_JA ? w->node[i]->branch_type !=
						 KNOD_BR_FORWARD_GOTO
					   : w->node[i]->branch_type !=
						 KNOD_BR_FORWARD_SKIP)))
				return false;
			if (t->linear_idx < count) {
				if (w->node[t->linear_idx] != t)
					return false;
				w->edge[i][1] = t->linear_idx;
			}
			if (op == BPF_JA)
				w->edge[i][0] = -1;
		}
	}
	w->in[0].reached = true;
	w->in[0].reg[1].kind = 3;
	for (i = 0; i < count; i++) {
		struct knod_mp_state st = w->in[i], fall;
		const struct bpf_insn *b = &w->node[i]->insn;
		unsigned cls = BPF_CLASS(b->code), op = BPF_OP(b->code),
			 d = b->dst_reg, s = b->src_reg;
		int off;
		if (!st.reached)
			continue;
		if (d > 10 || s > 10)
			return false;
		if (!b->code) {
			if (!i ||
			    w->node[i - 1]->insn.code !=
				(BPF_LD | BPF_DW | BPF_IMM) ||
			    w->node[i]->bpf_insn_idx !=
				w->node[i - 1]->bpf_insn_idx + 1 ||
			    b->dst_reg || b->src_reg || b->off ||
			    w->node[i]->is_merge_point)
				return false;
		} else if (b->code == (BPF_LDX | BPF_MEM | BPF_W) &&
			   w->node[i]->ptr.type == PTR_TO_CTX &&
			   st.reg[s].kind == 3 && !st.reg[s].off &&
			   (b->off == 0 || b->off == 4))
			st.reg[d] =
			    (struct knod_wr_expr){.kind = b->off ? 2 : 1};
		else if (b->code == (BPF_ALU64 | BPF_MOV | BPF_X))
			st.reg[d] = st.reg[s];
		else if (b->code == (BPF_ALU64 | BPF_ADD | BPF_K) &&
			 st.reg[d].kind && b->imm >= -2048 && b->imm <= 2048) {
			off = st.reg[d].off + b->imm;
			st.reg[d] =
			    off >= -2048 && off <= 2048
				? (struct knod_wr_expr){off, st.reg[d].kind}
				: (struct knod_wr_expr){};
		} else if (cls == BPF_LDX || cls == BPF_LD || cls == BPF_ALU ||
			   cls == BPF_ALU64)
			st.reg[d] = (struct knod_wr_expr){};
		else if ((cls == BPF_JMP || cls == BPF_JMP32) &&
			 op == BPF_CALL) {
			if (b->code != (BPF_JMP | BPF_CALL) || b->src_reg ||
			    b->imm != BPF_FUNC_map_lookup_elem)
				return false;
			for (j = 0; j < 6; j++)
				st.reg[j] = (struct knod_wr_expr){};
		} else if ((cls == BPF_ST || cls == BPF_STX) &&
			   BPF_MODE(b->code) != BPF_MEM)
			return false;
		fall = st;
		if (b->code == (BPF_JMP | BPF_JGT | BPF_X) &&
		    !w->node[i]->jump_neg_op && w->in[i].reg[d].kind == 1 &&
		    w->in[i].reg[s].kind == 2 && !w->in[i].reg[s].off &&
		    w->in[i].reg[d].off > fall.bound)
			fall.bound = w->in[i].reg[d].off;
		if (w->edge[i][0] >= 0)
			knod_mp_join(&w->in[w->edge[i][0]], &fall);
		if (w->edge[i][1] >= 0)
			knod_mp_join(&w->in[w->edge[i][1]], &st);
	}
	return w->in[count - 1].reached;
}
static bool knod_mp_cover(struct knod_mp_workspace *w)
{
	struct knod_insn_meta *tail = w->node[w->count - 1];
	unsigned at, base = tail->insn.src_reg, dst = tail->insn.dst_reg;
	w->color[w->count - 1] = 1;
	for (at = w->count; at-- > 0;) {
		struct knod_insn_meta *m = w->node[at];
		const struct bpf_insn *b = &m->insn;
		struct knod_bpf_effect e;
		unsigned i, k, preds = 0;
		if (!w->color[at])
			continue;
		if (w->depth[at] > KNOD_MP_PATH)
			return false;
		w->visits++;
		if (at != w->count - 1) {
			if (m->merge_read.role || m->sr.owner ||
			    !knod_bpf_load_pair_free(m))
				return false;
			/* Every selected producer path must reach tail; no side
			 * exit observes early dst. */
			if ((w->edge[at][0] >= 0) + (w->edge[at][1] >= 0) != 1)
				return false;
			if ((BPF_CLASS(b->code) == BPF_JMP ||
			     BPF_CLASS(b->code) == BPF_JMP32) &&
			    BPF_OP(b->code) != BPF_JA)
				return false;
			if (b->code == (BPF_LDX | BPF_MEM | BPF_H) &&
			    m->ptr.type == PTR_TO_PACKET &&
			    b->src_reg == base && b->dst_reg <= 9 && b->dst_reg != base &&
			    b->dst_reg != dst && b->off >= 0 &&
			    b->off + 2 == tail->insn.off) {
				if (!w->in[at].reached ||
				    w->in[at].reg[base].kind != 1 ||
				    w->in[at].reg[base].off ||
				    w->in[at].bound < tail->insn.off + 2)
					return false;
				if (w->producers == 2)
					return false;
				w->producer[w->producers++] = m;
				w->color[at] = 2;
				continue;
			}
			e = knod_bpf_effect(b);
			if (((e.uses | e.defs) & ((1U << dst) | (1U << base))))
				return false;
			if (e.barrier && !((BPF_CLASS(b->code) == BPF_JMP ||
					    BPF_CLASS(b->code) == BPF_JMP32) &&
					   BPF_OP(b->code) != BPF_CALL &&
					   BPF_OP(b->code) != BPF_EXIT))
				return false;
			if (BPF_CLASS(b->code) == BPF_LDX ||
			    BPF_CLASS(b->code) == BPF_LD)
				return false;
			if (BPF_CLASS(b->code) == BPF_ST ||
			    BPF_CLASS(b->code) == BPF_STX) {
				if (BPF_MODE(b->code) != BPF_MEM ||
				    m->ptr.type != PTR_TO_STACK)
					return false;
			}
		}

		for (i = 0; i < at; i++)
			for (k = 0; k < 2; k++)
				if (w->edge[i][k] == (int)at &&
				    w->in[i].reached) {
					preds++;
					w->color[i] = 1;
					if (w->depth[i] < w->depth[at] + 1)
						w->depth[i] = w->depth[at] + 1;
				}
		if (!preds)
			return false;
		w->color[at] = 2;
	}
	return true;
}
static bool knod_mp_try(struct knod_prog *prog, struct knod_insn_meta *tail,
			struct knod_mp_workspace *w)
{
	if (tail->insn.code != (BPF_LDX | BPF_MEM | BPF_H) ||
	    tail->ptr.type != PTR_TO_PACKET || tail->insn.src_reg > 9 ||
	    tail->insn.dst_reg > 9 ||
	    tail->insn.src_reg == tail->insn.dst_reg || tail->merge_read.role ||
	    !tail->is_merge_point || tail->sr.owner ||
	    !knod_bpf_load_pair_free(tail) || tail->insn.off < 2)
		return false;
	return knod_mp_certificate(prog, tail, w) && knod_mp_cover(w) &&
	       w->producers == 2;
}
static void knod_bpf_analyze_merge_reads(struct knod_bpf_priv *priv,
					 struct knod_prog *prog)
{
	struct knod_mp_workspace *w;
	struct knod_insn_meta *m;
	unsigned i;
	if (priv->isa_version != 10 || prog->jit_engine != 1 || !prog->wide_read_xdp)
		return;
	w = kzalloc(sizeof(*w), GFP_KERNEL);
	if (!w)
		return;
	list_for_each_entry(m, &prog->insns, l)
	{
		if (m->jit_engine != 1 || !knod_mp_try(prog, m, w))
			continue;
		if (w->producer[0]->jit_engine != 1 ||
		    w->producer[1]->jit_engine != 1)
			continue;
		/* All proof completed before publishing any selected metadata.
		 * One group per program avoids early definitions invalidating
		 * another certificate. */
		for (i = 0; i < 2; i++) {
			w->producer[i]->merge_read.off = m->insn.off;
			w->producer[i]->merge_read.dst = m->insn.dst_reg;
			w->producer[i]->merge_read.role = 1;
		}
		m->merge_read.role = 2;
		break;
	}
	kfree(w);
}
static void knod_bpf_emit_merge_read(struct knod_bpf_priv *priv,
				     struct knod_insn_meta *m)
{
	struct amdgcn_param32 zero, shift, mask;
	if (m->merge_read.role != 1)
		return;

	/* The certificate covers exactly these four private packet bytes.
	 * GFX10 native ABI12 queues enable UNALIGNED before dispatch.
	 */
	knod_emit(priv, m, global_load_dword, bpf_reg64[m->insn.dst_reg].lo,
		  bpf_reg64[m->insn.src_reg].lo, m->insn.off);
	knod_wait_vmcnt(priv, m);
	knod_iset32(&shift, 16);
	knod_emit(priv, m, v_lshrrev_b32, bpf_reg64[m->merge_read.dst].lo,
		  shift, bpf_reg64[m->insn.dst_reg].lo);
	knod_iset32(&mask, 0xffff);
	knod_emit(priv, m, v_and_b32_e32, bpf_reg64[m->insn.dst_reg].lo,
		  mask, bpf_reg64[m->insn.dst_reg].lo);
	knod_iset32(&zero, 0);
	knod_mov32(priv, m, bpf_reg64[m->insn.dst_reg].hi, zero);
	knod_mov32(priv, m, bpf_reg64[m->merge_read.dst].hi, zero);
}

static void knod_bpf_load_pair_group(struct knod_prog *prog,
		struct knod_insn_meta *a)
{
	struct knod_insn_meta *b, *c;
	int base = a->insn.src_reg, off;

	if (list_is_last(&a->l, &prog->insns))
		return;
	b = list_next_entry(a, l);
	if (list_is_last(&b->l, &prog->insns))
		return;
	c = list_next_entry(b, l);
	if (!knod_bpf_load_pair_free(a) || !knod_bpf_load_pair_free(b) ||
	    !knod_bpf_load_pair_free(c) || base > 9 ||
	    a->insn.code != (BPF_LDX | BPF_MEM | BPF_W) ||
	    c->insn.code != (BPF_LDX | BPF_MEM | BPF_W) ||
	    a->ptr.type != PTR_TO_PACKET || c->ptr.type != PTR_TO_PACKET ||
	    c->insn.src_reg != base || a->insn.dst_reg == base ||
	    a->insn.dst_reg > 9 || c->insn.dst_reg > 9 ||
	    b->insn.code != (BPF_STX | BPF_MEM | BPF_W) ||
	    b->ptr.type != PTR_TO_STACK || b->insn.dst_reg != BPF_REG_FP ||
	    b->dreg.stack_off + b->insn.off < -512 ||
	    b->dreg.stack_off + b->insn.off > -4 ||
	    b->insn.src_reg > 10 || ((b->dreg.stack_off + b->insn.off) & 3) ||
	    b->is_merge_point || c->is_merge_point ||
	    ((b->flags | c->flags) & FLAG_INSN_IS_JUMP_DST) ||
	    a->subprog_idx != b->subprog_idx || a->subprog_idx != c->subprog_idx ||
	    a->bpf_insn_idx < 0 || b->bpf_insn_idx != a->bpf_insn_idx + 1 ||
	    c->bpf_insn_idx != b->bpf_insn_idx + 1 ||
	    (c->insn.off != a->insn.off + 4 && a->insn.off != c->insn.off + 4))
		return;
	off = min(a->insn.off, c->insn.off);
	if (off < -2048 || off + 7 >= 2048)
		return;
	a->load_pair.first = c->load_pair.last = true;
	a->load_pair.owned = b->load_pair.owned = c->load_pair.owned = true;
	a->load_pair.off = off;
	a->load_pair.base = base;
	a->load_pair.word = (a->insn.off - off) / 4;
	c->load_pair.word = (c->insn.off - off) / 4;
}

static void knod_bpf_analyze_load_pairs(struct knod_bpf_priv *priv,
		struct knod_prog *prog)
{
	struct knod_insn_meta *m;
	list_for_each_entry(m, &prog->insns, l)
		memset(&m->load_pair, 0, sizeof(m->load_pair));
	if (priv->isa_version != 10)
		return;
	list_for_each_entry(m, &prog->insns, l)
		knod_bpf_load_pair_group(prog, m);
}

static void knod_bpf_emit_load_pair(struct knod_bpf_priv *priv,
		struct knod_insn_meta *m)
{
	struct amdgcn_param32 data, zero;
	if (m->load_pair.first) {
		knod_vset32(&data, KNOD_AMDGPU_TMP_VREG0_LO);
		knod_emit(priv, m, global_load_dwordx2, data,
			bpf_reg64[m->load_pair.base].lo, m->load_pair.off);
		knod_emit(priv, m, s_waitcnt_vmcnt);
	}
	knod_vset32(&data, KNOD_AMDGPU_TMP_VREG0_LO + m->load_pair.word);
	knod_mov32(priv, m, bpf_reg64[m->insn.dst_reg].lo, data);
	knod_iset32(&zero, 0);
	knod_mov32(priv, m, bpf_reg64[m->insn.dst_reg].hi, zero);
}

/* Bounded frontend proof: no runtime values or target ISA are inspected. */
static bool knod_bpf_addr_fact_boundary(const struct knod_insn_meta *m)
{
	return m->is_merge_point ||
		(m->flags & (FLAG_INSN_IS_JUMP_DST | FLAG_INSN_IS_SUBPROG_START));
}

static bool knod_bpf_key_literal(struct knod_prog *prog,
				struct knod_insn_meta *call, u32 *key)
{
	struct knod_insn_meta *m = call, *prev;
	int expected = call->bpf_insn_idx - 1, n, off;
	unsigned int width;
	u64 value;
	int key_off = call->kreg.stack_off;

	if (call->kreg.reg.type != PTR_TO_STACK || key_off < -512 || key_off > -4)
		return false;
	for (n = 0; n < 16; n++) {
		if (list_is_first(&m->l, &prog->insns))
			return false;
		m = list_prev_entry(m, l);
		if (m->bpf_insn_idx != expected-- || m->subprog_idx != call->subprog_idx ||
		    knod_bpf_addr_fact_boundary(m))
			return false;
		if (BPF_CLASS(m->insn.code) == BPF_JMP ||
		    BPF_CLASS(m->insn.code) == BPF_JMP32)
			return false;
		if (BPF_CLASS(m->insn.code) != BPF_ST &&
		    BPF_CLASS(m->insn.code) != BPF_STX)
			continue;
		if (BPF_MODE(m->insn.code) != BPF_MEM || m->ptr.type != PTR_TO_STACK ||
		    m->ptr.frameno != call->kreg.reg.frameno)
			return false;
		width = BPF_SIZE(m->insn.code) == BPF_B ? 1 :
			BPF_SIZE(m->insn.code) == BPF_H ? 2 :
			BPF_SIZE(m->insn.code) == BPF_W ? 4 : 8;
		off = m->dreg.stack_off + m->insn.off;
		if (off + (int)width <= key_off || off >= key_off + 4)
			continue;
		/* A partial latest write is ambiguous; do not scan through it. */
		if (off > key_off || off + (int)width < key_off + 4)
			return false;
		if (BPF_CLASS(m->insn.code) == BPF_ST) {
			value = (s64)m->insn.imm;
		} else {
			if (list_is_first(&m->l, &prog->insns))
				return false;
			prev = list_prev_entry(m, l);
			if (prev->bpf_insn_idx != m->bpf_insn_idx - 1 ||
			    prev->subprog_idx != call->subprog_idx ||
			    prev->insn.dst_reg != m->insn.src_reg)
				return false;
			if (prev->insn.code == (BPF_ALU64 | BPF_MOV | BPF_K))
				value = (s64)prev->insn.imm;
			else if (prev->insn.code == (BPF_ALU | BPF_MOV | BPF_K))
				value = (u32)prev->insn.imm;
			else
				return false;
		}
		*key = value >> ((key_off - off) * 8);
		return true;
	}
	return false;
}

static bool knod_bpf_percpu_addr_proven(struct knod_prog *prog,
				      struct knod_insn_meta *store)
{
	struct knod_insn_meta *m = store, *call = NULL, *lo, *hi;
	const struct bpf_map *map = store->dreg.reg.map_ptr;
	int expected = store->bpf_insn_idx - 1, n;
	bool null_guard = false;
	u64 literal;
	u32 key;

	if (!store->percpu_rmw_add || !store->percpu_rmw_uniform ||
	    store->insn.dst_reg != BPF_REG_0 || store->ptr.type != PTR_TO_MAP_VALUE ||
	    !map || map->map_type != BPF_MAP_TYPE_PERCPU_ARRAY || map->key_size != 4 ||
	    knod_bpf_addr_fact_boundary(store))
		return false;
	for (n = 0; n < 16; n++) {
		if (list_is_first(&m->l, &prog->insns))
			return false;
		m = list_prev_entry(m, l);
		if (m->bpf_insn_idx != expected-- || m->subprog_idx != store->subprog_idx ||
		    knod_bpf_addr_fact_boundary(m))
			return false;
		if (m->insn.code == (BPF_JMP | BPF_CALL)) {
			if (!null_guard || m->insn.src_reg || m->insn.imm != BPF_FUNC_map_lookup_elem)
				return false;
			call = m;
			break;
		}
		if (BPF_CLASS(m->insn.code) == BPF_JMP || BPF_CLASS(m->insn.code) == BPF_JMP32) {
			if (null_guard || m->insn.code != (BPF_JMP | BPF_JEQ | BPF_K) ||
			    m->insn.dst_reg != BPF_REG_0 || m->insn.imm || m->insn.off <= 0 ||
			    m->bpf_insn_idx + 1 + m->insn.off <= store->bpf_insn_idx)
				return false;
			null_guard = true;
			/* The guard must immediately follow the originating lookup. */
			continue;
		}
		if (null_guard || knod_bpf_effect(&m->insn).barrier ||
		    (knod_bpf_effect(&m->insn).defs & (1U << BPF_REG_0)))
			return false;
	}
	if (!call || list_is_first(&call->l, &prog->insns))
		return false;
	hi = list_prev_entry(call, l);
	if (list_is_first(&hi->l, &prog->insns))
		return false;
	lo = list_prev_entry(hi, l);
	/* Resolve the exact immutable map literal, not a stale verifier hint. */
	if (hi->bpf_insn_idx != call->bpf_insn_idx - 1 ||
	    lo->bpf_insn_idx != call->bpf_insn_idx - 2 || hi->insn.code ||
	    lo->insn.code != (BPF_LD | BPF_IMM | BPF_DW) ||
	    lo->insn.dst_reg != BPF_REG_1 || lo->insn.src_reg != BPF_PSEUDO_MAP_FD ||
	    knod_bpf_addr_fact_boundary(lo) || knod_bpf_addr_fact_boundary(hi))
		return false;
	literal = (u32)lo->insn.imm | ((u64)(u32)hi->insn.imm << 32);
	if (literal != (u64)(unsigned long)map || !knod_bpf_key_literal(prog, call, &key))
		return false;
	return key < map->max_entries;
}

static void knod_bpf_analyze_addr_proof(struct knod_bpf_priv *priv, struct knod_prog *prog)
{
	struct knod_insn_meta *m;

	list_for_each_entry(m, &prog->insns, l) {
		m->array_key_proven = false;
		m->array_key_literal = 0;
		if (priv->isa_version == 10 && m->jit_engine == 1 &&
		    !knod_bpf_addr_fact_boundary(m) &&
		    m->insn.code == (BPF_JMP | BPF_CALL) &&
		    !m->insn.src_reg && m->insn.imm == BPF_FUNC_map_lookup_elem)
			m->array_key_proven = knod_bpf_key_literal(prog, m,
							 &m->array_key_literal);
		m->percpu_addr_proven = false;
		if (priv->isa_version == 10)
			m->percpu_addr_proven = knod_bpf_percpu_addr_proven(prog, m);
	}
}

static int knod_bpf_jit(struct knod_dev *knodev,
			struct knod_prog *knod_prog)
{
	struct knod_bpf_priv *priv =
		(struct knod_bpf_priv *)knodev->accel->xdp.priv;
	short off, stack_off;
	struct knod_insn_meta *meta, *meta2;
	struct amdgcn_param64 param64[2];
	u32 insn_idx = 0;
	unsigned int copy_skip = 0, copied;
	struct amdgcn_param32 param[3];
	struct amdgcn_param32 p32[2];
	int s, d, imm, imm2;
	bool is_dw, fetch;
	bool skip = false;
	int atomic_op;
	int map_id;
	u64 imm64;
	u8 sreg;
	int ret;

	/* Analyze CFG before instruction emission */
	ret = knod_bpf_analyze_cfg(knod_prog);

	if (ret)
		return ret;

	knod_prog->jit_engine = knod_bpf_jit_engine;
	knod_prog->wide_read_xdp = knod_prog->type == BPF_PROG_TYPE_XDP;
	list_for_each_entry(meta, &knod_prog->insns, l)
		meta->jit_engine = knod_bpf_jit_engine;
	ret = knod_prog_prepare_insns(priv, knod_prog);
	if (ret)
		return ret;

	list_for_each_entry(meta, &knod_prog->insns, l) {
		memset(&meta->map_widen, 0, sizeof(meta->map_widen));
		memset(&meta->wide_read, 0, sizeof(meta->wide_read));
		memset(&meta->read_batch, 0, sizeof(meta->read_batch));
		memset(&meta->merge_read, 0, sizeof(meta->merge_read));
		meta->local_wait_member = false;
		meta->local_wait_defer = false;
		meta->local_issue_count = 0;
		memset(meta->local_issue_order, 0, sizeof(meta->local_issue_order));
		memset(&meta->load_pair, 0, sizeof(meta->load_pair));
		meta->percpu_dead = false;
		meta->percpu_delta_direct = false;
		memset(&meta->map_region, 0, sizeof(meta->map_region));
		meta->map_lds_head = false;
		meta->map_lds_tail = false;
	}

	knod_bpf_analyze_addr_proof(priv, knod_prog);
	knod_bpf_analyze_memory(knod_prog);
	knod_bpf_analyze_packet_regions(priv, knod_prog);
	knod_bpf_analyze_conststores(priv, knod_prog);
	knod_bpf_analyze_store_hoists(priv, knod_prog);
	knod_bpf_analyze_sinks(priv, knod_prog);
	knod_bpf_analyze_map_widen(priv, knod_prog);
	knod_bpf_analyze_wide_reads(priv, knod_prog);
	knod_bpf_analyze_read_batches(priv, knod_prog);
	knod_bpf_analyze_load_pairs(priv, knod_prog);
	knod_bpf_analyze_percpu_dead(priv, knod_prog);
	knod_bpf_analyze_store_regions(priv, knod_prog);
	knod_bpf_analyze_map_lds(priv, knod_prog);
	knod_bpf_analyze_map_regions(priv, knod_prog);
	knod_bpf_analyze_swapped_dead(priv, knod_prog);
	knod_bpf_analyze_merge_reads(priv, knod_prog);


	/* Fold the depth again from what will actually be emitted, on the same
	 * test the emitter makes, now that every pointer state is final.  The
	 * per-instruction folds above run inside the verifier hook, where a
	 * store's pointer may not have been seen yet.
	 */
	list_for_each_entry(meta, &knod_prog->insns, l) {
		int depth;

		if (meta->ptr.type != PTR_TO_STACK)
			continue;
		switch (BPF_CLASS(meta->insn.code)) {
		case BPF_LDX:
			depth = meta->sreg.stack_off + meta->insn.off;
			break;
		case BPF_STX:
		case BPF_ST:
			depth = meta->dreg.stack_off + meta->insn.off;
			break;
		default:
			continue;
		}
		if (knod_prog->max_stack_off > depth)
			knod_prog->max_stack_off = depth;
	}
	knod_prog->max_stack_off = -knod_prog->max_stack_off;
	knod_prog->max_stack_off = ALIGN(knod_prog->max_stack_off, 4);
	priv->lds_stack_base = 512 - knod_prog->max_stack_off;
	/* One slot past the top: the window moves two dwords at a time, so the
	 * highest slot's partner lands just beyond the stack, and it has to be
	 * a real, distinct place - not the wrap of a sixteen-bit offset back
	 * onto slot zero.
	 */
	knod_prog->lds_bytes = ALIGN((knod_prog->max_stack_off + 4) *
				     knod_bpf_workgroups, 1024);
	if (knod_prog->lds_bytes > priv->knod->lds_size) {
		pr_warn("knod_bpf: %d bytes of stack a lane times %u lanes is %u, more LDS than a workgroup has; use a smaller workgroup\n",
			knod_prog->max_stack_off, knod_bpf_workgroups,
			knod_prog->lds_bytes);
		return -E2BIG;
	}

	knod_bpf_analyze_local_waits(priv, knod_prog);

	/* Initialize all exec_save SGPRs to 0.
	 * Without this, merge points that restore from exec_save SGPRs
	 * of branches that were skipped (by an outer s_cbranch_execz)
	 * would OR garbage into EXEC, enabling invalid lanes.
	 * In the old code, BPF_EXIT used s_endpgm so execution never
	 * reached those merge points; now it does.
	 *
	 * The whole range, not the part this program uses, so that this is the
	 * same instructions for every program and can be built once.  It costs
	 * nothing: a wave declares all its registers whatever it does with
	 * them, so the ones past the end are not holding anyone back.
	 */
	meta = knod_prog_pre_last_meta(knod_prog);

	for (sreg = KNOD_BLOB_EXEC_SAVE_SREG;
	     sreg < KNOD_AMDGPU_EXEC_SAVE_SREG_MAX;
	     sreg += 2)
		knod_emit(priv, meta, s_mov_b64, sreg,
			  AMDGCN_SREG_INTEGER_0);

	insn_idx = 0;
	list_for_each_entry(meta, &knod_prog->pre_insns, l)
		insn_idx += knod_meta_bytes(meta) / 4;

	list_for_each_entry(meta, &knod_prog->insns, l) {
		if (copy_skip) {
			copy_skip--;
			meta->amdgpu_insn_idx = AMDGPU_INSN_SKIP;
			meta->amdgpu_insns = 0;
			meta->blob = NULL;
			meta->blob_size = 0;
			meta->blob_at = 0;
			continue;
		}
		if (skip) {
			skip = false;
			meta->amdgpu_insn_idx = AMDGPU_INSN_SKIP;
			continue;
		}
		s = meta->insn.src_reg;
		d = meta->insn.dst_reg;
		imm = meta->insn.imm;
		off = meta->insn.off;

		meta->amdgpu_insn_idx = insn_idx;
		meta->amdgpu_insns = 0;
		meta->sr.cursor = 0;
		meta->sr.error = false;
		/* Rewinding the cursor has to rewind the splice with it, or a
		 * second translation of the same metas keeps a routine from
		 * the first and puts it at an offset that no longer means
		 * anything.
		 */
		meta->blob = NULL;
		meta->blob_size = 0;
		meta->blob_at = 0;

		/* Structurized CFG: restore EXEC at merge points */
		if (meta->is_merge_point) {
			struct knod_insn_meta *br;

			list_for_each_entry(br, &knod_prog->insns, l) {
				if ((br->branch_type == KNOD_BR_FORWARD_SKIP ||
				     br->branch_type == KNOD_BR_FORWARD_GOTO) &&
				    br->merge_point == meta) {
					knod_emit(priv, meta, s_or_b64,
						  AMDGCN_SREG_EXEC_LO,
						  AMDGCN_SREG_EXEC_LO,
						  br->exec_save_sreg);
				}
			}
			/* Remove done lanes from restored EXEC */
			knod_emit(priv, meta, s_andn2_b64, AMDGCN_SREG_EXEC_LO,
				  AMDGCN_SREG_EXEC_LO,
				  knod_prog->done_mask_sreg);
				}

		if (meta->local_wait_member) {
			knod_bpf_emit_local_ready(priv, meta);
			goto insn_emitted;
		}

		if (meta->merge_read.role) {
			knod_bpf_emit_merge_read(priv, meta);
			goto insn_emitted;
		}
		if (meta->map_widen.owned) {
			knod_bpf_emit_map_widen(priv, meta);
			goto insn_emitted;
		}
		if (meta->wide_read.load) {
			knod_bpf_emit_wide_read(priv, meta);
			goto insn_emitted;
		}
		if (meta->read_batch.load) {
			knod_bpf_emit_read_batch(priv, meta);
			goto insn_emitted;
		}
		if (meta->percpu_dead)
			goto insn_emitted;
		if (meta->load_pair.first || meta->load_pair.last) {
			knod_bpf_emit_load_pair(priv, meta);
			goto insn_emitted;
		}
		if (meta->sink.member) {
			knod_bpf_emit_sink(priv, meta);
			goto insn_emitted;
		}
		if (meta->conststore.elide || meta->store_hoist.elide) {
			meta->amdgpu_insn_idx = AMDGPU_INSN_SKIP;
			meta->amdgpu_insns = 0;
			meta->blob = NULL;
			meta->blob_size = 0;
			meta->blob_at = 0;
			continue;
		}
		if (meta->conststore.bytes) {
			knod_bpf_emit_conststore(priv, meta);
			goto insn_emitted;
		}

		if (meta->store_hoist.emit) {
			knod_bpf_emit_store_hoist(priv, meta);
			goto insn_emitted;
		}

		copied = knod_bpf_emit_packet_region(priv, meta);
		if (!copied)
			copied = knod_bpf_emit_loads(priv, meta);
		if (!copied)
			copied = knod_bpf_emit_copy(priv, meta);
		if (!copied)
			copied = knod_bpf_emit_store(priv, meta);
		if (copied) {
			copy_skip = copied - 1;
			goto insn_emitted;
		}

		if (meta->packet_imm_valid) {
			knod_iset64(&p64[0], (u64)meta->packet_imm);
			knod_mov64(priv, meta, bpf_reg64[d], p64[0]);
			goto insn_emitted;
		}

		if (meta->percpu_rmw_add) {
			knod_bpf_emit_percpu_add(priv, meta);
			goto insn_emitted;
		}

		switch (meta->insn.code) {
		/* ALU
		 * If a destination register contains a pointer of STACK,
		 * offset should not be minus.
		 */
		case BPF_ALU | BPF_MOV | BPF_X:
			/* Nonzero offset encodes MOVSX, not an ordinary move. */
			if (off)
				return -EOPNOTSUPP;
			knod_mov32(priv, meta, bpf_reg64[d].lo, bpf_reg64[s].lo);
			knod_iset32(&p32[0], 0);
			knod_mov32(priv, meta, bpf_reg64[d].hi, p32[0]);
			break;
		case BPF_ALU64 | BPF_MOV | BPF_X:
			if (off)
				return -EOPNOTSUPP;
			knod_mov64(priv, meta, bpf_reg64[d], bpf_reg64[s]);
			break;
		case BPF_ALU | BPF_MOV | BPF_K:
			knod_iset64(&p64[0], (u32)imm);
			knod_mov64(priv, meta, bpf_reg64[d], p64[0]);
			break;
		case BPF_ALU64 | BPF_MOV | BPF_K:
			knod_iset64(&p64[0], imm);
			knod_mov64(priv, meta, bpf_reg64[d], p64[0]);
			break;
		case BPF_ALU | BPF_XOR | BPF_X:
			knod_xor32(priv, meta,
				       bpf_reg64[d].lo, bpf_reg64[d].lo,
				       bpf_reg64[s].lo);
			knod_iset64(&p64[0], 0);
			knod_mov32(priv, meta, bpf_reg64[d].hi, p64[0].lo);
			break;
		case BPF_ALU64 | BPF_XOR | BPF_X:
			//r[d] ^= r[s];
			knod_xor32(priv, meta,
				       bpf_reg64[d].lo, bpf_reg64[d].lo,
				       bpf_reg64[s].lo);
			knod_xor32(priv, meta,
				       bpf_reg64[d].hi, bpf_reg64[d].hi,
				       bpf_reg64[s].hi);
			break;
		case BPF_ALU | BPF_XOR | BPF_K:
		case BPF_ALU64 | BPF_XOR | BPF_K:
			knod_iset64(&p64[0], imm);
			/* VOP2's second source must be a VGPR. Keep the BPF
			 * operand live and put the immediate in the first slot.
			 */
			knod_xor32(priv, meta, bpf_reg64[d].lo,
				   p64[0].lo, bpf_reg64[d].lo);
			if (BPF_CLASS(meta->insn.code) == BPF_ALU64) {
				knod_xor32(priv, meta, bpf_reg64[d].hi,
					   p64[0].hi, bpf_reg64[d].hi);
			} else {
				knod_iset32(&p32[0], 0);
				knod_mov32(priv, meta, bpf_reg64[d].hi, p32[0]);
			}
			break;
		case BPF_ALU | BPF_MOD | BPF_X:
		case BPF_ALU64 | BPF_MOD | BPF_X:
			//r[d] %= r[s];
			knod_iset64(&p64[0], meta->umin_src);
			knod_mod(priv, meta, bpf_reg64[d], p64[0],
				     r64[0], r64[1], r64[2], r64[3], r64[4]);
			break;
		case BPF_ALU | BPF_MOD | BPF_K:
		case BPF_ALU64 | BPF_MOD | BPF_K:
			//r[d] %= imm;
			/* The dividend fits 32 bits (verifier rejects wider
			 * div/mod), so the 32-bit fold is valid even when
			 * clang emitted this as a 64-bit ALU op (e.g. u32
			 * hash % 65537 -> `r2 %= 65537`).
			 */
			if (meta->umax_dst <= U32_MAX && imm &&
			    knod_mod_k32(priv, meta, bpf_reg64[d], imm))
				break;
			knod_iset64(&p64[0], imm);
			knod_mod(priv, meta, bpf_reg64[d], p64[0],
				     r64[0], r64[1], r64[2], r64[3], r64[4]);
			break;
		case BPF_ALU | BPF_AND | BPF_X:
			knod_and32(priv, meta, bpf_reg64[d].lo,
				       bpf_reg64[d].lo, bpf_reg64[s].lo);
			knod_iset32(&p32[0], 0);
			knod_mov32(priv, meta, bpf_reg64[d].hi, p32[0]);
			break;
		case BPF_ALU64 | BPF_AND | BPF_X:
			//r[d] &= r[s];
			knod_and64(priv, meta, bpf_reg64[d],
				       bpf_reg64[d], bpf_reg64[s]);
			break;
		case BPF_ALU | BPF_AND | BPF_K:
		case BPF_ALU64 | BPF_AND | BPF_K:
			//r[d] &= imm;
			knod_iset32(&p32[0], imm);
			knod_and32(priv, meta, bpf_reg64[d].lo, p32[0],
				       bpf_reg64[d].lo);
			/* ALU64 immediates sign-extend: a negative mask keeps high. */
			if (BPF_CLASS(meta->insn.code) == BPF_ALU || imm >= 0) {
				knod_iset32(&p32[0], 0);
				knod_mov32(priv, meta, bpf_reg64[d].hi, p32[0]);
			}
			break;
		case BPF_ALU | BPF_OR | BPF_X:
			knod_or32(priv, meta, bpf_reg64[d].lo,
				  bpf_reg64[d].lo, bpf_reg64[s].lo);
			knod_iset32(&p32[0], 0);
			knod_mov32(priv, meta, bpf_reg64[d].hi, p32[0]);
			break;
		case BPF_ALU64 | BPF_OR | BPF_X:
			//r[d] |= r[s];
			knod_or32(priv, meta, bpf_reg64[d].lo,
				  bpf_reg64[d].lo, bpf_reg64[s].lo);
			knod_or32(priv, meta, bpf_reg64[d].hi,
				  bpf_reg64[d].hi, bpf_reg64[s].hi);
			break;
		case BPF_ALU | BPF_OR | BPF_K:
		case BPF_ALU64 | BPF_OR | BPF_K:
			//r[d] |= imm;
			knod_iset32(&p32[0], imm);
			knod_or32(priv, meta,
				bpf_reg64[d].lo, p32[0], bpf_reg64[d].lo);
			/* Positive ALU64 OR preserves high; negative OR sets it. */
			if (BPF_CLASS(meta->insn.code) == BPF_ALU || imm < 0) {
				knod_iset32(&p32[0],
					    BPF_CLASS(meta->insn.code) == BPF_ALU ? 0 : U32_MAX);
				knod_mov32(priv, meta, bpf_reg64[d].hi, p32[0]);
			}
			break;
		case BPF_ALU | BPF_ADD | BPF_X:
			knod_add32(priv, meta, bpf_reg64[d].lo,
				       bpf_reg64[d].lo, bpf_reg64[s].lo);
			knod_iset32(&p32[0], 0);
			knod_mov32(priv, meta, bpf_reg64[d].hi, p32[0]);
			break;
		case BPF_ALU64 | BPF_ADD | BPF_X:
			knod_add64(priv, meta, bpf_reg64[d],
				       bpf_reg64[d],
				       bpf_reg64[s]);

			//r[d] += r[s];
			break;
		case BPF_ALU | BPF_ADD | BPF_K:
			//r[d] += imm;
			knod_iset32(&p32[0], imm);
			knod_add32(priv, meta, bpf_reg64[d].lo,
				       p32[0], bpf_reg64[d].lo);
			knod_iset32(&p32[0], 0);
			knod_mov32(priv, meta, bpf_reg64[d].hi, p32[0]);
			break;
		case BPF_ALU64 | BPF_ADD | BPF_K:
			/* r[d] += imm, and the top half stays: this is what
			 * walks a pointer along, so clearing it puts the
			 * address somewhere else entirely.  The immediate is
			 * signed and widens to the whole register.
			 *
			 * It goes through a register first.  The add that
			 * carries reads VCC without being told to, and a
			 * literal cannot share an instruction with that.
			 */
			knod_iset64(&p64[0], (u64)(s64)imm);
			knod_mov64(priv, meta, r64[0], p64[0]);
			knod_add64(priv, meta, bpf_reg64[d], bpf_reg64[d],
				       r64[0]);
			break;
		case BPF_ALU | BPF_SUB | BPF_X:
			//r[d] -= r[s];
			knod_sub32(priv, meta, bpf_reg64[d].lo,
				       bpf_reg64[d].lo, bpf_reg64[s].lo);
			knod_iset32(&p32[0], 0);
			knod_mov32(priv, meta, bpf_reg64[d].hi, p32[0]);
			break;
		case BPF_ALU64 | BPF_SUB | BPF_X:
			//r[d] -= r[s];

			knod_sub64(priv, meta, bpf_reg64[d], bpf_reg64[d],
				       bpf_reg64[s]);
			break;
		case BPF_ALU | BPF_SUB | BPF_K:
			//r[d] -= imm;
			knod_iset64(&p64[0], (u64)(u32)imm);
			knod_mov64(priv, meta, r64[0], p64[0]);
			knod_sub64(priv, meta, bpf_reg64[d], bpf_reg64[d],
				       r64[0]);
			knod_iset32(&p32[0], 0);
			knod_mov32(priv, meta, bpf_reg64[d].hi, p32[0]);
			break;
		case BPF_ALU64 | BPF_SUB | BPF_K:
			/* r[d] -= imm, through a register for the same reason
			 * as the add above.
			 */
			knod_iset64(&p64[0], (u64)(s64)imm);
			knod_mov64(priv, meta, r64[0], p64[0]);
			knod_sub64(priv, meta, bpf_reg64[d], bpf_reg64[d],
				       r64[0]);
			break;
		case BPF_ALU | BPF_MUL | BPF_X:
			knod_mul_lo32(priv, meta, bpf_reg64[d].lo,
					  bpf_reg64[d].lo, bpf_reg64[s].lo);
			break;
		case BPF_ALU64 | BPF_MUL | BPF_X:
			//r[d] *= r[s];
			knod_mov64(priv, meta, r64[0], bpf_reg64[d]);
			knod_mov64(priv, meta, r64[1], bpf_reg64[s]);
			knod_mul64(priv, meta,
				       bpf_reg64[d],
				       r64[0],
				       r64[1],
				       r64[2]);
			break;
		case BPF_ALU | BPF_MUL | BPF_K:
			knod_iset32(&p32[0], imm);
			knod_mul_lo32(priv, meta, bpf_reg64[d].lo,
					  p32[0], bpf_reg64[d].lo);
			knod_iset32(&p32[0], 0);
			knod_mov32(priv, meta, bpf_reg64[d].hi, p32[0]);
			break;
		case BPF_ALU64 | BPF_MUL | BPF_K:
			//r[d] *= imm;
			knod_iset64(&p64[0], imm);
			knod_mov64(priv, meta, r64[0], bpf_reg64[d]);
			knod_mov64(priv, meta, r64[1], p64[0]);
			knod_mul64(priv, meta,
				       bpf_reg64[d],
				       r64[0],
				       r64[1],
				       r64[2]);
			break;
		case BPF_ALU | BPF_DIV | BPF_X:
		case BPF_ALU64 | BPF_DIV | BPF_X:
			//r[d] /= r[s];
			knod_iset64(&p64[0], meta->umin_src);
			knod_div(priv, meta, bpf_reg64[d], p64[0],
				     r64[0], r64[1], r64[2], r64[3]);
			break;
		case BPF_ALU | BPF_DIV | BPF_K:
		case BPF_ALU64 | BPF_DIV | BPF_K:
			//r[d] /= imm;
			knod_iset64(&p64[0], imm);
			knod_div(priv, meta, bpf_reg64[d], p64[0],
				     r64[0], r64[1], r64[2], r64[3]);
			break;
		case BPF_ALU | BPF_NEG:
			knod_iset32(&p32[0], 0);
			knod_sub32(priv, meta, bpf_reg64[d].lo, p32[0],
				       bpf_reg64[d].lo);
			knod_iset32(&p32[0], 0);
			knod_mov32(priv, meta, bpf_reg64[d].hi, p32[0]);
			break;
		case BPF_ALU64 | BPF_NEG:
			//r[d] = -r[d];
			WARN_ON_ONCE(1);
			break;
		case BPF_ALU | BPF_LSH | BPF_X:
			knod_lshlrev32(priv, meta, bpf_reg64[d].lo,
					   bpf_reg64[s].lo, bpf_reg64[d].lo);
			knod_iset32(&p32[0], 0);
			knod_mov32(priv, meta, bpf_reg64[d].hi, p32[0]);
			break;
		case BPF_ALU64 | BPF_LSH | BPF_X:
			//r[d] <<= r[s];
			knod_lshlrev64(priv, meta, bpf_reg64[d],
					   bpf_reg64[s], bpf_reg64[d]);
			break;
		case BPF_ALU | BPF_LSH | BPF_K:
			knod_iset32(&p32[0], imm);
			knod_lshlrev32(priv, meta, bpf_reg64[d].lo, p32[0],
					   bpf_reg64[d].lo);
			knod_iset32(&p32[0], 0);
			knod_mov32(priv, meta, bpf_reg64[d].hi, p32[0]);
			break;
		case BPF_ALU64 | BPF_LSH | BPF_K:
			//r[d] <<= imm;
			knod_iset64(&p64[0], imm);
			knod_lshlrev64(priv, meta, bpf_reg64[d], p64[0],
					   bpf_reg64[d]);
			break;
		case BPF_ALU | BPF_RSH | BPF_X:
			knod_lshrrev32(priv, meta, bpf_reg64[d].lo,
				       bpf_reg64[s].lo,
				       bpf_reg64[d].lo);
			knod_iset32(&p32[0], 0);
			knod_mov32(priv, meta, bpf_reg64[d].hi, p32[0]);
			break;
		case BPF_ALU64 | BPF_RSH | BPF_X:
			//r[d] >>= r[s];
			knod_lshrrev64(priv, meta, bpf_reg64[d],
					   bpf_reg64[s], bpf_reg64[d]);
			break;
		case BPF_ALU | BPF_RSH | BPF_K:
			knod_iset32(&p32[0], imm);
			knod_lshrrev32(priv, meta, bpf_reg64[d].lo, p32[0],
					   bpf_reg64[d].lo);
			knod_iset32(&p32[0], 0);
			knod_mov32(priv, meta, bpf_reg64[d].hi, p32[0]);
			break;
		case BPF_ALU64 | BPF_RSH | BPF_K:
			//r[d] >>= imm;
			knod_iset64(&p64[0], imm);
			knod_lshrrev64(priv, meta, bpf_reg64[d],
					   p64[0], bpf_reg64[d]);
			break;
		case BPF_ALU | BPF_ARSH | BPF_X:
			knod_ashrrev32(priv, meta, bpf_reg64[d].lo,
					   bpf_reg64[s].lo, bpf_reg64[d].lo);
			knod_iset32(&p32[0], 0);
			knod_mov32(priv, meta, bpf_reg64[d].hi, p32[0]);
			break;
		case BPF_ALU64 | BPF_ARSH | BPF_X:
			//r[d] >>= r[s];
			knod_ashrrev64(priv, meta, bpf_reg64[d],
					   bpf_reg64[s], bpf_reg64[d]);
			break;
		case BPF_ALU | BPF_ARSH | BPF_K:
			knod_iset32(&p32[0], imm);
			knod_ashrrev32(priv, meta, bpf_reg64[d].lo,
					   p32[0], bpf_reg64[d].lo);
			knod_iset32(&p32[0], 0);
			knod_mov32(priv, meta, bpf_reg64[d].hi, p32[0]);
			break;
		case BPF_ALU64 | BPF_ARSH | BPF_K:
			//r[d] >>= imm;
			knod_iset64(&p64[0], imm);
			knod_ashrrev64(priv, meta, bpf_reg64[d],
					   p64[0], bpf_reg64[d]);
			break;
		case BPF_LD | BPF_IMM | BPF_DW:
			meta2 = list_next_entry(meta, l);
			if (WARN_ON_ONCE(!meta2))
				return -EINVAL;
			imm2 = meta2->insn.imm;
			skip = true;
			imm64 = (u64)imm2 << 32 | (u32)imm;
			switch (s) {
			case 0x00:
				//r[d] = imm64;
				knod_mov64_imm(priv, meta, d * 2,
						   imm64);

				break;
			case 0x01:
				/* r[d] = param->maps[imm]; */
				imm64 = knod_bpf_get_map_gaddr(priv,
							       meta,
							       meta2);
				map_id = knod_bpf_get_map_id(priv,
							     meta,
							     meta2);
				knod_mov64_imm(priv, meta, d * 2,
						   imm64);
				break;
			default:
				WARN_ON_ONCE(1);
				break;
			}
			break;
			/* Legacy BPF packet access, not needed */
		case BPF_LD | BPF_ABS | BPF_B:
		case BPF_LD | BPF_ABS | BPF_H:
		case BPF_LD | BPF_ABS | BPF_W:
		case BPF_LD | BPF_IND | BPF_B:
		case BPF_LD | BPF_IND | BPF_H:
		case BPF_LD | BPF_IND | BPF_W:
			//err = pc | 0x0700;
			//exit = true;
			WARN_ON_ONCE(1);
			break;
		case BPF_LDX | BPF_MEM | BPF_B:
			if (meta->ptr.type == PTR_TO_STACK) {
				stack_off = meta->sreg.stack_off + off;
				knod_bpf_load_size(priv, meta,
						       &bpf_reg64[d],
						       &stack[0],
						       sizeof(unsigned char),
						       512 + stack_off);
			} else if (meta->ptr.type == PTR_TO_CTX) {
				knod_emit(priv, meta, global_load_ubyte,
					  bpf_reg64[d].lo,
					  bpf_reg64[s].lo, off * 2);
			} else if (meta->ptr.type == PTR_TO_MAP_VALUE) {
				knod_emit(priv, meta, global_load_ubyte,
					  bpf_reg64[d].lo,
					  bpf_reg64[s].lo, off);
			} else if (meta->ptr.type == PTR_TO_MAP_KEY) {
				knod_emit(priv, meta, global_load_ubyte,
					  bpf_reg64[d].lo,
					  bpf_reg64[s].lo, off);
			} else if (meta->ptr.type == PTR_TO_PACKET) {
				knod_emit(priv, meta, global_load_ubyte,
					  bpf_reg64[d].lo,
					  bpf_reg64[s].lo, off);
			} else {
				WARN_ON_ONCE(1);
			}
			knod_wait_vmcnt(priv, meta);
			if (priv->isa_version != 10 || meta->jit_engine != 1 ||
			    meta->ptr.type != PTR_TO_STACK) {
				knod_iset32(&p32[0], 0);
				knod_mov32(priv, meta, bpf_reg64[d].hi, p32[0]);
			}
			//ptr = (__global void *)r[s] + off;
			//r[d] = *(__global unsigned char *)ptr;
			break;
		case BPF_LDX | BPF_MEM | BPF_H:
			if (meta->ptr.type == PTR_TO_STACK) {
				stack_off = meta->sreg.stack_off + off;
				knod_bpf_load_size(priv, meta,
						       &bpf_reg64[d],
						       &stack[0],
						       sizeof(unsigned short),
						       512 + stack_off);
			} else if (meta->ptr.type == PTR_TO_CTX) {
				knod_emit(priv, meta, global_load_ushort,
					  bpf_reg64[d].lo,
					  bpf_reg64[s].lo, off * 2);
			} else if (meta->ptr.type == PTR_TO_MAP_VALUE) {
				knod_emit(priv, meta, global_load_ushort,
					  bpf_reg64[d].lo,
					  bpf_reg64[s].lo, off);
			} else if (meta->ptr.type == PTR_TO_MAP_KEY) {
				knod_emit(priv, meta, global_load_ushort,
					  bpf_reg64[d].lo,
					  bpf_reg64[s].lo, off);
			} else if (meta->ptr.type == SCALAR_VALUE) {
				knod_emit(priv, meta, global_load_ushort,
					  bpf_reg64[d].lo,
					  bpf_reg64[s].lo, off);
			} else if (meta->ptr.type == PTR_TO_PACKET) {
				knod_emit(priv, meta,
					  global_load_ushort,
					  bpf_reg64[d].lo,
					  bpf_reg64[s].lo, off);
			} else {
				knod_jit_err(" type = %d\n", meta->ptr.type);
				WARN_ON_ONCE(1);
			}
			//ptr = (__global void *)r[s] + off;
			//r[d] = *(__global unsigned short *)ptr;
			if (priv->isa_version != 10 || meta->jit_engine != 1 ||
			    meta->ptr.type != PTR_TO_STACK) {
				knod_iset32(&p32[0], 0);
				knod_mov32(priv, meta, bpf_reg64[d].hi, p32[0]);
			}
			knod_wait_vmcnt(priv, meta);
			break;
		case BPF_LDX | BPF_MEM | BPF_W:
			if (meta->ptr.type == PTR_TO_STACK) {
				stack_off = meta->sreg.stack_off + off;
				knod_bpf_load_size(priv, meta,
						       &bpf_reg64[d],
						       &stack[0],
						       sizeof(unsigned int),
						       512 + stack_off);
			} else if (meta->ptr.type == PTR_TO_CTX) {
				if (off == offsetof(struct xdp_md, data)) {
					knod_mov32(priv, meta,
						bpf_reg64[d].lo,
					    (struct amdgcn_param32){
					    .v = KNOD_AMDGPU_DATA_VREG_LO,
					    .type = AMDGCN_PARAM_TYPE_VGPR});
					knod_mov32(priv, meta,
						bpf_reg64[d].hi,
					    (struct amdgcn_param32){
					    .v = KNOD_AMDGPU_DATA_VREG_HI,
					    .type = AMDGCN_PARAM_TYPE_VGPR});
				} else if (off == offsetof(struct xdp_md,
							   data_end)) {
					knod_mov32(priv, meta,
						bpf_reg64[d].lo,
					    (struct amdgcn_param32){
					    .v = KNOD_AMDGPU_DATA_END_VREG_LO,
					    .type = AMDGCN_PARAM_TYPE_VGPR});
					knod_mov32(priv, meta,
						bpf_reg64[d].hi,
					    (struct amdgcn_param32){
					    .v = KNOD_AMDGPU_DATA_END_VREG_HI,
					    .type = AMDGCN_PARAM_TYPE_VGPR});
				} else {
					emit_global_load_dwordx2(
						priv->isa_version,
						&meta->amdgpu_insn[meta->amdgpu_insns],
						bpf_reg64[d].lo,
						bpf_reg64[s].lo,
						off * 2);
					meta->amdgpu_insns++;
				}
			} else if (meta->ptr.type == PTR_TO_MAP_VALUE) {
				knod_emit(priv, meta, global_load_dword,
					  bpf_reg64[d].lo,
					  bpf_reg64[s].lo, off);
			} else if (meta->ptr.type == PTR_TO_MAP_KEY) {
				knod_emit(priv, meta, global_load_dword,
					  bpf_reg64[d].lo,
					  bpf_reg64[s].lo, off);
			} else if (meta->ptr.type == PTR_TO_PACKET) {
				knod_emit(priv, meta, global_load_dword,
					  bpf_reg64[d].lo,
					  bpf_reg64[s].lo, off);
			} else {
				WARN_ON_ONCE(1);
			}
			//ptr = (__global void *)r[s] + off;
			//r[d] = *(__global unsigned int *)ptr;
			if (meta->ptr.type != PTR_TO_CTX &&
			    (priv->isa_version != 10 || meta->jit_engine != 1 ||
			    meta->ptr.type != PTR_TO_STACK)) {
				knod_iset32(&p32[0], 0);
				knod_mov32(priv, meta, bpf_reg64[d].hi,
					       p32[0]);
			}
			if (!meta->map_lds_head && !meta->local_wait_defer)
				knod_wait_vmcnt(priv, meta);
			break;
		case BPF_LDX | BPF_MEM | BPF_DW:
			if (meta->ptr.type == PTR_TO_STACK) {
				stack_off = meta->sreg.stack_off + off;
				knod_bpf_load_size(priv, meta,
						       &bpf_reg64[d],
						       &stack[0],
						       sizeof(unsigned long),
						       512+stack_off);
			} else if (meta->ptr.type == PTR_TO_CTX) {
				if (off == offsetof(struct xdp_md, data)) {
					knod_mov32(priv, meta,
						bpf_reg64[d].lo,
					    (struct amdgcn_param32){
					    .v = KNOD_AMDGPU_DATA_VREG_LO,
					    .type = AMDGCN_PARAM_TYPE_VGPR});
					knod_mov32(priv, meta,
						bpf_reg64[d].hi,
					    (struct amdgcn_param32){
					    .v = KNOD_AMDGPU_DATA_VREG_HI,
					    .type = AMDGCN_PARAM_TYPE_VGPR});
				} else if (off == offsetof(struct xdp_md,
							   data_end)) {
					knod_mov32(priv, meta,
						bpf_reg64[d].lo,
					    (struct amdgcn_param32){
					    .v = KNOD_AMDGPU_DATA_END_VREG_LO,
					    .type = AMDGCN_PARAM_TYPE_VGPR});
					knod_mov32(priv, meta,
						bpf_reg64[d].hi,
					    (struct amdgcn_param32){
					    .v = KNOD_AMDGPU_DATA_END_VREG_HI,
					    .type = AMDGCN_PARAM_TYPE_VGPR});
				} else {
					knod_emit(priv, meta,
						  global_load_dwordx2,
						  bpf_reg64[d].lo,
						  bpf_reg64[s].lo, off * 2);
				}
			} else if (meta->ptr.type == PTR_TO_MAP_VALUE) {
				knod_emit(priv, meta, global_load_dwordx2,
					  bpf_reg64[d].lo,
					  bpf_reg64[s].lo, off);
			} else if (meta->ptr.type == PTR_TO_MAP_KEY) {
				knod_emit(priv, meta, global_load_dwordx2,
					  bpf_reg64[d].lo,
					  bpf_reg64[s].lo, off);
			} else if (meta->ptr.type == PTR_TO_PACKET) {
				knod_emit(priv, meta,
					  global_load_dwordx2,
					  bpf_reg64[d].lo,
					  bpf_reg64[s].lo, off);
			} else {
				WARN_ON_ONCE(1);
			}
			//ptr = (__global void *)r[s] + off;
			//r[d] = *(__global unsigned long *)ptr;
			knod_wait_vmcnt(priv, meta);
			break;
		case BPF_STX | BPF_MEM | BPF_B:
			if (meta->ptr.type == PTR_TO_STACK) {
				stack_off = meta->dreg.stack_off + off;
				knod_bpf_store_cache_size(priv, meta,
						&bpf_reg64[s],
						&stack[0],
						sizeof(u8),
						512 + stack_off);
			} else if (meta->ptr.type == PTR_TO_CTX) {
				knod_emit(priv, meta, global_store_byte,
					  bpf_reg64[s].lo,
					  bpf_reg64[d].lo, off * 2);
			} else if (meta->ptr.type == PTR_TO_MAP_VALUE) {
				knod_emit(priv, meta, global_store_byte,
					  bpf_reg64[s].lo,
					  bpf_reg64[d].lo, off);
			} else if (meta->ptr.type == PTR_TO_MAP_KEY) {
				knod_emit(priv, meta, global_store_byte,
					  bpf_reg64[s].lo,
					  bpf_reg64[d].lo, off);
			} else if (meta->ptr.type == PTR_TO_PACKET) {
				knod_bpf_packet_store(priv, meta, 1, bpf_reg64[s].lo, d, off);
			} else {
				WARN_ON_ONCE(1);
			}
			break;
		case BPF_STX | BPF_MEM | BPF_H:
			if (meta->ptr.type == PTR_TO_STACK) {
				stack_off = meta->dreg.stack_off + off;
				knod_bpf_store_cache_size(priv, meta,
						&bpf_reg64[s],
						&stack[0],
						sizeof(u16),
						512 + stack_off);
			} else if (meta->ptr.type == PTR_TO_CTX) {
				knod_emit(priv, meta, global_store_short,
					  bpf_reg64[s].lo,
					  bpf_reg64[d].lo, off * 2);
			} else if (meta->ptr.type == PTR_TO_MAP_VALUE) {
				knod_emit(priv, meta, global_store_short,
					  bpf_reg64[s].lo,
					  bpf_reg64[d].lo, off);
			} else if (meta->ptr.type == PTR_TO_MAP_KEY) {
				knod_emit(priv, meta, global_store_short,
					  bpf_reg64[s].lo,
					  bpf_reg64[d].lo, off);
			} else if (meta->ptr.type == PTR_TO_PACKET) {
				knod_bpf_packet_store(priv, meta, 2, bpf_reg64[s].lo, d, off);
			} else {
				WARN_ON_ONCE(1);
			}
			break;
		case BPF_STX | BPF_MEM | BPF_W:
			if (meta->ptr.type == PTR_TO_STACK) {
				stack_off = meta->dreg.stack_off + off;
				knod_bpf_store_cache_size(priv, meta,
						&bpf_reg64[s],
						&stack[0],
						sizeof(u32),
						512 + stack_off);
			} else if (meta->ptr.type == PTR_TO_CTX) {
				knod_emit(priv, meta, global_store_dword,
					  bpf_reg64[s].lo,
					  bpf_reg64[d].lo, off * 2);
			} else if (meta->ptr.type == PTR_TO_MAP_VALUE) {
				knod_emit(priv, meta, global_store_dword,
					  bpf_reg64[s].lo,
					  bpf_reg64[d].lo, off);
			} else if (meta->ptr.type == PTR_TO_MAP_KEY) {
				knod_emit(priv, meta, global_store_dword,
					  bpf_reg64[s].lo,
					  bpf_reg64[d].lo, off);
			} else if (meta->ptr.type == PTR_TO_PACKET) {
				knod_bpf_packet_store(priv, meta, 4, bpf_reg64[s].lo, d, off);
			} else {
				WARN_ON_ONCE(1);
			}
			break;
		case BPF_STX | BPF_MEM | BPF_DW:
			if (meta->ptr.type == PTR_TO_STACK) {
				stack_off = meta->dreg.stack_off + off;
				knod_bpf_store_cache_size(priv, meta,
						&bpf_reg64[s],
						&stack[0],
						sizeof(u64),
						512 + stack_off);
			} else if (meta->ptr.type == PTR_TO_CTX) {
				knod_emit(priv, meta, global_store_dwordx2,
					  bpf_reg64[s].lo,
					  bpf_reg64[d].lo, off * 2);
			} else if (meta->ptr.type == PTR_TO_MAP_VALUE) {
				knod_emit(priv, meta, global_store_dwordx2,
					  bpf_reg64[s].lo,
					  bpf_reg64[d].lo, off);
			} else if (meta->ptr.type == PTR_TO_MAP_KEY) {
				knod_emit(priv, meta, global_store_dwordx2,
					  bpf_reg64[s].lo,
					  bpf_reg64[d].lo, off);
			} else if (meta->ptr.type == PTR_TO_PACKET) {
				knod_bpf_packet_store(priv, meta, 8, bpf_reg64[s].lo, d, off);
			} else {
				WARN_ON_ONCE(1);
			}
			break;
		case BPF_STX | BPF_ATOMIC | BPF_W:
		case BPF_STX | BPF_ATOMIC | BPF_DW:
			is_dw = BPF_SIZE(meta->insn.code) == BPF_DW;
			atomic_op = imm & ~BPF_FETCH;
			fetch = imm & BPF_FETCH;

			/*
			 * BPF atomic: *(dst_reg + off) op= src_reg
			 * If BPF_FETCH: src_reg = old value
			 * BPF_CMPXCHG: expect in r0, new in src_reg,
			 *   old value returned in r0.
			 *
			 * global_atomic_* with glc=1 returns old value in vdst.
			 * For non-FETCH ops use glc=0 (fire-and-forget).
			 *
			 * 64-bit atomics (global_atomic_*_x2) hang on GFX9
			 * VRAM. GFX10+ supports them.
			 */
			if (is_dw && priv->isa_version == 9) {
				pr_err("knod: 64-bit atomic not supported on GFX9\n");
				return -EOPNOTSUPP;
			}

			/*
			 * For CMPXCHG/FETCH: drain pending loads so addr/data
			 * VGPRs are ready. For non-fetch ADD wave reduction,
			 * addr was already waited for at map_lookup, and data
			 * is from ALU - no waitcnt needed.
			 */
			if (imm == BPF_CMPXCHG || fetch)
				knod_wait_vmcnt(priv, meta);

			if (imm == BPF_CMPXCHG) {
				/* cmpswap: data = {expect(r0), new(src)}.
				 * AMD cmpswap data reg pair must be
				 * consecutive:
				 *   32-bit: {cmp, new} = 2 consecutive VGPRs
				 *   64-bit: {cmp_lo, cmp_hi, new_lo, new_hi}
				 * Copy r0 and src into TMP consecutive pair.
				 */
				struct amdgcn_param32 tmp0_lo, tmp0_hi,
						      tmp1_lo, tmp1_hi;

				knod_vset32(&tmp0_lo,
					KNOD_AMDGPU_TMP_VREG0_LO);
				knod_vset32(&tmp0_hi,
					KNOD_AMDGPU_TMP_VREG0_HI);
				knod_vset32(&tmp1_lo,
					KNOD_AMDGPU_TMP_VREG1_LO);
				knod_vset32(&tmp1_hi,
					KNOD_AMDGPU_TMP_VREG1_HI);

				if (!is_dw) {
					/* TMP0_LO = r0 (expect),
					 * TMP0_HI = src (new)
					 */
					knod_mov32(priv, meta,
						       tmp0_lo,
						       bpf_reg64[0].lo);
					knod_mov32(priv, meta,
						       tmp0_hi,
						       bpf_reg64[s].lo);

					emit_global_atomic_cmpswap(
						priv->isa_version,
						&meta->amdgpu_insn[meta->amdgpu_insns],
						tmp0_lo, bpf_reg64[d].lo,
						tmp0_lo, off, 1);
					meta->amdgpu_insns++;
					knod_wait_vmcnt(priv, meta);
					/* Return old value in r0 */
					knod_mov32(priv, meta,
						       bpf_reg64[0].lo,
						       tmp0_lo);
				} else {
					/* 64-bit:
					 * {r0_lo, r0_hi, src_lo, src_hi}
					 */
					knod_mov32(priv, meta,
						       tmp0_lo,
						       bpf_reg64[0].lo);
					knod_mov32(priv, meta,
						       tmp0_hi,
						       bpf_reg64[0].hi);
					knod_mov32(priv, meta,
						       tmp1_lo,
						       bpf_reg64[s].lo);
					knod_mov32(priv, meta,
						       tmp1_hi,
						       bpf_reg64[s].hi);

					emit_global_atomic_cmpswap_x2(
						priv->isa_version,
						&meta->amdgpu_insn[meta->amdgpu_insns],
						tmp0_lo, bpf_reg64[d].lo,
						tmp0_lo, off, 1);
					meta->amdgpu_insns++;
					knod_wait_vmcnt(priv, meta);
					knod_mov32(priv, meta,
						       bpf_reg64[0].lo,
						       tmp0_lo);
					knod_mov32(priv, meta,
						       bpf_reg64[0].hi,
						       tmp0_hi);
				}
			} else if (!fetch && atomic_op == BPF_ADD) {
				/* One atomic per lane.  Counting the lanes and
				 * sending their total once needs them to be
				 * adding the same thing to the same place, and
				 * neither is known here: the amount is always a
				 * register, and the address is whatever each
				 * lane worked out.  It used to be folded anyway
				 * and put the wave's total on one lane's
				 * element.
				 */
				struct amdgcn_param32 v_tmp, v_tmp_hi;

				knod_vset32(&v_tmp, KNOD_AMDGPU_TMP_VREG0_LO);
				knod_vset32(&v_tmp_hi,
					    KNOD_AMDGPU_TMP_VREG0_HI);

				if (is_dw) {
					/* The pair the x2 atomic adds is one
					 * number, low half first.
					 */
					knod_emit(priv, meta, v_mov_b32_e32,
						  v_tmp, bpf_reg64[s].lo);
					knod_emit(priv, meta, v_mov_b32_e32,
						  v_tmp_hi, bpf_reg64[s].hi);
					knod_emit(priv, meta,
						  global_atomic_add_x2, v_tmp,
						  bpf_reg64[d].lo, v_tmp, off,
						  0);
				} else {
					knod_emit(priv, meta, global_atomic_add,
						  v_tmp, bpf_reg64[d].lo,
						  bpf_reg64[s].lo, off, 0);
				}
			} else {
				/* 64-bit: AND, OR, XOR, XCHG, or fetch ops */
				struct amdgcn_param32 vdst, data_p;

				if (fetch) {
					vdst = bpf_reg64[s].lo;
				} else {
					knod_vset32(&vdst,
						KNOD_AMDGPU_TMP_VREG0_LO);
				}
				data_p = bpf_reg64[s].lo;

				switch (atomic_op) {
				case BPF_ADD:
					emit_global_atomic_add_x2(
						priv->isa_version,
						&meta->amdgpu_insn[meta->amdgpu_insns],
						vdst, bpf_reg64[d].lo,
						data_p, off, fetch);
					break;
				case BPF_AND:
					emit_global_atomic_and_x2(
						priv->isa_version,
						&meta->amdgpu_insn[meta->amdgpu_insns],
						vdst, bpf_reg64[d].lo,
						data_p, off, fetch);
					break;
				case BPF_OR:
					emit_global_atomic_or_x2(
						priv->isa_version,
						&meta->amdgpu_insn[meta->amdgpu_insns],
						vdst, bpf_reg64[d].lo,
						data_p, off, fetch);
					break;
				case BPF_XOR:
					emit_global_atomic_xor_x2(
						priv->isa_version,
						&meta->amdgpu_insn[meta->amdgpu_insns],
						vdst, bpf_reg64[d].lo,
						data_p, off, fetch);
					break;
				default: /* BPF_XCHG */
					emit_global_atomic_swap_x2(
						priv->isa_version,
						&meta->amdgpu_insn[meta->amdgpu_insns],
						vdst, bpf_reg64[d].lo,
						data_p, off, fetch);
					break;
				}
				meta->amdgpu_insns++;
				/* Always wait for atomic completion */
				knod_wait_vmcnt(priv, meta);
			}
			break;
		case BPF_ST | BPF_MEM | BPF_B:
			knod_iset32(&p32[0], imm);
			if (meta->ptr.type == PTR_TO_STACK) {
				stack_off = meta->dreg.stack_off + off;
				knod_iset64(&p64[0], imm);
				knod_bpf_store_cache_size(priv, meta,
						&p64[0],
						&stack[0],
						sizeof(u8),
						512 + stack_off);
			} else if (meta->ptr.type == PTR_TO_CTX) {
				knod_emit(priv, meta, global_store_byte, p32[0],
					  bpf_reg64[d].lo, off * 2);
			} else if (meta->ptr.type == PTR_TO_MAP_VALUE) {
				knod_emit(priv, meta, global_store_byte, p32[0],
					  bpf_reg64[d].lo, off);
			} else if (meta->ptr.type == PTR_TO_MAP_KEY) {
				knod_emit(priv, meta, global_store_byte, p32[0],
					  bpf_reg64[d].lo, off);
			} else if (meta->ptr.type == PTR_TO_PACKET) {
				knod_iset64(&p64[0], imm);
				knod_emit(priv, meta, global_store_byte,
					  p64[0].lo,
					  bpf_reg64[d].lo, off);
			} else {
				WARN_ON_ONCE(1);
			}
			break;
		case BPF_ST | BPF_MEM | BPF_H:
			knod_iset32(&p32[0], imm);
			if (meta->ptr.type == PTR_TO_STACK) {
				stack_off = meta->dreg.stack_off + off;
				knod_iset64(&p64[0], imm);
				knod_bpf_store_cache_size(priv, meta,
						&p64[0],
						&stack[0],
						sizeof(u16),
						512 + stack_off);
			} else if (meta->ptr.type == PTR_TO_CTX) {
				knod_emit(priv, meta, global_store_short,
					  p32[0], bpf_reg64[d].lo, off * 2);
			} else if (meta->ptr.type == PTR_TO_MAP_VALUE) {
				knod_emit(priv, meta, global_store_short,
					  p32[0], bpf_reg64[d].lo, off);
			} else if (meta->ptr.type == PTR_TO_MAP_KEY) {
				knod_emit(priv, meta, global_store_short,
					  p32[0], bpf_reg64[d].lo, off);
			} else if (meta->ptr.type == PTR_TO_PACKET) {
				knod_iset64(&p64[0], imm);
				if (priv->isa_version == 10 ||
				    priv->isa_version == 11) {
					knod_bpf_store_packet_imm(priv, meta,
						p64[0], bpf_reg64[d], off, 2);
				} else {
					knod_emit(priv, meta,
						  global_store_short, p64[0].lo,
						  bpf_reg64[d].lo, off);
				}
			} else {
				WARN_ON_ONCE(1);
			}
			break;
		case BPF_ST | BPF_MEM | BPF_W:
			knod_iset32(&p32[0], imm);
			if (meta->ptr.type == PTR_TO_STACK) {
				stack_off = meta->dreg.stack_off + off;
				knod_iset64(&p64[0], imm);
				knod_bpf_store_cache_size(priv, meta,
						&p64[0],
						&stack[0],
						sizeof(u32),
						512 + stack_off);
			} else if (meta->ptr.type == PTR_TO_CTX) {
				knod_emit(priv, meta, global_store_dword,
					  p32[0], bpf_reg64[d].lo, off * 2);
			} else if (meta->ptr.type == PTR_TO_MAP_VALUE) {
				knod_emit(priv, meta, global_store_dword,
					  p32[0], bpf_reg64[d].lo, off);
			} else if (meta->ptr.type == PTR_TO_MAP_KEY) {
				knod_emit(priv, meta, global_store_dword,
					  p32[0], bpf_reg64[d].lo, off);
			} else if (meta->ptr.type == PTR_TO_PACKET) {
				knod_iset64(&p64[0], imm);
				if (priv->isa_version == 10 ||
				    priv->isa_version == 11) {
					knod_bpf_store_packet_imm(priv, meta,
						p64[0], bpf_reg64[d], off, 4);
				} else {
					knod_emit(priv, meta,
						  global_store_dword, p64[0].lo,
						  bpf_reg64[d].lo, off);
				}
			} else {
				WARN_ON_ONCE(1);
			}
			break;
		case BPF_ST | BPF_MEM | BPF_DW:
			knod_iset32(&p32[0], imm);
			if (meta->ptr.type == PTR_TO_STACK) {
				stack_off = meta->dreg.stack_off + off;
				knod_iset64(&p64[0], imm);
				knod_bpf_store_cache_size(priv, meta,
						&p64[0],
						&stack[0],
						sizeof(u64),
						512 + stack_off);
			} else if (meta->ptr.type == PTR_TO_CTX) {
				knod_emit(priv, meta, global_store_dwordx2,
					  p32[0], bpf_reg64[d].lo, off * 2);
			} else if (meta->ptr.type == PTR_TO_MAP_VALUE) {
				knod_emit(priv, meta, global_store_dwordx2,
					  p32[0], bpf_reg64[d].lo, off);
			} else if (meta->ptr.type == PTR_TO_MAP_KEY) {
				knod_emit(priv, meta, global_store_dwordx2,
					  p32[0], bpf_reg64[d].lo, off);
			} else if (meta->ptr.type == PTR_TO_PACKET) {
				knod_iset64(&p64[0], imm);
				if (priv->isa_version == 10 ||
				    priv->isa_version == 11) {
					knod_bpf_store_packet_imm(priv, meta,
						p64[0], bpf_reg64[d], off, 8);
				} else {
					knod_emit(priv, meta,
						  global_store_dwordx2,
						  p64[0].lo,
						  bpf_reg64[d].lo, off);
				}
				knod_iset32(&p32[0], imm);
			} else {
				WARN_ON_ONCE(1);
			}
			break;
		case BPF_JMP32 | BPF_JA | BPF_K:
			if (meta->branch_type == KNOD_BR_DIRECT_EXIT) {
				knod_bpf_emit_direct_exit_retval(priv, meta,
						meta->merge_point);

				/* Unconditional goto exit:
				 * all active lanes done
				 */
				knod_emit(priv, meta, s_or_b64,
					  knod_prog->done_mask_sreg,
					  knod_prog->done_mask_sreg,
					  AMDGCN_SREG_EXEC_LO);
				knod_emit(priv, meta, s_mov_b64,
					  AMDGCN_SREG_EXEC_LO,
					  AMDGCN_SREG_INTEGER_0);
			} else if (meta->branch_type == KNOD_BR_FORWARD_GOTO) {
				/* Structurized: save all active lanes, clear
				 * EXEC.  Lanes resume at merge_point (target).
				 */
				knod_emit(priv, meta, s_mov_b64,
					  meta->exec_save_sreg,
					  AMDGCN_SREG_EXEC_LO);
				knod_emit(priv, meta, s_mov_b64,
					  AMDGCN_SREG_EXEC_LO,
					  AMDGCN_SREG_INTEGER_0);
			} else {
				/* Reorder classifies every JA as FORWARD_GOTO
				 * or DIRECT_EXIT; reaching here is a bug.
				 */
				WARN_ON_ONCE(1);
			}
			break;
		case BPF_JMP | BPF_JA | BPF_K:
			if (meta->branch_type == KNOD_BR_DIRECT_EXIT) {
				knod_bpf_emit_direct_exit_retval(priv, meta,
						meta->merge_point);

				/* Unconditional goto exit:
				 * all active lanes done
				 */
				knod_emit(priv, meta, s_or_b64,
					  knod_prog->done_mask_sreg,
					  knod_prog->done_mask_sreg,
					  AMDGCN_SREG_EXEC_LO);
				knod_emit(priv, meta, s_mov_b64,
					  AMDGCN_SREG_EXEC_LO,
					  AMDGCN_SREG_INTEGER_0);
			} else if (meta->branch_type == KNOD_BR_FORWARD_GOTO) {
				/* Structurized: save all active lanes, clear
				 * EXEC.  Lanes resume at merge_point (target).
				 */
				knod_emit(priv, meta, s_mov_b64,
					  meta->exec_save_sreg,
					  AMDGCN_SREG_EXEC_LO);
				knod_emit(priv, meta, s_mov_b64,
					  AMDGCN_SREG_EXEC_LO,
					  AMDGCN_SREG_INTEGER_0);
			} else {
				/* Reorder classifies every JA as FORWARD_GOTO
				 * or DIRECT_EXIT; reaching here is a bug.
				 */
				WARN_ON_ONCE(1);
			}
			break;
		case BPF_JMP32 | BPF_JEQ | BPF_K:
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_iset32(&param[1], imm);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_emit(priv, meta, v_cmp_eq_u32, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP | BPF_JEQ | BPF_K:
			if (knod_bpf_emit_known_zero_cmp(priv, knod_prog, meta)) {
				knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
				break;
			}
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_iset32(&param[1], imm);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_HI);
			knod_iset32(&param[1], (s32)imm < 0 ? U32_MAX : 0);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_emit(priv, meta, v_cmp_eq_u64, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP32 | BPF_JEQ | BPF_X:
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], s * 2);
			knod_emit(priv, meta, v_cmp_eq_u32, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP | BPF_JEQ | BPF_X:
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], s * 2);
			knod_emit(priv, meta, v_cmp_eq_u64, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP32 | BPF_JGT | BPF_K:
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_iset32(&param[1], imm);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_emit(priv, meta, v_cmp_gt_u32, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP | BPF_JGT | BPF_K:
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_iset32(&param[1], imm);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_HI);
			knod_iset32(&param[1], (s32)imm < 0 ? U32_MAX : 0);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset64(&param64[0], d * 2);
			knod_vset64(&param64[1], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_emit(priv, meta, v_cmp_gt_u64, param64[0],
				  param64[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP32 | BPF_JGT | BPF_X:
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], s * 2);
			knod_emit(priv, meta, v_cmp_gt_u32, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP | BPF_JGT | BPF_X:
			knod_vset64(&param64[0], d * 2);
			knod_vset64(&param64[1], s * 2);
			knod_emit(priv, meta, v_cmp_gt_u64, param64[0],
				  param64[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP32 | BPF_JGE | BPF_K:
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_iset32(&param[1], imm);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_emit(priv, meta, v_cmp_ge_u32, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP | BPF_JGE | BPF_K:
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_iset32(&param[1], imm);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_HI);
			knod_iset32(&param[1], (s32)imm < 0 ? U32_MAX : 0);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset64(&param64[0], d * 2);
			knod_vset64(&param64[1], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_emit(priv, meta, v_cmp_ge_u64, param64[0],
				  param64[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP32 | BPF_JGE | BPF_X:
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], s * 2);
			knod_emit(priv, meta, v_cmp_ge_u32, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP | BPF_JGE | BPF_X:
			knod_vset64(&param64[0], d * 2);
			knod_vset64(&param64[1], s * 2);
			knod_emit(priv, meta, v_cmp_ge_u64, param64[0],
				  param64[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP32 | BPF_JLT | BPF_K:
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_iset32(&param[1], imm);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_emit(priv, meta, v_cmp_lt_u32, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP | BPF_JLT | BPF_K:
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_iset32(&param[1], imm);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_HI);
			knod_iset32(&param[1], (s32)imm < 0 ? U32_MAX : 0);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset64(&param64[0], d * 2);
			knod_vset64(&param64[1], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_emit(priv, meta, v_cmp_lt_u64, param64[0],
				  param64[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP32 | BPF_JLT | BPF_X:
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], s * 2);
			knod_emit(priv, meta, v_cmp_lt_u32, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP | BPF_JLT | BPF_X:
			knod_vset64(&param64[0], d * 2);
			knod_vset64(&param64[1], s * 2);
			knod_emit(priv, meta, v_cmp_lt_u64, param64[0],
				  param64[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP32 | BPF_JLE | BPF_K:
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_iset32(&param[1], imm);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_emit(priv, meta, v_cmp_le_u32, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP | BPF_JLE | BPF_K:
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_iset32(&param[1], imm);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_HI);
			knod_iset32(&param[1], (s32)imm < 0 ? U32_MAX : 0);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset64(&param64[0], d * 2);
			knod_vset64(&param64[1], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_emit(priv, meta, v_cmp_le_u64, param64[0],
				  param64[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP32 | BPF_JLE | BPF_X:
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], s * 2);
			knod_emit(priv, meta, v_cmp_le_u32, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP | BPF_JLE | BPF_X:
			knod_vset64(&param64[0], d * 2);
			knod_vset64(&param64[1], s * 2);
			knod_emit(priv, meta, v_cmp_le_u64, param64[0],
				  param64[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP32 | BPF_JSGT | BPF_K:
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_iset32(&param[1], imm);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_emit(priv, meta, v_cmp_gt_i32, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP | BPF_JSGT | BPF_K:
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_iset32(&param[1], imm);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_HI);
			knod_iset32(&param[1], (s32)imm < 0 ? U32_MAX : 0);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset64(&param64[0], d * 2);
			knod_vset64(&param64[1], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_emit(priv, meta, v_cmp_gt_i64, param64[0],
				  param64[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP32 | BPF_JSGT | BPF_X:
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], s * 2);
			knod_emit(priv, meta, v_cmp_gt_i32, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP | BPF_JSGT | BPF_X:
			knod_vset64(&param64[0], d * 2);
			knod_vset64(&param64[1], s * 2);
			knod_emit(priv, meta, v_cmp_gt_i64, param64[0],
				  param64[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP32 | BPF_JSGE | BPF_K:
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_iset32(&param[1], imm);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_emit(priv, meta, v_cmp_ge_i32, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP | BPF_JSGE | BPF_K:
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_iset32(&param[1], imm);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_HI);
			knod_iset32(&param[1], (s32)imm < 0 ? U32_MAX : 0);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset64(&param64[0], d * 2);
			knod_vset64(&param64[1], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_emit(priv, meta, v_cmp_ge_i64, param64[0],
				  param64[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP32 | BPF_JSGE | BPF_X:
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], s * 2);
			knod_emit(priv, meta, v_cmp_ge_i32, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP | BPF_JSGE | BPF_X:
			knod_vset64(&param64[0], d * 2);
			knod_vset64(&param64[1], s * 2);
			knod_emit(priv, meta, v_cmp_ge_i64, param64[0],
				  param64[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP32 | BPF_JSLT | BPF_K:
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_iset32(&param[1], imm);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_emit(priv, meta, v_cmp_lt_i32, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP | BPF_JSLT | BPF_K:
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_iset32(&param[1], imm);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_HI);
			knod_iset32(&param[1], (s32)imm < 0 ? U32_MAX : 0);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset64(&param64[0], d * 2);
			knod_vset64(&param64[1], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_emit(priv, meta, v_cmp_lt_i64, param64[0],
				  param64[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP32 | BPF_JSLT | BPF_X:
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], s * 2);
			knod_emit(priv, meta, v_cmp_lt_i32, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP | BPF_JSLT | BPF_X:
			knod_vset64(&param64[0], d * 2);
			knod_vset64(&param64[1], s * 2);
			knod_emit(priv, meta, v_cmp_lt_i64, param64[0],
				  param64[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP32 | BPF_JSLE | BPF_K:
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_iset32(&param[1], imm);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_emit(priv, meta, v_cmp_le_i32, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP | BPF_JSLE | BPF_K:
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_iset32(&param[1], imm);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_HI);
			knod_iset32(&param[1], (s32)imm < 0 ? U32_MAX : 0);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset64(&param64[0], d * 2);
			knod_vset64(&param64[1], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_emit(priv, meta, v_cmp_le_i64, param64[0],
				  param64[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP32 | BPF_JSLE | BPF_X:
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], s * 2);
			knod_emit(priv, meta, v_cmp_le_i32, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP | BPF_JSLE | BPF_X:
			knod_vset64(&param64[0], d * 2);
			knod_vset64(&param64[1], s * 2);
			knod_emit(priv, meta, v_cmp_le_i64, param64[0],
				  param64[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP32 | BPF_JSET | BPF_K:
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_iset32(&param[1], imm);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], d * 2);
			knod_vset32(&param[2], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_emit(priv, meta, v_and_b32_e32, param[0],
				  param[1], param[2]);
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_emit(priv, meta, v_cmp_eq_u32, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP | BPF_JSET | BPF_K:
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_iset32(&param[1], imm);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_HI);
			knod_iset32(&param[1], (s32)imm < 0 ? U32_MAX : 0);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], d * 2);
			knod_vset32(&param[2], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_emit(priv, meta, v_and_b32_e32, param[0],
				  param[1], param[2]);
			knod_vset32(&param[0], (d * 2) + 1);
			knod_vset32(&param[1], (d * 2) + 1);
			knod_vset32(&param[2], KNOD_AMDGPU_TMP_VREG0_HI);
			knod_emit(priv, meta, v_and_b32_e32, param[0],
				  param[1], param[2]);
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_emit(priv, meta, v_cmp_eq_u64, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP32 | BPF_JSET | BPF_X:
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], d * 2);
			knod_vset32(&param[2], s * 2);
			knod_emit(priv, meta, v_and_b32_e32, param[0],
				  param[1], param[2]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP | BPF_JSET | BPF_X:
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], d * 2);
			knod_vset32(&param[2], s * 2);
			knod_emit(priv, meta, v_and_b32_e32, param[0],
				  param[1], param[2]);
			knod_vset32(&param[0], (d * 2) + 1);
			knod_vset32(&param[1], (d * 2) + 1);
			knod_vset32(&param[2], (s * 2) + 1);
			knod_emit(priv, meta, v_and_b32_e32, param[0],
				  param[1], param[2]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP32 | BPF_JNE | BPF_K:
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_iset32(&param[1], imm);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_emit(priv, meta, v_cmp_eq_u32, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP | BPF_JNE | BPF_K:
			if (knod_bpf_emit_known_zero_cmp(priv, knod_prog, meta)) {
				knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
				break;
			}
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_iset32(&param[1], imm);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], KNOD_AMDGPU_TMP_VREG0_HI);
			knod_iset32(&param[1], (s32)imm < 0 ? U32_MAX : 0);
			knod_emit(priv, meta, v_mov_b32_e32, param[0],
				  param[1]);
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], KNOD_AMDGPU_TMP_VREG0_LO);
			knod_emit(priv, meta, v_cmp_eq_u64, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP32 | BPF_JNE | BPF_X:
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], s * 2);
			knod_emit(priv, meta, v_cmp_eq_u32, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP | BPF_JNE | BPF_X:
			knod_vset32(&param[0], d * 2);
			knod_vset32(&param[1], s * 2);
			knod_emit(priv, meta, v_cmp_eq_u64, param[0],
				  param[1]);
			knod_bpf_emit_branch_tail(priv, meta, knod_prog, off);
			break;
		case BPF_JMP32 | BPF_CALL:
		case BPF_JMP | BPF_CALL:
			switch (imm) {
			case 1:
				if (map_id == -1) {
					WARN_ON_ONCE(1);
					break;
				}
				if (!knod_bpf_map_op(priv, meta, map_id,
						     KNOD_BLOB_OP_LOOKUP))
					return -EOPNOTSUPP;
				map_id = -1;
				break;
			case 2:
				if (map_id == -1) {
					WARN_ON_ONCE(1);
					break;
				}
				if (!knod_bpf_map_op(priv, meta, map_id,
						     KNOD_BLOB_OP_UPDATE))
					return -EOPNOTSUPP;
				map_id = -1;
				break;
			case 3:
				if (map_id == -1) {
					WARN_ON_ONCE(1);
					break;
				}
				if (!knod_bpf_map_op(priv, meta, map_id,
						     KNOD_BLOB_OP_DELETE))
					return -EOPNOTSUPP;
				map_id = -1;
				break;
			case 5:
				knod_bpf_ktime_get_ns(priv, meta);
				break;
			case 44:
				knod_bpf_xdp_adjust_head(priv, meta);
				break;
			case 65:
				knod_bpf_xdp_adjust_tail(priv, meta);
				break;
			default:
				WARN_ON_ONCE(1);
				break;
			}
			break;
		case BPF_JMP32 | BPF_EXIT:
		case BPF_JMP | BPF_EXIT:
			/* Structurized CFG: BPF_EXIT is NOT a terminator.
			 * Mark all active lanes as done and clear EXEC.
			 * Actual exit handling (retval store,
			 * PASS block, s_endpgm) is in the unified
			 * fallthrough EXIT at the end of the stream.
			 * This follows the LLVM StructurizeCFG model where
			 * all lanes must reach the single exit point.
			 */
			knod_emit(priv, meta, s_or_b64,
				  knod_prog->done_mask_sreg,
				  knod_prog->done_mask_sreg,
				  AMDGCN_SREG_EXEC_LO);

			knod_emit(priv, meta, s_mov_b64, AMDGCN_SREG_EXEC_LO,
				  AMDGCN_SREG_INTEGER_0);
			break;
		case BPF_ALU | BPF_END | BPF_TO_BE: {
			struct amdgcn_param32 v_dst_lo, v_dst_hi, v_tmp, s_sel;

			knod_vset32(&v_dst_lo, d * 2);
			knod_vset32(&v_dst_hi, d * 2 + 1);
			knod_vset32(&v_tmp, KNOD_AMDGPU_TMP_VREG0_LO);
			knod_sset32(&s_sel, KNOD_AMDGPU_TMP_SREG0_LO);

			switch (imm) {
			case 16:
				/* bswap16+zext: {0,0,byte0,byte1} */
				knod_iset32(&param[0], 0x0C0C0001);
				knod_emit(priv, meta, s_mov_b32, s_sel,
					  param[0]);

				knod_emit(priv, meta, v_perm_b32, v_dst_lo,
					  v_dst_lo, v_dst_lo, s_sel);

				knod_iset32(&param[0], 0);
				knod_emit(priv, meta, v_mov_b32_e32, v_dst_hi,
					  param[0]);
				break;
			case 32:
				/* bswap32+zext */
				knod_iset32(&param[0], 0x00010203);
				knod_emit(priv, meta, s_mov_b32, s_sel,
					  param[0]);

				knod_emit(priv, meta, v_perm_b32, v_dst_lo,
					  v_dst_lo, v_dst_lo, s_sel);

				knod_iset32(&param[0], 0);
				knod_emit(priv, meta, v_mov_b32_e32, v_dst_hi,
					  param[0]);
				break;
			case 64: {
				struct amdgcn_param32 v_src_hi;

				knod_vset32(&v_src_hi, d * 2 + 1);

				/* bswap32 selector */
				knod_iset32(&param[0], 0x00010203);
				knod_emit(priv, meta, s_mov_b32, s_sel,
					  param[0]);

				/* tmp = bswap32(lo) */
				knod_emit(priv, meta, v_perm_b32, v_tmp,
					  v_dst_lo, v_dst_lo, s_sel);

				/* new_lo = bswap32(hi) */
				knod_emit(priv, meta, v_perm_b32, v_dst_lo,
					  v_src_hi, v_src_hi, s_sel);

				/* new_hi = tmp (bswap32(old_lo)) */
				knod_emit(priv, meta, v_mov_b32_e32, v_dst_hi,
					  v_tmp);
				break;
			}
			default:
				WARN_ON_ONCE(1);
				break;
			}
			break;
		}
		case BPF_ALU | BPF_END | BPF_TO_LE: {
			struct amdgcn_param32 v_dst_lo, v_dst_hi;

			knod_vset32(&v_dst_lo, d * 2);
			knod_vset32(&v_dst_hi, d * 2 + 1);

			switch (imm) {
			case 16:
				knod_iset32(&param[0], 0xFFFF);
				knod_emit(priv, meta, v_and_b32_e32, v_dst_lo,
					  param[0], v_dst_lo);

				knod_iset32(&param[0], 0);
				knod_emit(priv, meta, v_mov_b32_e32, v_dst_hi,
					  param[0]);
				break;
			case 32:
				knod_iset32(&param[0], 0);
				knod_emit(priv, meta, v_mov_b32_e32, v_dst_hi,
					  param[0]);
				break;
			case 64:
				break;
			default:
				WARN_ON_ONCE(1);
				break;
			}
			break;
		}
		default:
			WARN_ON_ONCE(1);
			break;
		}

insn_emitted:

		if (meta->map_region.error || meta->sr.error || (meta->sr.owner && meta->sr.cursor != meta->sr.count))
			return -EINVAL; /* Discard incomplete capture generation. */
		WARN_ON(meta->amdgpu_insns >= KNOD_META_INSNS);
		insn_idx += knod_meta_bytes(meta) / 4;
	}

	return knod_bpf_emit_epilogue(priv, knod_prog);
}

static int knod_bpf_translate(struct bpf_prog *prog)
{
	struct knod_prog *knod_prog = prog->aux->offload->dev_priv;
	struct knod_dev *knodev = knod_prog->knodev;
	int ret;

	knod_bpf_map_setup(prog);
	ret = knod_bpf_jit(knodev, knod_prog);
	if (ret < 0) {
		pr_err("knod: failed to JIT: %d\n", ret);
		return ret;
	}

	knod_setup_bpf_prog(prog);

	return 0;
}

static void knod_bpf_destroy_prog(struct bpf_prog *prog)
{
	struct knod_prog *knod_prog = prog->aux->offload->dev_priv;
	struct knod_dev *knodev = knod_prog->knodev;
	struct knod_bpf_priv *priv = knodev->accel->xdp.priv;

	/*
	 * Normally the prog was already uninstalled (offload with a NULL prog
	 * flipped back to pass).  Guard the abnormal path where the prog is
	 * freed while still tracked: flip to pass first so the worker stops
	 * dispatching this code.  The compiled code lives in a kernel slot and
	 * is no longer read once we flip away; knod_prog is CPU-only IR the GPU
	 * never touches, so it is safe to free synchronously.
	 */
	if (priv && READ_ONCE(priv->prog) == prog) {
		WRITE_ONCE(priv->prog, NULL);
		knod_bpf_reload_pass(knodev);
	}
	knod_prog_free(knod_prog);
}

static const struct bpf_prog_offload_ops knod_bpf_dev_ops = {
	.insn_hook      = knod_bpf_verify_insn,
	.finalize       = knod_bpf_finalize,
	.prepare        = knod_bpf_verifier_prep,
	.translate      = knod_bpf_translate,
	.destroy        = knod_bpf_destroy_prog,
};

static int knod_bpf_setup_prog_hw_checks(struct knod_dev *knodev,
					 struct netdev_bpf *bpf)
{
	if (!bpf->prog)
		return 0;

	return 0;
}

static int knod_bpf_map_get_next_key(struct bpf_offloaded_map *offmap,
				     void *key, void *next_key)
{
	unsigned int *nkey = (unsigned int *)next_key;
	unsigned int *_key = (unsigned int *)key;

	if (offmap->map.map_type == BPF_MAP_TYPE_ARRAY ||
	    offmap->map.map_type == BPF_MAP_TYPE_PERCPU_ARRAY) {
		if (key == NULL)
			*nkey = 0;
		else
			*nkey = (*_key) + 1;

		if (*nkey >= offmap->map.max_entries)
			return -ENOENT;
	} else if (offmap->map.map_type == BPF_MAP_TYPE_HASH ||
		   offmap->map.map_type == BPF_MAP_TYPE_PERCPU_HASH) {
		if (key == NULL)
			return knod_bpf_map_hash_get_first_key(offmap,
							       next_key);
		else
			return knod_bpf_map_hash_get_next_key(offmap, key,
							      nkey);
	}

	return 0;
}

static int knod_bpf_map_lookup_elem(struct bpf_offloaded_map *offmap,
				       void *key, void *value)
{
	return __knod_bpf_map_lookup_elem(offmap, key, value);
}

static int knod_bpf_map_update_elem(struct bpf_offloaded_map *offmap,
				    void *key, void *value, u64 flags)
{
	return __knod_bpf_map_update_elem(offmap, key, value, flags);
}

static int knod_bpf_map_delete_elem(struct bpf_offloaded_map *offmap, void *key)
{
	return __knod_bpf_map_delete_elem(offmap, key);
}

static const struct bpf_map_dev_ops knod_bpf_map_ops = {
	.map_get_next_key       = knod_bpf_map_get_next_key,
	.map_lookup_elem        = knod_bpf_map_lookup_elem,
	.map_update_elem        = knod_bpf_map_update_elem,
	.map_delete_elem        = knod_bpf_map_delete_elem,
};

static int knod_bpf_map_alloc(struct knod_dev *knodev,
			      struct bpf_offloaded_map *offmap)
{
	int err;

	if (offmap->map.map_type != BPF_MAP_TYPE_ARRAY &&
	    offmap->map.map_type != BPF_MAP_TYPE_HASH &&
	    offmap->map.map_type != BPF_MAP_TYPE_PERCPU_ARRAY &&
	    offmap->map.map_type != BPF_MAP_TYPE_PERCPU_HASH) {
		knod_jit_dbg(" unsupported map type: %d\n",
			offmap->map.map_type);
		return -EOPNOTSUPP;
	}

	err = __knod_bpf_map_alloc(knodev, offmap);
	if (err) {
		knod_jit_dbg(" err = %d\n", err);
		return err;
	}

	offmap->dev_ops = &knod_bpf_map_ops;
	return 0;
}

static int knod_bpf_xdp_install(struct knod_dev *knodev,
				struct netdev_bpf *bpf)
{
	int err = 0;

	ASSERT_RTNL();

	switch (bpf->command) {
	case XDP_SETUP_PROG:
		WARN_ON_ONCE(1);
		break;
	case XDP_SETUP_PROG_HW:
		err = knod_bpf_setup_prog_hw_checks(knodev, bpf);
		if (err)
			return err;

		err = knod_bpf_xdp_set_prog(knodev, bpf);
		break;
	case BPF_OFFLOAD_MAP_ALLOC:
		err = knod_bpf_map_alloc(knodev, bpf->offmap);
		break;
	case BPF_OFFLOAD_MAP_FREE:
		knod_bpf_map_free(knodev, bpf->offmap);
		break;
	default:
		knod_jit_dbg(" bpf->command = %d\n", bpf->command);
		err = -EINVAL;
		break;
	}

	return err;
}

static inline int bpf_debugfs_insn(struct knod_insn_meta *meta,
				   struct seq_file *m, int insn_idx)
{
	struct amdgcn_insn *insn = &meta->amdgpu_insn[insn_idx];

	debugfs_insn(insn, m);

	return insn->size;
}

/* Wide enough for the offset and the dwords of the longest instruction, so the
 * tags line up in a column of their own.
 */
#define KNOD_BPF_TAG_COLUMN		40

/* The dwords a meta holds, spliced and emitted alike, eight to a line.
 * @col carries the position within the line across metas so the run reads as
 * one block.  Returns how many bytes went out, which is what the offsets the
 * rest of the dump prints are counted in.
 */
static int bpf_debugfs_dwords(struct knod_insn_meta *meta, struct seq_file *m,
			      int *col)
{
	const u32 *dw;
	int n = 0, i, j;

	/* Same order knod_meta_write puts them in, or the dump describes a
	 * program that was never built.
	 */
	for (i = 0; i <= (int)meta->amdgpu_insns; i++) {
		if (meta->blob_size && i == (int)meta->blob_at) {
			for (j = 0; j < (int)(meta->blob_size / 4); j++) {
				seq_printf(m, "%08x%c", meta->blob[j],
					   ++(*col) % 8 ? ' ' : '\n');
				*col %= 8;
				n++;
			}
		}
		if (i == (int)meta->amdgpu_insns)
			break;

		dw = (const u32 *)&meta->amdgpu_insn[i];
		for (j = 0; j < (int)(meta->amdgpu_insn[i].size / 4); j++) {
			seq_printf(m, "%08x%c", dw[j],
				   ++(*col) % 8 ? ' ' : '\n');
			*col %= 8;
			n++;
		}
	}

	return n * 4;
}

/* A spliced routine in the annotated stream, one dword per line so the offsets
 * stay right.  Undecoded: the JIT did not build it and has no more idea what is
 * in it than the reader does.  Returns how many bytes it covered.
 */
static int bpf_debugfs_spliced(struct knod_insn_meta *meta, struct seq_file *m,
			       int offset)
{
	u32 i;

	for (i = 0; i < meta->blob_size / 4; i++)
		seq_printf(m, "%d:\t%08x%*s ; spliced\n", offset + i * 4,
			   meta->blob[i], KNOD_BPF_TAG_COLUMN - 16, "");

	return meta->blob_size;
}

/*
 * Print one GPU instruction at @offset, then drop the trailing newline and
 * append @tag as a right-hand comment aligned to a fixed column (tabs expand
 * to 8) so the origin lines up.  Returns the instruction size in dwords.
 */
static int bpf_debugfs_insn_tagged(struct knod_insn_meta *meta,
				   struct seq_file *m, int j,
				   int offset, const char *tag)
{
	size_t col, p, line_start = m->count;
	int sz;

	seq_printf(m, "%d:\t", offset);
	sz = bpf_debugfs_insn(meta, m, j);
	if (seq_has_overflowed(m))
		return sz;

	if (m->count > line_start && m->buf[m->count - 1] == '\n')
		m->count--;
	col = 0;
	for (p = line_start; p < m->count; p++)
		col = m->buf[p] == '\t' ? (col + 8) & ~(size_t)7 : col + 1;
	while (col < KNOD_BPF_TAG_COLUMN) {
		seq_putc(m, ' ');
		col++;
	}
	seq_printf(m, " ; %s\n", tag);

	return sz;
}

/*
 * Print the instructions a second time, re-sorted into BPF source order so the
 * dump reads like the program.  The offsets are the real (reordered) GPU
 * offsets, so they appear out of sequence - that shows where the reorder
 * placed each block.  Synthetic jumps have no BPF source insn and are last.
 */
static void bpf_insn_show_bpf_order(struct knod_bpf_priv *priv,
				    struct seq_file *m)
{
	struct knod_insn_meta *meta;
	int idx, max_idx = -1, off2, i;
	bool synth_hdr = false;
	char tag[24];

	seq_puts(m, "===[INSTRUCTIONS (bpf order)]===\n");
	seq_puts(m, "# format annotated\n");

	list_for_each_entry(meta, &priv->knod_prog->insns, l)
		if (meta->bpf_insn_idx > max_idx)
			max_idx = meta->bpf_insn_idx;

	for (idx = 0; idx <= max_idx; idx++) {
		list_for_each_entry(meta, &priv->knod_prog->insns, l) {
			if (meta->bpf_insn_idx != idx || !meta->amdgpu_insns)
				continue;
			scnprintf(tag, sizeof(tag), "bpf#%d", idx);
			off2 = meta->amdgpu_insn_idx;
			for (i = 0; i < meta->amdgpu_insns; i++)
				off2 += bpf_debugfs_insn_tagged(meta, m,
								i, off2, tag);
		}
	}

	list_for_each_entry(meta, &priv->knod_prog->insns, l) {
		if (meta->bpf_insn_idx >= 0 || !meta->amdgpu_insns)
			continue;
		if (!synth_hdr) {
			seq_puts(m, "  [synthetic jumps]\n");
			synth_hdr = true;
		}
		scnprintf(tag, sizeof(tag), "synth JA->#%d",
			  meta->jmp_dst ? meta->jmp_dst->bpf_insn_idx : -1);
		off2 = meta->amdgpu_insn_idx;
		for (i = 0; i < meta->amdgpu_insns; i++)
			off2 += bpf_debugfs_insn_tagged(meta, m,
							i, off2, tag);
	}
}

static int bpf_insn_show(struct seq_file *m, void *v)
{
	struct knod_bpf_priv *priv = (struct knod_bpf_priv *)m->private;
	struct knod_insn_meta *meta;
	struct knod_prog *kp;
	int i, insn_idx = 0;
	bool have_prog;
	int col = 0;

	if (!priv)
		return 0;

	knod_seq_dump_header(m, priv->knod, "BPF kernel", "annotated", 0, 0, 64);
	/* Which of the two builds this is.  Otherwise the only way to tell a
	 * dump apart is to recognise a routine in it, and the pieces the two
	 * engines share are byte for byte the same.
	 */
	seq_printf(m, "# jit_engine %d\n", knod_bpf_jit_engine);

	/*
	 * Show the kernel the GPU actually dispatches: the XDP prog when one is
	 * attached, otherwise the retained pass-through kernel.
	 */
	have_prog = READ_ONCE(priv->prog);
	if (have_prog) {
		kp = priv->knod_prog;
	} else {
		kp = priv->pass_knod_prog;
		seq_puts(m, "no XDP prog attached -- pass-through kernel:\n");
	}
	if (!kp)
		return 0;

	/* The prologue goes out as dwords rather than one instruction per line.
	 * Part of it may have been spliced in whole, and the JIT cannot say
	 * where the instructions in that part begin - so it says nothing about
	 * any of them, and the dump reads the same either way.  Nothing is lost:
	 * unlike the body, no line here belongs to a BPF instruction.
	 */
	seq_puts(m, "===[PROLOGUE]===\n");
	seq_puts(m, "# format block\n");
	seq_printf(m, "# base %d\n", insn_idx);
	list_for_each_entry(meta, &kp->pre_insns, l)
		insn_idx += bpf_debugfs_dwords(meta, m, &col);
	if (col)
		seq_putc(m, '\n');

	/* Emission (RPO) order - the actual GPU layout.  Each line is tagged
	 * with its origin BPF insn since the reorder makes this differ from the
	 * BPF byte order; synthetic jumps inserted by the reorder have none.
	 */
	seq_puts(m, "===[INSTRUCTIONS]===\n");
	seq_puts(m, "# format annotated\n");
	list_for_each_entry(meta, &kp->insns, l) {
		char tag[24];

		if (meta->bpf_insn_idx < 0)
			scnprintf(tag, sizeof(tag), "synth JA->#%d",
				  meta->jmp_dst ?
				  meta->jmp_dst->bpf_insn_idx : -1);
		else
			scnprintf(tag, sizeof(tag), "bpf#%d",
				  meta->bpf_insn_idx);

		/* Same walk knod_meta_write does, or the offsets drift from
		 * the program at the first routine spliced into the body.
		 */
		for (i = 0; i <= (int)meta->amdgpu_insns; i++) {
			if (meta->blob_size && i == (int)meta->blob_at)
				insn_idx += bpf_debugfs_spliced(meta, m,
								insn_idx);
			if (i == (int)meta->amdgpu_insns)
				break;
			insn_idx += bpf_debugfs_insn_tagged(meta, m, i,
							    insn_idx, tag);
		}
	}

	/* Dwords, for the same reason as the prologue: part of this may have
	 * been spliced in whole.  Nothing here belongs to a BPF instruction
	 * either, so the tags the body carries are not lost by going wide.
	 */
	seq_puts(m, "===[EPILOG]===\n");
	seq_puts(m, "# format block\n");
	seq_printf(m, "# base %d\n", insn_idx);
	col = 0;
	list_for_each_entry(meta, &kp->post_insns, l)
		insn_idx += bpf_debugfs_dwords(meta, m, &col);
	if (col)
		seq_putc(m, '\n');

	if (have_prog)
		bpf_insn_show_bpf_order(priv, m);

	return 0;
}

static int bpf_insn_open(struct inode *inode, struct file *file)
{
	return single_open(file, bpf_insn_show, inode->i_private);
}

static const struct file_operations bpf_insn_fops = {
	.owner   = THIS_MODULE,
	.open    = bpf_insn_open,
	.read    = seq_read,
	.llseek  = seq_lseek,
	.release = single_release,
};

static const char *knod_branch_type_str(enum knod_branch_type type)
{
	switch (type) {
	case KNOD_BR_NONE:		return "NONE";
	case KNOD_BR_DIRECT_EXIT:	return "DIRECT_EXIT";
	case KNOD_BR_FORWARD_SKIP:	return "FORWARD_SKIP";
	case KNOD_BR_FORWARD_GOTO:	return "FORWARD_GOTO";
	default:			return "UNKNOWN";
	}
}

static int bpf_cfg_show(struct seq_file *m, void *v)
{
	struct knod_bpf_priv *priv = (struct knod_bpf_priv *)m->private;
	struct knod_insn_meta *meta;

	if (!priv || !priv->knod_prog)
		return 0;

	seq_puts(m, "===[STRUCTURIZED CFG]===\n");
	seq_printf(m, "exec_save_pairs_used: %u\n",
		   priv->knod_prog->exec_save_pairs_used);
	seq_printf(m, "done_mask: s[%d:%d]\n",
		   priv->knod_prog->done_mask_sreg,
		   priv->knod_prog->done_mask_sreg + 1);
	seq_printf(m, "initial_exec: s[%d:%d]\n",
		   priv->knod_prog->initial_exec_sreg,
		   priv->knod_prog->initial_exec_sreg + 1);
	seq_puts(m, "\n");

	seq_printf(m, "%-6s %-8s %-14s %-10s %-10s %-8s\n",
		   "bpf#", "opcode", "branch_type", "exec_save", "merge_pt",
		   "is_merge");

	list_for_each_entry(meta, &priv->knod_prog->insns, l) {
		bool is_jmp = is_mbpf_jmp(meta);

		if (!is_jmp && !meta->is_merge_point)
			continue;

		seq_printf(m, "%-6d 0x%02x     ",
			   meta->bpf_insn_idx, meta->insn.code);

		if (meta->branch_type != KNOD_BR_NONE) {
			seq_printf(m, "%-14s s[%d:%d]    ",
				   knod_branch_type_str(meta->branch_type),
				   meta->exec_save_sreg,
				   meta->exec_save_sreg + 1);
			if (meta->merge_point)
				seq_printf(m, "%-10d ",
					   meta->merge_point->bpf_insn_idx);
			else
				seq_printf(m, "%-10s ", "-");
		} else if (is_jmp) {
			seq_printf(m, "%-14s %-10s %-10s ",
				   knod_branch_type_str(KNOD_BR_NONE),
				   "-", "-");
		} else {
			seq_printf(m, "%-14s %-10s %-10s ",
				   "", "", "");
		}

		if (meta->is_merge_point) {
			struct knod_insn_meta *br;

			seq_puts(m, "YES      restore:");
			list_for_each_entry(br, &priv->knod_prog->insns, l) {
				if ((br->branch_type == KNOD_BR_FORWARD_SKIP ||
				     br->branch_type == KNOD_BR_FORWARD_GOTO) &&
				    br->merge_point == meta)
					seq_printf(m, " s[%d:%d](from bpf#%d)",
						   br->exec_save_sreg,
						   br->exec_save_sreg + 1,
						   br->bpf_insn_idx);
			}
			seq_puts(m, "\n");
		} else {
			seq_puts(m, "\n");
		}
	}

	/* Basic-block CFG from the reorder analysis (origin BPF order). */
	if (priv->knod_prog->bbs) {
		struct knod_bb *bbs = priv->knod_prog->bbs;
		int nb = priv->knod_prog->n_bbs;
		int k, s;

		seq_printf(m, "\n[BASIC BLOCKS]  %d blocks, %d back-edge(s) -> %s\n",
			   nb, priv->knod_prog->n_back,
			   priv->knod_prog->n_back ? "HAS LOOP" : "DAG");

		for (k = 0; k < nb; k++) {
			struct knod_bb *bb = &bbs[k];

			seq_printf(m, "BB%-3d bpf#%d..#%d  rpo=%d  idom=#%d  succ={",
				   k, bb->leader->bpf_insn_idx,
				   bb->last->bpf_insn_idx, bb->rpo,
				   bb->idom ?
				   bb->idom->leader->bpf_insn_idx : -1);
			for (s = 0; s < bb->n_succ; s++)
				seq_printf(m, "%s#%d", s ? "," : "",
					   bb->succ[s]->leader->bpf_insn_idx);
			seq_printf(m, "}%s\n",
				   bb->loop_header ? "  LOOP_HDR" : "");
		}
	}

	return 0;
}

DEFINE_SHOW_ATTRIBUTE(bpf_cfg);

static int knod_stats_show(struct seq_file *s, void *unused)
{
	struct knod_bpf_priv *priv = s->private;
	u64 p50 = 0, p99 = 0, p999 = 0, acc;
	struct knod_bpf_stats *stats;
	u32 gfx = priv->knod->gfx_target_version;
	u64 ccnt, dcnt, elapsed, wall, end, mpps;
	int i;

	stats = &priv->stats;
	ccnt = stats->completion_count;
	dcnt = stats->dispatch_count;
	end = stats->stop_ns ? stats->stop_ns : ktime_get_ns();
	wall = stats->start_ns ? end - stats->start_ns : 0;

	/* Rate over the time dispatches were actually going out.  Measured from
	 * the reset instead, an idle link before the traffic started reads as
	 * throughput the device failed to deliver.
	 */
	elapsed = stats->last_dispatch_ns > stats->first_dispatch_ns ?
		  stats->last_dispatch_ns - stats->first_dispatch_ns : 0;

	seq_printf(s, "enabled:             %s\n",
		   static_branch_unlikely(&knod_stats_key) ? "yes" : "no");

	/* What the numbers below were taken on.  A dump that does not say is a
	 * dump that gets compared against the wrong one later.
	 */
	seq_puts(s, "\n--- geometry ---\n");
	seq_printf(s, "channels:            %d\n", priv->nr_works);
	seq_printf(s, "execution_mode:      %s\n", knod_bpf_persistent ? "persistent_chunks" : "aql_batches");
	seq_printf(s, "workgroup_size:      %u\n", knod_bpf_workgroups);
	seq_printf(s, "groups_per_queue:    %d\n",
		   knod_bpf_workgroups ? priv->batch_size /
		   (int)knod_bpf_workgroups : 0);
	seq_printf(s, "batch:               %d per queue, %d total\n",
		   priv->batch_size, priv->batch_size * priv->nr_works);
	seq_printf(s, "waves:               %d per queue, %d total\n",
		   DIV_ROUND_UP(priv->batch_size, KNOD_WAVE_LANES),
		   DIV_ROUND_UP(priv->batch_size, KNOD_WAVE_LANES) *
		   priv->nr_works);
	/* What is in force, and when that is not what was asked for, say so:
	 * reporting only the effective value turns a refusal into a mystery.
	 */
	seq_printf(s, "lds_per_wg:          %u\n", priv->knod->lds_size);
	seq_printf(s, "stack_bytes:         %d per lane\n",
		   priv->knod_prog ? priv->knod_prog->max_stack_off : 0);
	seq_printf(s, "lds_alloc:           %u\n",
		   priv->lds_bytes[READ_ONCE(priv->active_idx)]);
	seq_printf(s, "mcpu:                gfx%u%u%u\n",
		   gfx / 10000, (gfx / 100) % 100, gfx % 100);
	seq_printf(s, "jit_engine:          %s\n",
		   "blob");
	seq_printf(s, "poll_mode:           %s\n",
		   knod_bpf_poll_mode ? "spin" : "event");
	seq_printf(s, "dispatch_delay_us:   %u\n",
		   READ_ONCE(knod_bpf_dispatch_delay_us));
	seq_printf(s, "queue_expire_ms:     %u\n", knod_bpf_expire);
	seq_printf(s, "wgp:                 %s\n", knod_bpf_wgp ? "yes" : "no");
	seq_printf(s, "cycle_probe:         %u%s\n", knod_bpf_cycle_probe,
		   knod_bpf_cycle_probe ? " (costs throughput)" : "");

	seq_printf(s, "elapsed_ms:          %llu\n", wall / NSEC_PER_MSEC);
	if (elapsed) {
		mpps = stats->backlogs_total * 100000ULL / elapsed;
		seq_printf(s, "active_ms:           %llu\n",
			   elapsed / NSEC_PER_MSEC);
		seq_printf(s, "dispatch_per_s:      %llu\n",
			   dcnt * NSEC_PER_SEC / elapsed);
		seq_printf(s, "throughput:          %llu.%02llu Mpps\n",
			   mpps / 100, mpps % 100);
	}

	seq_puts(s, "\n--- dispatch ---\n");
	seq_printf(s, "count:               %llu\n", dcnt);
	seq_printf(s, "avg_ns:              %llu\n",
		   dcnt ? stats->dispatch_total_ns / dcnt : 0);
	seq_printf(s, "max_ns:              %llu\n", stats->dispatch_max_ns);
	seq_printf(s, "backlogs_avg:        %llu\n",
		   dcnt ? stats->backlogs_total / dcnt : 0);
	/* Force-retired without ever signalling.  Only dmesg used to say. */
	seq_printf(s, "expired:             %llu\n", stats->expire_count);

	seq_puts(s, "\nbacklogs histogram:\n");
	for (i = 0; i < KNOD_BL_BUCKETS; i++)
		seq_printf(s, "  %-10s %llu\n",
			   bl_labels[i], stats->backlogs_hist[i]);

	seq_puts(s, "\n--- completion ---\n");
	seq_printf(s, "count:               %llu\n", ccnt);
	seq_printf(s, "avg_ns:              %llu\n",
		   ccnt ? stats->completion_total_ns / ccnt : 0);
	seq_printf(s, "max_ns:              %llu\n",
		   stats->completion_max_ns);

	seq_puts(s, "\nlatency histogram:\n");
	for (i = 0; i < KNOD_LAT_BUCKETS; i++)
		seq_printf(s, "  %-10s %llu\n",
			   lat_labels[i], stats->completion_hist[i]);

	if (ccnt) {
		acc = 0;
		for (i = 0; i < KNOD_LAT_BUCKETS; i++) {
			acc += stats->completion_hist[i];
			if (!p50 && acc * 1000 >= ccnt * 500)
				p50 = i;
			if (!p99 && acc * 1000 >= ccnt * 990)
				p99 = i;
			if (!p999 && acc * 1000 >= ccnt * 999)
				p999 = i;
		}
		seq_printf(s, "\np50:  %s\n", lat_labels[p50]);
		seq_printf(s, "p99:  %s\n", lat_labels[p99]);
		seq_printf(s, "p999: %s\n", lat_labels[p999]);
	}

	seq_puts(s, "\n--- decode_act ---\n");
	seq_printf(s, "count:               %llu\n", stats->decode_act_count);
	seq_printf(s, "avg_ns:              %llu\n",
		   stats->decode_act_count ?
		   stats->decode_act_total_ns / stats->decode_act_count : 0);
	seq_printf(s, "max_ns:              %llu\n", stats->decode_act_max_ns);

	if (stats->cyc_count) {
		static const char * const part[KNOD_PROBE_PARTS] = {
			"prologue", "program", "epilogue",
		};
		u64 whole = 0;

		for (i = 0; i < KNOD_PROBE_PARTS; i++)
			whole += stats->cyc_total[i];

		seq_puts(s, "\n--- shader clocks ---\n");
		if (!whole) {
			seq_puts(s, "lanes:               (shader has no cycle probe)\n");
			goto no_cycles;
		}
		seq_printf(s, "lanes:               %llu\n", stats->cyc_count);
		for (i = 0; i < KNOD_PROBE_PARTS; i++)
			seq_printf(s, "%-9s avg %8llu  max %8llu  %2llu%%\n",
				   part[i],
				   stats->cyc_total[i] / stats->cyc_count,
				   stats->cyc_max[i],
				   stats->cyc_total[i] * 100 / whole);
		seq_printf(s, "%-9s avg %8llu\n", "total",
			   whole / stats->cyc_count);

no_cycles:
		;
	}

	/* The grid asks for groups_per_queue workgroups per queue; this says how
	 * many units they actually reached.
	 *
	 * Nothing emits the HW_ID read any more, on either engine, so the field
	 * this counts is always zero - which lands every packet in slot 0 and
	 * reads exactly like the answer it was built to look for, one unit doing
	 * everything.  Say so rather than let the shape of the output decide.
	 */
	for (i = 0, acc = 0; i < KNOD_HWID_SLOTS; i++)
		if (stats->hwid_hist[i])
			acc++;

	if (acc == 1 && stats->hwid_hist[0]) {
		seq_puts(s, "\n--- compute units ---\n");
		seq_puts(s, "units used:          (no HW_ID probe emitted)\n");
	} else if (acc) {
		seq_puts(s, "\n--- compute units ---\n");
		seq_printf(s, "units on device:     %llu\n", acc);
		seq_printf(s, "units per dispatch:  %llu.%02llu\n",
			   stats->hwid_dispatches ?
			   stats->hwid_units_total / stats->hwid_dispatches : 0,
			   stats->hwid_dispatches ?
			   stats->hwid_units_total * 100 /
			   stats->hwid_dispatches % 100 : 0);
		for (i = 0; i < KNOD_HWID_SLOTS; i++)
			if (stats->hwid_hist[i])
				seq_printf(s, "  se%u %s%u %s%-2u   %llu\n",
					   i >> 5,
					   priv->isa_version == 9 ? "sh" : "sa",
					   (i >> 4) & 1,
					   priv->isa_version == 9 ? "cu" : "wgp",
					   i & 0xf, stats->hwid_hist[i]);
	}

	return 0;
}

DEFINE_SHOW_ATTRIBUTE(knod_stats);

static ssize_t knod_stats_enable_write(struct file *file,
				       const char __user *buf,
				       size_t count, loff_t *ppos)
{
	struct knod_bpf_priv *priv = file->private_data;
	bool val;

	if (kstrtobool_from_user(buf, count, &val))
		return -EINVAL;

	if (val) {
		priv->stats.start_ns = ktime_get_ns();
		priv->stats.stop_ns = 0;
		priv->stats.first_dispatch_ns = 0;
		priv->stats.last_dispatch_ns = 0;
		static_branch_enable(&knod_stats_key);
	} else {
		static_branch_disable(&knod_stats_key);
		priv->stats.stop_ns = ktime_get_ns();
	}

	return count;
}

static ssize_t knod_stats_enable_read(struct file *file,
				      char __user *buf,
				      size_t count, loff_t *ppos)
{
	char tmp[4];
	int len;

	len = scnprintf(tmp, sizeof(tmp), "%d\n",
			static_branch_unlikely(&knod_stats_key) ? 1 : 0);

	return simple_read_from_buffer(buf, count, ppos, tmp, len);
}

static const struct file_operations knod_stats_enable_fops = {
	.owner = THIS_MODULE,
	.open  = simple_open,
	.read  = knod_stats_enable_read,
	.write = knod_stats_enable_write,
};

static ssize_t knod_stats_reset_write(struct file *file,
		const char __user *buf,
		size_t count, loff_t *ppos)
{
	struct knod_bpf_priv *priv = file->private_data;

	memset(&priv->stats, 0, sizeof(priv->stats));
	priv->stats.start_ns = ktime_get_ns();
	return count;
}

static const struct file_operations knod_stats_reset_fops = {
	.owner = THIS_MODULE,
	.open  = simple_open,
	.write = knod_stats_reset_write,
};

static int knod_debugfs_init(struct knod_bpf_priv *priv)
{
	struct dentry *dir = priv->knod->debug_dir;
	struct dentry *bpf_dir;

	if (!dir)
		return -ENOENT;

	bpf_dir = debugfs_create_dir("bpf", dir);
	if (IS_ERR(bpf_dir))
		return PTR_ERR(bpf_dir);

	priv->debug_dir = bpf_dir;

	debugfs_create_file("insn", 0644,
			    bpf_dir, priv, &bpf_insn_fops);
	debugfs_create_file("cfg", 0444, bpf_dir, priv,
			    &bpf_cfg_fops);
	debugfs_create_file("stats", 0444, bpf_dir, priv,
			    &knod_stats_fops);
	debugfs_create_file("stats_enable", 0644, bpf_dir, priv,
			    &knod_stats_enable_fops);
	debugfs_create_file("stats_reset", 0200, bpf_dir, priv,
			    &knod_stats_reset_fops);
	debugfs_create_bool("poll_mode", 0644, bpf_dir, &knod_bpf_poll_mode);
	debugfs_create_u32("dispatch_delay_us", 0644, bpf_dir,
			   &knod_bpf_dispatch_delay_us);

	return 0;
}

static void knod_debugfs_cleanup(struct knod_bpf_priv *priv)
{
	if (!priv->debug_dir)
		return;

	debugfs_remove_recursive(priv->debug_dir);
	priv->debug_dir = NULL;
}

/* Called when attached or module loading time */
/* attach: allocate the permanent per-attach priv struct. */
static int knod_accel_xdp_init(struct knod_dev *knodev)
{
	struct knod_accel *accel = knodev->accel;
	struct knod_bpf_priv *priv;

	priv = __knod_accel_xdp_init(accel, knodev);
	if (IS_ERR(priv))
		return PTR_ERR(priv);
	return 0;
}

/* detach: free the permanent priv struct. */
static void knod_accel_xdp_exit(struct knod_dev *knodev)
{
	struct knod_accel *accel = knodev->accel;
	struct knod_bpf_priv *priv = accel->xdp.priv;

	__knod_accel_xdp_exit(accel, priv);
}

/*
 * Feature select, phase B: register the BPF offload device so user XDP
 * progs/maps can bind to it.  Called after ->activate() set up the GPU
 * buffers, while xdp_ops already points at the BPF ops.
 */
static int knod_bpf_offload_init(struct knod_dev *knodev)
{
	struct knod_accel *accel = knodev->accel;
	struct knod_bpf_priv *priv = accel->xdp.priv;
	struct bpf_offload_dev *bpf_dev;
	int err;

	bpf_dev = bpf_offload_dev_create(&knod_bpf_dev_ops, priv);
	err = PTR_ERR_OR_ZERO(bpf_dev);
	if (err)
		return err;
	err = bpf_offload_dev_netdev_register(bpf_dev, knodev->netdev);
	if (err) {
		bpf_offload_dev_destroy(bpf_dev);
		return err;
	}
	knod_debugfs_init(priv);
	accel->xdp.bpf_dev = bpf_dev;
	return 0;
}

/*
 * Feature deselect, phase 1: unregister the BPF offload device.  This
 * force-frees any user XDP progs/maps still bound; the map-free ndo is
 * routed back through accel_ops.xdp_ops->xdp_install, so the caller keeps
 * xdp_ops pointed at the BPF ops until this returns.
 */
static void knod_bpf_offload_uninit(struct knod_dev *knodev)
{
	struct knod_accel *accel = knodev->accel;
	struct knod_bpf_priv *priv = accel->xdp.priv;

	knod_debugfs_cleanup(priv);
	bpf_offload_dev_netdev_unregister(accel->xdp.bpf_dev, knodev->netdev);
	bpf_offload_dev_destroy(accel->xdp.bpf_dev);
	accel->xdp.bpf_dev = NULL;
}

struct knod_accel_xdp_ops accel_xdp_ops = {
	/* attach/detach: permanent priv struct */
	.init = &knod_accel_xdp_init,
	.exit = &knod_accel_xdp_exit,
	/* feature select: GPU compute buffers (A) + offload dev (B) */
	.activate = &knod_bpf_activate,
	.deactivate = &knod_bpf_deactivate,
	.busy = &knod_bpf_busy,
	.xdp_offload_init = &knod_bpf_offload_init,
	.xdp_offload_uninit = &knod_bpf_offload_uninit,
	/* interface up/down (or feature switch): worker + GPU drain */
	.start = &knod_bpf_start,
	.stop = &knod_bpf_stop,
	.xdp_install = &knod_bpf_xdp_install,
};

static int __init knod_bpf_init_module(void)
{
	pr_info("knod-bpf module load\n");

	/* knod_accel_xdp_register() already calls xdp_ops->init() on every
	 * registered accel, so a second per-accel init loop here would just
	 * re-create the "bpf" debugfs dir ("already exists" warning) and leak
	 * a duplicate offload dev.
	 */
	knod_dev_lock();
	knod_accel_xdp_register(&accel_xdp_ops);
	knod_dev_unlock();

	return 0;
}
late_initcall(knod_bpf_init_module);

static void __exit knod_bpf_cleanup_module(void)
{
	struct knod_bpf_priv *priv, *tmp;
	struct knod_accel *accel;

	rtnl_lock();
	knod_dev_lock();
	list_for_each_entry_safe(priv, tmp, &priv_list, list) {
		accel = priv->accel;
		if (accel->knodev)
			accel_xdp_ops.exit(accel->knodev);
	}
	knod_accel_xdp_unregister();
	knod_dev_unlock();
	rtnl_unlock();
	pr_info("knod-bpf module unload\n");
}
module_exit(knod_bpf_cleanup_module);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Taehee Yoo <ap420073@gmail.com>");
MODULE_DESCRIPTION("AMDGPU BPF offload backend");
MODULE_VERSION("multi-aql");
