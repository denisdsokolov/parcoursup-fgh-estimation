from __future__ import annotations
from typing import Dict, Set, Iterable, Optional, List
import pandas as pd

# =============================================================
# FAST CDA core (stop on NO REJECTIONS) + pluggable
def students_proposed(
    program_id: int,                     # program making proposals now
    program_ranking: List[int],          # ordered student_ids (best→worst) for THIS program (may be a suffix)
    held_students: Set[int],             # set of student_ids currently held by THIS program
    never_proposed: Set[int],            # set of student_ids THIS program has NEVER proposed to yet
    programs: pd.DataFrame,              # df indexed by program_id with at least ['quota']
    students: pd.DataFrame,              # df indexed by student_id; may have 'b_trait'
    residents: pd.DataFrame,             # df indexed by student_id; may have 'r_trait'
    mechanism: str,                      # 'CDA' or 'SF-CDA'
) -> Iterable[int]:                      # returns iterable of student_ids to propose now
    """Proposal rule used by the engine.

    • CDA: take up to the current deficit best *never‑proposed*.
    • SF‑CDA: enforce target shares with the anchor/reshuffle/greedy procedure
      in the paper. The first anchor that fills capacity and both residual
      targets is the highest-priority stable anchor, ``k_left``. Targets are
      computed per program from
      'b_share' and 'r_share' in `programs` (percentages), each contributing
      ceil(share * quota / 100) target seats. A student with both traits counts
      toward **both** targets when tentatively admitted.
    """
    # ---------- Common: fast O(1) lookups ----------
    if program_id not in programs.index:
        return ()
    quota = int(programs.at[program_id, 'quota'])

    # Current main deficit (strict capacity constraint)
    q_def = quota - len(held_students)
    if q_def <= 0 or not never_proposed:
        return ()

    # ---------- CDA branch (simple, fast) ----------
    if mechanism != 'SF-CDA':
        out: List[int] = []
        append = out.append
        c = 0
        # walk the (suffix) ranking, pick up to q_def among never_proposed
        never = never_proposed
        for sid in program_ranking:
            if sid in never:
                append(sid)
                c += 1
                if c >= q_def:
                    break
        return out

    # ---------- SF‑CDA branch ----------
    # Fetch targets (percentages may be missing ⇒ treated as 0)
    b_share = float(programs.at[program_id, 'b_share']) if 'b_share' in programs.columns else 0.0
    r_share = float(programs.at[program_id, 'r_share']) if 'r_share' in programs.columns else 0.0

    # Trait maps
    # b-trait: program-agnostic, from students.b_trait if available
    b_map = students.attrs.get('b_map')
    if b_map is None:
        b_map = students['b_trait'].to_dict() if 'b_trait' in students.columns else {}

    # r-trait: program-specific — presence in residents (student_id, program_id)
    r_set_by_program = residents.attrs.get('r_set_by_program')
    if r_set_by_program is None:
        # build once (slower path if run here)
        r_set_by_program = {}
        for pid, sub in residents.groupby('program_id', sort=False):
            r_set_by_program[int(pid)] = set(int(s) for s in sub['student_id'].tolist())

    r_set_this_program: Set[int] = r_set_by_program.get(program_id, set())

    has_b = lambda s: 1 if b_map.get(s, 0) else 0
    has_r = lambda s: 1 if s in r_set_this_program else 0

    # Current trait counts among held (each both‑trait contributes to both counts)
    b_count = 0
    r_count = 0
    for s in held_students:
        b_count += has_b(s)
        r_count += has_r(s)

    # Targets (ceil) and non‑negative deficits for traits
    from math import ceil
    b_target = int(ceil(b_share * quota / 100.0))
    r_target = int(ceil(r_share * quota / 100.0))
    qb_def = max(0, b_target - b_count)
    qr_def = max(0, r_target - r_count)

    # Prepare the real ranking over *never‑proposed* only
    real_list = [s for s in program_ranking if s in never_proposed]
    if not real_list:
        return ()

    # Helper: greedy picker respecting deficits (counts both‑trait for both)
    def pick_student_subset(ordered: List[int], q: int, qb: int, qr: int) -> List[int]:
        chosen: List[int] = []
        chosen_set: Set[int] = set()
        seats = q
        bleft = qb
        rleft = qr

        # Step 1: fill b target up to bleft
        if bleft > 0 and seats > 0:
            for sid in ordered:
                if sid in chosen_set:
                    continue
                if has_b(sid):
                    chosen.append(sid)
                    chosen_set.add(sid)
                    seats -= 1
                    if bleft > 0:
                        bleft -= 1
                    if has_r(sid) and rleft > 0:
                        rleft -= 1
                    if seats == 0 or bleft == 0:
                        break

        # Step 2: fill r target up to rleft
        if rleft > 0 and seats > 0:
            for sid in ordered:
                if sid in chosen_set:
                    continue
                if has_r(sid):
                    chosen.append(sid)
                    chosen_set.add(sid)
                    seats -= 1
                    if rleft > 0:
                        rleft -= 1
                    if seats == 0 or rleft == 0:
                        break

        # Step 3: fill remaining seats by general merit
        if seats > 0:
            for sid in ordered:
                if sid in chosen_set:
                    continue
                chosen.append(sid)
                seats -= 1
                if seats == 0:
                    break

        return chosen

    # Cases based on which trait deficits are positive
    if qb_def == 0 and qr_def == 0:
        # Only capacity matters: take best q_def
        return real_list[:q_def]

    if (qb_def > 0) ^ (qr_def > 0):  # exactly one positive
        if qb_def > 0:
            # Prioritize b, then fill rest
            subset = pick_student_subset(real_list, q_def, qb_def, 0)
        else:
            # Prioritize r, then fill rest
            subset = pick_student_subset(real_list, q_def, 0, qr_def)
        return subset

    # Both qb_def > 0 and qr_def > 0: run anchor/reshuffle loop
    # Identify both‑trait candidates in the real list
    both_flags = [1 if (has_b(s) and has_r(s)) else 0 for s in real_list]
    both_positions = [i for i, f in enumerate(both_flags) if f]

    if not both_positions:
        # No both‑trait anchors available → fall back to greedy on real list
        return pick_student_subset(real_list, q_def, qb_def, qr_def)

    last_subset: List[int] = []

    # --- NEW: empty‑anchor pass ---
    # Treat the best both‑trait as anchor2 and consider reshuffle with *no* top both block.
    first_both_idx = both_positions[0]
    excluded_tail_empty = set(real_list[first_both_idx:])            # drop from first both onward
    reshuffled_empty = [s for s in real_list if s not in excluded_tail_empty]  # keep only strictly above first both
    subset = pick_student_subset(reshuffled_empty, q_def, qb_def, qr_def)
    last_subset = subset
    if subset:
        add_b = 0
        add_r = 0
        for s in subset:
            add_b += has_b(s)
            add_r += has_r(s)
        if (q_def - len(subset) == 0) and (max(0, qb_def - add_b) == 0) and (max(0, qr_def - add_r) == 0):
            return subset

    # --- Then consider actual both‑trait anchors in priority order ---
    # Returning the first stable one implements the paper's k_left choice.
    for k, anchor_idx in enumerate(both_positions):
        # Determine next both‑trait index (anchor2)
        next_idx = both_positions[k+1] if (k+1) < len(both_positions) else None

        # Top block: all both‑trait up to and including anchor, preserving order
        top_block = [s for i, s in enumerate(real_list[:anchor_idx+1]) if both_flags[i]]

        # Excluded tail: from next both‑trait onward (including it)
        excluded_tail: Set[int] = set(real_list[next_idx:]) if next_idx is not None else set()

        # Bottom block: the rest, excluding top_block and excluded tail, preserving order
        top_set = set(top_block)
        reshuffled = top_block + [s for s in real_list if s not in top_set and s not in excluded_tail]

        # Run greedy picker on this reshuffled order
        subset = pick_student_subset(reshuffled, q_def, qb_def, qr_def)
        last_subset = subset  # keep the last computed subset

        # Check if this subset zeros all deficits
        if subset:
            add_b = 0
            add_r = 0
            for s in subset:
                add_b += has_b(s)
                add_r += has_r(s)
            if (q_def - len(subset) == 0) and (max(0, qb_def - add_b) == 0) and (max(0, qr_def - add_r) == 0):
                return subset

    # If none of the anchors zeroed all deficits, return the last subset tried
    return last_subset



# ---------------------------- run_cda_fast ----------------------------

def run_cda_fast(
    programs: pd.DataFrame,               # columns: [program_id, quota, ...]
    students: pd.DataFrame,               # columns: [student_id, b_trait?, ...]
    residents: pd.DataFrame,              # columns: [program_id, student_id] (ONLY pairs that have r-trait for that program)
    prefs: pd.DataFrame,                  # columns: [student_id, program_id, rank] (1=top)
    ranks: pd.DataFrame,                  # columns: [program_id, student_id, call_rank, merit_rank]
    rank_col: str = 'call_rank',          # 'call_rank' (Parcoursup CDA) or 'merit_rank' (SF‑CDA later)
    mechanism: str = 'CDA',               # forwarded to students_proposed(); engine is neutral
    verbose: bool = False,                # optional logging per iteration
) -> tuple[pd.DataFrame, int, list[int]]:  # returns (final matching [student_id, program_id], number of iterations, list with numbers of filled seats per stage)
    """Ultra‑fast program‑proposing DA that stops on **no rejections**.

    Residents schema: `residents` contains ONLY (student_id, program_id) pairs
    where the student has the r‑trait for that specific program. There is **no**
    `r_trait` column; the trait is determined by row presence.
    """
    # ---------- validation & defensive copies ----------
    if rank_col not in ranks.columns:                                        # ensure chosen priority exists
        raise ValueError(f"rank_col='{rank_col}' not in ranks; available: {list(ranks.columns)}")

    programs = programs.copy()                                               # avoid mutating inputs
    students = students.copy()
    residents = residents.copy()
    prefs = prefs.copy()
    ranks = ranks.copy()

    # ---------- dtype normalization ----------
    if 'program_id' in programs.columns:                                     # int ids
        programs['program_id'] = programs['program_id'].astype(int)
    if 'quota' in programs.columns:
        programs['quota'] = programs['quota'].astype(int)
    students['student_id'] = students['student_id'].astype(int)
    prefs = prefs[['student_id', 'program_id', 'rank']].astype(int)
    ranks = ranks[['program_id', 'student_id', rank_col]].astype(int)

    # Ensure residents has the required columns and ints
    if not {'student_id','program_id'}.issubset(residents.columns):
        raise ValueError("residents must have columns {'student_id','program_id'} with composite PK indicating r-trait presence")
    residents[['student_id','program_id']] = residents[['student_id','program_id']].astype(int)

    # ---------- index for O(1) lookup ----------
    programs.set_index('program_id', inplace=True, drop=False)               # for .at[pid, 'quota']
    students.set_index('student_id', inplace=True, drop=False)
    # residents indexed not strictly needed for our lookups; keep as-is

    # ---------- precompute trait maps (store in .attrs for reuse) ----------
    # b-trait: program-agnostic (student-level) indicator if present
    if 'b_trait' in students.columns:
        students.attrs['b_map'] = students['b_trait'].to_dict()
    else:
        students.attrs['b_map'] = {}

    # r-trait: program-specific — build set of residents per program_id
    r_set_by_program: Dict[int, Set[int]] = {}
    for pid, sub in residents.groupby('program_id', sort=False):
        r_set_by_program[int(pid)] = set(int(s) for s in sub['student_id'].tolist())
    residents.attrs['r_set_by_program'] = r_set_by_program

    # ---------- student preference positions ----------
    stud_pref_pos: Dict[int, Dict[int, int]] = {}                            # sid -> {pid: position}
    for sid, sub in prefs.sort_values(['student_id', 'rank']).groupby('student_id', sort=False):
        progs = sub['program_id'].tolist()                                   # list of programs by preference
        stud_pref_pos[int(sid)] = {int(pid): i for i, pid in enumerate(progs)}

    # ---------- per‑program ranking lists ----------
    prog_ranking: Dict[int, List[int]] = {}                                   # pid -> ordered students
    for pid, sub in ranks.sort_values(['program_id', rank_col]).groupby('program_id', sort=False):
        prog_ranking[int(pid)] = [int(s) for s in sub['student_id'].tolist()]

    # ---------- quotas & initial state ----------
    prog_quota: Dict[int, int] = {int(row.program_id): int(row.quota)         # pid -> quota
                                  for row in programs.itertuples(index=False)}
    held_by_program: Dict[int, Set[int]] = {pid: set() for pid in prog_quota} # pid -> held students
    held_by_student: Dict[int, Optional[int]] = {int(s): None for s in students.index}  # sid -> held pid or None

    # ---------- remaining candidates (never‑proposed) & scan pointers ----------
    remain: Dict[int, Set[int]] = {pid: set(lst) for pid, lst in prog_ranking.items()}  # mutate as we propose
    ptr: Dict[int, int] = {pid: 0 for pid in prog_ranking}                              # next index to inspect

    # ---------- local bindings for loop speed ----------
    hb_prog = held_by_program
    hb_stud = held_by_student
    pref_pos = stud_pref_pos
    pranking = prog_ranking
    premain = remain
    pptr = ptr
    pquota = prog_quota
    INF = 10**12

    # ---------- preference comparator ----------
    def student_prefers(sid: int, new_p: int, old_p: Optional[int]) -> bool:
        if old_p is None:                                                    # unmatched → prefers any offer
            return True
        d = pref_pos.get(sid)                                                # student's pref map
        if d is None:                                                        # no prefs recorded → no switch
            return False
        return d.get(new_p, INF) < d.get(old_p, INF)                         # lower index = better

    # =========================== main loop ===========================
    iteration = 0
    filled_per_stage: list[int] = [] # for number of filled seats at the end of each stage
    while True:                                                              # iterate until no rejections
        iteration += 1
        rejections_made = 0                                                  # count rejections this round

        # ----- each program proposes up to its deficit -----
        for pid, quota in pquota.items():                                     # iterate over programs
            held = hb_prog[pid]                                              # current held at this program
            deficit = quota - len(held)                                      # seats still needed
            if deficit <= 0:                                                 # skip if full
                continue

            rank_list = pranking.get(pid, [])                                # full ranking for this pid
            if not rank_list:                                                # no candidates at all
                continue

            # Advance pointer past exhausted prefix (those not in remain)
            j = pptr[pid]                                                    # current scan pointer
            rset = premain[pid]                                              # set of never‑proposed candidates
            n = len(rank_list)                                               # cache length
            while j < n and (rank_list[j] not in rset):                      # skip already proposed IDs
                j += 1
            if j >= n:                                                       # no one left to approach
                pptr[pid] = j                                                # store pointer
                continue
            pptr[pid] = j                                                    # store pointer after skip

            # Choose proposals using the suffix starting at pointer
            subset = students_proposed(
                pid,                                                         # program id
                rank_list[j:],                                               # suffix from current pointer
                held,                                                        # current tentatively held
                rset,                                                        # set of never‑proposed
                programs,                                                    # for quota & shares lookup
                students,                                                    # has b_map attr
                residents,                                                   # has r_set_by_program attr
                mechanism,                                                   # mechanism tag
            )
            if not subset:                                                   # this program proposes to nobody now
                continue

            # Remove proposed from remain (so they will never be proposed again)
            rset.difference_update(subset)                                   # mutate in place; O(k)

            # Student responses (immediate)
            for sid in subset:                                               # each proposed student
                current = hb_stud.get(sid)                                   # their current held program
                if student_prefers(sid, pid, current):                       # accepts new offer if better
                    if current is not None:                                  # was matched → old program loses
                        hb_prog[current].discard(sid)                        # remove from old program's held set
                        rejections_made += 1                                 # displacement counts as rejection
                    hb_stud[sid] = pid                                       # now holds the new program
                    held.add(sid)                                            # add to this program's held set
                else:
                    rejections_made += 1                                     # declined new program (silent reject)

            # Move pointer forward over any IDs we just proposed
            j2 = pptr[pid]
            while j2 < n and (rank_list[j2] not in rset):                    # skip proposed/removed ids
                j2 += 1
            pptr[pid] = j2                                                   # store back pointer

        # record total filled seats at end of this stage
        filled_total = sum(len(hb_prog[pid]) for pid in pquota)
        filled_per_stage.append(filled_total)

        if verbose:                                                           # optional progress
            print(f"iter={iteration} rejections={rejections_made} filled={filled_total}")

        if rejections_made == 0:                                             # EARLY STOP: stable (no rejections)
            break

    # ---------- build and return final matching ----------
    out = [(sid, pid) for sid, pid in hb_stud.items() if pid is not None]    # (student_id, program_id)
    matching_df = pd.DataFrame(out, columns=['student_id', 'program_id']).astype(int)
    return matching_df, iteration, filled_per_stage

# ======================= Justified Envy Counter =======================

def number_of_students_with_je(
    matching: pd.DataFrame,
    prefs: pd.DataFrame,
    ranks: pd.DataFrame,
    programs: pd.DataFrame,
    students: pd.DataFrame,
    residents: pd.DataFrame,
) -> tuple[int,int]:
    """Count students who have at least one justified envy under the current matching.

    Inputs
    ------
    matching  : DataFrame with columns ['student_id','program_id'] (tentative matches)
    prefs     : DataFrame with ['student_id','program_id','rank'] (1=top)
    ranks     : DataFrame with ['program_id','student_id','merit_rank'] (must include merit_rank)
    programs  : DataFrame with ['program_id','quota','b_share','r_share'] (shares in percent; shares optional→0)
    students  : DataFrame with ['student_id','b_trait'] (b_trait optional→0)
    residents : DataFrame with ['program_id','student_id'] meaning r-trait for that program

    Returns
    -------
    int : number of distinct students s1 who have at least one justified envy.
    int : number of distinct b-students s1 who have at least one justified envy.

    Notes
    -----
    • A both-trait student contributes to BOTH b and r counts.
    • Deficits are Δb= max(0, b_target - B), Δr= max(0, r_target - R), with targets
      b_target = ceil(b_share*quota/100), r_target = ceil(r_share*quota/100).
    • For existence checks we aggregate current holders of each program by category:
      (b2,r2) ∈ {(0,0),(1,0),(0,1),(1,1)}, storing their counts and worst merit rank.
    • For clause (c) (priority tie-break), we check if s1's merit position at f2 is
      strictly better (smaller index) than the **worst** position among the category
      that keeps deficits unchanged.
    """
    # ---------- defensive copies & dtypes ----------
    matching = matching.copy()
    prefs = prefs.copy()
    ranks = ranks.copy()
    programs = programs.copy()
    students = students.copy()
    residents = residents.copy()

    matching[['student_id','program_id']] = matching[['student_id','program_id']].astype(int)
    prefs[['student_id','program_id','rank']] = prefs[['student_id','program_id','rank']].astype(int)
    ranks_cols = set(ranks.columns)
    if 'merit_rank' not in ranks_cols:
        raise ValueError("ranks must include 'merit_rank' for justified envy checks")
    ranks = ranks[['program_id','student_id','merit_rank']].astype(int)
    programs[['program_id','quota']] = programs[['program_id','quota']].astype(int)
    if 'b_share' in programs.columns:
        programs['b_share'] = programs['b_share'].astype(float)
    else:
        programs['b_share'] = 0.0
    if 'r_share' in programs.columns:
        programs['r_share'] = programs['r_share'].astype(float)
    else:
        programs['r_share'] = 0.0
    students['student_id'] = students['student_id'].astype(int)
    if 'b_trait' in students.columns:
        students['b_trait'] = students['b_trait'].astype(int)
    else:
        students['b_trait'] = 0
    residents[['student_id','program_id']] = residents[['student_id','program_id']].astype(int)

    # ---------- indices for O(1) lookups ----------
    programs.set_index('program_id', inplace=True, drop=False)
    students.set_index('student_id', inplace=True, drop=False)

    # ---------- maps: preferences & merit positions ----------
    # student preferences: pos 0 = top
    stud_pref_pos: Dict[int, Dict[int, int]] = {}
    stud_pref_order: Dict[int, List[int]] = {}
    for sid, sub in prefs.sort_values(['student_id','rank']).groupby('student_id', sort=False):
        lst = [int(p) for p in sub['program_id'].tolist()]
        stud_pref_order[int(sid)] = lst
        stud_pref_pos[int(sid)] = {int(p): i for i, p in enumerate(lst)}

    # program merit ranking positions: pos 0 = best
    prog_merit_pos: Dict[int, Dict[int, int]] = {}
    for pid, sub in ranks.sort_values(['program_id','merit_rank']).groupby('program_id', sort=False):
        prog_merit_pos[int(pid)] = {int(s): i for i, s in enumerate(sub['student_id'].tolist())}

    # ---------- r-trait sets per program ----------
    r_set_by_program: Dict[int, Set[int]] = {}
    for pid, sub in residents.groupby('program_id', sort=False):
        r_set_by_program[int(pid)] = set(int(s) for s in sub['student_id'].tolist())

    # ---------- b-trait map (program-agnostic) ----------
    b_map: Dict[int, int] = students['b_trait'].to_dict()

    # ---------- current assignment maps ----------
    current_prog_of: Dict[int, Optional[int]] = {int(s): None for s in students.index}
    for row in matching.itertuples(index=False):
        current_prog_of[int(row.student_id)] = int(row.program_id)

    held_by_program: Dict[int, List[int]] = {}
    for pid, sub in matching.groupby('program_id', sort=False):
        held_by_program[int(pid)] = [int(s) for s in sub['student_id'].tolist()]

    # ---------- per-program aggregates: counts, deficits, category stats ----------
    from math import ceil
    program_info = {}  # pid -> dict with B,R,Db,Dr, targets, cat_counts, cat_worst

    for pid in programs.index:
        quota = int(programs.at[pid, 'quota'])
        b_target = int(ceil(float(programs.at[pid, 'b_share']) * quota / 100.0))
        r_target = int(ceil(float(programs.at[pid, 'r_share']) * quota / 100.0))
        held = held_by_program.get(int(pid), [])

        rset = r_set_by_program.get(int(pid), set())
        B = 0
        R = 0
        # category: (bflag, rflag) -> count and worst merit pos among matched
        cat_count: Dict[tuple, int] = {(0,0):0,(1,0):0,(0,1):0,(1,1):0}
        cat_worst: Dict[tuple, int] = {(0,0):-1,(1,0):-1,(0,1):-1,(1,1):-1}

        pos_map = prog_merit_pos.get(int(pid), {})
        for s in held:
            bflag = 1 if b_map.get(s, 0) else 0
            rflag = 1 if s in rset else 0
            B += bflag
            R += rflag
            cat = (bflag, rflag)
            cat_count[cat] += 1
            s_pos = pos_map.get(s, 10**12)
            if s_pos > cat_worst[cat]:
                cat_worst[cat] = s_pos

        Db = max(0, b_target - B)
        Dr = max(0, r_target - R)
        program_info[int(pid)] = {
            'quota': quota,
            'b_target': b_target,
            'r_target': r_target,
            'B': B,
            'R': R,
            'Db': Db,
            'Dr': Dr,
            'rset': rset,
            'pos_map': pos_map,
            'cat_count': cat_count,
            'cat_worst': cat_worst,
        }

    # ---------- helper: quick JE existence for (s1, f2) ----------
    INF = 10**12
    def has_je_for_program(s1: int, f2: int) -> bool:
        info = program_info.get(f2)
        if info is None:
            return False
        cat_count = info['cat_count']
        if (cat_count[(0,0)] + cat_count[(1,0)] + cat_count[(0,1)] + cat_count[(1,1)]) == 0:
            return False  # no s2 at f2 to swap with

        # s1 traits for this program
        t1b = 1 if b_map.get(s1, 0) else 0
        t1r = 1 if s1 in info['rset'] else 0

        B = info['B']; R = info['R']
        Db = info['Db']; Dr = info['Dr']

        # Try each category that exists; stop on first that satisfies (a) or (b) or (c)
        for (b2, r2) in ((1,1),(1,0),(0,1),(0,0)):
            if cat_count[(b2,r2)] == 0:
                continue
            Bp = B - b2 + t1b
            Rp = R - r2 + t1r
            Dbp = max(0, info['b_target'] - Bp)
            Drp = max(0, info['r_target'] - Rp)

            if Dbp < Db:
                return True                           # (a) improves b deficit
            if (Dbp == Db) and (Drp < Dr):
                return True                           # (b) ties b, improves r
            if (Dbp == Db) and (Drp == Dr):           # (c) tie on deficits → check priority
                s1_pos = info['pos_map'].get(s1, INF)
                worst_pos = info['cat_worst'][(b2,r2)]
                if s1_pos < worst_pos:               # exists s2 in this category ranked below s1
                    return True
        return False

    # ---------- main count loop over students ----------
    counted = 0
    counted_b = 0
    seen = set()

    all_students = set(students.index) | set(prefs['student_id'].tolist()) | set(matching['student_id'].tolist())
    stu_b = set(students[students['b_trait'] == 1]['student_id']) # list of all scholarship holders
    for s1 in all_students:
        if s1 in seen:
            continue
        seen.add(s1)

        # preference list (in order); if empty → no acceptable programs → no envy
        pref_list = stud_pref_order.get(s1)
        if not pref_list:
            continue

        # current program and its position in s1 prefs
        f1 = current_prog_of.get(s1, None)
        pos1 = stud_pref_pos.get(s1, {}).get(f1, INF)

        has_envy = False
        for f2 in pref_list:
            if f1 is not None and f2 == f1:
                break                           # reached current match; less-preferred programs follow
            # unmatched students consider all programs in their list
            if has_je_for_program(s1, f2):
                counted += 1
                if s1 in stu_b:
                    counted_b += 1
                has_envy = True
                break
        # if not has_envy: continue to next student

    return (counted, counted_b)

# ======================================== b and r compliance avgs =================================

def avg_traits_compl(
    matching: pd.DataFrame,
    programs: pd.DataFrame,
    students: pd.DataFrame,
    residents: pd.DataFrame,
) -> tuple[float, float, float, float]:

    matching = matching.copy()
    programs = programs.copy()
    students = students.copy()
    residents = residents.copy()

    programs.set_index('program_id', inplace=True, drop=False)
    students.set_index('student_id', inplace=True, drop=False)

    b_map: Dict[int, int] = students['b_trait'].to_dict()
    r_set_by_program: Dict[int, Set[int]] = {}
    for pid, sub in residents.groupby('program_id', sort=False):
        r_set_by_program[int(pid)] = set(int(s) for s in sub['student_id'].tolist())

    held_by_program: Dict[int, List[int]] = {}
    for pid, sub in matching.groupby('program_id', sort=False):
        held_by_program[int(pid)] = [int(s) for s in sub['student_id'].tolist()]

    from math import ceil

    sum_b = 0.0
    cnt_b = 0
    sum_r = 0.0
    cnt_r = 0

    sum_b_total = 0
    sum_r_total = 0

    for pid in programs.index:
        quota = int(programs.at[pid, 'quota'])
        b_share = float(programs.at[pid, 'b_share'])
        r_share = float(programs.at[pid, 'r_share'])
        b_target = int(ceil(b_share * quota / 100.0)) if b_share > 0 else 0
        r_target = int(ceil(r_share * quota / 100.0)) if r_share > 0 else 0

        held = held_by_program.get(int(pid), [])
        rset = r_set_by_program.get(int(pid), set())

        B = sum(1 for s in held if b_map.get(s, 0))
        R = sum(1 for s in held if s in rset)

        if b_target > 0:
            val_b = min(1.0, B / b_target)
            sum_b += val_b
            cnt_b += 1
            val_b_total = min(B,b_target) # number of filled b target seats
            sum_b_total += val_b_total

        if r_target > 0:
            val_r = min(1.0, R / r_target)
            sum_r += val_r
            cnt_r += 1
            val_r_total = min(R,r_target) # number of filled r target seats
            sum_r_total += val_r_total

    avg_b = (sum_b / cnt_b) if cnt_b > 0 else float('nan')
    avg_r = (sum_r / cnt_r) if cnt_r > 0 else float('nan')

    return (avg_b, avg_r, sum_b_total, sum_r_total)
