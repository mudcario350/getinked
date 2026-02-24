#!/usr/bin/env python3



"""
Student-Speaker Assignment App (Streamlit)

Run with: streamlit run app.py
"""

import streamlit as st
import csv
import io
import re
import random
from collections import defaultdict

# ─────────────────────────────────────────────
# Canonical speaker order (for cleaning raw CSV)
# ─────────────────────────────────────────────
CANONICAL_SPEAKERS = [
    "barshaw", "bilen", "kat h-c", "johnson", "korn", "lenz", "maloche",
    "nash", "tebbetts", "broome", "cordi", "flynn", "alex h-c", "layman",
    "stutsman", "twietmeyer", "lederman"
]

# Column index of the student email field in the raw survey CSV (0-based).
EMAIL_COL = 3

# ─────────────────────────────────────────────
# Speaker session info  (title + room)
# Keys must match the entries in CANONICAL_SPEAKERS exactly.
# Note: "maloche" here matches CANONICAL_SPEAKERS; the schedule spells it
# "MELOCHE" — fix both if the canonical spelling needs to change.
# ─────────────────────────────────────────────
SPEAKER_INFO = {
    # Room 211A
    "nash":       "NASH - Behind the Scenes (Room 211A)",
    "lenz":       "LENZ - Writing Warm Ups (Room 211A)",
    "flynn":      "FLYNN - Writing Music with Words (Room 211A)",
    "maloche":    "MALOCHE - Make 'Em Feel (Room 211A)",
    # Room 211B
    "korn":       "KORN - Better Backstory (Room 211B)",
    "cordi":      "CORDI & ALLEN - Exploring Story Through Play and Discovery (Room 211B)",
    "bilen":      "BILEN - Screenwriting 101 (Room 211B)",
    # Room 254A
    "tebbetts":   "TEBBETTS - Film School For Writers (Room 254A)",
    "johnson":    "JOHNSON - The Art of Conversation (Room 254A)",
    # Room 254B
    "broome":     "BROOME - Writing the Femme Fatale (Room 254B)",
    "alex h-c":   "ALEX HIGGS-COULTHARD - Coffee With Your Character (Room 254B)",
    "kat h-c":    "KAT HIGGS-COULTHARD - Core Memories (Room 254B)",
    # Room 210
    "stutsman":   "STUTSMAN - Making Magic (Room 210)",
    "lederman":   "LEDERMAN - The Discovery of Creative Nonfiction (Room 210)",
    "twietmeyer": "TWIETMEYER - Crafting the Best POV for Your Story (Room 210)",
    # Room 207
    "layman":     "LAYMAN - Art of the Scar (Room 207)",
    "barshaw":    "BARSHAW - How to use art to get you writing (Room 207)",
}

# ─────────────────────────────────────────────
# CSV Cleaning Logic
# ─────────────────────────────────────────────

def parse_choice(val):
    val = val.strip()
    if not val:
        return None
    m = re.match(r'(\d+)', val)
    return int(m.group(1)) if m else None


def clean_student_csv(raw_bytes):
    text = raw_bytes.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return None, "Empty CSV"

    headers = rows[0]
    data = rows[1:]

    student_col = 3
    skip_cols = {0, 1, 2, 4, 5, 6, 7}
    speaker_cols = [i for i in range(len(headers)) if i not in skip_cols and i != student_col]

    if len(speaker_cols) != len(CANONICAL_SPEAKERS):
        return None, (
            f"Expected {len(CANONICAL_SPEAKERS)} speaker columns, "
            f"found {len(speaker_cols)}. Check that the CSV format matches."
        )

    col_to_canon = {}
    for i, col_idx in enumerate(speaker_cols):
        if i < len(CANONICAL_SPEAKERS):
            col_to_canon[col_idx] = CANONICAL_SPEAKERS[i]

    # Keyed by email (lowercased) to deduplicate multiple submissions from the
    # same student.  Later rows overwrite earlier ones so the most-recent
    # preferences win.  Insertion order is preserved so the final list stays
    # in roughly the same order as the first appearance of each student.
    seen: dict[str, tuple[str, dict]] = {}  # email -> (name, ranks)
    duplicate_count = 0

    for row in data:
        if len(row) <= student_col:
            continue
        name = row[student_col].strip()
        if not name:
            continue

        email = row[EMAIL_COL].strip().lower() if EMAIL_COL < len(row) else ""
        # Fall back to name as the dedup key when no email is present.
        dedup_key = email if email else name.lower()

        ranks = {}
        for i in speaker_cols:
            if i in col_to_canon and i < len(row):
                rank = parse_choice(row[i])
                if rank is not None:
                    ranks[col_to_canon[i]] = rank

        if dedup_key in seen:
            duplicate_count += 1
            seen[dedup_key] = (name, ranks)  # update with latest submission
        else:
            seen[dedup_key] = (name, ranks)

    students = list(seen.values())
    return students, None, duplicate_count


# ─────────────────────────────────────────────
# Assignment Solver (Greedy + Backtracking)
# ─────────────────────────────────────────────

def solve_student_max_pref(ranked_speakers, speaker_sessions, all_sessions, remaining):
    """
    Exhaustive backtracking search over all sessions.  For each session we
    either assign a preferred speaker (in rank order) or skip it (leaving it
    for fill_remaining_sessions).  We keep track of the best assignment seen
    so far, where "best" means:
      1. maximise the number of sessions covered by a preferred speaker, then
      2. minimise the sum of rank numbers (rank 1 is better than rank 2, etc.).

    The search uses a *scratch copy* of `remaining` so the global dict is
    never touched during exploration.  Once the best assignment is found we
    commit it to the real `remaining` dict.

    Returns (assignment dict {session: speaker}, speakers_used set).
    """
    local_rem = dict(remaining)   # scratch copy — never modifies global during search
    best = {"assignment": {}, "count": 0, "rank_sum": float("inf")}

    def backtrack(sess_idx, current, used, count, rank_sum):
        # Record if this path is better than anything seen so far.
        if count > best["count"] or (
            count == best["count"] and rank_sum < best["rank_sum"]
        ):
            best["assignment"] = dict(current)
            best["count"] = count
            best["rank_sum"] = rank_sum

        if sess_idx == len(all_sessions):
            return

        session = all_sessions[sess_idx]

        # Branch A: assign a preferred speaker to this session.
        for rank_idx, speaker in enumerate(ranked_speakers, start=1):
            if speaker in used:
                continue
            if session not in speaker_sessions.get(speaker, set()):
                continue
            if local_rem.get((speaker, session), 0) <= 0:
                continue
            current[session] = speaker
            used.add(speaker)
            local_rem[(speaker, session)] -= 1
            backtrack(sess_idx + 1, current, used, count + 1, rank_sum + rank_idx)
            del current[session]
            used.remove(speaker)
            local_rem[(speaker, session)] += 1

        # Branch B: leave this session for fill_remaining_sessions.
        backtrack(sess_idx + 1, current, used, count, rank_sum)

    backtrack(0, {}, set(), 0, 0.0)

    # Commit the winning assignment to the shared remaining dict.
    speakers_used = set()
    for session, speaker in best["assignment"].items():
        remaining[(speaker, session)] -= 1
        speakers_used.add(speaker)

    return best["assignment"], speakers_used


def fill_remaining_sessions(assignment, all_sessions, remaining, speakers_used, speaker_sessions, capacity, min_cap):
    unfilled = [s for s in all_sessions if s not in assignment]
    if not unfilled:
        return True
    return _fill_backtrack(unfilled, 0, assignment, remaining, speakers_used, speaker_sessions, capacity, min_cap)


def _fill_backtrack(unfilled, idx, assignment, remaining, speakers_used, speaker_sessions, capacity, min_cap):
    if idx == len(unfilled):
        return True
    session = unfilled[idx]

    # Try deficit speakers first (those below minimum fill) so unpreferred
    # assignments preferentially go to under-filled slots.
    def deficit_priority(sp):
        assigned = capacity - remaining.get((sp, session), 0)
        return 0 if assigned < min_cap else 1

    ordered_speakers = sorted(speaker_sessions.keys(), key=deficit_priority)

    for speaker in ordered_speakers:
        if session not in speaker_sessions[speaker] or speaker in speakers_used:
            continue
        if remaining.get((speaker, session), 0) <= 0:
            continue
        assignment[session] = speaker
        speakers_used.add(speaker)
        remaining[(speaker, session)] -= 1
        if _fill_backtrack(unfilled, idx + 1, assignment, remaining, speakers_used, speaker_sessions, capacity, min_cap):
            return True
        del assignment[session]
        speakers_used.remove(speaker)
        remaining[(speaker, session)] += 1
    return False


def fix_deficits(results, remaining, capacity, min_cap, all_sessions):
    """
    Post-processing pass: move students from over-minimum slots into deficit
    slots (those with fewer than min_cap students assigned).

    For each deficit slot we rank every possible donor student by how much the
    move costs in terms of preference satisfaction:
      - Moving to a preferred speaker ranked HIGHER than the current one is
        beneficial (negative cost).
      - Moving between two unpreferred speakers is neutral.
      - Moving away from a preferred speaker to an unpreferred one is penalised.

    Only moves that keep the donor slot at or above min_cap are considered.
    Iterates until no further improving moves are possible.

    Mutates `results` assignment dicts and `remaining` in-place.
    Returns a list of warning strings for slots that remain below minimum.
    """
    if min_cap <= 0:
        return []

    changed = True
    while changed:
        changed = False
        for (def_sp, def_sess) in list(remaining.keys()):
            if def_sess not in all_sessions:
                continue
            if remaining[(def_sp, def_sess)] <= 0:
                continue  # Slot is at max capacity already
            def_assigned = capacity - remaining[(def_sp, def_sess)]
            if def_assigned >= min_cap:
                continue  # Already at or above minimum

            # Collect candidates: students in def_sess assigned to a different
            # speaker whose slot can afford to lose one student.
            candidates = []
            for i, (name, assignment, ranks) in enumerate(results):
                src_sp = assignment.get(def_sess)
                if src_sp is None or src_sp == def_sp:
                    continue
                src_assigned = capacity - remaining.get((src_sp, def_sess), 0)
                if src_assigned - 1 < min_cap:
                    continue  # Donor slot would drop below minimum

                # Score the move (lower = better for the student).
                src_rank = ranks.get(src_sp)   # None if assigned without preference
                def_rank = ranks.get(def_sp)   # None if deficit speaker not ranked

                if def_rank is not None and src_rank is not None:
                    score = def_rank - src_rank   # negative = upgrade for student
                elif def_rank is not None and src_rank is None:
                    score = -1                    # moving to a preferred speaker
                elif def_rank is None and src_rank is None:
                    score = 0                     # both unpreferred: neutral
                else:
                    score = 100 + src_rank        # losing a preferred speaker: bad

                candidates.append((score, i, src_sp))

            if not candidates:
                continue

            # Pick the least-cost move.
            candidates.sort(key=lambda c: c[0])
            _, best_idx, src_sp = candidates[0]

            remaining[(src_sp, def_sess)] += 1
            remaining[(def_sp, def_sess)] -= 1
            results[best_idx][1][def_sess] = def_sp
            changed = True

    warnings = []
    for (sp, sess), rem in remaining.items():
        if sess not in all_sessions:
            continue
        assigned = capacity - rem
        if assigned < min_cap:
            info = SPEAKER_INFO.get(sp, "")
            label = f"{sp} — {info}" if info else sp
            warnings.append(
                f"Under-minimum: {label} | Session {sess} has "
                f"{assigned}/{min_cap} students (not enough students to fill)."
            )
    return warnings


def assign_unranked_students(unranked_names, speaker_sessions, remaining, capacity, all_sessions):
    """
    Assign students who submitted no preferences to the least-populated slots.
    For each session the student is placed with whichever speaker currently has
    the fewest students assigned; ties are broken randomly.  A student will not
    be sent to the same speaker twice across sessions.
    Returns a list of [name, assignment dict, {}] entries (empty ranks).
    """
    results = []
    for name in unranked_names:
        assignment = {}
        speakers_used = set()
        for session in all_sessions:
            candidates = []
            for sp in speaker_sessions:
                if session not in speaker_sessions[sp] or sp in speakers_used:
                    continue
                rem = remaining.get((sp, session), 0)
                if rem <= 0:
                    continue
                assigned = capacity - rem
                candidates.append((assigned, sp))

            if not candidates:
                continue

            # Among the least-populated speakers, pick one at random.
            min_assigned = min(c[0] for c in candidates)
            least_full = [sp for assigned, sp in candidates if assigned == min_assigned]
            speaker = random.choice(least_full)

            assignment[session] = speaker
            speakers_used.add(speaker)
            remaining[(speaker, session)] -= 1

        results.append([name, assignment, {}])
    return results


def run_solver(session_data, students_data, capacity, min_cap, unranked_names=None):
    """
    session_data:    dict[session_num] -> list of speaker names
    students_data:   list of (name, {speaker: rank})
    capacity:        int (max students per speaker per session)
    min_cap:         int (min students per speaker per session)
    unranked_names:  optional list of student names with no preferences
    """
    speaker_sessions = defaultdict(set)
    remaining = {}
    for sess_num, speakers in session_data.items():
        for sp in speakers:
            sp = sp.strip().lower()
            if sp:
                speaker_sessions[sp].add(sess_num)
                remaining[(sp, sess_num)] = capacity

    all_sessions = sorted(session_data.keys())

    # Convert student ranks to ordered lists
    students = []
    for name, ranks in students_data:
        ranked = sorted(ranks.items(), key=lambda x: x[1])
        ranked_speakers = [sp for sp, _ in ranked]
        students.append((name, ranked_speakers, ranks))

    # Sort by constraint level
    students.sort(key=lambda s: len(s[1]))

    results = []
    warnings = []

    for student_name, ranked_speakers, ranks in students:
        assignment, speakers_used = solve_student_max_pref(
            ranked_speakers, speaker_sessions, all_sessions, remaining
        )

        if len(assignment) < len(all_sessions):
            fill_remaining_sessions(assignment, all_sessions, remaining, speakers_used, speaker_sessions, capacity, min_cap)

        if len(assignment) < len(all_sessions):
            warnings.append(f"Could not fully assign {student_name} ({len(assignment)}/{len(all_sessions)} sessions)")

        results.append([student_name, assignment, ranks])

    # Post-processing: move students to satisfy minimum slot fill.
    # results entries are lists so fix_deficits can mutate the assignment dict.
    deficit_warnings = fix_deficits(results, remaining, capacity, min_cap, all_sessions)
    warnings.extend(deficit_warnings)

    # Assign unranked students to the least-populated remaining slots.
    if unranked_names:
        unranked_results = assign_unranked_students(
            unranked_names, speaker_sessions, remaining, capacity, all_sessions
        )
        results.extend(unranked_results)

    # Restore input order for ranked students; unranked appear at the end.
    input_order = {name: i for i, (name, _) in enumerate(students_data)}
    results.sort(key=lambda r: input_order.get(r[0], 999999))

    return results, all_sessions, warnings


def compute_stats(results, all_sessions):
    rank_hits = defaultdict(int)
    total_pref = 0
    total = len(results) * len(all_sessions)
    unassigned_count = 0

    for name, assignment, ranks in results:
        ranked_list = sorted(ranks.items(), key=lambda x: x[1])
        ranked_speakers = [sp for sp, _ in ranked_list]
        for s in all_sessions:
            sp = assignment.get(s)
            if not sp or sp == "UNASSIGNED":
                unassigned_count += 1
                continue
            if sp in ranked_speakers:
                rank_hits[ranked_speakers.index(sp) + 1] += 1
                total_pref += 1

    return {
        "total": total,
        "total_pref": total_pref,
        "rank_hits": dict(rank_hits),
        "no_pref": total - total_pref - unassigned_count,
        "unassigned": unassigned_count,
    }


def format_assignment(speaker, session_num):
    """Return a display string for one assignment cell, enriched with session info."""
    if not speaker or speaker == "UNASSIGNED":
        return "UNASSIGNED"
    info = SPEAKER_INFO.get(speaker, "")
    return f"{speaker} — {info}" if info else speaker


def results_to_csv(results, all_sessions):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Student"] + [f"Session{s}" for s in all_sessions])
    for name, assignment, _ in results:
        row = [name] + [format_assignment(assignment.get(s, "UNASSIGNED"), s) for s in all_sessions]
        writer.writerow(row)
    return output.getvalue()


# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────

st.set_page_config(page_title="Student-Speaker Assignments", layout="wide")
st.title("🎤 Student-Speaker Assignment Tool")

# ── Session Data Entry ──
st.header("1. Session Data")

session_mode = st.radio("How would you like to enter session data?", ["Manual entry", "Upload CSV"], horizontal=True)

if session_mode == "Upload CSV":
    st.write("Upload a CSV with columns: `Speaker`, `Session`, `Capacity`")
    sessions_file = st.file_uploader("Choose sessions CSV", type=["csv"], key="sessions_upload")
    if sessions_file is not None:
        text = sessions_file.read().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        session_data = defaultdict(list)
        uploaded_capacity = None
        for row in reader:
            sp = row["Speaker"].strip().lower()
            sess = int(row["Session"].strip())
            cap = int(row["Capacity"].strip())
            if sp:
                session_data[sess].append(sp)
            uploaded_capacity = cap
        st.session_state.session_data = dict(session_data)
        if uploaded_capacity is not None:
            st.session_state.uploaded_capacity = uploaded_capacity
        st.success(f"Loaded {len(session_data)} sessions!")
        for s in sorted(session_data):
            st.write(f"**Session {s}:** {', '.join(session_data[s])}")

if session_mode == "Manual entry":
    st.write("Add speakers to each session. Use the **+** buttons to add more slots or sessions.")

if session_mode == "Manual entry":
    if "num_sessions" not in st.session_state:
        st.session_state.num_sessions = 4
    if "session_slots" not in st.session_state:
        st.session_state.session_slots = {i: 4 for i in range(1, 5)}
    if "session_values" not in st.session_state:
        st.session_state.session_values = {}

    def add_session():
        st.session_state.num_sessions += 1
        n = st.session_state.num_sessions
        st.session_state.session_slots[n] = 4

    def add_slot(sess):
        st.session_state.session_slots[sess] += 1

    num_sess = st.session_state.num_sessions
    cols = st.columns(num_sess + 1)

    for sess_idx in range(1, num_sess + 1):
        with cols[sess_idx - 1]:
            st.subheader(f"Session {sess_idx}")
            num_slots = st.session_state.session_slots[sess_idx]
            for slot in range(num_slots):
                key = f"sess_{sess_idx}_slot_{slot}"
                default = st.session_state.session_values.get(key, "")
                val = st.text_input(
                    f"Speaker {slot + 1}",
                    value=default,
                    key=key,
                    label_visibility="collapsed",
                    placeholder=f"Speaker {slot + 1}"
                )
                st.session_state.session_values[key] = val
            st.button("➕ Add slot", key=f"add_slot_{sess_idx}", on_click=add_slot, args=(sess_idx,))

    with cols[num_sess]:
        st.subheader("")
        st.button("➕ Add session", on_click=add_session)

    if st.button("✅ Submit Session Data", type="primary"):
        session_data = {}
        for sess_idx in range(1, num_sess + 1):
            speakers = []
            num_slots = st.session_state.session_slots[sess_idx]
            for slot in range(num_slots):
                key = f"sess_{sess_idx}_slot_{slot}"
                val = st.session_state.session_values.get(key, "").strip()
                if val:
                    speakers.append(val.lower())
            if speakers:
                session_data[sess_idx] = speakers
        st.session_state.session_data = session_data
        st.success(f"Saved {len(session_data)} sessions!")
        for s, spkrs in session_data.items():
            st.write(f"**Session {s}:** {', '.join(spkrs)}")

# Capacity inputs
default_cap = st.session_state.get("uploaded_capacity", 15)
cap_col, min_col = st.columns(2)
with cap_col:
    capacity = st.number_input("Max students per speaker per session", min_value=1, value=default_cap, step=1)
with min_col:
    min_cap = st.number_input("Min students per speaker per session", min_value=0, value=0, step=1)
if min_cap > capacity:
    st.error("Minimum cannot exceed maximum capacity.")

# ── Student File Upload ──
st.header("2. Upload Student Rankings")
st.write("Upload the raw student survey CSV. It will be cleaned automatically.")

uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

if uploaded_file is not None:
    raw_bytes = uploaded_file.read()
    students_data, error, duplicate_count = clean_student_csv(raw_bytes)
    if error:
        st.error(f"Error cleaning CSV: {error}")
    else:
        st.session_state.students_data = students_data
        msg = f"Loaded {len(students_data)} students!"
        if duplicate_count > 0:
            msg += (
                f" ({duplicate_count} duplicate "
                f"submission{'s' if duplicate_count != 1 else ''} detected — "
                "preferences updated to the most recent response.)"
            )
        st.success(msg)
        with st.expander("Preview student data"):
            preview_rows = []
            for name, ranks in students_data[:10]:
                ranked = sorted(ranks.items(), key=lambda x: x[1])
                top = ", ".join([f"{sp} (#{r})" for sp, r in ranked[:5]])
                preview_rows.append({"Student": name, "Top 5 Choices": top})
            st.table(preview_rows)

# ── Unranked Students ──
st.header("2b. Students Without Preferences (Optional)")
st.write("Students who registered but did not submit preferences. They will be placed in the least-populated sessions.")

unranked_mode = st.radio(
    "How would you like to enter these students?",
    ["Manual entry", "Upload TXT"],
    horizontal=True,
    key="unranked_mode",
)

if unranked_mode == "Manual entry":
    names_text = st.text_area("Enter one student name per line", key="unranked_manual_text")
    if st.button("✅ Save Unranked Students", key="save_unranked"):
        names = [n.strip() for n in names_text.splitlines() if n.strip()]
        st.session_state.unranked_students = names
        st.success(f"Saved {len(names)} unranked student{'s' if len(names) != 1 else ''}.")

elif unranked_mode == "Upload TXT":
    unranked_file = st.file_uploader("Upload TXT file (one name per line)", type=["txt"], key="unranked_upload")
    if unranked_file is not None:
        content = unranked_file.read().decode("utf-8")
        names = [n.strip() for n in content.splitlines() if n.strip()]
        st.session_state.unranked_students = names
        st.success(f"Loaded {len(names)} unranked student{'s' if len(names) != 1 else ''}.")

if "unranked_students" in st.session_state and st.session_state.unranked_students:
    st.caption(f"{len(st.session_state.unranked_students)} unranked students queued.")
    if st.button("Clear unranked students", key="clear_unranked"):
        st.session_state.unranked_students = []
        st.rerun()

# ── Run Solver ──
st.header("3. Generate Assignments")

can_run = "session_data" in st.session_state and "students_data" in st.session_state

if not can_run:
    missing = []
    if "session_data" not in st.session_state:
        missing.append("session data")
    if "students_data" not in st.session_state:
        missing.append("student rankings")
    st.info(f"Please submit {' and '.join(missing)} first.")

if can_run and st.button("🚀 Generate Student Assignments", type="primary", disabled=(min_cap > capacity)):
    with st.spinner("Running assignment solver..."):
        results, all_sessions, warnings = run_solver(
            st.session_state.session_data,
            st.session_state.students_data,
            capacity,
            min_cap,
            st.session_state.get("unranked_students", []),
        )
    st.session_state.results = results
    st.session_state.all_sessions = all_sessions
    st.session_state.warnings = warnings

if "results" in st.session_state:
    results = st.session_state.results
    all_sessions = st.session_state.all_sessions
    warnings = st.session_state.warnings

    if warnings:
        for w in warnings:
            st.warning(w)

    # Display results table
    st.subheader("Assignment Results")
    table_rows = []
    for name, assignment, _ in results:
        row = {"Student": name}
        for s in all_sessions:
            row[f"Session {s}"] = format_assignment(assignment.get(s, "UNASSIGNED"), s)
        table_rows.append(row)
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    # Stats
    stats = compute_stats(results, all_sessions)
    st.subheader("Assignment Statistics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Assignments", stats["total"])
    col2.metric("Matched a Preference", stats["total_pref"])
    col3.metric("No Preference Match", stats["no_pref"])

    if stats["unassigned"] > 0:
        st.error(f"⚠️ {stats['unassigned']} unassigned slots")

    st.write("**Preference breakdown:**")
    rank_hits = stats["rank_hits"]
    if rank_hits:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        sorted_ranks = sorted(rank_hits.keys())
        labels = [f"Rank {r}" for r in sorted_ranks]
        values = [rank_hits[r] for r in sorted_ranks]

        fig = plt.figure(figsize=(5, 2.5), dpi=100)
        ax = fig.add_subplot(111)
        ax.bar(range(len(labels)), values, width=0.6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Assignments", fontsize=8)
        ax.set_xlabel("Rank", fontsize=8)
        ax.tick_params(axis='y', labelsize=7)
        fig.tight_layout()

        col_chart, col_spacer = st.columns([2, 3])
        with col_chart:
            st.pyplot(fig)
        plt.close(fig)

        for r in sorted_ranks:
            st.write(f"- **Rank {r}:** {rank_hits[r]} assignments")

    # Download
    csv_output = results_to_csv(results, all_sessions)
    st.download_button(
        label="📥 Download Assignments CSV",
        data=csv_output,
        file_name="assignments.csv",
        mime="text/csv",
        type="primary"
    )
