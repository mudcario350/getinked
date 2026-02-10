#!/usr/bin/env python3




"""
Student-Speaker Assignment App (Streamlit)

Run with: streamlit run app.py
"""

import streamlit as st
import csv
import io
import re
from collections import defaultdict

# ─────────────────────────────────────────────
# Canonical speaker order (for cleaning raw CSV)
# ─────────────────────────────────────────────
CANONICAL_SPEAKERS = [
    "barshaw", "bilen", "kat h-c", "johnson", "korn", "lenz", "maloche",
    "nash", "tebbetts", "broome", "cordi", "flynn", "alex h-c", "layman",
    "stutsman", "twietmeyer", "lederman"
]

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

    students = []
    for row in data:
        if len(row) <= student_col:
            continue
        name = row[student_col].strip()
        if not name:
            continue
        ranks = {}
        for i in speaker_cols:
            if i in col_to_canon and i < len(row):
                rank = parse_choice(row[i])
                if rank is not None:
                    ranks[col_to_canon[i]] = rank
        students.append((name, ranks))

    return students, None


# ─────────────────────────────────────────────
# Assignment Solver (Greedy + Backtracking)
# ─────────────────────────────────────────────

def build_student_options(ranked_speakers, speaker_sessions, all_sessions):
    options = []
    for speaker in ranked_speakers:
        if speaker in speaker_sessions:
            for session in sorted(speaker_sessions[speaker]):
                options.append((speaker, session))
    return options


def solve_student(options, all_sessions, remaining, assignment, sessions_filled, speakers_used, idx=0):
    if len(sessions_filled) == len(all_sessions):
        return True
    for i in range(idx, len(options)):
        speaker, session = options[i]
        if session in sessions_filled or speaker in speakers_used:
            continue
        if remaining.get((speaker, session), 0) <= 0:
            continue
        assignment[session] = speaker
        sessions_filled.add(session)
        speakers_used.add(speaker)
        remaining[(speaker, session)] -= 1
        if solve_student(options, all_sessions, remaining, assignment, sessions_filled, speakers_used, i + 1):
            return True
        del assignment[session]
        sessions_filled.remove(session)
        speakers_used.remove(speaker)
        remaining[(speaker, session)] += 1
    return False


def fill_remaining_sessions(assignment, all_sessions, remaining, speakers_used, speaker_sessions):
    unfilled = [s for s in all_sessions if s not in assignment]
    if not unfilled:
        return True
    return _fill_backtrack(unfilled, 0, assignment, remaining, speakers_used, speaker_sessions)


def _fill_backtrack(unfilled, idx, assignment, remaining, speakers_used, speaker_sessions):
    if idx == len(unfilled):
        return True
    session = unfilled[idx]
    for speaker, sessions in speaker_sessions.items():
        if session not in sessions or speaker in speakers_used:
            continue
        if remaining.get((speaker, session), 0) <= 0:
            continue
        assignment[session] = speaker
        speakers_used.add(speaker)
        remaining[(speaker, session)] -= 1
        if _fill_backtrack(unfilled, idx + 1, assignment, remaining, speakers_used, speaker_sessions):
            return True
        del assignment[session]
        speakers_used.remove(speaker)
        remaining[(speaker, session)] += 1
    return False


def run_solver(session_data, students_data, capacity):
    """
    session_data: dict[session_num] -> list of speaker names
    students_data: list of (name, {speaker: rank})
    capacity: int (max per speaker per session)
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
        options = build_student_options(ranked_speakers, speaker_sessions, all_sessions)
        assignment = {}
        sessions_filled = set()
        speakers_used = set()

        solve_student(options, all_sessions, remaining, assignment, sessions_filled, speakers_used)

        if len(assignment) < len(all_sessions):
            fill_remaining_sessions(assignment, all_sessions, remaining, speakers_used, speaker_sessions)

        if len(assignment) < len(all_sessions):
            warnings.append(f"Could not fully assign {student_name} ({len(assignment)}/{len(all_sessions)} sessions)")

        results.append((student_name, assignment, ranks))

    # Restore original order
    original_order = {name: i for i, (name, _, _) in enumerate(
        sorted([(n, rs, r) for n, rs, r in [(name, ranks, ranks) for name, ranks in students_data]], key=lambda x: 0)
    )}
    # Actually just use input order
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


def results_to_csv(results, all_sessions):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Student"] + [f"Session{s}" for s in all_sessions])
    for name, assignment, _ in results:
        row = [name] + [assignment.get(s, "UNASSIGNED") for s in all_sessions]
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

# Capacity (use uploaded value as default if available)
default_cap = st.session_state.get("uploaded_capacity", 15)
capacity = st.number_input("Max students per speaker per session", min_value=1, value=default_cap, step=1)

# ── Student File Upload ──
st.header("2. Upload Student Rankings")
st.write("Upload the raw student survey CSV. It will be cleaned automatically.")

uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

if uploaded_file is not None:
    raw_bytes = uploaded_file.read()
    students_data, error = clean_student_csv(raw_bytes)
    if error:
        st.error(f"Error cleaning CSV: {error}")
    else:
        st.session_state.students_data = students_data
        st.success(f"Loaded {len(students_data)} students!")
        with st.expander("Preview student data"):
            preview_rows = []
            for name, ranks in students_data[:10]:
                ranked = sorted(ranks.items(), key=lambda x: x[1])
                top = ", ".join([f"{sp} (#{r})" for sp, r in ranked[:5]])
                preview_rows.append({"Student": name, "Top 5 Choices": top})
            st.table(preview_rows)

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

if can_run and st.button("🚀 Generate Student Assignments", type="primary"):
    with st.spinner("Running assignment solver..."):
        results, all_sessions, warnings = run_solver(
            st.session_state.session_data,
            st.session_state.students_data,
            capacity
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
            row[f"Session {s}"] = assignment.get(s, "UNASSIGNED")
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
        sorted_ranks = sorted(rank_hits.keys())
        labels = [f"Rank {r}" for r in sorted_ranks]
        values = [rank_hits[r] for r in sorted_ranks]

        fig, ax = plt.subplots()
        ax.bar(range(len(labels)), values)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Assignments")
        ax.set_xlabel("Rank")
        fig.tight_layout()
        st.pyplot(fig)

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
