import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import random
from datetime import datetime

# ============================================================
# CONSTANTS
# ============================================================
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TIME_SLOTS_PER_DAY = 5
TOTAL_SLOTS = len(DAYS) * TIME_SLOTS_PER_DAY
HARD_PENALTY = 1000
SOFT_PENALTY = 1
TIME_LABELS = ["8:00-10:00", "10:00-12:00", "12:00-14:00", "14:00-16:00", "16:00-18:00"]

def slot_to_day_and_time(slot_id):
    slot_index = slot_id - 1
    day_index = slot_index // TIME_SLOTS_PER_DAY
    time_index = slot_index % TIME_SLOTS_PER_DAY
    return DAYS[day_index], TIME_LABELS[time_index]

# ============================================================
# DATA LOADING AND VALIDATION
# ============================================================
def load_csv(file):
    return pd.read_csv(file)

def validate_data(courses, instructors, rooms, enrollments):
    errors = []
    course_ids = set(courses['course_id'].tolist())
    instructor_ids = set(instructors['instructor_id'].tolist())
    for _, course in courses.iterrows():
        if course['instructor_id'] not in instructor_ids:
            errors.append(f"Course {course['course_id']} references unknown instructor")
    for _, enroll in enrollments.iterrows():
        if enroll['course_id'] not in course_ids:
            errors.append(f"Enrollment {enroll['enrollment_id']} references unknown course")
    return errors

# ============================================================
# CONSTRAINT CHECKER
# ============================================================
def check_constraints(chromosome, courses, instructors, rooms, enrollments):
    num_courses = len(courses)
    violations = {k: 0 for k in [
        'instructor_clash', 'room_clash', 'student_clash',
        'capacity_violation', 'room_type_violation',
        'instructor_unavailability', 'instructor_preference',
        'student_gaps', 'workload_imbalance', 'back_to_back_buildings']}
    course_list = courses.to_dict('records')
    instructor_list = instructors.to_dict('records')
    room_list = rooms.to_dict('records')
    instructor_by_id = {i['instructor_id']: i for i in instructor_list}

    assignments = {i: (int(chromosome[i*2]), int(chromosome[i*2+1])) for i in range(num_courses)}

    # HC1 Instructor clash
    inst_slot = {}
    for i in range(num_courses):
        key = (course_list[i]['instructor_id'], assignments[i][0])
        inst_slot[key] = inst_slot.get(key, 0) + 1
    for count in inst_slot.values():
        if count > 1:
            violations['instructor_clash'] += count - 1

    # HC2 Room clash
    room_slot = {}
    for i in range(num_courses):
        key = (assignments[i][1], assignments[i][0])
        room_slot[key] = room_slot.get(key, 0) + 1
    for count in room_slot.values():
        if count > 1:
            violations['room_clash'] += count - 1

        # HC3 Student clash (optimised with precomputed course index)
    course_index = {c['course_id']: i for i, c in enumerate(course_list)}
    student_courses = {}
    for _, e in enrollments.iterrows():
        student_courses.setdefault(e['student_id'], []).append(e['course_id'])
    for sid, cids in student_courses.items():
        slot_count = {}
        for cid in cids:
            idx = course_index.get(cid)
            if idx is not None:
                slot = assignments[idx][0]
                slot_count[slot] = slot_count.get(slot, 0) + 1
        for count in slot_count.values():
            if count > 1:
                violations['student_clash'] += count - 1

    # HC4 Capacity
    for i in range(num_courses):
        if assignments[i][1] < len(room_list):
            if course_list[i]['enrollment_count'] > room_list[assignments[i][1]]['capacity']:
                violations['capacity_violation'] += 1

    # HC5 Room type
    for i in range(num_courses):
        if assignments[i][1] < len(room_list):
            if course_list[i]['required_room_type'] != room_list[assignments[i][1]]['room_type']:
                violations['room_type_violation'] += 1

    # Soft constraints
    for i in range(num_courses):
        inst = instructor_by_id.get(course_list[i]['instructor_id'])
        slot = assignments[i][0]
        if inst:
            unavail = [int(x.strip()) for x in str(inst['unavailable_slots']).split(',') if x.strip().isdigit()]
            if slot in unavail:
                violations['instructor_unavailability'] += 1
            preferred = [int(x.strip()) for x in str(inst['preferred_slots']).split(',') if x.strip().isdigit()]
            if preferred and slot not in preferred:
                violations['instructor_preference'] += 1

    # Workload imbalance
    inst_day = {}
    for i in range(num_courses):
        day, _ = slot_to_day_and_time(assignments[i][0])
        key = (course_list[i]['instructor_id'], day)
        inst_day[key] = inst_day.get(key, 0) + 1
    for count in inst_day.values():
        if count > 3:
            violations['workload_imbalance'] += count - 3

    return violations

def calculate_fitness(violations):
    hard = ['instructor_clash', 'room_clash', 'student_clash',
            'capacity_violation', 'room_type_violation']
    soft = ['instructor_unavailability', 'instructor_preference',
            'student_gaps', 'workload_imbalance', 'back_to_back_buildings']
    total = sum(violations[v] * HARD_PENALTY for v in hard)
    total += sum(violations[v] * SOFT_PENALTY for v in soft)
    return 1.0 / (1.0 + total), total

# ============================================================
# GENETIC ALGORITHM
# ============================================================
def init_population(size, num_courses, courses, rooms):
    room_list = rooms.to_dict('records')
    course_list = courses.to_dict('records')
    population = []
    for _ in range(size):
        chrom = []
        for i in range(num_courses):
            slot = random.randint(1, TOTAL_SLOTS)
            course = course_list[i]
            compatible = [r for r, rm in enumerate(room_list)
                          if rm['capacity'] >= course['enrollment_count']
                          and rm['room_type'] == course['required_room_type']]
            room_idx = random.choice(compatible) if compatible else random.randint(0, len(room_list)-1)
            chrom.append(slot)
            chrom.append(room_idx)
        population.append(np.array(chrom))
    return population

def tournament_selection(population, fitnesses, k=3):
    idxs = random.sample(range(len(population)), k)
    best = max(idxs, key=lambda i: fitnesses[i])
    return population[best].copy()

def uniform_crossover(p1, p2, rate=0.9):
    if random.random() > rate:
        return p1.copy()
    child = np.zeros_like(p1)
    for i in range(len(p1)):
        child[i] = p1[i] if random.random() < 0.5 else p2[i]
    return child

def mutate(chrom, courses, rooms, rate=0.05):
    course_list = courses.to_dict('records')
    room_list = rooms.to_dict('records')
    for i in range(len(course_list)):
        if random.random() < rate:
            chrom[i*2] = random.randint(1, TOTAL_SLOTS)
        if random.random() < rate:
            course = course_list[i]
            compatible = [r for r, rm in enumerate(room_list)
                          if rm['capacity'] >= course['enrollment_count']
                          and rm['room_type'] == course['required_room_type']]
            chrom[i*2+1] = random.choice(compatible) if compatible else random.randint(0, len(room_list)-1)
    return chrom

def run_ga(courses, instructors, rooms, enrollments,
           pop_size=200, generations=500, mutation_rate=0.05,
           elitism=2, stagnation_limit=20):
    num_courses = len(courses)
    population = init_population(pop_size, num_courses, courses, rooms)
    history = []
    best_fit = 0
    best_chrom = None
    stagnation = 0
    progress = st.progress(0)
    status = st.empty()
    start = time.time()

    for gen in range(generations):
        fitnesses = []
        for chrom in population:
            v = check_constraints(chrom, courses, instructors, rooms, enrollments)
            f, _ = calculate_fitness(v)
            fitnesses.append(f)
        gen_best_idx = int(np.argmax(fitnesses))
        gen_best = fitnesses[gen_best_idx]
        if gen_best > best_fit:
            best_fit = gen_best
            best_chrom = population[gen_best_idx].copy()
            stagnation = 0
        else:
            stagnation += 1
        history.append(best_fit)
        progress.progress(min((gen+1)/generations, 1.0))
        status.text(f"Generation {gen+1}/{generations} | Best Fitness: {best_fit:.4f}")
        if best_fit >= 1.0 or stagnation >= stagnation_limit:
            break

        new_pop = []
        sorted_idx = np.argsort(fitnesses)[::-1]
        for i in range(elitism):
            new_pop.append(population[sorted_idx[i]].copy())
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            child = uniform_crossover(p1, p2)
            child = mutate(child, courses, rooms, mutation_rate)
            new_pop.append(child)
        population = new_pop

    end = time.time()
    progress.empty()
    status.empty()
    final_v = check_constraints(best_chrom, courses, instructors, rooms, enrollments)
    final_fit, _ = calculate_fitness(final_v)
    return {
        'chromosome': best_chrom,
        'fitness': final_fit,
        'violations': final_v,
        'history': history,
        'time': end - start,
        'generations': len(history)
    }

# ============================================================
# TIMETABLE FORMATTER
# ============================================================
def format_timetable(chrom, courses, rooms):
    course_list = courses.to_dict('records')
    room_list = rooms.to_dict('records')
    grid = {d: {t: [] for t in TIME_LABELS} for d in DAYS}
    for i in range(len(course_list)):
        slot = int(chrom[i*2])
        room_idx = int(chrom[i*2+1])
        day, time_label = slot_to_day_and_time(slot)
        entry = f"{course_list[i]['course_id']} - {room_list[room_idx]['name']}"
        grid[day][time_label].append(entry)
    rows = []
    for t in TIME_LABELS:
        row = {'Time': t}
        for d in DAYS:
            row[d] = ", ".join(grid[d][t]) if grid[d][t] else "Free"
        rows.append(row)
    return pd.DataFrame(rows)

def convergence_chart(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(history)+1)), y=history,
                             mode='lines', name='Best Fitness',
                             line=dict(color='blue', width=2)))
    fig.update_layout(title='Genetic Algorithm Convergence',
                      xaxis_title='Generation', yaxis_title='Best Fitness',
                      yaxis=dict(range=[0, 1.05]), template='plotly_white')
    return fig

# ============================================================
# STREAMLIT INTERFACE
# ============================================================
st.set_page_config(page_title="AI Class Scheduler", layout="wide")
st.title("AI-Based Class Scheduling System")
st.subheader("Genetic Algorithm Optimization for Departmental Timetabling")

st.sidebar.header("1. Upload Data Files")
cf = st.sidebar.file_uploader("Upload courses.csv", type=['csv'])
inf = st.sidebar.file_uploader("Upload instructors.csv", type=['csv'])
rf = st.sidebar.file_uploader("Upload rooms.csv", type=['csv'])
ef = st.sidebar.file_uploader("Upload enrollments.csv", type=['csv'])

st.sidebar.header("2. Configure GA Parameters")
pop = st.sidebar.slider("Population Size", 50, 500, 200, 50)
gens = st.sidebar.slider("Number of Generations", 50, 1000, 500, 50)
mut = st.sidebar.slider("Mutation Rate", 0.01, 0.20, 0.05, 0.01)
elit = st.sidebar.slider("Elitism Count", 1, 10, 2, 1)

t1, t2, t3, t4 = st.tabs(["Input Summary", "Generated Timetable", "Fitness Chart", "Export"])

if cf and inf and rf and ef:
    courses = load_csv(cf)
    instructors = load_csv(inf)
    rooms = load_csv(rf)
    enrollments = load_csv(ef)
    errors = validate_data(courses, instructors, rooms, enrollments)

    with t1:
        st.header("Uploaded Data Summary")
        if errors:
            for e in errors:
                st.error(e)
        else:
            st.success("All data validated successfully.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Courses", len(courses))
            c2.metric("Instructors", len(instructors))
            c3.metric("Rooms", len(rooms))
            c4.metric("Enrollments", len(enrollments))
            st.subheader("Courses")
            st.dataframe(courses)

    if st.sidebar.button("Generate Schedule") and not errors:
        results = run_ga(courses, instructors, rooms, enrollments,
                         pop_size=pop, generations=gens,
                         mutation_rate=mut, elitism=elit)
        st.session_state['results'] = results
        st.session_state['courses'] = courses
        st.session_state['rooms'] = rooms

    if 'results' in st.session_state:
        r = st.session_state['results']
        courses = st.session_state['courses']
        rooms = st.session_state['rooms']

        with t2:
            st.header("Generated Timetable")
            st.dataframe(format_timetable(r['chromosome'], courses, rooms),
                         use_container_width=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Best Fitness", f"{r['fitness']:.4f}")
            c2.metric("Execution Time", f"{r['time']:.2f} s")
            c3.metric("Generations Run", r['generations'])
            hard = sum(r['violations'][k] for k in
                       ['instructor_clash', 'room_clash', 'student_clash',
                        'capacity_violation', 'room_type_violation'])
            c4.metric("Hard Violations", hard)

        with t3:
            st.header("Fitness Convergence")
            st.plotly_chart(convergence_chart(r['history']), use_container_width=True)

        with t4:
            st.header("Export Results")
            df = format_timetable(r['chromosome'], courses, rooms)
            st.download_button("Download Timetable as CSV",
                               df.to_csv(index=False),
                               file_name=f"timetable_{datetime.now().strftime('%Y%m%d')}.csv",
                               mime="text/csv")
            st.json(r['violations'])
else:
    st.info("Please upload all four CSV files using the sidebar.")