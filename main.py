import streamlit as st
import numpy as np
import pandas as pd
import random
from collections import defaultdict
import json
import os
import io  # <--- Додайте цей імпорт


class SeatingArrangement:
    def __init__(self, students):
        """
        Ініціалізація з списком імен учнів
        """
        self.students = students
        self.num_students = len(students)
        # Словник для зберігання пріоритетів кожного учня
        self.preferences = {}
        # Матриця ваг для оптимізації
        self.weight_matrix = np.zeros((self.num_students, self.num_students))
        # Словник для відстеження, хто з ким сидів останнім часом
        self.recent_seatings = defaultdict(set)

    def add_student_preferences(self, student, preferences, weights=None):
        """
        Додавання переваг для учня
        student: ім'я учня
        preferences: список з 4 імен учнів у порядку пріоритету
        weights: вага для кожного пріоритету, за замовчуванням [4, 3, 2, 1]
        """
        if weights is None:
            weights = [4, 3, 2, 1]

        if len(preferences) != 4:
            raise ValueError("Кожен учень має надати 4 пріоритети")

        self.preferences[student] = preferences

        # Оновлення матриці ваг
        student_idx = self.students.index(student)
        for pref, weight in zip(preferences, weights):
            if pref in self.students:
                pref_idx = self.students.index(pref)
                self.weight_matrix[student_idx, pref_idx] = weight

    def update_recent_seatings(self, arrangement):
        """
        Оновлення історії розсадки
        arrangement: список пар учнів
        """
        for pair in arrangement:
            if len(pair) == 2:  # Пара учнів
                student1, student2 = pair
                self.recent_seatings[student1].add(student2)
                self.recent_seatings[student2].add(student1)

        # Лімітуємо кількість останніх розсаджень, які ми пам'ятаємо
        for student in self.students:
            if len(self.recent_seatings[student]) > 3:  # пам'ятаємо 3 останні тижні
                self.recent_seatings[student] = set(list(self.recent_seatings[student])[-3:])

    def compute_seating_score(self, pair):
        """
        Обчислення оцінки для пари учнів
        """
        student1, student2 = pair
        idx1 = self.students.index(student1)
        idx2 = self.students.index(student2)

        # Основна оцінка на основі пріоритетів
        score = self.weight_matrix[idx1, idx2] + self.weight_matrix[idx2, idx1]

        # Штраф, якщо учні вже сиділи разом нещодавно
        if student2 in self.recent_seatings[student1]:
            score -= 5

        return score

    def optimize_seating(self):
        """
        Створення оптимальної розсадки учнів
        """
        available_students = set(self.students)
        arrangement = []

        # Спочатку формуємо можливі пари та сортуємо їх за оцінкою
        all_possible_pairs = []
        for i, student1 in enumerate(self.students):
            for j, student2 in enumerate(self.students[i + 1:], i + 1):
                score = self.compute_seating_score((student1, student2))
                all_possible_pairs.append((student1, student2, score))

        # Сортуємо пари за оцінкою (від вищої до нижчої)
        all_possible_pairs.sort(key=lambda x: x[2], reverse=True)

        # Додаємо випадковість, щоб не було передбачуваним
        random.shuffle(all_possible_pairs[:len(all_possible_pairs) // 3])

        # Створюємо розсадку
        for student1, student2, score in all_possible_pairs:
            if student1 in available_students and student2 in available_students:
                arrangement.append((student1, student2))
                available_students.remove(student1)
                available_students.remove(student2)

        # Якщо залишився непарний учень, він сидить сам
        if available_students:
            arrangement.append((list(available_students)[0],))

        return arrangement

    def generate_new_arrangement(self):
        """
        Генерує нову розсадку, оновлює історію та повертає результат
        """
        arrangement = self.optimize_seating()
        self.update_recent_seatings(arrangement)
        return arrangement


def save_data(students, preferences, history):
    """Зберігає дані у файл"""
    data = {
        "students": students,
        "preferences": preferences,
        "history": history
    }

    with open("class_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_data():
    """Завантажує дані з файлу"""
    if os.path.exists("class_data.json"):
        with open("class_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["students"], data["preferences"], data["history"]
    return [], {}, []


def process_csv(csv_file):
    """Обробляє завантажений CSV файл з пріоритетами учнів"""
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')

        # Перевірка структури файлу
        if 'Ім\'я' not in df.columns and df.shape[1] >= 5:
            # Якщо немає колонки "Ім'я", але є 5 колонок, припускаємо, що перша колонка - імена
            df.columns = ['Ім\'я', '1', '2', '3', '4']

        if 'Ім\'я' not in df.columns or '1' not in df.columns or '2' not in df.columns or '3' not in df.columns or '4' not in df.columns:
            return False, "Файл має неправильний формат. Перевірте колонки: 'Ім'я', '1', '2', '3', '4'", None, None

        # Витягуємо список учнів
        students = df['Ім\'я'].tolist()

        # Витягуємо переваги
        preferences = {}
        for _, row in df.iterrows():
            student = row['Ім\'я']
            prefs = [row['1'], row['2'], row['3'], row['4']]
            preferences[student] = prefs

        return True, "Дані успішно завантажено", students, preferences
    except Exception as e:
        return False, f"Помилка при обробці файлу: {str(e)}", None, None


# Streamlit додаток
st.set_page_config(page_title="Система розсадки учнів", layout="wide")

st.title("🏫 Система оптимальної розсадки учнів")

st.logo("stul.png")

st.markdown(
    """
    <link rel="icon" href="stul.png" type="image/png">
    """,
    unsafe_allow_html=True
)

# Завантаження даних з файлу
saved_students, saved_preferences, saved_history = load_data()

# Sidebar для управління списком учнів
with st.sidebar:
    st.header("📋 Список учнів класу")

    # Текстове поле для введення списку учнів
    students_text = st.text_area(
        "Введіть імена учнів (кожне ім'я з нового рядка):",
        "\n".join(saved_students) if saved_students else "",
        height=200
    )

    # Кнопка для оновлення списку учнів
    if st.button("Оновити список учнів"):
        students = [name.strip() for name in students_text.split("\n") if name.strip()]
        st.success(f"Список учнів оновлено! Усього учнів: {len(students)}")
        # Очищаємо історію при зміні складу класу
        saved_history = []
        saved_preferences = {}
        save_data(students, saved_preferences, saved_history)
    else:
        students = saved_students if saved_students else []

    # Завантаження CSV
    st.subheader("📤 Завантаження з CSV")
    uploaded_file = st.file_uploader("Завантажити пріоритети з CSV файлу", type="csv")

    if uploaded_file is not None:
        success, message, csv_students, csv_preferences = process_csv(uploaded_file)

        if success:
            st.success(message)

            # Додаємо кнопку для застосування даних з CSV
            if st.button("Застосувати дані з CSV"):
                saved_students = csv_students
                saved_preferences = csv_preferences
                # Очищаємо історію при зміні пріоритетів
                saved_history = []
                save_data(saved_students, saved_preferences, saved_history)
                st.success("Дані з CSV файлу успішно застосовано!")
                st.rerun()
        else:
            st.error(message)

    # Кнопка експорту поточних пріоритетів у CSV
    if saved_students and saved_preferences:
        st.subheader("📥 Експорт у CSV")
        if st.button("Експортувати поточні пріоритети в CSV"):
            data = []
            for student in saved_students:
                if student in saved_preferences:
                    prefs = saved_preferences[student]
                    data.append([student] + prefs)
                else:
                    data.append([student, "", "", "", ""])

            df_export = pd.DataFrame(data, columns=["Ім'я", "1", "2", "3", "4"])
            csv = df_export.to_csv(index=False).encode('utf-8')

            st.download_button(
                "Завантажити CSV файл",
                csv,
                "priorities.csv",
                "text/csv",
                key='download-priorities-csv'
            )

# Основний інтерфейс
tab1, tab2, tab3 = st.tabs(["Переваги учнів", "Створити розсадку", "Історія розсадок"])

# Вкладка для введення переваг учнів
with tab1:
    st.header("🔄 Внесіть переваги учнів")

    if not students:
        st.warning("Спершу додайте список учнів у бічній панелі")
    else:
        col1, col2 = st.columns(2)

        with col1:
            # Вибір учня для редагування переваг
            student_to_edit = st.selectbox(
                "Виберіть учня для введення переваг:",
                students
            )

            other_students = [s for s in students if s != student_to_edit]

            # Отримуємо поточні переваги (якщо є)
            current_prefs = saved_preferences.get(student_to_edit, [])

            # Вибір однокласників за пріоритетами
            pref1 = st.selectbox(
                "1-й пріоритет (найбільше бажання):",
                [""] + other_students,
                index=other_students.index(current_prefs[0]) + 1 if current_prefs and len(current_prefs) > 0 and
                                                                    current_prefs[0] in other_students else 0
            )

            remaining_students1 = [s for s in other_students if s != pref1]
            pref2 = st.selectbox(
                "2-й пріоритет:",
                [""] + remaining_students1,
                index=remaining_students1.index(current_prefs[1]) + 1 if current_prefs and len(current_prefs) > 1 and
                                                                         current_prefs[1] in remaining_students1 else 0
            )

            remaining_students2 = [s for s in remaining_students1 if s != pref2]
            pref3 = st.selectbox(
                "3-й пріоритет:",
                [""] + remaining_students2,
                index=remaining_students2.index(current_prefs[2]) + 1 if current_prefs and len(current_prefs) > 2 and
                                                                         current_prefs[2] in remaining_students2 else 0
            )

            remaining_students3 = [s for s in remaining_students2 if s != pref3]
            pref4 = st.selectbox(
                "4-й пріоритет:",
                [""] + remaining_students3,
                index=remaining_students3.index(current_prefs[3]) + 1 if current_prefs and len(current_prefs) > 3 and
                                                                         current_prefs[3] in remaining_students3 else 0
            )

            if st.button("Зберегти переваги"):
                preferences = [p for p in [pref1, pref2, pref3, pref4] if p]
                if len(preferences) != 4:
                    st.error("Необхідно вибрати 4 різних учнів!")
                else:
                    saved_preferences[student_to_edit] = preferences
                    save_data(students, saved_preferences, saved_history)
                    st.success(f"Переваги для {student_to_edit} збережено!")

        with col2:
            st.subheader("Поточні переваги")

            # Показуємо таблицю вже введених переваг
            prefs_data = []
            for s in students:
                if s in saved_preferences:
                    prefs_data.append([s] + saved_preferences[s])
                else:
                    prefs_data.append([s, "", "", "", ""])

            df = pd.DataFrame(
                prefs_data,
                columns=["Учень", "1-й вибір", "2-й вибір", "3-й вибір", "4-й вибір"]
            )
            st.dataframe(df, use_container_width=True)

            # Повнота даних
            complete = len(saved_preferences) == len(students)
            st.progress(len(saved_preferences) / max(1, len(students)))
            if complete:
                st.success(f"Дані повні! Всі {len(students)} учнів внесли переваги.")
            else:
                st.info(f"Внесено {len(saved_preferences)} з {len(students)} учнів")

# Вкладка для створення розсадки
with tab2:
    st.header("🪑 Створення розсадки")

    # Ініціалізація списку заборонених пар
    if 'forbidden_pairs' not in st.session_state:
        st.session_state['forbidden_pairs'] = []

    if not students:
        st.warning("Спершу додайте список учнів у бічній панелі")
    elif len(saved_preferences) < len(students):
        st.warning("Необхідно ввести переваги для всіх учнів")
    else:
        st.subheader("Генерація нової розсадки")

        col_settings_1, col_settings_2 = st.columns(2)

        with col_settings_1:
            # Налаштування алгоритму
            with st.expander("⚙️ Налаштування алгоритму", expanded=False):
                randomness = st.slider(
                    "Рівень випадковості (0-100%)",
                    min_value=0,
                    max_value=100,
                    value=30
                )
                penalty = st.slider(
                    "Штраф за повторні розсадки (1-10)",
                    min_value=1,
                    max_value=10,
                    value=5
                )

            # Блок заборонених пар
            with st.expander("⛔ Хто НЕ МОЖЕ сидіти разом", expanded=True):
                f_col1, f_col2, f_col3 = st.columns([3, 3, 2.5])
                with f_col1:
                    bad_pair_1 = st.selectbox("Учень 1", students, key="bp_1", label_visibility="collapsed")
                with f_col2:
                    bad_pair_2 = st.selectbox("Учень 2", [s for s in students if s != bad_pair_1], key="bp_2",
                                              label_visibility="collapsed")
                with f_col3:
                    if st.button("⛔ Заборонити", key="btn_forbid", use_container_width=True):
                        exists = False
                        current_pair_set = {bad_pair_1, bad_pair_2}
                        for p1, p2 in st.session_state['forbidden_pairs']:
                            if {p1, p2} == current_pair_set:
                                exists = True
                                break
                        if not exists:
                            st.session_state['forbidden_pairs'].append((bad_pair_1, bad_pair_2))
                            st.rerun()

                if st.session_state['forbidden_pairs']:
                    st.markdown("---")
                    pairs_to_remove = []
                    for idx, (p1, p2) in enumerate(st.session_state['forbidden_pairs']):
                        p_col1, p_col2 = st.columns([0.85, 0.15])
                        with p_col1:
                            st.markdown(f":no_entry_sign: **{p1}** — **{p2}**")
                        with p_col2:
                            if st.button("🗑️", key=f"del_pair_{idx}"):
                                pairs_to_remove.append(idx)
                    if pairs_to_remove:
                        for idx in sorted(pairs_to_remove, reverse=True):
                            st.session_state['forbidden_pairs'].pop(idx)
                        st.rerun()

        with col_settings_2:
            # Налаштування старости
            with st.expander("⭐ Налаштування старости", expanded=True):
                use_starosta = st.checkbox("Призначити місце старости вручну")
                starosta = None
                starosta_neighbor = None
                if use_starosta:
                    starosta = st.selectbox("Хто староста?", students, key="starosta_select")
                    potential_neighbors = [s for s in students if s != starosta]
                    starosta_neighbor = st.selectbox("З ким сидить староста?", potential_neighbors,
                                                     key="starosta_neighbor_select")
                    st.info(f"Пара {starosta} + {starosta_neighbor} буде закріплена.")

        st.markdown("---")

        if st.button("Створити нову розсадку", type="primary", use_container_width=True):
            # Логіка генерації розсадки
            seating = SeatingArrangement(students)
            for student, prefs in saved_preferences.items():
                seating.add_student_preferences(student, prefs)
            for past_arrangement in saved_history:
                seating.update_recent_seatings(past_arrangement)


            def optimize_seating_custom(self):
                available_students = set(self.students)
                arrangement = []

                # 1. Староста
                if use_starosta and starosta and starosta_neighbor:
                    if starosta in available_students and starosta_neighbor in available_students:
                        arrangement.append((starosta, starosta_neighbor))
                        available_students.remove(starosta)
                        available_students.remove(starosta_neighbor)

                # 2. Решта пар
                all_possible_pairs = []
                remaining_list = list(available_students)
                for i, s1 in enumerate(remaining_list):
                    for j, s2 in enumerate(remaining_list[i + 1:], i + 1):
                        # Перевірка заборони
                        if any({s1, s2} == {p1, p2} for p1, p2 in st.session_state['forbidden_pairs']):
                            continue
                        score = self.compute_seating_score((s1, s2))
                        all_possible_pairs.append((s1, s2, score))

                all_possible_pairs.sort(key=lambda x: x[2], reverse=True)
                if all_possible_pairs:
                    count = int(len(all_possible_pairs) * randomness / 100)
                    random.shuffle(all_possible_pairs[:max(1, count)])

                for s1, s2, score in all_possible_pairs:
                    if s1 in available_students and s2 in available_students:
                        arrangement.append((s1, s2))
                        available_students.remove(s1)
                        available_students.remove(s2)

                # Одинаки
                while available_students:
                    s = list(available_students)[0]
                    available_students.remove(s)
                    found = False
                    if available_students:
                        for partner in list(available_students):
                            if not any({s, partner} == {p1, p2} for p1, p2 in st.session_state['forbidden_pairs']):
                                arrangement.append((s, partner))
                                available_students.remove(partner)
                                found = True
                                break
                    if not found:
                        arrangement.append((s,))
                return arrangement


            def compute_score_custom(self, pair):
                s1, s2 = pair
                idx1, idx2 = self.students.index(s1), self.students.index(s2)
                score = self.weight_matrix[idx1, idx2] + self.weight_matrix[idx2, idx1]
                if s2 in self.recent_seatings[s1]:
                    score -= penalty
                return score


            seating.optimize_seating = lambda: optimize_seating_custom(seating)
            seating.compute_seating_score = lambda pair: compute_score_custom(seating, pair)

            new_arrangement = seating.generate_new_arrangement()
            saved_history.append(new_arrangement)
            save_data(students, saved_preferences, saved_history)
            st.success("Нову розсадку створено!")

            # Відображення
            st.subheader("Розсадка на цей тиждень:")

            FIXED_DESKS_COUNT = 15
            display_arrangement = new_arrangement[:]
            while len(display_arrangement) < FIXED_DESKS_COUNT:
                display_arrangement.append(None)

            # Таблиця для перегляду в UI
            ui_data = []
            for i, pair in enumerate(display_arrangement, 1):
                if pair is None:
                    ui_data.append([i, "---", "---"])
                elif len(pair) == 2:
                    ui_data.append([i, pair[0], pair[1]])
                else:
                    ui_data.append([i, pair[0], "---"])
            st.dataframe(pd.DataFrame(ui_data, columns=["Парта", "Учень 1", "Учень 2"]), use_container_width=True)

            # Схема HTML
            st.subheader("Схема розсадки у класі")
            desk_html = "<div style='text-align:center; margin-bottom:20px;'><strong>ВЧИТЕЛЬ</strong></div>"
            desk_html += "<div style='display:flex; justify-content:center;'>"
            desk_html += "<div style='border:2px solid black; padding:10px; width:300px; text-align:center; background:white;'>Дошка</div></div><br>"

            for row in range(5):
                desk_html += "<div style='display:flex; justify-content:center;'>"
                for col in range(3):
                    idx = row * 3 + col
                    pair = display_arrangement[idx] if idx < len(display_arrangement) else None

                    if pair is None:
                        border, bg, style = "#ccc", "#f9f9f9", "dashed"
                        content = "<div style='color:#aaa;'>Вільна</div><hr style='border-top:1px dashed #ccc; margin:5px 0;'><div style='color:#aaa;'>Вільна</div>"
                    else:
                        style = "solid"
                        is_starosta = False
                        if use_starosta and len(pair) == 2:
                            if {pair[0], pair[1]} == {starosta, starosta_neighbor}: is_starosta = True
                        border = "#F4B400" if is_starosta else "#4285F4"
                        bg = "#FFF8E1" if is_starosta else "#E8F0FE"
                        p2_name = pair[1] if len(pair) > 1 else "---"
                        content = f"<div>{pair[0]}</div><hr style='border-color:{border}; margin:5px 0;'><div>{p2_name}</div>"

                    desk_html += f"<div style='border:2px {style} {border}; bg:{bg}; width:150px; height:90px; margin:10px; padding:10px; border-radius:8px; background-color:{bg}; text-align:center; display:flex; flex-direction:column; justify-content:center; color:black;'>{content}</div>"
                desk_html += "</div>"
            st.html(desk_html)

            # --- ЕКСПОРТ В EXCEL (Шаблон) ---
            # Готуємо дані для Excel у форматі шаблону (3 колонки парт, 5 рядів)
            excel_data = []

            # Заголовки (верхні)
            excel_data.append(["", "ДВЕРІ", "", "", "ЦЕНТР", "", "", "ВІКНА", "", ""])
            excel_data.append(["", "1", "", "", "2", "", "", "3", "", ""])

            # Ряди парт (від 5 до 1, тобто від задніх до передніх)
            # Логіка: Індекс 0-2 це Row 1 (передні), 12-14 це Row 5 (задні)
            for r in range(5, 0, -1):  # 5, 4, 3, 2, 1
                row_idx = r - 1
                desk_start_idx = row_idx * 3

                # Отримуємо пари для Лівого, Центрального та Правого ряду
                pairs_in_row = []
                for i in range(3):
                    d_idx = desk_start_idx + i
                    pair = display_arrangement[d_idx] if d_idx < len(display_arrangement) else None

                    s1, s2 = "", ""
                    if pair:
                        s1 = pair[0]
                        if len(pair) > 1:
                            s2 = pair[1]
                    pairs_in_row.append((s1, s2))

                left, center, right = pairs_in_row[0], pairs_in_row[1], pairs_in_row[2]

                # Формуємо рядок: [Row#, L1, L2, Row#, C1, C2, Row#, R1, R2, Row#]
                excel_row = [
                    str(r), left[0], left[1],
                    str(r), center[0], center[1],
                    str(r), right[0], right[1],
                    str(r)
                ]
                excel_data.append(excel_row)

            # Заголовки (нижні, повторюються)
            excel_data.append(["", "1", "", "", "2", "", "", "3", "", ""])
            excel_data.append(["", "ДВЕРІ", "", "", "ЦЕНТР", "", "", "ВІКНА", "", ""])

            # Створюємо DataFrame
            df_excel = pd.DataFrame(excel_data)

            # Зберігаємо в буфер
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_excel.to_excel(writer, index=False, header=False, sheet_name='Rozsadka')

                # Автоматичне налаштування ширини колонок (опціонально)
                worksheet = writer.sheets['Rozsadka']
                worksheet.set_column('A:A', 3)  # Вузькі колонки для номерів
                worksheet.set_column('D:D', 3)
                worksheet.set_column('G:G', 3)
                worksheet.set_column('J:J', 3)
                worksheet.set_column('B:C', 15)  # Ширші для імен
                worksheet.set_column('E:F', 15)
                worksheet.set_column('H:I', 15)

            buffer.seek(0)

            st.download_button(
                label="📥 Завантажити Excel (Шаблон)",
                data=buffer,
                file_name="rozsadka_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
# Вкладка для історії розсадок
with tab3:
    st.header("📜 Історія розсадок")

    if not saved_history:
        st.info("Історія розсадок порожня")
    else:
        for week, arrangement in enumerate(saved_history, 1):
            with st.expander(f"Тиждень {week}"):
                seating_data = []
                for i, pair in enumerate(arrangement, 1):
                    if len(pair) == 2:
                        seating_data.append([i, pair[0], pair[1]])
                    else:
                        seating_data.append([i, pair[0], "---"])

                df = pd.DataFrame(seating_data, columns=["Парта", "Учень 1", "Учень 2"])
                st.dataframe(df, use_container_width=True)

        if st.button("Очистити історію"):
            saved_history = []
            save_data(students, saved_preferences, saved_history)
            st.success("Історію розсадок очищено!")
            st.rerun()
