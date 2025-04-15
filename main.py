import streamlit as st
import numpy as np
import pandas as pd
import random
from collections import defaultdict
import json
import os


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

    if not students:
        st.warning("Спершу додайте список учнів у бічній панелі")
    elif len(saved_preferences) < len(students):
        st.warning("Необхідно ввести переваги для всіх учнів")
    else:
        st.subheader("Генерація нової розсадки")

        # Додаємо опцію налаштування параметрів
        with st.expander("Налаштування алгоритму"):
            randomness = st.slider(
                "Рівень випадковості (0-100%)",
                min_value=0,
                max_value=100,
                value=30,
                help="Чим вище значення, тим більша випадковість при розсадці"
            )

            penalty = st.slider(
                "Штраф за повторні розсадки (1-10)",
                min_value=1,
                max_value=10,
                value=5,
                help="Чим вище значення, тим менша ймовірність, що учні сидітимуть разом повторно"
            )

        if st.button("Створити нову розсадку", type="primary"):
            # Створюємо розсадку
            seating = SeatingArrangement(students)

            # Додаємо переваги
            for student, prefs in saved_preferences.items():
                seating.add_student_preferences(student, prefs)

            # Завантажуємо історію розсадок
            for past_arrangement in saved_history:
                seating.update_recent_seatings(past_arrangement)


            # Застосовуємо налаштування
            # Змінюємо рівень випадковості
            def optimize_seating_custom(self):
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

                # Додаємо випадковість, штраф за повторні розсадки
                shuffle_count = int(len(all_possible_pairs) * randomness / 100)
                random.shuffle(all_possible_pairs[:max(1, shuffle_count)])

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


            # Перевизначаємо метод для обчислення оцінки з кастомним штрафом
            def compute_seating_score_custom(self, pair):
                student1, student2 = pair
                idx1 = self.students.index(student1)
                idx2 = self.students.index(student2)

                # Основна оцінка на основі пріоритетів
                score = self.weight_matrix[idx1, idx2] + self.weight_matrix[idx2, idx1]

                # Штраф, якщо учні вже сиділи разом нещодавно
                if student2 in self.recent_seatings[student1]:
                    score -= penalty

                return score


            # Застосовуємо кастомні методи
            seating.optimize_seating = lambda: optimize_seating_custom(seating)
            seating.compute_seating_score = lambda pair: compute_seating_score_custom(seating, pair)

            # Генеруємо нову розсадку
            new_arrangement = seating.generate_new_arrangement()

            # Зберігаємо в історію
            saved_history.append(new_arrangement)
            save_data(students, saved_preferences, saved_history)

            st.success("Нову розсадку створено!")

            # Відображаємо нову розсадку
            st.subheader("Розсадка на цей тиждень:")

            seating_data = []
            for i, pair in enumerate(new_arrangement, 1):
                if len(pair) == 2:
                    seating_data.append([i, pair[0], pair[1]])
                else:
                    seating_data.append([i, pair[0], "---"])

            df = pd.DataFrame(seating_data, columns=["Парта", "Учень 1", "Учень 2"])
            st.dataframe(df, use_container_width=True)

            # Візуалізація розсадки у вигляді таблиці або схеми класу
            st.subheader("Схема розсадки у класі")

            # Створюємо сітку для візуалізації
            max_rows = 5  # 4 парти в ряд

            # Візуалізуємо схему класу
            desk_html = "<div style='text-align:center; margin-bottom:20px;'><strong>ВЧИТЕЛЬ</strong></div>"
            desk_html += "<div style='display:flex; justify-content:center;'>"
            desk_html += "<div style='border:2px solid black; padding:10px; text-align:center; margin:5px;'>Дошка</div>"
            desk_html += "</div><br>"

            for row in range(max_rows):
                desk_html += "<div style='display:flex; justify-content:center;'>"
                for col in range(3):
                    desk_idx = row * 3 + col
                    if desk_idx < len(new_arrangement):
                        pair = new_arrangement[desk_idx]
                        if len(pair) == 2:
                            desk_html += f"<div style='border:2px solid #4285F4; color: black; padding:10px; width:150px; height:80px; margin:10px; text-align:center; background-color:#E8F0FE; border-radius:5px;'>"
                            desk_html += f"<div>{pair[0]}</div><hr style='margin:5px 0;'><div>{pair[1]}</div>"
                        else:
                            desk_html += f"<div style='border:2px solid #4285F4; padding:10px; width:150px; height:80px; margin:10px; text-align:center; background-color:#E8F0FE; border-radius:5px;'>"
                            desk_html += f"<div>{pair[0]}</div><hr style='margin:5px 0;'><div>---</div>"
                        desk_html += "</div>"
                desk_html += "</div>"

            st.markdown(desk_html, unsafe_allow_html=True)

            # Додаємо можливість експорту
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Завантажити розсадку як CSV",
                csv,
                "rozсadka.csv",
                "text/csv",
                key='download-csv'
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
