import json
from datetime import timedelta
import re
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st
from langfuse.openai import OpenAI
from langfuse import Langfuse
from dotenv import load_dotenv
from pathlib import Path
import os
import time 

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

langfuse = Langfuse()
langfuse.auth_check()
client = OpenAI()

# Model i dane
MODEL_FILE = "half_maraton_predictor_model_v2.pkl"
CSV_2023 = "clean_halfmarathon_2023.csv"
CSV_2024 = "clean_halfmarathon_2024.csv"
IMAGE_FILE = "Bieg.png"

model = joblib.load(MODEL_FILE)
df_2023 = pd.read_csv(CSV_2023)
df_2024 = pd.read_csv(CSV_2024)
df = pd.concat([df_2023, df_2024], ignore_index=True)

# Przygotowanie danych
def ensure_sex_encoded(df_local):
    if "Płeć_encoded" not in df_local.columns:
        if "Płeć" in df_local.columns:
            df_local["Płeć_encoded"] = df_local["Płeć"].map({"M": 0, "K": 1})
        elif "plec" in df_local.columns:
            df_local["Płeć_encoded"] = df_local["plec"].map({"M": 0, "K": 1})
        else:
            df_local["Płeć_encoded"] = np.nan
    return df_local

df = ensure_sex_encoded(df)
df["Wiek"] = pd.to_numeric(df["Wiek"], errors="coerce")
df = df.dropna(subset=["Wiek"])
df["halfmarathon_minutes"] = df["total_seconds"] / 60.0

age_bins = [0, 18, 30, 40, 50, 60, 70, 120]
age_labels = ["<18", "18-29", "30-39", "40-49", "50-59", "60-69", ">70"]
df["group"] = pd.cut(df["Wiek"], bins=age_bins, labels=age_labels, right=False)
df["group"] = df["group"].astype(pd.CategoricalDtype(categories=age_labels, ordered=True))
BAR_COLORS = ["#D9F6EE", "#A6F7E7", "#83C8B9", "#4F8E80", "#305E53", "#1B4E61", "#0B2C35"]

# Funkcja analizująca dane użytkownika
def get_user_data(user_input: str):
    prompt = f"""
Użytkownik napisał: "{user_input}".

Wyodrębnij dane i zwróć poprawny JSON:
{{
  "sex": "kobieta" lub "mężczyzna" lub null,
  "age": liczba lat lub null,
  "time_5k": czas w minutach lub null,
  "distance_km": liczba kilometrów lub null
}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Zwróć zawsze poprawny JSON z danymi biegacza."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        raw_output = response.choices[0].message.content.strip()
        raw_output = re.sub(r"```(json)?", "", raw_output).strip()
        m = re.search(r"\{.*\}", raw_output, re.S)
        if m:
            raw_output = m.group(0)
        data = json.loads(raw_output)

        sex = data.get("sex", None)
        if isinstance(sex, str):
            s = sex.lower()
            if s.startswith("k") or "kob" in s:
                sex = "kobieta"
            elif s.startswith("m") or "męż" in s or "mezc" in s:
                sex = "mężczyzna"
            else:
                sex = None
        else:
            sex = None

        age = int(data.get("age")) if data.get("age") else None
        time_5k = float(data.get("time_5k")) if data.get("time_5k") else None
        distance_km = float(data.get("distance_km")) if data.get("distance_km") else None

        return {"sex": sex, "age": age, "time_5k": time_5k, "distance_km": distance_km}

    except Exception as e:
        print("Błąd w get_user_data:", e)
        return {"sex": None, "age": None, "time_5k": None, "distance_km": None}


# Interfejs Streamlit
st.set_page_config(page_title="HALF MARATON PREDICTOR", layout="centered")

st.markdown("<h1 style='text-align:center; font-size:40px; font-weight:900;'>HALF MARATON PREDICTOR</h1>", unsafe_allow_html=True)
st.image(IMAGE_FILE, use_container_width=True)

st.markdown(
    "<h3 style='text-align:center; font-size:26px;'>"
    "<span style='display:inline-block; transform: scaleX(-1);'>🏃‍♀️</span> "
    "Sprawdź ile czasu zajmie Ci ukończenie półmaratonu"
    "</h3>",
    unsafe_allow_html=True,
)

st.write(
    "Podaj swoje dane: płeć, wiek oraz czas, w którym przebiegasz 5 km. "
    "Aplikacja policzy Twój przewidywany czas ukończenia półmaratonu, "
    "porównując Cię z danymi z Półmaratonu Wrocławskiego 2023–2024."
)

st.markdown("<p style='font-size:20px; font-weight:bold; margin-top:28px;'>Tu wpisz dane ⬇️</p>", unsafe_allow_html=True)
st.markdown("<p style='font-size:14px; color:black; margin-bottom:-30px;'>Przykład: Jestem kobietą, mam 35 lat, mój czas biegu na 5 km to 28 minut</p>", unsafe_allow_html=True)
user_input = st.text_input("", key="user_input", value="")

clicked = st.button("🏁 Oblicz mój czas")

# Logika po kliknięciu przycisku
if clicked:
    if not user_input.strip():
        st.error("❌ Wpisz swoje dane przed obliczeniem.")
    else:
        # 🚀 Pasek postępu z ikonkami
        placeholder_title = st.empty()
        placeholder_title.markdown("### ⏳ Postęp analizy:")
        progress_text = st.empty()
        progress_bar = st.progress(0)
        progress_text.markdown("🏁 **Krok 1/3:** Odczytywanie danych użytkownika...")

        with st.spinner("Przetwarzam dane..."):
            time.sleep(0.7)
            progress_bar.progress(33)
            data = get_user_data(user_input)
            time.sleep(0.7)
            progress_text.markdown("🧠 **Krok 2/3:** Analizuję dane i przygotowuję predykcję...")
            progress_bar.progress(66)
            time.sleep(0.7)

        progress_text.markdown("📊 **Krok 3/3:** Gotowe! Generuję wynik 🎉")
        progress_bar.progress(100)
        time.sleep(0.5)

        progress_bar.empty()
        progress_text.empty()
        placeholder_title.empty()

        missing = [k for k, v in data.items() if k != "distance_km" and v in (None, "", 0)]
        mapping = {"sex": "płeć", "age": "wiek", "time_5k": "czas na 5 km"}

        # Wykrycie dystansu innego niż 5 km
        if data.get("distance_km") and data["distance_km"] != 5:
            dist = int(data["distance_km"])
            st.warning(f"⚠️ Hej! Wykryłem, że podałeś czas na dystans {dist} km. Obliczenia dotyczą 5 km — podaj czas na 5 km.")
        elif missing:
            missing_human = [mapping[k] for k in missing]
            if len(missing_human) == 1:
                key = missing[0]
                if key == "sex":
                    msg = "Hej! Nie udało mi się wychwycić Twojej płci 😅 Dopisz ją proszę."
                elif key == "age":
                    msg = "Hej! Nie udało mi się wychwycić Twojego wieku 😅 Dopisz go proszę."
                elif key == "time_5k":
                    msg = "Hej! Nie udało mi się wychwycić Twojego czasu na 5 km 😅 Dopisz go proszę."
                else:
                    msg = f"Hej! Brakuje mi informacji: {mapping.get(key, key)} 😅 Uzupełnij ją proszę."
            else:
                joined = " i ".join(missing_human)
                msg = f"Hej! Brakuje mi kilku informacji: {joined} 🤔 Uzupełnij je, a policzę Twój czas!"
            st.warning(f"⚠️ {msg}")

        else:
            sex_encoded = 0 if data["sex"] == "mężczyzna" else 1
            X_new = np.array([[sex_encoded, data["age"], data["time_5k"]]])
            predicted_minutes = model.predict(X_new)[0]
            total_seconds = int(predicted_minutes * 60)
            time_str = str(timedelta(seconds=total_seconds))

            st.markdown(
                f"<div style='background-color:#83c8b9;padding:28px;border-radius:14px;margin-top:14px;text-align:center;'>"
                f"<div style='font-size:26px;font-weight:700;'>🏁 Twój przewidywany czas to:</div>"
                f"<div style='font-size:40px;font-weight:900;margin-top:8px;'>{time_str}</div>"
                f"<div style='font-size:20px;margin-top:8px;'>Może pobijesz rekord? 🏆</div>"
                "</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div style='font-size:26px; font-weight:700; text-align:center; margin-top:40px;'>"
                "Porównaj swój przewidywany czas do wyników uczestników Półmaratonu Wrocławskiego"
                "</div>",
                unsafe_allow_html=True,
            )

            # Wykres porównawczy
            df_local = ensure_sex_encoded(df)
            fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
            fig.subplots_adjust(wspace=0.35, bottom=0.15, top=0.92)

            for idx, sex_val in enumerate([0, 1]):
                subset = df_local[df_local["Płeć_encoded"] == sex_val]
                avg_by_group = (
                    subset.groupby("group")["halfmarathon_minutes"]
                    .mean()
                    .reindex(age_labels)
                    .fillna(0)
                )

                ax = axes[idx]
                bars = ax.bar(age_labels, avg_by_group.values, color=BAR_COLORS, edgecolor="black")
                ax.set_title("Mężczyźni" if sex_val == 0 else "Kobiety", fontsize=13, weight="bold")
                ax.set_xlabel("Grupa wiekowa")
                if idx == 0:
                    ax.set_ylabel("Średni czas (HH:MM)")

                ticks = np.arange(0, int(np.nanmax(avg_by_group.values)) + 60, 30)
                ax.set_yticks(ticks)
                ax.set_yticklabels([f"{int(t//60):02d}:{int(t%60):02d}" for t in ticks])

                # Etykiety
                for bar, val in zip(bars, avg_by_group.values):
                    if val > 0:
                        label = f"{int(val//60):02d}:{int(val%60):02d}"
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            val + 2,
                            label,
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            fontweight="bold"
                        )

            st.pyplot(fig)

