import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import random

st.set_page_config(page_title="Lottery Pattern Analyzer", layout="wide")

st.title("🎯 Lottery Pattern Analyzer & Number Generator")

# Upload past draw data
lotto_file = st.file_uploader("Upload UK Lotto CSV (6 columns of numbers)", type="csv")
euromillions_file = st.file_uploader("Upload EuroMillions CSV (5+2 columns)", type="csv")

def load_data(file, expected_cols):
    try:
        df = pd.read_csv(file, header=None)
        if df.shape[1] < expected_cols:
            st.error("Uploaded file does not match expected number of columns.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def frequency_analysis(numbers):
    flat = [num for row in numbers for num in row]
    freq = dict(Counter(flat))
    return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))

def markov_chain(numbers):
    transitions = defaultdict(Counter)
    for i in range(len(numbers) - 1):
        for a in numbers.iloc[i]:
            for b in numbers.iloc[i + 1]:
                transitions[a][b] += 1
    # Normalize
    for a in transitions:
        total = sum(transitions[a].values())
        for b in transitions[a]:
            transitions[a][b] /= total
    return transitions

def generate_weighted_numbers(freq_dict, count):
    numbers, weights = zip(*freq_dict.items())
    total = sum(weights)
    weights = [w / total for w in weights]
    return sorted(random.choices(numbers, weights=weights, k=count))

def generate_from_markov(transitions, start, count):
    sequence = [start]
    current = start
    while len(sequence) < count:
        next_candidates = transitions.get(current)
        if not next_candidates:
            break
        current = random.choices(
            population=list(next_candidates.keys()),
            weights=list(next_candidates.values()),
            k=1
        )[0]
        if current not in sequence:
            sequence.append(current)
    return sorted(sequence)

# Run analysis and generation
if st.button("Generate Sets 🚀"):

    if lotto_file:
        st.subheader("📊 UK Lotto Analysis")
        df_lotto = load_data(lotto_file, 6)
        if df_lotto is not None:
            freq = frequency_analysis(df_lotto.values.tolist())
            transitions = markov_chain(df_lotto)
            most_common = list(freq.keys())[:6]

            st.write("▶ **Hot Numbers (Frequency-based)**:", sorted(most_common))
            weighted_set = generate_weighted_numbers(freq, 6)
            st.write("▶ **Weighted Random Set**:", sorted(weighted_set))

            markov_set = generate_from_markov(transitions, random.choice(list(freq.keys())), 6)
            st.write("▶ **Markov Modeled Set**:", sorted(markov_set))

            # Chart
            st.write("📈 Frequency Distribution:")
            fig, ax = plt.subplots()
            ax.bar(freq.keys(), freq.values())
            ax.set_title("UK Lotto Number Frequency")
            st.pyplot(fig)

    if euromillions_file:
        st.subheader("📊 EuroMillions Analysis")
        df_euro = load_data(euromillions_file, 7)
        if df_euro is not None:
            main_numbers = df_euro.iloc[:, :5]
            stars = df_euro.iloc[:, 5:]

            freq_main = frequency_analysis(main_numbers.values.tolist())
            freq_stars = frequency_analysis(stars.values.tolist())

            transitions_main = markov_chain(main_numbers)
            transitions_stars = markov_chain(stars)

            most_common_main = list(freq_main.keys())[:5]
            most_common_stars = list(freq_stars.keys())[:2]

            st.write("▶ **Hot Numbers (Main)**:", sorted(most_common_main))
            st.write("▶ **Hot Stars**:", sorted(most_common_stars))

            weighted_main = generate_weighted_numbers(freq_main, 5)
            weighted_stars = generate_weighted_numbers(freq_stars, 2)
            st.write("▶ **Weighted Random EuroMillions Set**:", sorted(weighted_main), "+ Stars:", sorted(weighted_stars))

            markov_main = generate_from_markov(transitions_main, random.choice(list(freq_main.keys())), 5)
            markov_stars = generate_from_markov(transitions_stars, random.choice(list(freq_stars.keys())), 2)
            st.write("▶ **Markov Modeled EuroMillions Set**:", sorted(markov_main), "+ Stars:", sorted(markov_stars))

            # Charts
            st.write("📈 EuroMillions Main Number Frequency:")
            fig, ax = plt.subplots()
            ax.bar(freq_main.keys(), freq_main.values())
            ax.set_title("EuroMillions Main Number Frequency")
            st.pyplot(fig)

            st.write("📈 Lucky Star Frequency:")
            fig2, ax2 = plt.subplots()
            ax2.bar(freq_stars.keys(), freq_stars.values(), color='orange')
            ax2.set_title("Lucky Stars Frequency")
            st.pyplot(fig2)

# ----------------------------------------
# Section: Generate predictions
# ----------------------------------------

## 🧪 Sample CSV Format

You will need to upload draw history files in CSV format:

- **UK Lotto**: 6 columns, each row like:
