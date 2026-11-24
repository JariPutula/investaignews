"""Standalone entry point, also discovered by the original portfolio app."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from learning.microsoft_exercise_page import show_exercise

st.set_page_config(page_title="Microsoft: hinta ja odotukset", page_icon=":material/school:", layout="wide")
show_exercise()
