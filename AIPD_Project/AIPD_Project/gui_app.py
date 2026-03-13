import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
ENCODER_PATH = BASE_DIR / "encoder.pkl"

# Load model
try:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    print("✓ Model and encoder loaded")
except Exception as e:
    print("Error loading model:", e)
    sys.exit(1)

def predict():
    try:
        machine_type = type_entry.get().strip().upper()

        if machine_type not in ["L", "M", "H"]:
            messagebox.showerror("Input Error", "Machine Type must be L, M, or H")
            return

        machine_type = encoder.transform([machine_type])[0]

        air_temp = float(air_temp_entry.get())
        process_temp = float(process_temp_entry.get())
        rpm = float(rpm_entry.get())
        torque = float(torque_entry.get())
        tool_wear = float(tool_wear_entry.get())

        data = pd.DataFrame([[
            machine_type,
            air_temp,
            process_temp,
            rpm,
            torque,
            tool_wear
        ]], columns=[
            'Type',
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]'
        ])

        prediction = model.predict(data)

        if prediction[0] == 1:
            result_label.config(
                text="⚠ FAILURE PREDICTED",
                fg="red"
            )
        else:
            result_label.config(
                text="✓ MACHINE NORMAL",
                fg="green"
            )

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers")

# GUI Window
root = tk.Tk()
root.title("AI Predictive Maintenance System")
root.geometry("500x600")
root.configure(bg="#f5f7fa")

title = tk.Label(
    root,
    text="AI Predictive Maintenance",
    font=("Arial", 20, "bold"),
    bg="#2196F3",
    fg="white",
    pady=15
)
title.pack(fill="x")

frame = tk.Frame(root, bg="white", padx=30, pady=30)
frame.pack(pady=30)

def field(label):
    tk.Label(frame, text=label, bg="white").pack(anchor="w")
    entry = tk.Entry(frame)
    entry.pack(fill="x", pady=5)
    return entry

type_entry = field("Machine Type (L/M/H)")
air_temp_entry = field("Air Temperature (K)")
process_temp_entry = field("Process Temperature (K)")
rpm_entry = field("Rotational Speed (rpm)")
torque_entry = field("Torque (Nm)")
tool_wear_entry = field("Tool Wear (min)")

predict_btn = tk.Button(
    root,
    text="Predict Machine Health",
    command=predict,
    bg="#4CAF50",
    fg="white",
    font=("Arial", 12, "bold"),
    padx=20,
    pady=10
)
predict_btn.pack(pady=10)

result_label = tk.Label(
    root,
    text="",
    font=("Arial", 14, "bold"),
    bg="#f5f7fa"
)
result_label.pack(pady=20)

root.mainloop()