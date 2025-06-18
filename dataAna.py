import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import pandas as pd
import numpy as np
import os
import re
from difflib import SequenceMatcher

BG = "#f5faff"
ACCENT = "#005691"
FONT = ("Segoe UI", 11)
TITLE_FONT = ("Segoe UI", 16, "bold")

current_df = None
selections = {}

recent_files = []
settings = {
    "use_similarity": False,
    "similarity_threshold": 0.85,
    "separators": [",", ";", "\n"],
    "remove_measurements": False,
    "case_sensitive": False
}

def open_settings():
    win = tk.Toplevel(root)
    win.title("Settings")
    win.configure(bg=BG)

    similarity_var = tk.BooleanVar(value=settings["use_similarity"])
    threshold_var = tk.DoubleVar(value=100 * settings["similarity_threshold"])
    tk.Checkbutton(win, text="Use Similarity Threshold", variable=similarity_var, bg=BG, font=FONT).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 0))
    tk.Label(win, text="Similarity Threshold (%):", font=FONT, bg=BG).grid(row=1, column=0, sticky="w", padx=10)
    threshold_entry = tk.Entry(win, textvariable=threshold_var, width=6)
    threshold_entry.grid(row=1, column=1, sticky="w")

    tk.Label(win, text="Separators:", font=FONT, bg=BG).grid(row=2, column=0, sticky="nw", padx=10, pady=(10, 0))
    sep_frame = tk.Frame(win, bg=BG)
    sep_frame.grid(row=2, column=1, sticky="w", pady=(10, 0))

    sep_vars = [tk.StringVar(value=sep) for sep in settings["separators"]]
    
    def refresh_separators():
        for widget in sep_frame.winfo_children():
            widget.destroy()
        for i, var in enumerate(sep_vars):
            frame = tk.Frame(sep_frame, bg=BG)
            entry = tk.Entry(frame, textvariable=var, width=6)
            entry.pack(side="left")
            remove_btn = tk.Button(frame, text="X", command=lambda i=i: [sep_vars.pop(i), refresh_separators()])
            remove_btn.pack(side="left", padx=2)
            frame.pack(anchor="w", pady=2)
        tk.Button(sep_frame, text="+ Add", command=lambda: [sep_vars.append(tk.StringVar()), refresh_separators()]).pack(pady=5)

    refresh_separators()

    remove_meas_var = tk.BooleanVar(value=settings["remove_measurements"])
    case_sensitive_var = tk.BooleanVar(value=settings["case_sensitive"])

    tk.Checkbutton(win, text="Remove Measurements (e.g., '5 mg')", variable=remove_meas_var, bg=BG, font=FONT).grid(row=3, column=0, columnspan=2, sticky="w", padx=10, pady=10)
    tk.Checkbutton(win, text="Case Sensitive Matching", variable=case_sensitive_var, bg=BG, font=FONT).grid(row=4, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))

    def save_settings():
        try:
            settings["use_similarity"] = similarity_var.get()
            settings["similarity_threshold"] = float(threshold_var.get()) / 100
            settings["separators"] = [v.get() for v in sep_vars if v.get()]
            settings["remove_measurements"] = remove_meas_var.get()
            settings["case_sensitive"] = case_sensitive_var.get()
            win.destroy()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid threshold.")

    ttk.Button(win, text="Save", command=save_settings).grid(row=5, column=0, columnspan=2, pady=15)

def similar_grouping(entries):
    if not settings["use_similarity"]:
        return {item: [item] for item in entries}
    grouped = {}
    for item in entries:
        found = False
        for key in grouped:
            if len(item) >= 6 and len(key) >= 6:
                if SequenceMatcher(None, item, key).ratio() > settings["similarity_threshold"]:
                    grouped[key].append(item)
                    found = True
                    break
        if not found:
            grouped[item] = [item]
    return grouped
def show_tooltip(widget, text):
    tooltip = tk.Toplevel(widget)
    tooltip.wm_overrideredirect(True)
    tooltip.wm_geometry(f"+{widget.winfo_rootx()+20}+{widget.winfo_rooty()+20}")
    label = tk.Label(tooltip, text=text, bg="#ffffe0", relief="solid", borderwidth=1, font=("Segoe UI", 9))
    label.pack()
    def leave(event):
        tooltip.destroy()
    widget.bind("<Leave>", leave, add="+")
    widget.after(2000, tooltip.destroy)
    
def analyze_column(data, colname, include_blanks=False, ignore_blanks=False):
    results = ""
    total = len(data)
    blank_count = data.isna().sum()
    non_blank = data.dropna()
    analysis_targets = []

    if include_blanks:
        analysis_targets.append((data, "*Including blanks*"))
    if ignore_blanks:
        analysis_targets.append((non_blank, "*Ignoring blanks*"))

    for dset, note in analysis_targets:
        results += f"{colname} ({blank_count}/{total} cells blank) {note}:\n"
        dset_numeric = pd.to_numeric(dset, errors='coerce')
        if dset_numeric.notna().sum() > 0:
            decimals = 0
            for val in dset_numeric.dropna():
                s = str(val)
                if "." in s:
                    decimals = max(decimals, len(s.split(".")[1]))
            mean = dset_numeric.mean()
            std = dset_numeric.std()
            fmt = f"{{:.{decimals}f}}"
            results += f"    {fmt.format(mean)} ± {fmt.format(std)}\n\n"
        elif set(dset.dropna().astype(str).str.lower().unique()) <= {"yes", "no"}:
            yes = dset.str.lower().value_counts().get("yes", 0)
            no = dset.str.lower().value_counts().get("no", 0)
            denom = yes + no
            if denom > 0:
                results += f"    Yes ({yes} / {denom},  {100*yes/denom:.1f}%)\n"
                results += f"    No  ({no} / {denom},  {100*no/denom:.1f}%)\n\n"
        else:
            all_items = []
            sep_pattern = "[" + re.escape("".join(settings["separators"])) + "]+"
            for val in dset:
                if pd.isna(val):
                    all_items.append("*blank Cell*")
                else:
                    entries = re.split(sep_pattern, str(val))
                    for e in entries:
                        e = e.strip().lower()
                        if e:
                            all_items.append(e)
            vc = pd.Series(all_items).value_counts()
            grouped = similar_grouping(vc.index)
            combined = {}
            for key, variants in grouped.items():
                total_count = sum(vc[v] for v in variants)
                label = "///".join(v.capitalize() for v in sorted(variants))
                combined[label] = total_count
            vc = pd.Series(combined).sort_index()
            for val, count in vc.items():
                results += f"    {val} ({count}/{len(dset)}, {100*count/len(dset):.1f}%)\n"
            results += "\n"
    return results

def update_recent_files(path):
    if path in recent_files:
        recent_files.remove(path)
    recent_files.insert(0, path)
    if len(recent_files) > 5:
        recent_files.pop()
    recent_menu.menu.delete(0, "end")
    for f in recent_files:
        recent_menu.menu.add_command(label=os.path.basename(f), command=lambda p=f: load_file_path(p))
def load_file_path(path):
    try:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
            load_sheet_from_df(df)
        else:
            xl = pd.ExcelFile(path)
            if len(xl.sheet_names) == 1:
                load_sheet(xl, xl.sheet_names[0])
            else:
                choose_sheet(xl)
        update_recent_files(path)
    except Exception as e:
        messagebox.showerror("Error", f"Could not load file:\n{e}")

def select_file():
    path = filedialog.askopenfilename(filetypes=[("Excel or CSV files", "*.xlsx *.xls *.csv")])
    if path:
        load_file_path(path)

def choose_sheet(xl):
    popup = tk.Toplevel(root)
    popup.title("Select Sheet")
    popup.configure(bg=BG)
    tk.Label(popup, text="Select a sheet to analyze:", bg=BG, font=FONT).pack(padx=10, pady=(10, 5))
    selected = tk.StringVar(value=xl.sheet_names[0])
    sheet_dropdown = ttk.Combobox(popup, values=xl.sheet_names, textvariable=selected, state="readonly", width=30)
    sheet_dropdown.pack(pady=5)
    def confirm():
        popup.destroy()
        load_sheet(xl, selected.get())
    ttk.Button(popup, text="Load Sheet", command=confirm).pack(pady=(10, 15))

def load_sheet(xl, sheet_name):
    df = xl.parse(sheet_name)
    load_sheet_from_df(df)

def load_sheet_from_df(df):
    global current_df
    df.columns = df.columns.str.strip().str.replace('\n', ' ')
    current_df = df
    build_column_selection(df)

def build_column_selection(df):
    for widget in scrollable_frame.winfo_children():
        widget.destroy()
    selections.clear()
    tk.Label(scrollable_frame, text="Column", bg=BG, font=FONT).grid(row=0, column=0, padx=5, pady=5)
    tk.Label(scrollable_frame, text="Include Blanks", bg=BG, font=FONT).grid(row=0, column=1, padx=5)
    tk.Label(scrollable_frame, text="Ignore Blanks", bg=BG, font=FONT).grid(row=0, column=2, padx=5)
    for idx, col in enumerate(df.columns):
        tk.Label(scrollable_frame, text=col, bg=BG, font=FONT).grid(row=idx+1, column=0, sticky="w", padx=5)
        var_include = tk.BooleanVar()
        var_ignore = tk.BooleanVar()
        selections[col] = (var_include, var_ignore)
        ttk.Checkbutton(scrollable_frame, variable=var_include).grid(row=idx+1, column=1)
        ttk.Checkbutton(scrollable_frame, variable=var_ignore).grid(row=idx+1, column=2)

def run_analysis():
    if current_df is None:
        messagebox.showerror("Error", "No data loaded.")
        return
    output.delete(1.0, tk.END)
    for col, (inc, ign) in selections.items():
        if inc.get() or ign.get():
            try:
                result = analyze_column(current_df[col], col, inc.get(), ign.get())
                output.insert(tk.END, result + "\n")
            except Exception as e:
                output.insert(tk.END, f"Error analyzing {col}: {e}\n\n")

def download_results():
    file = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
    if file:
        try:
            with open(file, "w", encoding="utf-8") as f:
                f.write(output.get(1.0, tk.END))
            messagebox.showinfo("Success", "Results saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save file:\n{e}")

root = tk.Tk()
root.title("Virginia Urology Data Analyzer")
root.configure(bg=BG)
root.geometry("900x600")
root.resizable(True, True)

title_frame = tk.Frame(root, bg=BG)
title_frame.pack(fill="x", pady=(10, 0))
tk.Label(title_frame, text="Virginia Urology Data Analyzer", font=TITLE_FONT, bg=BG, fg=ACCENT).pack(side="left", padx=20)
help_btn = ttk.Button(title_frame, text="?", width=3, command=lambda: messagebox.showinfo("Help", "Load a file, select columns, and click Analyze."))
help_btn.pack(side="right", padx=10)
help_btn.bind("<Enter>", lambda e: show_tooltip(help_btn, "Click for help."))

top_frame = tk.Frame(root, bg=BG)
top_frame.pack(fill="x", padx=20, pady=5)
ttk.Button(top_frame, text="⚙ Settings", command=open_settings).pack(side="left")

recent_menu = ttk.Menubutton(top_frame, text="Recent Files")
recent_menu.menu = tk.Menu(recent_menu, tearoff=0)
recent_menu["menu"] = recent_menu.menu
recent_menu.pack(side="left", padx=10)

file_frame = tk.Frame(root, bg=BG)
file_frame.pack(fill="x", pady=(10, 0))
ttk.Button(file_frame, text="Load Excel/CSV File", command=select_file).pack(side="left", padx=20)

columns_frame = tk.LabelFrame(root, text="Column Selection", bg=BG, font=FONT, fg=ACCENT)
columns_frame.pack(fill="x", padx=20, pady=10)
canvas = tk.Canvas(columns_frame, bg=BG, highlightthickness=0, height=200)
scrollbar = ttk.Scrollbar(columns_frame, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg=BG)
scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

analyze_btn = ttk.Button(root, text="Analyze", command=run_analysis)
analyze_btn.pack(pady=10)

ttk.Button(root, text="Download Results", command=download_results).pack()

output_frame = tk.LabelFrame(root, text="Analysis Output", bg=BG, font=FONT, fg=ACCENT)
output_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
output = ScrolledText(output_frame, font=("Consolas", 11), bg="#fafdff", wrap="word", height=12)
output.pack(fill="both", expand=True, padx=10, pady=10)

root.mainloop()
