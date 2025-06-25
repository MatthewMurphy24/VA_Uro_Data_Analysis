from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from difflib import SequenceMatcher
from scipy.stats import fisher_exact, chi2_contingency
import tkinter as tk
import pandas as pd
import numpy as np
import re
import sys
from collections import defaultdict

BG = "#f5faff"  # Background color for the UI
ACCENT = "#005691"  # Accent color for headings and highlights
FONT = ("Segoe UI", 11)  # Default font for UI elements
TITLE_FONT = ("Segoe UI", 16, "bold")  # Font for main window title

class AnalysisManager:
    """
    Manages in-memory storage of analyses.
    Each analysis is stored as a tuple: (summary, counts, total).
    """
    def __init__(self):
        """Initialize the analysis manager with an empty dictionary."""
        self.analyses = {}  # Dictionary storing analyses: name -> (summary, counts, total)

    def save(self, name, summary, counts, total):
        """
        Save an analysis in memory.

        Args:
            name (str): Name for the analysis.
            summary (list): List of summary lines.
            counts (dict): Dictionary of label counts.
            total (int): Total number of entries.
        """
        self.analyses[name] = (summary, counts, total)

    def names(self):
        """
        Get a list of all saved analysis names.

        Returns:
            list: List of analysis names.
        """
        return list(self.analyses.keys())

    def get(self, name):
        """
        Retrieve an analysis by name.

        Args:
            name (str): Name of the analysis.

        Returns:
            tuple: (summary, counts, total) or None if not found.
        """
        return self.analyses.get(name)

class TrieNode:
    """
    Node for Trie data structure used in mapping.
    """
    def __init__(self):
        """Initialize a Trie node with empty children and no value."""
        self.children = {}  # Dictionary of child TrieNodes for each character
        self.value = None   # Value stored at this node (if any)

class Trie:
    """
    Trie data structure for efficient prefix-based mapping.
    """
    def __init__(self):
        """Initialize the Trie with a root node."""
        self.root = TrieNode()  # Root node of the Trie

    def insert(self, key, value):
        """
        Insert a key-value pair into the Trie.

        Args:
            key (str): The key to insert.
            value (str): The value to associate with the key.
        """
        node = self.root
        for char in key.lower():
            node = node.children.setdefault(char, TrieNode())
        node.value = value

    def search(self, text):
        """
        Search for the longest prefix match in the Trie.

        Args:
            text (str): The text to search.

        Returns:
            tuple: (value, end_index) if found, else (None, None).
        """
        node = self.root
        last_value, last_index = None, 0
        for i, char in enumerate(text.lower()):
            if char not in node.children:
                break
            node = node.children[char]
            if node.value is not None:
                last_value, last_index = node.value, i+1
        return (last_value, last_index) if last_value else (None, None)

class DataAnalyzerApp:
    """
    Main application class for the Virginia Urology Data Analyzer.
    Handles UI, settings, mapping, analysis, saving, and comparison.
    """
    def __init__(self, root):
        """
        Initialize the application and build the UI.

        Args:
            root (tk.Tk): The root Tkinter window.
        """
        self.root = root  # The main Tkinter window
        self.root.title("Virginia Urology Data Analyzer")
        self.root.configure(bg=BG)
        self.current_df = None  # The currently loaded pandas DataFrame
        self.selections = {}  # Dict: column name -> (include_var, ignore_var) for checkboxes
        self.mapping_trie = Trie()  # Trie for mapping label replacements
        self.mapping_loaded = False  # Whether a mapping has been loaded
        self.settings = {  # Dictionary of user settings
            "use_similarity": False,  # Whether to use label similarity grouping
            "similarity_threshold": 0.85,  # Similarity threshold for grouping
            "separators": [",", ";", "\n"],  # List of separators for splitting entries
            "remove_measurements": False,  # Whether to remove measurements from labels
            "case_sensitive": False  # Whether label matching is case sensitive
        }
        self.analysis_manager = AnalysisManager()  # Stores all saved analyses
        self.last_summary = []  # Last analysis summary (list of lines)
        self.last_counts = {}  # Last analysis label counts (dict)
        self.last_total = 0  # Last analysis total count (int)
        # UI elements:
        self.button_frame = None  # Frame for main action buttons
        self.analyze_btn = None  # Analyze button widget
        self.save_btn = None  # Save Analysis button widget
        self.columns_frame = None  # Frame for column selection checkboxes
        self.scrollable_frame = None  # Frame for scrollable column selection
        self.output_frame = None  # Frame for analysis output
        self.output = None  # ScrolledText widget for output display
        self.build_ui()  # Build the UI

    def build_ui(self):
        """
        Build the main user interface for the application.
        """
        tk.Label(self.root, text="Virginia Urology Data Analyzer", font=TITLE_FONT, bg=BG, fg=ACCENT).pack(pady=10)
        top = tk.Frame(self.root, bg=BG); top.pack()
        ttk.Button(top, text="⚙ Settings", command=self.open_settings).pack(side="left", padx=5)
        ttk.Button(top, text="Paste Mapping List", command=self.open_mapping_window).pack(side="left", padx=5)
        ttk.Button(top, text="Load Excel/CSV File", command=self.select_file).pack(side="left", padx=5)
        self.button_frame = tk.Frame(self.root, bg=BG); self.button_frame.pack(pady=5)
        self.analyze_btn = ttk.Button(self.button_frame, text="Analyze", command=self.run_analysis)
        self.analyze_btn.pack(side="left", padx=5)
        self.save_btn = ttk.Button(self.button_frame, text="Save Analysis", command=self.save_analysis, state="disabled")
        self.save_btn.pack(side="left", padx=5)
        ttk.Button(self.button_frame, text="Download Results", command=self.download_results).pack(side="left", padx=5)
        ttk.Button(self.button_frame, text="Compare Results", command=self.compare_analyses_window).pack(side="left", padx=5)
        self.columns_frame = tk.LabelFrame(self.root, text="Column Selection", bg=BG, font=FONT, fg=ACCENT)
        self.columns_frame.pack(fill="x", padx=20, pady=10)
        self.scrollable_frame = tk.Frame(self.columns_frame, bg=BG)
        self.scrollable_frame.pack(fill="x")
        self.output_frame = tk.LabelFrame(self.root, text="Analysis Output", bg=BG, font=FONT, fg=ACCENT)
        self.output_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self.output = ScrolledText(self.output_frame, font=("Consolas", 11), bg="#fafdff", wrap="word", height=12)
        self.output.pack(fill="both", expand=True, padx=10, pady=10)

    def select_file(self):
        """
        Open a file dialog to select and load an Excel or CSV file.
        """
        path = filedialog.askopenfilename(filetypes=[("Excel or CSV files", "*.xlsx *.xls *.csv")])
        if path:
            try:
                if path.endswith(".csv"):
                    df = pd.read_csv(path)
                else:
                    df = pd.read_excel(path)
                self.current_df = df
                self.build_column_selection(df)
            except Exception as e:
                messagebox.showerror("Error", f"Could not load file:\n{e}")

    def build_column_selection(self, df):
        """
        Build the UI for column selection based on the loaded DataFrame.

        Args:
            df (pd.DataFrame): The loaded DataFrame.
        """
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.selections.clear()
        for idx, col in enumerate(df.columns):
            tk.Label(self.scrollable_frame, text=col, bg=BG, font=FONT).grid(row=idx, column=0, sticky="w", padx=5)
            var_include = tk.BooleanVar()
            var_ignore = tk.BooleanVar()
            self.selections[col] = (var_include, var_ignore)
            ttk.Checkbutton(self.scrollable_frame, variable=var_include).grid(row=idx, column=1)
            ttk.Checkbutton(self.scrollable_frame, variable=var_ignore).grid(row=idx, column=2)

    def open_settings(self):
        """
        Open the settings window for adjusting analysis options.
        """
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.configure(bg=BG)
        similarity_var = tk.BooleanVar(value=self.settings["use_similarity"])
        threshold_var = tk.DoubleVar(value=100 * self.settings["similarity_threshold"])
        tk.Checkbutton(win, text="Use Similarity Threshold", variable=similarity_var, bg=BG, font=FONT).pack(anchor="w", padx=10, pady=5)
        tk.Label(win, text="Similarity Threshold (%):", font=FONT, bg=BG).pack(anchor="w", padx=10)
        threshold_entry = tk.Entry(win, textvariable=threshold_var, width=6)
        threshold_entry.pack(anchor="w", padx=10)
        tk.Label(win, text="Separators (comma separated):", font=FONT, bg=BG).pack(anchor="w", padx=10, pady=(10,0))
        sep_var = tk.StringVar(value=",".join(self.settings["separators"]))
        tk.Entry(win, textvariable=sep_var, width=30).pack(anchor="w", padx=10)
        remove_meas_var = tk.BooleanVar(value=self.settings["remove_measurements"])
        case_sensitive_var = tk.BooleanVar(value=self.settings["case_sensitive"])
        tk.Checkbutton(win, text="Remove Measurements", variable=remove_meas_var, bg=BG, font=FONT).pack(anchor="w", padx=10, pady=5)
        tk.Checkbutton(win, text="Case Sensitive Matching", variable=case_sensitive_var, bg=BG, font=FONT).pack(anchor="w", padx=10, pady=5)
        def save_settings():
            self.settings["use_similarity"] = similarity_var.get()
            self.settings["similarity_threshold"] = float(threshold_var.get()) / 100
            self.settings["separators"] = [s.strip() for s in sep_var.get().split(",") if s.strip()]
            self.settings["remove_measurements"] = remove_meas_var.get()
            self.settings["case_sensitive"] = case_sensitive_var.get()
            win.destroy()
        ttk.Button(win, text="Save", command=save_settings).pack(pady=10)

    def open_mapping_window(self):
        """
        Open a window to paste and process a mapping list for label replacement.
        """
        win = tk.Toplevel(self.root)
        win.title("Paste Mapping List")
        win.configure(bg=BG)
        tk.Label(win, text="Paste mapping list (left = right, use /// for synonyms):", font=FONT, bg=BG).pack(padx=10, pady=(10, 0))
        text = ScrolledText(win, width=80, height=20, font=("Consolas", 10))
        text.pack(padx=10, pady=10)
        def process():
            self.mapping_trie = Trie()
            mapping_text = text.get("1.0", tk.END)
            for line in mapping_text.splitlines():
                line = line.strip()
                if not line or "=" not in line: continue
                left, right = line.split("=", 1)
                right = right.strip()
                for synonym in left.split("///"):
                    self.mapping_trie.insert(synonym.strip(), right)
            self.mapping_loaded = True
            messagebox.showinfo("Mapping Loaded", "Mapping list processed and loaded.")
            win.destroy()
        ttk.Button(win, text="Process Mapping", command=process).pack(pady=(0, 10))

    def remove_measurements(self, val):
        """
        Remove measurement units (e.g., '5mg', '10 ml') from a string.

        Args:
            val (str): The string to process.

        Returns:
            str: The string with measurements removed.
        """
        return re.sub(r'\s*\b\d+(\.\d+)?\s*(mg|mcg|g|ml|units|iu|tabs?|tablets?|caps?|drops?|pills?|mcL|mL|L|cc|times|x|per|every|day|qd|bid|tid|q\d+h|q\d+d|qhs|prn|twice|once|weekly|daily|hour|hrs?|minutes?|mins?)\b', '', val, flags=re.IGNORECASE).strip()

    def replace_with_mapping(self, val):
        """
        Replace a label using the loaded mapping Trie.

        Args:
            val (str): The label to replace.

        Returns:
            str: The mapped label, or the original if no mapping found.
        """
        if not self.mapping_loaded or not isinstance(val, str) or not val.strip():
            return val
        mapped, idx = self.mapping_trie.search(val.strip())
        if mapped:
            if idx == len(val.strip()):
                return mapped
            return mapped + val.strip()[idx:]
        return val

    def is_single_char_or_symbol(self, label):
        """
        Check if a label is a single character or symbol.

        Args:
            label (str): The label to check.

        Returns:
            bool: True if single char/symbol, False otherwise.
        """
        stripped = label.strip()
        return (len(stripped) == 1 and not stripped.isalnum()) or (len(stripped) == 1)

    def ask_user_group_action(self, labels):
        """
        Ask the user how to group similar labels.

        Args:
            labels (list): List of similar labels.

        Returns:
            str: The chosen label or "separate".
        """
        win = tk.Toplevel(self.root)
        win.title("Group Similar Labels")
        win.grab_set()
        tk.Label(win, text="These labels are similar:\n\n" + "\n".join(f"- {l}" for l in labels) + "\n\nWhich label should be used for all, or keep them separate?", font=FONT, bg=BG).pack(padx=20, pady=10)
        choice = tk.StringVar(value="separate")
        btn_frame = tk.Frame(win, bg=BG)
        btn_frame.pack(pady=5)
        for l in labels:
            ttk.Button(btn_frame, text=l, command=lambda val=l: [choice.set(val), win.destroy()]).pack(fill="x", pady=2)
        ttk.Button(win, text="Keep separate", command=lambda: [choice.set("separate"), win.destroy()]).pack(pady=5)
        win.wait_window()
        return choice.get()

    def similar_grouping(self, entries):
        """
        Group similar labels based on the similarity threshold.

        Args:
            entries (list): List of label strings.

        Returns:
            dict: Mapping of chosen label to list of grouped labels.
        """
        if not self.settings["use_similarity"]:
            return {item: [item] for item in entries if not self.is_single_char_or_symbol(item)}
        grouped, used = {}, set()
        entries = [e for e in entries if not self.is_single_char_or_symbol(e)]
        n = len(entries)
        i = 0
        while i < n:
            item = entries[i]
            if item in used:
                i += 1
                continue
            group = [item]
            for j in range(i + 1, n):
                other = entries[j]
                if other in used: continue
                if len(item) >= 3 and len(other) >= 3:
                    if SequenceMatcher(None, item, other).ratio() > self.settings["similarity_threshold"]:
                        group.append(other)
            if len(group) > 1:
                action = self.ask_user_group_action(group)
                if action == "separate":
                    for g in group:
                        grouped[g] = [g]
                        used.add(g)
                else:
                    grouped[action] = group
                    used.update(group)
            else:
                grouped[item] = [item]
                used.add(item)
            i += 1
        return grouped

    def analyze_column(self, data, colname, include_blanks=False, ignore_blanks=False):
        """
        Analyze a column of data and return a formatted summary.

        Args:
            data (pd.Series): The column data.
            colname (str): The column name.
            include_blanks (bool): Whether to include blanks in analysis.
            ignore_blanks (bool): Whether to ignore blanks in analysis.

        Returns:
            str: The formatted analysis summary.
        """
        results = ""
        total = len(data)
        blank_count = data.isna().sum()
        non_blank = data.dropna()
        analysis_targets = []
        if include_blanks: analysis_targets.append((data, "*Including blanks*"))
        if ignore_blanks: analysis_targets.append((non_blank, "*Ignoring blanks*"))
        for dset, note in analysis_targets:
            results += f"{colname} ({blank_count}/{total} cells blank) {note}:\n"
            dset_numeric = pd.to_numeric(dset, errors='coerce')
            if dset_numeric.notna().sum() > 0:
                decimals = max((len(str(val).split(".")[1]) if "." in str(val) else 0) for val in dset_numeric.dropna())
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
                sep_pattern = "[" + re.escape("".join(self.settings["separators"])) + "]+"
                for val in dset:
                    if pd.isna(val):
                        all_items.append("*blank Cell*")
                    else:
                        entries = re.split(sep_pattern, str(val))
                        for e in entries:
                            e = e.strip()
                            if not self.settings["case_sensitive"]:
                                e = e.lower()
                            if e:
                                if self.settings["remove_measurements"]:
                                    e = self.remove_measurements(e)
                                e = self.replace_with_mapping(e)
                                all_items.append(e)
                vc = pd.Series(all_items).value_counts()
                grouped = self.similar_grouping(list(vc.index))
                combined = {}
                for key, variants in grouped.items():
                    if key == "X": continue
                    total_count = sum(vc[v] for v in variants if v in vc)
                    combined[key.capitalize()] = total_count
                vc = pd.Series(combined).sort_index()
                for val, count in vc.items():
                    results += f"    {val} ({count}/{len(dset)}, {100*count/len(dset):.1f}%)\n"
                results += "\n"
        return results

    def run_analysis(self):
        """
        Run analysis on the selected columns and update the output.
        Also enables the Save Analysis button if results are available.
        """
        self.output.delete(1.0, tk.END)
        self.last_summary, self.last_counts, self.last_total = [], {}, 0
        if self.current_df is None:
            messagebox.showerror("Error", "No data loaded.")
            return
        for col, (inc, ign) in self.selections.items():
            if inc.get() or ign.get():
                try:
                    result = self.analyze_column(self.current_df[col], col, inc.get(), ign.get())
                    self.output.insert(tk.END, result + "\n")
                    if not self.last_summary:
                        lines = result.strip().split("\n")
                        counts, total = {}, 0
                        for line in lines:
                            m = re.match(r"\s*(.*?)\s+\(?(\d+)/(\d+)\)?", line)
                            if m:
                                label = m.group(1).strip()
                                count = int(m.group(2))
                                total = int(m.group(3))
                                counts[label] = count
                        self.last_summary = lines
                        self.last_counts = counts
                        self.last_total = total
                except Exception as e:
                    self.output.insert(tk.END, f"Error analyzing {col}: {e}\n\n")
        if self.last_summary and self.last_counts and self.last_total:
            self.save_btn["state"] = "normal"
        else:
            self.save_btn["state"] = "disabled"

    def save_analysis(self):
        """
        Open a dialog to save the last analysis in memory under a user-specified name.
        """
        win = tk.Toplevel(self.root)
        win.title("Save Analysis")
        win.configure(bg=BG)
        tk.Label(win, text="Enter a name for this analysis:", font=FONT, bg=BG).pack(padx=10, pady=10)
        name_var = tk.StringVar()
        entry = ttk.Entry(win, textvariable=name_var, width=40)
        entry.pack(padx=10, pady=5)
        def save():
            name = name_var.get().strip()
            if not name:
                messagebox.showerror("Error", "Please enter a name.")
                return
            self.analysis_manager.save(name, self.last_summary, self.last_counts, self.last_total)
            messagebox.showinfo("Saved", f"Analysis '{name}' saved in program.")
            win.destroy()
        ttk.Button(win, text="Save", command=save).pack(pady=10)

    def download_results(self):
        """
        Download the current analysis output as a text file.
        """
        file = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if file:
            try:
                with open(file, "w", encoding="utf-8") as f:
                    f.write(self.output.get(1.0, tk.END))
                messagebox.showinfo("Success", "Results saved.")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file:\n{e}")

    def compare_analyses_window(self):
        """
        This feature has been removed.
        """
        messagebox.showinfo("Compare Results", "The compare feature has been removed from this version.")

def parse_file(filename):
    with open(filename, encoding='utf-8') as f:
        content = f.read()

    tests = {}
    # Split by test headers
    test_blocks = re.split(r'([A-Za-z ]+ test test result \(\d+/\d+ cells blank\) \*Including blanks\*:)', content)
    for i in range(1, len(test_blocks), 2):
        header = test_blocks[i].strip()
        data = test_blocks[i+1]
        test_name = header.split(' test test result')[0]
        lines = [line.strip() for line in data.strip().split('\n') if line.strip()]
        test_data = {}
        for line in lines:
            # Category with count and percent
            m = re.match(r'(.+?) \((\d+)/(\d+), ([\d.]+)%\)', line)
            if m:
                cat = m.group(1)
                val = f"{m.group(2)}/{m.group(3)}, {m.group(4)}%"
                test_data[cat] = val
            else:
                # Mean ± stddev
                m2 = re.match(r'([\d.]+) ± ([\d.]+)', line)
                if m2:
                    test_data['mean±std'] = f"{m2.group(1)} ± {m2.group(2)}"
        tests[test_name] = test_data
    return tests

def compare_tests(tests1, tests2):
    all_test_names = set(tests1) & set(tests2)
    for test in all_test_names:
        print(f"-----{test} test result-----")
        cats = set(tests1[test]) | set(tests2[test])
        # Remove 'mean±std' if not in both
        if 'mean±std' in cats and ('mean±std' not in tests1[test] or 'mean±std' not in tests2[test]):
            cats.remove('mean±std')
        for cat in sorted(cats):
            val1 = tests1[test].get(cat, "0/0, 0.0%")
            val2 = tests2[test].get(cat, "0/0, 0.0%")
            print(f"{cat:12} {val1:18} {val2:18}  1.000")
        print()

if __name__ == "__main__":
    """
    Entry point for the application.
    """
    root = tk.Tk()
    app = DataAnalyzerApp(root)
    root.mainloop()
