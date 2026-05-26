#!/usr/bin/env python3
import csv
import os
import sys

TABLE_CONFIGS = [
    {"file": "../results/p1_analyze_linear.csv", "name": "Linear Scan (Baseline CPU)"},
    {"file": "../results/p1_analyze_batch.csv",  "name": "Batching (NumPy CPU)"},
    {"file": "../results/p2_analyze.csv",        "name": "Batching (CuPy GPU)"},
    {"file": "../results/p3_analyze.csv",        "name": "Matrix Multiplication Memory Optimization (NumPy CPU)"},
    {"file": "../results/p4_analyze.csv",        "name": "Matrix Multiplication Memory Optimization  (CuPy GPU)"},
    {"file": "../results/p5_analyze.csv",        "name": "CUVS Baseline Implementation"},
]

def format_cell(val):
    """Clean data elements and formats scientific floating points for LaTeX."""
    val_clean = val.strip()
    
    if val_clean.upper() in ["SKIPPED", "OOM", "OOM_KILLED", "CRASH", ""]:
        return r"\textit{Skipped}"
        
    try:
        if val_clean.isdigit():
            return f"{int(val_clean):,}"
            
        num = float(val_clean)
        
        if 'e' in val_clean.lower() or (0 < abs(num) < 0.001):
            base, exponent = f"{num:.2e}".split('e')
            return f"${base} \\times 10^{{{int(exponent)}}}$"
            
        return f"{num:.4f}" if num % 1 != 0 else f"{int(num):,}"
    except ValueError:
        return val_clean.replace('_', r'\_')

def build_latex_table(file_path, title_caption, file_id):
    """Parses a generic CSV and maps it into standard LaTeX with a vertical separator."""
    if not os.path.exists(file_path):
        return f"% Error: File '{file_path}' was not found. Generation skipped.\n"

    with open(file_path, mode='r') as f:
        reader = csv.reader(f)
        try:
            raw_headers = [h.strip() for h in next(reader)]
        except StopIteration:
            return f"% Error: File '{file_path}' is completely empty.\n"
        
        rows = [[col.strip() for col in row] for row in reader if row]

    display_headers = [h.replace('_', ' ').title() for h in raw_headers]
    num_cols = len(display_headers)
    
    # Custom alignment string: 'l | c c c c' 
    # This places a vertical line directly after the first column (Dataset Name)
    col_alignment = "l | " + " ".join(["c"] * (num_cols - 1))
    
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"  \centering")
    latex.append(f"  \\caption{{Performance Results: {title_caption}}}")
    latex.append(f"  \\label{{tab:{file_id}}}")
    latex.append(r"  \resizebox{\textwidth}{!}{%")
    latex.append(f"    \\begin{{tabular}}{{{col_alignment}}}")
    
    # Header block with a single horizontal line below it
    latex.append("    " + " & ".join(display_headers) + r" \\")
    latex.append(r"    \hline")  # The single horizontal line requested
    
    for row in rows:
        while len(row) < num_cols:
            row.append("")
        formatted_cells = [format_cell(cell) for cell in row]
        latex.append("    " + " & ".join(formatted_cells) + r" \\")
        
    latex.append(r"    \end{tabular}%")
    latex.append(r"  }")
    latex.append(r"\end{table}") 
    return "\n".join(latex)

def main():
    output_filename = "all_latex_tables.tex"
    results_dir = "results"
    
    print(f"Reading target files from directory: '{results_dir}/'")
    
    with open(output_filename, "w") as out_f:
        out_f.write("% ==========================================================\n")
        out_f.write("% AUTOMATICALLY GENERATED LATEX TABLES FOR ASSIGNMENT 7\n")
        out_f.write("% ==========================================================\n\n")
        
        for config in TABLE_CONFIGS:
            csv_path = os.path.join(results_dir, config["file"])
            file_id = config["file"].split(".")[0]
            
            print(f"Processing structural layout for: {csv_path}")
            table_latex = build_latex_table(csv_path, config["name"], file_id)
            
            out_f.write(f"% --- Table for {config['file']} ---\n")
            out_f.write(table_latex)
            out_f.write("\n\n")
            
    print(f"\nSuccess! All LaTeX table snippets written cleanly to: {output_filename}")

if __name__ == "__main__":
    main()
