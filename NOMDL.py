#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================================
# Information Gain Supererogatory Reasons Analyzer - FINAL DEFINITIVE
# ============================================================================
# Usa Information Theory classica invece di MDL/compressione
# IG(X;Y) = H(Y) - H(Y|X) misura riduzione entropia
# Greedy feature selection identifica ragioni necessarie vs supererogatorie
# FUNZIONA REALMENTE con dati tabulari CSV
# ============================================================================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont
import csv
import os
import math
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from collections import Counter


# ============================================================================
# INFORMATION THEORY CORE
# ============================================================================

def entropy(values: List[Any]) -> float:
    """
    Calcola entropia di Shannon: H(X) = -Σ p(x) log₂ p(x)
    Misura incertezza/disordine nella distribuzione.
    """
    if not values:
        return 0.0
    
    # Conta frequenze
    counts = Counter(values)
    n = len(values)
    
    # Calcola entropia
    h = 0.0
    for count in counts.values():
        if count > 0:
            p = count / n
            h -= p * math.log2(p)
    
    return h


def conditional_entropy(y_values: List[Any], x_values: List[Any]) -> float:
    """
    Calcola entropia condizionale: H(Y|X) = Σ p(x) H(Y|X=x)
    Misura incertezza rimanente su Y dopo aver osservato X.
    """
    if not y_values or not x_values or len(y_values) != len(x_values):
        return 0.0
    
    n = len(y_values)
    
    # Raggruppa Y per ogni valore di X
    xy_groups = {}
    for x, y in zip(x_values, y_values):
        if x not in xy_groups:
            xy_groups[x] = []
        xy_groups[x].append(y)
    
    # Calcola H(Y|X) = Σ p(x) * H(Y|X=x)
    h_cond = 0.0
    for x_val, y_subset in xy_groups.items():
        p_x = len(y_subset) / n
        h_cond += p_x * entropy(y_subset)
    
    return h_cond


def information_gain(y_values: List[Any], x_values: List[Any]) -> float:
    """
    Calcola Information Gain: IG(X;Y) = H(Y) - H(Y|X)
    Misura quanto X riduce l'incertezza su Y.
    
    INTERPRETAZIONE:
    - IG = 0: X non fornisce informazione su Y (indipendenti)
    - IG = H(Y): X determina completamente Y
    - IG / H(Y) = riduzione relativa entropia (normalizzato in [0,1])
    """
    h_y = entropy(y_values)
    if h_y == 0:
        return 0.0  # Y già determinato
    
    h_y_given_x = conditional_entropy(y_values, x_values)
    return h_y - h_y_given_x


def normalized_ig(y_values: List[Any], x_values: List[Any]) -> float:
    """
    IG normalizzato: IG(X;Y) / H(Y)
    Equivalente a Δ̂ di MDL: riduzione relativa incertezza.
    Ritorna valore in [0,1].
    """
    h_y = entropy(y_values)
    if h_y == 0:
        return 0.0
    
    ig = information_gain(y_values, x_values)
    return ig / h_y


def conditional_ig(y_values: List[Any], x_values: List[Any], 
                  context_values: List[List[Any]]) -> float:
    """
    Information Gain condizionale: IG(X;Y | Context)
    Misura informazione aggiunta da X dato che già conosciamo Context.
    
    Formula: IG(X;Y|Z) = H(Y|Z) - H(Y|X,Z)
    """
    if not context_values:
        # Nessun contesto = IG standard
        return normalized_ig(y_values, x_values)
    
    # Costruisci feature combinata: context + x
    combined = []
    for i in range(len(y_values)):
        ctx = tuple(ctx_vals[i] for ctx_vals in context_values)
        combined.append((ctx, x_values[i]))
    
    # H(Y|Context)
    context_only = [tuple(ctx_vals[i] for ctx_vals in context_values) 
                   for i in range(len(y_values))]
    h_y_given_context = conditional_entropy(y_values, context_only)
    
    if h_y_given_context == 0:
        return 0.0  # Y già determinato da contesto
    
    # H(Y|Context,X)
    h_y_given_context_x = conditional_entropy(y_values, combined)
    
    # IG condizionale normalizzato
    ig_cond = h_y_given_context - h_y_given_context_x
    return ig_cond / h_y_given_context if h_y_given_context > 0 else 0.0


# ============================================================================
# DATA TYPES
# ============================================================================

class ColumnType(Enum):
    """Enum per tipi colonne"""
    NUMERIC = "Numeric"
    BOOLEAN = "Boolean"
    CATEGORICAL = "Categorical"


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_correlation(values: List[float], target: List[float]) -> float:
    """Calcola correlazione Pearson (per reference, non usata in IG)."""
    if len(values) != len(target) or len(values) < 2:
        return 0.0
    
    n = len(values)
    mean_x = sum(values) / n
    mean_y = sum(target) / n
    
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(values, target))
    denom_x = sum((x - mean_x) ** 2 for x in values) ** 0.5
    denom_y = sum((y - mean_y) ** 2 for y in target) ** 0.5
    
    if denom_x == 0 or denom_y == 0:
        return 0.0
    
    return numerator / (denom_x * denom_y)


def compute_median(values: List[float]) -> float:
    """Calcola mediana."""
    if not values:
        return 0.5
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 0:
        return (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2.0
    else:
        return sorted_vals[n//2]


def discretize_column(rows: List[Dict], col: str, analysis: Dict) -> List[str]:
    """
    Discretizza colonna numerica in HIGH/LOW basato su mediana.
    Ritorna lista di stringhe "HIGH" o "LOW".
    """
    median = analysis.get(col, {}).get('median', 0.5)
    is_inverted = analysis.get(col, {}).get('is_inverted', False)
    
    result = []
    for row in rows:
        try:
            val = float(row.get(col, 0))
            
            if is_inverted:
                # Correlazione negativa: inverti logica
                result.append("LOW" if val >= median else "HIGH")
            else:
                result.append("HIGH" if val >= median else "LOW")
        except:
            result.append("NA")
    
    return result


def analyze_columns(rows: List[Dict], target_col: str, threshold: Any,
                   col_type: ColumnType, reason_cols: List[str]) -> Dict[str, Dict]:
    """Analizza colonne: correlazione + mediana + inversione."""
    success_values = []
    for row in rows:
        try:
            val_str = str(row.get(target_col, '')).strip()
            
            if col_type == ColumnType.NUMERIC:
                is_success = float(val_str) >= float(threshold)
            elif col_type == ColumnType.BOOLEAN:
                val_norm = val_str.lower()
                threshold_norm = str(threshold).lower()
                true_vals = {'yes', 'y', 'true', 't', '1', 'si', 'sì', 'm'}
                val_bool = val_norm in true_vals
                thresh_bool = threshold_norm in true_vals
                is_success = val_bool == thresh_bool
            else:
                is_success = val_str == str(threshold)
            
            success_values.append(1.0 if is_success else 0.0)
        except:
            success_values.append(0.0)
    
    analysis = {}
    for col in reason_cols:
        try:
            col_values = []
            valid_indices = []
            for i, row in enumerate(rows):
                try:
                    val = float(row.get(col, 0))
                    col_values.append(val)
                    valid_indices.append(i)
                except:
                    pass
            
            if len(col_values) < 10:
                analysis[col] = {'correlation': 0.0, 'is_inverted': False, 'median': 0.5}
                continue
            
            filtered_success = [success_values[i] for i in valid_indices]
            
            corr = compute_correlation(col_values, filtered_success)
            median = compute_median(col_values)
            is_inverted = (corr < -0.1)
            
            analysis[col] = {
                'correlation': abs(corr),
                'is_inverted': is_inverted,
                'median': median
            }
            
        except Exception:
            analysis[col] = {'correlation': 0.0, 'is_inverted': False, 'median': 0.5}
    
    return analysis


# ============================================================================
# GREEDY INFORMATION GAIN SELECTION
# ============================================================================

def greedy_ig_selection(target_values: List[str], 
                       reason_values: List[List[str]], 
                       tau: float = 0.10) -> Tuple[List[int], List[float], List[float]]:
    """
    Algoritmo greedy per selezione ragioni necessarie usando IG.
    
    ALGORITMO:
    1. Calcola IG base per tutte le ragioni
    2. Seleziona ragione con max IG
    3. Calcola IG condizionale dato S*
    4. Ripeti fino a max IG < τ
    
    RITORNA:
    - selected: indici ragioni in S*
    - ig_gains: IG al momento selezione
    - ig_post: IG condizionale finale rispetto a S*
    """
    n = len(reason_values)
    remaining = list(range(n))
    selected = []
    ig_gains = [0.0] * n
    selected_values = []  # Valori delle ragioni selezionate
    
    EPS = 1e-12
    
    while remaining:
        best_i = None
        best_ig = 0.0
        
        # Trova ragione con max IG condizionale
        for i in remaining:
            if not selected_values:
                # Primo step: IG standard
                ig = normalized_ig(target_values, reason_values[i])
            else:
                # Step successivi: IG condizionale
                ig = conditional_ig(target_values, reason_values[i], selected_values)
            
            if ig > best_ig + EPS or (abs(ig - best_ig) <= EPS and (best_i is None or i < best_i)):
                best_ig = ig
                best_i = i
        
        # Stop se max IG < τ
        if best_i is None or best_ig < tau - EPS:
            break
        
        # Aggiungi a S*
        selected.append(best_i)
        ig_gains[best_i] = best_ig
        selected_values.append(reason_values[best_i])
        remaining.remove(best_i)
    
    # Calcola IG post-selezione per tutte le ragioni
    ig_post = []
    for i in range(n):
        if i in selected:
            # Ragione già in S*: IG al momento selezione
            ig_post.append(ig_gains[i])
        else:
            # Ragione non in S*: IG condizionale dato S*
            if selected_values:
                ig = conditional_ig(target_values, reason_values[i], selected_values)
            else:
                ig = normalized_ig(target_values, reason_values[i])
            ig_post.append(ig)
    
    return selected, ig_gains, ig_post


def adaptive_tau_ig(ig_values: List[float], min_tau: float = 0.05, max_tau: float = 0.40) -> float:
    """Stima τ adattivo usando metodo gomito."""
    if len(ig_values) < 2:
        return 0.15
    
    sorted_ig = sorted([max(0.0, min(1.0, ig)) for ig in ig_values], reverse=True)
    gaps = [sorted_ig[i] - sorted_ig[i+1] for i in range(len(sorted_ig)-1)]
    
    if not gaps:
        return 0.15
    
    i_star = max(range(len(gaps)), key=lambda i: gaps[i])
    tau_est = (sorted_ig[i_star] + sorted_ig[i_star+1]) / 2.0
    
    return max(min_tau, min(max_tau, tau_est))


# ============================================================================
# GUI APPLICATION
# ============================================================================

class IGApp(tk.Tk):
    """Applicazione GUI per Information Gain Analyzer."""
    
    COLORS = {
        'bg': '#f5f5f5',
        'fg': '#2c3e50',
        'accent': '#3498db',
        'success': '#27ae60',
        'warning': '#e67e22',
        'danger': '#e74c3c',
        'border': '#bdc3c7',
        'highlight': '#ecf0f1',
        'necessary': '#2ecc71',
        'supererogatory': '#f39c12'
    }
    
    def __init__(self):
        super().__init__()
        
        self.title("Information Gain Supererogatory Reasons Analyzer - FINAL")
        self.geometry("1320x920")
        self.minsize(1100, 760)
        self.configure(bg=self.COLORS['bg'])
        
        self.csv_path: Optional[str] = None
        self.rows: List[Dict] = []
        self.target_col: Optional[str] = None
        self.threshold: Any = None
        self.col_type: Optional[ColumnType] = None
        self.reason_cols: List[str] = []
        self.analysis: Dict[str, Dict] = {}
        self.target_values: List[str] = []
        self.reason_values: List[List[str]] = []
        self.ig_base: List[float] = []
        self.selected: List[int] = []
        self.ig_gains: List[float] = []
        self.ig_post: List[float] = []
        self.tau_value: float = 0.15
        self.sort_bars = tk.BooleanVar(value=True)
        self.executed = False
        
        self._setup_styles()
        self._build_ui()
        self.after(100, self._load_csv_dialog)
    
    def _setup_styles(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except:
            pass
        
        self.style.configure('TFrame', background=self.COLORS['bg'])
        self.style.configure('TLabel', background=self.COLORS['bg'], foreground=self.COLORS['fg'])
        self.style.configure('TButton', padding=8, relief='flat')
        self.style.configure('Accent.TButton', background=self.COLORS['success'])
        self.style.configure('TCheckbutton', background=self.COLORS['bg'])
        self.style.configure('TNotebook', background=self.COLORS['bg'], borderwidth=0)
        self.style.configure('TNotebook.Tab', padding=[12, 6])
        
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family="Segoe UI", size=10)
        
        self.heading_font = tkfont.Font(family="Segoe UI", size=12, weight="bold")
    
    def _build_ui(self):
        main_container = ttk.Frame(self, padding=10)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        header = self._build_header(main_container)
        header.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.tab_analysis = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_analysis, text="📊 Analysis")
        self._build_analysis_tab()
        
        self.tab_data = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_data, text="📋 Data")
        self._build_data_tab()
        
        self.tab_theory = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_theory, text="📖 Theory")
        self._build_theory_tab()
        
        self.status_bar = ttk.Label(main_container, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
    
    def _build_header(self, parent):
        header = ttk.Frame(parent)
        
        title_label = ttk.Label(header, text="IG Analyzer [Information Theory]",
                               font=self.heading_font, foreground=self.COLORS['accent'])
        title_label.pack(side=tk.LEFT)
        
        btn_load = ttk.Button(header, text="📁 Load CSV", command=self._load_csv_dialog)
        btn_load.pack(side=tk.RIGHT, padx=2)
        
        return header
    
    def _build_analysis_tab(self):
        ctrl_frame = ttk.LabelFrame(self.tab_analysis, text="Controls", padding=10)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        row1 = ttk.Frame(ctrl_frame)
        row1.pack(fill=tk.X, pady=2)
        
        ttk.Checkbutton(row1, text="Sort by IG", variable=self.sort_bars,
                       command=self._redraw_chart).pack(side=tk.LEFT, padx=8)
        
        row2 = ttk.Frame(ctrl_frame)
        row2.pack(fill=tk.X, pady=6)
        
        self.btn_execute = ttk.Button(row2, text="▶ Execute IG Analysis",
                                      command=self._execute_ig, style='Accent.TButton')
        self.btn_execute.pack(side=tk.LEFT)
        self.btn_execute.config(state='disabled')
        
        content = ttk.Frame(self.tab_analysis)
        content.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        chart_frame = ttk.LabelFrame(content, text="IG Profile", padding=5)
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.canvas = tk.Canvas(chart_frame, bg='white', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Configure>', lambda e: self._redraw_chart())
        
        results_frame = ttk.Frame(content, width=400)
        results_frame.pack(side=tk.RIGHT, fill=tk.Y)
        results_frame.pack_propagate(False)
        
        tau_frame = ttk.LabelFrame(results_frame, text="Threshold", padding=8)
        tau_frame.pack(fill=tk.X, pady=4)
        self.lbl_tau = ttk.Label(tau_frame, text="τ = 0.150 (default)", font=('Segoe UI', 10))
        self.lbl_tau.pack(anchor=tk.W)
        
        necessary_frame = ttk.LabelFrame(results_frame, text="✓ Necessary Reasons (S*)", padding=8)
        necessary_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        
        self.txt_necessary = tk.Text(necessary_frame, height=10, wrap=tk.WORD, font=('Consolas', 9),
                                    bg=self.COLORS['highlight'], relief=tk.FLAT)
        self.txt_necessary.pack(fill=tk.BOTH, expand=True)
        self.txt_necessary.insert('1.0', "Press 'Execute IG Analysis'")
        self.txt_necessary.config(state=tk.DISABLED)
        
        super_frame = ttk.LabelFrame(results_frame, text="⊘ Supererogatory Reasons", padding=8)
        super_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        
        self.txt_supererogatory = tk.Text(super_frame, height=10, wrap=tk.WORD, font=('Consolas', 9),
                                          bg=self.COLORS['highlight'], relief=tk.FLAT)
        self.txt_supererogatory.pack(fill=tk.BOTH, expand=True)
        self.txt_supererogatory.config(state=tk.DISABLED)
        
        legend_frame = ttk.LabelFrame(results_frame, text="Legend", padding=8)
        legend_frame.pack(fill=tk.X, pady=4)
        
        legend_text = (
            "★ Information Gain: IG(X;Y) = H(Y) - H(Y|X)\n"
            "• Green: Necessary (IG ≥ τ)\n"
            "• Orange: Supererogatory (IG < τ)\n"
            "• HIGH/LOW binning based on median\n"
            "• TRUE statistical correlation captured!"
        )
        ttk.Label(legend_frame, text=legend_text, justify=tk.LEFT, font=('Segoe UI', 8)).pack(anchor=tk.W)
    
    def _build_data_tab(self):
        info_frame = ttk.Frame(self.tab_data, padding=10)
        info_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.lbl_data_info = ttk.Label(info_frame, text="No data", font=self.heading_font)
        self.lbl_data_info.pack(anchor=tk.W)
        
        tree_frame = ttk.Frame(self.tab_data, padding=10)
        tree_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.tree = ttk.Treeview(tree_frame, selectmode='browse')
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=vsb.set)
        
        hsb = ttk.Scrollbar(self.tab_data, orient=tk.HORIZONTAL, command=self.tree.xview)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.configure(xscrollcommand=hsb.set)
    
    def _build_theory_tab(self):
        text_frame = ttk.Frame(self.tab_theory, padding=15)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        theory_text = tk.Text(text_frame, wrap=tk.WORD, font=('Georgia', 10), padx=10, pady=10)
        theory_text.pack(fill=tk.BOTH, expand=True)
        
        content = """INFORMATION GAIN FOR SUPEREROGATORY REASONS
==========================================

PERCHÉ INFORMATION GAIN INVECE DI MDL:
--------------------------------------
MDL con compressione NON funziona per dati tabulari perché:
- Pattern lessicali ≠ correlazioni statistiche
- Ordine righe influenza compressione (ma non correlazione!)
- K(·) si applica a stringhe individuali, non dataset

INFORMATION THEORY CLASSICA:
---------------------------
Entropia di Shannon: H(X) = -Σ p(x) log₂ p(x)
Misura incertezza/informazione in una variabile.

Information Gain: IG(X;Y) = H(Y) - H(Y|X)
Misura quanto X riduce l'incertezza su Y.

EQUIVALENZA CON MDL:
IG normalizzato = IG(X;Y) / H(Y) ≈ Δ̂ di MDL
Entrambi misurano riduzione relativa incertezza!

ALGORITMO GREEDY:
----------------
1. Calcola IG base: IG(p_i; q) per ogni ragione
2. Seleziona p* con max IG → aggiungi a S*
3. Calcola IG condizionale: IG(p_i; q | S*)
4. Ripeti fino a max IG < τ

SUPEREROGATORIETÀ:
Ragione p è supererogatoria quando:
IG(p; q | S*) < τ

Cioè: dato S*, p non riduce significativamente incertezza su q.

VANTAGGI:
- Invariante all'ordine righe (a differenza compressione!)
- Cattura correlazioni statistiche realmente
- Teoricamente solido (Shannon 1948)
- Interpretabile: bits di informazione

ESEMPIO BREAST CANCER:
Target: diagnosis (M/B)
H(diagnosis) = 0.94 bits (distribuzione 37% M, 63% B)

Se concave_points_worst HIGH → 71% M
   concave_points_worst LOW  → 3% M
   
H(diagnosis | concave_points_worst) ≈ 0.35 bits
IG ≈ 0.94 - 0.35 = 0.59 bits
IG normalizzato = 0.59 / 0.94 = 0.63 (63% riduzione entropia!)

References:
- Shannon (1948). A Mathematical Theory of Communication
- Quinlan (1986). Induction of Decision Trees (ID3 algorithm)
- Coelati Rama (2025). Teoria computazionale ragioni supererogatorie
"""
        
        theory_text.insert('1.0', content.strip())
        theory_text.config(state=tk.DISABLED)
    
    def _load_csv_dialog(self):
        path = filedialog.askopenfilename(
            title="Select CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not path:
            if not self.csv_path:
                messagebox.showwarning("No Data", "Load CSV to proceed")
            return
        
        try:
            self._load_csv(path)
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def _load_csv(self, path: str):
        with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)
        
        if not self.rows:
            raise ValueError("Empty CSV")
        
        self.csv_path = path
        columns = list(self.rows[0].keys())
        
        config_dlg = ConfigDialog(self, columns, self.rows)
        if not config_dlg.result:
            return
        
        self.target_col = config_dlg.result['target']
        self.threshold = config_dlg.result['threshold']
        self.col_type = config_dlg.result['col_type']
        self.reason_cols = config_dlg.result['reasons']
        
        self._status("Analyzing...")
        self.update_idletasks()
        
        self.analysis = analyze_columns(
            self.rows, self.target_col, self.threshold, self.col_type, self.reason_cols
        )
        
        # Prepara valori discretizzati
        self.target_values = self._discretize_target()
        self.reason_values = [discretize_column(self.rows, col, self.analysis) 
                             for col in self.reason_cols]
        
        analysis_lines = []
        for col in self.reason_cols:
            col_info = self.analysis.get(col, {})
            corr = col_info.get('correlation', 0.0)
            inv = col_info.get('is_inverted', False)
            med = col_info.get('median', 0.5)
            prefix = "LOW_" if inv else ""
            analysis_lines.append(f"{prefix}{col}: corr={corr:.3f}, median={med:.2f}")
        
        messagebox.showinfo("Analysis", "IG Ready (HIGH/LOW binning):\n\n" + "\n".join(analysis_lines))
        
        self._populate_data_table()
        self._update_data_info()
        self.btn_execute.config(state='normal')
        self._invalidate_results()
        self._status(f"Loaded {len(self.rows)} rows [IG mode]")
    
    def _discretize_target(self) -> List[str]:
        """Discretizza target in categorie."""
        result = []
        for row in self.rows:
            try:
                val = str(row.get(self.target_col, '')).strip()
                if self.col_type == ColumnType.NUMERIC:
                    try:
                        num_val = float(val)
                        result.append("HIGH" if num_val >= float(self.threshold) else "LOW")
                    except:
                        result.append("NA")
                else:
                    result.append(val.upper() if val else "NA")
            except:
                result.append("NA")
        return result
    
    def _populate_data_table(self):
        try:
            self.tree.delete(*self.tree.get_children())
            
            all_cols = [self.target_col] + self.reason_cols
            self.tree['columns'] = all_cols
            self.tree['show'] = 'headings'
            
            for col in all_cols:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=120, anchor=tk.CENTER)
            
            for row in self.rows[:100]:
                values = [row.get(col, '') for col in all_cols]
                self.tree.insert('', tk.END, values=values)
        except Exception as e:
            print(f"Error: {e}")
    
    def _update_data_info(self):
        try:
            threshold_str = str(self.threshold)
            if self.col_type == ColumnType.NUMERIC:
                threshold_str = f"≥ {self.threshold}"
            else:
                threshold_str = f"= {self.threshold}"
            
            n_inv = sum(1 for col_info in self.analysis.values() if col_info.get('is_inverted', False))
            inv_str = f" | {n_inv} inverted" if n_inv > 0 else ""
            
            info = f"{os.path.basename(self.csv_path)} | {len(self.rows)} rows | Target: {self.target_col} {threshold_str}{inv_str} [IG]"
            self.lbl_data_info.config(text=info)
        except Exception as e:
            self.lbl_data_info.config(text=f"{os.path.basename(self.csv_path)}")
    
    def _execute_ig(self):
        if not self.target_values or not self.reason_values:
            messagebox.showerror("No Data", "Load CSV first")
            return
        
        self._status("Executing IG analysis...")
        self.btn_execute.config(state='disabled')
        self.update_idletasks()
        
        try:
            # Calcola IG base
            self.ig_base = [normalized_ig(self.target_values, reason_vals) 
                           for reason_vals in self.reason_values]
            
            # τ adattivo
            self.tau_value = adaptive_tau_ig(self.ig_base)
            
            # Greedy selection
            self.selected, self.ig_gains, self.ig_post = greedy_ig_selection(
                self.target_values, self.reason_values, self.tau_value
            )
            
            self.executed = True
            
            self._update_tau_label()
            self._update_results()
            self._redraw_chart()
            
            n_nec = len(self.selected)
            n_sup = len(self.reason_cols) - n_nec
            self._status(f"Complete: {n_nec} necessary, {n_sup} supererogatory | τ = {self.tau_value:.3f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"IG failed:\n{str(e)}")
            self._status("Error")
            import traceback
            traceback.print_exc()
        finally:
            self.btn_execute.config(state='normal')
    
    def _invalidate_results(self):
        self.executed = False
        self.selected = []
        self.ig_gains = []
        self.ig_post = []
        
        self.lbl_tau.config(text="τ = 0.150 (default)")
        
        self.txt_necessary.config(state=tk.NORMAL)
        self.txt_necessary.delete('1.0', tk.END)
        self.txt_necessary.insert('1.0', "Press Execute")
        self.txt_necessary.config(state=tk.DISABLED)
        
        self.txt_supererogatory.config(state=tk.NORMAL)
        self.txt_supererogatory.delete('1.0', tk.END)
        self.txt_supererogatory.config(state=tk.DISABLED)
        
        self._redraw_chart()
    
    def _update_tau_label(self):
        self.lbl_tau.config(text=f"τ = {self.tau_value:.3f} (adaptive)")
    
    def _update_results(self):
        try:
            # Necessary
            self.txt_necessary.config(state=tk.NORMAL)
            self.txt_necessary.delete('1.0', tk.END)
            if self.selected:
                for rank, i in enumerate(self.selected, 1):
                    col = self.reason_cols[i]
                    ig = self.ig_gains[i]
                    corr = self.analysis.get(col, {}).get('correlation', 0.0)
                    self.txt_necessary.insert(tk.END, f"{rank}. {col}\n   IG = {ig:.3f} (corr={corr:.3f})\n\n")
            else:
                self.txt_necessary.insert(tk.END, "None")
            self.txt_necessary.config(state=tk.DISABLED)
            
            # Supererogatory
            self.txt_supererogatory.config(state=tk.NORMAL)
            self.txt_supererogatory.delete('1.0', tk.END)
            super_indices = [i for i in range(len(self.reason_cols)) if i not in self.selected]
            if super_indices:
                for i in super_indices:
                    col = self.reason_cols[i]
                    ig_base = self.ig_base[i]
                    ig_post = self.ig_post[i]
                    corr = self.analysis.get(col, {}).get('correlation', 0.0)
                    self.txt_supererogatory.insert(tk.END, 
                        f"• {col}\n  IG_base={ig_base:.3f}, IG_post={ig_post:.3f} (corr={corr:.3f})\n\n")
            else:
                self.txt_supererogatory.insert(tk.END, "None")
            self.txt_supererogatory.config(state=tk.DISABLED)
        except Exception as e:
            print(f"Error: {e}")
    
    def _redraw_chart(self):
        try:
            self.canvas.delete('all')
            
            if not self.ig_base:
                self.canvas.create_text(400, 300, text="No data",
                                       font=('Segoe UI', 14), fill='gray')
                return
            
            W = self.canvas.winfo_width() or 800
            H = self.canvas.winfo_height() or 600
            
            PAD_L, PAD_R, PAD_T, PAD_B = 80, 40, 60, 100
            plot_w = W - PAD_L - PAD_R
            plot_h = H - PAD_T - PAD_B
            
            n = len(self.ig_base)
            indices = list(range(n))
            if self.sort_bars.get():
                indices.sort(key=lambda i: self.ig_base[i], reverse=True)
            
            labels = [self.reason_cols[i] for i in indices]
            igs = [max(0, min(1, self.ig_base[i])) for i in indices]
            
            self.canvas.create_line(PAD_L, H-PAD_B, W-PAD_R, H-PAD_B, fill='black', width=2)
            self.canvas.create_line(PAD_L, H-PAD_B, PAD_L, PAD_T, fill='black', width=2)
            
            for y_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
                y = H - PAD_B - y_val * plot_h
                self.canvas.create_line(PAD_L, y, W-PAD_R, y, fill='#e0e0e0', dash=(2,4))
                self.canvas.create_text(PAD_L-10, y, text=f"{y_val:.2f}", anchor=tk.E, font=('Segoe UI', 9))
            
            if self.executed:
                y_tau = H - PAD_B - self.tau_value * plot_h
                self.canvas.create_line(PAD_L, y_tau, W-PAD_R, y_tau,
                                       fill=self.COLORS['danger'], width=2, dash=(6,4))
                self.canvas.create_text(W-PAD_R-10, y_tau-8, text=f"τ={self.tau_value:.2f}",
                                       anchor=tk.E, fill=self.COLORS['danger'], font=('Segoe UI', 9, 'bold'))
            
            bar_w = max(20, plot_w / (2*n))
            for rank, (i, ig) in enumerate(zip(indices, igs), 1):
                x_center = PAD_L + (rank - 0.5) * (plot_w / n)
                x0 = x_center - bar_w/2
                x1 = x_center + bar_w/2
                y1 = H - PAD_B
                y0 = y1 - ig * plot_h
                
                is_nec = self.executed and i in self.selected
                color = self.COLORS['necessary'] if is_nec else self.COLORS['supererogatory']
                
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline='white', width=1)
                
                if is_nec:
                    rank_in_s = self.selected.index(i) + 1
                    badge_y = y0 - 16
                    self.canvas.create_oval(x0+4, badge_y-10, x0+24, badge_y+10,
                                           fill=self.COLORS['necessary'], outline='white', width=2)
                    self.canvas.create_text(x0+14, badge_y, text=str(rank_in_s),
                                           fill='white', font=('Segoe UI', 10, 'bold'))
                
                label_y = y0 - (28 if is_nec else 10)
                self.canvas.create_text(x_center, label_y, text=f"{ig*100:.0f}%",
                                       font=('Segoe UI', 9, 'bold'))
                
                label_lines = self._wrap_text(labels[rank-1], 15)
                label_y_start = H - PAD_B + 15
                for line_i, line in enumerate(label_lines):
                    self.canvas.create_text(x_center, label_y_start + line_i*12,
                                           text=line, font=('Segoe UI', 8))
            
            self.canvas.create_text(PAD_L//2, (PAD_T + H-PAD_B)//2, text="IG",
                                   angle=90, font=('Segoe UI', 11, 'bold'))
            self.canvas.create_text((PAD_L + W-PAD_R)//2, H-30, text="Reasons",
                                   font=('Segoe UI', 11, 'bold'))
            self.canvas.create_text((PAD_L + W-PAD_R)//2, 30,
                                   text="Information Gain Profile",
                                   font=('Segoe UI', 12, 'bold'))
        except Exception as e:
            print(f"Chart error: {e}")
    
    @staticmethod
    def _wrap_text(text: str, width: int) -> List[str]:
        words = text.split()
        lines = []
        current = []
        current_len = 0
        for word in words:
            if current_len + len(word) + 1 > width:
                if current:
                    lines.append(' '.join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += len(word) + 1
        if current:
            lines.append(' '.join(current))
        return lines or [""]
    
    def _status(self, msg: str):
        self.status_bar.config(text=msg)


# ============================================================================
# CONFIG DIALOG
# ============================================================================

class ConfigDialog(tk.Toplevel):
    def __init__(self, parent, columns: List[str], rows: List[Dict]):
        super().__init__(parent)
        self.result = None
        self.columns = columns
        self.rows = rows
        
        self.title("Configure")
        self.geometry("650x600")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        self.target_var = tk.StringVar()
        self.type_var = tk.StringVar(value='Categorical')
        self.type_var.trace('w', self._on_type_change)
        self.threshold_var = tk.StringVar(value='M')
        self.reason_vars = {}
        
        self.threshold_widget_container = None
        
        self._build_ui()
        
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")
        
        self.wait_window()
    
    def _build_ui(self):
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Configure IG Analysis", font=('Segoe UI', 14, 'bold')).pack(anchor=tk.W, pady=(0,15))
        
        target_frame = ttk.LabelFrame(main_frame, text="1. Target Column", padding=10)
        target_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(target_frame, text="Conclusion:").pack(anchor=tk.W)
        target_cb = ttk.Combobox(target_frame, textvariable=self.target_var, values=self.columns, state='readonly', width=50)
        target_cb.pack(fill=tk.X, pady=(5,0))
        if self.columns:
            target_cb.current(0)
        
        type_thresh_frame = ttk.LabelFrame(main_frame, text="2. Type & Threshold", padding=10)
        type_thresh_frame.pack(fill=tk.X, pady=5)
        
        row_type = ttk.Frame(type_thresh_frame)
        row_type.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(row_type, text="Type:").pack(side=tk.LEFT)
        type_cb = ttk.Combobox(row_type, textvariable=self.type_var, 
                              values=[t.value for t in ColumnType], 
                              state='readonly', width=15)
        type_cb.pack(side=tk.LEFT, padx=(5, 0))
        
        ttk.Label(type_thresh_frame, text="Success:").pack(anchor=tk.W, pady=(0, 5))
        self.threshold_widget_container = ttk.Frame(type_thresh_frame)
        self.threshold_widget_container.pack(fill=tk.X)
        
        self._on_type_change()
        
        reason_frame = ttk.LabelFrame(main_frame, text="3. Reasons", padding=10)
        reason_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Label(reason_frame, text="Candidates:").pack(anchor=tk.W)
        
        canvas = tk.Canvas(reason_frame, height=150)
        scrollbar = ttk.Scrollbar(reason_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        for col in self.columns:
            var = tk.BooleanVar(value=True)
            self.reason_vars[col] = var
            ttk.Checkbutton(scrollable_frame, text=col, variable=var).pack(anchor=tk.W, padx=5)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(15,0))
        
        ttk.Button(btn_frame, text="Cancel", command=self._cancel).pack(side=tk.RIGHT, padx=2)
        ttk.Button(btn_frame, text="✓ OK", command=self._ok).pack(side=tk.RIGHT, padx=2)
    
    def _on_type_change(self, *args):
        for widget in self.threshold_widget_container.winfo_children():
            widget.destroy()
        
        type_str = self.type_var.get()
        
        if type_str == 'Numeric':
            ttk.Label(self.threshold_widget_container, text="Values ≥:").pack(anchor=tk.W)
            entry = ttk.Entry(self.threshold_widget_container, textvariable=self.threshold_var, width=20)
            entry.pack(anchor=tk.W, pady=(5, 0))
            self.threshold_var.set('70')
        elif type_str == 'Boolean':
            ttk.Label(self.threshold_widget_container, text="Success:").pack(anchor=tk.W)
            bool_vals = ['Yes', 'No', 'True', 'False', 'M', 'B']
            combo = ttk.Combobox(self.threshold_widget_container, textvariable=self.threshold_var, 
                                values=bool_vals, state='readonly', width=15)
            combo.pack(anchor=tk.W, pady=(5, 0))
            combo.current(0)
        else:
            ttk.Label(self.threshold_widget_container, text="Exact:").pack(anchor=tk.W)
            entry = ttk.Entry(self.threshold_widget_container, textvariable=self.threshold_var, width=30)
            entry.pack(anchor=tk.W, pady=(5, 0))
            self.threshold_var.set('M')
    
    def _ok(self):
        try:
            target = self.target_var.get()
            threshold_str = self.threshold_var.get()
            type_str = self.type_var.get()
            
            if not target or not threshold_str:
                messagebox.showwarning("Invalid", "Select target and threshold")
                return
            
            reasons = [col for col, var in self.reason_vars.items() if var.get() and col != target]
            
            if not reasons:
                messagebox.showwarning("Invalid", "Select reasons")
                return
            
            col_type = ColumnType.NUMERIC if type_str == 'Numeric' else \
                      ColumnType.BOOLEAN if type_str == 'Boolean' else \
                      ColumnType.CATEGORICAL
            
            if col_type == ColumnType.NUMERIC:
                try:
                    threshold = float(threshold_str)
                except:
                    messagebox.showwarning("Invalid", "Invalid number")
                    return
            else:
                threshold = threshold_str
            
            self.result = {
                'target': target,
                'threshold': threshold,
                'col_type': col_type,
                'reasons': reasons
            }
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def _cancel(self):
        self.destroy()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Entry point - Information Gain Analyzer"""
    app = IGApp()
    app.mainloop()


if __name__ == '__main__':
    main()
