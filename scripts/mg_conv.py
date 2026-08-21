import os
import glob
import subprocess
import csv
import matplotlib.pyplot as plt
import argparse
import numpy as np
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# Configuration
# ==============================================================================
SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
EXECUTABLE = os.path.join(SCRIPT_FOLDER, "../build/mgconvergence")
MESH_FOLDER = os.path.join(SCRIPT_FOLDER, "../geo/mg_meshes")
OUTPUT_FOLDER = "data/multigrid_results"
PLOT_FOLDER = "data/multigrid_plots"

NUM_JOBS = 16  # Number of parallel jobs
PURGE_OUTPUTS = True

# Default number of refinements for all meshes
DEFAULT_REFINEMENTS = 4

# Penalty Configuration
PENALTY_VALUES = [10.0, 50., 100.]

# Time-stepping (Tau) Configuration
TAU_VALUES = [0., 32., 1000.]

# Multigrid smoothing configuration: (pre_smooth, post_smooth)
SMOOTHING_ITERATIONS = [(1, 1), (0, 1), (0, 2)]

MESH_CONFIG = {
    # "ball.msh": DEFAULT_REFINEMENTS,
    "corner.msh": DEFAULT_REFINEMENTS,
    # "corner_structured.msh": DEFAULT_REFINEMENTS,
    "cube.msh": DEFAULT_REFINEMENTS,
    # "cube_hole.msh": DEFAULT_REFINEMENTS,
    # "cube_two_voids.msh": DEFAULT_REFINEMENTS,
    "cube_void.msh": DEFAULT_REFINEMENTS,
    # "cylinder.msh": DEFAULT_REFINEMENTS,
    "tetra.msh": DEFAULT_REFINEMENTS
}

MESH_CONFIG = {
    mesh_path: MESH_CONFIG[os.path.basename(mesh_path)]
    for mesh_path in sorted(glob.glob(os.path.join(MESH_FOLDER, "**", "*.msh"), recursive=True))
    if os.path.basename(mesh_path) in MESH_CONFIG
}

GMRES_RUNS = 8
NEV = 2
EW_TOL = 1e-2
GMRES_TOL = 1e-6

# ==============================================================================
# Plotting Bounds (to ignore diverging outliers)
# ==============================================================================
MAX_EIG_PLOT = 1.2     # Cap Eigenvalue plots at 1.2 (since >1 means divergence)
MAX_GMRES_PLOT = 100   # Cap GMRES plots at 100 iterations

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run StokesMG penalty and tau sensitivity study.")
    parser.add_argument('--rerun', action='store_true', help="Force re-run of all simulations.")
    parser.add_argument('--plot-only', action='store_true', help="Only plot existing data without running.")
    parser.add_argument(
        '--reuse-data',
        action='store_true',
        help="Reuse existing CSV data and skip running new simulations."
    )
    parser.add_argument(
        '--plot-set',
        choices=['all', 'post-smoothing'],
        default='all',
        help="Choose which plots to generate: all plots or only post-smoothing sweep plots."
    )
    return parser.parse_args()

def purge_old_files():
    """Deletes existing CSVs in the output folder and PDFs in the plots folder."""
    print("Automatically purging old files...")
    
    csv_files = glob.glob(os.path.join(OUTPUT_FOLDER, "*.csv"))
    for f in csv_files:
        try: os.remove(f)
        except OSError as e: print(f"Error deleting {f}: {e}")
            
    pdf_files = glob.glob(os.path.join(PLOT_FOLDER, "*.pdf"))
    for f in pdf_files:
        try: os.remove(f)
        except OSError as e: print(f"Error deleting {f}: {e}")
            
    print(f"Purged {len(csv_files)} CSV files and {len(pdf_files)} PDF plots.\n")

def run_job(cmd, out_file, rerun):
    if not rerun and os.path.exists(out_file):
        print(f"Skipping {os.path.basename(out_file)}, already exists.")
        return True
    
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Job failed: {' '.join(cmd)}\n{e}")
        return False

def read_csv(filepath):
    data = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key not in data: data[key] = []
                data[key].append(float(val) if val != 'NaN' else np.nan)
    for key in data:
        data[key] = np.array(data[key])
    return data

def plot_capped_line(ax, x_data, y_data, max_val, base_label, **kwargs):
    """
    Plots the line, caps at max_val. 
    If capped, appends the true max value to the legend label and overlays an upward triangle.
    """
    y_data = np.array(y_data)
    y_capped = np.minimum(y_data, max_val)
    
    # Check if any point exceeds the cap
    max_true = np.max(y_data)
    if max_true > max_val:
        # Format for legend: 2 decimals for Eigenvalues, integer for GMRES
        val_str = f"{max_true:.2f}" if max_val < 10 else f"{int(max_true)}"
        final_label = f"{base_label} [max: {val_str}]"
    else:
        final_label = base_label
        
    # Plot the main capped line
    line = ax.plot(x_data, y_capped, label=final_label, **kwargs)
    color = line[0].get_color()
    
    # Overlay upward-pointing triangles for the points that were capped
    over_mask = y_data > max_val
    if np.any(over_mask):
        x_over = np.array(x_data)[over_mask]
        y_over = y_capped[over_mask]
        # Draw the triangle (^) marker
        ax.plot(x_over, y_over, marker='^', color=color, linestyle='None', markersize=8)

def run_smoothing_pair(smoothing_pair, args):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(PLOT_FOLDER, exist_ok=True)

    if not args.plot_only and PURGE_OUTPUTS:
        purge_old_files()

    jobs = []

    # 1. Setup Jobs
    for tau in TAU_VALUES:
        for p in PENALTY_VALUES:
            for smooth_pair in (smoothing_pair,):
                pre_smooth, post_smooth = smooth_pair
                for cycle in ['V']:
                    for mesh_path, refs in MESH_CONFIG.items():
                        mesh_name = os.path.basename(mesh_path)
                        
                        p_val = round(p, 2)
                        out_file = os.path.join(
                            OUTPUT_FOLDER,
                            f"out_{mesh_name}_tau_{tau}_p_{p_val}_pre_{pre_smooth}_post_{post_smooth}_{cycle}.csv"
                        )
                        cmd = [
                            EXECUTABLE,
                            "--mesh", mesh_path,
                            "--refinements", str(refs),
                            "--output", out_file,
                            "--tau", str(tau),
                            "--penalty", str(p_val),
                            "--pre_smooth", str(pre_smooth),
                            "--post_smooth", str(post_smooth),
                            "--cycle", cycle,
                            "--nev", str(NEV),
                            "--gmres", str(GMRES_RUNS),
                            "--gmres_tol", str(GMRES_TOL),
                            "--eval_tol", str(EW_TOL)
                        ]
                        jobs.append((cmd, out_file))

    # 2. Execute Jobs
    if not args.plot_only:
        with ThreadPoolExecutor(max_workers=NUM_JOBS) as executor:
            futures = [executor.submit(run_job, cmd, out_file, args.rerun) for cmd, out_file in jobs]
            for future in as_completed(futures): future.result()

    # 3. Read Results
    pre_smooth, post_smooth = smoothing_pair
    all_results = {t: {round(p, 2): {'V': {}, 'W': {}} for p in PENALTY_VALUES} for t in TAU_VALUES}
    
    for tau in TAU_VALUES:
        for p in PENALTY_VALUES:
            p_val = round(p, 2)
            for cycle in ['V', 'W']:
                for mesh_path in MESH_CONFIG.keys():
                    mesh_name = os.path.basename(mesh_path)
                    out_file = os.path.join(
                        OUTPUT_FOLDER,
                        f"out_{mesh_name}_tau_{tau}_p_{p_val}_pre_{pre_smooth}_post_{post_smooth}_{cycle}.csv"
                    )
                    if os.path.exists(out_file):
                        all_results[tau][p_val][cycle][mesh_name] = read_csv(out_file)

    # Calculate extremes
    tau_ext = list(dict.fromkeys([TAU_VALUES[0], TAU_VALUES[-1]]))
    pen_ext = list(dict.fromkeys([PENALTY_VALUES[0], PENALTY_VALUES[-1]]))
    extremes_combos = list(itertools.product(tau_ext, pen_ext))

    if args.plot_set != 'all':
        return

    # 4. Plot Results
    for mesh_path in MESH_CONFIG.keys():
        mesh_name = os.path.basename(mesh_path)
        max_ref = MESH_CONFIG[mesh_path]
        
        for cycle in ['V', 'W']:
            has_data = any(mesh_name in all_results[t][round(p, 2)][cycle] for t in TAU_VALUES for p in PENALTY_VALUES)
            if not has_data: continue

            # ==================================================================
            # PLOT A: 2x2 Summary Plot
            # ==================================================================
            fig, axes = plt.subplots(2, 2, figsize=(15, 11))
            fig.suptitle(
                rf"Mesh: {mesh_name} | Cycle: {cycle} | "
                rf"Pre-smooth: {pre_smooth}, Post-smooth: {post_smooth}",
                fontsize=18
            )

            ax_ref_cvg = axes[0, 0]
            ax_ref_gmres = axes[0, 1]
            ax_pen_sens = axes[1, 0]
            ax_tau_sens = axes[1, 1]
            
            # Twin axes for sensitivity plots
            ax_pen_sens_gmres = ax_pen_sens.twinx()
            ax_tau_sens_gmres = ax_tau_sens.twinx()

            # --- SET STRICT Y-LIMITS + SLIGHT PADDING FOR MARKERS ---
            ax_ref_cvg.set_ylim(bottom=0, top=MAX_EIG_PLOT * 1.05)
            ax_ref_gmres.set_ylim(bottom=0, top=MAX_GMRES_PLOT * 1.05)
            
            ax_pen_sens.set_ylim(bottom=0, top=MAX_EIG_PLOT * 1.05)
            ax_pen_sens_gmres.set_ylim(bottom=0, top=MAX_GMRES_PLOT * 1.05)
            
            ax_tau_sens.set_ylim(bottom=0, top=MAX_EIG_PLOT * 1.05)
            ax_tau_sens_gmres.set_ylim(bottom=0, top=MAX_GMRES_PLOT * 1.05)

            # Draw a faint dotted line across the plot to show exactly where the cap is
            for ax_e in [ax_ref_cvg, ax_pen_sens, ax_tau_sens]:
                ax_e.axhline(MAX_EIG_PLOT, color='gray', linestyle=':', alpha=0.5)
            for ax_g in [ax_ref_gmres, ax_pen_sens_gmres, ax_tau_sens_gmres]:
                ax_g.axhline(MAX_GMRES_PLOT, color='gray', linestyle=':', alpha=0.5)

            colors_top = plt.cm.tab10(np.linspace(0, 1, max(1, len(extremes_combos))))
            colors_bot = plt.cm.Set1(np.linspace(0, 1, max(1, max(len(tau_ext), len(pen_ext)))))

            # --- Top-Left & Top-Right: Refinements vs CVG and GMRES ---
            for idx, (t, p) in enumerate(extremes_combos):
                p_val = round(p, 2)
                refs_plot, cvg_plot, gmres_plot = [], [], []
                if mesh_name in all_results[t][p_val][cycle]:
                    df = all_results[t][p_val][cycle][mesh_name]
                    for ref in range(1, max_ref + 1):
                        mask = df['Refinements'] == ref
                        if np.any(mask):
                            refs_plot.append(ref)
                            target_col = 'AbsEval1' if 'AbsEval1' in df else 'AbsEval0'
                            cvg_plot.append(df[target_col][mask][0])
                            gmres_plot.append(df['AvgGMRES'][mask][0])

                if refs_plot:
                    base_label = rf"$\tau$={t}, $C_w$={p_val}"
                    ls = '-' if idx < len(extremes_combos)/2 else '--'
                    
                    plot_capped_line(ax_ref_cvg, refs_plot, cvg_plot, MAX_EIG_PLOT, base_label,
                                     marker='o', color=colors_top[idx], linestyle=ls)
                    plot_capped_line(ax_ref_gmres, refs_plot, gmres_plot, MAX_GMRES_PLOT, base_label,
                                     marker='s', color=colors_top[idx], linestyle=ls)

            ax_ref_cvg.set_title("Convergence vs Refinements")
            ax_ref_cvg.set_xlabel("Number of Refinements")
            ax_ref_cvg.set_ylabel(f"Max Eigenvalue (capped at {MAX_EIG_PLOT})")
            ax_ref_cvg.set_xticks(range(1, max_ref + 1))
            ax_ref_cvg.grid(True, alpha=0.4)
            if ax_ref_cvg.has_data(): ax_ref_cvg.legend(loc='best', fontsize=8)

            ax_ref_gmres.set_title("GMRES Iterations vs Refinements")
            ax_ref_gmres.set_xlabel("Number of Refinements")
            ax_ref_gmres.set_ylabel(f"Avg GMRES Iterations (capped at {MAX_GMRES_PLOT})")
            ax_ref_gmres.set_xticks(range(1, max_ref + 1))
            ax_ref_gmres.grid(True, alpha=0.4)
            if ax_ref_gmres.has_data(): ax_ref_gmres.legend(loc='best', fontsize=8)

            # --- Bottom-Left: C_w Sensitivity ---
            for idx, t in enumerate(tau_ext):
                p_plot, c_plot, g_plot = [], [], []
                for p in PENALTY_VALUES:
                    p_val = round(p, 2)
                    if mesh_name in all_results[t][p_val][cycle]:
                        df = all_results[t][p_val][cycle][mesh_name]
                        mask = df['Refinements'] == max_ref
                        if np.any(mask):
                            p_plot.append(p_val)
                            target_col = 'AbsEval1' if 'AbsEval1' in df else 'AbsEval0'
                            c_plot.append(df[target_col][mask][0])
                            g_plot.append(df['AvgGMRES'][mask][0])
                if p_plot:
                    plot_capped_line(ax_pen_sens, p_plot, c_plot, MAX_EIG_PLOT, rf"$\tau$={t} (Eig)",
                                     marker='o', linestyle='-', color=colors_bot[idx])
                    plot_capped_line(ax_pen_sens_gmres, p_plot, g_plot, MAX_GMRES_PLOT, rf"$\tau$={t} (GMRES)",
                                     marker='x', linestyle='--', color=colors_bot[idx])

            ax_pen_sens.set_title(rf"$C_w$ Sensitivity (at ref = {max_ref})")
            ax_pen_sens.set_xlabel(rf"$C_w$")
            ax_pen_sens.set_ylabel(f"Max Eigenvalue (capped at {MAX_EIG_PLOT})")
            ax_pen_sens_gmres.set_ylabel(f"Avg GMRES (capped at {MAX_GMRES_PLOT})")
            ax_pen_sens.grid(True, alpha=0.4)
            
            lines_1, labels_1 = ax_pen_sens.get_legend_handles_labels()
            lines_2, labels_2 = ax_pen_sens_gmres.get_legend_handles_labels()
            if lines_1 or lines_2:
                ax_pen_sens.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best', fontsize=8)

            # --- Bottom-Right: Tau Sensitivity ---
            for idx, p in enumerate(pen_ext):
                p_val = round(p, 2)
                t_plot, c_plot, g_plot = [], [], []
                for t in TAU_VALUES:
                    if mesh_name in all_results[t][p_val][cycle]:
                        df = all_results[t][p_val][cycle][mesh_name]
                        mask = df['Refinements'] == max_ref
                        if np.any(mask):
                            t_plot.append(t)
                            target_col = 'AbsEval1' if 'AbsEval1' in df else 'AbsEval0'
                            c_plot.append(df[target_col][mask][0])
                            g_plot.append(df['AvgGMRES'][mask][0])
                if t_plot:
                    t_strs = [str(x) for x in t_plot]
                    plot_capped_line(ax_tau_sens, t_strs, c_plot, MAX_EIG_PLOT, rf"$C_w$={p_val} (Eig)",
                                     marker='o', linestyle='-', color=colors_bot[idx])
                    plot_capped_line(ax_tau_sens_gmres, t_strs, g_plot, MAX_GMRES_PLOT, rf"$C_w$={p_val} (GMRES)",
                                     marker='x', linestyle='--', color=colors_bot[idx])

            ax_tau_sens.set_title(rf"$\tau$ Sensitivity (at ref = {max_ref})")
            ax_tau_sens.set_xlabel(rf"$\tau$")
            ax_tau_sens.set_ylabel(f"Max Eigenvalue (capped at {MAX_EIG_PLOT})")
            ax_tau_sens_gmres.set_ylabel(f"Avg GMRES (capped at {MAX_GMRES_PLOT})")
            ax_tau_sens.grid(True, alpha=0.4)
            
            lines_3, labels_3 = ax_tau_sens.get_legend_handles_labels()
            lines_4, labels_4 = ax_tau_sens_gmres.get_legend_handles_labels()
            if lines_3 or lines_4:
                ax_tau_sens.legend(lines_3 + lines_4, labels_3 + labels_4, loc='best', fontsize=8)

            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            summary_plot_file = os.path.join(
                PLOT_FOLDER,
                f"{mesh_name}_{cycle}_pre_{pre_smooth}_post_{post_smooth}_summary.pdf"
            )
            plt.savefig(summary_plot_file, bbox_inches='tight')
            plt.close(fig)

            # ==================================================================
            # PLOT B: Heatmaps showing dependence on BOTH Tau and Penalty
            # ==================================================================
            fig_hm, axes_hm = plt.subplots(1, 2, figsize=(16, 7))
            fig_hm.suptitle(
                rf"Convergence Heatmaps (Mesh: {mesh_name} | {cycle}-Cycle)" + "\n"
                + f"At Maximum Refinement ({max_ref}) | "
                + f"Pre-smooth: {pre_smooth}, Post-smooth: {post_smooth}",
                fontsize=16
            )

            Z_eig = np.full((len(PENALTY_VALUES), len(TAU_VALUES)), np.nan)
            Z_gmres = np.full((len(PENALTY_VALUES), len(TAU_VALUES)), np.nan)

            for i_p, p in enumerate(PENALTY_VALUES):
                p_val = round(p, 2)
                for i_t, t in enumerate(TAU_VALUES):
                    if mesh_name in all_results[t][p_val][cycle]:
                        df = all_results[t][p_val][cycle][mesh_name]
                        mask = df['Refinements'] == max_ref
                        if np.any(mask):
                            target_col = 'AbsEval1' if 'AbsEval1' in df else 'AbsEval0'
                            Z_eig[i_p, i_t] = df[target_col][mask][0]
                            Z_gmres[i_p, i_t] = df['AvgGMRES'][mask][0]

            def plot_single_heatmap(ax, Z_data, title, val_fmt):
                if np.all(np.isnan(Z_data)):
                    ax.set_visible(False)
                    return

                vmin = np.nanmin(Z_data)
                
                if "Eigenvalue" in title:
                    vmax_raw = min(np.nanmax(Z_data), MAX_EIG_PLOT)
                else:
                    vmax_raw = min(np.nanmax(Z_data), MAX_GMRES_PLOT)
                
                vmax = max(vmax_raw, vmin + 1e-3)

                cmap = 'viridis_r'
                cax = ax.imshow(Z_data, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
                fig_hm.colorbar(cax, ax=ax, label=title + " (capped)")

                ax.set_xticks(np.arange(len(TAU_VALUES)))
                ax.set_xticklabels([str(t) for t in TAU_VALUES])
                ax.set_yticks(np.arange(len(PENALTY_VALUES)))
                ax.set_yticklabels([str(p) for p in PENALTY_VALUES])

                ax.set_xlabel(rf"$\tau$", fontsize=12)
                ax.set_ylabel(rf"$C_w$", fontsize=12)
                ax.set_title(title, fontsize=14)

                for i_p in range(len(PENALTY_VALUES)):
                    for i_t in range(len(TAU_VALUES)):
                        val = Z_data[i_p, i_t]
                        if not np.isnan(val):
                            norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                            text_col = "white" if norm_val > 0.5 else "black"
                            fmt_str = f"{{:{val_fmt}}}"
                            
                            val_str = fmt_str.format(val)
                            if val > vmax_raw:
                                val_str = ">" + fmt_str.format(vmax_raw)

                            ax.text(i_t, i_p, val_str, ha="center", va="center", 
                                    color=text_col, fontweight='bold', fontsize=11)

            plot_single_heatmap(axes_hm[0], Z_eig, 'Max Eigenvalue', '.3f')
            plot_single_heatmap(axes_hm[1], Z_gmres, 'Avg GMRES Iterations', '.1f')

            plt.tight_layout()
            plt.subplots_adjust(top=0.88)
            heatmap_plot_file = os.path.join(
                PLOT_FOLDER,
                f"{mesh_name}_{cycle}_pre_{pre_smooth}_post_{post_smooth}_heatmap.pdf"
            )
            plt.savefig(heatmap_plot_file, bbox_inches='tight')
            plt.close(fig_hm)

    print(
        f"Saved plots for pre={pre_smooth}, post={post_smooth} "
        f"in '{PLOT_FOLDER}'."
    )

def plot_post_smoothing_vs_convergence():
    """Plot post-smoothing steps vs convergence and GMRES for pre=0 configurations."""
    post_smoothing_values = sorted({post for pre, post in SMOOTHING_ITERATIONS if pre == 0})
    if not post_smoothing_values:
        print("No pre=0 smoothing pairs configured; skipping post-smoothing sweep plots.")
        return

    tau_ext = list(dict.fromkeys([TAU_VALUES[0], TAU_VALUES[-1]]))
    pen_ext = list(dict.fromkeys([PENALTY_VALUES[0], PENALTY_VALUES[-1]]))
    extremes_combos = list(itertools.product(tau_ext, pen_ext))

    for mesh_path in MESH_CONFIG.keys():
        mesh_name = os.path.basename(mesh_path)
        max_ref = MESH_CONFIG[mesh_path]

        for cycle in ['V', 'W']:
            fig, (ax_eig, ax_gmres) = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
            has_data = False
            colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(extremes_combos))))

            ax_eig.axhline(MAX_EIG_PLOT, color='gray', linestyle=':', alpha=0.5)
            ax_gmres.axhline(MAX_GMRES_PLOT, color='gray', linestyle=':', alpha=0.5)

            for idx, (tau, penalty) in enumerate(extremes_combos):
                penalty_val = round(penalty, 2)
                post_plot = []
                conv_plot = []
                gmres_plot = []

                for post_smooth in post_smoothing_values:
                    out_file = os.path.join(
                        OUTPUT_FOLDER,
                        f"out_{mesh_name}_tau_{tau}_p_{penalty_val}_pre_0_post_{post_smooth}_{cycle}.csv"
                    )
                    if not os.path.exists(out_file):
                        continue

                    df = read_csv(out_file)
                    mask = df['Refinements'] == max_ref
                    if np.any(mask):
                        post_plot.append(post_smooth)
                        target_col = 'AbsEval1' if 'AbsEval1' in df else 'AbsEval0'
                        conv_plot.append(df[target_col][mask][0])
                        gmres_plot.append(df['AvgGMRES'][mask][0])

                if post_plot:
                    has_data = True
                    sorted_triplets = sorted(zip(post_plot, conv_plot, gmres_plot))
                    post_plot = [item[0] for item in sorted_triplets]
                    conv_plot = [item[1] for item in sorted_triplets]
                    gmres_plot = [item[2] for item in sorted_triplets]
                    label = rf"$\tau$={tau}, $C_w$={penalty_val}"
                    plot_capped_line(
                        ax_eig,
                        post_plot,
                        conv_plot,
                        MAX_EIG_PLOT,
                        f"{label} (Eig)",
                        marker='o',
                        color=colors[idx],
                        linestyle='-'
                    )
                    plot_capped_line(
                        ax_gmres,
                        post_plot,
                        gmres_plot,
                        MAX_GMRES_PLOT,
                        f"{label} (GMRES)",
                        marker='x',
                        color=colors[idx],
                        linestyle='--'
                    )

            if not has_data:
                plt.close(fig)
                continue

            fig.suptitle(
                rf"Post-smoothing Sweep (Mesh: {mesh_name} | {cycle}-Cycle)"
                + "\n"
                + rf"(Pre-smoothing fixed at 0, ref = {max_ref})"
            )

            ax_eig.set_title("Convergence Rate")
            ax_eig.set_xlabel("Post-smoothing steps")
            ax_eig.set_ylabel(f"Max Eigenvalue (capped at {MAX_EIG_PLOT})")
            ax_gmres.set_ylabel(f"Avg GMRES Iterations (capped at {MAX_GMRES_PLOT})")
            ax_gmres.set_title("GMRES Iterations")
            ax_gmres.set_xlabel("Post-smoothing steps")

            ax_eig.set_xticks(post_smoothing_values)
            ax_gmres.set_xticks(post_smoothing_values)
            ax_eig.set_ylim(bottom=0, top=MAX_EIG_PLOT * 1.05)
            ax_gmres.set_ylim(bottom=0, top=MAX_GMRES_PLOT * 1.05)
            ax_eig.grid(True, alpha=0.4)
            ax_gmres.grid(True, alpha=0.4)

            lines_1, labels_1 = ax_eig.get_legend_handles_labels()
            if lines_1:
                ax_eig.legend(lines_1, labels_1, loc='best', fontsize=8)

            lines_2, labels_2 = ax_gmres.get_legend_handles_labels()
            if lines_2:
                ax_gmres.legend(lines_2, labels_2, loc='best', fontsize=8)

            plt.tight_layout()
            plt.subplots_adjust(top=0.82)
            post_smooth_plot_file = os.path.join(
                PLOT_FOLDER,
                f"{mesh_name}_{cycle}_pre_0_post_smoothing_vs_convergence.pdf"
            )
            plt.savefig(post_smooth_plot_file, bbox_inches='tight')
            plt.close(fig)

    print("Saved post-smoothing sweep plots for pre=0 configurations.")

def main():
    args = parse_arguments()
    if args.reuse_data:
        args.plot_only = True

    for smoothing_pair_index, smoothing_pair in enumerate(SMOOTHING_ITERATIONS):
        global PURGE_OUTPUTS
        PURGE_OUTPUTS = smoothing_pair_index == 0
        print(
            f"Running smoothing configuration {smoothing_pair_index + 1}/"
            f"{len(SMOOTHING_ITERATIONS)}: pre={smoothing_pair[0]}, "
            f"post={smoothing_pair[1]}"
        )
        run_smoothing_pair(smoothing_pair, args)

    if args.plot_set in ('all', 'post-smoothing'):
        plot_post_smoothing_vs_convergence()

if __name__ == "__main__":
    main()
