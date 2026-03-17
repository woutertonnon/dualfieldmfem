"""Post-processing utilities for convergence and diagnostic data.

This module intentionally mirrors the legacy plotting/collection behavior so
existing benchmark scripts keep working during migration.
"""

from __future__ import annotations

import glob
import os
import re
import stat
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import paramiko
from matplotlib.ticker import MultipleLocator


class SimulationDataProcessor:
    """Collect simulation CSV output and produce standard plots.

    Parameters
    ----------
    name
        Benchmark name used in `data/config`, `out/data`, and plot folders.
    """

    def __init__(self, name: str):
        self.name = name
        self.data = None
        self.error_columns = ["u1_err_L2"]
        self.cons_columns = []

    def collect_data(self):
        """Collect final-row error values for each order/refinement case.

        Returns
        -------
        dict
            Mapping `order -> {"refs": np.ndarray, "errs": {col: np.ndarray}}`.
        """
        pattern = re.compile(self.name + r"_conv_order(\d+)_ref(\d+)_vars\.csv")
        self.data = {}

        for fname in glob.glob("out/data/" + self.name + "/" + self.name + "_conv_order*_ref*_vars.csv"):
            base = os.path.basename(fname)
            m = pattern.match(base)
            if not m:
                continue

            order = int(m.group(1))
            ref = int(m.group(2))

            df = pd.read_csv(fname)
            if len(df) == 0:
                continue

            for error_column in self.error_columns:
                if error_column not in df.columns:
                    raise ValueError(f"'{error_column}' column not found in {fname}")

            row = df.iloc[-1]
            errs = {col: float(row[col]) for col in self.error_columns}

            if order not in self.data:
                self.data[order] = []
            self.data[order].append((ref, errs))

        for order in list(self.data.keys()):
            if not self.data[order]:
                continue
            self.data[order].sort(key=lambda x: x[0])
            refs = np.array([r for r, _ in self.data[order]], dtype=float)
            temp = {"refs": refs, "errs": {}}
            for error_column in self.error_columns:
                temp["errs"][error_column] = np.array([e[error_column] for _, e in self.data[order]], dtype=float)
            self.data[order] = temp

        return self.data

    def add_reference_triangles(self, ax, order, x_anchor, y_anchor):
        """Draw an `O(h^order)` reference segment on a semilogy axis."""
        if x_anchor is None or y_anchor is None or y_anchor <= 0:
            return
        x0 = x_anchor - 1.0
        x1 = x_anchor
        y1 = y_anchor
        y0 = y_anchor * (2.0 ** order)
        ax.semilogy([x0, x1], [y0, y1], ":k")
        ax.text(x0 + 0.05, np.sqrt(y0 * y1), f"O(h{order})", va="center", ha="left")

    def plot_convergence(self, show_plot=False, reference_order=lambda order: order):
        """Create and save convergence plots for configured error columns.

        Parameters
        ----------
        show_plot
            If ``True``, display figures interactively.
        reference_order
            Mapping from polynomial order to expected slope order.
        """
        figs = {}
        axs = {}
        for error_column in self.error_columns:
            figs[error_column], axs[error_column] = plt.subplots()

        markers = {1: "o-", 2: "s-", 3: "^-", 4: "v-"}

        if not self.data:
            print("Error: data array not initialized. Call collect_data() first")
            sys.exit(1)

        for order, d in sorted(self.data.items()):
            refs = d["refs"]
            errs = d["errs"]
            style = markers.get(order, "o-")
            for error_column in self.error_columns:
                axs[error_column].semilogy(refs, errs[error_column], style, label=f"order {order}", linewidth=1.5, markersize=6)
                self.add_reference_triangles(axs[error_column], reference_order(order), refs[-1], errs[error_column][-1])

        for error_column in self.error_columns:
            axs[error_column].set_xlabel("Refinement level (ref index)")
            axs[error_column].set_ylabel(f"L2 error '{error_column}'")
            axs[error_column].set_title("Convergence vs mesh refinement")
            axs[error_column].grid(True, which="both", linestyle="--", linewidth=0.5)
            axs[error_column].xaxis.set_major_locator(MultipleLocator(1.0))
            axs[error_column].legend(loc="best")

            directory = "./out/plots/" + self.name + "/convergence"
            os.makedirs(directory, exist_ok=True)
            outname = directory + "/" + self.name + "_" + error_column
            figs[error_column].tight_layout()
            figs[error_column].savefig(outname, dpi=300)
            if show_plot:
                plt.show()
            else:
                plt.close(figs[error_column])

    def plot_conserved_variables(self):
        """Plot configured conservation/diagnostic columns versus time."""
        directory = "./out/plots/" + self.name + "/conservation"
        os.makedirs(directory, exist_ok=True)

        pattern = re.compile(self.name + r"_conv_order(\d+)_ref(\d+)_vars\.csv")
        for fname in glob.glob("out/data/" + self.name + "/" + self.name + "_conv_order*_ref*_vars.csv"):
            base = os.path.basename(fname)
            m = pattern.match(base)
            if not m:
                continue

            order = int(m.group(1))
            ref = int(m.group(2))
            df = pd.read_csv(fname)

            if "time_full" not in df.columns:
                raise ValueError(f"'time_full' column not found in {fname}")

            for cons_column in self.cons_columns:
                if cons_column not in df.columns:
                    raise ValueError(f"'{cons_column}' column not found in {fname}")
                fig, ax = plt.subplots()
                ax.plot(np.array(df["time_full"]), np.array(df[cons_column]))
                ax.set_xlabel("time_full [s]")
                ax.set_ylabel(cons_column)
                ax.grid()
                plt.savefig(directory + "/" + self.name + "_" + cons_column + "_order" + str(order) + "_ref" + str(ref) + ".png")
                plt.close(fig)

    def _sftp_pull_recursive(self, sftp, remote_dir, local_dir):
        """Recursively copy a remote SFTP directory to a local directory."""
        os.makedirs(local_dir, exist_ok=True)
        for entry in sftp.listdir_attr(remote_dir):
            remote_path = remote_dir + "/" + entry.filename
            local_path = os.path.join(local_dir, entry.filename)
            if stat.S_ISDIR(entry.st_mode):
                self._sftp_pull_recursive(sftp, remote_path, local_path)
            else:
                sftp.get(remote_path, local_path)

    def pull_data_from_euler(self):
        """Download benchmark CSV and ParaView data from Euler via SFTP."""
        hostname = "euler.ethz.ch"
        username = "wtonnon"
        key_path = os.path.expanduser("~/.ssh/id_ed25519")

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            client.connect(hostname=hostname, username=username, key_filename=key_path, port=22, timeout=10)

            config_directory = "data/config/" + self.name + "/"
            out_directory = "out/data/" + self.name + "/"

            try:
                files = os.listdir(config_directory)
            except Exception:
                print("Error: directory " + config_directory + " not found! Cannot continue.")
                sys.exit(1)

            os.makedirs(out_directory, exist_ok=True)
            sftp = client.open_sftp()

            for config_file in files:
                out_file, _ = os.path.splitext(config_file)
                out_file = out_file + "_vars.csv"
                local_path = "./" + out_directory + out_file
                remote_path = "/cluster/home/wtonnon/dualfieldmfem/" + out_directory + out_file
                try:
                    sftp.get(remote_path, local_path)
                except FileNotFoundError:
                    pass

            remote_base = "/cluster/home/wtonnon/dualfieldmfem/"
            paraview_rel = "data/visualisation/paraview/" + self.name + "/"
            remote_paraview = remote_base + paraview_rel
            local_paraview = "./" + paraview_rel
            try:
                sftp.stat(remote_paraview)
                self._sftp_pull_recursive(sftp, remote_paraview.rstrip("/"), local_paraview)
            except FileNotFoundError:
                pass

            sftp.close()
        finally:
            client.close()
