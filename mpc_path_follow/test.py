# plot_csv_points_tkinter_with_yaw_labels.py
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import cm

class CSVPointPlotter:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Path Plotter – with yaw labels")
        self.root.geometry("1100x800")

        self.label = tk.Label(
            root,
            text="Load CSV file (semicolon separated) – shows yaw next to each point",
            font=("Arial", 12)
        )
        self.label.pack(pady=12)

        self.btn_load = tk.Button(
            root,
            text="Select CSV File",
            command=self.load_csv,
            font=("Arial", 11),
            width=20
        )
        self.btn_load.pack(pady=10)

        self.status = tk.Label(
            root,
            text="No file loaded",
            fg="gray",
            font=("Arial", 10)
        )
        self.status.pack(pady=6)

        self.figure = None
        self.canvas = None
        self.toolbar = None

    def load_csv(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            # Read with semicolon delimiter (common in your use-case)
            df = pd.read_csv(file_path, sep=';')

            if df.shape[1] < 2:
                messagebox.showerror("Error", "CSV must have at least 2 columns (x, y)")
                return

            # Take first two columns as x and y
            x = df.iloc[:, 0].to_numpy()
            y = df.iloc[:, 1].to_numpy()

            # If there is a yaw column (3rd column), use it; otherwise calculate it
            if df.shape[1] >= 3:
                yaw = df.iloc[:, 2].to_numpy()
                yaw_source = "from 3rd column"
            else:
                # Calculate yaw from consecutive points
                yaw = self.calculate_yaw(x, y)
                yaw_source = "calculated from points"

            self.status.config(
                text=f"Loaded {len(df)} points  |  yaw: {yaw_source}",
                fg="green"
            )

            self.plot_points(x, y, yaw, title=f"Path from {file_path.split('/')[-1]}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to read file:\n{str(e)}")

    def calculate_yaw(self, x, y):
        """Calculate yaw (heading) from each point to the next"""
        if len(x) < 2:
            return np.zeros_like(x)

        dx = np.diff(x)
        dy = np.diff(y)
        yaw = np.arctan2(dy, dx)
        # Last point gets yaw of previous segment
        yaw = np.append(yaw, yaw[-1])

        return yaw

    def plot_points(self, x, y, yaw, title="Path with yaw labels"):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        if self.toolbar:
            self.toolbar.destroy()

        self.figure = plt.Figure(figsize=(10, 8), dpi=100)
        ax = self.figure.add_subplot(111)

        n = len(x)

        # Color gradient to show order (start blue → end red/yellow)
        colors = cm.viridis(np.linspace(0, 1, n))

        # Main scatter plot
        sc = ax.scatter(
            x, y,
            c=colors,
            s=40,
            alpha=0.9,
            edgecolor='none',
            cmap='viridis'
        )

        # Label each point with its yaw value (in radians, 3 decimals)
        for i in range(n):
            yaw_str = f"{yaw[i]:.3f}"
            ax.text(
                x[i] + 0.06, y[i] + 0.06,
                yaw_str,
                fontsize=7.5,
                color='black',
                ha='left',
                va='bottom',
                bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', pad=1.2)
            )

        # Colorbar showing order
        cbar = self.figure.colorbar(sc, ax=ax, shrink=0.75, pad=0.04)
        cbar.set_label("Point order (0 = start, 1 = end)", rotation=270, labelpad=18)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.set_aspect('equal')

        # Thin line connecting points to show sequence
        ax.plot(x, y, color='gray', linewidth=0.8, alpha=0.55, zorder=0)

        # Embed in Tkinter
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = CSVPointPlotter(root)
    root.mainloop()