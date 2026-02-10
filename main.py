from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from app import App


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    App(root)
    root.minsize(1200, 600)
    root.mainloop()


if __name__ == "__main__":
    main()
