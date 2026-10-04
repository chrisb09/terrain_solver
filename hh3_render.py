"""Shared PGF/PDF rendering hook for the HH3 analysis figures.

If '--pgf' is in sys.argv, figures are saved through the thesis style
(thesis_style.save -> .pgf + .pdf + .png); otherwise plain PNG via Agg.

Usage in a renderer:
    from hh3_render import use_pgf, save_figure, close_figure
    ...
    fig = plt.subplots(...)[0]
    save_figure(fig, out_path)      # respects --pgf
    close_figure(fig)
"""
import sys
from pathlib import Path

PGF = "--pgf" in sys.argv
if PGF:
    sys.path.insert(0, str(Path.home() / "Master-Thesis" / "gitlab" / "sourcecode"))
    import thesis_style as ts

    ts.apply()
    import matplotlib.pyplot as plt
else:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


def save_figure(fig, out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if PGF:
        ts.save(fig, out.with_suffix(""))
        print(f"Wrote {out.with_suffix('.pgf')} (+ .pdf, .png)")
    else:
        fig.savefig(out, dpi=170)
        print(f"Wrote {out}")


def close_figure(fig) -> None:
    if not PGF:
        import matplotlib.pyplot as plt
        plt.close(fig)
