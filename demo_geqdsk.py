"""Pure GEQDSK payload round-trip using the fusionprime-base I/O boundary."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from fusionprime_base.io.geqdsk import load_geqdsk, save_geqdsk

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "SOLOVEV.geqdsk"


def main() -> int:
    """Read and write a GEQDSK payload without constructing an Equilibrium."""

    with TemporaryDirectory(prefix="veqpy-geqdsk-") as directory:
        output = Path(directory) / "roundtrip.geqdsk"
        payload = load_geqdsk(INPUT)
        save_geqdsk(payload, output)
        restored = load_geqdsk(output)
        print("fusionprime-base GEQDSK payload round-trip")
        print(f"grid: NR={restored.NR}, NZ={restored.NZ}")
        print(f"boundary points: {restored.boundary.shape[0]}")
        print(f"output: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
