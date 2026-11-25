import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate shifted-attention checkpoints on released real series")
    cmds = parser.add_subparsers(dest="action", required=True)
    prep = cmds.add_parser("prepare", help="Download and verify public benchmark assets")
    prep.add_argument("--data", required=True, type=Path)
    prep.add_argument("--assets", nargs="+", choices=["tsad","metrics","chamber","msds","author"], default=["tsad","chamber","msds"])
    for name in ("timercd","tsb","rca"):
        cmd = cmds.add_parser(name)
        cmd.add_argument("--checkpoint", type=Path, help="Required for the shifted model")
        cmd.add_argument("--predictor",choices=["shifted","severity"] if name=="rca" else ["shifted","severity","author"],
                         default="shifted")
        cmd.add_argument("--data", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--device", default="cuda")
        if name == "rca":
            cmd.add_argument("--ledger", type=Path, required=True)
            cmd.add_argument("--threshold", type=float, help="Locked source calibration threshold for alarm-conditioned evaluation")
        else:
            cmd.add_argument("--width", type=int)
            cmd.add_argument("--stride", type=int, default=None)
            cmd.add_argument("--split", choices=["U","M","both"], default="both")
            if name == "timercd":
                cmd.add_argument("--metric-engine",choices=["fast","default"],default="fast",
                                 help="Published integration implements both threshold grids; fast is its CLI default")
    industrial = cmds.add_parser("industrial", help="Ingest a release-specific SWaT/WADI or MSDS entity spec")
    industrial.add_argument("--spec", type=Path, required=True)
    industrial.add_argument("--data", type=Path, required=True)
    args = parser.parse_args(argv)
    if getattr(args,"predictor",None)=="shifted" and args.checkpoint is None:
        parser.error("the shifted predictor requires --checkpoint")
    if args.action == "prepare":
        from .assets import prepare
        prepare(args.data, args.assets)
    elif args.action == "industrial":
        from .adapters import industrial_import
        industrial_import(args.spec, args.data)
    elif args.action == "rca":
        from .rca import run
        run(args)
    else:
        from .tsad import run
        run(args)


if __name__ == "__main__":
    main()
