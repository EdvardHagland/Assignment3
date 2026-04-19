from __future__ import annotations

import argparse
from pathlib import Path
import sys

LABELING_DIR = Path(__file__).resolve().parent
if str(LABELING_DIR) not in sys.path:
    sys.path.insert(0, str(LABELING_DIR))

from webapp import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local SEC annotation app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
