"""Run a configured job."""
from tc.cli import parse_args

def main() -> None:
    parse_args().execute()

if __name__ == "__main__":
    main()
