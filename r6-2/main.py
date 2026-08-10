"""Convenience dispatcher: python main.py video ... / camera ... / evaluate ..."""

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"video", "camera", "evaluate"}:
        print("Usage: python main.py {video|camera|evaluate} [arguments]")
        return 2
    command = sys.argv.pop(1)
    if command == "video":
        from infer_video import main as command_main
    elif command == "camera":
        from infer_camera import main as command_main
    else:
        from evaluate import main as command_main
    return command_main()


if __name__ == "__main__":
    raise SystemExit(main())
