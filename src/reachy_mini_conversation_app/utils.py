from __future__ import annotations
import os
import sys
import logging
import argparse
import warnings
import subprocess
from typing import TYPE_CHECKING, Optional

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.camera_worker import CameraWorker
from reachy_mini_conversation_app.vision.head_tracking import HeadTracker


if TYPE_CHECKING:
    from reachy_mini_conversation_app.vision.local_vision import VisionProcessor
    from reachy_mini_conversation_app.vision.remote_vision import RemoteVisionProcessor


class CameraVisionInitializationError(Exception):
    """Raised when camera or vision setup fails in an expected way."""


def parse_args() -> tuple[argparse.Namespace, list]:  # type: ignore
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Reachy Mini Conversation App")
    parser.add_argument(
        "--head-tracker",
        choices=["yolo", "mediapipe"],
        default=None,
        help=(
            "Optional head-tracking backend: yolo uses a local face detector in a subprocess, "
            "mediapipe uses reachy_mini_toolbox in process. Disabled by default."
        ),
    )
    parser.add_argument("--no-camera", default=False, action="store_true", help="Disable camera usage")
    parser.add_argument(
        "--local-vision",
        default=False,
        action="store_true",
        help="Use local vision model instead of the selected realtime backend vision",
    )
    parser.add_argument("--gradio", default=False, action="store_true", help="Open gradio interface")
    parser.add_argument("--debug", default=False, action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--cascade",
        default=False,
        action="store_true",
        help="Use the cascade backend (ASR→LLM→TTS pipeline) instead of a realtime backend.",
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        default=None,
        help="[Optional] Robot name to target. Must match the daemon's --robot-name when connecting to a specific robot, mainly useful for development with multiple robots.",
    )
    subparsers = parser.add_subparsers(dest="command")
    tool_spaces_parser = subparsers.add_parser("tool-spaces", help="Manage installed Hugging Face Space tool sources")
    tool_spaces_subparsers = tool_spaces_parser.add_subparsers(dest="tool_spaces_command", required=True)

    add_parser = tool_spaces_subparsers.add_parser("add", help="Install one public Space tool source by slug")
    add_parser.add_argument("space_slug", help="Public Hugging Face Space slug in the form owner/space-name")
    add_parser.add_argument(
        "--install-only",
        action="store_true",
        default=False,
        help="Install the Space without enabling its tools in any profile.",
    )
    add_parser.add_argument(
        "--profile",
        dest="profile",
        default=None,
        metavar="PROFILE",
        help="Enable tools in this profile instead of the active profile.",
    )

    remove_parser = tool_spaces_subparsers.add_parser("remove", help="Remove one installed Space tool source")
    remove_parser.add_argument("space_slug", help="Installed Hugging Face Space slug in the form owner/space-name")

    tool_spaces_subparsers.add_parser("list", help="List installed Space tool sources")
    return parser.parse_known_args()


def initialize_camera_and_vision(
    args: argparse.Namespace,
    current_robot: ReachyMini,
) -> tuple[CameraWorker | None, "VisionProcessor | RemoteVisionProcessor | None"]:
    """Initialize camera capture, optional head tracking, and optional vision.

    Vision backend precedence: --local-vision (local SmolVLM2) > VL_BASE_URL env
    (remote Qwen3-VL returning text) > None (the realtime backend handles vision).
    The remote path is what the text-only DM brain uses to "read the table".
    """
    camera_worker: Optional[CameraWorker] = None
    head_tracker: HeadTracker | None = None
    vision_processor: Optional["VisionProcessor | RemoteVisionProcessor"] = None

    if not args.no_camera:
        if args.head_tracker is not None:
            try:
                if args.head_tracker == "yolo":
                    from reachy_mini_conversation_app.vision.head_tracking.yolo_process import (
                        YoloHeadTrackerProcess,
                    )

                    head_tracker = YoloHeadTrackerProcess()
                    logging.getLogger(__name__).info("Using yolo head tracker subprocess")
                else:
                    from reachy_mini_conversation_app.vision.head_tracking.mediapipe import (
                        MediapipeHeadTracker,
                    )

                    head_tracker = MediapipeHeadTracker()
                    logging.getLogger(__name__).info("Using mediapipe head tracker in process")
            except Exception as e:
                raise CameraVisionInitializationError(
                    f"Failed to initialize {args.head_tracker} head tracker: {e}",
                ) from e

        camera_worker = CameraWorker(current_robot, head_tracker)

        if args.local_vision:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from reachy_mini_conversation_app.vision.local_vision import VisionProcessor",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode < 0:
                raise CameraVisionInitializationError(
                    "Local vision import crashed on this machine. "
                    "Run without --local-vision or install compatible dependencies.",
                )
            try:
                from reachy_mini_conversation_app.vision.local_vision import initialize_vision_processor

            except ImportError as e:
                raise CameraVisionInitializationError(
                    "To use --local-vision, please install the extra dependencies: pip install '.[local_vision]'",
                ) from e

            vision_processor = initialize_vision_processor()
        elif os.environ.get("VL_BASE_URL"):
            from reachy_mini_conversation_app.vision.remote_vision import RemoteVisionProcessor

            base_url = os.environ["VL_BASE_URL"]
            vision_processor = RemoteVisionProcessor(base_url)
            logging.getLogger(__name__).info(
                "Using remote Qwen3-VL vision backend at %s (camera tool returns text).",
                base_url,
            )
        else:
            logging.getLogger(__name__).info(
                "Using the selected realtime backend for vision (default). Use --local-vision for "
                "local processing, or set VL_BASE_URL for the remote Qwen3-VL camera tool.",
            )

    return camera_worker, vision_processor


def setup_logger(debug: bool) -> logging.Logger:
    """Setups the logger."""
    log_level = "DEBUG" if debug else "INFO"
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)

    # Suppress WebRTC warnings
    warnings.filterwarnings("ignore", message=".*AVCaptureDeviceTypeExternal.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="aiortc")

    # Tame third-party noise (looser in DEBUG)
    if log_level == "DEBUG":
        logging.getLogger("aiortc").setLevel(logging.INFO)
        logging.getLogger("fastrtc").setLevel(logging.INFO)
        logging.getLogger("aioice").setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.INFO)
        logging.getLogger("websockets").setLevel(logging.INFO)
    else:
        logging.getLogger("aiortc").setLevel(logging.ERROR)
        logging.getLogger("fastrtc").setLevel(logging.ERROR)
        logging.getLogger("aioice").setLevel(logging.WARNING)
    return logger


def log_connection_troubleshooting(logger: logging.Logger, robot_name: Optional[str]) -> None:
    """Log troubleshooting steps for connection issues."""
    logger.error("Troubleshooting steps:")
    logger.error("  1. Verify reachy-mini-daemon is running")

    if robot_name is not None:
        logger.error(f"  2. Daemon must be started with: --robot-name '{robot_name}'")
    else:
        logger.error("  2. If daemon uses --robot-name, add the same flag here: --robot-name <name>")

    logger.error("  3. For wireless: check network connectivity")
    logger.error("  4. Review daemon logs")
    logger.error("  5. Restart the daemon")
