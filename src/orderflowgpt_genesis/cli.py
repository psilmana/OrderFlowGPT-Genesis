"""Command-line interface for Genesis."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .config import GenesisConfiguration
from .runner import GenesisRunner, RunnerResult

VERSION = "0.1.0"


class GenesisCLI:
    """Parse command-line arguments and execute the Genesis runner."""

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="orderflowgpt_genesis")
        source = parser.add_mutually_exclusive_group()
        source.add_argument("--video", type=Path)
        source.add_argument("--folder", type=Path)
        parser.add_argument("--transcript", type=Path)
        parser.add_argument("--output", type=Path)
        parser.add_argument("--overwrite", action="store_true")
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument("--config", type=Path)
        parser.add_argument(
            "--version", action="version", version=f"orderflowgpt-genesis {VERSION}"
        )
        return parser

    def parse(self, argv: Sequence[str] | None = None) -> argparse.Namespace:
        return self.build_parser().parse_args(argv)

    def run(self, argv: Sequence[str] | None = None) -> int:
        args = self.parse(argv)
        config = self._configuration(args)
        runner = GenesisRunner(config)
        try:
            if args.video is not None:
                results: tuple[RunnerResult, ...] = (
                    runner.run_video(args.video, args.transcript),
                )
            elif args.folder is not None:
                results = runner.run_folder(args.folder)
            else:
                self.build_parser().print_help()
                return 0
        except Exception as exc:
            print(f"Genesis error: {exc}")
            return 2
        if args.verbose:
            for result in results:
                print(f"{result.lesson_id}: {result.report}")
        return 0

    def _configuration(self, args: argparse.Namespace) -> GenesisConfiguration:
        config = GenesisConfiguration()
        if args.config is not None:
            data = json.loads(args.config.read_text(encoding="utf-8"))
            config = GenesisConfiguration(
                video_folder=Path(data.get("video_folder", config.video_folder)),
                transcript_folder=Path(
                    data.get("transcript_folder", config.transcript_folder)
                ),
                output_folder=Path(data.get("output_folder", config.output_folder)),
                frame_extraction_interval=int(
                    data.get(
                        "frame_extraction_interval", config.frame_extraction_interval
                    )
                ),
                logging=bool(data.get("logging", config.logging)),
                overwrite=bool(data.get("overwrite", config.overwrite)),
            )
        if args.output is not None:
            config = replace(config, output_folder=args.output)
        if args.overwrite:
            config = replace(config, overwrite=True)
        return config


def main(argv: Sequence[str] | None = None) -> int:
    return GenesisCLI().run(argv)
