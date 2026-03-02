#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


ENV_MAPPING = {
	"day_high": "昼多い",
	"day_low": "昼少ない",
	"night_high": "夜多い",
	"night_low": "夜少ない",
}


@dataclass
class VideoAnno:
	person_id: str
	env_name: str
	video_name: str
	start: Optional[int]
	mid: Optional[int]
	end: Optional[int]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Check SAM3D outputs against mini.json start/mid/end and video counts"
	)
	parser.add_argument(
		"--mini-json",
		type=Path,
		default=Path("/work/SSR/share/data/drive/annotation/split_mid_end/mini.json"),
		help="Path to mini.json",
	)
	parser.add_argument(
		"--video-root",
		type=Path,
		default=Path("/work/SSR/share/data/drive/videos_split"),
		help="Path to source videos root",
	)
	parser.add_argument(
		"--result-root",
		type=Path,
		default=Path("/work/SSR/share/data/drive/sam3d_body_results_right"),
		help="Path to sam3d result root",
	)
	parser.add_argument(
		"--views",
		nargs="+",
		default=["front", "left", "right"],
		help="Views to check",
	)
	parser.add_argument(
		"--strict-missing",
		action="store_true",
		help="Treat missing frames in [mid, end) as mismatch (default only checks out-of-range).",
	)
	parser.add_argument(
		"--range-mode",
		choices=["start_end", "mid_end"],
		default="start_end",
		help="Frame range source: start_end => [start, end), mid_end => [mid, end).",
	)
	parser.add_argument(
		"--csv-out",
		type=Path,
		default=None,
		help="Optional CSV output path for detailed per-task check results.",
	)
	return parser.parse_args()


def _parse_video_name(video_field: str) -> Optional[Tuple[str, str, str]]:
	base = os.path.basename(video_field)
	m = re.match(r"^(person_\d+)_(day_high|day_low|night_high|night_low).*", base)
	if not m:
		return None

	person = m.group(1)
	env_key = m.group(2)
	env_name = ENV_MAPPING.get(env_key)
	person_id = person.split("_")[1]
	return person_id, env_name, base


def _extract_start_mid_end(item: dict) -> Tuple[Optional[int], Optional[int], Optional[int]]:
	frames = {"start": None, "mid": None, "end": None}
	for label_obj in item.get("videoLabels", []):
		labels = label_obj.get("timelinelabels", [])
		if not labels:
			continue
		label_name = labels[0]
		frame_num = label_obj.get("ranges", [{}])[0].get("start")
		if label_name in frames:
			frames[label_name] = frame_num
	return frames["start"], frames["mid"], frames["end"]


def load_annotations(mini_json: Path) -> List[VideoAnno]:
	data = json.loads(mini_json.read_text(encoding="utf-8"))
	annos: List[VideoAnno] = []
	for item in data:
		parsed = _parse_video_name(item.get("video", ""))
		if parsed is None:
			continue
		person_id, env_name, video_name = parsed
		start, mid, end = _extract_start_mid_end(item)
		annos.append(
			VideoAnno(
				person_id=person_id,
				env_name=env_name,
				video_name=video_name,
				start=start,
				mid=mid,
				end=end,
			)
		)
	return annos


def parse_result_frame_indices(result_view_dir: Path) -> Set[int]:
	indices: Set[int] = set()
	for npz_path in result_view_dir.glob("*_sam3d_body.npz"):
		stem = npz_path.stem  # e.g. 000619_sam3d_body
		frame_str = stem.split("_", 1)[0]
		if frame_str.isdigit():
			indices.add(int(frame_str))
	return indices


def check_one_video(
	anno: VideoAnno,
	video_root: Path,
	result_root: Path,
	views: List[str],
	strict_missing: bool,
	range_mode: str,
) -> Dict[str, object]:
	person_dir = video_root / anno.person_id
	env_dir = person_dir / anno.env_name
	result_env_dir = result_root / anno.person_id / anno.env_name

	source_exists = env_dir.exists()
	result_exists = result_env_dir.exists()

	expected_range = None
	expected_start = None
	expected_end = None
	if range_mode == "start_end":
		expected_start = anno.start
		expected_end = anno.end
	elif range_mode == "mid_end":
		expected_start = anno.mid
		expected_end = anno.end

	if (
		expected_start is not None
		and expected_end is not None
		and int(expected_end) >= int(expected_start)
	):
		expected_range = set(range(int(expected_start), int(expected_end)))

	missing_views = []
	per_view = {}
	hard_mismatch = False

	for view in views:
		view_dir = result_env_dir / view
		if not view_dir.exists():
			missing_views.append(view)
			per_view[view] = {
				"count": 0,
				"out_of_range": 0,
				"missing_in_range": None,
			}
			hard_mismatch = True
			continue

		indices = parse_result_frame_indices(view_dir)
		out_of_range = 0
		missing_in_range = None

		if expected_range is not None:
			out_of_range = len([x for x in indices if x not in expected_range])
			missing_in_range = len(expected_range - indices)
			if out_of_range > 0:
				hard_mismatch = True
			if strict_missing and missing_in_range > 0:
				hard_mismatch = True

		per_view[view] = {
			"count": len(indices),
			"out_of_range": out_of_range,
			"missing_in_range": missing_in_range,
		}

	if not source_exists or not result_exists:
		hard_mismatch = True

	return {
		"key": f"{anno.person_id}/{anno.env_name}",
		"video_name": anno.video_name,
		"source_exists": source_exists,
		"result_exists": result_exists,
		"start": anno.start,
		"mid": anno.mid,
		"end": anno.end,
		"range_mode": range_mode,
		"expected_start": expected_start,
		"expected_end": expected_end,
		"expected_range_size": len(expected_range) if expected_range is not None else None,
		"missing_views": missing_views,
		"per_view": per_view,
		"match": not hard_mismatch,
	}


def main() -> None:
	args = parse_args()
	annos = load_annotations(args.mini_json)

	if not annos:
		print("[ERROR] No valid entries parsed from mini.json")
		return

	unique_anno_tasks = sorted({(a.person_id, a.env_name) for a in annos})
	print(f"[INFO] mini.json entries parsed: {len(annos)}")
	print(f"[INFO] unique person/env tasks: {len(unique_anno_tasks)}")
	print(f"[INFO] range mode: {args.range_mode}")

	source_tasks = set()
	if args.video_root.exists():
		for person_dir in args.video_root.iterdir():
			if not person_dir.is_dir():
				continue
			for env_dir in person_dir.iterdir():
				if env_dir.is_dir():
					source_tasks.add((person_dir.name, env_dir.name))

	result_tasks = set()
	if args.result_root.exists():
		for person_dir in args.result_root.iterdir():
			if not person_dir.is_dir():
				continue
			for env_dir in person_dir.iterdir():
				if env_dir.is_dir():
					result_tasks.add((person_dir.name, env_dir.name))

	anno_task_set = set(unique_anno_tasks)
	miss_in_source = sorted(anno_task_set - source_tasks)
	miss_in_result = sorted(anno_task_set - result_tasks)
	extra_in_result = sorted(result_tasks - anno_task_set)

	print("\n=== Video Count Check ===")
	print(f"annotated tasks: {len(anno_task_set)}")
	print(f"source tasks:    {len(source_tasks)}")
	print(f"result tasks:    {len(result_tasks)}")
	print(f"missing in source: {len(miss_in_source)}")
	print(f"missing in result: {len(miss_in_result)}")
	print(f"extra in result:   {len(extra_in_result)}")

	checks = [
		check_one_video(
			anno=a,
			video_root=args.video_root,
			result_root=args.result_root,
			views=args.views,
			strict_missing=args.strict_missing,
			range_mode=args.range_mode,
		)
		for a in annos
	]

	mismatches = [c for c in checks if not c["match"]]

	if args.csv_out is not None:
		args.csv_out.parent.mkdir(parents=True, exist_ok=True)
		with args.csv_out.open("w", newline="", encoding="utf-8") as f:
			writer = csv.writer(f)
			writer.writerow(
				[
					"key",
					"video_name",
					"source_exists",
					"result_exists",
					"match",
					"range_mode",
					"start",
					"mid",
					"end",
					"expected_start",
					"expected_end",
					"expected_range_size",
					"missing_views",
					"front_count",
					"front_out_of_range",
					"front_missing_in_range",
					"left_count",
					"left_out_of_range",
					"left_missing_in_range",
					"right_count",
					"right_out_of_range",
					"right_missing_in_range",
				]
			)

			for item in checks:
				front = item["per_view"].get(
					"front", {"count": None, "out_of_range": None, "missing_in_range": None}
				)
				left = item["per_view"].get(
					"left", {"count": None, "out_of_range": None, "missing_in_range": None}
				)
				right = item["per_view"].get(
					"right", {"count": None, "out_of_range": None, "missing_in_range": None}
				)

				writer.writerow(
					[
						item["key"],
						item["video_name"],
						item["source_exists"],
						item["result_exists"],
						item["match"],
						item["range_mode"],
						item["start"],
						item["mid"],
						item["end"],
						item["expected_start"],
						item["expected_end"],
						item["expected_range_size"],
						"|".join(item["missing_views"]),
						front["count"],
						front["out_of_range"],
						front["missing_in_range"],
						left["count"],
						left["out_of_range"],
						left["missing_in_range"],
						right["count"],
						right["out_of_range"],
						right["missing_in_range"],
					]
				)

		print(f"\n[INFO] CSV report saved: {args.csv_out}")

	print("\n=== Frame Range Check ===")
	print(f"checked entries: {len(checks)}")
	print(f"mismatch entries: {len(mismatches)}")

	preview = mismatches[:20]
	if preview:
		print("\n--- mismatch preview (first 20) ---")
		for item in preview:
			print(
				f"[{item['key']}] source={item['source_exists']} result={item['result_exists']} "
				f"start={item['start']} mid={item['mid']} end={item['end']} "
				f"expected=[{item['expected_start']},{item['expected_end']}) "
				f"missing_views={item['missing_views']}"
			)
			for view, stats in item["per_view"].items():
				print(
					f"  - {view}: count={stats['count']}, out_of_range={stats['out_of_range']}, "
					f"missing_in_range={stats['missing_in_range']}"
				)

	if miss_in_source:
		print("\n--- tasks missing in source (first 20) ---")
		for t in miss_in_source[:20]:
			print(t)

	if miss_in_result:
		print("\n--- tasks missing in result (first 20) ---")
		for t in miss_in_result[:20]:
			print(t)

	if extra_in_result:
		print("\n--- extra tasks in result (first 20) ---")
		for t in extra_in_result[:20]:
			print(t)


if __name__ == "__main__":
	main()
