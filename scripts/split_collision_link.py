#!/usr/bin/env python3
"""Split one link's collision spheres into spatial clusters of child links.

The generated collision checker (cricket -> VAMP) tests two links' bounding
spheres before testing any of their spheres against each other. The
chassis carries most of the robot's spheres inside one bounding sphere of
~0.8 m radius that overlaps the arm in every pose, so every check paid for
every chassis-sphere x arm-sphere pair. Moving each spatial cluster of those
spheres onto its own fixed child link gives each cluster a tight bounding
sphere, and the broad phase skips the clusters the arm is nowhere near.

The spheres themselves are unchanged. Each new link is attached to the
original one by an identity fixed joint, so it moves identically, and every
``disable_collisions`` entry of the original link is mirrored onto it in the
SRDF, so the self-collision pairs are unchanged too.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def kmeans(points: np.ndarray, k: int, seed: int = 0, iters: int = 100) -> np.ndarray:
    """Labels of a k-means++ clustering (best of 10 restarts)."""
    rng = np.random.default_rng(seed)
    best, best_cost = None, np.inf
    for _ in range(10):
        centers = [points[rng.integers(len(points))]]
        for _ in range(k - 1):
            d2 = np.min(
                ((points[:, None] - np.array(centers)[None]) ** 2).sum(-1), axis=1
            )
            centers.append(points[rng.choice(len(points), p=d2 / d2.sum())])
        centers = np.array(centers)
        for _ in range(iters):
            labels = ((points[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
            new = np.array(
                [
                    points[labels == j].mean(0) if np.any(labels == j) else centers[j]
                    for j in range(k)
                ]
            )
            if np.allclose(new, centers):
                break
            centers = new
        cost = ((points - centers[labels]) ** 2).sum()
        if cost < best_cost:
            best, best_cost = labels, cost
    return best


def split(
    urdf: ET.ElementTree, srdf: ET.ElementTree, link_name: str, clusters: int
) -> list[str]:
    robot = urdf.getroot()
    link = next(lk for lk in robot.findall("link") if lk.get("name") == link_name)
    collisions = link.findall("collision")
    for c in collisions:
        if c.find("geometry/sphere") is None:
            raise ValueError(
                f"{link_name}: only sphere collision geometry can be split"
            )
        rpy = c.find("origin").get("rpy") or "0 0 0"
        if any(float(v) != 0.0 for v in rpy.split()):
            raise ValueError(f"{link_name}: sphere origins must have zero rpy")
    if len(collisions) <= clusters:
        return []
    centers = np.array(
        [[float(v) for v in c.find("origin").get("xyz").split()] for c in collisions]
    )
    labels = kmeans(centers, clusters)

    new_links = []
    insert_at = list(robot).index(link) + 1
    for j in range(clusters):
        name = f"{link_name}_collision_{j}"
        new_links.append(name)
        child = ET.Element("link", name=name)
        for c, label in zip(collisions, labels):
            if label == j:
                link.remove(c)
                child.append(c)
        joint = ET.Element("joint", name=f"{name}_joint", type="fixed")
        ET.SubElement(joint, "parent", link=link_name)
        ET.SubElement(joint, "child", link=name)
        ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
        robot.insert(insert_at, child)
        robot.insert(insert_at + 1, joint)
        insert_at += 2

    srdf_root = srdf.getroot()
    for d in list(srdf_root.findall("disable_collisions")):
        pair = (d.get("link1"), d.get("link2"))
        if link_name not in pair:
            continue
        other = pair[1] if pair[0] == link_name else pair[0]
        for name in new_links:
            srdf_root.append(
                ET.Element(
                    "disable_collisions",
                    link1=name,
                    link2=other,
                    reason=d.get("reason", "Default"),
                )
            )
    return new_links


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--urdf", required=True, type=Path, help="Spherized URDF, rewritten in place"
    )
    ap.add_argument("--srdf", required=True, type=Path, help="SRDF, rewritten in place")
    ap.add_argument("--link", required=True, help="Link whose spheres are split")
    ap.add_argument("--clusters", type=int, default=12)
    args = ap.parse_args()

    urdf, srdf = ET.parse(args.urdf), ET.parse(args.srdf)
    prefix = f"{args.link}_collision_"
    if any(
        lk.get("name", "").startswith(prefix) for lk in urdf.getroot().findall("link")
    ):
        raise SystemExit(
            f"{args.urdf} already has {prefix}* links; split a fresh FOAM output."
        )
    # Re-running the pipeline must not stack duplicates: drop the SRDF
    # entries of an earlier split of this link first.
    for d in list(srdf.getroot().findall("disable_collisions")):
        if d.get("link1", "").startswith(prefix) or d.get("link2", "").startswith(
            prefix
        ):
            srdf.getroot().remove(d)
    new_links = split(urdf, srdf, args.link, args.clusters)
    for tree, path in ((urdf, args.urdf), (srdf, args.srdf)):
        ET.indent(tree, space="  ")
        body = ET.tostring(tree.getroot(), encoding="unicode")
        path.write_text(f"<?xml version='1.0' encoding='utf-8'?>\n{body}\n")
    print(f"Split {args.link} into {len(new_links)} collision links.")


if __name__ == "__main__":
    main()
