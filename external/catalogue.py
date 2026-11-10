"""The open dataset we use: SkillCorner's, to learn possession from touches (possession/touch_possession.py).

The bar: free, on a public host anyone can download from, and a licence that lets us use it (CC BY / MIT). External
data is allowed by RULES.md as long as the organisers can reach it, which a public host with an open licence
guarantees. Checked 2026-10-26.
"""

DATASETS = {
    "skillcorner": {
        "name": "SkillCorner Open Data",
        "provider": "SkillCorner with PySport",
        "licence": "MIT",
        "host": "GitHub (tracking in Git LFS)",
        "url": "https://github.com/SkillCorner/opendata",
        "games": 20,
        "contents": "Broadcast tracking (10 Hz, extrapolated off-screen players), dynamic events "
                    "(possessions, pressing, off-ball runs, passing options) and phases of play. "
                    "A-League 2024/25.",
        "video": False,
        "size": "2 GB",
    },
}
