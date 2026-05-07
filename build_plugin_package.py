from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

PLUGIN_ROOT_NAME = "amazon_bedrock_llms"
DIST_DIR_NAME = "dist"
FILES_TO_INCLUDE = [
    "plugin.json",
    "requirements.txt",
    "settings.json",
    "bedrock_llms.py",
    "bedrock_price_estimator.py",
    "cached_model_costs.json",
    "cached_model_pricing.json",
    "README.md",
]
DIRS_TO_INCLUDE = ["assets"]


def build_package() -> Path:
    root = Path(__file__).resolve().parent
    dist_dir = root / DIST_DIR_NAME
    dist_dir.mkdir(exist_ok=True)
    package_path = dist_dir / f"{PLUGIN_ROOT_NAME}.zip"

    with ZipFile(package_path, "w", compression=ZIP_DEFLATED) as archive:
        for relative_name in FILES_TO_INCLUDE:
            source = root / relative_name
            if source.exists():
                archive.write(source, Path(PLUGIN_ROOT_NAME) / relative_name)

        for relative_dir in DIRS_TO_INCLUDE:
            source_dir = root / relative_dir
            if not source_dir.exists():
                continue
            for source in source_dir.rglob("*"):
                if source.is_file():
                    archive.write(source, Path(PLUGIN_ROOT_NAME) / source.relative_to(root))

    return package_path


def main() -> int:
    package_path = build_package()
    print(f"Created plugin package: {package_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

