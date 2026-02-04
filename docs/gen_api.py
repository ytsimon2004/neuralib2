import importlib
import sys
import textwrap
from pathlib import Path

# === Configuration ===
PACKAGES_ROOT = Path('../packages').resolve()
DST = Path('source/api')
DST.mkdir(parents=True, exist_ok=True)

AUTOSUMMARY_DIR = DST / '_autosummary'
AUTOSUMMARY_DIR.mkdir(exist_ok=True)

CONTENT_FILE = """\
{module}
{underline}

.. currentmodule:: {module}

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

{autosummary_list}
"""

CONTENT_DIR = """\
{module}
{underline}

.. automodule:: {module}
   :members:

.. toctree::
    :maxdepth: 1
    :caption: Modules:

{module_list}
"""

# Track generated .rst files and autosummary targets
generated_rst_files = ['neuralib.rst']
autosummary_targets = []


def discover_packages():
    """Discover all neuralib packages in the packages/ directory."""
    packages = []
    for pkg_dir in PACKAGES_ROOT.glob('neuralib-*'):
        if pkg_dir.is_dir():
            src_path = pkg_dir / 'src'
            if src_path.exists():
                packages.append(src_path)
                # Add to sys.path for import resolution
                sys.path.insert(0, str(src_path))
                print(f"[Discovered] {pkg_dir.name}")
    return packages


def get_module_all(module_path: Path, src_root: Path) -> list:
    """Dynamically import a module and extract __all__, if available."""
    try:
        rel_path = module_path.relative_to(src_root)
        modname = '.'.join(rel_path.with_suffix('').parts)
        mod = importlib.import_module(modname)
        return getattr(mod, '__all__', [])
    except Exception as e:
        print(f"[Warning] Could not extract __all__ from {module_path.name}: {e}")
        return []


def write_module_file(module: str, output_path: Path, all_list: list):
    """Write an .rst file for a module using only autosummary."""
    if not all_list:
        print(f"[Skipped] {module} has no __all__")
        return

    autosummary_list = textwrap.indent('\n'.join(all_list), '   ')
    content = CONTENT_FILE.format(
        module=module,
        underline='=' * len(module),
        autosummary_list=autosummary_list
    )
    autosummary_targets.extend([f"{module}.{name}" for name in all_list])
    output_path.write_text(content)
    print(f"[Created] {output_path}")


def write_directory_index(module: str, output_path: Path, module_list: list):
    """Write an .rst index file for a package directory."""
    module_list.sort()
    formatted_list = textwrap.indent('\n'.join(module_list), '    ')
    content = CONTENT_DIR.format(
        module=module,
        underline='=' * len(module),
        module_list=formatted_list
    )
    output_path.write_text(content)
    print(f"[Created] {output_path}")


def process_source_tree(src_root: Path):
    """Walk through a source tree and generate .rst files."""
    neuralib_root = src_root / 'neuralib'
    if not neuralib_root.exists():
        print(f"[Warning] No neuralib module found in {src_root}")
        return

    for path in neuralib_root.rglob('*'):
        rel = path.relative_to(src_root)
        module_path = DST / (str(rel.with_suffix('.rst')).replace('/', '.'))

        if path.is_file() and path.suffix == '.py' and not path.name.startswith('_'):
            generated_rst_files.append(module_path.name)
            if not module_path.exists():
                modname = '.'.join(rel.with_suffix('').parts)
                all_list = get_module_all(path, src_root)
                write_module_file(modname, module_path, all_list)

        elif path.is_dir() and (path / '__init__.py').exists() and '__pycache__' not in path.parts:
            generated_rst_files.append(module_path.name)
            if not module_path.exists():
                modname = '.'.join(rel.parts)
                submodules = []
                for child in path.iterdir():
                    if child.suffix == '.py' and not child.name.startswith('_'):
                        submodules.append(f"{modname}.{child.stem}")
                    elif child.is_dir() and (child / '__init__.py').exists():
                        submodules.append(f"{modname}.{child.name}")
                write_directory_index(modname, module_path, submodules)


def write_main_index():
    """Write the main neuralib.rst index file."""
    # Collect all top-level neuralib submodules
    submodules = set()
    for src_root in discover_packages():
        neuralib_root = src_root / 'neuralib'
        if neuralib_root.exists():
            for child in neuralib_root.iterdir():
                if child.is_dir() and (child / '__init__.py').exists() and not child.name.startswith('_'):
                    submodules.add(f"neuralib.{child.name}")
                elif child.suffix == '.py' and not child.name.startswith('_'):
                    submodules.add(f"neuralib.{child.stem}")

    if submodules:
        write_directory_index("neuralib", DST / "neuralib.rst", sorted(list(submodules)))


def cleanup_stale_rst():
    """Remove stale .rst files that are no longer needed."""
    for f in DST.glob('*.rst'):
        if f.name not in generated_rst_files:
            print(f"[Stale] {f.name}")
            # f.unlink()  # Uncomment to auto-delete


if __name__ == '__main__':
    print("=== Discovering neuralib packages ===")
    packages = discover_packages()

    print("\n=== Processing source trees ===")
    for src_root in packages:
        print(f"\nProcessing: {src_root}")
        process_source_tree(src_root)

    print("\n=== Writing main index ===")
    write_main_index()

    print("\n=== Checking for stale files ===")
    cleanup_stale_rst()

    print("\n=== Done ===")
    print(f"Generated {len(generated_rst_files)} .rst files")
    print(f"Tracked {len(autosummary_targets)} autosummary targets")
