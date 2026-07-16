"""py2app setup script for dictate-mac (Phase 10).

Builds ``dist/DictateMac.app`` — a macOS bundle that contains every
Python dependency (mlx + mlx_whisper, silero_vad with its model, sounddevice,
rumps, pyobjc frameworks) and starts ``dictate_mac.cli:main`` as the
entry point. The whisper STT weights are NOT bundled; they download
from Hugging Face on the first launch on a new Mac.

Run via ``./build.sh`` (handles ``.icns`` generation + python path).

Notes / sharp edges
===================

* ``mlx.metallib`` (~162 MB binary blob) lives at
  ``site-packages/mlx/lib/mlx.metallib`` and is loaded at runtime via
  Apple's Metal C++ API. py2app preserves package directory layouts
  when packages are listed in ``OPTIONS['packages']``, but the ``mlx``
  namespace package has historically been inconsistent under py2app
  resource scanning. We list the entire ``mlx/lib`` directory
  in ``data_files`` explicitly so the metallib lands at
  ``DictateMac.app/Contents/Resources/data/mlx-lib/mlx.metallib`` —
  reachable from ``mlx`` via the standard search path py2app sets up.
* ``silero_vad`` ships its model at ``silero_vad/data/*.jit|.onnx|...
  safetensors`` and resolves it at runtime via ``importlib.resources``.
  Listing ``silero_vad.data`` in ``packages`` is enough — py2app
  bundles non-``.py`` files inside packages automatically and rewires
  ``importlib.resources.files`` to look inside the bundle.
* ``sounddevice`` ships a native ``libportaudio.dylib`` as the private
  package ``_sounddevice_data``. ``dlopen()`` cannot load files that
  live inside ``python313.zip``, so the loader fails at startup. We
  extract ``_sounddevice_data`` after py2app finishes — see
  :func:`_extract_native_runtime_libs` below.
* ``py2app`` (≤0.28.x) refuses to run when ``distribution.install_requires``
  is non-empty. Modern setuptools copies ``[project] dependencies``
  from ``pyproject.toml`` straight into ``install_requires`` — see
  :func:`_noop_dependencies` below for the monkeypatch.
* No code signing (``setup.py`` does not call ``codesign``). Local use
  only — TCC permissions (Microphone, Accessibility, Input Monitoring)
  are granted to ``com.local.dictate-mac`` manually in System
  Settings → Privacy & Security.
"""

from __future__ import annotations

import re
import shutil
import sys
import warnings
from pathlib import Path


def _noop_dependencies(dist, val, _root_dir=None):  # noqa: ANN001
    """Drop-in replacement that ignores ``dependencies`` from pyproject.

    ``py2app`` rejects a non-empty ``install_requires`` (it considers
    runtime deps out of scope for an app bundle — the app is meant to
    be self-contained, not pip-installed by the user). Modern
    setuptools pre-populates ``install_requires`` from
    ``[project] dependencies`` in ``pyproject.toml``. To keep the
    runtime dependency list in one canonical place (pyproject.toml for
    the dev install, setup.py for the bundle), we swap setuptools'
    applier with this no-op before :func:`setuptools.setup` runs.
    """
    return None


def _install_pyproject_compat() -> None:
    """Replace setuptools' ``[project] dependencies`` applier.

    Loaded eagerly — must run before any Distribution reads pyproject.
    """
    try:
        from setuptools.config import _apply_pyprojecttoml as _apt
    except ImportError:
        return
    # The applier dict lives at module level as ``PYPROJECT_CORRESPONDENCE``.
    # Replace the ``dependencies`` entry so it lands on the distribution
    # as an empty list — py2app bails out the moment it sees a
    # non-empty ``install_requires`` and we want a bundled .app, not a
    # pip-installable wheel.
    correspondence = getattr(_apt, "PYPROJECT_CORRESPONDENCE", None)
    if isinstance(correspondence, dict):
        correspondence["dependencies"] = _noop_dependencies


# modulegraph 0.19.x (used by py2app 0.28.x) recurses deeply through
# deeply-nested AST nodes in Python 3.13 wheels (mlx_whisper / torch
# paths). The default 1000-frame Python recursion limit trips well
# before modulegraph finishes. Bump it before importing setuptools so
# the new limit sticks through the whole build.
sys.setrecursionlimit(5000)


def _patch_py2app_for_ns_packages() -> None:
    """``mlx`` is a PEP 420 namespace package with no ``__init__.py``.

    py2app 0.28.x uses ``imp.find_module`` to resolve package paths,
    which raises ``ImportError`` for namespaces. We override
    ``BuildApp.get_bootstrap`` so it falls back to the namespace path
    derived from ``importlib``.
    """
    try:
        from importlib import util as _imputil
    except ImportError:
        return
    try:
        from py2app.build_app import py2app as _BuildApp
    except ImportError:
        return

    if getattr(_BuildApp.get_bootstrap, "_dictate_patched", False):
        return

    _orig = _BuildApp.get_bootstrap

    def get_bootstrap(self, bootstrap):  # type: ignore[no-redef]
        if isinstance(bootstrap, str) and not os.path.exists(bootstrap):
            spec = _imputil.find_spec(bootstrap)
            if spec is not None:
                # Prefer the package directory (submodule_search_locations)
                # — ``collect_packagedirs`` only keeps paths whose
                # ``os.path.exists`` is True AND that look like directories.
                # Returning ``__init__.py`` (a file path) makes the
                # downstream ``os.path.join(realpath(''), "")`` resolve to
                # a non-existent path and the package gets silently
                # dropped.
                if spec.submodule_search_locations:
                    for p in spec.submodule_search_locations:
                        if os.path.exists(p):
                            return p
                if spec.origin is not None and os.path.exists(spec.origin):
                    return spec.origin
            # Last resort: try the original imp-based lookup (may raise).
            return _orig(self, bootstrap)
        return _orig(self, bootstrap)

    get_bootstrap._dictate_patched = True  # type: ignore[attr-defined]
    _BuildApp.get_bootstrap = get_bootstrap


import os  # noqa: E402  — kept under the monkeypatch section

_patch_py2app_for_ns_packages()


_install_pyproject_compat()


try:
    from setuptools import setup
except ImportError:
    sys.stderr.write(
        "setuptools is required to run setup.py — install with "
        "`uv pip install setuptools` or activate a venv that has it.\n"
    )
    raise

PROJECT_ROOT = Path(__file__).resolve().parent

APP = ["src/dictate_mac/__main__.py"]
ICNS_FILE = PROJECT_ROOT / "assets" / "DictateMac.icns"
ICONSET_STAGE = PROJECT_ROOT / "assets" / "DictateMac.iconset"


def _stage_runtime_files() -> tuple[list[tuple[str, list[str]]], list[str]]:
    """Stage and return ``(data_files, resources)`` for ``py2app``.

    ``py2app`` interprets ``data_files`` strictly as
    ``[(bundle_dir, [files_to_copy_there])]`` — directory trees are
    NOT auto-expanded, so we enumerate every file explicitly. The
    bundle directory path is relative to ``Contents/Resources``.

    For files that need to be at exact paths inside the bundle (e.g.
    the Python interpreter has to find ``mlx.metallib``), we use
    ``resources`` instead — py2app copies each file as-is, preserving
    the basename.

    Returns
    -------
    data_files
        List of ``(bundle_dir, [abs_paths])`` tuples passed as
        ``data_files`` to :func:`setup`.
    resources
        List of absolute file paths passed as ``resources`` to
        :func:`setup`. These end up at
        ``Contents/Resources/<basename>``.
    """
    import site

    site_packages = Path(site.getsitepackages()[0])

    bundle_staging = PROJECT_ROOT / "assets" / "_py2app"
    if bundle_staging.is_symlink() or bundle_staging.exists():
        if bundle_staging.is_symlink():
            bundle_staging.unlink()
        elif bundle_staging.is_dir():
            shutil.rmtree(bundle_staging)
        else:
            bundle_staging.unlink()
    bundle_staging.mkdir(parents=True, exist_ok=True)

    staged: dict[str, Path] = {}

    pairs = {
        # Stage the silero-vad model directory so the bundle includes
        # silero_vad.jit / silero_vad.onnx / silero_vad_*.safetensors.
        "silero-vad-data": site_packages / "silero_vad" / "data",
        # mlx.metallib + the small companion .dylibs that the metal
        # backend dlopens at runtime.
        "mlx-lib": site_packages / "mlx" / "lib",
    }

    for stage_name, src in pairs.items():
        if not src.exists():
            warnings.warn(
                f"{src} does not exist; the .app may fail at runtime. "
                "Did you `uv pip install -e .` in this venv?",
                stacklevel=2,
            )
            continue
        dst = bundle_staging / stage_name
        if dst.is_symlink() or dst.exists():
            if dst.is_symlink():
                dst.unlink()
            elif dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copytree(src, dst, symlinks=True)
        files = sorted(p for p in dst.rglob("*") if p.is_file())
        print(f"  staged {stage_name} ({len(files)} files)")
        staged[stage_name] = dst

    data_files: list[tuple[str, list[str]]] = []
    resources: list[str] = []

    # silero-vad: the runtime resolves them via importlib.resources on
    # the package, so copying them anywhere under Resources/ works as
    # long as the importlib bootstrap is wired correctly. py2app's
    # bundle boot adds Resources/ to the package search path.
    if "silero-vad-data" in staged:
        silero_files = sorted(
            str(p) for p in staged["silero-vad-data"].rglob("*") if p.is_file()
        )
        # data_files = [(dir_in_bundle, [files])] — each file gets
        # joined with the dir; basename is preserved by py2app.
        data_files.append(("data/silero_vad", silero_files))
        # Same files are also listed as resources, so each keeps a
        # stable top-level basename for tools that grep the bundle.
        resources.extend(silero_files)

    # mlx.metallib + libmlx.dylib + libjaccl.dylib: do NOT add them
    # to ``resources``. py2app already copies them into the on-disk
    # mirror at ``lib/python3.13/mlx/lib/`` because ``mlx`` is listed
    # in ``OPTIONS['packages']``. ``mlx.core`` resolves the metallib
    # path via ``importlib.resources.files('mlx').joinpath('lib/mlx.metallib')``
    # — that hits the on-disk mirror.
    #
    # Adding each ``mlx/lib/*`` file to ``resources`` was an old
    # belt-and-braces measure for Phase 10 days when py2app's handling
    # of the namespace package was inconsistent. It cost us a 162 MB
    # duplicate of ``mlx.metallib`` plus 16 MB of ``libmlx.dylib``
    # plus 855 KB of ``libjaccl.dylib`` at ``Contents/Resources/`` —
    # a 179 MB penalty for code that no one calls. Confirmed by
    # deleting the top-level copies and re-running selftest (5/5
    # PASS): the on-disk mirror is the one mlx.core uses.
    #
    # If a future py2app regression re-breaks this, a quick recovery
    # is to uncomment the loop below; nothing else depends on it.

    return data_files, resources


def _pre_patch_source_venv() -> dict[Path, Path]:
    """Replace source-venv wheel copies of stubbed modules BEFORE py2app
    scans the dependency graph.

    Why this exists
    ---------------

    py2app 0.28.x uses ``modulegraph`` to build the bundle's dependency
    graph via **static AST analysis**. It reads files from the active
    Python's ``site-packages``, not from any pre-existing bundle. If
    modulegraph sees ``from scipy import signal`` at the top of
    ``mlx_whisper/timing.py`` in the source venv, scipy enters the
    graph *before* ``setup.py`` ever returns — and ``OPTIONS['excludes']``
    cannot remove it (known py2app behavior: namespace packages
    reachable via ``sys.path`` are copied regardless of excludes).

    After setup() returns, our post-build ``_install_timing_stub()``
    overwrites ``mlx_whisper/timing.py`` inside the bundle with a
    raise-stub that does not import scipy. But the **graph was already
    built** with the original file in view, and the copy step that
    follows the graph builder has already pulled scipy onto disk in
    ``Contents/Resources/lib/python3.13/scipy/``.

    Confirmed at runtime: scipy is never imported by any module in
    the import chain (verified with ``sys.meta_path`` blocker →
    5/5 selftest PASS; also confirmed with ``scipy/`` physically
    removed from the source venv while running selftest).

    The fix
    -------

    Patch the source venv in-place **before** setup() runs:

    1. Save each file we are about to overwrite to ``<path>.bak``.
    2. Copy our stub from ``assets/_bundling/`` over the wheel file.
    3. Clear ``__pycache__/`` for the patched packages so the next
       ``import`` resolves to the new source.
    4. Return a ``{original_path: backup_path}`` map so the caller can
       restore the originals in a ``finally:`` clause — the venv must
       be left in a state where ``uv run dictate-mac`` (and the
       ``selftest`` in particular) still works.

    The post-build ``_install_silero_vad_stub()`` and
    ``_install_timing_stub()`` still run as belt-and-braces — they
    copy the same stubs into the *bundle* regardless of what the
    source venv now contains.
    """
    import site

    try:
        site_packages = Path(site.getsitepackages()[0])
    except IndexError:  # pragma: no cover — running outside a venv
        return {}

    stubs_dir = PROJECT_ROOT / "assets" / "_bundling"
    if not stubs_dir.is_dir():
        return {}

    saved: dict[Path, Path] = {}

    pairs = [
        # venv_relpath                stub_subpath (relative to stubs_dir)
        ("mlx_whisper/timing.py",     "mlx_whisper/timing.py"),
        ("silero_vad/__init__.py",    "silero_vad/__init__.py"),
        ("silero_vad/model.py",       "silero_vad/model.py"),
        ("silero_vad/utils_vad.py",   "silero_vad/utils_vad.py"),
    ]

    for venv_rel, stub_rel in pairs:
        target = site_packages / venv_rel
        if not target.exists():
            print(f"  pre-patch: {target} not in venv — skipping")
            continue
        stub_source = stubs_dir / stub_rel
        if not stub_source.is_file():
            print(f"  pre-patch: {stub_source} not found — skipping")
            continue

        backup = target.with_suffix(target.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(target, backup)
        shutil.copyfile(stub_source, target)
        saved[target] = backup
        print(f"  pre-patched venv {venv_rel} with stub")

    # Drop .pyc caches so the next import sees the new source.
    for sub in ("mlx_whisper", "silero_vad"):
        pycache = site_packages / sub / "__pycache__"
        if pycache.is_dir():
            shutil.rmtree(pycache)

    return saved


def _restore_source_venv(saved: dict[Path, Path]) -> None:
    """Reverse of :func:`_pre_patch_source_venv`.

    Called in a ``finally:`` clause from :func:`main` so the venv is
    left intact regardless of whether ``setup()`` succeeded. We copy
    the backup back, then unlink the ``.bak`` so we never accumulate
    stale backups across rebuilds. ``__pycache__/`` is also cleared
    so subsequent ``uv run dictate-mac`` invocations see the restored
    source rather than a stale compiled ``.pyc``.
    """
    for original, backup in saved.items():
        try:
            shutil.copyfile(backup, original)
        finally:
            try:
                backup.unlink()
            except FileNotFoundError:
                pass
        pycache = original.parent / "__pycache__"
        if pycache.is_dir():
            shutil.rmtree(pycache)
        print(f"  restored venv {original.relative_to(original.parent.parent)}")


def _extract_native_runtime_libs(app_dist: Path) -> None:
    """Reconcile on-disk packages and python313.zip for runtime use.

    py2app's default mode keeps most ``.pyc`` inside ``python313.zip``
    while also mirroring some packages as directories on disk. For
    packages that ship a native dylib (``_sounddevice_data`` carrying
    ``libportaudio.dylib``) the dylib must live on the filesystem —
    ``dlopen()`` cannot read inside the zip.

    Worse, packages that appear *both* in the zip AND on disk (e.g.
    the namespace package ``mlx`` — the dir has ``core.so``, the zip
    has the rest) confuse the import machinery: Python treats the
    zip's ``__init__.pyc`` as the package definition and never looks
    inside ``mlx/__pycache__`` or finds the on-disk ``core.so``. The
    result is ``ModuleNotFoundError: No module named 'mlx.core'``.

    This function:

    1. Copies ``_sounddevice_data`` (and any other native-lib carrier)
       from the zip to real directories.
    2. Removes ALL zip entries for ``mlx/`` so the namespace package
       is only visible through the on-disk directory (which has the
       ``core.cpython-313-darwin.so`` we need).
    """
    import shutil
    import tempfile
    import zipfile

    lib_root = app_dist / "Contents" / "Resources" / "lib"
    candidates = [
        lib_root / "python3.13" / "python313.zip",
        lib_root / "python313.zip",
    ]
    zip_path = next((p for p in candidates if p.exists()), None)
    if zip_path is None:
        print(
            "  no python313.zip found — skipping extract step"
        )
        return

    extract_root = zip_path.parent

    # (prefix, mode) — 'extract' copies the package out of the zip;
    # 'remove' deletes the zip entries entirely (package stays where
    # py2app put it on disk).
    actions = {
        # sounddevice's _sounddevice_data package carries
        # libportaudio.dylib — dlopen cannot read inside the zip.
        "_sounddevice_data/": "extract",
        # mlx is shipped as a PEP 420 namespace package split between
        # the zip and the on-disk mirror — the disk copy contains the
        # C extension ``core.cpython-313-darwin.so``. Keeping the zip
        # copy too causes Python to bind to the zip version and miss
        # the .so. Drop the zip side entirely.
        "mlx/": "remove",
        "mlx-0.32.0.dist-info/": "remove",
        "mlx_metal-0.32.0.dist-info/": "remove",
        # scipy.optimize's __init__ performs a final submodule import
        # that breaks when invoked from inside the ZipImporter path —
        # the loader-time ``__getattr__`` chain returns a partial
        # module. The on-disk mirror is a real directory tree, so we
        # delete the zip copy to force Python through it.
        "scipy/": "remove",
        "scipy-": "remove",
        "numpy-": "remove",
        # numba triggers NumPy's runtime version check and the same
        # "partially initialized module" failure mode from the zip
        # path; identical fix.
        "numba/": "remove",
        "numba-": "remove",
        # The silero_vad stub is installed on disk only (see
        # _install_silero_vad_stub). Keeping any of the upstream
        # wheel's .pyc copies inside the zip would let ZipImporter
        # shadow our stub and re-introduce the torch dependency.
        "silero_vad/": "remove",
        "silero_vad-": "remove",
        # The mlx_whisper.timing stub is installed on disk only (see
        # _install_timing_stub). Same rationale: ZipImporter must
        # not find a cached .pyc that imports numba.
        "mlx_whisper/": "remove",
        "mlx_whisper-": "remove",
        # onnxruntime's pybind11_state.so + libonnxruntime.dylib must
        # live on disk (dlopen() can't read into a zip), and the rest
        # of the package follows for consistency. The on-disk mirror
        # is at lib/python3.13/onnxruntime/.
        "onnxruntime/": "remove",
        "onnxruntime-": "remove",
        # Heavy transitive wheels that should never be reached now
        # that silero_vad + mlx_whisper.timing are stubbed.
        # Stripping them from the zip keeps ZipImporter from finding
        # them even if the disk strip (_strip_bundle_junk) hasn't
        # finished yet.
        "llvmlite/": "remove",
        "llvmlite-": "remove",
        "sympy/": "remove",
        "sympy-": "remove",
        "networkx/": "remove",
        "networkx-": "remove",
    }

    extracted: list[str] = []
    removed_prefixes: list[str] = []

    with zipfile.ZipFile(zip_path) as zf:
        for prefix, mode in actions.items():
            members = [n for n in zf.namelist() if n.startswith(prefix)]
            if not members:
                continue

            # Both 'extract' and 'remove' end up dropping the prefix
            # from the zip — 'extract' also writes the files out.
            removed_prefixes.append(prefix)

            if mode == "remove":
                # on-disk copy remains the source of truth
                continue

            # mode == "extract"
            for m in members:
                dest = extract_root / m
                if dest.is_symlink():
                    dest.unlink()
                elif dest.exists() and not dest.is_dir():
                    dest.unlink()
                # Clobber any py2app placeholder files left behind
                # (e.g. an empty ``_sounddevice_data`` file).
                ancestor = dest.parent
                while ancestor != extract_root:
                    if ancestor.is_file() or ancestor.is_symlink():
                        ancestor.unlink()
                    ancestor = ancestor.parent
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(m) as src, open(dest, "wb") as out:
                    out.write(src.read())
                extracted.append(str(dest))

    # Rewrite the zip without the removed entries — the ZipImporter
    # would otherwise shadow the on-disk package.
    if removed_prefixes:
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=str(zip_path.parent), suffix=".tmp.zip"
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        with zipfile.ZipFile(zip_path) as src, zipfile.ZipFile(
            tmp_path, "w", zipfile.ZIP_DEFLATED
        ) as dst:
            for item in src.infolist():
                if any(
                    item.filename.startswith(p)
                    for p in removed_prefixes
                ):
                    continue
                data = src.read(item.filename)
                dst.writestr(item, data)
        shutil.move(str(tmp_path), str(zip_path))

    if extracted:
        print(
            f"  extracted {len(extracted)} native-lib files "
            f"(first: {extracted[0]!r})"
        )
    if removed_prefixes:
        print(
            f"  removed zip entries for: {', '.join(removed_prefixes)} "
            f"(only on-disk copy remains)"
        )


def _rewrite_boot_script(app_dist: Path) -> None:
    """Patch py2app's hardcoded build-time prefix in __boot__.py.

    py2app emits ``__boot__.py`` with a literal call:

        _site_packages('<build-time venv path>', ...)

    pointing at the **build machine's** virtualenv. When the bundle
    ships to another Mac (or even runs from a different working
    directory on the same one), that venv doesn't exist — every
    module load falls back to the system Python, which doesn't have
    mlx / silero_vad / etc.

    We rewrite the call to use ``os.environ['RESOURCEPATH']`` as the
    bundle root and prepend TWO site-package directories to ``sys.path``:

    * ``lib/python3.13`` — where py2app writes most packages
      (mlx, silero_vad, numpy, scipy, numba, ...)
    * ``lib`` itself — where py2app sometimes lands native-lib
      packages (e.g. sounddevice's ``_sounddevice_data``)

    A custom helper is inlined because py2app's ``_site_packages``
    only adds ``prefix/lib/pythonX.Y/site-packages`` — it can't add
    arbitrary multiple paths in a single call.
    """
    boot = (
        app_dist
        / "Contents/Resources/__boot__.py"
    )
    if not boot.exists():
        return

    original = boot.read_text()
    pattern = re.compile(
        r"_site_packages\(\s*['\"][^'\"]+['\"]\s*,[^)]*\)"
    )
    # We inline a tiny site-packages helper: import site, then walk
    # both candidate directories under RESOURCEPATH and add them via
    # site.addsitedir (which respects .pth files). We don't need the
    # _site_packages() function at all after this.
    replacement = (
        "import os as _os, site as _site, sys as _sys\n"
        "_rp = _os.environ['RESOURCEPATH']\n"
        "_pyver = 'python{}.{}'.format(*_sys.version_info[:2])\n"
        "for _p in (_os.path.join(_rp, 'lib', _pyver), _os.path.join(_rp, 'lib')):\n"
        "    if _os.path.isdir(_p): _site.addsitedir(_p)\n"
    )
    patched = pattern.sub(replacement, original, count=1)
    if patched == original:
        print(
            "  no hardcoded prefix found in __boot__.py — skipping "
            "boot-script rewrite (assumes py2app is portable already)"
        )
        return
    boot.write_text(patched)
    print("  patched __boot__.py to use RESOURCEPATH for both "
          "lib/pythonX.Y and lib/ (catches _sounddevice_data)")


def _strip_info_plist_paths(app_dist: Path) -> None:
    """Strip build-machine paths from Info.plist and re-sign the bundle.

    py2app writes the build-time venv's ``python`` absolute path into
    ``PythonInfoDict.PythonExecutable`` (e.g.
    ``<build-time venv path>``). That string carries the developer's
    macOS username out of the build environment and into every shipped
    bundle. ``__boot__.py`` no longer consults this field — it resolves
    its interpreter via ``os.environ['RESOURCEPATH']`` — so we can
    safely overwrite the value with the ``@executable_path`` placeholder
    that ``PyRuntimeLocations`` already references.

    Modifying ``Info.plist`` after py2app sealed the bundle breaks the
    embedded code-signature's ``Info.plist`` hash. Gatekeeper then
    rejects the bundle at every ``spctl --assess`` and — more
    importantly for users — refuses TCC permission prompts at runtime,
    which manifests as a silent mic-denied state. We therefore re-sign
    the bundle with ``codesign --force --deep --sign -`` immediately
    after the patch.

    No-op (no patch, no re-sign) if the field is already clean.
    """
    import re as _re
    import subprocess

    plist = app_dist / "Contents" / "Info.plist"
    if not plist.exists():
        return

    original = plist.read_text()
    pattern = _re.compile(
        r"(<key>PythonExecutable</key>\s*<string>)(/Users/[^<\"']+)(</string>)"
    )
    replacement = (
        r"\1@executable_path/../Frameworks/Python.framework/Versions/3.13/Python\3"
    )
    patched, n = pattern.subn(replacement, original, count=1)
    if n == 0:
        print("  no PythonExecutable path leak in Info.plist — skipping")
        return
    plist.write_text(patched)
    print(
        f"  rewrote PythonExecutable in Info.plist to @executable_path "
        f"({n} occurrence)"
    )

    # The post-build patch above invalidates the embedded code-signature
    # (it hashes Info.plist contents, and we just changed them). Re-apply
    # ad-hoc signing so Gatekeeper sees a consistent bundle. Without
    # this, ``spctl --assess`` rejects the bundle, and TCC silently
    # denies permission prompts at runtime — see AGENTS.md §5.
    try:
        subprocess.run(
            [
                "codesign",
                "--force",
                "--deep",
                "--sign", "-",
                str(app_dist),
            ],
            check=True,
            capture_output=True,
        )
        print(f"  re-signed {app_dist.name} with ad-hoc signature "
              f"(Gatekeeper-consistent)")
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        print(f"  WARNING: codesign re-sign failed: {stderr}")


def _install_torchaudio_stub(app_dist: Path) -> None:
    """Replace the bundled torchaudio package with a stub.

    The real torchaudio wheel ships a C extension
    (``_torchaudio.abi3.so``) whose ``libtorchaudio`` binary fails to
    ``dlopen`` under our bundle — its ``install_name`` chain
    references wheel dependencies that py2app doesn't bundle. We don't
    actually need torchaudio at runtime: we feed raw ``numpy.ndarray``
    from PortAudio into silero-vad, which only uses
    ``torchaudio.read_audio`` / ``save_audio`` / ``transforms.Resample``
    for *file* I/O that we never invoke. ``mlx_whisper`` doesn't
    touch torchaudio at all.

    Workaround: install a stub ``torchaudio`` package that exports
    the attribute paths silero-vad looks up at module top-level,
    with no-op fallbacks where possible.

    Drop the stub directory if the user later needs torchaudio-backed
    audio I/O in the bundle.
    """
    target_dir = (
        app_dist
        / "Contents/Resources/lib/python3.13/torchaudio"
    )
    # py2app wrote the real wheel here — wipe it.
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "__init__.py").write_text(
        '''"""Stub torchaudio for the bundled DictateMac.

silero-vad imports ``torchaudio`` at module top-level but only uses
parts of it (``read_audio``, ``save_audio``, ``transforms.Resample``)
that we never call — silero-vad's VAD pipeline operates on raw
``numpy.ndarray`` samples, not torchaudio tensors.

The real torchaudio wheel ships a ``libtorchaudio`` binary whose
``install_name`` chain references wheel dependencies that py2app
doesn't bundle, so it fails to ``dlopen`` under our bundle. Rather
than patch the wheel's C-level linking, we replace the torchaudio
import with this stub. The stub exposes the attribute paths
silero-vad and mlx_whisper look up, returning no-ops that keep them
happy.

If the user later needs torchaudio-backed audio I/O, drop this
directory and let the real wheel take over.
"""

__version__ = "2.9.0+stub"


class _NoOp:
    """Stand-in callable/object that swallows every attribute access."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return _NoOp()

    def __call__(self, *args, **kwargs):
        return _NoOp()


def load(*args, **kwargs):
    raise RuntimeError(
        "torchaudio is stubbed in this DictateMac.app — file/audio I/O "
        "via torchaudio.load is not available. Use the venv-installed "
        "build for full audio I/O."
    )


def save(*args, **kwargs):
    raise RuntimeError("torchaudio is stubbed — see torchaudio.load docstring")


def list_audio_backends(*args, **kwargs):
    return ["stub"]


sox_effects = _NoOp()
transforms = _NoOp()
compliance = _NoOp()
datasets = _NoOp()
models = _NoOp()
pipelines = _NoOp()
IOStream = _NoOp
'''
    )
    print(
        f"  installed torchaudio stub at {target_dir} "
        "(silero-vad/mlx_whisper only — file I/O not supported)"
    )


def _install_silero_vad_stub(app_dist: Path) -> None:
    """Replace the bundled ``silero_vad`` package with a numpy+onnxruntime stub.

    The upstream wheel's ``silero_vad.utils_vad`` has top-level
    ``import torch`` and ``import torchaudio`` and ends up transitively
    pulling ``libtorch_cpu.dylib`` (~263 MB on disk) into the bundle
    even though dictation only invokes ``silero_vad.load_silero_vad``
    (which defaults to ``onnx=False`` → the JIT path → ``torch.jit.load``).

    We replace ``silero_vad/{__init__,utils_vad,model}.py`` with a pure
    numpy + onnxruntime implementation that targets the ONNX model
    variant only (``silero_vad.onnx``). The replacement is loaded from
    ``assets/_bundling/silero_vad/*`` so the upstream wheel continues to
    be used by the venv install — only the bundle gets the stub.

    After this step:

    * Bundle's ``import silero_vad`` → our stub (no torch, no torchaudio).
    * Bundle's ``from silero_vad import load_silero_vad`` / ``get_speech_timestamps``
      keep the same call signatures as upstream.
    * ``silero_vad.data`` still ships ``silero_vad.onnx`` (one model,
      ~2 MB); the JIT model + extra ONNX variants are stripped via
      ``_strip_bundle_junk()``.

    Caveat: ``vad_forward`` (the streaming per-chunk variant) calls
    ``model(chunk, sr)`` for each 512-sample window at 16 kHz. Our
    stub returns a numpy ndarray whose ``.item()`` returns the speech
    probability, exactly matching upstream's behavior.
    """
    target_dir = (
        app_dist
        / "Contents/Resources/lib/python3.13/silero_vad"
    )
    if not target_dir.exists():
        print(
            "  no bundled silero_vad/ — skipping stub install "
            "(may have been excluded from packages)"
        )
        return

    stub_dir = (
        PROJECT_ROOT / "assets" / "_bundling" / "silero_vad"
    )
    if not stub_dir.is_dir():
        print(f"  WARNING: {stub_dir} not found; skipping silero_vad stub")
        return

    # Replace the .py files; keep the on-disk data/ directory as-is so
    # importlib.resources can still find silero_vad.onnx.
    written: list[str] = []
    for src in sorted(stub_dir.glob("*.py")):
        dst = target_dir / src.name
        shutil.copyfile(src, dst)
        written.append(dst.name)

    # Drop any .py files that are part of the upstream wheel but not
    # part of our stub (tinygrad_model.py — silero-vad's alternate
    # inference backend we don't call).
    stub_filenames = {p.name for p in stub_dir.glob("*.py")}
    for entry in list(target_dir.iterdir()):
        if (
            entry.is_file()
            and entry.suffix == ".py"
            and entry.name not in stub_filenames
        ):
            entry.unlink()

    # Strip pyc cache so the bundle imports our stub instead of any
    # cached .pyc from the wheel.
    pycache = target_dir / "__pycache__"
    if pycache.is_dir():
        shutil.rmtree(pycache)
    utils_pycache = target_dir / "utils_vad" / "__pycache__"
    if utils_pycache.is_dir():  # pragma: no cover
        shutil.rmtree(utils_pycache)

    print(
        f"  installed silero_vad stub at {target_dir} "
        f"(files: {', '.join(written)}; no torch/torchaudio required)"
    )


def _install_timing_stub(app_dist: Path) -> None:
    """Replace the bundled ``mlx_whisper.timing`` with a raise-stub.

    The real ``timing.py`` has top-level ``import numba`` and
    ``from scipy import signal``. Even though we never call
    ``add_word_timestamps`` (passes ``word_timestamps=False`` to
    ``mlx_whisper.transcribe``), the top-level ``from .timing import
    add_word_timestamps`` in ``mlx_whisper.transcribe`` causes
    ``modulegraph`` to drag numba + scipy + llvmlite + sympy into the
    bundle (~265 MB).

    The stub replaces the file with a single ``add_word_timestamps``
    function that raises. After this step the import succeeds and the
    dependency chain is broken.
    """
    target = (
        app_dist
        / "Contents/Resources/lib/python3.13/mlx_whisper/timing.py"
    )
    if not target.exists():
        print(
            "  bundled mlx_whisper/timing.py not found — skipping "
            "timing stub install (likely excluded from packages)"
        )
        return

    stub_path = (
        PROJECT_ROOT
        / "assets"
        / "_bundling"
        / "mlx_whisper"
        / "timing.py"
    )
    if not stub_path.is_file():
        print(f"  WARNING: {stub_path} not found; skipping timing stub")
        return

    shutil.copyfile(stub_path, target)

    # Drop the cached .pyc so the interpreter sees the new file.
    parent = target.parent / "__pycache__"
    cached = parent / "timing.cpython-313.pyc" if parent.is_dir() else None
    if cached is not None and cached.exists():
        cached.unlink()

    print(
        f"  installed mlx_whisper.timing stub at {target} "
        "(no numba/scipy/llvmlite needed; word_timestamps unsupported)"
    )


def _strip_bundle_junk(app_dist: Path) -> None:
    """Drop directories that py2app shipped but the runtime does not need.

    The bundle is built around the strict subset of dependencies
    ``dictate_mac.audio`` actually invokes (numpy, sounddevice,
    onnxruntime-backed silero_vad, mlx-whisper-on-mlx, huggingface_hub,
    pyobjc-framework-{Quartz, Cocoa}, rumps). Anything else that
    modulegraph pulled in via transitive imports is fair game.

    Targets removed (relative to ``Contents/Resources/lib/python3.13/``
    unless noted otherwise):

    * ``torch/`` sub-packages (``_inductor``, ``_dynamo``,
      ``distributed``, ``bin``, ``testing``, ``fx``, ``ao``, ``optim``,
      ``onnx``, ``include``, JIT ``__pycache__``, ``jit``,
      ``autograd``, ``backends``, ``signal``, ``utils``) plus
      ``torch-2.13.0.dist-info/`` — only the C-extension entry points
      (``torch/__init__.py``, ``torch/_C``, ``libtorch_cpu.dylib``)
      are needed if anything tries to ``import torch``; we strip them
      entirely because the silero_vad stub no longer touches torch.

    * ``scipy/optimize/`` and the entire ``sympy*/`` tree — pulled by
      mlx_whisper.timing before the stub broke that chain. The
      rest of scipy is kept (other code paths may still reference it).
      If a post-stub import fails because scipy.optimize is missing,
      remove this strip instead.

    * ``numba/`` and its dist-info — same reason as scipy.optimize.
      ``numba`` ships with the actually-needed llvmlite at runtime,
      so the two go together.

    * ``llvmlite/`` and its dist-info — pulled in via the numba wheel.

    * ``networkx/`` and its dist-info — pulled in via huggingface_hub
      / hf_xet for graph operations the daemon never invokes.

    * ``onnxruntime/transformers``, ``onnxruntime/quantization``,
      ``onnxruntime/tools``, ``onnxruntime/backend``,
      ``onnxruntime/datasets`` — extra onnxruntime tools that the
      runtime does not call into.

    * All ``*.dSYM/`` directories — Apple-style debug symbols shipped
      by some wheels; not needed at runtime.

    * Python's stdlib ``test/`` directory from ``python313.zip`` —
      regression test data; never imported by app code.

    * Silero-vad model variants we don't use:
      ``silero_vad.jit``, ``silero_vad_16k.safetensors``,
      ``silero_vad_16k_op15.onnx``, ``silero_vad_half.onnx``,
      ``silero_vad_op18_ifless.onnx``. We keep ``silero_vad.onnx``.

    * Top-level duplicates:
      ``mlx.metallib``, ``libmlx.dylib``, ``libjaccl.dylib`` that
      py2app also drops into ``Contents/Resources/`` as resources (we
      do not strip these — both copies are needed because the
      on-disk mirror at ``lib/python3.13/mlx/lib/`` is what
      ``mlx.core`` resolves via the DYLD search path).

    The post-build step is idempotent: missing targets are silently
    skipped.
    """
    lib_root = app_dist / "Contents" / "Resources" / "lib" / "python3.13"
    zip_candidates = [
        app_dist / "Contents" / "Resources" / "lib" / "python3.13" / "python313.zip",
        app_dist / "Contents" / "Resources" / "lib" / "python313.zip",
    ]
    zip_path = next((p for p in zip_candidates if p.exists()), None)

    # (lib_root-relative path, description) — directories to remove
    # in-place. None means "doesn't exist; skip silently".
    disk_deletions = [
        # torch — entire tree once silero_vad no longer references it.
        # py2app split torch between zip and disk originally; we keep
        # a no-op entry-point if anything imports torch by accident,
        # but we strip what's still around at this point.
        ("torch/_inductor", "torch JIT/AOT compiler (inductor)"),
        ("torch/_dynamo", "torch dynamo tracer"),
        ("torch/distributed", "torch.distributed"),
        ("torch/bin", "torch C++ binaries"),
        ("torch/testing", "torch.testing"),
        ("torch/fx", "torch FX graph tracer"),
        ("torch/ao", "torch.ao (quantization)"),
        ("torch/optim", "torch.optim"),
        ("torch/onnx", "torch.onnx exporter"),
        ("torch/include", "torch C++ headers"),
        ("torch/__pycache__", "torch bytecode cache"),
        ("torch/jit", "torch.jit (already unreachable)"),
        ("torch/autograd", "torch.autograd"),
        ("torch/backends", "torch.backends"),
        ("torch/signal", "torch.signal"),
        ("torch/utils/benchmark", "torch.utils.benchmark"),
        ("torch/utils/checkpoint", "torch.utils.checkpoint"),
        ("torch/utils/data", "torch.utils.data"),
        ("torch/utils/deterministic", "torch.utils.deterministic"),
        ("torch/utils/flop_counter", "torch.utils.flop_counter"),
        ("torch/utils/_sympy", "torch sympy helpers"),
        ("torch/utils/_pytree", "torch pytree helpers (NUMA/CCP only)"),
        ("torch/futures", "torch.futures"),
        ("torch/_higher_order_ops", "torch higher-order ops"),
        ("torch/_inductor/codegen", "torch.inductor codegen"),
        ("torch/_inductor/codegen/cuda", "torch.inductor CUDA codegen"),
        # scipy — Phase 14 audit confirmed the runtime never imports
        # it (the source-venv pre-patch in main() cuts the graph at
        # mlx_whisper.timing before py2app scans; verified via
        # meta_path blocker → 5/5 selftest PASS). Strip every
        # submodule so a stray reference still resolves cleanly
        # without dragging the whole 68 MB tree in. The whole
        # package is then wiped by the `scipy` entry below — listed
        # per-submodule so the strip report shows the breakdown.
        ("scipy/optimize", "scipy.optimize (triggers sympy)"),
        ("scipy/stats", "scipy.stats (distributions, unused)"),
        ("scipy/sparse", "scipy.sparse (sparse matrices, unused)"),
        ("scipy/special", "scipy.special (Bessel/gamma, unused)"),
        ("scipy/linalg", "scipy.linalg (LAPACK wrappers, unused)"),
        ("scipy/spatial", "scipy.spatial (kd-tree, unused)"),
        ("scipy/signal", "scipy.signal (signal processing, unused)"),
        ("scipy/io", "scipy.io (matlab/wavfile, unused)"),
        ("scipy/interpolate", "scipy.interpolate (unused)"),
        ("scipy/integrate", "scipy.integrate (ODE, unused)"),
        ("scipy/ndimage", "scipy.ndimage (image processing, unused)"),
        ("scipy/fft", "scipy.fft (FFTW bindings, unused)"),
        ("scipy/fftpack", "scipy.fftpack (legacy FFT, unused)"),
        ("scipy/cluster", "scipy.cluster (k-means, unused)"),
        ("scipy/constants", "scipy.constants (physical consts, unused)"),
        ("scipy/odr", "scipy.odr (orthogonal dist regression, unused)"),
        ("scipy/datasets", "scipy.datasets (test data)"),
        ("scipy/differentiate", "scipy.differentiate (finite-diff, unused)"),
        ("scipy/misc", "scipy.misc (deprecated module)"),
        ("scipy/_external", "vendored numpy/scipy helpers"),
        # Catch-all: removes scipy/__init__.py, scipy/__config__.py,
        # scipy/_cyutility.cpython-313-darwin.so, scipy/_lib/, etc.,
        # AND any scipy subdirectory not enumerated above.
        ("scipy", "scipy (catch-all: __init__/__config__/_lib/_external)"),
        # scipy also ships libgfortran / libquadmath / libgcc_s under
        # scipy/.dylibs/ — Fortran runtime pulled by scipy.linalg and
        # scipy.sparse. With the whole tree gone, the dylibs are dead
        # weight.
        ("scipy/.dylibs", "scipy Fortran runtime (libgfortran/libquadmath)"),
        # sympy — gone after scipy.optimize is gone.
        # (handled via disk + zip wildcards below)
        # numba — entire package (we never JIT-compile anything).
        ("numba", "numba (llvmlite-backed JIT)"),
        # llvmlite — paired with numba.
        ("llvmlite", "llvmlite (LLVM bindings for numba)"),
        # networkx — pulled by hf_xet for graph ops.
        ("networkx", "networkx (graph library; hf_xet uses it)"),
        # onnxruntime sub-tools we don't call.
        ("onnxruntime/transformers", "onnxruntime.transformers"),
        ("onnxruntime/quantization", "onnxruntime.quantization"),
        ("onnxruntime/tools", "onnxruntime.tools"),
        ("onnxruntime/backend", "onnxruntime.backend"),
        ("onnxruntime/datasets", "onnxruntime.datasets"),
        # numpy — Phase 14 second pass: drop the test directories,
        # the f2py Fortran wrapper generator, the .pyi typing stubs
        # (only used by static type-checkers), and the random
        # _examples/ directory which contains numba/cython/cffi
        # example kernels we never invoke. Runtime imports nothing
        # here; selftest still 5/5 PASS after stripping.
        ("numpy/_core/tests", "numpy._core.tests (3.6 MB pytest data)"),
        ("numpy/lib/tests", "numpy.lib.tests (852 KB pytest data)"),
        ("numpy/f2py", "numpy.f2py (1.8 MB Fortran wrapper generator)"),
        ("numpy/random/_examples", "numpy random example kernels"),
        # numpy also ships C headers and a static libnpymath.a — both
        # needed only for downstream C extensions building against
        # numpy. Our bundle has no such build step.
        ("numpy/_core/include", "numpy C headers (.h) for downstream"),
        # numpy typing stubs (.pyi) — these are static type-hint stubs
        # for downstream C extensions; they are consumed ONLY by
        # static type-checkers (mypy, pyright), never by the runtime.
        # Distributed across ``numpy/{typing,_typing,char,strings,
        # core,rec,polynomial,ma,matrixlib,ctypeslib,linalg,fft,
        # random,testing}/...pyi``. Strip each individually so the
        # strip report shows the breakdown.
        #
        # Runtime notes (verified by deleting each dir from a copy
        # of the bundle and running selftest):
        # - ``numpy/_utils`` MUST stay (numpy._globals imports
        #   ``set_module`` from it at import time).
        # - ``numpy/_typing`` MUST stay (numpy.linalg imports it).
        # - ``numpy/matrixlib`` MUST stay (loaded transitively by
        #   numpy.lib via numpy.matrixlib.defmatrix).
        # All other submodules below are strippable.
        ("numpy/typing", "numpy.typing (1.0 MB .pyi stubs)"),
        ("numpy/strings", "numpy.strings (stubs)"),
        ("numpy/char", "numpy.char (legacy + stubs)"),
        ("numpy/core", "numpy.core (legacy compat shim + stubs)"),
        ("numpy/rec", "numpy.rec (record arrays + stubs)"),
        ("numpy/ctypeslib", "numpy.ctypeslib (stubs + ctypes helper)"),
        ("numpy/dtypes", "numpy.dtypes (compatibility alias)"),
        ("numpy/exceptions", "numpy.exceptions (compatibility alias)"),
        ("numpy/ma", "numpy.ma (masked arrays — not used)"),
        ("numpy/polynomial", "numpy.polynomial (not used)"),
        # All *.pyi stubs across the bundle — none are needed at runtime.
        # We can't list every directory individually here because
        # ``_strip_bundle_junk()`` treats this list as path prefixes;
        # the catch-all glob ``**/*.pyi`` is handled below in
        # ``_strip_pyi_files`` (a separate post-build step).
        # mlx ships a heavy C++ Metal header tree (~3.9 MB) used only
        # when building C extensions against mlx. Runtime never
        # touches it; the actual ``mlx.core`` C extension loads via
        # ``dlopen`` of ``libmlx.dylib`` + ``mlx.metallib``.
        ("mlx/include", "mlx C++ headers (3.9 MB, build-only)"),
        # mlx ships a heavy C++ Metal header tree (~3.9 MB) used only
        # when building C extensions against mlx. Runtime never
        # touches it; the actual ``mlx.core`` C extension loads via
        # ``dlopen`` of ``libmlx.dylib`` + ``mlx.metallib``.
        ("mlx/include", "mlx C++ headers (3.9 MB, build-only)"),
        # Quartz — Quartz/CoreGraphics is the only Quartz submodule our
        # code calls (CGEvent tap + CGEventKeyboardSetUnicodeString).
        # DO NOT strip CoreVideo, PDFKit, or ImageKit: their modules
        # are imported by ``Quartz.__init__`` for namespace export
        # resolution and cause a "circular import" error at
        # ``from Quartz import ...`` time if any of them is missing.
        # The ~530 KB cost is unavoidable.
        # lib-dynload/objc — keep. Verified during the Phase-14 second
        # pass: py2app 0.28.x does NOT copy the ``objc/`` package's
        # ``_objc.cpython-313-darwin.so`` into the bundle at all; the
        # only copy of the C extension ends up under
        # ``lib-dynload/objc/_objc.so``. When the bundle's hotkey tap
        # imports Quartz, ``Quartz.CoreGraphics._metadata`` does
        # ``import objc`` which transitively triggers
        # ``import objc._objc`` — that lookup ONLY succeeds via the
        # lib-dynload/objc/ path. Removing it produces
        # ``'objc/_objc.so' not found`` at runtime and the tap
        # thread dies. Keep the 1.6 MB.
        # mlx_whisper — drop three modules that the bundle's runtime
        # path does not call: torch_whisper (alternative torch
        # backend), writers (SRT/VTT/TSV), cli (`python -m
        # mlx_whisper`). Selftest 5/5 PASS after stripping.
        ("mlx_whisper/torch_whisper.py", "mlx_whisper torch backend"),
        ("mlx_whisper/writers.py", "mlx_whisper SRT/VTT writers"),
        ("mlx_whisper/cli.py", "mlx_whisper CLI"),
        # huggingface_hub — drop dead subtrees we don't call at
        # runtime. Verified safe: our import path (snapshot_download)
        # only uses hf_api / file_download / _snapshot_download /
        # utils.* / _local_folder / _buckets / _commit_api /
        # _upload_pipeline / _upload_large_folder / repocard* (the
        # last six are imported indirectly by _snapshot_download at
        # module load, even though we never *call* them). The drop
        # list below removes only confirmed-unused subtrees:
        # CLI, Inference API, Hub serialization helpers, the
        # user-class Mixin.
        ("huggingface_hub/cli", "huggingface-cli (terminal tool)"),
        ("huggingface_hub/inference", "HF Inference API client"),
        ("huggingface_hub/serialization", "HF save/load model helpers"),
        ("huggingface_hub/hub_mixin.py", "user-class Mixin (we don't subclass)"),
    ]

    # Top-level dist-info directories to drop. Matched by prefix.
    disk_distinfo_prefixes = (
        "torch-2.13.0.dist-info",
        "torchgen-",
        "numba-",
        "llvmlite-",
        "sympy-",
        "scipy-",
        "networkx-",
        "onnxruntime-",
    )

    # Silero-vad models we don't use. Keep only silero_vad.onnx.
    # py2app writes the data package in TWO places:
    # (a) `silero_vad.data/` as a sibling of `silero_vad/` (PEP-style
    #     namespace data package — the format importlib.resources uses).
    # (b) `silero_vad/data/` as a subdirectory of the silero_vad package
    #     (the format the wheel's __init__.py uses for `impresources.files`).
    # Both must be stripped.
    silero_data_dirs = [
        lib_root / "silero_vad.data",
        lib_root / "silero_vad" / "data",
    ]
    silero_models_to_remove = [
        "silero_vad.jit",
        "silero_vad_16k.safetensors",
        "silero_vad_16k_op15.onnx",
        "silero_vad_half.onnx",
        "silero_vad_op18_ifless.onnx",
    ]



    # python313.zip top-level prefixes to drop. We rewrite the zip
    # to drop these so ZipImporter doesn't shadow the on-disk versions.
    zip_remove_prefixes = [
        "test/",
        "objc/_objc.cpython-313-darwin.so.dSYM/",
        "mpmath/",
        "sympy",
        "networkx",
        "numba",
        "llvmlite",
        "scipy",
        "torch",
    ]

    # Same idea as the on-disk __pycache__ sweep above: the .pyc files
    # inside the zip also carry the developer's build path in their
    # headers. Match by suffix so we catch every package's __pycache__
    # (numpy, huggingface_hub, mlx, Foundation, …).
    zip_remove_suffixes = ("/__pycache__/",)

    removed_disk: list[tuple[str, int]] = []  # (label, byte_count)
    removed_zip: list[str] = []

    if not lib_root.is_dir():
        print(f"  expected lib dir missing: {lib_root} — skipping strip")
    else:
        # (1) disk directories
        for relpath, label in disk_deletions:
            target = lib_root / relpath
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.is_dir():
                size = sum(
                    f.stat().st_size for f in target.rglob("*") if f.is_file()
                )
                shutil.rmtree(target)
                removed_disk.append((label, size))
            # else: missing — skip silently

        # (1b) __pycache__/ directories: py2app compiles modules during
        #      the build and leaves the .pyc files in place. Each .pyc
        #      embeds the *build machine's* source path in its header
        #      (``/Users/<name>/.venv/lib/python3.13/...``) — that
        #      leaks the developer's home directory into every shipped
        #      bundle. The .app's embedded Python recompiles on first
        #      import with the bundle's own paths, so dropping these
        #      caches costs nothing at runtime.
        for pycache in lib_root.rglob("__pycache__"):
            if pycache.is_dir():
                size = sum(
                    f.stat().st_size for f in pycache.rglob("*") if f.is_file()
                )
                shutil.rmtree(pycache)
                removed_disk.append(
                    (f"__pycache__ ({pycache.relative_to(lib_root)})", size)
                )

        # (2) dist-info at the top level of lib/python3.13/
        for entry in list(lib_root.iterdir()):
            if entry.is_dir() and any(
                entry.name.startswith(p) for p in disk_distinfo_prefixes
            ):
                size = sum(
                    f.stat().st_size for f in entry.rglob("*") if f.is_file()
                )
                shutil.rmtree(entry)
                removed_disk.append((f"dist-info {entry.name}", size))

        # (3) silero_vad model files we don't ship. Four locations
        #     because py2app mirrored them in different places:
        #     a) ``silero_vad.data/`` package directory (kept — our
        #        stub needs ``silero_vad.onnx`` for inference).
        #     b) ``silero_vad/data/`` subdirectory (kept — the stub's
        #        fall-back resource lookup uses this format too).
        #     c) top-level ``Contents/Resources/`` (the ``resources``
        #        mechanism drops a flat copy here).
        #     d) ``Contents/Resources/data/silero_vad/`` (the
        #        ``data_files`` mechanism drops a tree here).
        for silero_root in (
            *silero_data_dirs,
            app_dist / "Contents" / "Resources",
            app_dist / "Contents" / "Resources" / "data" / "silero_vad",
        ):
            if not silero_root.is_dir():
                continue
            for name in silero_models_to_remove:
                f = silero_root / name
                if f.exists() and f.is_file():
                    size = f.stat().st_size
                    f.unlink()
                    removed_disk.append((f"silero_vad model {name}", size))

        # (3b) Drop any top-level Resources/ duplicates of files that
        #      already live inside an on-disk Python package. These
        #      would be the result of accidentally adding wheel files
        #      to ``resources`` after py2app already wrote them into
        #      ``lib/python3.13/<pkg>/lib/`` (defense-in-depth for the
        #      mlx.metallib / libmlx.dylib / libjaccl.dylib triple).
        resources_dir = app_dist / "Contents" / "Resources"
        duplicates_to_drop = [
            "mlx.metallib",
            "libmlx.dylib",
            "libjaccl.dylib",
            # silero_vad.onnx is already at
            # ``lib/python3.13/silero_vad.data/silero_vad.onnx`` (used
            # at runtime by our stub). The top-level copy and the
            # ``data/silero_vad/`` mirror copy were dropped here as
            # belt-and-braces; they cost 4.4 MB.
            "silero_vad.onnx",
        ]
        for name in duplicates_to_drop:
            f = resources_dir / name
            if f.exists() and f.is_file():
                size = f.stat().st_size
                f.unlink()
                removed_disk.append((f"top-level {name}", size))

        # (3c) Drop the secondary silero_vad model copy under
        #      ``Contents/Resources/data/silero_vad/silero_vad.onnx``
        #      that ``data_files`` mirrored alongside the ``lib/`` copy
        #      (the lib/ copy is the one our stub reads).
        secondary_silero = resources_dir / "data" / "silero_vad" / "silero_vad.onnx"
        if secondary_silero.exists() and secondary_silero.is_file():
            size = secondary_silero.stat().st_size
            secondary_silero.unlink()
            removed_disk.append(("secondary silero_vad.onnx mirror", size))

        # (3d) Drop Resources/include (Python C headers — only needed
        #      for compiling C extensions against the bundled Python;
        #      runtime never touches them) and Resources/openssl.ca
        #      (a CA-bundle shipped by py2app's standard recipe; the
        #      OpenSSL that ships in Frameworks/ provides its own
        #      cert store, and libssl.dylib uses the system trust store
        #      by default on macOS — verified by curl-style tests
        #      through libssl during selftest's HTTPS model download).
        resources_include = resources_dir / "include"
        if resources_include.is_dir():
            size = sum(
                f.stat().st_size for f in resources_include.rglob("*")
                if f.is_file()
            )
            shutil.rmtree(resources_include)
            removed_disk.append(("Resources/include (Python C headers)", size))

        # (3e) __pycache__/ under Contents/Resources/ (not under
        #      lib/python3.13/ — that's handled in (1b) above). py2app
        #      also drops a small number of compiled .pyc files at
        #      top-level resources paths (e.g. the silero-vad data
        #      mirror) and these carry the same build-machine path
        #      leak as the lib/ copies.
        for pycache in resources_dir.rglob("__pycache__"):
            if pycache.is_dir():
                size = sum(
                    f.stat().st_size for f in pycache.rglob("*") if f.is_file()
                )
                shutil.rmtree(pycache)
                removed_disk.append(
                    (f"Resources __pycache__ ({pycache.relative_to(resources_dir)})", size)
                )
        # And any stragglers — top-level __init__.cpython-313.pyc at
        # Contents/Resources/ and Contents/Resources/data/silero_vad/
        # that py2app created when it dropped those packages there.
        for leftover in (
            resources_dir / "__init__.cpython-313.pyc",
            resources_dir / "data" / "silero_vad" / "__init__.cpython-313.pyc",
        ):
            if leftover.is_file():
                size = leftover.stat().st_size
                leftover.unlink()
                removed_disk.append((str(leftover.relative_to(app_dist)), size))

        resources_openssl = resources_dir / "openssl.ca"
        if resources_openssl.exists() and (
            resources_openssl.is_dir() or resources_openssl.is_file()
        ):
            if resources_openssl.is_dir():
                size = sum(
                    f.stat().st_size for f in resources_openssl.rglob("*")
                    if f.is_file()
                )
                shutil.rmtree(resources_openssl)
            else:
                size = resources_openssl.stat().st_size
                resources_openssl.unlink()
            removed_disk.append(("Resources/openssl.ca (CA-bundle)", size))

        # (4) python313.zip rewrite — drop the stdlib test/ and
        # transitive deps that we no longer need (the rest of the
        # zip is preserved, including onnxruntime wheels).
        if zip_path is not None:
            try:
                import tempfile
                import zipfile

                tmp_fd, tmp_name = tempfile.mkstemp(
                    dir=str(zip_path.parent),
                    suffix=".tmp.zip",
                )
                import os as _os
                _os.close(tmp_fd)
                tmp_path = Path(tmp_name)
                counts = {p: 0 for p in zip_remove_prefixes}
                counts_suffix = {s: 0 for s in zip_remove_suffixes}
                bytes_saved = 0
                with zipfile.ZipFile(zip_path) as src, zipfile.ZipFile(
                    tmp_path, "w", zipfile.ZIP_DEFLATED
                ) as dst:
                    for item in src.infolist():
                        prefix_matched = next(
                            (
                                p
                                for p in zip_remove_prefixes
                                if item.filename.startswith(p)
                            ),
                            None,
                        )
                        suffix_matched = next(
                            (
                                s
                                for s in zip_remove_suffixes
                                if item.filename.endswith(s)
                            ),
                            None,
                        )
                        if prefix_matched is None and suffix_matched is None:
                            dst.writestr(item, src.read(item.filename))
                        elif prefix_matched is not None:
                            counts[prefix_matched] += 1
                            bytes_saved += item.file_size
                        else:
                            counts_suffix[suffix_matched] += 1
                            bytes_saved += item.file_size
                shutil.move(str(tmp_path), str(zip_path))
                for prefix, count in counts.items():
                    if count:
                        removed_zip.append(f"{prefix} ({count} entries)")
                for suffix, count in counts_suffix.items():
                    if count:
                        removed_zip.append(f"*{suffix} ({count} entries)")
            except Exception as exc:  # noqa: BLE001
                print(f"  WARNING: zip rewrite failed: {exc}")

        # (5) *.dSYM/ anywhere under lib/. Debug symbols are useless
        #     at runtime and add hundreds of KBs each.
        for dsym in lib_root.rglob("*.dSYM"):
            if dsym.is_dir():
                size = sum(
                    f.stat().st_size
                    for f in dsym.rglob("*")
                    if f.is_file()
                )
                shutil.rmtree(dsym)
                removed_disk.append((f"dSYM {dsym.relative_to(lib_root)}", size))

        # (5b) *.pyi stubs anywhere under lib/. These are static
        #      type-hint files consumed only by mypy / pyright; the
        #      Python interpreter never reads them. They are scattered
        #      across every wheel (numpy, huggingface_hub, etc.) and
        #      collectively add ~3 MB of dead weight.
        pyi_count = 0
        pyi_bytes = 0
        for pyi in lib_root.rglob("*.pyi"):
            if pyi.is_file():
                pyi_bytes += pyi.stat().st_size
                pyi.unlink()
                pyi_count += 1
        if pyi_count:
            removed_disk.append(
                (f"*.pyi stubs ({pyi_count} files)", pyi_bytes)
            )

        # (5c) *.bak/.orig backup files anywhere under lib/. These
        #      show up because ``_pre_patch_source_venv()`` writes
        #      ``<path>.bak`` backups of the wheel files it patches
        #      before setup() runs, and py2app then copies the entire
        #      patched package directory into the bundle. Runtime
        #      never reads them.
        bak_count = 0
        bak_bytes = 0
        for bak_glob in ("*.bak", "*.orig"):
            for bak in lib_root.rglob(bak_glob):
                if bak.is_file():
                    bak_bytes += bak.stat().st_size
                    bak.unlink()
                    bak_count += 1
        if bak_count:
            removed_disk.append(
                (f"*.bak/.orig backup files ({bak_count} files)", bak_bytes)
            )

    if removed_disk:
        total_disk = sum(sz for _, sz in removed_disk)
        print(
            f"  stripped {len(removed_disk)} on-disk tree(s); "
            f"~{total_disk / 1e6:.1f} MB freed:"
        )
        for label, size in removed_disk[:10]:
            print(f"    {label:<60}  {size / 1e6:7.2f} MB")
        if len(removed_disk) > 10:
            print(f"    ... and {len(removed_disk) - 10} more")
    if removed_zip:
        print(
            f"  stripped from python313.zip: {', '.join(removed_zip)}"
        )


def main() -> None:
    if not ICNS_FILE.exists():
        sys.stderr.write(
            f"{ICNS_FILE} is missing — run `./build.sh` first, or\n"
            f"`{PROJECT_ROOT / '.venv' / 'bin' / 'python'} "
            f"{PROJECT_ROOT / 'assets' / 'icon' / 'make_icon.py'}` "
            "to generate it.\n"
        )
        sys.exit(1)

    data_files, resources = _stage_runtime_files()

    OPTIONS = {
        "argv_emulation": False,
        "resources": resources,
        "iconfile": str(ICNS_FILE),
        "plist": {
            "CFBundleName": "DictateMac",
            "CFBundleDisplayName": "Dictate Mac",
            "CFBundleIdentifier": "com.local.dictate-mac",
            "CFBundleShortVersionString": "0.2.0",
            "CFBundleVersion": "0.2.0",
            "LSMinimumSystemVersion": "11.0",
            "LSUIElement": True,
            "NSHighResolutionCapable": True,
            "NSMicrophoneUsageDescription": (
                "Dictate Mac needs microphone access to recognize "
                "your speech."
            ),
            "NSAccessibilityUsageDescription": (
                "Dictate Mac synthesizes Unicode keystrokes to inject "
                "recognized text into the focused window."
            ),
            "NSHumanReadableCopyright": "© dictate-mac authors",
            "CFBundleExecutable": "DictateMac",
            "CFBundlePackageType": "APPL",
            "CFBundleSignature": "????",
        },
        "packages": [
            "dictate_mac",
            "mlx",
            "mlx_whisper",
            "silero_vad",
            "silero_vad.data",
            "sounddevice",
            "rumps",
            "AppKit",
            "Foundation",
            "Quartz",
            # onnxruntime is required by the silero_vad stub. Without
            # this entry py2app misses the C extension binary in
            # python313.zip and the stub fails at first inference.
            "onnxruntime",
            # numpy and scipy are kept as directories because the
            # zip-import path breaks their relative-import story.
            # Post-build, scipy.optimize is stripped (it transitively
            # pulls sympy); the rest of scipy is preserved for any
            # edge-case import that still lands on it.
            "numpy",
            "scipy",
        ],
        "includes": [
            "numpy",
            "huggingface_hub",
            "pyobjc",
            "importlib.metadata",
            "importlib.resources",
            "importlib_resources",
            "json",
            "threading",
            "asyncio",
            "queue",
            "ctypes",
            "objc",
        ],
        "excludes": [
            "tkinter",
            "PyQt5",
            "PyQt6",
            "PySide2",
            "PySide6",
            "IPython",
            "notebook",
            "jupyter",
            "matplotlib",
            "pytest",
            "setuptools",
            "pip",
            "wheel",
            "_pytest",
            "sphinx",
            "mkl",
            # Heavy ML frameworks we never use. The silero_vad stub
            # removes the only reason modulegraph would chase torch;
            # listing them here gives py2app a strong signal to skip
            # the scan entirely. _strip_bundle_junk() re-removes any
            # straggler sub-trees that the broader packages list
            # accidentally dragged in.
            "torch",
            "torchvision",
            "torchaudio",
            "numba",
            "llvmlite",
            "sympy",
            "networkx",
            "pandas",
            "transformers",
            "datasets",
            "PIL",
            "sklearn",
            "skimage",
            "cv2",
            "tornado",
            "zmq",
        ],
        "strip": True,
        "optimize": 1,
        "site_packages": True,
        "use_pythonpath": False,
        "compressed": False,
        "alias": False,
    }

    # Pre-patch the source venv so py2app sees our stubs while it
    # builds the dependency graph. Without this, modulegraph reads the
    # wheel's ``mlx_whisper/timing.py`` (which has
    # ``from scipy import signal`` at module top-level) and pulls scipy
    # into the bundle before we ever replace the file. See
    # ``_pre_patch_source_venv`` for the full rationale.
    saved_venv_files = _pre_patch_source_venv()
    try:
        setup(
            app=APP,
            name="DictateMac",
            data_files=data_files,
            options={"py2app": OPTIONS, "build_py": {"optimize": 1}},
        )
    finally:
        # Always restore the venv — both on success and on any
        # exception during setup() — so subsequent ``uv run`` / ``make``
        # invocations see the original wheel files.
        _restore_source_venv(saved_venv_files)

    # Post-build: pull native-lib packages out of python313.zip so
    # dlopen() can find them. Done outside setup() because setup() only
    # returns after the ``run()`` finishes, and we want this on the
    # same code path that the user invokes.
    app_dist = PROJECT_ROOT / "dist" / "DictateMac.app"
    if app_dist.exists():
        _extract_native_runtime_libs(app_dist)
        _install_torchaudio_stub(app_dist)
        _install_silero_vad_stub(app_dist)
        _install_timing_stub(app_dist)
        _strip_bundle_junk(app_dist)
        _rewrite_boot_script(app_dist)
        _strip_info_plist_paths(app_dist)


if __name__ == "__main__":
    main()
