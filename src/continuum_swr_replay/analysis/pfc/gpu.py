"""Pick a free GPU for the batched fitter, on a machine other people share.

The batched Poisson IRLS is the one GPU-shaped piece of the PFC pipeline, and
the host it runs on carries other people's jobs. Grabbing a busy card would
evict them, and JAX grabs most of a card's memory by default, so a run has to
say which card it wants and how much of it **before JAX initialises a
backend**. That is later than "before ``import jax``": the module import is
cheap and lazy, and the environment is read when the first device is asked
for. Callers should still configure as early as they can -- the margin is
narrow and invisible -- and :func:`verify_jax_device` closes it by checking
afterwards that the process really did end up on one card.

Selection reads ``nvidia-smi`` and sets environment variables; only the
verification touches JAX, and nothing here imports Spyglass.

Three rules keep it from colliding with other work:

- a card counts as free only if its used memory is under a threshold (an idle
  card still shows a few hundred MiB of other users' contexts) *and* its
  utilisation is zero;
- a chosen card is claimed with a lock file naming the claiming process, so two
  workers of the same batch starting at the same moment cannot both read the
  same card as idle and land on it. A claim whose process is gone is ignored,
  and a process that already holds a claim still holds it;
- a process configures exactly once. A worker handed a second session returns
  the card it is already on rather than selecting again -- JAX's backend is
  bound by then, so a second ``CUDA_VISIBLE_DEVICES`` would change nothing
  except what the second session's sidecar claims, while consuming a free card
  that never gets released.
"""

from __future__ import annotations

import atexit
import errno
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: A card with more than this in use is somebody's job, not a free card. An
#: idle card still reports a few hundred MiB of other users' contexts.
DEFAULT_MAX_USED_MIB = 2048

#: Fraction of the chosen card JAX may take. Modest on purpose: the batched
#: fitter's working set is a few GB, and the rest of the card stays available.
DEFAULT_MEM_FRACTION = 0.5

#: Where claims live. One file per card, holding the claiming PID.
CLAIM_DIRECTORY = Path(os.environ.get("TMPDIR", "/tmp")) / "pfc_gpu_claims"

_QUERY = "index,memory.used,memory.total,utilization.gpu,uuid,name"


class NoFreeGPUError(RuntimeError):
    """Raised when no card is free enough to fit on."""


@dataclass(frozen=True)
class GPUStatus:
    """One card, as ``nvidia-smi`` reports it."""

    index: int
    used_mib: int
    total_mib: int
    utilization_pct: int
    uuid: str
    name: str


def query_gpus() -> list[GPUStatus]:
    """Every visible card's memory and utilisation.

    Returns
    -------
    list of GPUStatus
        Empty when ``nvidia-smi`` is absent or fails -- a machine with no
        NVIDIA driver, which the caller reports as "no GPU" rather than as an
        error of its own.
    """
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_gpu_query(output)


def parse_gpu_query(output: str) -> list[GPUStatus]:
    """Parse ``nvidia-smi --format=csv,noheader,nounits`` rows.

    Parameters
    ----------
    output : str
        One row per card: index, used MiB, total MiB, utilisation %, uuid, name.

    Returns
    -------
    list of GPUStatus
    """
    cards: list[GPUStatus] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 6:
            continue
        cards.append(
            GPUStatus(
                index=int(fields[0]),
                used_mib=int(fields[1]),
                total_mib=int(fields[2]),
                utilization_pct=int(fields[3]),
                uuid=fields[4],
                name=fields[5],
            )
        )
    return cards


def occupancy_table(cards: list[GPUStatus]) -> str:
    """The cards and what is on them, for an error message or a log line."""
    if not cards:
        return "no NVIDIA GPUs are visible"
    return "\n".join(
        f"  GPU {card.index} ({card.name}): {card.used_mib}/{card.total_mib} MiB "
        f"used, {card.utilization_pct}% utilisation"
        for card in cards
    )


def _claim(index: int) -> bool:
    """Take the claim file for ``index``, or report that someone else holds it.

    A claim this process already holds is still this process's claim, so
    re-taking it succeeds; a claim whose process is no longer alive is stale
    and is taken over. The claim is released when this process exits.
    """
    CLAIM_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = CLAIM_DIRECTORY / f"gpu{index}.claim"
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise
        try:
            holder = int(path.read_text().strip() or "0")
        except (OSError, ValueError):
            holder = 0
        if holder == os.getpid():
            return True
        if holder and _process_is_alive(holder):
            return False
        # Stale claim: the holder is gone.
        try:
            path.unlink()
        except OSError:
            return False
        return _claim(index)
    with os.fdopen(handle, "w") as claim_file:
        claim_file.write(str(os.getpid()))
    atexit.register(_release, path)
    return True


def _release(path: Path) -> None:
    try:
        if path.exists() and path.read_text().strip() == str(os.getpid()):
            path.unlink()
    except OSError:
        pass


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def select_free_gpu(
    max_used_mib: int = DEFAULT_MAX_USED_MIB,
    max_util_pct: int = 0,
) -> GPUStatus:
    """The emptiest card that nobody -- person or sibling worker -- is using.

    Parameters
    ----------
    max_used_mib : int
        A card with more than this in use is somebody's job.
    max_util_pct : int
        A card busier than this is somebody's job.

    Returns
    -------
    GPUStatus
        The chosen card, claimed for this process.

    Raises
    ------
    NoFreeGPUError
        With the occupancy of every card, so the message says what is in the
        way rather than only that something is.
    """
    cards = query_gpus()
    free = [
        card
        for card in cards
        if card.used_mib <= max_used_mib and card.utilization_pct <= max_util_pct
    ]
    for card in sorted(free, key=lambda card: (card.used_mib, card.index)):
        if _claim(card.index):
            return card
    raise NoFreeGPUError(
        "No GPU is free (needs <= "
        f"{max_used_mib} MiB used and <= {max_util_pct}% utilisation, and no "
        "claim by another worker):\n" + occupancy_table(cards)
    )


class WrongJAXDeviceError(RuntimeError):
    """Raised when JAX did not end up on the single card that was configured."""


def verify_jax_device(
    record: Mapping[str, object], devices: Sequence[Any] | None = None
) -> None:
    """Check that JAX really is on the one device ``record`` names.

    ``CUDA_VISIBLE_DEVICES`` is read when JAX initialises its backend, not when
    it is imported, so configuring after ``import jax`` still works -- but only
    until something has already asked for a device. When that happens the
    environment variable is ignored silently and the process is on **every**
    card, including the ones other people are running on, each preallocated.
    That failure has no symptom until somebody else's job dies, so it is turned
    into a startup error here.

    Parameters
    ----------
    record : Mapping
        A :func:`configure_jax_gpu` record.
    devices : sequence, optional
        The devices to check, for testing. Defaults to ``jax.devices()``, which
        initialises the backend -- that is the point: the check must be made on
        the backend the run will use.

    Raises
    ------
    WrongJAXDeviceError
        When JAX sees a number of devices other than one, a device of the wrong
        platform, or a card that is not the one that was chosen.
    """
    if devices is None:
        import jax

        devices = jax.devices()
    expected_platform = "cpu" if record["jax_backend"] == "cpu" else "gpu"
    if len(devices) != 1:
        raise WrongJAXDeviceError(
            f"JAX sees {len(devices)} devices, not the one that was "
            f"configured ({devices}). The backend was initialised before "
            "CUDA_VISIBLE_DEVICES was set, so this process is on every card "
            "on the host -- including other people's."
        )
    device = devices[0]
    platform = str(getattr(device, "platform", ""))
    if platform != expected_platform:
        raise WrongJAXDeviceError(
            f"JAX is on a {platform!r} device but {expected_platform!r} was "
            f"configured ({device!r})."
        )
    if expected_platform == "cpu":
        return
    # With one card visible, that card is ordinal 0 inside this process; a
    # non-zero ordinal means the mapping never took effect.
    ordinal = getattr(device, "local_hardware_id", 0)
    kind = str(getattr(device, "device_kind", ""))
    if ordinal != 0 or kind != record["gpu_name"]:
        raise WrongJAXDeviceError(
            f"JAX is on {kind!r} at local ordinal {ordinal}, but GPU "
            f"{record['gpu_index']} ({record['gpu_name']}, "
            f"{record['gpu_uuid']}) was configured."
        )


def _cpu_record() -> dict[str, object]:
    """The record a CPU run carries: a backend and no card."""
    return {
        "jax_backend": "cpu",
        "gpu_index": None,
        "gpu_uuid": None,
        "gpu_name": None,
    }


#: This process's configuration, once it has one. A process configures exactly
#: once: JAX's backend is bound after the first fit, so a second selection
#: could not move it, and taking a second card would record the wrong one in
#: the next session's sidecar while leaving the first claimed until exit.
_CONFIGURED: dict[str, object] | None = None


def configure_jax_gpu(
    gpu: int | None = None,
    allow_cpu: bool = False,
    mem_fraction: float = DEFAULT_MEM_FRACTION,
    max_used_mib: int = DEFAULT_MAX_USED_MIB,
) -> dict[str, object]:
    """Point JAX at one card, before JAX initialises a backend.

    Sets ``CUDA_VISIBLE_DEVICES`` to the chosen card,
    ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` and
    ``XLA_PYTHON_CLIENT_MEM_FRACTION`` so the process takes what it needs
    rather than the whole card, then checks with :func:`verify_jax_device` that
    the process really did end up there.

    Called again in the same process -- a worker handed a second session --
    this returns the card it is already on without selecting another. JAX is
    bound to a device by then, so a second selection could only mislabel the
    second session's sidecar and consume a free card nothing releases until
    the process exits.

    Parameters
    ----------
    gpu : int, optional
        Force this card (still claimed, but its occupancy is not checked --
        the caller has said they know what is on it).
    allow_cpu : bool
        Fall back to CPU when no card is free. Off by default: a silent CPU
        run is many times slower and would not be noticed until the batch
        overran.
    mem_fraction : float
        Fraction of the card JAX may take.
    max_used_mib : int
        Passed to :func:`select_free_gpu`.

    Returns
    -------
    dict
        ``jax_backend`` (``"gpu"``/``"cpu"``), ``gpu_index``, ``gpu_uuid``,
        ``gpu_name`` -- for the run's provenance sidecar. The last three are
        ``None`` on CPU.

    Raises
    ------
    NoFreeGPUError
        When no card is free and ``allow_cpu`` is False.
    WrongJAXDeviceError
        When JAX did not end up on the configured card.
    """
    global _CONFIGURED
    if _CONFIGURED is not None:
        return dict(_CONFIGURED)

    # ``JAX_PLATFORMS`` wins over anything decided here: with it pinned away
    # from CUDA the process cannot reach a card at all, so claiming one would
    # park it for nothing and then fail verification. Treated as "no card is
    # free", which the caller already knows how to refuse.
    platforms = os.environ.get("JAX_PLATFORMS", "")
    if platforms and not {"cuda", "gpu"} & {
        part.strip().lower() for part in platforms.split(",")
    }:
        if not allow_cpu:
            raise NoFreeGPUError(
                f"JAX_PLATFORMS={platforms!r} excludes CUDA, so no GPU can be "
                "used. Unset it, or pass allow_cpu."
            )
        record = _cpu_record()
        verify_jax_device(record)
        _CONFIGURED = record
        return dict(record)

    cards = {card.index: card for card in query_gpus()}
    if gpu is not None:
        _claim(gpu)
        chosen = cards.get(gpu)
        if chosen is None:
            raise NoFreeGPUError(
                f"GPU {gpu} was requested but is not visible:\n"
                + occupancy_table(list(cards.values()))
            )
    else:
        try:
            chosen = select_free_gpu(max_used_mib=max_used_mib)
        except NoFreeGPUError:
            if not allow_cpu:
                raise
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            record = _cpu_record()
            verify_jax_device(record)
            _CONFIGURED = record
            return dict(record)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen.index)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
    record = {
        "jax_backend": "gpu",
        "gpu_index": chosen.index,
        "gpu_uuid": chosen.uuid,
        "gpu_name": chosen.name,
    }
    verify_jax_device(record)
    _CONFIGURED = record
    return dict(record)
