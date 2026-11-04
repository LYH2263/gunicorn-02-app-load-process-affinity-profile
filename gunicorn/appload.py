#
# This file is part of gunicorn released under the MIT license.
# See the NOTICE for more information.

"""
Application loading process-affinity profiles.

Gunicorn can put the application callable into three structurally
different places, and the three ledgers are not interchangeable:

``master_preload`` / ``worker_inherited``
    With ``preload_app`` the master calls ``app.wsgi()`` in
    :meth:`gunicorn.arbiter.Arbiter.setup` before forking.  The callable
    is cached in the master address space and every HTTP worker inherits
    the copy-on-write pages; ``Worker.load_wsgi`` hits the cached
    callable instead of walking the full import path again.

``worker_lazy``
    Without ``preload_app`` each HTTP worker (sync, gthread, gevent, ...)
    calls ``load_wsgi -> app.wsgi() -> load()`` from
    :meth:`gunicorn.workers.base.Worker.init_process` in its own address
    space.  With ``reload`` the reloader flips ``alive`` to false and the
    worker exits so the master respawns it; the load still happens inside
    the (new) worker process (``worker_reload_exit`` keeps the *worker*
    pin, it must never masquerade as a dirty out-of-process load).

``dirty_init`` / ``dirty_call``
    The master spawns a separate :class:`gunicorn.dirty.arbiter.DirtyArbiter`
    which forks dirty workers.  ``DirtyWorker.load_apps`` instantiates the
    apps and runs ``DirtyApp.init`` out of the HTTP workers' address space;
    the HTTP side only reaches the app through ``DirtyClient`` calls over a
    Unix socket.

This module gives a *host* (the embedding program, not the WSGI app) one
nestable profiling semantics shared by all three loading pins:

* profiling is opt-in and **off by default** - every ``note_*`` hook is a
  cheap no-op until a profile is enabled for the Arbiter tree, and loading
  behaviour stays byte-for-byte identical when it is off;
* the profile is bound to a single Arbiter **tree identity** that survives
  ``fork()`` (module state is inherited and the identity is also exported
  in the environment for the dirty subtree), so preload master, HTTP
  workers, the dirty arbiter and dirty workers all report against the same
  tree - never two unrelated Arbiters;
* hosts open :class:`ProfileWindow` contexts, which may be nested.  A
  nested window gets its *own* intent ledger: events observed inside it
  resolve intents declared inside it and can never drain or write through
  an outer window's pending intents.  When the inner window closes the
  outer window's loading-affinity intent is visible again, unchanged;
* every event is also appended to a cross-process JSON-lines sink tagged
  with tree/window/pid/ppid/pin, which is what lets a host assert temporal
  causality across master, HTTP worker and dirty worker processes.

The module deliberately does *not* wrap ``app.wsgi()``/``load()`` behind a
single factory: each loading exit keeps its own pin and its own
instrumentation point, they merely share the window/ledger semantics
defined here.
"""

import contextlib
import json
import os
import tempfile
import threading
import time
import uuid


# ---------------------------------------------------------------------------
# Loading pins
# ---------------------------------------------------------------------------

class Pin:
    """Canonical process affinities an application load can be pinned to."""

    #: ``load()`` ran in the master before fork (Arbiter.setup, preload_app).
    MASTER_PRELOAD = "master_preload"

    #: A forked HTTP worker hit the callable cached by the master preload.
    WORKER_INHERITED = "worker_inherited"

    #: ``load()`` ran freshly inside an HTTP worker child (no preload).
    WORKER_LAZY = "worker_lazy"

    #: The reloader flipped ``alive`` to false in an HTTP worker so the
    #: master respawns it.  Same address-space family as ``worker_lazy`` -
    #: it is explicitly *not* an out-of-process/dirty load.
    WORKER_RELOAD_EXIT = "worker_reload_exit"

    #: Fork boundaries on the HTTP worker side (master spawned a worker /
    #: the worker child booted).  They belong to the worker ledger and give
    #: cross-process tests the causal ordering around ``load_wsgi``.
    WORKER_SPAWN = "worker_spawn"
    WORKER_BOOT = "worker_boot"

    #: Master forked the dirty arbiter / dirty arbiter came up (boundary).
    DIRTY_ARBITER_SPAWN = "dirty_arbiter_spawn"
    DIRTY_ARBITER_BOOT = "dirty_arbiter_boot"

    #: Dirty arbiter forked a dirty worker, handing it its (possibly
    #: limited) app assignment (boundary).
    DIRTY_WORKER_SPAWN = "dirty_worker_spawn"
    DIRTY_WORKER_BOOT = "dirty_worker_boot"

    #: A dirty app was instantiated / ``DirtyApp.init`` ran inside a dirty
    #: worker process, i.e. outside the HTTP workers' address space.
    DIRTY_APP_LOAD = "dirty_app_load"
    DIRTY_INIT = "dirty_init"

    #: The HTTP side only invoked the app through ``DirtyClient`` over a
    #: Unix socket; no dirty app class was imported/loaded there.
    DIRTY_CALL = "dirty_call"


#: Which of the three loading ledgers a pin belongs to.  The families are
#: intentionally kept distinct so a profile can detect pin bleed-through
#: (e.g. a reload respawn being labelled as a dirty out-of-process load).
PIN_FAMILY = {
    Pin.MASTER_PRELOAD: "preload",
    Pin.WORKER_INHERITED: "preload",
    Pin.WORKER_LAZY: "worker",
    Pin.WORKER_RELOAD_EXIT: "worker",
    Pin.WORKER_SPAWN: "worker",
    Pin.WORKER_BOOT: "worker",
    Pin.DIRTY_ARBITER_SPAWN: "dirty",
    Pin.DIRTY_ARBITER_BOOT: "dirty",
    Pin.DIRTY_WORKER_SPAWN: "dirty",
    Pin.DIRTY_WORKER_BOOT: "dirty",
    Pin.DIRTY_APP_LOAD: "dirty",
    Pin.DIRTY_INIT: "dirty",
    Pin.DIRTY_CALL: "dirty",
}

#: Environment used to carry the tree identity/sink across forks, including
#: the master -> dirty arbiter -> dirty worker subtree.
ENV_TREE = "GUNICORN_APPLOAD_TREE"
ENV_SINK = "GUNICORN_APPLOAD_SINK"


def describe_app(app):
    """Best-effort, import-free description of a gunicorn application.

    Used as the default ``target`` of WSGI loading events so the three
    carriers can be compared against the same declared intent.
    """
    if app is None:
        return None
    app_uri = getattr(app, "app_uri", None)
    if app_uri:
        return app_uri
    cfg = getattr(app, "cfg", None)
    wsgi_app = getattr(cfg, "wsgi_app", None)
    if wsgi_app:
        return wsgi_app
    return type(app).__name__


# ---------------------------------------------------------------------------
# Window / intent ledger
# ---------------------------------------------------------------------------

class Intent:
    """A host-declared, still-expected loading affinity inside a window."""

    __slots__ = ("pin", "target", "key", "state", "event")

    PENDING = "pending"
    RESOLVED = "resolved"

    def __init__(self, pin, target, key):
        self.pin = pin
        self.target = target
        self.key = key
        self.state = self.PENDING
        self.event = None

    def as_dict(self):
        return {
            "pin": self.pin,
            "target": self.target,
            "state": self.state,
            "resolved_by": None if self.event is None else self.event["seq"],
        }


class ProfileWindow:
    """A nestable application-load profiling window.

    Each window owns a private intent ledger and a private list of the
    events observed while it is on top of the stack.  Events recorded in a
    nested window can never resolve an intent of an outer window, and
    closing the nested window makes the outer ledger visible again exactly
    as it was.
    """

    def __init__(self, depth, reason=None):
        self.id = uuid.uuid4().hex
        self.depth = depth
        self.reason = reason
        self.pid = os.getpid()
        self.intents = {}  # key -> Intent
        self.events = []
        self.violations = []
        self.closed = False

    def declare(self, pin, target=None, key=None):
        """Declare that a load of ``target`` is expected on ``pin``.

        The intent stays *pending* until an event with the same
        ``(pin, target)`` is observed inside this very window.  Declaring
        the same key more than once just returns the existing intent.
        """
        if key is None:
            key = (pin, target)
        intent = self.intents.get(key)
        if intent is None:
            intent = Intent(pin, target, key)
            self.intents[key] = intent
        return intent

    @property
    def pending(self):
        return [i for i in self.intents.values()
                if i.state == Intent.PENDING]

    @property
    def resolved(self):
        return [i for i in self.intents.values()
                if i.state == Intent.RESOLVED]

    def snapshot(self):
        """Return a plain-data view of the window ledger."""
        return {
            "id": self.id,
            "depth": self.depth,
            "reason": self.reason,
            "pid": self.pid,
            "pending": [i.as_dict() for i in self.pending],
            "resolved": [i.as_dict() for i in self.resolved],
            "violations": list(self.violations),
            "events": list(self.events),
        }

    def _absorb(self, event):
        key = (event["pin"], event["target"])
        intent = self.intents.get(key)
        if intent is not None and intent.state == Intent.PENDING:
            intent.state = Intent.RESOLVED
            intent.event = event
        else:
            # A pending intent for the same target on a pin from a
            # *different loading ledger* (preload vs worker vs dirty) is an
            # affinity violation: the load landed on the wrong process
            # surface. Sibling pins of the same family (e.g. dirty_init vs
            # dirty_app_load) are not cross-ledger bleed.
            observed_family = event.get("family")
            others = [i for i in self.intents.values()
                      if i.state == Intent.PENDING
                      and i.target == event["target"]
                      and i.pin != event["pin"]
                      and PIN_FAMILY.get(i.pin) != observed_family]
            if others:
                self.violations.append({
                    "expected_pin": others[0].pin,
                    "expected_family": PIN_FAMILY.get(others[0].pin),
                    "observed_pin": event["pin"],
                    "observed_family": observed_family,
                    "target": event["target"],
                    "seq": event["seq"],
                })
        self.events.append(event)


# ---------------------------------------------------------------------------
# Per-process registry
# ---------------------------------------------------------------------------

class _Registry:
    """Process-local binding of a profile to one Arbiter tree.

    Forked processes inherit the binding copy-on-write; each process then
    appends to its own window stack while the shared sink gathers the
    cross-process trace.
    """

    def __init__(self, tree_id, sink_path):
        self.tree_id = tree_id
        self.sink_path = sink_path
        self.windows = []
        self.seq = 0
        self.lock = threading.Lock()

    def next_seq(self):
        with self.lock:
            seq = self.seq
            self.seq += 1
            return seq


_state = None
_state_lock = threading.Lock()


def reset():
    """Forget the current process binding.

    This is intended for tests that drive the registry in-process; real
    deployments bind once for the lifetime of the Arbiter tree.
    """
    global _state
    with _state_lock:
        _state = None
    os.environ.pop(ENV_TREE, None)
    os.environ.pop(ENV_SINK, None)


def is_enabled():
    """Return True when this process belongs to a profiled Arbiter tree."""
    return _state is not None or ENV_TREE in os.environ


def bind_if_configured():
    """Bind the registry from the inherited environment after a fork.

    No-op when profiling is disabled for the tree.  Explicit bindings
    (master side) always win.
    """
    global _state
    if _state is not None:
        return _state
    tree_id = os.environ.get(ENV_TREE)
    if not tree_id:
        return None
    with _state_lock:
        if _state is None:
            _state = _Registry(tree_id, os.environ.get(ENV_SINK, ""))
        return _state


def enable(tree_id=None, sink=None):
    """Enable application-load profiling for this process and its forks.

    Args:
        tree_id: explicit tree identity; a fresh UUID is generated when
            omitted.
        sink: path to the JSON-lines trace file; a temp file is created
            when omitted.

    Returns:
        A :class:`LoadProfile` handle bound to the tree.
    """
    global _state
    tree_id = tree_id or uuid.uuid4().hex
    if not sink:
        sink = os.path.join(
            tempfile.gettempdir(),
            "gunicorn-appload-%s-%s.jsonl" % (os.getpid(), tree_id[:8]),
        )
    with _state_lock:
        _state = _Registry(tree_id, sink)
    os.environ[ENV_TREE] = tree_id
    os.environ[ENV_SINK] = sink
    return LoadProfile(tree_id, sink)


@contextlib.contextmanager
def window(reason=None):
    """Open a nested profiling window (no-op when profiling is disabled).

    On exit the previously current window is restored verbatim; events and
    intent resolutions that happened inside this window stay in it.
    """
    registry = bind_if_configured()
    if registry is None:
        yield None
        return

    with registry.lock:
        win = ProfileWindow(depth=len(registry.windows), reason=reason)
        registry.windows.append(win)
    try:
        yield win
    finally:
        with registry.lock:
            win.closed = True
            try:
                registry.windows.remove(win)
            except ValueError:
                pass


def declare(pin, target=None):
    """Declare a pending intent on the current window.

    Returns the intent, or ``None`` when profiling is disabled / no window
    is open.
    """
    registry = bind_if_configured()
    if registry is None or not registry.windows:
        return None
    with registry.lock:
        return registry.windows[-1].declare(pin, target)


def current_window():
    registry = bind_if_configured()
    if registry is None or not registry.windows:
        return None
    return registry.windows[-1]


def record(pin, target=None, **attrs):
    """Record a loading-affinity event for the current Arbiter tree.

    The event is:

    * resolved against the innermost open window's pending intents in the
      current process only (outer windows are never touched);
    * appended, best-effort, to the cross-process JSON-lines sink.

    Returns the event dict, or ``None`` when profiling is disabled (the
    common production path, which stays allocation-light).
    """
    registry = bind_if_configured()
    if registry is None:
        return None

    seq = registry.next_seq()
    event = {
        "tree": registry.tree_id,
        "window": registry.windows[-1].id if registry.windows else None,
        "window_depth": len(registry.windows),
        "pid": os.getpid(),
        "ppid": os.getppid() if hasattr(os, "getppid") else 0,
        "seq": seq,
        "ts": time.time(),
        "pin": pin,
        "family": PIN_FAMILY.get(pin, "unknown"),
        "target": target,
    }
    for key, value in attrs.items():
        event[key] = value

    _append_sink(registry.sink_path, event)

    with registry.lock:
        if registry.windows:
            registry.windows[-1]._absorb(event)
    return event


def _append_sink(sink_path, event):
    if not sink_path:
        return
    try:
        line = json.dumps(event, separators=(",", ":"), default=str)
        # One short line per open()-append write; POSIX guarantees atomic
        # O_APPEND writes below the pipe buffer size, which keeps traces
        # intact across concurrent processes/threads.
        with open(sink_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        # Tracing must never break a loading path.
        pass


def collect(sink=None, tree=None):
    """Read back cross-process trace events from a JSON-lines sink."""
    if sink is None:
        registry = bind_if_configured()
        if registry is None:
            return []
        sink = registry.sink_path
    events = []
    try:
        with open(sink, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if tree is None or event.get("tree") == tree:
                    events.append(event)
    except OSError:
        return []
    events.sort(key=lambda e: (e.get("ts", 0.0), e.get("pid", 0),
                               e.get("seq", 0)))
    return events


# ---------------------------------------------------------------------------
# Host-facing handle
# ---------------------------------------------------------------------------

class LoadProfile:
    """Handle returned when a host enables profiling for an Arbiter tree."""

    def __init__(self, tree_id, sink_path):
        self.tree = tree_id
        self.sink = sink_path

    def window(self, reason=None):
        """Open a (possibly nested) profiling window."""
        return window(reason=reason)

    def declare(self, pin, target=None):
        return declare(pin, target)

    def collect(self):
        return collect(self.sink, self.tree)

    def local_events(self):
        """Events absorbed by the current process' innermost window."""
        win = current_window()
        return list(win.events) if win is not None else []

    def __repr__(self):
        return "<LoadProfile tree=%s sink=%s>" % (
            self.tree[:8], self.sink)


# ---------------------------------------------------------------------------
# Instrumentation hooks - one per loading exit, distinct pins
# ---------------------------------------------------------------------------

def note_master_preload(app):
    """``Arbiter.setup`` is about to call ``app.wsgi()`` pre-fork."""
    if bind_if_configured() is None:
        return
    cached = getattr(app, "callable", None) is not None
    record(
        Pin.MASTER_PRELOAD,
        target=describe_app(app),
        phase="cache_hit" if cached else "fresh",
        callable_cached=cached,
    )


def note_worker_spawn(arbiter, worker, pid):
    """Master forked an HTTP worker."""
    if bind_if_configured() is None:
        return
    record(
        Pin.WORKER_SPAWN,
        target=describe_app(getattr(worker, "app", None)),
        child_pid=pid,
        age=getattr(worker, "age", None),
        worker_class=type(worker).__name__,
        preload=bool(getattr(arbiter.cfg, "preload_app", False)),
    )


def note_worker_boot(worker):
    """An HTTP worker child entered ``init_process``."""
    if bind_if_configured() is None:
        return
    record(
        Pin.WORKER_BOOT,
        target=describe_app(getattr(worker, "app", None)),
        age=getattr(worker, "age", None),
        reload=bool(getattr(worker.cfg, "reload", False)),
        worker_class=type(worker).__name__,
        respawn=getattr(worker, "age", 1) > 1,
    )


def note_worker_wsgi(worker, phase, cache_hit, fail_app=False):
    """``Worker.load_wsgi`` resolved the WSGI callable in a worker child.

    ``cache_hit`` distinguishes the preload child side (inherited cached
    callable) from the lazy per-worker load; ``reload``/``age`` mark
    respawns after code changes, which stay on the worker ledger.
    """
    if bind_if_configured() is None:
        return
    pin = Pin.WORKER_INHERITED if cache_hit else Pin.WORKER_LAZY
    record(
        pin,
        target=describe_app(getattr(worker, "app", None)),
        phase=phase,
        cache_hit=cache_hit,
        fail_app=fail_app,
        reload=bool(getattr(worker.cfg, "reload", False)),
        age=getattr(worker, "age", None),
        respawn=getattr(worker, "age", 1) > 1,
        worker_class=type(worker).__name__,
    )


def note_worker_reload_exit(worker, fname):
    """The reloader is about to flip ``alive`` to false and exit.

    This stays on the *worker* pin: the replacement worker reloads in its
    own address space, it is not a dirty out-of-process load.
    """
    if bind_if_configured() is None:
        return
    record(
        Pin.WORKER_RELOAD_EXIT,
        target=describe_app(getattr(worker, "app", None)),
        fname=fname,
        age=getattr(worker, "age", None),
        alive_before=bool(worker.alive),
        respawn=getattr(worker, "age", 1) > 1,
        reload=True,
    )


def note_dirty_arbiter_spawn(arbiter, pid, socket_path):
    """Master forked the dirty arbiter."""
    if bind_if_configured() is None:
        return
    record(
        Pin.DIRTY_ARBITER_SPAWN,
        target="dirty-arbiter",
        child_pid=pid,
        socket_path=socket_path,
        dirty_apps=list(getattr(arbiter.cfg, "dirty_apps", []) or []),
        dirty_workers=int(getattr(arbiter.cfg, "dirty_workers", 0) or 0),
    )


def note_dirty_arbiter_boot(dirty_arbiter):
    """The dirty arbiter process entered its run loop."""
    if bind_if_configured() is None:
        return
    record(
        Pin.DIRTY_ARBITER_BOOT,
        target="dirty-arbiter",
        apps=list(dirty_arbiter.app_specs.keys()),
        num_workers=dirty_arbiter.num_workers,
        socket_path=dirty_arbiter.socket_path,
    )


def note_dirty_worker_spawn(dirty_arbiter, pid, app_paths):
    """Dirty arbiter forked a dirty worker with its (possibly limited) apps.

    ``app_paths`` is the exact assignment computed from ``module:Class:N``
    specs / ``app_worker_map``, not the full configured app list.
    """
    if bind_if_configured() is None:
        return
    record(
        Pin.DIRTY_WORKER_SPAWN,
        target=",".join(app_paths),
        child_pid=pid,
        app_paths=list(app_paths),
        limited=len(app_paths) < len(dirty_arbiter.app_specs),
    )


def note_dirty_worker_boot(worker):
    """A dirty worker child entered ``init_process``."""
    if bind_if_configured() is None:
        return
    record(
        Pin.DIRTY_WORKER_BOOT,
        target=",".join(worker.app_paths),
        app_paths=list(worker.app_paths),
        age=getattr(worker, "age", None),
    )


def note_dirty_apps_load(worker):
    """A dirty worker is about to instantiate its assigned apps."""
    if bind_if_configured() is None:
        return
    record(
        Pin.DIRTY_APP_LOAD,
        target=",".join(worker.app_paths),
        app_paths=list(worker.app_paths),
        phase="instantiate",
        age=getattr(worker, "age", None),
    )


def note_dirty_init(worker, app_path, phase, error=None):
    """``DirtyApp.init`` ran (or failed) inside a dirty worker process."""
    if bind_if_configured() is None:
        return
    record(
        Pin.DIRTY_INIT,
        target=app_path,
        phase=phase,
        ok=error is None,
        error=None if error is None else str(error),
        age=getattr(worker, "age", None),
    )


def note_dirty_call(app_path, action, mode, socket_path):
    """The HTTP side invoked a dirty app *only* through ``DirtyClient``."""
    if bind_if_configured() is None:
        return
    record(
        Pin.DIRTY_CALL,
        target=app_path,
        action=action,
        mode=mode,
        via="unix_socket",
        socket_path=socket_path,
    )
