#
# This file is part of gunicorn released under the MIT license.
# See the NOTICE for more information.

"""Acceptance tests for the application-loading process-affinity profile.

The assertions below deliberately track *temporal causality across
loading paths inside one Arbiter tree*:

* preload (``preload_app``): master preloads before fork, HTTP worker
  children hit the cached callable (``master_preload`` ->
  ``worker_inherited``);
* lazy (no preload): each HTTP worker runs the full ``load()`` path in
  its own address space (``worker_lazy``), including reload respawns
  (``worker_reload_exit`` stays on the worker ledger, never dirty);
* dirty: ``DirtyApp.init`` runs in a dirty worker process forked from
  the dirty arbiter, while the HTTP side can only call actions through
  ``DirtyClient`` over a Unix socket (``dirty_init`` vs ``dirty_call``).

Nested profile windows isolate their intent ledgers: an inner window can
never drain or write through an outer window's pending intent.
"""

import json
import os
import signal
import socket
import time
from types import SimpleNamespace

import pytest
from unittest import mock

from gunicorn import appload
from gunicorn.app.base import BaseApplication
from gunicorn.arbiter import Arbiter
from gunicorn.config import Config
from gunicorn.workers.base import Worker
from gunicorn.dirty.worker import DirtyWorker
from gunicorn.dirty import get_dirty_client, close_dirty_client

HAS_FORK = hasattr(os, "fork")
HAS_UNIX = hasattr(socket, "AF_UNIX")
fork_only = pytest.mark.skipif(not HAS_FORK, reason="requires os.fork()")
fork_unix_only = pytest.mark.skipif(
    not (HAS_FORK and HAS_UNIX), reason="requires fork() and AF_UNIX")

# Dedicated support module - nothing else in the suite imports it, so the
# HTTP side can prove the dirty app class was never imported there.
DIRTY_MODULE = "tests.support_appload_dirty"
# Plain import path: what dirty workers load and what clients route to.
DIRTY_SPEC = DIRTY_MODULE + ":ProfileDirtyApp"
# Full spec with an explicit per-app worker limit: config-only; this is
# also what keeps the master from importing the module up-front.
DIRTY_SPEC_FULL = DIRTY_SPEC + ":1"

LEDGER_ENV = "APPLOAD_LOAD_LEDGER"


# ---------------------------------------------------------------------------
# Support objects
# ---------------------------------------------------------------------------

class MockLog:
    def debug(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass

    def exception(self, *a, **k):
        pass

    def close_on_exec(self):
        pass

    def reopen_files(self):
        pass


class ProfileApplication(BaseApplication):
    """Minimal gunicorn application with a unique, observable target."""

    def __init__(self, options=None, wsgi_name=None):
        self._options = dict(options or {})
        if wsgi_name is not None:
            self._options.setdefault("wsgi_app", wsgi_name)
        self.cfg = None
        self.load_count = 0
        super().__init__()

    def load_config(self):
        for key, value in self._options.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        self.load_count += 1
        ledger = os.environ.get(LEDGER_ENV)
        if ledger:
            with open(ledger, "a", encoding="utf-8") as fh:
                fh.write("%d\n" % os.getpid())

        def wsgi(environ, start_response):
            start_response("200 OK", [("Content-Type", "text/plain")])
            return [b"ok"]

        return wsgi


class ExitWorker(Worker):
    """HTTP worker that boots (runs init_process/load_wsgi) and exits 0.

    ``os._exit`` is deliberate: the child forked inside pytest must not
    unwind ``SystemExit`` through the inherited pytest call stack.
    """

    def run(self):
        os._exit(0)


class ReloadExitWorker(Worker):
    """First generation asks the real reloader to respawn it; later ones exit."""

    def run(self):
        if self.age == 1 and self.reloader is not None:
            # Same call the real filesystem reloader thread makes.
            try:
                self.reloader._callback("changed_app.py")
            except SystemExit as exc:
                # The reloader closure ends with sys.exit(0); leave the
                # process directly instead of unwinding through pytest.
                code = exc.code if isinstance(exc.code, int) else 1
                os._exit(code)
        os._exit(0)


def make_lazy_worker(app, reload=False, age=1):
    """A bare worker for exercising the real load_wsgi path in-process."""
    worker = Worker.__new__(Worker)
    worker.app = app
    worker.cfg = SimpleNamespace(reload=reload)
    worker.age = age
    worker.alive = True
    worker.log = MockLog()
    worker.reloader = None
    return worker


def make_dirty_worker(tmpdir, app_paths=(DIRTY_SPEC,), age=1):
    cfg = Config()
    worker = DirtyWorker(
        age=age,
        ppid=os.getpid(),
        app_paths=list(app_paths),
        cfg=cfg,
        log=MockLog(),
        socket_path=os.path.join(str(tmpdir), "dirty-worker.sock"),
    )
    return worker


@pytest.fixture(autouse=True)
def _reset_appload():
    appload.reset()
    os.environ.pop(LEDGER_ENV, None)
    yield
    appload.reset()
    os.environ.pop(LEDGER_ENV, None)


def reap_child(pid, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            wpid, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return None
        if wpid:
            return status
        time.sleep(0.02)
    os.kill(pid, signal.SIGKILL)
    _, status = os.waitpid(pid, 0)
    return status


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def drain_children(deadline_s=2.0):
    """Best-effort reap of reparented grandchildren (e.g. dirty workers)."""
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            wpid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if not wpid:
            return
        time.sleep(0.02)


def spawn_one(arbiter, worker_cls):
    arbiter.worker_class = worker_cls
    pid = arbiter.spawn_worker()
    status = reap_child(pid)
    arbiter.WORKERS.pop(pid, None)
    assert status is not None and os.WIFEXITED(status), status
    return pid, os.WEXITSTATUS(status)


# ---------------------------------------------------------------------------
# Part A: profile is off by default, windows nest, ledgers don't bleed
# ---------------------------------------------------------------------------

def test_disabled_by_default_loading_paths_untouched(tmp_path):
    assert appload.is_enabled() is False
    assert os.environ.get(appload.ENV_TREE) is None
    assert appload.record(appload.Pin.WORKER_LAZY, target="x") is None

    app = ProfileApplication(wsgi_name="tests:appload-off")
    worker = make_lazy_worker(app)
    worker.load_wsgi()

    assert callable(worker.wsgi)
    assert app.load_count == 1
    # No sink file was created and the hooks returned nothing.
    assert list(tmp_path.iterdir()) == []


def test_nested_window_inner_load_wsgi_does_not_drain_outer(tmp_path):
    profile = appload.enable(sink=str(tmp_path / "trace.jsonl"))
    app_a = ProfileApplication(wsgi_name="tests:appload-inner-a")
    app_b = ProfileApplication(wsgi_name="tests:appload-outer-b")
    target_b = appload.describe_app(app_b)

    with profile.window("outer") as outer:
        outer.declare(appload.Pin.WORKER_LAZY, target_b)

        with profile.window("inner") as inner:
            inner.declare(appload.Pin.WORKER_LAZY,
                          appload.describe_app(app_a))
            make_lazy_worker(app_a).load_wsgi()

            assert len(inner.resolved) == 1
            assert inner.pending == []
            assert inner.violations == []

        # Inner window closed: the outer loading-affinity intent is still
        # pending and nothing from the inner ledger wrote through.
        assert appload.current_window() is outer
        assert [i.target for i in outer.pending] == [target_b]
        assert outer.events == []
        assert outer.violations == []

        # Now the outer load happens and resolves the outer intent.
        make_lazy_worker(app_b).load_wsgi()
        assert len(outer.resolved) == 1
        assert outer.pending == []

    # Both loads reached the shared sink, tagged with *different* windows
    # but the same tree.
    events = profile.collect()
    assert {e["tree"] for e in events} == {profile.tree}
    lazy = [e for e in events if e["pin"] == appload.Pin.WORKER_LAZY]
    assert len(lazy) == 2
    assert len({e["window"] for e in lazy}) == 2


def test_nested_window_inner_dirty_init_does_not_write_through(tmp_path):
    profile = appload.enable(sink=str(tmp_path / "trace.jsonl"))

    with profile.window("outer") as outer:
        outer.declare(appload.Pin.DIRTY_INIT, DIRTY_SPEC)

        with profile.window("inner") as inner:
            inner.declare(appload.Pin.DIRTY_INIT, DIRTY_SPEC)
            dw1 = make_dirty_worker(tmp_path / "inner")
            try:
                dw1.load_apps()
            finally:
                dw1.tmp.close()

            assert inner.pending == []
            assert len(inner.resolved) == 1
            init_events = [e for e in inner.events
                           if e["pin"] == appload.Pin.DIRTY_INIT]
            assert len(init_events) == 1
            assert init_events[0]["family"] == "dirty"
            assert init_events[0]["phase"] == "done"
            assert dw1.apps[DIRTY_SPEC].initialized is True

        assert appload.current_window() is outer
        assert len(outer.pending) == 1
        assert outer.events == []

        dw2 = make_dirty_worker(tmp_path / "outer")
        try:
            dw2.load_apps()
        finally:
            dw2.tmp.close()
        assert len(outer.resolved) == 1
        assert outer.pending == []


def test_nested_window_inner_reload_exit_does_not_drain_outer(tmp_path):
    profile = appload.enable(sink=str(tmp_path / "trace.jsonl"))
    app = ProfileApplication(wsgi_name="tests:appload-reload")
    target = appload.describe_app(app)

    with profile.window("outer") as outer:
        outer.declare(appload.Pin.WORKER_RELOAD_EXIT, target)

        with profile.window("inner") as inner:
            inner.declare(appload.Pin.WORKER_RELOAD_EXIT, target)
            gen1 = make_lazy_worker(app, reload=True, age=1)
            gen1._appload_note_reload_exit("changed_app.py")

            assert len(inner.resolved) == 1
            event = inner.resolved[0].event
            assert event["pin"] == appload.Pin.WORKER_RELOAD_EXIT
            assert event["family"] == "worker"
            assert event["reload"] is True
            assert event["alive_before"] is True
            # The note is taken while alive is still true; the closure in
            # init_process flips alive immediately afterwards.
            assert gen1.alive is True

        assert appload.current_window() is outer
        assert len(outer.pending) == 1
        assert outer.events == []

        gen2 = make_lazy_worker(app, reload=True, age=2)
        gen2._appload_note_reload_exit("changed_app.py")
        assert len(outer.resolved) == 1


def test_reload_exit_cannot_masquerade_as_dirty_in_a_window(tmp_path):
    profile = appload.enable(sink=str(tmp_path / "trace.jsonl"))
    target = "tests:appload-reload-vs-dirty"

    with profile.window("taxonomy") as win:
        win.declare(appload.Pin.DIRTY_INIT, target)
        # A reload event for the same target lands on the worker ledger.
        appload.record(appload.Pin.WORKER_RELOAD_EXIT, target=target,
                       fname="x.py", reload=True, alive_before=True,
                       age=1, respawn=False)

        assert len(win.pending) == 1
        assert len(win.violations) == 1
        violation = win.violations[0]
        assert violation["expected_family"] == "dirty"
        assert violation["observed_family"] == "worker"


def test_window_restores_outer_on_exception(tmp_path):
    profile = appload.enable(sink=str(tmp_path / "trace.jsonl"))

    with profile.window("outer") as outer:
        outer.declare(appload.Pin.WORKER_LAZY, "tests:appload-exc")
        with pytest.raises(RuntimeError):
            with profile.window("inner"):
                appload.record(appload.Pin.DIRTY_CALL,
                               "tests.support_appload_dirty:ProfileDirtyApp:1",
                               action="compute", mode="execute",
                               via="unix_socket", socket_path="x")
                raise RuntimeError("boom")

        assert appload.current_window() is outer
        assert len(outer.pending) == 1
        assert outer.events == []


# ---------------------------------------------------------------------------
# Part B: the SAME declared-intent semantics holds on every carrier.
# Removing any carrier instrumentation makes one of these cases red.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("carrier,pin,family", [
    ("preload", appload.Pin.MASTER_PRELOAD, "preload"),
    ("lazy", appload.Pin.WORKER_LAZY, "worker"),
    ("dirty", appload.Pin.DIRTY_INIT, "dirty"),
])
def test_declared_intent_resolves_on_each_carrier(carrier, pin, family,
                                                  tmp_path):
    profile = appload.enable(sink=str(tmp_path / "trace.jsonl"))

    with profile.window("outer") as outer:
        with profile.window("distractor-inner") as inner:
            # Event on another ledger: must not resolve outer and must not
            # bleed into it when the inner window closes.
            other_target = "other:%s" % carrier
            other_pin = (
                appload.Pin.DIRTY_CALL if family != "dirty"
                else appload.Pin.WORKER_LAZY
            )
            inner.declare(other_pin, other_target)
            appload.record(other_pin, target=other_target)
            assert inner.pending == []

        if carrier == "preload":
            app = ProfileApplication(
                {"preload_app": True}, wsgi_name="tests:carrier-preload")
            outer.declare(pin, appload.describe_app(app))
            assert outer.pending and not outer.resolved
            # Real Arbiter.setup preload branch (no fork, no start()).
            arbiter = Arbiter(app)
            assert arbiter.app.callable is not None

        elif carrier == "lazy":
            app = ProfileApplication(wsgi_name="tests:carrier-lazy")
            outer.declare(pin, appload.describe_app(app))
            assert outer.pending and not outer.resolved
            # Real HTTP worker loading exit.
            make_lazy_worker(app).load_wsgi()

        else:
            outer.declare(pin, DIRTY_SPEC)
            assert outer.pending and not outer.resolved
            # Real out-of-process carrier entry point, in this process.
            dw = make_dirty_worker(tmp_path / "carrier-dirty")
            try:
                dw.load_apps()
            finally:
                dw.tmp.close()

        assert outer.violations == []
        assert len(outer.resolved) == 1
        resolved = outer.resolved[0]
        assert resolved.pin == pin
        assert resolved.event["family"] == family
        assert outer.pending == []

    trace = profile.collect()
    assert {e["tree"] for e in trace} == {profile.tree}
    assert any(e["pin"] == pin for e in trace)


# ---------------------------------------------------------------------------
# Part C: same Arbiter tree, real forks - temporal/process causality.
# ---------------------------------------------------------------------------

@fork_only
def test_same_tree_preload_master_then_child_cache_hit(tmp_path):
    sink = tmp_path / "trace.jsonl"
    ledger = tmp_path / "loads.log"
    os.environ[LEDGER_ENV] = str(ledger)

    profile = appload.enable(sink=str(sink))
    app = ProfileApplication({"preload_app": True, "workers": 1},
                             wsgi_name="tests:appload-tree-preload")
    arbiter = Arbiter(app)  # setup() preloads in this process
    arbiter.pid = os.getpid()
    master = os.getpid()

    child, code = spawn_one(arbiter, ExitWorker)
    assert code == 0

    events = profile.collect()
    assert {e["tree"] for e in events} == {profile.tree}

    preloads = [e for e in events
                if e["pin"] == appload.Pin.MASTER_PRELOAD]
    assert len(preloads) == 1
    pre = preloads[0]
    assert pre["pid"] == master
    assert pre["phase"] == "fresh"
    assert pre["callable_cached"] is False
    assert pre["target"] == "tests:appload-tree-preload"

    boots = [e for e in events
             if e["pin"] == appload.Pin.WORKER_BOOT and e["pid"] == child]
    assert len(boots) == 1
    spawns = [e for e in events
              if e["pin"] == appload.Pin.WORKER_SPAWN
              and e["child_pid"] == child]
    assert len(spawns) == 1
    assert spawns[0]["pid"] == master
    assert spawns[0]["preload"] is True

    hits = [e for e in events
            if e["pin"] == appload.Pin.WORKER_INHERITED]
    assert len(hits) == 1
    hit = hits[0]
    assert hit["pid"] == child
    assert hit["cache_hit"] is True
    assert hit["phase"] == "cache_hit"
    assert child != master

    # No fresh per-worker load happened anywhere on this tree.
    assert not [e for e in events
                if e["pin"] == appload.Pin.WORKER_LAZY]

    # Causal order: master preload precedes the child cache hit.
    assert pre["ts"] < hit["ts"]

    # load() itself only ran in the master process.
    load_pids = [int(x) for x in ledger.read_text().split()]
    assert load_pids == [master]


@fork_only
def test_same_tree_lazy_worker_loads_in_child(tmp_path):
    sink = tmp_path / "trace.jsonl"
    ledger = tmp_path / "loads.log"
    os.environ[LEDGER_ENV] = str(ledger)

    profile = appload.enable(sink=str(sink))
    app = ProfileApplication({"workers": 1},
                             wsgi_name="tests:appload-tree-lazy")
    arbiter = Arbiter(app)  # preload_app defaults to False
    arbiter.pid = os.getpid()
    master = os.getpid()

    # The master never loaded the app.
    assert arbiter.app.callable is None
    assert not [e for e in profile.collect()
                if e["pin"] == appload.Pin.MASTER_PRELOAD]

    child, code = spawn_one(arbiter, ExitWorker)
    assert code == 0

    events = profile.collect()
    assert {e["tree"] for e in events} == {profile.tree}

    lazy = [e for e in events
            if e["pin"] == appload.Pin.WORKER_LAZY]
    assert len(lazy) == 1
    load_event = lazy[0]
    assert load_event["pid"] == child
    assert load_event["cache_hit"] is False
    assert load_event["phase"] == "fresh"
    assert child != master

    boots = [e for e in events
             if e["pin"] == appload.Pin.WORKER_BOOT and e["pid"] == child]
    assert len(boots) == 1
    assert boots[0]["reload"] is False

    assert not [e for e in events
                if e["pin"] in (appload.Pin.MASTER_PRELOAD,
                                appload.Pin.WORKER_INHERITED)]

    load_pids = [int(x) for x in ledger.read_text().split()]
    assert load_pids == [child]


@fork_only
def test_same_tree_reload_respawn_stays_worker_ledger(tmp_path):
    sink = tmp_path / "trace.jsonl"
    ledger = tmp_path / "loads.log"
    os.environ[LEDGER_ENV] = str(ledger)

    profile = appload.enable(sink=str(sink))
    app = ProfileApplication({"reload": True, "reload_engine": "poll",
                              "workers": 1},
                             wsgi_name="tests:appload-tree-reload")
    arbiter = Arbiter(app)
    arbiter.pid = os.getpid()

    gen1, code1 = spawn_one(arbiter, ReloadExitWorker)
    assert code1 == 0
    gen2, code2 = spawn_one(arbiter, ReloadExitWorker)
    assert code2 == 0
    assert gen1 != gen2

    events = profile.collect()
    assert {e["tree"] for e in events} == {profile.tree}

    exits = [e for e in events
             if e["pin"] == appload.Pin.WORKER_RELOAD_EXIT]
    assert len(exits) == 1
    exit_event = exits[0]
    assert exit_event["pid"] == gen1
    assert exit_event["family"] == "worker"
    assert exit_event["alive_before"] is True
    assert exit_event["reload"] is True

    boots = sorted(
        (e for e in events if e["pin"] == appload.Pin.WORKER_BOOT),
        key=lambda e: e["age"])
    assert [e["pid"] for e in boots] == [gen1, gen2]
    assert [e["age"] for e in boots] == [1, 2]
    assert [e["respawn"] for e in boots] == [False, True]

    # Both generations loaded freshly in their own address spaces; the
    # respawn is a worker-lazy load, never an out-of-process/dirty load.
    lazy = sorted(
        (e for e in events if e["pin"] == appload.Pin.WORKER_LAZY),
        key=lambda e: e["age"])
    assert [e["pid"] for e in lazy] == [gen1, gen2]
    assert all(e["cache_hit"] is False for e in lazy)
    assert all(e["reload"] is True for e in lazy)

    assert exit_event["ts"] < boots[1]["ts"]

    dirty_pins = {appload.Pin.DIRTY_ARBITER_SPAWN,
                  appload.Pin.DIRTY_ARBITER_BOOT,
                  appload.Pin.DIRTY_WORKER_SPAWN,
                  appload.Pin.DIRTY_WORKER_BOOT,
                  appload.Pin.DIRTY_APP_LOAD, appload.Pin.DIRTY_INIT,
                  appload.Pin.DIRTY_CALL}
    assert not [e for e in events if e["pin"] in dirty_pins]
    assert not [e for e in events
                if e["pin"] in (appload.Pin.MASTER_PRELOAD,
                                appload.Pin.WORKER_INHERITED)]

    load_pids = sorted(int(x) for x in ledger.read_text().split())
    assert load_pids == sorted([gen1, gen2])


@fork_unix_only
def test_same_tree_dirty_init_out_of_process_http_uses_client(tmp_path):
    # Earlier in-process tests may have imported the dedicated module; the
    # whole point here is that the *master/HTTP side* of this tree must not
    # have it in sys.modules, so drop it before forking anything.
    import sys
    sys.modules.pop(DIRTY_MODULE, None)

    sink = tmp_path / "trace.jsonl"
    http_result = tmp_path / "http_result.json"

    profile = appload.enable(sink=str(sink))
    app = ProfileApplication(
        {"dirty_workers": 1, "dirty_apps": [DIRTY_SPEC_FULL]},
        wsgi_name="tests:appload-tree-dirty")
    arbiter = Arbiter(app)
    arbiter.pid = os.getpid()
    master = os.getpid()

    assert DIRTY_MODULE not in sys.modules
    da_pid = arbiter.spawn_dirty_arbiter()
    http_pid = None
    dw_pid = None
    try:
        # Readiness through the real DirtyClient over the Unix socket.
        client = get_dirty_client(timeout=5.0)
        deadline = time.monotonic() + 25
        while True:
            try:
                assert client.execute(DIRTY_SPEC, "compute", 2, 3) == 5
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.1)

        # The master process itself never imported the dirty app class
        # (the ":1" spec even avoids reading its workers attribute here).
        assert DIRTY_MODULE not in sys.modules

        # Fork an HTTP-worker-like process: it may only touch the app
        # through DirtyClient.
        http_pid = os.fork()
        if http_pid == 0:  # pragma: no cover - child branch
            result = {}
            try:
                # Fresh client in the child: never reuse a connection
                # inherited from the master across fork.
                from gunicorn.dirty.client import DirtyClient
                c = DirtyClient(os.environ["GUNICORN_DIRTY_SOCKET"],
                                timeout=5.0)
                where = c.execute(DIRTY_SPEC, "where_am_i")
                total = c.execute(DIRTY_SPEC, "compute", 40, 2)
                result = {
                    "code": 0,
                    "pid": os.getpid(),
                    "compute": total,
                    "where": where,
                    "module_absent": DIRTY_MODULE not in sys.modules,
                }
            except BaseException as exc:  # noqa: BLE001
                result = {"code": 1, "error": repr(exc)}
            try:
                with open(http_result, "w", encoding="utf-8") as fh:
                    json.dump(result, fh)
            finally:
                os._exit(result.get("code", 1))

        status = reap_child(http_pid)
        assert status is not None and os.WIFEXITED(status), status
        assert os.WEXITSTATUS(status) == 0
        data = json.loads(http_result.read_text())
        assert data["compute"] == 42
        assert data["module_absent"] is True
        assert data["where"]["initialized"] is True

        events = profile.collect()
        assert {e["tree"] for e in events} == {profile.tree}

        da_boots = [e for e in events
                    if e["pin"] == appload.Pin.DIRTY_ARBITER_BOOT]
        assert len(da_boots) == 1
        assert da_boots[0]["pid"] == da_pid

        dw_spawns = [e for e in events
                     if e["pin"] == appload.Pin.DIRTY_WORKER_SPAWN]
        assert len(dw_spawns) == 1
        dw_spawn = dw_spawns[0]
        dw_pid = dw_spawn["child_pid"]
        assert dw_spawn["app_paths"] == [DIRTY_SPEC]
        assert dw_pid not in (master, http_pid, da_pid)

        dw_boots = [e for e in events
                    if e["pin"] == appload.Pin.DIRTY_WORKER_BOOT]
        assert len(dw_boots) == 1
        assert dw_boots[0]["pid"] == dw_pid

        inits = [e for e in events
                 if e["pin"] == appload.Pin.DIRTY_INIT]
        assert len(inits) == 1
        init_event = inits[0]
        assert init_event["pid"] == dw_pid
        assert init_event["target"] == DIRTY_SPEC
        assert init_event["phase"] == "done"
        assert init_event["ok"] is True
        # The app itself reports the same init process as the trace.
        assert data["where"]["init_pid"] == dw_pid

        calls = [e for e in events
                 if e["pin"] == appload.Pin.DIRTY_CALL]
        assert calls
        assert {e["target"] for e in calls} == {DIRTY_SPEC}
        assert all(e["via"] == "unix_socket" for e in calls)
        assert all(e["socket_path"] for e in calls)
        assert any(e["pid"] == http_pid and e["mode"] == "execute"
                   for e in calls)

        # DirtyApp.init never ran in an HTTP-side/master process and the
        # HTTP side never emitted an init event.
        init_pids = {e["pid"] for e in inits}
        call_pids = {e["pid"] for e in calls}
        assert init_pids.isdisjoint(call_pids)
        assert http_pid not in init_pids
        assert master not in init_pids

        # Causal order across the tree:
        # arbiter boot -> worker fork -> init -> HTTP socket call.
        assert da_boots[0]["ts"] <= dw_spawn["ts"]
        assert dw_spawn["ts"] < init_event["ts"]
        http_calls = [e for e in calls if e["pid"] == http_pid]
        assert init_event["ts"] < min(e["ts"] for e in http_calls)

        # No WSGI preload/lazy event exists for the dirty app target.
        assert not [e for e in events
                    if e["pin"] in (appload.Pin.MASTER_PRELOAD,
                                    appload.Pin.WORKER_INHERITED,
                                    appload.Pin.WORKER_LAZY)
                    and e["target"] == DIRTY_SPEC]

    finally:
        # Tear the whole dirty subtree down deterministically.
        # 1. The master-side readiness client keeps a persistent accepted
        #    connection to the arbiter; close it so the arbiter's
        #    serve_forever() shutdown (which on 3.12 awaits
        #    server.wait_closed()) is not blocked on this side.
        close_dirty_client()
        # Collect every dirty worker this tree spawned, including when the
        # test fails before its own pid extraction ran.
        worker_pids = set()
        try:
            worker_pids = {
                e["child_pid"] for e in profile.collect()
                if e["pin"] == appload.Pin.DIRTY_WORKER_SPAWN}
        except Exception:
            pass
        if dw_pid is not None:
            worker_pids.add(dw_pid)
        # 2. SIGTERM the arbiter: it stops serving, sets alive=False (so
        #    SIGCHLD handling never respawns), and enters graceful stop().
        arbiter.kill_dirty_arbiter(signal.SIGTERM)
        # Let the arbiter install its shutdown state before its worker goes
        # away, otherwise a SIGCHLD could trigger a respawn.
        time.sleep(0.5)
        # 3. The dirty worker can finish its own wait_closed() drain only
        #    after the arbiter closes its persistent routing connection,
        #    and the arbiter closes it only after reaping the worker - a
        #    chicken-and-egg that production resolves by waiting out
        #    dirty_graceful_timeout (default 30s) and SIGKILLing. Skip the
        #    30s idle wait: remove the idle worker promptly.
        for pid in worker_pids:
            if pid_alive(pid):
                os.kill(pid, signal.SIGKILL)
        # 4. Arbiter reaps the worker(s), finds its pool empty and exits.
        status = reap_child(da_pid, timeout=10)
        if status is None:
            arbiter.kill_dirty_arbiter(signal.SIGKILL)
            reap_child(da_pid, timeout=5)
        for pid in worker_pids:
            if pid_alive(pid):
                os.kill(pid, signal.SIGKILL)
        drain_children()
