"""gdsearch tests that need no game. A toy cube world with the bridge's interface drives the search."""

import json

import pytest

from gdbridge.client import RunResult, State
from gdenv import StartCheckpoint
from gdsearch import (
    SearchBot,
    SearchConfig,
    SearchError,
    Solution,
    aligned_checkpoint_tick,
    backward_schedule,
    checkpoint_at,
    schedule_inputs,
    suffix_decision_offset,
    suffix_decisions,
    suffix_policy,
)


class ToyBridge:
    """A toy level with the bridge's interface. x advances one unit per tick alive.

    A press on the ground starts a jump that lasts JUMP ticks; a held button jumps again on landing.
    A floor spike at tick f kills a cube on the ground at f, a ceiling spike at tick c kills one in the
    air at c, and the level completes when x reaches `length`. Hold-forever always dies at the first
    ceiling spike, so the search has to release, and the three spikes at 120..122 need a jump timed
    inside a 23-tick window.
    """

    JUMP = 26

    def __init__(self, floor=(50, 120, 121, 122, 200), ceiling=(80, 160), length=300, level_id=1):
        self.floor, self.ceiling, self.length, self.level_id = set(floor), set(ceiling), length, level_id
        self.resets = self.runs = self.attempt = 0
        self.render_on = True
        self._restart()

    def _restart(self):
        self._tick = self._x = self._air_until = 0
        self._held = self._dead = self._completed = False

    def _state(self) -> State:
        return State(
            tick=self._tick, x=float(self._x), y=0.0, yvel=0.0, rot=0.0, mode="cube", upside_down=False,
            mini=False, speed=0.9, on_ground=self._tick >= self._air_until, dead=self._dead, started=True,
            completed=self._completed, percent=100.0 * min(self._x, self.length) / self.length,
            attempts=self.attempt, practice=False, level_id=self.level_id, level_length=float(self.length),
        )

    def ping(self):
        return {"ok": True, "bridge": "toy", "gd": "toy", "in_level": True}

    def state(self):
        return self._state()

    def load_main(self, level_id, timeout=0.0):
        self.level_id = level_id
        self._restart()
        return self._state()

    def render(self, on):
        self.render_on = on

    def hold(self, down):
        self._held = bool(down)
        return self._state()

    def reset(self, from_start=False):
        self.resets += 1
        self.attempt += 1
        self._restart()
        return self._state()

    def run(self, ticks, inputs=(), trace=False):
        self.runs += 1
        inputs = sorted((int(t), bool(d)) for t, d in inputs)
        nxt, tr = 0, []
        for _ in range(ticks):
            t = self._tick
            while nxt < len(inputs) and inputs[nxt][0] <= t:
                self._held = inputs[nxt][1]
                nxt += 1
            if not self._dead:
                airborne = t < self._air_until
                if not airborne and self._held:
                    self._air_until = t + self.JUMP
                    airborne = True
                if (airborne and t in self.ceiling) or (not airborne and t in self.floor):
                    self._dead = True
                elif not self._completed:
                    self._x = t + 1
                    if self._x >= self.length:
                        self._completed = True
            tr.append([float(self._x), 1.0 if t < self._air_until else 0.0, 0.0, int(self._dead)])
            self._tick += 1
        return RunResult(self._state(), str(hash(tuple(tuple(e) for e in tr))), tr if trace else None)


def toy_config(**overrides):
    base = dict(level_id=1, windows=(60, 120, 240), pair_window_ticks=120, horizon_ticks=200)
    base.update(overrides)
    return SearchConfig(**base)


# -- toy world sanity ------------------------------------------------------------------------------

def test_toy_dies_at_first_spike_without_input():
    toy = ToyBridge()
    r = toy.run(100, [(0, False)], trace=True)
    assert r.state.dead and r.state.x == 50.0
    assert [t[3] for t in r.trace].index(1) == 50


def test_toy_hold_forever_dies_at_the_ceiling():
    toy = ToyBridge()
    r = toy.run(100, [(3, True)], trace=True)
    assert [t[3] for t in r.trace].index(1) == 80


# -- the search --------------------------------------------------------------------------------------

def test_search_clears_the_toy_level():
    toy = ToyBridge()
    bot = SearchBot(toy, toy_config())
    sol = bot.solve()
    assert sol.completed and sol.ticks == 300 and sol.final_x == 300.0
    assert sol.attempts == bot.attempts and sol.attempts + bot.trim_attempts == toy.resets
    assert sol.attempts < 300, sol.attempts
    assert bot.commits >= 2, "a tap over the first spike, a tap over the triple, then the tap that completes"
    assert [d for _, d in sol.inputs] == [i % 2 == 0 for i in range(len(sol.inputs))]
    assert all(t < sol.ticks for t, _ in sol.inputs)
    assert sol.search["commits"] == bot.commits and sol.search["windows"] == [60, 120, 240]
    assert sol.progress[0][0] == 1 and sol.progress[-1][1] == 300.0
    assert toy.render_on, "rendering is restored after the search"
    # The schedule replays to completion on a fresh toy and the toy's trajectory hash matches.
    fresh = ToyBridge()
    r = sol.replay(fresh)
    assert r.state.completed and r.hash == sol.hash and all(t[3] == 0 for t in r.trace)
    # One tick less does not complete: ticks is the exact completion length.
    assert not sol.replay(ToyBridge(), ticks=sol.ticks - 1).state.completed


def test_search_is_deterministic():
    a = SearchBot(ToyBridge(), toy_config()).solve()
    b = SearchBot(ToyBridge(), toy_config()).solve()
    assert a.inputs == b.inputs and a.attempts == b.attempts and a.progress == b.progress and a.hash == b.hash


def test_target_x_stops_early():
    toy = ToyBridge()
    full = SearchBot(ToyBridge(), toy_config()).solve()
    sol = SearchBot(toy, toy_config(target_x=100.0)).solve()
    assert not sol.completed and sol.hash is None
    assert sol.final_x >= 100.0 and sol.ticks == 100, "x reaches 100 after 100 ticks"
    assert sol.attempts < full.attempts
    assert sol.inputs == full.inputs[: len(sol.inputs)], "a partial search is a prefix of the full one"
    r = sol.replay(ToyBridge())
    assert not r.state.dead and r.state.x >= 100.0


def test_attempt_budget_raises():
    with pytest.raises(SearchError, match="budget"):
        SearchBot(ToyBridge(), toy_config(max_attempts=5)).solve()


def test_impossible_level_exhausts_the_search():
    toy = ToyBridge(floor=(50,), ceiling=(50,))  # dead on the ground and dead in the air at tick 50
    bot = SearchBot(toy, toy_config())
    with pytest.raises(SearchError, match="exhausted"):
        bot.solve()
    assert bot.backtracks == 1 and bot.commits == 0
    # root; the first window holds decisions 0..11 (ticks 3..47, clipped at the level start): 12 taps,
    # then 12 toggles; the wider windows are empty; then the 66 pairs of those 12 decisions minus the
    # 11 that were already tried as taps
    assert bot.attempts == 1 + 12 + 12 + (66 - 11)


def test_decision_windows_and_toggles():
    bot = SearchBot(ToyBridge(), toy_config(windows=(16, 32)))
    # decisions at ticks 3, 7, 11, ...; death at tick 40: last decision is 9 (tick 39)
    assert bot._last_decision(40) == 9 and bot._last_decision(2) == -1
    assert bot._chunk_decisions(40, 0) == [9, 8, 7, 6]  # ticks 39..27 >= 24
    assert bot._chunk_decisions(40, 1) == [5, 4, 3, 2]  # ticks 23..11 >= 8
    assert bot._window_decisions(40, 8) == [8, 9]
    assert bot._toggle((2, 5), 5) == (2,) and bot._toggle((2, 5), 3) == (2, 3, 5)
    assert bot._truncate((2, 5, 9, 10), 39) == (2, 5, 9)


# -- solutions ------------------------------------------------------------------------------------------

def make_solution(**overrides):
    base = dict(
        level_id=1,
        inputs=((47, True), (71, False), (119, True), (143, False), (199, True), (211, False)),
        ticks=300,
        attempts=42,
        wall_time_s=1.5,
    )
    base.update(overrides)
    return Solution(**base)


def test_solution_round_trips_through_json(tmp_path):
    sol = make_solution(search={"windows": [120, 240]}, progress=[[1, 50.0], [3, 300.0]], hash="123")
    path = sol.save(tmp_path / "solutions" / "1.json")
    assert path == tmp_path / "solutions" / "1.json"
    loaded = Solution.load(path)
    assert loaded == sol
    raw = json.loads(path.read_text())
    assert raw["inputs"] == [[47, True], [71, False], [119, True], [143, False], [199, True], [211, False]]
    assert raw["level_id"] == 1 and raw["ticks"] == 300 and raw["attempts"] == 42
    assert Solution.default_path(7).as_posix() == "data/solutions/7.json"


@pytest.mark.parametrize(
    "inputs",
    [
        ((48, True),),  # not a decision tick
        ((47, False),),  # must start with a press
        ((47, True), (51, True)),  # must alternate
        ((71, True), (47, False)),  # must increase
        ((47, True), (299, False), (303, True)),  # at or after the end
        ((2, True),),  # before the first decision tick
    ],
)
def test_solution_rejects_bad_schedules(inputs):
    with pytest.raises(ValueError):
        make_solution(inputs=inputs)


def test_solution_schedule_queries():
    sol = make_solution()
    assert [sol.hold_at(t) for t in (0, 46, 47, 70, 71, 118, 119, 198, 199, 210, 211, 999)] == [
        False, False, True, True, False, False, True, False, True, True, False, False,
    ]
    assert sol.toggles() == [11, 17, 29, 35, 49, 52]
    assert sol.action_at(10) == 0 and sol.action_at(11) == 1 and sol.action_at(16) == 1 and sol.action_at(17) == 0
    assert sol.n_decisions == 75, "decisions at ticks 3 .. 299"
    actions = sol.env_actions()
    assert len(actions) == 75 and sum(actions) == (17 - 11) + (35 - 29) + (52 - 49)
    assert sol.prefix(119) == ((47, True), (71, False)) and sol.prefix(120) == sol.inputs[:3]
    assert schedule_inputs(sol.toggles(), 3, 4) == list(sol.inputs)


# -- curriculum ----------------------------------------------------------------------------------------

def test_aligned_checkpoint_ticks():
    sol = make_solution()
    assert [aligned_checkpoint_tick(sol, t) for t in (2, 3, 5, 6, 1000, 1001, 1002)] == [2, 2, 2, 6, 998, 998, 1002]
    with pytest.raises(ValueError):
        aligned_checkpoint_tick(sol, 1)


def test_checkpoint_at_replays_the_prefix_and_lines_up_decisions():
    sol = make_solution()
    cp = checkpoint_at(sol, 121)
    assert cp == StartCheckpoint(tick=118, inputs=((47, True), (71, False)))
    assert suffix_decision_offset(sol, cp) == 29, "env decision 0 is solution decision 29 (tick 119)"
    policy = suffix_policy(sol, cp)
    assert [policy(k) for k in range(8)] == [sol.action_at(29 + k) for k in range(8)] == [1, 1, 1, 1, 1, 1, 0, 0]
    assert suffix_decisions(sol, cp) == sol.n_decisions - 29
    assert suffix_decision_offset(sol, None) == 0 and suffix_decisions(sol, None) == sol.n_decisions
    assert checkpoint_at(sol, 300).tick == 298, "aligned down, still before the end"
    with pytest.raises(ValueError):
        checkpoint_at(sol, 302)
    with pytest.raises(ValueError):
        checkpoint_at(sol, 100, warmup_ticks=0)
    with pytest.raises(ValueError):
        suffix_decision_offset(sol, StartCheckpoint(tick=100))


def test_backward_schedule_runs_from_the_end_to_the_start():
    sol = make_solution()
    cps = backward_schedule(sol, spacing_ticks=100)
    assert cps[-1] is None
    ticks = [cp.tick for cp in cps[:-1]]
    assert ticks == [198, 98]
    assert all((t - 2) % 4 == 0 for t in ticks)
    assert cps[0].inputs == sol.prefix(198)
    # released_only moves a checkpoint that lands inside a hold back to before the press
    held = backward_schedule(sol, spacing_ticks=170, released_only=False)
    assert [cp.tick for cp in held[:-1]] == [130] and sol.hold_at(130)
    released = backward_schedule(sol, spacing_ticks=170)
    assert [cp.tick for cp in released[:-1]] == [118] and not sol.hold_at(118)
    assert backward_schedule(sol, spacing_ticks=1000) == [None]
    with pytest.raises(ValueError):
        backward_schedule(sol, spacing_ticks=1)
