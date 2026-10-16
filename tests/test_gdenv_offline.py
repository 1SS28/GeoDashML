"""gdenv tests that need no running game: config, delay buffer, object table, obs layout, CSV log."""

import json

import numpy as np
import pandas as pd
import pytest

from gdbridge.objects import ROLES, GameObjectType
from gdenv import ActionDelayBuffer, GDEnvConfig, ObjectTable, StartCheckpoint, decode, feature_names, obs_size
from gdenv.episode_log import FIELDS, EpisodeLogger
from gdenv.observation import N_OBJECT, N_PLAYER, write_objects


def obj(x, y, w, h, type_=GameObjectType.Solid, decoration=False):
    return {"id": 1, "type": int(type_), "x": x + w / 2, "y": y + h / 2, "rot": 0.0, "sx": 1.0, "sy": 1.0,
            "decoration": decoration, "rect": [x, y, w, h]}


# -- config ----------------------------------------------------------------------------------

def test_config_round_trips_through_json(tmp_path):
    cfg = GDEnvConfig(level_id=1, action_delay_ticks=6, start_checkpoint=StartCheckpoint(400, ((90, True), (96, False))))
    path = cfg.save(tmp_path / "config.json")
    loaded = GDEnvConfig.load(path)
    assert loaded == cfg
    assert json.loads(path.read_text())["start_checkpoint"]["inputs"] == [[90, True], [96, False]]


@pytest.mark.parametrize("bad", [dict(frame_skip=0), dict(action_delay_ticks=-1), dict(lookahead_distance=0), dict(max_objects=-1)])
def test_config_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        GDEnvConfig(**bad)


# -- delay buffer -----------------------------------------------------------------------------

@pytest.mark.parametrize("delay,frame_skip", [(0, 4), (1, 4), (4, 4), (6, 4), (12, 4), (7, 3), (12, 1)])
def test_delay_buffer_releases_each_action_exactly_delay_ticks_later(delay, frame_skip):
    buf = ActionDelayBuffer(delay, frame_skip)
    released = {}
    for k in range(40):
        t = k * frame_skip
        buf.push(t, k)
        for due, action in buf.pop_due(t + frame_skip):
            assert t <= due < t + frame_skip, "released inside the step that contains its due tick"
            released[action] = due
    for action, due in released.items():
        assert due == action * frame_skip + delay
    assert len(buf) == len([k for k in range(40) if k * frame_skip + delay >= 40 * frame_skip])
    assert len(buf) <= buf.capacity - 1


def test_delay_buffer_clear_and_overflow():
    buf = ActionDelayBuffer(8, 4)
    buf.push(0, 1)
    buf.push(4, 0)
    assert buf.pending() == [(8, 1), (12, 0)]
    buf.clear()
    assert len(buf) == 0 and buf.pending() == []
    for k in range(buf.capacity):
        buf.push(4 * k, 1)
    with pytest.raises(OverflowError):
        buf.push(4 * buf.capacity, 1)


# -- object table -----------------------------------------------------------------------------

def test_table_drops_decoration_and_roleless_and_sorts_by_left_edge():
    g = {"level_id": 0, "level_length": 1000.0, "objects": [
        obj(300, 90, 30, 30),
        obj(100, 90, 30, 30, GameObjectType.Hazard),
        obj(200, 90, 30, 30, decoration=True),
        obj(200, 90, 40, 40, GameObjectType.SecretCoin),
        obj(100, 120, 30, 30),
    ]}
    t = ObjectTable.from_geometry(g)
    assert len(t) == 3
    assert t.stats.dropped_decoration == 1 and t.stats.dropped_no_role == 1
    assert list(t.left) == [100.0, 100.0, 300.0]
    assert list(t.cy) == [105.0, 135.0, 105.0]  # same left edge: lower object first
    assert [ROLES[r] for r in t.role] == ["hazard", "solid", "solid"]


def test_ahead_matches_brute_force_on_random_geometry():
    rng = np.random.default_rng(1)
    objects = []
    for _ in range(500):
        w = float(rng.choice([6.0, 9.0, 30.0, 86.0]))
        objects.append(obj(float(rng.integers(0, 5000)), float(rng.integers(90, 400)), w, 30.0,
                           GameObjectType(int(rng.choice([0, 2, 8, 11, 5])))))
    g = {"level_id": 0, "level_length": 5000.0, "objects": objects}
    t = ObjectTable.from_geometry(g)
    la = 300.0
    for px in rng.uniform(-200, 5300, 400):
        idx = t.ahead(px, la)
        got = sorted(zip(t.left[idx], t.cy[idx], t.cx[idx]))
        exp = sorted((o["rect"][0], o["rect"][1] + o["rect"][3] / 2, o["rect"][0] + o["rect"][2] / 2)
                     for o in objects if o["rect"][0] + o["rect"][2] >= px and o["rect"][0] <= px + la)
        assert got == exp, px
        assert list(t.left[idx]) == sorted(t.left[idx]), "nearest left edge first"
    assert len(t.ahead(0.0, la, max_objects=5)) == 5
    assert len(t.ahead(0.0, la, max_objects=0)) == 0
    assert t.max_in_window(la) >= len(t.ahead(0.0, la))


# -- observation layout -------------------------------------------------------------------------

def test_obs_layout_sizes_and_names():
    assert N_PLAYER == 15 and N_OBJECT == 4 + len(ROLES)
    assert obs_size(40) == 15 + 40 * N_OBJECT
    names = feature_names(2)
    assert len(names) == obs_size(2)
    assert names[0] == "x_frac" and names[15] == "obj0_rel_x" and names[-1] == f"obj1_role_{ROLES[-1]}"


def test_write_objects_and_decode_round_trip():
    g = {"level_id": 0, "level_length": 1000.0, "objects": [obj(130, 90, 30, 30), obj(200, 99, 6, 12, GameObjectType.Hazard)]}
    t = ObjectTable.from_geometry(g)
    out = np.zeros((3, N_OBJECT), dtype=np.float32)
    write_objects(out, t, t.ahead(100.0, 600.0), 100.0, 105.0, 600.0, 300.0, 30.0)
    obs = np.concatenate([np.zeros(N_PLAYER, np.float32), out.ravel()])
    d = decode(obs, 3)
    assert len(d["objects"]) == 2, "padding slot has no role and is dropped"
    a, b = d["objects"]
    assert a["role"] == "solid" and np.isclose(a["rel_x"], 45 / 600) and np.isclose(a["rel_y"], 0.0) and a["width"] == 1.0
    assert b["role"] == "hazard" and np.isclose(b["rel_x"], 103 / 600) and np.isclose(b["width"], 0.2) and np.isclose(b["height"], 0.4)
    assert not out[2].any()


# -- episode log ---------------------------------------------------------------------------------

def test_episode_logger_writes_pandas_readable_csv(tmp_path):
    path = tmp_path / "episodes.csv"
    log = EpisodeLogger(path)
    log.log(episode=1, attempt=5, percent=1.9, ticks=400, decisions=100, reward=160.0, outcome="death", start_checkpoint="start", wall_time_s=0.05)
    log.close()
    log = EpisodeLogger(path)  # appending keeps a single header
    log.log(episode=2, attempt=6, percent=2.5, ticks=520, decisions=130, reward=280.0, outcome="death", start_checkpoint="tick400", wall_time_s=0.06)
    log.close()
    df = pd.read_csv(path)
    assert list(df.columns) == list(FIELDS)
    assert len(df) == 2 and df["ticks"].sum() == 920 and df["outcome"].tolist() == ["death", "death"]
