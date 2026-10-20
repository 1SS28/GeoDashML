// GDML bridge: lockstep control of Geometry Dash over a loopback TCP socket.
// Protocol: docs/bridge-protocol.md.
//
// Design:
//   * A non-blocking listener on 127.0.0.1:41414 is created when the mod loads.
//   * Every frame, after the game's own mainLoop, the bridge accepts a client if none is
//     attached, then services commands until a per-frame budget runs out
//     (8 ms with rendering on, 1000 ms with rendering off).
//   * Music is not started while a client is attached: the game's music start spins on
//     FMOD's stream state in a way that can livelock inside resetLevel.
//   * While a client is attached, the game's own per-frame physics update is swallowed
//     by the GJBaseGameLayer::update hook. The bridge advances physics itself by calling
//     update(1/240) once per tick. Death does not auto-reset while attached.
//   * Everything runs on the game's main thread; there are no extra threads.
//   * Playback: with no client attached, pressing A in a level resets it from the start and the
//     mod replays ~/Library/Application Support/GeoDashML/solutions/<level_id>.json (written by
//     gdsearch) in lockstep at 240 ticks per second of real time, so the game plays itself with
//     the schedule's exact tick timing. A again stops it.

#include <Geode/Geode.hpp>
#include <Geode/modify/CCDisplayLinkDirector.hpp>
#include <Geode/modify/CCKeyboardDispatcher.hpp>
#include <Geode/modify/FMODAudioEngine.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>
#include <matjson.hpp>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace geode::prelude;

namespace {

constexpr int         kPort          = 41414;
constexpr float       kTickDt        = 1.0f / 240.0f;
constexpr char const* kBridgeVersion = "0.1.6";
constexpr char const* kGdVersion     = "2.2081";
constexpr int         kMaxTicks      = 2'000'000;

struct Bridge {
    int         listenFd = -1;
    int         clientFd = -1;
    std::string inbuf;
    bool        stepping = false;  // true only while the bridge itself drives update()
    bool        marking  = false;  // true only while the bridge itself places a practice checkpoint
    bool        renderOn = true;
    uint64_t    tick     = 0;      // ticks advanced since level entry or last reset
    bool attached() const { return clientFd >= 0; }
};
Bridge g;

// Solution playback (the A key).
struct Playback {
    bool                                   active = false;
    std::vector<std::pair<int64_t, bool>>  inputs;
    size_t                                 next   = 0;
    int64_t                                ticks  = 0;  // ticks to play
    int64_t                                tick   = 0;  // ticks played
    double                                 carry  = 0;  // fractional ticks owed
    std::chrono::steady_clock::time_point  last;
};
Playback play;
void startPlayback(PlayLayer* pl);

using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------- socket plumbing

void setNonBlocking(int fd) {
    int fl = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, fl | O_NONBLOCK);
}

bool startServer() {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { log::error("bridge: socket() failed: {}", strerror(errno)); return false; }
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(kPort);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) < 0) {
        log::error("bridge: bind(127.0.0.1:{}) failed: {}", kPort, strerror(errno));
        close(fd);
        return false;
    }
    if (listen(fd, 1) < 0) {
        log::error("bridge: listen() failed: {}", strerror(errno));
        close(fd);
        return false;
    }
    setNonBlocking(fd);
    g.listenFd = fd;
    log::info("bridge: listening on 127.0.0.1:{}", kPort);
    return true;
}

void dropClient(char const* why) {
    if (g.clientFd >= 0) close(g.clientFd);
    g.clientFd = -1;
    g.inbuf.clear();
    g.stepping = false;
    log::info("bridge: client detached ({})", why);
}

void acceptClient() {
    sockaddr_in a{};
    socklen_t   len = sizeof a;
    int fd = accept(g.listenFd, reinterpret_cast<sockaddr*>(&a), &len);
    if (fd < 0) {
        if (errno != EAGAIN && errno != EWOULDBLOCK) log::warn("bridge: accept() failed: {}", strerror(errno));
        return;
    }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    g.clientFd = fd;
    g.inbuf.clear();
    log::info("bridge: client attached");
}

// The accepted socket inherits O_NONBLOCK from the listener on macOS, so a reply larger than the
// socket buffer gets EAGAIN part-way; wait for the client to drain it instead of dropping the client
// (a 6 MB trace reply did exactly that, 2026-10-17). Gives up after 30 s without progress.
bool sendAll(std::string const& s) {
    size_t off = 0;
    while (off < s.size()) {
        ssize_t n = send(g.clientFd, s.data() + off, s.size() - off, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                pollfd p{g.clientFd, POLLOUT, 0};
                if (poll(&p, 1, 30000) <= 0) return false;
                continue;
            }
            return false;
        }
        off += static_cast<size_t>(n);
    }
    return true;
}

// Reads one line, waiting until `deadline` at most. Returns false on timeout or disconnect.
bool readLine(std::string& line, Clock::time_point deadline) {
    while (true) {
        auto nl = g.inbuf.find('\n');
        if (nl != std::string::npos) {
            line = g.inbuf.substr(0, nl);
            g.inbuf.erase(0, nl + 1);
            return true;
        }
        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
        if (remaining < 0) return false;
        pollfd p{g.clientFd, POLLIN, 0};
        int r = poll(&p, 1, static_cast<int>(remaining));
        if (r <= 0) return false;
        char buf[65536];
        ssize_t n = recv(g.clientFd, buf, sizeof buf, 0);
        if (n <= 0) {
            dropClient(n == 0 ? "closed by client" : strerror(errno));
            return false;
        }
        g.inbuf.append(buf, static_cast<size_t>(n));
    }
}

// ---------------------------------------------------------------- game access

char const* modeName(PlayerObject* p) {
    if (p->m_isShip)   return "ship";
    if (p->m_isBall)   return "ball";
    if (p->m_isBird)   return "ufo";
    if (p->m_isDart)   return "wave";
    if (p->m_isRobot)  return "robot";
    if (p->m_isSpider) return "spider";
    if (p->m_isSwing)  return "swing";
    return "cube";
}

std::string stateJson(PlayLayer* pl) {
    auto p = pl->m_player1;
    return fmt::format(
        R"({{"tick":{},"x":{:.6f},"y":{:.6f},"yvel":{:.6f},"rot":{:.4f},"mode":"{}","upside_down":{},"mini":{},"speed":{:.4f},"on_ground":{},"dead":{},"started":{},"completed":{},"end_anim":{},"locked":{},"percent":{:.4f},"attempts":{},"practice":{},"playing":{},"level_id":{},"level_length":{:.3f}}})",
        g.tick,
        p->getPositionX(), p->getPositionY(), p->m_yVelocity, p->getRotation(),
        modeName(p), p->m_isUpsideDown, p->m_vehicleSize < 1.0f, p->m_playerSpeed,
        p->m_isOnGround, p->m_isDead, pl->m_started, pl->m_hasCompletedLevel,
        pl->m_levelEndAnimationStarted, p->m_isLocked, pl->getCurrentPercent(),
        pl->m_attempts, pl->m_isPracticeMode, play.active,
        pl->m_level ? static_cast<int>(pl->m_level->m_levelID.value()) : 0,
        pl->m_levelLength);
}

void doTick(PlayLayer* pl) {
    g.stepping = true;
    pl->update(kTickDt);
    g.stepping = false;
    g.tick++;
}

void applyHold(PlayLayer* pl, bool down) {
    pl->handleButton(down, 1, true);
}

struct Fnv {
    uint64_t h = 1469598103934665603ull;
    void add(void const* data, size_t n) {
        auto b = static_cast<unsigned char const*>(data);
        for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    }
    void addTick(PlayerObject* p) {
        double x = p->getPositionX(), y = p->getPositionY(), v = p->m_yVelocity;
        unsigned char d = p->m_isDead ? 1 : 0;
        add(&x, sizeof x); add(&y, sizeof y); add(&v, sizeof v); add(&d, 1);
    }
};

void appendTrace(std::string& out, PlayerObject* p) {
    out += fmt::format("[{:.6f},{:.6f},{:.6f},{}]", p->getPositionX(), p->getPositionY(), p->m_yVelocity, p->m_isDead ? 1 : 0);
}

// ---------------------------------------------------------------- json helpers

std::string err(std::string_view what) { return fmt::format(R"({{"ok":false,"error":"{}"}})", what); }

bool getBool(matjson::Value const& v, std::string_view key, bool def) {
    auto f = v.get(key);
    if (!f.isOk()) return def;
    auto r = f.unwrap().asBool();
    return r.isOk() ? r.unwrap() : def;
}
int64_t getInt(matjson::Value const& v, std::string_view key, int64_t def) {
    auto f = v.get(key);
    if (!f.isOk()) return def;
    auto r = f.unwrap().asInt();
    return r.isOk() ? static_cast<int64_t>(r.unwrap()) : def;
}
bool hasNonNull(matjson::Value const& v, std::string_view key) {
    auto f = v.get(key);
    return f.isOk() && !f.unwrap().isNull();
}

// ---------------------------------------------------------------- commands

std::string cmdGeometry(PlayLayer* pl) {
    char const* home = getenv("HOME");
    if (!home) return err("no HOME");
    std::filesystem::path dir = std::filesystem::path(home) / "Library/Application Support/GeoDashML/levels";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    int levelId = pl->m_level ? static_cast<int>(pl->m_level->m_levelID.value()) : 0;
    auto path = dir / fmt::format("{}.json", levelId);
    std::ofstream f(path);
    if (!f) return err("cannot open output file");
    f << fmt::format(R"({{"level_id":{},"level_length":{:.3f},"objects":[)", levelId, pl->m_levelLength);
    int count = 0;
    for (auto obj : CCArrayExt<GameObject*>(pl->m_objects)) {
        if (!obj) continue;
        auto const& r = obj->getObjectRect();
        f << (count ? ",\n" : "\n")
          << fmt::format(R"({{"id":{},"type":{},"x":{:.3f},"y":{:.3f},"rot":{:.2f},"sx":{:.3f},"sy":{:.3f},"decoration":{},"rect":[{:.3f},{:.3f},{:.3f},{:.3f}]}})",
                 obj->m_objectID, static_cast<int>(obj->m_objectType),
                 obj->getPositionX(), obj->getPositionY(), obj->getRotation(),
                 obj->getScaleX(), obj->getScaleY(), obj->m_isDecoration,
                 r.origin.x, r.origin.y, r.size.width, r.size.height);
        count++;
    }
    f << "\n]}\n";
    return fmt::format(R"({{"ok":true,"path":"{}","count":{}}})", path.string(), count);
}

std::string cmdLoadMain(int64_t id) {
    auto level = GameLevelManager::sharedState()->getMainLevel(static_cast<int>(id), false);
    if (!level) return err("no such main level");
    auto scene = PlayLayer::scene(level, false, false);
    if (!scene) return err("scene creation failed");
    CCDirector::sharedDirector()->replaceScene(scene);
    g.tick = 0;
    return R"({"ok":true})";
}

std::string handle(std::string const& line) {
    auto parsed = matjson::parse(line);
    if (!parsed.isOk()) return err("bad json");
    matjson::Value const v = parsed.unwrap();
    if (!v.isObject()) return err("bad json");
    auto cmdF = v.get("cmd");
    if (!cmdF.isOk()) return err("bad json");
    auto cmdR = cmdF.unwrap().asString();
    if (!cmdR.isOk()) return err("bad json");
    std::string cmd = cmdR.unwrap();

    auto pl = PlayLayer::get();

    if (cmd == "ping") {
        return fmt::format(R"({{"ok":true,"bridge":"{}","gd":"{}","in_level":{}}})", kBridgeVersion, kGdVersion, pl != nullptr);
    }
    if (cmd == "render") {
        g.renderOn = getBool(v, "on", true);
        return R"({"ok":true})";
    }
    if (cmd == "load_main") {
        return cmdLoadMain(getInt(v, "id", 1));
    }
    if (cmd == "detach") {
        return R"({"ok":true,"detach":true})";
    }

    if (!pl || !pl->m_player1) return err("no level");

    if (cmd == "state") {
        return fmt::format(R"({{"ok":true,"state":{}}})", stateJson(pl));
    }
    if (cmd == "hold") {
        applyHold(pl, getBool(v, "down", true));
        return fmt::format(R"({{"ok":true,"state":{}}})", stateJson(pl));
    }
    if (cmd == "step") {
        int64_t n = getInt(v, "n", 1);
        if (n < 0 || n > kMaxTicks) return err("bad args");
        bool trace = getBool(v, "trace", false);
        if (hasNonNull(v, "hold")) applyHold(pl, getBool(v, "hold", false));
        std::string tr;
        if (trace) tr.reserve(static_cast<size_t>(n) * 48);
        for (int64_t i = 0; i < n; i++) {
            doTick(pl);
            if (trace) { if (i) tr += ','; appendTrace(tr, pl->m_player1); }
        }
        if (trace) return fmt::format(R"({{"ok":true,"state":{},"trace":[{}]}})", stateJson(pl), tr);
        return fmt::format(R"({{"ok":true,"state":{}}})", stateJson(pl));
    }
    if (cmd == "run") {
        int64_t ticks = getInt(v, "ticks", 0);
        if (ticks < 0 || ticks > kMaxTicks) return err("bad args");
        bool trace = getBool(v, "trace", false);
        bool stop  = getBool(v, "stop", false);  // end the run after the tick that kills the player or ends the level
        std::vector<std::pair<int64_t, bool>> inputs;
        auto inputsF = v.get("inputs");
        if (inputsF.isOk()) {
            auto const& arr = inputsF.unwrap();
            if (!arr.isArray()) return err("bad args");
            for (size_t i = 0;; i++) {
                auto item = arr.get(i);
                if (!item.isOk()) break;
                auto const& pair = item.unwrap();
                auto t = pair.get(0);
                auto d = pair.get(1);
                if (!t.isOk() || !d.isOk()) return err("bad args");
                auto ti = t.unwrap().asInt();
                auto db = d.unwrap().asBool();
                if (!ti.isOk() || !db.isOk()) return err("bad args");
                inputs.emplace_back(static_cast<int64_t>(ti.unwrap()), db.unwrap());
            }
        }
        Fnv hash;
        std::string tr;
        if (trace) tr.reserve(static_cast<size_t>(ticks) * 48);
        size_t next = 0;
        for (int64_t t = 0; t < ticks; t++) {
            while (next < inputs.size() && inputs[next].first <= t) {
                applyHold(pl, inputs[next].second);
                next++;
            }
            doTick(pl);
            hash.addTick(pl->m_player1);
            if (trace) { if (t) tr += ','; appendTrace(tr, pl->m_player1); }
            if (stop && (pl->m_player1->m_isDead || pl->m_hasCompletedLevel || pl->m_levelEndAnimationStarted)) break;
        }
        if (trace) return fmt::format(R"({{"ok":true,"state":{},"hash":"{}","trace":[{}]}})", stateJson(pl), hash.h, tr);
        return fmt::format(R"({{"ok":true,"state":{},"hash":"{}"}})", stateJson(pl), hash.h);
    }
    if (cmd == "reset") {
        if (getBool(v, "from_start", false)) pl->resetLevelFromStart();
        else pl->resetLevel();
        g.tick = 0;
        return fmt::format(R"({{"ok":true,"state":{}}})", stateJson(pl));
    }
    if (cmd == "practice") {
        pl->togglePracticeMode(getBool(v, "on", true));
        return fmt::format(R"({{"ok":true,"state":{}}})", stateJson(pl));
    }
    if (cmd == "checkpoint") {
        if (!pl->m_isPracticeMode) return err("practice off");
        g.marking = true;
        pl->markCheckpoint();
        g.marking = false;
        return fmt::format(R"({{"ok":true,"state":{}}})", stateJson(pl));
    }
    if (cmd == "geometry") {
        return cmdGeometry(pl);
    }
    if (cmd == "play") {
        // Same as pressing A: starts playback of solutions/<level_id>.json. It runs once the client
        // detaches (playback and lockstep are exclusive), so a client sends play, then detach.
        startPlayback(pl);
        if (!play.active) return err("no solution");
        return fmt::format(R"({{"ok":true,"inputs":{},"ticks":{}}})", play.inputs.size() - 1, play.ticks);
    }
    return err("unknown cmd");
}

// ---------------------------------------------------------------- solution playback

std::filesystem::path gdmlDir() {
    char const* home = getenv("HOME");
    return std::filesystem::path(home ? home : "") / "Library/Application Support/GeoDashML";
}

void stopPlayback(char const* why) {
    if (!play.active) return;
    play.active = false;
    log::info("bridge: playback stopped ({}) after {} ticks", why, play.tick);
}

// Loads solutions/<level_id>.json (gdsearch's format: "inputs": [[tick, down], ...], "ticks": n),
// resets the level from the start and begins playing it.
void startPlayback(PlayLayer* pl) {
    int levelId = pl->m_level ? static_cast<int>(pl->m_level->m_levelID.value()) : 0;
    auto path = gdmlDir() / "solutions" / fmt::format("{}.json", levelId);
    std::ifstream f(path);
    if (!f) {
        Notification::create(fmt::format("GDML: no solution for level {}", levelId), NotificationIcon::Error)->show();
        log::warn("bridge: no solution file at {}", path.string());
        return;
    }
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto parsed = matjson::parse(text);
    if (!parsed.isOk()) { log::warn("bridge: bad solution json at {}", path.string()); return; }
    auto const& v = parsed.unwrap();
    play.inputs.clear();
    play.inputs.emplace_back(0, false);  // start released, like the search and the env
    auto inputsF = v.get("inputs");
    if (inputsF.isOk()) {
        auto const& arr = inputsF.unwrap();
        for (size_t i = 0;; i++) {
            auto item = arr.get(i);
            if (!item.isOk()) break;
            auto t = item.unwrap().get(0), d = item.unwrap().get(1);
            if (!t.isOk() || !d.isOk()) break;
            auto ti = t.unwrap().asInt(); auto db = d.unwrap().asBool();
            if (ti.isOk() && db.isOk()) play.inputs.emplace_back(static_cast<int64_t>(ti.unwrap()), db.unwrap());
        }
    }
    play.ticks  = getInt(v, "ticks", 0) + 240;  // a second past the end so the end portal is reached
    play.next   = 0;
    play.tick   = 0;
    play.carry  = 0;
    play.last   = Clock::now();
    play.active = true;
    pl->resetLevelFromStart();
    g.tick = 0;
    int attempts = static_cast<int>(getInt(v, "attempts", 0));
    Notification::create(fmt::format("GDML: playing solution ({} inputs, found in {} attempts)", play.inputs.size() - 1, attempts))->show();
    log::info("bridge: playback started from {} ({} inputs, {} ticks)", path.string(), play.inputs.size() - 1, play.ticks);
}

// Called every frame: advances the physics by the real time elapsed, 240 ticks per second.
void servicePlayback() {
    if (!play.active || g.attached()) return;
    auto pl = PlayLayer::get();
    if (!pl || !pl->m_player1) { stopPlayback("left the level"); return; }
    auto now = Clock::now();
    play.carry += std::chrono::duration<double>(now - play.last).count() * 240.0;
    play.last = now;
    int64_t due = static_cast<int64_t>(play.carry);
    if (due > 24) due = 24;  // after a stall, do not spiral
    play.carry -= static_cast<double>(due);
    for (int64_t i = 0; i < due && play.active; i++) {
        while (play.next < play.inputs.size() && play.inputs[play.next].first <= play.tick) {
            applyHold(pl, play.inputs[play.next].second);
            play.next++;
        }
        doTick(pl);
        play.tick++;
        if (pl->m_player1->m_isDead) stopPlayback("died");
        else if (pl->m_levelEndAnimationStarted || pl->m_hasCompletedLevel) stopPlayback("reached the end");
        else if (play.tick >= play.ticks) stopPlayback("schedule exhausted");
    }
}

// Services the attached client until the frame budget runs out.
void service() {
    if (g.listenFd < 0) return;
    if (!g.attached()) {
        acceptClient();
        if (!g.attached()) return;
    }
    auto deadline = Clock::now() + std::chrono::milliseconds(g.renderOn ? 8 : 1000);
    while (g.attached()) {
        std::string line;
        if (!readLine(line, deadline)) break;
        std::string reply = handle(line);
        reply += '\n';
        if (!sendAll(reply)) { dropClient("send failed"); break; }
        if (reply.find("\"detach\":true") != std::string::npos) { dropClient("detach"); break; }
    }
}

} // namespace

// ---------------------------------------------------------------- hooks

// The per-frame hook. CCScheduler::update looked like the obvious choice but on macOS it
// fires only sporadically outside the loading screen; the display-link director's mainLoop
// runs once per rendered frame everywhere.
class $modify(BridgeDirector, CCDisplayLinkDirector) {
    void mainLoop() {
        CCDisplayLinkDirector::mainLoop();
        service();
        servicePlayback();
    }
};

// A starts or stops solution playback while in a level with no client attached. The key is
// swallowed so the game's own binding (platformer left) does not see it.
class $modify(BridgeKeys, CCKeyboardDispatcher) {
    bool dispatchKeyboardMSG(enumKeyCodes key, bool down, bool repeat, double timestamp) {
        if (key == enumKeyCodes::KEY_A && !g.attached()) {
            auto pl = PlayLayer::get();
            if (pl && pl->m_player1) {
                if (down && !repeat) {
                    if (play.active) stopPlayback("key");
                    else startPlayback(pl);
                }
                return true;
            }
        }
        return CCKeyboardDispatcher::dispatchKeyboardMSG(key, down, repeat, timestamp);
    }
};

class $modify(BridgeGameLayer, GJBaseGameLayer) {
    void update(float dt) {
        // While attached or playing back, the game's own frame update is swallowed; only bridge-driven ticks pass.
        if ((g.attached() || play.active) && !g.stepping) return;
        GJBaseGameLayer::update(dt);
    }
};

class $modify(BridgePlayLayer, PlayLayer) {
    void delayedResetLevel() {
        if (g.attached()) return;  // resets happen only on command while attached
        stopPlayback("reset");
        PlayLayer::delayedResetLevel();
    }
    void setupHasCompleted() {
        PlayLayer::setupHasCompleted();
        g.tick = 0;
    }
    // Reaching the end portal starts an end animation that moves the player toward the portal with
    // cocos actions, driven by the scheduler in real time, which runs while the window is frontmost.
    // Under the bridge the next reset and checkpoint placement happen while it still runs, so it
    // dragged the player away from respawns and freshly placed checkpoints (x off by a few units, by
    // hundreds, or to 3.4 million; 2026-10-20), and its completion callback showed the game's level
    // complete screen. While a client is attached only the state changes are kept: the flag the
    // bridge reports as end_anim and the player lock.
    void playEndAnimationToPos(cocos2d::CCPoint position) {
        if (!g.attached()) {
            PlayLayer::playEndAnimationToPos(position);
            return;
        }
        m_levelEndAnimationStarted = true;
        if (m_player1) m_player1->m_isLocked = true;
        if (m_player2) m_player2->m_isLocked = true;
    }
    // Practice mode places its own checkpoints every couple of seconds of play; after a long episode
    // resetLevel() then respawns at the game's checkpoint, not the client's (2026-10-17). While a
    // client is attached only the bridge's `checkpoint` command may place one.
    void markCheckpoint() {
        if (g.attached() && !g.marking) return;
        PlayLayer::markCheckpoint();
    }
    // Music start spins on FMOD's stream-open state while holding the audio lock in a tight
    // loop, which can starve FMOD's own thread and livelock inside resetLevel. The agent does
    // not need audio, so skip it entirely while a client is attached.
    void startMusic() {
        if (g.attached()) return;
        PlayLayer::startMusic();
    }
    void prepareMusic(bool dontWait) {
        if (g.attached()) return;
        PlayLayer::prepareMusic(dontWait);
    }
};

// The automatic practice checkpoints start here (PlayerObject::updateCheckpointTest decides when).
class $modify(BridgePlayer, PlayerObject) {
    void tryPlaceCheckpoint() {
        if (g.attached()) return;
        PlayerObject::tryPlaceCheckpoint();
    }
};

// Entering practice mode plays the practice song through FMODAudioEngine::playMusic, which goes
// loadMusic -> preloadMusic -> the same Sound::getOpenState spin, and livelocked the game in
// Phase 2 (see the notebook, 2026-10-16). Loading a level while attached goes
// PlayLayer::setupHasCompleted -> loadMusic -> preloadMusic and livelocked the same way on
// 2026-10-20 (bridge 0.1.5). No music of any kind while attached: loadMusic is the common step.
class $modify(BridgeAudio, FMODAudioEngine) {
    void playMusic(gd::string path, bool shouldLoop, float fadeInTime, int channel) {
        if (g.attached()) return;
        FMODAudioEngine::playMusic(path, shouldLoop, fadeInTime, channel);
    }
    void loadMusic(gd::string path, float speed, float unknown, float volume, bool shouldLoop, int musicID, int channelID, bool dontReset) {
        if (g.attached()) return;
        FMODAudioEngine::loadMusic(path, speed, unknown, volume, shouldLoop, musicID, channelID, dontReset);
    }
    void loadMusic(gd::string path) {
        if (g.attached()) return;
        FMODAudioEngine::loadMusic(path);
    }
};

$on_mod(Loaded) {
    startServer();
}
