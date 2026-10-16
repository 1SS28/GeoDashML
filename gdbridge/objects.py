"""GameObjectType values as exported by the bridge, from Geode's Enums.hpp for GD 2.2081."""

from enum import IntEnum


class GameObjectType(IntEnum):
    Solid = 0
    Hazard = 2
    InverseGravityPortal = 3
    NormalGravityPortal = 4
    ShipPortal = 5
    CubePortal = 6
    Decoration = 7
    YellowJumpPad = 8
    PinkJumpPad = 9
    GravityPad = 10
    YellowJumpRing = 11
    PinkJumpRing = 12
    GravityRing = 13
    InverseMirrorPortal = 14
    NormalMirrorPortal = 15
    BallPortal = 16
    RegularSizePortal = 17
    MiniSizePortal = 18
    UfoPortal = 19
    Modifier = 20
    Breakable = 21
    SecretCoin = 22
    DualPortal = 23
    SoloPortal = 24
    Slope = 25
    WavePortal = 26
    RobotPortal = 27
    TeleportPortal = 28
    GreenRing = 29
    Collectible = 30
    UserCoin = 31
    DropRing = 32
    SpiderPortal = 33
    RedJumpPad = 34
    RedJumpRing = 35
    CustomRing = 36
    DashRing = 37
    GravityDashRing = 38
    CollisionObject = 39
    Special = 40
    SwingPortal = 41
    GravityTogglePortal = 42
    SpiderOrb = 43
    SpiderPad = 44
    EnterEffectObject = 45
    TeleportOrb = 46
    AnimatedHazard = 47


# Coarse roles for the observation one-hot. Everything not listed is ignored by the agent.
ROLE_OF_TYPE = {
    GameObjectType.Solid: "solid",
    GameObjectType.Slope: "solid",
    GameObjectType.Breakable: "solid",
    GameObjectType.Hazard: "hazard",
    GameObjectType.AnimatedHazard: "hazard",
    GameObjectType.YellowJumpPad: "pad",
    GameObjectType.PinkJumpPad: "pad",
    GameObjectType.RedJumpPad: "pad",
    GameObjectType.GravityPad: "pad",
    GameObjectType.SpiderPad: "pad",
    GameObjectType.YellowJumpRing: "orb",
    GameObjectType.PinkJumpRing: "orb",
    GameObjectType.RedJumpRing: "orb",
    GameObjectType.GravityRing: "orb",
    GameObjectType.GreenRing: "orb",
    GameObjectType.DropRing: "orb",
    GameObjectType.DashRing: "orb",
    GameObjectType.GravityDashRing: "orb",
    GameObjectType.SpiderOrb: "orb",
    GameObjectType.TeleportOrb: "orb",
    GameObjectType.CustomRing: "orb",
    GameObjectType.CubePortal: "portal",
    GameObjectType.ShipPortal: "portal",
    GameObjectType.BallPortal: "portal",
    GameObjectType.UfoPortal: "portal",
    GameObjectType.WavePortal: "portal",
    GameObjectType.RobotPortal: "portal",
    GameObjectType.SpiderPortal: "portal",
    GameObjectType.SwingPortal: "portal",
    GameObjectType.NormalGravityPortal: "gravity",
    GameObjectType.InverseGravityPortal: "gravity",
    GameObjectType.GravityTogglePortal: "gravity",
    GameObjectType.RegularSizePortal: "size",
    GameObjectType.MiniSizePortal: "size",
    GameObjectType.NormalMirrorPortal: "mirror",
    GameObjectType.InverseMirrorPortal: "mirror",
    GameObjectType.DualPortal: "dual",
    GameObjectType.SoloPortal: "dual",
    GameObjectType.TeleportPortal: "teleport",
    GameObjectType.Modifier: "modifier",
}

ROLES = sorted(set(ROLE_OF_TYPE.values()))


def role_of(type_value: int) -> str | None:
    try:
        return ROLE_OF_TYPE.get(GameObjectType(type_value))
    except ValueError:
        return None
