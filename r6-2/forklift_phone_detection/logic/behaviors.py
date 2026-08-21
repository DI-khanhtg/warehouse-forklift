"""Canonical behavior names shared by geometry, state, events, and drawing."""


NORMAL = "NORMAL"
PHONE_PRESENT = "PHONE_PRESENT"
PHONE_CALL = "PHONE_CALL"
HANDHELD_PHONE_USE = "HANDHELD_PHONE_USE"
WATCHING_PHONE = "WATCHING_PHONE"

POSITIVE_BEHAVIORS = frozenset(
    {PHONE_CALL, HANDHELD_PHONE_USE, WATCHING_PHONE}
)

_ALIASES = {
    NORMAL: NORMAL,
    PHONE_PRESENT: PHONE_PRESENT,
    PHONE_CALL: PHONE_CALL,
    "PHONE_NEAR_HEAD": PHONE_CALL,
    "CALLING": PHONE_CALL,
    HANDHELD_PHONE_USE: HANDHELD_PHONE_USE,
    "TEXTING_OR_HOLDING_PHONE": HANDHELD_PHONE_USE,
    "HANDHELD": HANDHELD_PHONE_USE,
    WATCHING_PHONE: WATCHING_PHONE,
    "WATCHING": WATCHING_PHONE,
}


def canonical_behavior(value) -> str:
    """Normalize legacy/visualization labels before final-state fusion."""
    name = str(value or NORMAL).strip().upper()
    return _ALIASES.get(name, NORMAL)


def is_using_phone_behavior(value) -> bool:
    return canonical_behavior(value) in POSITIVE_BEHAVIORS
