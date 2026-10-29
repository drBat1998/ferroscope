# My colors of choise
COLOR_KEYS = {
    "m": "magenta",
    "c": "cyan",
    "y": "yellow",
    "g": "gray",
    "mg": "magma",
    "h": "HiLo",
}

ALIASES = {"grey": "gray", "m": "mg"}

# The best of general observation

DEFAULT_ORDER = ["magenta", "cyan", "yellow"]


def default_color(i) -> str:
    index = i % len(DEFAULT_ORDER)
    return DEFAULT_ORDER[index]


def parse_color(answer: str, default: str) -> str:
    cleaned = answer.strip().lower()

    if cleaned == "":
        return default

    if cleaned in COLOR_KEYS:
        return COLOR_KEYS[cleaned]

    for name in COLOR_KEYS.values():
        if cleaned == name.lower():
            return COLOR_KEYS[name.lower()]

    if cleaned in ALIASES:
        return ALIASES[cleaned]

    else:
        raise ValueError(f"unknown color {answer!r}, use {'/'.join(COLOR_KEYS)}")


def colormap_list(channel_names: list[str], colors: dict[str, str]) -> list[str]:
    result = []
    for i, name in enumerate(channel_names):
        color = colors.get(name, default_color(i))
        result.append(color)
    return result
