from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ENVIRONMENT_TAGS = {
    "AbandonedCable": ["infra", "mix", "large", "weather"],
    "AbandonedFactory": ["infra", "outdoor", "medium", "dusty"],
    "AbandonedFactory2": ["infra", "indoor", "small"],
    "AbandonedSchool": ["infra", "mix", "large"],
    "AmericanDiner": ["domestic", "indoor", "medium"],
    "AmusementPark": ["rural", "outdoor", "medium"],
    "AncientTowns": ["rural", "outdoor", "medium"],
    "Antiquity3D": ["thematic", "outdoor", "large", "day-night"],
    "Apocalyptic": ["thematic", "outdoor", "medium"],
    "ArchVizTinyHouseDay": ["domestic", "indoor", "small", "day-night"],
    "ArchVizTinyHouseNight": ["domestic", "indoor", "small", "day-night"],
    "BrushifyMoon": ["nature", "outdoor", "large"],
    "CarWelding": ["infra", "indoor", "medium", "dynamic"],
    "CastleFortress": ["rural", "mix", "large"],
    "CoalMine": ["infra", "indoor", "medium"],
    "ConstructionSite": ["infra", "outdoor", "medium", "lighting"],
    "CountryHouse": ["domestic", "indoor", "small"],
    "CyberPunkDowntown": ["thematic", "indoor", "medium", "dynamic"],
    "Cyberpunk": ["thematic", "outdoor", "medium"],
    "DesertGasStation": ["rural", "outdoor", "small", "lighting"],
    "Downtown": ["urban", "outdoor", "medium"],
    "EndofTheWorld": ["rural", "outdoor", "medium", "dusty"],
    "FactoryWeather": ["infra", "outdoor", "large", "weather"],
    "Fantasy": ["rural", "outdoor", "medium"],
    "ForestEnv": ["nature", "outdoor", "large"],
    "Gascola": ["nature", "outdoor", "medium", "day-night"],
    "GothicIsland": ["rural", "mix", "large"],
    "GreatMarsh": ["rural", "outdoor", "medium", "fog"],
    "HQWesternSaloon": ["rural", "mix", "medium"],
    "HongKong": ["urban", "outdoor", "small", "night"],
    "Hospital": ["infra", "indoor", "large"],
    "House": ["domestic", "indoor", "medium"],
    "IndustrialHangar": ["infra", "outdoor", "medium"],
    "JapaneseAlley": ["urban", "mix", "small", "rain", "night"],
    "JapaneseCity": ["urban", "mix", "medium"],
    "MiddleEast": ["urban", "outdoor", "medium", "weather"],
    "ModUrbanCity": ["urban", "mix", "medium"],
    "ModernCityDowntown": ["urban", "mix", "medium"],
    "ModularNeighborhood": ["urban", "outdoor", "large"],
    "ModularNeighborhoodIntExt": ["urban", "mix", "large"],
    "NordicHarbor": ["urban", "outdoor", "medium"],
    "Ocean": ["nature", "outdoor", "medium", "dynamic"],
    "Office": ["domestic", "indoor", "medium"],
    "OldBrickHouseDay": ["domestic", "indoor", "small", "day-night"],
    "OldBrickHouseNight": ["domestic", "indoor", "small", "day-night"],
    "OldIndustrialCity": ["infra", "outdoor", "medium"],
    "OldScandinavia": ["nature", "outdoor", "large", "day-night"],
    "OldTownFall": ["urban", "outdoor", "medium", "season"],
    "OldTownNight": ["urban", "outdoor", "medium", "season"],
    "OldTownSummer": ["urban", "outdoor", "medium", "season"],
    "OldTownWinter": ["urban", "outdoor", "medium", "season"],
    "PolarSciFi": ["thematic", "mix", "medium", "snow"],
    "Prison": ["infra", "mix", "large"],
    "Restaurant": ["domestic", "indoor", "medium"],
    "RetroOffice": ["domestic", "indoor", "small"],
    "Rome": ["thematic", "outdoor", "medium"],
    "Ruins": ["nature", "outdoor", "medium"],
    "SeasideTown": ["rural", "outdoor", "medium"],
    "SeasonalForestAutumn": ["nature", "outdoor", "large", "season"],
    "SeasonalForestSpring": ["nature", "outdoor", "large", "season"],
    "SeasonalForestSummerNight": ["nature", "outdoor", "large", "season", "night"],
    "SeasonalForestWinter": ["nature", "outdoor", "large", "season"],
    "SeasonalForestWinterNight": ["nature", "outdoor", "large", "season", "night"],
    "Sewerage": ["infra", "indoor", "large"],
    "ShoreCaves": ["nature", "outdoor", "small"],
    "Slaughter": ["thematic", "mix", "medium"],
    "SoulCity": ["rural", "mix", "medium", "dynamic"],
    "Supermarket": ["domestic", "indoor", "medium", "rain"],
    "TerrainBlending": ["nature", "outdoor", "small"],
    "UrbanConstruction": ["urban", "outdoor", "medium"],
    "VictorianStreet": ["urban", "outdoor", "medium"],
    "WaterMillDay": ["rural", "outdoor", "medium", "day-night"],
    "WaterMillNight": ["rural", "outdoor", "medium", "day-night"],
    "WesternDesertTown": ["rural", "mix", "large", "day-night"],
}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_yaml(payload), encoding="utf-8")


def scene_tags(env_name: str) -> list[str]:
    return ["scene", *ENVIRONMENT_TAGS.get(env_name, [])]


def _yaml(value: Any, indent: int = 0) -> str:
    lines = _yaml_lines(value, indent)
    return "\n".join(lines) + "\n"


def _yaml_lines(value: Any, indent: int) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if _is_inline_number_list(item):
                lines.append(f"{prefix}{key}: {_inline_list(item)}")
            elif isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_scalar(item)}")
        return lines
    if isinstance(value, list):
        if _is_inline_number_list(value):
            return [f"{prefix}{_inline_list(value)}"]
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_scalar(item)}")
        return lines
    return [f"{prefix}{_scalar(value)}"]


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value == 0.0:
            return "0.0"
        return str(value)
    return json.dumps(str(value))


def _is_inline_number_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, (int, float)) and not isinstance(item, bool)
        for item in value
    )


def _inline_list(value: list[Any]) -> str:
    return "[" + ",".join(_scalar(item) for item in value) + "]"
