"""Material choices for the interactive device builder.

Only the baseline selection is represented in the current COMSOL campaign.
Alternative materials are deliberately selectable for design exploration, but
they must block predictions until matching simulation data exists.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerSpec:
    key: str
    label: str
    role: str
    baseline: str
    alternatives: tuple[str, ...]
    color: str
    thickness: str

    @property
    def options(self) -> tuple[str, ...]:
        return (self.baseline, *self.alternatives)


LAYERS: tuple[LayerSpec, ...] = (
    LayerSpec(
        key="front_barrier",
        label="Front barrier (optional)",
        role="Environmental barrier or encapsulant on the light-facing side",
        baseline="None",
        alternatives=("Al2O3", "Parylene C", "EVA / POE encapsulant"),
        color="#CBD5E1",
        thickness="0 nm",
    ),
    LayerSpec(
        key="substrate",
        label="Substrate",
        role="Mechanical support and light entry",
        baseline="Glass",
        alternatives=("Flexible PET",),
        color="#DCE9F7",
        thickness="1.1 mm",
    ),
    LayerSpec(
        key="front_contact",
        label="Front contact",
        role="Transparent conducting electrode",
        baseline="ITO",
        alternatives=("FTO", "AZO"),
        color="#0891B2",
        thickness="150 nm",
    ),
    LayerSpec(
        key="htl",
        label="Hole transport layer",
        role="Carries holes to the front contact",
        baseline="NiOx + MeO-2PACz",
        alternatives=("PTAA", "PEDOT:PSS", "Spiro-OMeTAD"),
        color="#7C3AED",
        thickness="20 nm",
    ),
    LayerSpec(
        key="htl_absorber_barrier",
        label="HTL / absorber interlayer (optional)",
        role="Internal passivation or ion-blocking layer at the hole-selective interface",
        baseline="None",
        alternatives=(
            "MeO-2PACz SAM",
            "2D perovskite",
            "Ultrathin Al2O3",
            "SAM + Al2O3 bilayer",
        ),
        color="#A78BFA",
        thickness="0 nm",
    ),
    LayerSpec(
        key="absorber",
        label="Perovskite absorber",
        role="Absorbs light and generates charge",
        baseline="Cs0.2FA0.8PbI3",
        alternatives=("FAPbI3", "MAPbI3", "Mixed I/Br perovskite"),
        color="#2E1A47",
        thickness="450 nm",
    ),
    LayerSpec(
        key="absorber_etl_barrier",
        label="Absorber / ETL interlayer (optional)",
        role="Internal passivation or ion-blocking layer at the electron-selective interface",
        baseline="None",
        alternatives=(
            "2D perovskite",
            "Ultrathin Al2O3",
            "LiF",
            "2D perovskite + LiF bilayer",
        ),
        color="#60A5FA",
        thickness="0 nm",
    ),
    LayerSpec(
        key="etl",
        label="Electron transport layer",
        role="Carries electrons to the rear contact",
        baseline="C60 + BCP",
        alternatives=("PCBM + BCP", "SnO2", "TiO2"),
        color="#2563EB",
        thickness="29 nm",
    ),
    LayerSpec(
        key="rear_contact",
        label="Rear contact",
        role="Metal electrode",
        baseline="Silver (Ag)",
        alternatives=("Gold (Au)", "Copper (Cu)", "Carbon"),
        color="#94A3B8",
        thickness="100 nm",
    ),
    LayerSpec(
        key="rear_barrier",
        label="Rear barrier (optional)",
        role="Environmental barrier or edge-seal layer behind the rear electrode",
        baseline="None",
        alternatives=("Al2O3", "Parylene C", "Epoxy edge seal"),
        color="#CBD5E1",
        thickness="0 nm",
    ),
)

BASELINE_MATERIALS = {layer.key: layer.baseline for layer in LAYERS}


def selected_materials(state: dict) -> dict[str, str]:
    """Read the builder selection from state, defaulting to the trained stack."""
    return {
        layer.key: str(state.get(f"material_{layer.key}", layer.baseline))
        for layer in LAYERS
    }


def is_baseline_design(materials: dict[str, str]) -> bool:
    return all(materials.get(key) == value for key, value in BASELINE_MATERIALS.items())


def changed_layers(materials: dict[str, str]) -> tuple[LayerSpec, ...]:
    return tuple(layer for layer in LAYERS if materials.get(layer.key) != layer.baseline)
