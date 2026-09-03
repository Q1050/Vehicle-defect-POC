"""Small, explicit relationship rules for troubleshooting hypotheses."""

HYPOTHESIS_RULES: dict[str, dict[str, float]] = {
    "lubrication_issue": {
        "oil_leak": 1.00,
        "low_engine_oil_warning": 1.00,
        "possible_knocking": 0.65,
        "low_oil_report": 0.85,
        "dashboard_indicator": 0.30,
    },
    "engine_running_irregularly": {
        "possible_misfire_pattern": 1.00,
        "possible_engine_vibration": 0.90,
        "vehicle_shaking": 0.85,
        "check_engine_warning": 0.75,
        "dashboard_indicator": 0.35,
    },
    "cooling_or_combustion_issue": {
        "possible_smoke": 1.00,
        "overheating_warning": 0.95,
        "overheating_report": 0.85,
        "dashboard_indicator": 0.25,
    },
    "belt_or_accessory_issue": {
        "broken_belt": 1.00,
        "possible_bearing_noise": 0.80,
        "possible_hissing": 0.35,
        "charging_warning": 0.75,
    },
    "tire_condition_issue": {
        "tire_wear": 1.00,
        "tire_pressure_warning": 0.90,
        "vehicle_pulling": 0.75,
        "steering_vibration": 0.70,
    },
    "corrosion_issue": {
        "rust": 1.00,
        "corrosion_report": 0.80,
    },
}


CONFLICT_RULES = [
    (
        {"possible_smoke"},
        {"no_visible_smoke"},
        "Video or visual evidence suggests smoke while another source reports no visible smoke.",
    ),
    (
        {"dashboard_indicator", "check_engine_warning", "low_engine_oil_warning",
         "overheating_warning", "tire_pressure_warning", "charging_warning"},
        {"no_warning_lights"},
        "Warning-indicator evidence conflicts with a report that no warning lights are visible.",
    ),
]
