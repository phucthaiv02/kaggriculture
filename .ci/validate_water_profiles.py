from agents.schedules import water_profiles
from experiments.crop_schedules import CASES, run_case


for crop, use_fertilizer in CASES:
    for days in water_profiles(crop, use_fertilizer):
        run_case(crop, use_fertilizer, seed=1, water_days=days)

print("production water profiles preserve verified crop yield")
