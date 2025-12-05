import json
import random
from pathlib import Path

# Load data
mortar_data = json.loads(Path("../../frontend/data/mortar.json").read_text())
rock_data = json.loads(Path("../../frontend/data/rock.json").read_text())


# Function to calculate Hirsch model with noise addition
def calculate_hirsch_with_noise(mortar_id, rock_id, rock_ratio, eta=0.3, noise_std=2.0):
    mortar = mortar_data[mortar_id]["properties"]
    rock = rock_data[rock_id]["properties"]
    fm = mortar["compressive_strength_mpa"]
    fr = rock["compressive_strength_mpa"]  # / 4
    r = rock_ratio
    # Hirsch model calculation
    # add some noise to eta to make it more realistic
    eta = eta + abs(random.gauss(0, 0.1))
    hirsch_strength = (1 - r) * fm + r * fr + eta * r * (1 - r) * (fr - fm)
    # Add Gaussian noise
    noise = random.gauss(0, noise_std)
    return max(hirsch_strength + noise, 0)  # Ensure non-negative strength


# Generate synthetic data
concrete_synthetic = {}
mortar_ids = list(mortar_data.keys())
rock_ids = list(rock_data.keys())
rock_ratios = [0.3, 0.4, 0.5]

for mortar_id in mortar_ids:
    for rock_id in rock_ids:
        for ratio in rock_ratios:
            key = f"{mortar_id}-{rock_id}-{int(ratio * 10)}"
            strength = calculate_hirsch_with_noise(mortar_id, rock_id, ratio)
            concrete_synthetic[key] = {
                "mortar_id": mortar_id,
                "rock_id": rock_id,
                "rock_ratio": ratio,
                "concrete_compressive_strength_mpa": round(strength, 2),
            }

# Write out the synthetic data
output_path = Path("../../frontend/data/concrete.json")
output_path.write_text(json.dumps(concrete_synthetic, indent=4))

print("Synthetic data generated and saved to ../../frontend/data/concrete.json")

# print out the range of the values
print(
    f"Range of values: {min(concrete_synthetic.values(), key=lambda x: x['concrete_compressive_strength_mpa'])} to {max(concrete_synthetic.values(), key=lambda x: x['concrete_compressive_strength_mpa'])}"
)
