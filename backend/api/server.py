import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import litserve as ls
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

# Environment configurations
PORT = int(os.environ.get("API_PORT", "9600"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
NUM_API_SERVERS = int(os.environ.get("NUM_API_SERVERS", "1"))
WORKERS_PER_DEVICE = int(os.environ.get("WORKERS_PER_DEVICE", "1"))

# Resolve data directory: if not set, try relative to current directory
if "DATA_DIR" in os.environ:
    DATA_DIR = os.environ.get("DATA_DIR")
else:
    DATA_DIR = "./data"  # can be symlink (dev) or real files (docker image - prod)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Pydantic models for request validation and Swagger docs
class MortarProperties(BaseModel):
    splitting_strength_mpa: float
    shrinkage_inches: float
    flexural_strength_mpa: float
    slump_inches: float
    compressive_strength_mpa: float
    poissons_ratio: float


class MortarMeta(BaseModel):
    gwp: float
    product_name_long: str
    manufacturer: str
    cost_per_pound: float


class CustomMortar(BaseModel):
    name: str
    description: str
    properties: MortarProperties
    meta: MortarMeta


class InputRequest(BaseModel):
    region_id: str = Field("R001", description="The ID of the region to predict for")
    desired_compressive_strength_mpa: float = Field(
        50, description="The desired compressive strength in MPa"
    )
    custom_mortars: List[CustomMortar] = []


def load_data_files(data_dir: str = DATA_DIR) -> Dict[str, Any]:
    """Load region, rock, mortar, and concrete data files."""
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"Data directory not found: {p}")
    return {
        "regions": json.loads((p / "region.json").read_text()),
        "rocks": json.loads((p / "rock.json").read_text()),
        "mortars": json.loads((p / "mortar.json").read_text()),
        "concrete": json.loads((p / "concrete.json").read_text()),
    }


def filter_concrete_samples(
    region_id: str,
    desired_strength: float,
    available_rocks: List[str],
    concrete_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Filter concrete.json experimental samples that meet the strength requirements.

    Returns samples where:
    - rock_id is in available_rocks for the region
    - concrete_compressive_strength_mpa >= desired_strength
    - concrete_compressive_strength_mpa < desired_strength * 1.333
    """
    if not available_rocks:
        return []

    # Convert available_rocks to a set for O(1) lookup
    available_rocks_set = set(available_rocks)
    upper_bound = desired_strength * 1.333

    # Filter in a single pass
    filtered = [
        {
            "rock_id": entry["rock_id"],
            "mortar_id": entry["mortar_id"],
            "rock_ratio": entry["rock_ratio"],
            "predicted_compressive_strength_mpa": entry[
                "concrete_compressive_strength_mpa"
            ],
        }
        for entry in concrete_data.values()
        if entry["rock_id"] in available_rocks_set
        and entry["concrete_compressive_strength_mpa"] >= desired_strength
        and entry["concrete_compressive_strength_mpa"] < upper_bound
    ]

    # Sort by predicted strength (ascending)
    filtered.sort(key=lambda x: x["predicted_compressive_strength_mpa"])
    return filtered


class ConcretePredictionAPI(ls.LitAPI):
    def setup(self, device):
        if device != "cpu":
            logger.warning(
                "ConcretePredictionAPI does not benefit from hardware acceleration. Use 'cpu'."
            )

        # Load data files
        try:
            self.data = load_data_files()
            logger.info("Loaded region, rock, mortar, and concrete data files.")
        except Exception as e:
            logger.error(f"Failed to load data files: {e}")
            self.data = {"regions": {}, "rocks": {}, "mortars": {}, "concrete": {}}

        # TODO: Load model here (e.g., PhysicsNN or PhysicsGPR)
        # self.model = ...
        logger.info("ConcretePredictionAPI initialized.")

    def decode_request(self, request: InputRequest) -> Dict[str, Any]:
        return request.model_dump()

    def predict(self, request_data) -> Dict[str, Any]:
        """Generate predictions for concrete strength."""
        region_id = request_data["region_id"]
        desired_strength = request_data["desired_compressive_strength_mpa"]
        custom_mortars = request_data["custom_mortars"]

        # Get available rocks for the region
        region = self.data["regions"].get(region_id)
        if not region:
            return {
                "predictions": [],
                "status": "error",
                "error": f"Unknown region_id: {region_id}",
            }

        available_rocks = region.get("available_rocks", [])
        if not available_rocks:
            logger.warning(f"No available rocks for region {region_id}, defaulting to all rocks")
            available_rocks = list(self.data["rocks"].keys())

        # if custom mortars are provided, we nede to simulate these using the model for the available rocks, and rock ratio in [0.4, 0.6]
        simulations = []
        if custom_mortars:
            simulations = self.simulate_custom_mortars(custom_mortars, available_rocks)

        # print(custom_mortars, available_rocks)
        # Filter concrete.json experimental samples that meet the strength requirements
        filtered_samples = filter_concrete_samples(
            region_id=region_id,
            desired_strength=desired_strength,
            available_rocks=available_rocks,
            concrete_data=self.data["concrete"],
        )

        # combine filtered experiments and simulations
        predictions = filtered_samples + simulations

        # Determine status based on results
        if not predictions:
            status = "partial"
        else:
            status = "success"

        return {
            "predictions": predictions,
            "status": status,
            "mocked": False,  # frontend displays this
        }

    def simulate_custom_mortars(self, custom_mortars: List[CustomMortar|dict], available_rocks: List[str]) -> List[Dict[str, Any]]:
        available_rock_info = {rock_id: self.data["rocks"].get(rock_id) for rock_id in available_rocks}
        all_mortars = self.data["mortars"]

        print(available_rock_info)
        print(f"Number of custom mortars: {len(custom_mortars)}")
        print(custom_mortars)
        print(f"Type of all mortars: {type(all_mortars)}")
        return []

if __name__ == "__main__":
    api = ConcretePredictionAPI(max_batch_size=1, api_path="/predict")
    server = ls.LitServer(
        api,
        accelerator="cpu",
        track_requests=True,
        workers_per_device=WORKERS_PER_DEVICE,
    )

    # Add CORS middleware to handle OPTIONS preflight requests
    # litserve uses FastAPI under the hood, so we can access the app
    try:
        app = server.app  # Access the underlying FastAPI app
        YES_VALUES = ["true", "1", "yes", "y"]
        if os.environ.get("DISABLE_RATE_LIMIT", "false").lower() in YES_VALUES:
            logger.info("Rate limiting disabled")
            pass
        else:
            # Rate limiting setup
            # We use default_limits and SlowAPIMiddleware to enforce limits globally
            # without needing to decorate specific routes (since litserve hides them)
            limiter = Limiter(key_func=get_remote_address, default_limits=["20/minute"])
            app.state.limiter = limiter
            app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
            app.add_middleware(SlowAPIMiddleware)

            logger.info("Rate limiter added: 20 requests/minute per IP")

        if os.environ.get("DISABLE_CORS", "false").lower() in YES_VALUES:
            logger.info("CORS disabled")
            pass
        else:
            allowed_origins = (
                ["*"]
                if os.environ.get("ENVIRONMENT") == "dev"
                else ["https://mindthemath.github.io", "https://concrete.math.computer"]
            )
            app.add_middleware(
                CORSMiddleware,
                allow_origins=allowed_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            logger.info(
                f"CORS middleware added successfully with origins: {allowed_origins}"
            )
    except AttributeError:
        logger.warning(
            "Could not access FastAPI app directly. CORS/RateLimit may not work properly."
        )

    server.run(
        port=PORT,
        host="0.0.0.0",
        log_level=LOG_LEVEL.lower(),
        num_api_servers=NUM_API_SERVERS,
        generate_client_file=False,
    )
