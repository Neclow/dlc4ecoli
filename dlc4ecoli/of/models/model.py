import json

from types import SimpleNamespace

from torchvision.models.optical_flow import raft_small, raft_large

from .sea_raft.core.raft import RAFT


def load_model(
    model_name,
    weights="DEFAULT",
    config="spring-M",
    url="MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
):
    # FIXME: torch.compile causes torch._dynamo.exc.InternalTorchDynamoError
    if model_name == "raft_large":
        model = raft_large(weights=weights)
    elif model_name == "raft_small":
        model = raft_small(weights=weights)
    elif model_name == "sea_raft":
        if not config.endswith(".json"):
            config = config + ".json"
        json_path = f"dlc4ecoli/of/models/sea_raft/config/eval/{config}"

        with open(json_path, "r", encoding="utf-8") as f:
            args = SimpleNamespace(**json.load(f))

        model = RAFT.from_pretrained(url, args=args)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model
