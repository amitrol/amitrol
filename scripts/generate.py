"""Erzeugt ein Bild mit FLUX.1 Krea (Hugging Face Space) aus prompts/aktuell.txt."""
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from gradio_client import Client

SPACE = "mcp-tools/FLUX.1-Krea-dev"
ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = ROOT / "prompts" / "aktuell.txt"
OUT_DIR = ROOT / "bilder"


def main() -> int:
    prompt = PROMPT_FILE.read_text(encoding="utf-8").strip()
    if not prompt:
        print("Kein Prompt in prompts/aktuell.txt")
        return 1

    width = int(os.environ.get("WIDTH", "1024"))
    height = int(os.environ.get("HEIGHT", "768"))
    token = os.environ.get("HF_TOKEN") or None

    # Neuere gradio_client-Versionen nutzen "token", ältere "hf_token".
    try:
        client = Client(SPACE, token=token)
    except TypeError:
        client = Client(SPACE, hf_token=token)
    result = client.predict(
        prompt=prompt,
        width=width,
        height=height,
        randomize_seed=True,
        api_name="/infer",
    )

    # Ergebnis ist meist (bild, seed); bild kann Pfad oder Dict sein.
    image, seed = (result[0], result[1]) if isinstance(result, (list, tuple)) else (result, None)
    if isinstance(image, dict):
        image = image.get("path") or image.get("url")

    OUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = Path(str(image)).suffix or ".webp"
    target = OUT_DIR / f"{stamp}{suffix}"
    shutil.copy(image, target)

    meta = {"prompt": prompt, "seed": seed, "width": width, "height": height, "space": SPACE}
    target.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Gespeichert: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
