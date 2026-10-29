"""Execute a notebook top to bottom in a fresh kernel and store the outputs in place.
(Minimal replacement for `jupyter nbconvert --execute`, using only jupyter_client + ipykernel.)

    python tools/execute_notebook.py notebooks/velib_question_A.ipynb
"""
import json
import sys
import time
from pathlib import Path

from jupyter_client.manager import start_new_kernel


def execute(path: Path, timeout: int = 3600) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    km, kc = start_new_kernel(kernel_name="python3", cwd=str(path.parent))
    failed = None
    try:
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        for i, cell in enumerate(code_cells, 1):
            start = time.time()
            source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            msg_id = kc.execute(source)
            outputs, count = [], None
            while True:
                msg = kc.get_iopub_msg(timeout=timeout)
                if msg["parent_header"].get("msg_id") != msg_id:
                    continue
                kind, content = msg["msg_type"], msg["content"]
                if kind == "status" and content["execution_state"] == "idle":
                    break
                if kind == "execute_input":
                    count = content["execution_count"]
                elif kind == "stream":
                    if outputs and outputs[-1]["output_type"] == "stream" and outputs[-1]["name"] == content["name"]:
                        outputs[-1]["text"] += content["text"]
                    else:
                        outputs.append({"output_type": "stream", "name": content["name"], "text": content["text"]})
                elif kind in ("execute_result", "display_data"):
                    out = {"output_type": kind, "data": content["data"], "metadata": content.get("metadata", {})}
                    if kind == "execute_result":
                        out["execution_count"] = content["execution_count"]
                    outputs.append(out)
                elif kind == "error":
                    outputs.append({"output_type": "error", "ename": content["ename"],
                                    "evalue": content["evalue"], "traceback": content["traceback"]})
            kc.get_shell_msg(timeout=timeout)
            cell["outputs"], cell["execution_count"] = outputs, count
            print(f"cell {i}/{len(code_cells)} done in {time.time() - start:.0f}s", flush=True)
            errors = [o for o in outputs if o["output_type"] == "error"]
            if errors:
                failed = f"cell {i}: {errors[0]['ename']}: {errors[0]['evalue']}"
                break
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if failed:
        raise SystemExit(f"notebook failed at {failed}")


if __name__ == "__main__":
    execute(Path(sys.argv[1]).resolve())
