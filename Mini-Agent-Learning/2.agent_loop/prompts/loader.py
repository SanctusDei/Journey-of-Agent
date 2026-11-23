from pathlib import Path



def load_prompt(path: str) -> str:
    prompt_file = Path(path)
    if not prompt_file.exists():
        raise FileNotFoundError(f"提示词文件不存在: {prompt_file.resolve()}")
    return prompt_file.read_text(encoding="utf-8").strip()

if __name__ == "__main__":
    
    print(load_prompt("./maid.md"))
