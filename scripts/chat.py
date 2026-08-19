"""Interactive Terminal Chat with Fine-Tuned Ultron (113M).

Supports streaming multi-turn conversation with ChatML formatting,
custom system prompts, and sampling temperature controls.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from rich.console import Console

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import UltronConfig
from model import UltronModel, load_ultron_state_dict
from scripts.generate import select_next_token
from tokenizer import UltronTokenizer

console = Console()


def load_instruct_model(
    checkpoint_dir: str | Path,
    device: torch.device,
) -> tuple[UltronModel, UltronTokenizer]:
    """Load model weights from fine-tuned SFT checkpoint directory."""
    path = Path(checkpoint_dir)
    config = UltronConfig()

    candidates = (
        path / "model.safetensors",
        path / "pytorch_model.bin",
        path / "pytorch_model" / "model.safetensors",
        path / "pytorch_model" / "pytorch_model.bin",
    )
    weight_path = next((p for p in candidates if p.is_file()), None)

    model = UltronModel(config)
    if weight_path is not None:
        if weight_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(str(weight_path))
        else:
            state_dict = torch.load(
                str(weight_path), map_location="cpu", weights_only=True
            )
        load_ultron_state_dict(model, state_dict)
        console.print(f"[dim]Loaded fine-tuned weights from {weight_path}[/dim]")
    else:
        console.print(
            f"[yellow]Warning: No checkpoint found at {checkpoint_dir}. Using randomly initialized model.[/yellow]"
        )

    model.to(device).eval()
    tokenizer = UltronTokenizer(config.tokenizer_name)
    return model, tokenizer


def chat_loop(
    model: UltronModel,
    tokenizer: UltronTokenizer,
    system_prompt: str,
    device: torch.device,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    max_new_tokens: int = 512,
) -> None:
    """Run interactive multi-turn chat session."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    console.print(
        "[bold cyan]═══════════════════════════════════════════════[/bold cyan]"
    )
    console.print(
        "[bold cyan]     🤖 ULTRON (113M) INTERACTIVE CHAT         [/bold cyan]"
    )
    console.print(
        "[bold cyan]═══════════════════════════════════════════════[/bold cyan]"
    )
    console.print(
        "[dim]Type your message below. Type 'exit', 'quit', or 'clear' to reset.[/dim]\n"
    )

    while True:
        try:
            user_input = console.input("[bold green]You > [/bold green]").strip()
        except KeyboardInterrupt, EOFError:
            console.print("\n[yellow]Exiting chat...[/yellow]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            console.print("[yellow]Farewell, Sire.[/yellow]")
            break
        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": system_prompt}]
            console.print("[dim]Conversation history cleared.[/dim]")
            continue

        messages.append({"role": "user", "content": user_input})

        # Apply ChatML template
        prompt_text = str(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        console.print("[bold blue]Ultron > [/bold blue]", end="")

        generated_tokens: list[int] = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                output = model(input_tensor)
                next_token_logits = output.logits[:, -1, :]

                selected = select_next_token(
                    next_token_logits,
                    input_tensor,
                    greedy=temperature == 0.0,
                    temperature=max(temperature, 1e-5),
                    top_k=top_k,
                    top_p=top_p,
                    min_p=0.0,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                )

                token_id = int(selected.item())
                if token_id in (tokenizer.im_end_id, tokenizer.eos_token_id):
                    break

                generated_tokens.append(token_id)
                token_text = tokenizer.decode([token_id])
                print(token_text, end="", flush=True)

                input_tensor = torch.cat([input_tensor, selected], dim=1)

        print("\n")
        assistant_reply = tokenizer.decode(generated_tokens).strip()
        messages.append({"role": "assistant", "content": assistant_reply})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chat interactively with fine-tuned Ultron-113M."
    )
    parser.add_argument(
        "--checkpoint",
        "--checkpoint-dir",
        dest="checkpoint",
        type=str,
        default="accelerate_sft_checkpoint",
        help="Path to fine-tuned SFT checkpoint directory.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="You are Ultron, an intelligent, helpful, and concise AI assistant.",
        help="System prompt.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling threshold (default: 0.9).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k sampling threshold (default: 40).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens per response.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution.",
    )
    args = parser.parse_args()

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    model, tokenizer = load_instruct_model(args.checkpoint, device)

    chat_loop(
        model=model,
        tokenizer=tokenizer,
        system_prompt=args.system,
        device=device,
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
