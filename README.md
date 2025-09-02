# Telegram Diffusers Bot (SDXL Edition)

A Telegram bot that generates images using **Stable Diffusion XL** (SDXL) with optional LoRA models and optional LLM‑generated prompts (via Nvidia NIM).  
The bot is designed to run in Google Colab (or any Python environment with a GPU).

## Project structure

```
telegram_diffusers_bot/
│
├─ requirements.txt                # pip dependencies
├─ config/
│   └─ .env.example                # copy to .env and fill in your values
│
├─ bot/
│   ├─ __init__.py
│   ├─ main.py                     # entry point – builds the Application and registers handlers
│   ├─ llm.py                      # Nvidia NIM LLM wrapper (generate_prompt_from_idea)
│   ├─ diffusion.py                # SDXL pipeline with optional LoRA loading
│   ├─ queue.py                    # global & per‑user semaphores + task‑cancellation map
│   └─ handlers.py                 # all Telegram command handlers
│
├─ models/                         # placeholder for custom SD checkpoints (optional)
│   └─ .gitkeep
└─ loras/                          # directory where downloaded LoRA files are stored
    └─ .gitkeep
```

## Setup (Google Colab)

1. **Create a new notebook** and add the following cells (run them in order).

   ```python
   # Cell 1 – Install dependencies
   !pip install -q -r https://raw.githubusercontent.com/yourusername/telegram_diffusers_bot/main/requirements.txt
   ```

2. **Clone the repository** (or upload the files directly).

   ```python
   !git clone https://github.com/yourusername/telegram_diffusers_bot.git
   %cd telegram_diffusers_bot
   ```

3. **Create a real `.env` file** from the template.

   ```python
   %%writefile config/.env
   TELEGRAM_TOKEN=YOUR_TELEGRAM_BOT_TOKEN
   CHANNEL_ID=YOUR_TARGET_CHANNEL_ID   # e.g. -1001234567890
   NVIDIA_API_KEY=YOUR_NVIDIA_API_KEY   # optional, needed for LLM prompt generation
   ```

4. **Start the bot** (the cell will keep running as long‑running async loop).

   ```python
   !python bot/main.py
   ```

   The bot will start polling. Keep the cell alive; you can stop it with the “Interrupt execution” button.

## Bot commands

| Command | Description |
|---------|-------------|
| `/start` | Show a help message. |
| `/idea <description>` | LLM creates a prompt from your description, then generates an image. |
| `/gen <prompt>` | Directly generate an image from a prompt. |
| `/random` | Generate a random preset scene. |
| `/lora <url>` | Download a LoRA checkpoint and activate it for you. |
| `/loras` | List all stored LoRAs. |
| `/clear_loras` | Remove all active LoRAs for the calling user. |
| `/cancel` | Cancel your current generation task. |
| `/setting` | (placeholder) future settings (size, steps, LoRAs). |

## Extending the bot

* **Settings** – implement `/setting` to let users choose image size, inference steps, and active LoRAs via inline keyboards.  
* **Upscaling** – integrate `StableDiffusionUpscalePipeline` and expose an “🔍 Upscale” button.  
* **Persistence** – store user‑LoRA selections and queue state in SQLite for resilience across restarts.  
* **Deployment** – switch from polling to webhooks on a VPS for production‑grade reliability.

Enjoy generating high‑quality images with SDXL! 🚀
