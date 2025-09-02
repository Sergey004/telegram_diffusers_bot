import os
import asyncio
import logging
import re
from typing import List, Tuple, Optional

import requests
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import ContextTypes
from urllib.parse import urlparse

from .llm import generate_prompt_from_idea
from .diffusion import generate_image
from .queue import (
    global_semaphore,
    get_user_semaphore,
    user_tasks,
)

# Simple in‑memory storage for per‑user active LoRAs
_user_active_loras: dict[int, List[str]] = {}

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
async def _acquire_semaphores(user_id: int) -> Tuple[asyncio.Semaphore, asyncio.Semaphore]:
    """
    Acquire the global and per‑user semaphores.
    Returns the two semaphore objects so the caller can release them later.
    """
    await global_semaphore.acquire()
    user_sem = get_user_semaphore(user_id)
    await user_sem.acquire()
    return global_semaphore, user_sem


def _release_semaphores(global_sem: asyncio.Semaphore, user_sem: asyncio.Semaphore):
    """Release the previously acquired semaphores."""
    user_sem.release()
    global_sem.release()


async def _send_progress_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str):
    """Send a temporary progress message and return its message object."""
    return await context.bot.send_message(chat_id=chat_id, text=text)


async def _run_generation_task(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    prompt: str,
    loras: Optional[List[str]] = None,
    size: Tuple[int, int] = (512, 512),
    steps: int = 30,
):
    """
    Core generation workflow:
    1. Acquire concurrency limits.
    2. Send a “Generating…” placeholder.
    3. Run diffusion in a thread.
    4. Send the resulting image.
    5. Clean up.
    """
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # Acquire semaphores = await _acquire_semaphores(user_id)
    global_sem, user_sem = await _acquire_semaphores(user_id)

    # Store the task so it can be cancelled via /cancel
    current_task = asyncio.current_task()
    user_tasks[user_id] = current_task

    # Send placeholder message
    progress_msg = await _send_progress_message(context, chat_id, "🖼️ Generating image…")

    try:
        # Run the heavy diffusion call in a thread to avoid blocking the event loop
        image = await generate_image(
            prompt=prompt,
            size=size,
            steps=steps,
            lora_names=loras,
        )
        # Send the image as a file (document)
        from io import BytesIO
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)
        await context.bot.send_document(chat_id=chat_id, document=image_bytes, filename="generated.png")
        # Edit the placeholder to indicate completion
        await progress_msg.edit_text("✅ Image generated.")
    except Exception as exc:
        logging.exception("Image generation failed")
        await progress_msg.edit_text(f"❌ Generation failed: {exc}")
    finally:
        # Cleanup: release semaphores and remove task reference
        _release_semaphores(global_sem, user_sem)
        user_tasks.pop(user_id, None)


# --------------------------------------------------------------------------- #
# Command handlers
# --------------------------------------------------------------------------- #
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Welcome message with brief usage."""
    welcome_text = (
        "🤖 *Telegram Diffusers Bot*\n\n"
        "I can generate images using Stable Diffusion + optional LoRA models.\n\n"
        "Available commands:\n"
        "/idea <description> – let the LLM create a prompt and generate an image.\n"
        "/gen <prompt> – generate directly from a prompt.\n"
        "/random – generate a random scene.\n"
        "/lora <url> – download and activate a LoRA.\n"
        "/loras – list available LoRAs.\n"
        "/clear_loras – remove all active LoRAs for you.\n"
        "/cancel – cancel your current generation.\n"
        "/setting – (future) adjust size, steps, etc.\n"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


async def idea_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /idea <text>
    Uses the LLM to turn a user description into a Stable Diffusion prompt,
    then enqueues the generation.
    """
    if not context.args:
        await update.message.reply_text("❗️ Usage: /idea <your description>")
        return

    user_text = " ".join(context.args)
    await update.message.reply_text("🧠 Generating prompt with LLM…")
    prompt = await generate_prompt_from_idea(user_text)

    # Retrieve any active LoRAs for this user
    loras = _user_active_loras.get(update.effective_user.id, [])
    # Enqueue generation
    await _run_generation_task(update, context, prompt, loras=loras)


async def gen_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /gen <prompt>
    Directly generate an image from a user‑provided prompt.
    """
    if not context.args:
        await update.message.reply_text("❗️ Usage: /gen <prompt>")
        return

    prompt = " ".join(context.args)
    loras = _user_active_loras.get(update.effective_user.id, [])
    await _run_generation_task(update, context, prompt, loras=loras)


async def random_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /random
    Generates a random scene using the LLM (fallback to a static prompt if no API key).
    """
    await update.message.reply_text("🎲 Generating a random prompt…")
    # Simple static fallback – you can replace this with a more sophisticated LLM call
    random_prompt = "tiger and dragon duo, fantasy landscape, vivid colors, detailed illustration"
    loras = _user_active_loras.get(update.effective_user.id, [])
    await _run_generation_task(update, context, random_prompt, loras=loras)


async def lora_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /lora <url>
    Downloads a LoRA checkpoint from the given URL and registers it for the user.
    """
    if not context.args:
        await update.message.reply_text("❗️ Usage: /lora <direct‑download‑url>")
        return

    url = context.args[0]
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        filename = os.path.basename(url.split("?")[0])  # strip query params
        loras_path = os.path.join(os.path.dirname(__file__), "..", "loras", filename)
        with open(loras_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        # Register for the user
        user_id = update.effective_user.id
        _user_active_loras.setdefault(user_id, []).append(filename)
        await update.message.reply_text(f"✅ LoRA `{filename}` downloaded and activated.", parse_mode="Markdown")
    except Exception as exc:
        logging.exception("Failed to download LoRA")
        await update.message.reply_text(f"❌ Failed to download LoRA: {exc}")


async def loras_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /loras
    Lists all LoRA files currently stored on the server.
    """
    loras_dir = os.path.join(os.path.dirname(__file__), "..", "loras")
    try:
        files = [f for f in os.listdir(loras_dir) if os.path.isfile(os.path.join(loras_dir, f))]
        if not files:
            await update.message.reply_text("📂 No LoRA files are available.")
            return
        file_path = "\n".join(f"- `{f}`" for f in files)
        await update.message.reply_text(f"📂 Available LoRAs:\n{file_path}", parse_mode="Markdown")
    except Exception as exc:
        logging.exception("Failed to list LoRAs")
        await update.message.reply_text(f"❌ Error listing LoRAs: {exc}")


async def clear_loras_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /clear_loras
    Removes all active LoRAs for the calling user (does not delete files).
    """
    user_id = update.effective_user.id
    _user_active_loras[user_id] = []
    await update.message.reply_text("🗑️ All your active LoRAs have been cleared.")


async def cancel_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /cancel
    Cancels the current generation task for the user, if any.
    """
    user_id = update.effective_user.id
    task = user_tasks.get(user_id)
    if task and not task.done():
        task.cancel()
        await update.message.reply_text("🛑 Generation cancelled.")
    else:
        await update.message.reply_text("ℹ️ No active generation to cancel.")


async def setting_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /setting
    Placeholder for future settings (size, steps, active LoRAs, etc.).
    """
    await update.message.reply_text(
        "⚙️ Settings are not yet implemented. Future versions will allow you to adjust image size, steps, and active LoRAs."
    )

# --------------------------------------------------------------------------- #
# Civitai model upload command
# --------------------------------------------------------------------------- #

async def model_upload_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /model_upload <url_or_id>
    Downloads a model or LoRA from Civitai and stores it in the appropriate folder.
    If the model version provides multiple files, the bot presents an inline keyboard
    for the user to choose which file to download.
    """
    if not context.args:
        await update.message.reply_text("❗️ Usage: /model_upload <Civitai URL or model ID>")
        return

    url_or_id = context.args[0]
    try:
        model_id = _extract_civitai_id(url_or_id)
    except ValueError as e:
        await update.message.reply_text(f"❌ {e}")
        return

    await update.message.reply_text("🔎 Fetching model metadata from Civitai…")
    api_url = f"https://civitai.com/api/v1/models/{model_id}"
    try:
        resp = requests.get(api_url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        await update.message.reply_text(f"❌ Failed to retrieve model data: {exc}")
        return

    versions = data.get("modelVersions", [])
    if not versions:
        await update.message.reply_text("❌ No model versions found.")
        return
    files = versions[0].get("files", [])
    if not files:
        await update.message.reply_text("❌ No downloadable files found for this model.")
        return

    if len(files) > 1:
        keyboard = []
        file_map = {}
        for idx, f in enumerate(files):
            name = f.get("name", f"file_{idx}")
            callback_data = f"civitai_file_{idx}"
            file_map[callback_data] = f
            keyboard.append([InlineKeyboardButton(name, callback_data=callback_data)])
        context.user_data["civitai_file_map"] = file_map
        await update.message.reply_text(
            "Multiple files found – please select the one to download:",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
        return

    # Single file – download directly
    file_info = files[0]
    await _download_and_store_file(update, context, file_info)

async def civitai_file_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Callback query handler for the inline keyboard presented by /model_upload.
    """
    query = update.callback_query
    await query.answer()
    callback_data = query.data
    file_map = context.user_data.get("civitai_file_map", {})
    file_info = file_map.get(callback_data)
    if not file_info:
        await query.edit_message_text("❌ Selected file information not found.")
        return

    context.user_data.pop("civitai_file_map", None)
    await query.edit_message_text("⬇️ Downloading selected file…")
    await _download_and_store_file(update, context, file_info)

async def _download_and_store_file(update: Update, context: ContextTypes.DEFAULT_TYPE, file_info: dict) -> None:
    """
    Helper that downloads a file from Civitai and stores it in the appropriate folder.
    Registers LoRAs for the user when applicable.
    """
    download_url = file_info.get("downloadUrl")
    filename = file_info.get("name")
    if not download_url or not filename:
        await update.message.reply_text("❌ Invalid file information received.")
        return

    is_lora = _is_lora(filename)
    target_dir = os.path.join(os.path.dirname(__file__), "..", "loras" if is_lora else "models")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, filename)

    try:
        with requests.get(download_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(target_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except Exception as exc:
        await update.message.reply_text(f"❌ Download failed: {exc}")
        return

    if is_lora:
        user_id = update.effective_user.id
        _user_active_loras.setdefault(user_id, []).append(filename)

    await update.message.reply_text(
        f"✅ `{filename}` saved to `{'loras' if is_lora else 'models'}`.",
        parse_mode="Markdown",
    )
