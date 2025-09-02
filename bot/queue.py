import asyncio
from typing import Dict

# Global concurrency limit: maximum number of simultaneous generations across all users
MAX_GLOBAL_CONCURRENT = 5
global_semaphore = asyncio.Semaphore(MAX_GLOBAL_CONCURRENT)

# Per‑user concurrency limit: each user may have up to 3 active generations
MAX_PER_USER = 3
_user_semaphores: Dict[int, asyncio.Semaphore] = {}

# Mapping of user_id → currently running asyncio.Task (used for /cancel)
user_tasks: Dict[int, asyncio.Task] = {}

def get_user_semaphore(user_id: int) -> asyncio.Semaphore:
    """
    Retrieve (or create) a semaphore that limits the number of concurrent
    generations for a specific user.
    """
    if user_id not in _user_semaphores:
        _user_semaphores[user_id] = asyncio.Semaphore(MAX_PER_USER)
    return _user_semaphores[user_id]
