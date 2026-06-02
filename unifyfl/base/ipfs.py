import asyncio
from datetime import datetime
from typing import List, Tuple
import torch
import os

import aioipfs

import async_timeout

# TODO: switch to sync ipfs framework?

# Ensure directories exist
os.makedirs("upload", exist_ok=True)
os.makedirs("download", exist_ok=True)


async def save_model_ipfs(state_dict, ipfs_host: str) -> str:
    """Save model to IPFS with DinD retry logic + Shashwati's BytesIO/shell fallback."""
    import io
    import subprocess
    max_retries = 3
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            client = aioipfs.AsyncIPFS(maddr=ipfs_host)
            cur_time = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".pickle")
            torch.save(state_dict, f"upload/{cur_time}")

            with open(f"upload/{cur_time}", "rb") as f:
                data = f.read()

            try:
                # BytesIO has getbuffer() which aiohttp expects
                results = [entry["Hash"] async for entry in client.add(io.BytesIO(data))]
                cid = str(results[0])
            except aioipfs.exceptions.APIError as e:
                print(f"IPFS API Error: {e}")
                # Fallback to the ipfs shell binary if the library call fails
                process = subprocess.run(
                    ["ipfs", "add", "-q", f"upload/{cur_time}"],
                    capture_output=True, text=True,
                )
                cid = process.stdout.strip()
                if not cid:
                    raise e

            await client.close()
            return cid
        except Exception as e:
            await asyncio.sleep(retry_delay)
            if attempt == max_retries - 1:
                raise
            print(f"IPFS save retry {attempt + 1}/{max_retries}: {str(e)}")


async def load_model_ipfs(cid: str, ipfs_host: str):
    """Load model from IPFS with retry logic."""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            client = aioipfs.AsyncIPFS(maddr=ipfs_host)
            async with async_timeout.timeout(100):
                await client.get(path=cid, dstdir="download")
            await client.close()
            return torch.load(f"download/{cid}")
        except Exception as e:
            await asyncio.sleep(retry_delay)
            if attempt == max_retries - 1:
                raise
            print(f"IPFS load retry {attempt + 1}/{max_retries}: {str(e)}")


async def load_models(cid_list: List[str], ipfs_host: str) -> List:
    return await asyncio.gather(*[load_model_ipfs(cid, ipfs_host) for cid in cid_list])
