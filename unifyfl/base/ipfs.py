import asyncio
from datetime import datetime
from typing import List, Tuple
import torch

import aioipfs

import async_timeout

# TODO: switch to sync ipfs framework?


async def save_model_ipfs(state_dict, ipfs_host: str) -> str:
    client = aioipfs.AsyncIPFS(maddr=ipfs_host)
    import os
    import io
    os.makedirs("upload", exist_ok=True)
    cur_time = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".pickle")
    # print(type(state_dict))
    torch.save(state_dict, f"upload/{cur_time}")
    
    with open(f"upload/{cur_time}", "rb") as f:
        data = f.read()

    try:
        # Using BytesIO because it has getbuffer() which aiohttp expects
        # We use a list comprehension to get the CID from the generator
        results = [entry["Hash"] async for entry in client.add(io.BytesIO(data))]
        cid = str(results[0])
    except aioipfs.exceptions.APIError as e:
        print(f"IPFS API Error: {e}")
        # Fallback to shell if library fails
        import subprocess
        process = subprocess.run(['ipfs', 'add', '-q', f"upload/{cur_time}"], capture_output=True, text=True)
        cid = process.stdout.strip()
        if not cid:
            raise e
    
    await client.close()
    return cid


async def load_model_ipfs(cid: str, ipfs_host: str):
    client = aioipfs.AsyncIPFS(maddr=ipfs_host)
    async with async_timeout.timeout(100):
        await client.get(path=cid, dstdir="download")
    await client.close()
    return torch.load(f"download/{cid}")
    # return np.load(f"download/{cid}", allow_pickle=True)
    # return pickle.load(open(f"download/{cid}", "rb"))


async def load_models(cid_list: List[str], ipfs_host: str) -> List:
    return await asyncio.gather(*[load_model_ipfs(cid, ipfs_host) for cid in cid_list])
