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
    import os
    import uuid
    import shutil
    import subprocess

    max_retries = 3
    for attempt in range(max_retries):
        unique_dir = f"download/{cid}_{uuid.uuid4().hex}"
        os.makedirs(unique_dir, exist_ok=True)
        try:
            # Try library first
            client = aioipfs.AsyncIPFS(maddr=ipfs_host)
            try:
                async with async_timeout.timeout(60):
                    await client.get(path=cid, dstdir=unique_dir)
            finally:
                await client.close()

            file_path = f"{unique_dir}/{cid}"
            
            # Verify and load
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                raise IOError(f"Downloaded file {file_path} is empty or missing.")
            
            state_dict = torch.load(file_path, map_location=lambda storage, loc: storage)
            return state_dict

        except Exception as e:
            print(f"IPFS library load failed/corrupted for {cid} (attempt {attempt + 1}/{max_retries}): {e}")
            # Fallback to CLI
            try:
                file_path = f"{unique_dir}/{cid}"
                if os.path.exists(unique_dir):
                    shutil.rmtree(unique_dir)
                os.makedirs(unique_dir, exist_ok=True)
                
                # Run ipfs get cid -o unique_dir/cid
                process = subprocess.run(
                    ['ipfs', 'get', cid, '-o', file_path],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if process.returncode == 0 and os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    state_dict = torch.load(file_path, map_location=lambda storage, loc: storage)
                    return state_dict
                else:
                    raise IOError(f"CLI ipfs get failed: {process.stderr}")
            except Exception as cli_e:
                print(f"IPFS CLI fallback failed: {cli_e}")
            
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(2)
        finally:
            if os.path.exists(unique_dir):
                shutil.rmtree(unique_dir)


async def load_models(cid_list: List[str], ipfs_host: str) -> List:
    return await asyncio.gather(*[load_model_ipfs(cid, ipfs_host) for cid in cid_list])
