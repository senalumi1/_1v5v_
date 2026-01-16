from __future__ import annotations

import torch

import asyncio
from queue import Queue
from typing import TYPE_CHECKING, Optional


from transformers.generation import BaseStreamer


class AudioStreamer(BaseStreamer):
    """
    Audio streamer that stores audio chunks in queues for each sample in the batch.
    This allows streaming audio generation for multiple samples simultaneously.
    
    Parameters:
        batch_size (`int`):
            The batch size for generation
        stop_signal (`any`, *optional*):
            The signal to put in the queue when generation ends. Defaults to None.
        timeout (`float`, *optional*):
            The timeout for the audio queue. If `None`, the queue will block indefinitely.
    """
    
    def __init__(
        self, 
        batch_size: int,
        stop_signal: Optional[any] = None,
        timeout: Optional[float] = None,
    ):
        self.batch_size = batch_size
        self.stop_signal = stop_signal
        self.timeout = timeout
        
        # Create a queue for each sample in the batch
        self.audio_queues = [Queue() for _ in range(batch_size)]
        self.finished_flags = [False for _ in range(batch_size)]
        self.sample_indices_map = {}  # Maps from sample index to queue index
        
    def put(self, audio_chunks: torch.Tensor, sample_indices: torch.Tensor):
        """
        Receives audio chunks and puts them in the appropriate queues.
        """

        # sample_indices → 무조건 Python int 리스트로 변환
        if torch.is_tensor(sample_indices):
            sample_indices = sample_indices.detach().cpu().tolist()
        if isinstance(sample_indices, int):
            sample_indices = [sample_indices]

        # audio_chunks → CPU로 이동 
        if torch.is_tensor(audio_chunks):
            audio_chunks = audio_chunks.detach().cpu()

        for i, idx in enumerate(sample_indices):
            if idx < self.batch_size and not self.finished_flags[idx]:
                audio_chunk = audio_chunks[i]

                # (1, T), (1,1,T) 같은 경우 → (T,)로 정리
                if hasattr(audio_chunk, "ndim") and audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk.reshape(-1)

                self.audio_queues[idx].put(audio_chunk, timeout=self.timeout)

    def end(self, sample_indices: Optional[torch.Tensor] = None):
        """
        Signals the end of generation for specified samples or all samples.
        """

        # 1) sample_indices 없으면 전부 종료
        if sample_indices is None:
            for idx in range(self.batch_size):
                if not self.finished_flags[idx]:
                    self.audio_queues[idx].put(self.stop_signal, timeout=self.timeout)
                    self.finished_flags[idx] = True
            return

        # 2) sample_indices를 무조건 "int 리스트"로 변환
        if torch.is_tensor(sample_indices):
            sample_indices = sample_indices.detach().cpu().tolist()
        if isinstance(sample_indices, int):
            sample_indices = [sample_indices]

        # 3) 종료 처리
        for idx in sample_indices:
            if 0 <= idx < self.batch_size and not self.finished_flags[idx]:
                self.audio_queues[idx].put(self.stop_signal, timeout=self.timeout)
                self.finished_flags[idx] = True
                
    def __iter__(self):
        """Returns an iterator over the batch of audio streams."""
        return AudioBatchIterator(self)
    
    def get_stream(self, sample_idx: int):
        """Get the audio stream for a specific sample."""
        if sample_idx >= self.batch_size:
            raise ValueError(f"Sample index {sample_idx} exceeds batch size {self.batch_size}")
        return AudioSampleIterator(self, sample_idx)


class AudioSampleIterator:
    """Iterator for a single audio stream from the batch."""
    
    def __init__(self, streamer: AudioStreamer, sample_idx: int):
        self.streamer = streamer
        self.sample_idx = sample_idx
        
    def __iter__(self):
        return self
    
    def __next__(self):
        value = self.streamer.audio_queues[self.sample_idx].get(timeout=self.streamer.timeout)
        if value == self.streamer.stop_signal:
            raise StopIteration()
        return value


class AudioBatchIterator:
    """Iterator that yields audio chunks for all samples in the batch."""
    
    def __init__(self, streamer: AudioStreamer):
        self.streamer = streamer
        self.active_samples = set(range(streamer.batch_size))
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.active_samples:
            raise StopIteration()
            
        batch_chunks = {}
        samples_to_remove = set()
        
        # Try to get chunks from all active samples
        for idx in self.active_samples:
            try:
                value = self.streamer.audio_queues[idx].get(block=False)
                if value == self.streamer.stop_signal:
                    samples_to_remove.add(idx)
                else:
                    batch_chunks[idx] = value
            except:
                # Queue is empty for this sample, skip it this iteration
                pass
        
        # Remove finished samples
        self.active_samples -= samples_to_remove
        
        if batch_chunks:
            return batch_chunks
        elif self.active_samples:
            # If no chunks were ready but we still have active samples, 
            # wait a bit and try again
            import time
            time.sleep(0.01)
            return self.__next__()
        else:
            raise StopIteration()


class AsyncAudioStreamer(AudioStreamer):
    """
    Async version of AudioStreamer for use in async contexts.
    """
    
    def __init__(
        self, 
        batch_size: int,
        stop_signal: Optional[any] = None,
        timeout: Optional[float] = None,
    ):
        super().__init__(batch_size, stop_signal, timeout)
        # Replace regular queues with async queues
        self.audio_queues = [asyncio.Queue() for _ in range(batch_size)]
        self.loop = asyncio.get_running_loop()
        
    def put(self, audio_chunks: torch.Tensor, sample_indices: torch.Tensor):
        """Put audio chunks in the appropriate async queues."""
        for i, sample_idx in enumerate(sample_indices):
            idx = sample_idx.item()
            if idx < self.batch_size and not self.finished_flags[idx]:
                audio_chunk = audio_chunks[i].detach().cpu()
                self.loop.call_soon_threadsafe(
                    self.audio_queues[idx].put_nowait, audio_chunk
                )
    
    def end(self, sample_indices: Optional[torch.Tensor] = None):
        """Signal the end of generation for specified samples."""
        if sample_indices is None:
            indices_to_end = range(self.batch_size)
        else:
            indices_to_end = [s.item() if torch.is_tensor(s) else s for s in sample_indices]
            
        for idx in indices_to_end:
            if idx < self.batch_size and not self.finished_flags[idx]:
                self.loop.call_soon_threadsafe(
                    self.audio_queues[idx].put_nowait, self.stop_signal
                )
                self.finished_flags[idx] = True
    
    async def get_stream(self, sample_idx: int):
        """Get async iterator for a specific sample's audio stream."""
        if sample_idx >= self.batch_size:
            raise ValueError(f"Sample index {sample_idx} exceeds batch size {self.batch_size}")
            
        while True:
            value = await self.audio_queues[sample_idx].get()
            if value == self.stop_signal:
                break
            yield value
    
    def __aiter__(self):
        """Returns an async iterator over all audio streams."""
        return AsyncAudioBatchIterator(self)


class AsyncAudioBatchIterator:
    """Async iterator for batch audio streaming."""
    
    def __init__(self, streamer: AsyncAudioStreamer):
        self.streamer = streamer
        self.active_samples = set(range(streamer.batch_size))
        
    def __aiter__(self):
        return self
        
    async def __anext__(self):
        if not self.active_samples:
            raise StopAsyncIteration()
            
        batch_chunks = {}
        samples_to_remove = set()
        
        # Create tasks for all active samples
        tasks = {
            idx: asyncio.create_task(self._get_chunk(idx)) 
            for idx in self.active_samples
        }
        
        # Wait for at least one chunk to be ready
        done, pending = await asyncio.wait(
            tasks.values(), 
            return_when=asyncio.FIRST_COMPLETED,
            timeout=self.streamer.timeout
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
            
        # Process completed tasks
        for idx, task in tasks.items():
            if task in done:
                try:
                    value = await task
                    if value == self.streamer.stop_signal:
                        samples_to_remove.add(idx)
                    else:
                        batch_chunks[idx] = value
                except asyncio.CancelledError:
                    pass
                    
        self.active_samples -= samples_to_remove
        
        if batch_chunks:
            return batch_chunks
        elif self.active_samples:
            # Try again if we still have active samples
            return await self.__anext__()
        else:
            raise StopAsyncIteration()
    
    async def _get_chunk(self, idx):
        """Helper to get a chunk from a specific queue."""
        return await self.streamer.audio_queues[idx].get()