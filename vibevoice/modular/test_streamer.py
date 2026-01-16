import torch
from streamer import AudioStreamer

# 1) 가짜 스트리머 생성 (batch_size = 3)
streamer = AudioStreamer(batch_size=3)

# 2) 가짜 오디오 데이터 (3개 샘플)
# 각 샘플당 길이 16000짜리 fake waveform
fake_audio = torch.randn(3, 16000)

# 3) 가짜 인덱스
fake_indices = torch.tensor([0, 1, 2])

# 4) put 테스트
streamer.put(fake_audio, fake_indices)

print("put() 성공")

# 5) end 테스트
streamer.end(fake_indices)

print("end() 성공")

# 6) 각 큐에서 실제로 데이터가 들어갔는지 확인
for i in range(3):
    q = streamer.audio_queues[i]
    print(f"Queue {i} size:", q.qsize())
