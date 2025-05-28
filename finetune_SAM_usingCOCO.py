import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from segment_anything import sam_model_registry
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import os
from tqdm import tqdm
import urllib.request
import zipfile

# FFT 기반 셀프어텐션 클래스 정의
class FFTSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 쿼리, 키, 밸류 프로젝션 레이어
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 쿼리, 키, 밸류 프로젝션
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # FFT 기반 어텐션 계산
        q_fft = torch.fft.rfft(q, dim=-1)
        k_fft = torch.fft.rfft(k, dim=-1)
        
        # 복소수 곱셈 (컨볼루션에 해당)
        res = q_fft * k_fft.conj()
        
        # 역변환
        attn_output = torch.fft.irfft(res, dim=-1, n=self.head_dim)
        
        # 결과와 밸류의 곱
        output = attn_output * v
        
        # 원래 형태로 변환
        output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.dim)
        output = self.out_proj(output)
        
        return output

# SAM 모델 수정하는 함수
def replace_attention_with_fft(model):
    """SAM 모델의 셀프어텐션 레이어를 FFT 버전으로 교체"""
    for name, module in model.named_children():
        if isinstance(module, nn.MultiheadAttention):
            # MultiheadAttention의 embed_dim 가져오기
            embed_dim = module.embed_dim
            # FFT 어텐션으로 교체
            setattr(model, name, FFTSelfAttention(dim=embed_dim, num_heads=module.num_heads, dropout=module.dropout))
        else:
            replace_attention_with_fft(module)
    
    return model

# COCO 데이터셋 다운로드 함수
def download_coco_dataset(data_dir="./coco_data"):
    """COCO 데이터셋을 다운로드하고 압축을 푸는 함수"""
    # 디렉토리 생성
    os.makedirs(data_dir, exist_ok=True)
    
    # 다운로드 URL 정의
    urls = {
        "images": "http://images.cocodataset.org/zips/val2017.zip",  # 검증 이미지 (더 작음)
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    }
    
    # 데이터 다운로드 및 압축 해제
    for name, url in urls.items():
        zip_path = os.path.join(data_dir, f"{name}.zip")
        
        # 이미 다운로드되었는지 확인
        if not os.path.exists(zip_path):
            print(f"{name} 다운로드 중...")
            urllib.request.urlretrieve(url, zip_path)
            
        # 압축 해제
        extract_dir = os.path.join(data_dir, name)
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir, exist_ok=True)
            print(f"{name} 압축 해제 중...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
    
    # 필요한 경로 반환
    images_dir = os.path.join(data_dir, "val2017")
    ann_file = os.path.join(data_dir, "annotations", "instances_val2017.json")
    
    return images_dir, ann_file

# 수정된 COCO 데이터셋 로더 클래스
class COCOSegmentationDataset(Dataset):
    def __init__(self, root_dir, annFile, transform=None, target_size=(1024, 1024)):
        self.root_dir = root_dir
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_size = target_size
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.root_dir, img_info['file_name'])
        
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        
        # 해당 이미지의 세그멘테이션 마스크 가져오기
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((img_info['height'], img_info['width']))
        
        # 모든 객체 마스크 합치기
        for ann in anns:
            if 'segmentation' in ann:
                current_mask = self.coco.annToMask(ann)
                mask = np.maximum(mask, current_mask * ann['category_id'])
        
        # 이미지와 마스크 리사이즈 (동일한 크기로)
        image = image.resize(self.target_size, Image.BILINEAR)
        mask = Image.fromarray(mask.astype(np.uint8)).resize(self.target_size, Image.NEAREST)
        mask = np.array(mask)
        
        # 변환 적용
        if self.transform:
            image = self.transform(image)
        
        # 마스크를 텐서로 변환
        mask = torch.from_numpy(mask).long()
        
        return image, mask

# 커스텀 콜레이트 함수 (배치 생성)
def custom_collate_fn(batch):
    images = []
    masks = []
    
    for image, mask in batch:
        images.append(image)
        masks.append(mask)
    
    images = torch.stack(images, 0)
    masks = torch.stack(masks, 0)
    
    return images, masks

# SAM 모델 다운로드 함수
def download_sam_model(model_type="vit_b", save_dir="./sam_checkpoints"):
    """SAM 모델 체크포인트를 다운로드하는 함수"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint_urls = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
    
    checkpoint_path = os.path.join(save_dir, f"sam_{model_type}.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"SAM {model_type} 모델 다운로드 중...")
        urllib.request.urlretrieve(checkpoint_urls[model_type], checkpoint_path)
    
    return checkpoint_path

# SAM 모델 수정 (학습을 위한 헤드 추가)
class ModifiedSAM(nn.Module):
    def __init__(self, sam_model, num_classes=91):  # COCO 데이터셋은 90개 클래스 + 배경
        super().__init__()
        self.sam = sam_model
        
        # SAM 출력 차원 가져오기 (임베딩 차원)
        # ViT-H: 256, ViT-L: 256, ViT-B: 256
        self.embedding_dim = 256
        
        # 세그멘테이션 헤드 추가
        self.seg_head = nn.Sequential(
            nn.Conv2d(self.embedding_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        # 이미지 인코더 실행
        features = self.sam.image_encoder(x)  # [B, 256, H/16, W/16]
        
        # 특징맵 업샘플링 (원본 이미지 크기로)
        features = F.interpolate(features, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # 세그멘테이션 헤드
        logits = self.seg_head(features)  # [B, num_classes, H, W]
        
        return logits

# 학습 함수
def train_sam_with_fft(model, dataloader, optimizer, criterion, device, num_epochs=5):
    model.train()
    model.to(device)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for images, masks in tqdm(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # 순방향 전파
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 역전파 및 최적화
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    return model

# 메인 실행 함수
def main():
    # 디바이스 설정
    ##device = torch.device("mps")
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"사용 디바이스: {device}")
    
    # COCO 데이터셋 다운로드
    print("COCO 데이터셋 준비 중...")
    images_dir, ann_file = download_coco_dataset()
    print(f"이미지 디렉토리: {images_dir}")
    print(f"어노테이션 파일: {ann_file}")
    
    # SAM 모델 체크포인트 다운로드
    print("SAM 모델 준비 중...")
    checkpoint_path = download_sam_model()
    
    # SAM 모델 불러오기
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    
    # 셀프어텐션을 FFT 버전으로 교체
    print("SAM 모델의 셀프어텐션을 FFT 버전으로 교체 중...")
    sam = replace_attention_with_fft(sam)
    
    # SAM 모델 수정 (세그멘테이션 헤드 추가)
    model = ModifiedSAM(sam)
    
    # 데이터 변환 설정
    target_size = (1024, 1024)  # 더 작은 이미지 크기로 설정 (메모리 절약)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # COCO 데이터셋 로드
    print("데이터셋 로드 중...")
    dataset = COCOSegmentationDataset(
        root_dir=images_dir,
        annFile=ann_file,
        transform=transform,
        target_size=target_size
    )
    
    # 데이터셋이 너무 크면 일부만 사용
    subset_size = min(100, len(dataset))  # 메모리 이슈로 더 적은 샘플 사용
    indices = torch.randperm(len(dataset))[:subset_size]
    subset_dataset = torch.utils.data.Subset(dataset, indices)
    
    print(f"전체 데이터셋 크기: {len(dataset)}, 사용할 샘플 수: {subset_size}")
    
    # 커스텀 콜레이트 함수로 데이터로더 생성
    dataloader = DataLoader(
        subset_dataset, 
        batch_size=1,  # 배치 크기 1로 줄임 (메모리 이슈 방지)
        shuffle=True, 
        num_workers=0,  # 워커 수 0으로 설정 (데이터 로딩 문제 방지)
        collate_fn=custom_collate_fn
    )
    
    # 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    # 학습 실행
    print("FFT 기반 SAM 모델 학습 시작...")
    trained_model = train_sam_with_fft(model, dataloader, optimizer, criterion, device, num_epochs=3)
    
    # 모델 저장
    save_path = "fft_sam_finetuned.pth"
    torch.save(trained_model.state_dict(), save_path)
    print(f"모델 학습 완료! 저장 경로: {save_path}")

if __name__ == "__main__":
    main()
