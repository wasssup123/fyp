import os
import math
import random
import time  # For timing
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence

from Bio import SeqIO
import matplotlib.pyplot as plt

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

###################################
# Distributed Initialization
###################################
def init_distributed_mode():
    # MASTER_ADDR, MASTER_PORT, RANK, and WORLD_SIZE must be set externally.
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"Distributed mode: rank {rank}, world size {world_size}")
    else:
        print("Not using distributed mode")
        rank = 0
        world_size = 1
    backend = "gloo"  # Suitable for mixed environments.
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size

###################################
# Device Selection: Prefer CUDA > MPS > CPU
###################################
def get_local_device(rank):
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

###################################
# Data Preparation Functions
###################################
def parse_fasta_with_labels(fasta_file):
    data = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        header = record.description.strip()
        sequence = str(record.seq).upper()
        label = header.split()[0]
        data.append((label, sequence))
    return data

def create_train_test_split(raw_data):
    label_to_samples = defaultdict(list)
    for label, seq in raw_data:
        label_to_samples[label].append(seq)
    train_data = []
    test_data = []
    for label, seqs in label_to_samples.items():
        random.shuffle(seqs)
        test_seq = seqs[0]
        train_seqs = seqs[1:]
        test_data.append((label, test_seq))
        for s in train_seqs:
            train_data.append((label, s))
    return train_data, test_data

def generate_kmers(sequence, k=6):
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmers.append(sequence[i:i+k])
    return kmers

def build_kmer_vocab(dataset, k=6):
    kmer_set = set()
    for _, seq in dataset:
        kmers = generate_kmers(seq, k)
        kmer_set.update(kmers)
    vocab = {"<UNK>": 0}
    for i, kmer in enumerate(sorted(kmer_set), start=1):
        vocab[kmer] = i
    return vocab

def encode_sequence(sequence, vocab, k=6):
    kmers = generate_kmers(sequence, k)
    encoded = [vocab.get(kmer, vocab["<UNK>"]) for kmer in kmers]
    return encoded

def filter_classes(raw_data, min_count=5):
    label_counts = Counter([label for (label, _) in raw_data])
    filtered_data = [(label, seq) for (label, seq) in raw_data if label_counts[label] >= min_count]
    return filtered_data

def reverse_complement(seq):
    # For simplicity, just reverse the sequence.
    return seq[::-1]

def create_paired_data(data_list):
    paired = []
    for label, seq in data_list:
        rev_seq = reverse_complement(seq)
        paired.append((label, seq, rev_seq))
    return paired

# Resampling function (for stages 6 and 5)
def resample_dataset(train_data):
    label_to_samples = defaultdict(list)
    for label, seq in train_data:
        label_to_samples[label].append(seq)
    max_count = max(len(seqs) for seqs in label_to_samples.values())
    resampled_data = []
    for label, seqs in label_to_samples.items():
        sampled_seqs = random.choices(seqs, k=max_count)
        for seq in sampled_seqs:
            resampled_data.append((label, seq))
    random.shuffle(resampled_data)
    return resampled_data

###################################
# Dataset Class
###################################
class TwoFastaKmerDataset(Dataset):
    def __init__(self, paired_data, vocab, k=6):
        super().__init__()
        self.vocab = vocab
        self.k = k
        labels = sorted(set(item[0] for item in paired_data))
        self.label2idx = {lbl: i for i, lbl in enumerate(labels)}
        self.encoded_data = []
        for label, fwd_seq, rev_seq in paired_data:
            x1 = encode_sequence(fwd_seq, self.vocab, k=self.k)
            x2 = encode_sequence(rev_seq, self.vocab, k=self.k)
            y = self.label2idx[label]
            self.encoded_data.append((x1, x2, y))
    
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        return self.encoded_data[idx]
    
    def get_vocab_size(self):
        return len(self.vocab)
    
    def get_num_classes(self):
        return len(self.label2idx)

def collate_fn_two(batch):
    seqs_fwd, seqs_rev, labels = zip(*batch)
    seqs_fwd_tensors = [torch.tensor(s, dtype=torch.long) for s in seqs_fwd]
    seqs_rev_tensors = [torch.tensor(s, dtype=torch.long) for s in seqs_rev]
    padded_fwd = pad_sequence(seqs_fwd_tensors, batch_first=True, padding_value=0)
    padded_rev = pad_sequence(seqs_rev_tensors, batch_first=True, padding_value=0)
    labels_tensors = torch.tensor(labels, dtype=torch.long)
    return padded_fwd, padded_rev, labels_tensors

###################################
# Model Architecture
###################################
class ViTDeepSEAEncoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim=128,
                 d_model=256,
                 num_conv_filters=(320, 480, 960),
                 conv_kernel_sizes=(8, 8, 8),
                 pool_kernel_sizes=(4, 4),
                 num_transformer_layers=2,
                 nhead=8,
                 dropout=0.2,
                 max_seq_len=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.deepsea_conv = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=num_conv_filters[0], kernel_size=conv_kernel_sizes[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel_sizes[0]),
            nn.Conv1d(in_channels=num_conv_filters[0], out_channels=num_conv_filters[1], kernel_size=conv_kernel_sizes[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel_sizes[1]),
            nn.Conv1d(in_channels=num_conv_filters[1], out_channels=num_conv_filters[2], kernel_size=conv_kernel_sizes[2]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.bn = nn.BatchNorm1d(num_conv_filters[2])
        self.proj = nn.Linear(num_conv_filters[2], d_model)
        self.max_tokens = 150
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_tokens, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = self.embedding(x)          # [B, seq_len, embed_dim]
        x = x.transpose(1, 2)          # [B, embed_dim, seq_len]
        x = self.deepsea_conv(x)       # [B, num_conv_filters[-1], L_out]
        x = self.bn(x)
        x = x.transpose(1, 2)          # [B, L_out, num_conv_filters[-1]]
        x = self.proj(x)               # [B, L_out, d_model]
        B, L, _ = x.size()
        pos_embed = self.pos_embedding[:, :L, :]
        x = x + pos_embed
        x = self.transformer_encoder(x)
        x = self.final_norm(x)
        x = x.mean(dim=1)
        return x

class TwoViTDeepSEAFusionDNAClassifierWithFC(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 embed_dim=128,
                 d_model=256,
                 num_conv_filters=(320,480,960),
                 conv_kernel_sizes=(8,8,8),
                 pool_kernel_sizes=(4,4),
                 num_transformer_layers=2,
                 nhead=8,
                 dropout=0.2,
                 max_seq_len=1000):
        super().__init__()
        self.vit_branch1 = ViTDeepSEAEncoder(vocab_size=vocab_size,
                                              embed_dim=embed_dim,
                                              d_model=d_model,
                                              num_conv_filters=num_conv_filters,
                                              conv_kernel_sizes=conv_kernel_sizes,
                                              pool_kernel_sizes=pool_kernel_sizes,
                                              num_transformer_layers=num_transformer_layers,
                                              nhead=nhead,
                                              dropout=dropout,
                                              max_seq_len=max_seq_len)
        self.vit_branch2 = ViTDeepSEAEncoder(vocab_size=vocab_size,
                                              embed_dim=embed_dim,
                                              d_model=d_model,
                                              num_conv_filters=num_conv_filters,
                                              conv_kernel_sizes=conv_kernel_sizes,
                                              pool_kernel_sizes=pool_kernel_sizes,
                                              num_transformer_layers=num_transformer_layers,
                                              nhead=nhead,
                                              dropout=dropout,
                                              max_seq_len=max_seq_len)
        self.fc = nn.Linear(2 * d_model, num_classes)
    
    def forward(self, x1, x2):
        f1 = self.vit_branch1(x1)
        f2 = self.vit_branch2(x2)
        fused = torch.cat([f1, f2], dim=1)
        logits = self.fc(fused)
        return logits

###################################
# Helper Functions for Distillation
###################################
def get_overlapping_indices(teacher_label2idx, student_label2idx):
    teacher_indices = []
    student_indices = []
    for label, t_idx in teacher_label2idx.items():
        if label in student_label2idx:
            teacher_indices.append(t_idx)
            student_indices.append(student_label2idx[label])
    return teacher_indices, student_indices

def distillation_loss(student_logits, teacher_logits, student_overlap, teacher_overlap, T, clip_threshold=0.9):
    s_overlap = student_logits[:, student_overlap]  # [B, num_overlap]
    t_overlap = teacher_logits[:, teacher_overlap]  # [B, num_overlap]
    teacher_probs = F.softmax(t_overlap / T, dim=1)
    teacher_probs = torch.clamp(teacher_probs, max=clip_threshold)
    teacher_probs = teacher_probs / teacher_probs.sum(dim=1, keepdim=True)
    kd_loss = F.kl_div(F.log_softmax(s_overlap / T, dim=1),
                       teacher_probs,
                       reduction="batchmean") * (T * T)
    return kd_loss

###################################
# Build Vocabulary and Load Raw Data
###################################
fasta_file = "/Users/longheishe/Documents/Github/fyp/data2/fungi_ITS_cleaned.fasta"
raw_data = parse_fasta_with_labels(fasta_file)
vocab = build_kmer_vocab(raw_data, k=6)

###################################
# Progressive Training Pipeline Functions
###################################
def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    ce_loss = F.cross_entropy(inputs, targets, reduction="none")
    pt = torch.exp(-ce_loss)
    focal_loss_val = alpha * (1 - pt) ** gamma * ce_loss
    if reduction == "mean":
        return focal_loss_val.mean()
    elif reduction == "sum":
        return focal_loss_val.sum()
    else:
        return focal_loss_val

def train_stage(stage_name, 
                student_data_filter, 
                teacher_model=None, 
                use_resampling=False, 
                use_focal_loss=False, 
                temperature=4.5, alpha=0.5):
    """
    Generic training function for one stage.
      - stage_name: e.g. "Student 10"
      - student_data_filter: minimum sample count (10, 8, 7, 6, 5)
      - teacher_model: if provided, used for distillation
      - use_resampling: if True, resample training data
      - use_focal_loss: if True, use focal loss (for Student 7)
    Returns:
      - best_state: best model state from this stage (for use as teacher later)
      - avg_best_acc: average best test accuracy over 10 runs
      - label2idx: mapping for this stage
    """
    data_filtered = filter_classes(raw_data, min_count=student_data_filter)
    train_data, test_data = create_train_test_split(data_filtered)
    if use_resampling:
        train_data = resample_dataset(train_data)
    paired_train = create_paired_data(train_data)
    paired_test = create_paired_data(test_data)
    dataset_train = TwoFastaKmerDataset(paired_train, vocab, k=6)
    dataset_test = TwoFastaKmerDataset(paired_test, vocab, k=6)
    
    label2idx = dataset_train.label2idx
    num_classes = dataset_train.get_num_classes()
    vocab_size = dataset_train.get_vocab_size()
    
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    train_sampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    
    loader_train = DataLoader(dataset_train, batch_size=16, sampler=train_sampler,
                              shuffle=(False if train_sampler is not None else True), collate_fn=collate_fn_two)
    loader_test = DataLoader(dataset_test, batch_size=16, shuffle=False, collate_fn=collate_fn_two)
    
    runs = 10
    best_accs = []
    best_states = []
    local_device = get_local_device(rank)
    
    for run in range(1, runs+1):
        print(f"\n=== {stage_name} Run {run}/{runs} ===")
        model = TwoViTDeepSEAFusionDNAClassifierWithFC(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embed_dim=128,
            d_model=256,
            num_conv_filters=(320,480,960),
            conv_kernel_sizes=(8,8,8),
            pool_kernel_sizes=(4,4),
            num_transformer_layers=2,
            nhead=8,
            dropout=0.2,
            max_seq_len=1000
        ).to(local_device)
        if teacher_model is not None:
            teacher_model.eval()
            teacher_model.to(local_device)
        model = DDP(model, device_ids=[local_device.index] if local_device.type=="cuda" else None)
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        
        num_epochs = 100
        patience_limit = 10
        best_acc = 0.0
        best_state = None
        patience_counter = 0
        
        for epoch in range(1, num_epochs+1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            total_loss = 0.0
            for fwd, rev, labels in loader_train:
                fwd, rev, labels = fwd.to(local_device), rev.to(local_device), labels.to(local_device)
                optimizer.zero_grad()
                logits = model(fwd, rev)
                if teacher_model is None:
                    loss = criterion(logits, labels)
                else:
                    with torch.no_grad():
                        teacher_logits = teacher_model(fwd, rev)
                    ce_loss = criterion(logits, labels)
                    try:
                        t_overlap, s_overlap = get_overlapping_indices(teacher_model.module.label2idx, model.module.label2idx)
                    except AttributeError:
                        t_overlap, s_overlap = [], []
                    kd_loss = distillation_loss(logits, teacher_logits, t_overlap, s_overlap, temperature, clip_threshold=0.9)
                    if use_focal_loss:
                        ce_loss = focal_loss(logits, labels, alpha=0.25, gamma=2.0, reduction="mean")
                    loss = alpha * kd_loss + (1 - alpha) * ce_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader_train)
            
            if rank == 0:
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for fwd, rev, labels in loader_test:
                        fwd, rev, labels = fwd.to(local_device), rev.to(local_device), labels.to(local_device)
                        preds = torch.argmax(model(fwd, rev), dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)
                test_acc = 100.0 * correct / total
                scheduler.step(test_acc)
                print(f"[{stage_name}] Run {run} Epoch {epoch}/100 | Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2f}%")
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_state = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience_limit:
                    print(f"Early stopping triggered at {stage_name} run {run} epoch {epoch}.")
                    break
        if rank == 0:
            print(f"{stage_name} Run {run} Best Accuracy: {best_acc:.2f}%")
            best_accs.append(best_acc)
            best_states.append(best_state)
    if rank == 0:
        avg_best_acc = sum(best_accs) / len(best_accs)
        print(f"\n*** Average {stage_name} Test Accuracy over {runs} runs: {avg_best_acc:.2f}% ***")
    else:
        avg_best_acc = None
    return best_states[0] if best_states else None, avg_best_acc, label2idx

###################################
# Main Progressive Training Pipeline with Timing
###################################
def main():
    # Start timing
    start_time = time.time()
    
    rank, world_size = init_distributed_mode()
    local_device = get_local_device(rank)
    
    # Stage 1: Student 10 (min_count>=10; no resampling, no teacher)
    best_state_10, avg_acc_10, label2idx_10 = train_stage("Student 10", student_data_filter=10,
                                                          teacher_model=None, use_resampling=False)
    teacher_student10 = TwoViTDeepSEAFusionDNAClassifierWithFC(
        vocab_size=len(vocab),
        num_classes=len(label2idx_10),
        embed_dim=128,
        d_model=256,
        num_conv_filters=(320,480,960),
        conv_kernel_sizes=(8,8,8),
        pool_kernel_sizes=(4,4),
        num_transformer_layers=2,
        nhead=8,
        dropout=0.2,
        max_seq_len=1000
    ).to(local_device)
    if best_state_10 is not None:
        teacher_student10.load_state_dict(best_state_10)
    teacher_student10.eval()
    
    # Stage 2: Student 8 (min_count>=8), distill from Student 10
    best_state_8, avg_acc_8, label2idx_8 = train_stage("Student 8", student_data_filter=8,
                                                        teacher_model=teacher_student10, use_resampling=False)
    teacher_student8 = TwoViTDeepSEAFusionDNAClassifierWithFC(
        vocab_size=len(vocab),
        num_classes=len(label2idx_8),
        embed_dim=128,
        d_model=256,
        num_conv_filters=(320,480,960),
        conv_kernel_sizes=(8,8,8),
        pool_kernel_sizes=(4,4),
        num_transformer_layers=2,
        nhead=8,
        dropout=0.2,
        max_seq_len=1000
    ).to(local_device)
    if best_state_8 is not None:
        teacher_student8.load_state_dict(best_state_8)
    teacher_student8.eval()
    
    # Stage 3: Student 7 (min_count>=7), distill from Student 8 using focal loss
    best_state_7, avg_acc_7, label2idx_7 = train_stage("Student 7", student_data_filter=7,
                                                        teacher_model=teacher_student8, use_resampling=False, use_focal_loss=True)
    teacher_student7 = TwoViTDeepSEAFusionDNAClassifierWithFC(
        vocab_size=len(vocab),
        num_classes=len(label2idx_7),
        embed_dim=128,
        d_model=256,
        num_conv_filters=(320,480,960),
        conv_kernel_sizes=(8,8,8),
        pool_kernel_sizes=(4,4),
        num_transformer_layers=2,
        nhead=8,
        dropout=0.2,
        max_seq_len=1000
    ).to(local_device)
    if best_state_7 is not None:
        teacher_student7.load_state_dict(best_state_7)
    teacher_student7.eval()
    
    # Stage 4: Student 6 (min_count>=6), resample training data, distill from Student 7
    best_state_6, avg_acc_6, label2idx_6 = train_stage("Student 6", student_data_filter=6,
                                                        teacher_model=teacher_student7, use_resampling=False)
    teacher_student6 = TwoViTDeepSEAFusionDNAClassifierWithFC(
        vocab_size=len(vocab),
        num_classes=len(label2idx_6),
        embed_dim=128,
        d_model=256,
        num_conv_filters=(320,480,960),
        conv_kernel_sizes=(8,8,8),
        pool_kernel_sizes=(4,4),
        num_transformer_layers=2,
        nhead=8,
        dropout=0.2,
        max_seq_len=1000
    ).to(local_device)
    if best_state_6 is not None:
        teacher_student6.load_state_dict(best_state_6)
    teacher_student6.eval()
    
    # Stage 5: Student 5 (min_count>=5), resample training data, distill from Student 6
    best_state_5, avg_acc_5, label2idx_5 = train_stage("Student 5", student_data_filter=5,
                                                        teacher_model=teacher_student6, use_resampling=False)
    
    if rank == 0:
        print("\n--- Progressive Training Summary ---")
        print(f"Student 10 Avg Test Acc: {avg_acc_10:.2f}%")
        print(f"Student 8 Avg Test Acc: {avg_acc_8:.2f}%")
        print(f"Student 7 Avg Test Acc: {avg_acc_7:.2f}%")
        print(f"Student 6 Avg Test Acc: {avg_acc_6:.2f}%")
        print(f"Student 5 Avg Test Acc: {avg_acc_5:.2f}%")
        # Calculate total elapsed time
        elapsed_time = time.time() - start_time
        print(f"\nTotal Training Time: {elapsed_time:.2f} seconds")
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
