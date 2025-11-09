import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrNetLoss(nn.Module):
    def __init__(self, loss_weights, blank_id=0):
        super().__init__()
        self.loss_weights = loss_weights
        self.blank_id = blank_id
        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction='mean')

        print(f"CorrNetLoss initialized with weights: {loss_weights}")

    def forward(self, data):
        """
        修复版本：确保所有输入格式正确
        """
        # 确保必要的输入存在
        required_keys = ['sequence_logits', 'label', 'feat_len', 'label_lgt']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key in data: {key}")

        total_loss = torch.tensor(0.0, device=data['sequence_logits'].device)
        loss_components = {}

        # 修复：确保长度张量为整数类型
        feat_len = data['feat_len'].to(torch.int32)
        label_lgt = data['label_lgt'].to(torch.int32)

        # 1. 主要的SeqCTC损失
        if 'SeqCTC' in self.loss_weights and self.loss_weights['SeqCTC'] > 0:
            sequence_logits = data['sequence_logits']

            # 确保logits格式为 [T, B, C]
            if sequence_logits.dim() == 3:
                if sequence_logits.shape[0] == feat_len.numel():  # [B, T, C]
                    sequence_logits = sequence_logits.transpose(0, 1)  # [T, B, C]
                elif sequence_logits.shape[1] == feat_len.numel():  # [T, B, C] 已经是正确格式
                    pass
                else:
                    raise ValueError(f"Unexpected sequence_logits shape: {sequence_logits.shape}")

            log_probs = F.log_softmax(sequence_logits, dim=-1)
            seq_ctc_loss = self.ctc_loss(
                log_probs,
                data['label'],
                feat_len,  # 使用修复后的整数长度
                label_lgt  # 使用修复后的整数长度
            )
            weighted_loss = self.loss_weights['SeqCTC'] * seq_ctc_loss
            loss_components['SeqCTC'] = weighted_loss.item()
            total_loss += weighted_loss

        # 2. ConvCTC损失（可选）
        if 'ConvCTC' in self.loss_weights and self.loss_weights['ConvCTC'] > 0 and 'conv_logits' in data:
            conv_logits = data['conv_logits']

            # 同样的格式转换
            if conv_logits.dim() == 3:
                if conv_logits.shape[0] == feat_len.numel():  # [B, T, C]
                    conv_logits = conv_logits.transpose(0, 1)  # [T, B, C]

            log_probs = F.log_softmax(conv_logits, dim=-1)
            conv_ctc_loss = self.ctc_loss(
                log_probs,
                data['label'],
                feat_len,
                label_lgt
            )
            weighted_loss = self.loss_weights['ConvCTC'] * conv_ctc_loss
            loss_components['ConvCTC'] = weighted_loss.item()
            total_loss += weighted_loss

        # 3. 蒸馏损失（可选）
        if 'Dist' in self.loss_weights and self.loss_weights['Dist'] > 0:
            if 'conv_logits' in data and 'sequence_logits' in data:
                conv_logits = data['conv_logits']
                sequence_logits = data['sequence_logits']

                # 确保形状匹配
                if conv_logits.shape != sequence_logits.shape:
                    # 简单的形状调整：取相同的时间维度
                    min_time = min(conv_logits.shape[1], sequence_logits.shape[1])
                    conv_logits = conv_logits[:, :min_time, :]
                    sequence_logits = sequence_logits[:, :min_time, :]

                # 计算KL散度
                student_log_probs = F.log_softmax(conv_logits, dim=-1)
                teacher_probs = F.softmax(sequence_logits.detach(), dim=-1)
                dist_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

                weighted_loss = self.loss_weights['Dist'] * dist_loss
                loss_components['Dist'] = weighted_loss.item()
                total_loss += weighted_loss

        # 返回符合框架的格式
        return {
            "loss": total_loss,
            "total_loss": loss_components
        }