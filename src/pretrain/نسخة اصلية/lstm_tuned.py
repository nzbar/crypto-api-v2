"""
File: pretrain/lstm_tuned.py
Description: تعريف بنية نموذج LSTM المطور (ثنائي الاتجاه مع Dropout).
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl

class LSTMTuned(pl.LightningModule):
    def __init__(self, n_features, hidden_units, n_layers, lr, dropout_prob=0.2):
        super().__init__()
        self.save_hyperparameters()
        
        # --- التحسينات الجوهرية ---
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_units,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,  # 1. تفعيل النموذج ثنائي الاتجاه
            dropout=dropout_prob if n_layers > 1 else 0 # 2. إضافة Dropout بين الطبقات
        )
        
        # المدخل للطبقة الخطية يتضاعف لأن النموذج ثنائي الاتجاه
        self.linear = nn.Linear(in_features=hidden_units * 2, out_features=1)
        
        # 3. إضافة طبقة Dropout مستقلة
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        
        # تطبيق Dropout على المخرجات لمنع الحفظ الزائد
        out = self.dropout(lstm_out[:, -1, :])
        
        y_predicted = self.linear(out)
        return y_predicted, hidden

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
