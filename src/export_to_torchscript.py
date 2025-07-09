import os
import sys
import json
import torch

# 📌 تحديد المسارات
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(os.path.join(ROOT_DIR, "src"))

# ✅ استيراد كلاس LSTM
from pretrain.lstm import LSTM

# ✅ تحميل إعدادات النموذج
with open(os.path.join(ROOT_DIR, "src", "examples", "config_nn.json")) as f:
    config = json.load(f)

# ✅ إنشاء النموذج
model = LSTM(
    n_features=246,
    hidden_units=config["hidden_units"],
    n_layers=config["n_layers"],
    lr=config["learning_rate"]
)

# ✅ تحميل الأوزان
weights_path = os.path.join(ROOT_DIR, "models", "lstm_14062025.pth")
model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
model.eval()

# ✅ إنشاء إدخال وهمي (dummy input)
example_input = torch.rand(1, 60, 246)

# ✅ تحويل النموذج إلى TorchScript
traced_script_module = torch.jit.trace(model, example_input)

# ✅ حفظ النموذج بصيغة .pt
output_path = os.path.join(ROOT_DIR, "models", "lstm_14062025.pt")
traced_script_module.save(output_path)

print(f"✅ TorchScript model saved at: {output_path}")
