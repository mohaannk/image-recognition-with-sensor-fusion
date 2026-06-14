# ==========================================
# pothole_detection_12_Apr_working.py
# Full cleaned & fixed version for VS Code
# ==========================================

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install scikit-learn seaborn gradio tqdm scipy matplotlib pillow numpy -q

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# IEEE-Style Plotting Configuration
# ==========================================
import matplotlib as mpl

try:
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.linewidth'] = 0.8
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['xtick.major.width'] = 0.8
    mpl.rcParams['ytick.major.width'] = 0.8
    mpl.rcParams['grid.linestyle'] = '--'
    mpl.rcParams['grid.linewidth'] = 0.5
    mpl.rcParams['grid.alpha'] = 0.7
    mpl.rcParams['axes.titlesize'] = 10
    mpl.rcParams['axes.labelsize'] = 9
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['legend.fontsize'] = 8
    mpl.rcParams['savefig.dpi'] = 600
except RuntimeError:
    print("⚠️ Times New Roman not found. Falling back to default font.")
    pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🖥️  Device: {device}')

# ==========================================
# Attention Modules
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, max(in_planes // ratio, 1), 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(max(in_planes // ratio, 1), in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))

class AttentionBlock(nn.Module):
    def __init__(self, in_planes):
        super(AttentionBlock, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        return x * self.ca(x) * self.sa(x)

class PotholeDetector(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(PotholeDetector, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.attention = AttentionBlock(512)
        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        if x.requires_grad:
            x.register_hook(self.activations_hook)
            
        x = self.attention(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.backbone.fc(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

model = PotholeDetector(dropout_rate=0.3).to(device)
print('✅ Model initialized')

# ==========================================
# GRADIO UI WRAPPERS
# ==========================================
def gradio_inference(img, sld_lidar, sld_sonar, sld_gpr, chk_ambiguous):
    if img is None:
        return None, "❌ Select an image first."

    if chk_ambiguous and img is not None:
        ambiguous_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.6, contrast=0.7, saturation=0.6),
            transforms.GaussianBlur(kernel_size=5, sigma=(1.5, 2.5))
        ])
        img = ambiguous_transform(img)

    try:
        model.load_state_dict(torch.load('pothole_detector.pth', map_location=device))
        model.eval()
    except:
        return None, "⚠️ Model file 'pothole_detector.pth' not found. Please ensure the pre-trained model file is in the same directory as the script."

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
    probs = torch.softmax(outputs, dim=1)
    v_prob = probs[0][1].item()
    vision_src = "CNN"

    # SENSOR FUSION
    w_v, w_l, w_s, w_g = 1.0, 0.0, 0.0, 0.0

    if chk_ambiguous:
        w_v = abs(v_prob - 0.5) * 2
        w_l = abs(sld_lidar - 0.5) * 2
        w_s = abs(sld_sonar - 0.5) * 2
        w_g = abs(sld_gpr - 0.5) * 2

    total_w = w_v + w_l + w_s + w_g
    if total_w < 1e-6:
        w_v, w_l, w_s, w_g = 0.25, 0.25, 0.25, 0.25
    else:
        w_v /= total_w
        w_l /= total_w
        w_s /= total_w
        w_g /= total_w

    fused_score = (v_prob * w_v) + (sld_lidar * w_l) + (sld_sonar * w_s) + (sld_gpr * w_g)
    decision = "⚠️ POTHOLE" if fused_score > 0.5 else "🛣️ NORMAL"

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), gridspec_kw={'width_ratios': [1.5, 1.2, 1.2]})

    axes[0].imshow(img)
    axes[0].set_title(f"Fused Decision: {decision.strip('⚠️🛣️ ')}\n(Confidence: {fused_score:.1%})")
    axes[0].axis('off')

    sensors = ['Vision', 'LiDAR', 'Ultrasonic', 'GPR']
    readings = [v_prob, sld_lidar, sld_sonar, sld_gpr]
    weights = [w_v, w_l, w_s, w_g]
    bar_colors = ['cyan', 'red', 'blue', 'yellow']

    bars1 = axes[1].barh(sensors, readings, color=bar_colors, height=0.6)
    axes[1].set_xlim([0, 1.1])
    axes[1].set_title('Sensor Readings')
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    for bar in bars1:
        width = bar.get_width()
        axes[1].text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width:.1%}', ha='left', va='center', fontsize=8)

    bars2 = axes[2].barh(sensors, weights, color=bar_colors, height=0.6)
    max_weight = max(weights) if any(weights) else 1.0
    axes[2].set_xlim([0, max_weight * 1.3])
    axes[2].set_title('Sensor Weights')
    axes[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    for bar in bars2:
        width = bar.get_width()
        axes[2].text(width + (max_weight * 0.02), bar.get_y() + bar.get_height()/2, f'{width:.1%}', ha='left', va='center', fontsize=8)

    for i, ax in enumerate(axes[1:]):
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.xaxis.grid(True)
        if i > 0:
            ax.set_yticklabels([])

    plt.tight_layout(pad=0.5, w_pad=1.0)

    report = f"📊 SENSOR FUSION ANALYSIS\n"
    report += "="*40 + "\n"
    report += f"Vision ({vision_src}) : {v_prob*100:.1f}% (Weight: {w_v*100:.1f}%)\n"
    report += f"LiDAR        : {sld_lidar*100:.1f}% (Weight: {w_l*100:.1f}%)\n"
    report += f"Ultrasonic   : {sld_sonar*100:.1f}% (Weight: {w_s*100:.1f}%)\n"
    report += f"GPR          : {sld_gpr*100:.1f}% (Weight: {w_g*100:.1f}%)\n"
    report += "="*40 + "\n"
    report += f"🎯 FUSED DECISION: {decision}\n"
    report += f"   Confidence: {fused_score:.1%}\n"
    if chk_ambiguous:
        report += "\n🌫️ AMBIGUOUS MODE ACTIVE - Sensor weights are dynamically calculated based on confidence."
        
    return fig, report

# ==========================================
# GRADIO APP
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛣️ Pothole Detection & Sensor Fusion Analysis")
    gr.Markdown("Upload a test image and simulate multi-sensor readings to get a final fused decision.")
    
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Select Test Image")
            gr.Markdown("### Adjust Simulated Sensor Readings (0.0 = Normal, 1.0 = Pothole)")
            lidar_slider = gr.Slider(0, 1, value=0, step=0.05, label="🔴 LiDAR", interactive=False)
            sonar_slider = gr.Slider(0, 1, value=0, step=0.05, label="🔵 Ultrasonic", interactive=False)
            gpr_slider = gr.Slider(0, 1, value=0, step=0.05, label="🟡 GPR", interactive=False)
            ambiguous_checkbox = gr.Checkbox(label="🌫️ Ambiguous Mode (Dynamic Sensor Weights)")
            inf_btn = gr.Button("🔍 Analyze", variant="primary")
            
        with gr.Column(scale=2):
            inf_plot = gr.Plot(label="Sensor Fusion Analysis")
            inf_text = gr.Textbox(label="Detailed Results", lines=12)

        inf_btn.click(fn=gradio_inference, 
                      inputs=[img_input, lidar_slider, sonar_slider, gpr_slider, ambiguous_checkbox], 
                      outputs=[inf_plot, inf_text])

        def toggle_sensor_sliders(is_ambiguous):
            if is_ambiguous:
                return gr.update(interactive=True, value=0.5), gr.update(interactive=True, value=0.5), gr.update(interactive=True, value=0.5)
            else:
                return gr.update(interactive=False, value=0), gr.update(interactive=False, value=0), gr.update(interactive=False, value=0)

        ambiguous_checkbox.change(fn=toggle_sensor_sliders, 
                                  inputs=ambiguous_checkbox, 
                                  outputs=[lidar_slider, sonar_slider, gpr_slider])

if __name__ == "__main__":
    demo.launch(share=True, inbrowser=True)