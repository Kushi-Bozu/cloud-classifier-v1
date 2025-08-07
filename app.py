import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from datetime import datetime

# 日本語フォント設定（Windows用）
rcParams['font.family'] = 'Meiryo'  # 'Meiryo' ot 'MS Gothic' or 'DejaVu Sans'

# ランク別カラー定義（1位: 金, 2位: 銀, 3位: 銅）
rank_colors = ['gold', 'silver', '#cd7f32']  # ブロンズは HEXで指定

#雲情報（辞書）
cloud_info = {
    "Altocumulus": {
        "和名": "高積雲（こうせきうん）",
        "特徴": "小さな塊状の雲が広がり、天気の変化を予兆することが多い。",
        "高度分類": "中層雲（2,000〜7,000m）"
    },
    "Altostratus": {
        "和名": "高層雲（こうそううん）",
        "特徴": "灰色または青灰色のベール状雲。太陽や月がぼんやり透ける。",
        "高度分類": "中層雲（2,000〜7,000m）"
    },
    "Capcloud": {
        "和名": "( ﾟДﾟ) 笠雲（かさくも）",
        "特徴": "山や高峰の上に笠状にかかる雲。風が強い兆候。",
        "高度分類": "低層雲（〜2,000m）"
    },
    "Cirrocumulus": {
        "和名": "巻積雲（けんせきうん）",
        "特徴": "小さい白い斑点状の雲が魚の鱗のように並ぶ。",
        "高度分類": "上層雲（5,000〜13,000m）"
    },
    "Cirrostratus": {
        "和名": "巻層雲（けんそううん）",
        "特徴": "薄いベールのような雲で、暈（かさ）が見えることが多い。",
        "高度分類": "上層雲（5,000〜13,000m）"
    },
    "Cirrus": {
        "和名": "巻雲（けんうん）",
        "特徴": "白く繊細な筋状の雲。天気が崩れる前兆。",
        "高度分類": "上層雲（5,000〜13,000m）"
    },
    "Cumulonimbus": {
        "和名": "積乱雲（せきらんうん）",
        "特徴": "入道雲とも呼ばれ、雷雨や突風を伴うことがある。",
        "高度分類": "対流雲（地上〜13,000m）"
    },
    "Cumulus": {
        "和名": "積雲（せきうん）",
        "特徴": "モクモクと盛り上がった形の雲。晴天時に見られる。",
        "高度分類": "対流雲（地上〜2,000m）"
    },
    "Haze": {
        "和名": "( ﾟДﾟ) 煙霧（えんむ）",
        "特徴": "大気中の微粒子で視界がぼやける現象。",
        "高度分類": "地表付近"
    },
    "Lenticular": {
        "和名": "( ﾟДﾟ) レンズ雲（れんずくも）",
        "特徴": "風下側の空にできるレンズ形の雲。強風の兆候。",
        "高度分類": "中層雲〜上層雲（2,000〜7,000m）"
    },
    "Nimbostratus": {
        "和名": "乱層雲（らんそううん）",
        "特徴": "厚く広がり長時間の雨や雪を降らせる。",
        "高度分類": "低層雲（〜2,000m）"
    },
    "Shelf": {
        "和名": "( ﾟДﾟ) 棚雲（たなぐも）",
        "特徴": "積乱雲の前面に形成される棚状の雲。突風や雷雨を伴うことがある。",
        "高度分類": "対流雲（地上〜2,000m）"
    },
    "Stratocumulus": {
        "和名": "層積雲（そうせきうん）",
        "特徴": "灰色や白色の塊が一面に広がる雲。",
        "高度分類": "低層雲（〜2,000m）"
    },
    "Stratus": {
        "和名": "層雲（そううん）",
        "特徴": "一様に広がる灰色の雲。小雨や霧雨を伴うことがある。",
        "高度分類": "低層雲（〜2,000m）"
    }
}

@st.cache_resource
def load_model():
    #checkpoint = torch.load("vit_model.pth", map_location=torch.device("cpu"))
    checkpoint = torch.load("vit_model.pth", map_location=torch.device("cpu"), weights_only=False)
    classes = checkpoint['classes']
    model = timm.create_model("vit_small_patch16_224", pretrained=False)
    model.head = nn.Linear(model.head.in_features, len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, classes

model, class_names = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

if "log_df" not in st.session_state:
    st.session_state.log_df = pd.DataFrame(columns=["日時", "画像名", "予測クラス", "確率(%)"])

st.title("雲の分類")
st.write("画像をアップロードして解析を開始してください。")

uploaded_file = st.file_uploader("画像を選択", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="アップロード画像", use_column_width=True)
    if st.button("解析開始"):
        with st.spinner("解析中..."):
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                pred_class = torch.argmax(probs).item()
                
            # 予測クラスと確率を取得
            pred_label = class_names[pred_class]
            probs = torch.softmax(outputs, dim=1)[0]
            prob_values = probs.numpy()
            
            # 上位3クラス抽出
            top3_idx = prob_values.argsort()[-3:][::-1]
            top3_labels = [class_names[i] for i in top3_idx]
            top3_probs = prob_values[top3_idx]

            # レイアウト
            col1, col2 = st.columns(2)

            with col1:
                st.success(f"予測結果: {pred_label}")
                if pred_label in cloud_info:
                    info = cloud_info[pred_label]
                    card_html = f"""
                    <div style="background-color:#f0f8ff; padding:15px; border-radius:10px; margin-bottom:10px;">
                        <h4 style="margin:0;">和名</h4>
                        <p style="margin:0;">{info['和名']}</p>
                    </div>
                    <div style="background-color:#f0fff0; padding:15px; border-radius:10px; margin-bottom:10px;">
                        <h4 style="margin:0;">特徴</h4>
                        <p style="margin:0;">{info['特徴']}</p>
                    </div>
                    <div style="background-color:#fff5ee; padding:15px; border-radius:10px; margin-bottom:10px;">
                        <h4 style="margin:0;">高度分類</h4>
                        <p style="margin:0;">{info['高度分類']}</p>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
            with col2:
                # 横棒グラフ（色をrank_colorsで指定）
                fig, ax = plt.subplots()
                #bars = ax.barh(top3_labels, top3_probs, color="skyblue")
                bars = ax.barh(top3_labels, top3_probs, color=rank_colors)

                # 値を右横に表示（%）
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f"{top3_probs[i]*100:.1f}%", va='center', fontsize=10)
                
                # 棒グラフ（最下端）にラベルとタイトル表示
                #ax.set_xlabel("確率")
                ax.set_xlabel("Probability")
                #ax.set_title("上位3クラス（予測分布）")
                ax.set_title("Top 3 classes (prediction distribution)")
                plt.gca().invert_yaxis()  # 上位を上に表示
                st.pyplot(fig)

            pred_label = class_names[pred_class]
            
            st.success(f"予測結果: {class_names[pred_class]}")
            
            # 和名・特徴・高度分類をカード風表示
            # if pred_label in cloud_info:
            #     info = cloud_info[pred_label]
    
            # # カード風HTML
            #     card_html = f"""
            #     <div style="background-color:#f0f8ff; padding:15px; border-radius:10px; margin-bottom:10px;">
            #         <h4 style="margin:0;">和名</h4>
            #         <p style="margin:0;">{info['和名']}</p>
            #     </div>
            #     <div style="background-color:#f0fff0; padding:15px; border-radius:10px; margin-bottom:10px;">
            #         <h4 style="margin:0;">特徴</h4>
            #         <p style="margin:0;">{info['特徴']}</p>
            #     </div>
            #     <div style="background-color:#fff5ee; padding:15px; border-radius:10px; margin-bottom:10px;">
            #         <h4 style="margin:0;">高度分類</h4>
            #         <p style="margin:0;">{info['高度分類']}</p>
            #     </div>
            #     """
            #     st.markdown(card_html, unsafe_allow_html=True)
            # else:
            #     st.write("追加情報は未登録です。")
            
            #分類確率を棒グラフで可視化
            probs = torch.softmax(outputs, dim=1)[0]
            
            fig, ax = plt.subplots()
            ax.bar(class_names, probs.numpy())
            #ax.set_ylabel("確率")
            ax.set_ylabel("Probability")
            #ax.set_title("予測分布")
            ax.set_title("Classification probability(predictive distribution)")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
            log_entry = {
                "日時": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "画像名": uploaded_file.name,
                "予測クラス": class_names[pred_class],
                "確率(%)": f"{probs[pred_class].item()*100:.2f}"
            }
            st.session_state.log_df = pd.concat([st.session_state.log_df, pd.DataFrame([log_entry])], ignore_index=True)

st.subheader("推論ログ（セッション中のみ保持）")
st.dataframe(st.session_state.log_df)
