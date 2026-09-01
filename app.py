import os

# =========================================================
# ตั้งค่า Keras สำหรับโมเดล Teachable Machine
# =========================================================
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import tf_keras as keras


# =========================================================
# ตั้งค่าหน้าเว็บ
# =========================================================
st.set_page_config(
    page_title="AI Food Scanner",
    page_icon="🥗",
    layout="centered"
)


# =========================================================
# CSS ตกแต่ง
# =========================================================
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #2E7D32;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        color: #666666;
        margin-bottom: 25px;
    }

    .food-name {
        font-size: 28px;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
    }

    .confidence {
        text-align: center;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# โหลด Nutrition Database
# =========================================================
@st.cache_data
def load_data():

    df = pd.read_csv(
        "nutrition_data.csv",
        encoding="utf-8-sig"
    )

    # ลบช่องว่างหน้าหลังชื่อคอลัมน์
    df.columns = df.columns.str.strip()

    # รองรับชื่อคอลัมน์แบบเดิมที่มีคำอธิบายภาษาไทย
    rename_columns = {
        "class_label (ชื่อใน AI)": "class_label",
        "menu_name (ชื่อเมนู)": "menu_name",
        "serving_size (ปริมาณ)": "serving_size",
        "calories (kcal)": "calories",
        "protein (g)": "protein",
        "carbs (g)": "carbs",
        "fat (g)": "fat",
        "source (ที่มาของข้อมูล)": "source"
    }

    df.rename(
        columns=rename_columns,
        inplace=True
    )

    # ตรวจสอบคอลัมน์ที่จำเป็น
    required_columns = [
        "class_label",
        "menu_name",
        "serving_size",
        "calories",
        "protein",
        "carbs",
        "fat"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "ไม่พบคอลัมน์ต่อไปนี้ใน nutrition_data.csv: "
            + ", ".join(missing_columns)
            + "\n\nคอลัมน์ที่พบจริง: "
            + ", ".join(df.columns.tolist())
        )

    # ทำความสะอาด class_label
    df["class_label"] = (
        df["class_label"]
        .astype(str)
        .str.strip()
    )

    return df


# =========================================================
# โหลดโมเดล + Labels
# =========================================================
@st.cache_resource
def load_model_and_labels():

    # โหลดโมเดล
    model = keras.models.load_model(
        "keras_model.h5",
        compile=False
    )

    # โหลด labels.txt
    with open(
        "labels.txt",
        "r",
        encoding="utf-8"
    ) as file:

        class_names = []

        for line in file.readlines():

            line = line.strip()

            if not line:
                continue

            # รองรับ:
            # 0 khao_man_gai
            # 1 pad_thai
            #
            # และ:
            # khao_man_gai
            if " " in line:
                label = line.split(" ", 1)[1].strip()
            else:
                label = line

            class_names.append(label)

    return model, class_names


# =========================================================
# หัวข้อ
# =========================================================
st.markdown(
    '<div class="main-title">🥗 AI Food Scanner</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'ระบบ AI สแกนอาหารและวิเคราะห์โภชนาการ'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# โหลดไฟล์ระบบ
# =========================================================
try:

    df = load_data()
    model, class_names = load_model_and_labels()

except Exception as error:

    st.error("❌ ไม่สามารถโหลดไฟล์ระบบได้")

    st.code(str(error))

    st.info("""
ตรวจสอบว่าไฟล์ต่อไปนี้อยู่ในโฟลเดอร์เดียวกับ app.py:

1. keras_model.h5
2. labels.txt
3. nutrition_data.csv
""")

    st.stop()


# =========================================================
# ตรวจสอบระบบ
# =========================================================
with st.expander("🔧 ตรวจสอบข้อมูลระบบ"):

    st.write("### คอลัมน์ CSV")

    st.write(
        df.columns.tolist()
    )

    st.write("### จำนวนเมนู")

    st.write(
        f"{len(df)} เมนู"
    )

    st.write("### Labels จาก AI")

    st.write(
        class_names
    )


# =========================================================
# Upload รูป
# =========================================================
st.subheader("📷 เลือกรูปอาหาร")

uploaded_file = st.file_uploader(
    "อัปโหลดภาพอาหาร",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)

st.write("หรือ")

camera_file = st.camera_input(
    "📸 ถ่ายภาพอาหาร"
)


# เลือกไฟล์
input_image = uploaded_file

if input_image is None:
    input_image = camera_file


# =========================================================
# เริ่มประมวลผลเมื่อมีรูป
# =========================================================
if input_image is not None:

    # -----------------------------------------------------
    # เปิดรูป
    # -----------------------------------------------------
    try:

        image = Image.open(
            input_image
        ).convert("RGB")

    except Exception as error:

        st.error(
            "❌ ไม่สามารถเปิดรูปภาพได้"
        )

        st.code(
            str(error)
        )

        st.stop()


    # -----------------------------------------------------
    # แสดงรูป
    # -----------------------------------------------------
    st.image(
        image,
        caption="ภาพอาหาร",
        use_container_width=True
    )


    # =====================================================
    # เตรียมภาพสำหรับ AI
    # =====================================================

    size = (
        224,
        224
    )

    image_resized = ImageOps.fit(
        image,
        size,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        image_resized
    )

    normalized_image_array = (
        image_array.astype(
            np.float32
        ) / 127.5
    ) - 1.0

    data = np.ndarray(
        shape=(
            1,
            224,
            224,
            3
        ),
        dtype=np.float32
    )

    data[0] = normalized_image_array


    # =====================================================
    # AI Prediction
    # =====================================================
    try:

        prediction = model.predict(
            data,
            verbose=0
        )

    except Exception as error:

        st.error(
            "❌ เกิดข้อผิดพลาดขณะให้ AI วิเคราะห์ภาพ"
        )

        st.code(
            str(error)
        )

        st.stop()


    # =====================================================
    # ตรวจสอบ Prediction
    # =====================================================

    if prediction is None:

        st.error(
            "❌ AI ไม่สามารถทำนายผลได้"
        )

        st.stop()


    # prediction[0] = probability ของแต่ละ class
    probabilities = prediction[0]

    index = int(
        np.argmax(probabilities)
    )


    # ตรวจสอบจำนวน labels
    if index >= len(class_names):

        st.error(
            "❌ จำนวน Class ของโมเดลไม่ตรงกับ labels.txt"
        )

        st.write(
            "จำนวน Output จากโมเดล:",
            len(probabilities)
        )

        st.write(
            "จำนวน Labels:",
            len(class_names)
        )

        st.stop()


    # =====================================================
    # ผล AI
    # =====================================================

    predicted_class = (
        class_names[index]
        .strip()
    )

    confidence_score = (
        float(probabilities[index])
        * 100
    )


    # =====================================================
    # แสดงผล AI
    # =====================================================

    st.divider()

    st.subheader(
        "🤖 ผลการวิเคราะห์จาก AI"
    )

    st.markdown(
        f'<div class="food-name">{predicted_class}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="confidence">'
        f'ความมั่นใจ: <b>{confidence_score:.2f}%</b>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.progress(
        min(
            confidence_score / 100,
            1.0
        )
    )


    # =====================================================
    # ค้นหาข้อมูลจาก CSV
    # =====================================================

    matched_data = df[
        df["class_label"]
        .astype(str)
        .str.strip()
        .str.lower()
        ==
        predicted_class.lower()
    ]


    # =====================================================
    # พบข้อมูล
    # =====================================================

    if not matched_data.empty:

        row = matched_data.iloc[0]


        # -------------------------------------------------
        # ชื่อเมนู
        # -------------------------------------------------

        st.divider()

        st.success(
            f"🍽️ เมนูที่ตรวจพบ: {row['menu_name']}"
        )


        # -------------------------------------------------
        # Serving Size
        # -------------------------------------------------

        st.info(
            f"🥣 ปริมาณ: {row['serving_size']}"
        )


        # =================================================
        # Nutrition
        # =================================================

        st.subheader(
            "🥗 ข้อมูลโภชนาการ"
        )


        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "🔥 พลังงาน",
                f"{row['calories']} kcal"
            )

            st.metric(
                "🥩 โปรตีน",
                f"{row['protein']} g"
            )


        with col2:

            st.metric(
                "🍚 คาร์โบไฮเดรต",
                f"{row['carbs']} g"
            )

            st.metric(
                "🧈 ไขมัน",
                f"{row['fat']} g"
            )


        # =================================================
        # ตารางข้อมูล
        # =================================================

        st.subheader(
            "📋 รายละเอียดโภชนาการ"
        )

        nutrition_table = pd.DataFrame(
            {
                "รายการ": [
                    "เมนู",
                    "ปริมาณ",
                    "พลังงาน",
                    "โปรตีน",
                    "คาร์โบไฮเดรต",
                    "ไขมัน"
                ],

                "ข้อมูล": [
                    row["menu_name"],
                    row["serving_size"],
                    f"{row['calories']} kcal",
                    f"{row['protein']} g",
                    f"{row['carbs']} g",
                    f"{row['fat']} g"
                ]
            }
        )

        st.table(
            nutrition_table
        )


        # =================================================
        # Source
        # =================================================

        if "source" in df.columns:

            source = row["source"]

            if pd.notna(source):

                st.caption(
                    f"📚 แหล่งข้อมูล: {source}"
                )


    # =====================================================
    # ไม่พบข้อมูล
    # =====================================================

    else:

        st.warning(
            "⚠️ AI สามารถทำนายภาพได้ "
            "แต่ไม่พบ class นี้ใน nutrition_data.csv"
        )

        st.write(
            "Class ที่ AI ทำนาย:"
        )

        st.code(
            predicted_class
        )

        st.write(
            "Class ที่มีอยู่ในฐานข้อมูล:"
        )

        st.write(
            df["class_label"].tolist()
        )


# =========================================================
# Footer
# =========================================================

st.divider()

st.caption(
    "🥗 AI Food Scanner | "
    "AI Food Classification & Nutrition Analysis"
)