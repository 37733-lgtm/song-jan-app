import os

# =========================================================
# ตั้งค่าก่อนโหลด TensorFlow
# =========================================================
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# ลดการใช้ทรัพยากรของ TensorFlow
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import gc

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import tf_keras as keras


# =========================================================
# ตั้งค่า Flask
# =========================================================
app = Flask(__name__)
CORS(app)


# =========================================================
# โหลด Nutrition Database
# =========================================================
def load_data():
    df = pd.read_csv(
        "nutrition_data.csv",
        encoding="utf-8-sig"
    )

    # ตัดช่องว่างหน้า/หลังชื่อคอลัมน์
    df.columns = df.columns.str.strip()

    # รองรับชื่อคอลัมน์จากไฟล์เดิม
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
            + "\nคอลัมน์ที่มี: "
            + ", ".join(df.columns.tolist())
        )

    # ทำความสะอาด class label
    df["class_label"] = (
        df["class_label"]
        .astype(str)
        .str.strip()
    )

    return df


# =========================================================
# โหลด AI Model + Labels
# =========================================================
def load_model_and_labels():

    # โหลดโมเดล
    model = keras.models.load_model(
        "keras_model.h5",
        compile=False
    )

    class_names = []

    with open(
        "labels.txt",
        "r",
        encoding="utf-8"
    ) as file:

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
# โหลดข้อมูลทั้งหมดครั้งเดียว
# =========================================================
df = load_data()
model, class_names = load_model_and_labels()

# ช่วยคืน memory ที่ไม่จำเป็น
gc.collect()


# =========================================================
# ฟังก์ชันทำนายอาหาร
# =========================================================
def predict_food(image):

    # ขนาดภาพที่โมเดล Teachable Machine ใช้
    size = (224, 224)

    # ปรับขนาดภาพ
    image_resized = ImageOps.fit(
        image,
        size,
        Image.Resampling.LANCZOS
    )

    # แปลงเป็น numpy
    image_array = np.asarray(
        image_resized,
        dtype=np.float32
    )

    # Normalize แบบ Teachable Machine
    normalized_image_array = (
        image_array / 127.5
    ) - 1.0

    # เตรียม input
    data = np.empty(
        (1, 224, 224, 3),
        dtype=np.float32
    )

    data[0] = normalized_image_array

    # AI Prediction
    prediction = model.predict(
        data,
        batch_size=1,
        verbose=0
    )

    # Probability ของแต่ละ class
    probabilities = prediction[0]

    # หาคลาสที่มั่นใจที่สุด
    index = int(
        np.argmax(probabilities)
    )

    # ตรวจสอบ labels
    if index >= len(class_names):
        raise ValueError(
            "จำนวน Output ของโมเดลไม่ตรงกับ labels.txt"
        )

    predicted_class = (
        class_names[index]
        .strip()
    )

    confidence = (
        float(probabilities[index]) * 100
    )

    return (
        predicted_class,
        confidence
    )


# =========================================================
# หน้าเว็บหลัก
# =========================================================
@app.route("/", methods=["GET"])
def home():
    return send_from_directory(
        ".",
        "index.html"
    )


# =========================================================
# API: วิเคราะห์อาหาร
# =========================================================
@app.route("/predict", methods=["POST"])
def predict():

    try:

        # -------------------------------------------------
        # ตรวจสอบไฟล์
        # -------------------------------------------------
        if "image" not in request.files:
            return jsonify({
                "error": "ไม่พบไฟล์รูปภาพ"
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "error": "ไม่ได้เลือกรูปภาพ"
            }), 400

        # -------------------------------------------------
        # เปิดรูป
        # -------------------------------------------------
        image = Image.open(
            file.stream
        ).convert("RGB")

        # -------------------------------------------------
        # AI วิเคราะห์
        # -------------------------------------------------
        predicted_class, confidence = predict_food(
            image
        )

        # -------------------------------------------------
        # ค้นหาโภชนาการจาก CSV
        # -------------------------------------------------
        matched_data = df[
            df["class_label"]
            .astype(str)
            .str.strip()
            .str.lower()
            ==
            predicted_class.lower()
        ]

        # -------------------------------------------------
        # ถ้าไม่พบข้อมูล
        # -------------------------------------------------
        if matched_data.empty:

            return jsonify({
                "foodName": predicted_class,
                "portionSize": "-",
                "calories": 0,
                "sodium": 0,
                "carbs": 0,
                "sugar": 0,
                "fat": 0,
                "protein": 0,
                "fiber": 0,
                "confidence": round(
                    confidence,
                    2
                ),
                "adviceLevel": "warn",
                "advice": (
                    "AI ตรวจพบเมนูนี้ "
                    "แต่ยังไม่มีข้อมูลโภชนาการในฐานข้อมูล"
                ),
                "highlights": [],
                "alternatives": []
            })

        row = matched_data.iloc[0]

        # -------------------------------------------------
        # แปลงค่าตัวเลขให้ปลอดภัย
        # -------------------------------------------------
        def num(value):
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0

        # -------------------------------------------------
        # สร้างข้อมูลส่งกลับ HTML
        # -------------------------------------------------
        result = {

            "foodName": str(
                row["menu_name"]
            ),

            "portionSize": str(
                row["serving_size"]
            ),

            "calories": num(
                row["calories"]
            ),

            # CSV ตอนนี้ไม่มี sodium
            "sodium": 0,

            "carbs": num(
                row["carbs"]
            ),

            # CSV ตอนนี้ไม่มี sugar
            "sugar": 0,

            "fat": num(
                row["fat"]
            ),

            "protein": num(
                row["protein"]
            ),

            # CSV ตอนนี้ไม่มี fiber
            "fiber": 0,

            "confidence": round(
                confidence,
                2
            ),

            "adviceLevel": "good",

            "advice": (
                f"AI ระบุว่าเป็น "
                f"{row['menu_name']} "
                f"ด้วยความมั่นใจ "
                f"{confidence:.2f}%"
            ),

            "highlights": [],

            "alternatives": []
        }

        # -------------------------------------------------
        # Source
        # -------------------------------------------------
        if "source" in df.columns:

            if pd.notna(row["source"]):

                result["source"] = str(
                    row["source"]
                )

        # -------------------------------------------------
        # คืน JSON
        # -------------------------------------------------
        return jsonify(result)

    except Exception as error:

        # แสดง error ใน Render Logs
        print("========================================")
        print("PREDICT ERROR")
        print(str(error))
        print("========================================")

        return jsonify({
            "error": str(error)
        }), 500


# =========================================================
# Health Check
# =========================================================
@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "menu_count": len(df),
        "label_count": len(class_names)
    })


# =========================================================
# Run
# =========================================================
if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print("========================================")
    print(" Song-Jan Food AI Server")
    print("========================================")
    print(
        f"จำนวนเมนู: {len(df)}"
    )
    print(
        f"จำนวน AI labels: {len(class_names)}"
    )
    print(
        f"Port: {port}"
    )
    print("========================================")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
