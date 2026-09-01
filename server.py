import os

# ต้องตั้งก่อน import tf_keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"

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

    df.columns = df.columns.str.strip()

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
            "ไม่พบคอลัมน์ใน nutrition_data.csv: "
            + ", ".join(missing_columns)
        )

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
            # หรือ:
            # khao_man_gai

            if " " in line:
                label = line.split(" ", 1)[1].strip()
            else:
                label = line

            class_names.append(label)

    return model, class_names


# โหลดครั้งเดียวตอนเปิด server
df = load_data()
model, class_names = load_model_and_labels()


# =========================================================
# ทำนายรูป
# =========================================================
def predict_food(image):

    # ขนาดที่ Teachable Machine ใช้
    size = (224, 224)

    image = ImageOps.fit(
        image,
        size,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(image)

    normalized_image_array = (
        image_array.astype(np.float32) / 127.5
    ) - 1.0

    data = np.ndarray(
        shape=(1, 224, 224, 3),
        dtype=np.float32
    )

    data[0] = normalized_image_array

    prediction = model.predict(
    data,
    batch_size=1,
    verbose=0
)

    probabilities = prediction[0]

    index = int(np.argmax(probabilities))

    if index >= len(class_names):
        raise ValueError(
            "จำนวน Output ของโมเดลไม่ตรงกับ labels.txt"
        )

    predicted_class = class_names[index].strip()

    confidence_score = (
        float(probabilities[index]) * 100
    )

    return predicted_class, confidence_score


# =========================================================
# API
# =========================================================
@app.route("/predict", methods=["POST"])
def predict():

    try:

        # -------------------------------------------------
        # เช็กรูป
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
        # AI ทำนาย
        # -------------------------------------------------
        predicted_class, confidence = predict_food(
            image
        )


        # -------------------------------------------------
        # ค้นข้อมูลโภชนาการ
        # -------------------------------------------------
        matched_data = df[
            df["class_label"]
            .astype(str)
            .str.strip()
            .str.lower()
            ==
            predicted_class.lower()
        ]


        # ถ้า AI เจอเมนูแต่ไม่มีใน CSV
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
                "confidence": round(confidence, 2),
                "adviceLevel": "warn",
                "advice": "AI ตรวจพบเมนูนี้ แต่ยังไม่มีข้อมูลโภชนาการในฐานข้อมูล",
                "highlights": [],
                "alternatives": []
            })


        row = matched_data.iloc[0]


        # -------------------------------------------------
        # เตรียมข้อมูลส่งกลับ HTML
        # -------------------------------------------------
        result = {

            "foodName": str(row["menu_name"]),

            "portionSize": str(
                row["serving_size"]
            ),

            "calories": float(
                row["calories"]
            ),

            "sodium": 0,

            "carbs": float(
                row["carbs"]
            ),

            "sugar": 0,

            "fat": float(
                row["fat"]
            ),

            "protein": float(
                row["protein"]
            ),

            "fiber": 0,

            "confidence": round(
                confidence,
                2
            ),

            "adviceLevel": "good",

            "advice": (
                f"AI ระบุว่าเป็น {row['menu_name']} "
                f"ด้วยความมั่นใจ {confidence:.2f}%"
            ),

            "highlights": [],

            "alternatives": []
        }


        # source ถ้ามี
        if "source" in df.columns:
            if pd.notna(row["source"]):
                result["source"] = str(
                    row["source"]
                )


        return jsonify(result)


    except Exception as error:

        print("ERROR:", error)

        return jsonify({
            "error": str(error)
        }), 500


# =========================================================
# ทดสอบ server
# =========================================================
@app.route("/", methods=["GET"])
def home():
    return send_from_directory(".", "index.html")


# =========================================================
# Run
# =========================================================
if __name__ == "__main__":

    print("========================================")
    print(" Song-Jan Food AI Server")
    print("========================================")
    print(f"จำนวนเมนู: {len(df)}")
    print(f"จำนวน AI labels: {len(class_names)}")
    print("Server: http://127.0.0.1:5000")
    print("========================================")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
