from flask import Flask, request, render_template, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)

# ตั้งค่าที่เก็บไฟล์ที่อัปโหลด
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ตาราง HSV สำหรับแถบสีของตัวต้านทาน
COLOUR_TABLE = [
     # [lower_bound_HSV, upper_bound_HSV, color_name, tolerance, box_color]
    [(0, 10, 0), (179, 255, 45), "BLACK", 0, (0, 0, 0)],
    [(0, 105, 40), (25, 200, 70), "BROWN", 1, (0, 51, 102)],
    [(130, 30, 35), (178, 255, 255), "RED", 2, (0, 0, 255)],
    [(0, 140, 120), (10, 255, 255), "ORANGE", 3, (0, 128, 255)],
    [(25, 0, 0), (100,255, 255), "YELLOW", 4, (0, 255, 255)],
    [(30, 30, 0), (70, 255, 255), "GREEN", 5, (0, 255, 0)],
    [(75, 45, 0), (115, 255, 255), "BLUE", 6, (255, 0, 0)],
    [(270, 30, 35), (280, 255, 255), "PURPLE", 7, (255, 0, 125)],
    [(0, 0, 50), (179, 45, 90), "GRAY", 8, (128, 128, 128)],
    [(0, 0, 175), (179, 15, 250), "WHITE", 9, (255, 255, 255)],
]

COLOUR_TABLE_HIGH_BRIGHTNESS = [
    [(0, 10, 0), (179, 255, 45), "BLACK", 0, (0, 0, 0)],
    [(0, 105, 40), (25, 200, 70), "BROWN", 1, (42, 42, 165)],
    [(130, 30, 35), (178, 255, 255), "RED", 2, (0, 0, 255)],
    [(0, 140, 120), (10, 255, 255), "ORANGE", 3, (0, 128, 255)],
    [(25, 0,0), (60, 255, 255), "YELLOW", 4, (0, 255, 255)],
    [(30, 30, 0), (70, 255, 255), "GREEN", 5, (0, 255, 0)],
    [(75, 45, 0), (115, 255, 255), "BLUE", 6, (255, 0, 0)],
    [(270, 30, 35), (280, 255, 255), "PURPLE", 7, (255, 0, 125)],
    [(0, 0, 50), (179, 45, 90), "GRAY", 8, (128, 128, 128)],
    [(0, 0, 175), (179, 15, 250), "WHITE", 9, (255, 255, 255)],
]

COLOUR_TABLE_LOW_BRIGHTNESS = [
    [(0, 20, 0), (179, 150, 45), "BLACK", 0, (0, 0, 0)],
    [(0, 60, 20), (25, 140, 60), "BROWN", 1, (42, 42, 165)],
    [(120, 30, 30), (170, 200, 200), "RED", 2, (0, 0, 255)],
    [(0, 100, 100), (15, 200, 200), "ORANGE", 3, (0, 128, 255)],
    [(20, 40, 40), (50, 200, 200), "YELLOW", 4, (0, 255, 255)],
    [(30, 20, 0), (65, 200, 200), "GREEN", 5, (0, 255, 0)],
    [(75, 30, 0), (110, 200, 200), "BLUE", 6, (255, 0, 0)],
    [(270, 30, 35), (280, 255, 255), "PURPLE", 7, (255, 0, 125)],
    [(0, 0, 30), (179, 35, 70), "GRAY", 8, (128, 128, 128)],
    [(0, 0, 150), (179, 15, 255), "WHITE", 9, (255, 255, 255)],
]

def get_average_brightness(image):
    # แปลงเป็นโหมดสี HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # ดึงช่อง V (Value หรือ Brightness)
    brightness = hsv_image[:, :, 2]
    # คำนวณค่าเฉลี่ยของ Brightness
    avg_brightness = np.mean(brightness)
    return avg_brightness

def select_colour_table(image):
    avg_brightness = get_average_brightness(image)
    print(f"Average Brightness: {avg_brightness}")
    
    # เกณฑ์ความสว่างในการเลือกตารางสี
    if avg_brightness > 150:  # รูปที่สว่างมาก
        print("Using COLOUR_TABLE for high brightness")
        return COLOUR_TABLE_HIGH_BRIGHTNESS  # เพิ่มตารางสีสำหรับแสงมาก
    elif avg_brightness < 80:  # รูปที่มืดมาก
        print("Using COLOUR_TABLE for low brightness")
        return COLOUR_TABLE_LOW_BRIGHTNESS  # เพิ่มตารางสีสำหรับแสงน้อย
    else:
        print("Using default COLOUR_TABLE")
        return COLOUR_TABLE  # ใช้ตารางสีปกติ

def adjust_image_for_reflections(image):
    gamma_corrected = adjust_gamma(image, gamma=1.5)
    blurred_image = cv2.medianBlur(gamma_corrected, 5)
    filtered_image = cv2.bilateralFilter(blurred_image, 6, 80, 80)
    return filtered_image

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def process_image(file_path):
    try:
        original_image = cv2.imread(file_path)
        if original_image is None:
            return [{"error": "Invalid image file"}]
        adjusted_image = adjust_image_for_reflections(original_image)
        gray_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
        _, segmented = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        eroded = cv2.erode(segmented, kernel, iterations=5)
        dilated = cv2.dilate(eroded, kernel, iterations=7)
        mask = cv2.erode(dilated, kernel, iterations=1)

        resistor_mask = filter_resistors(mask)
        resistor_only = cv2.bitwise_and(original_image, original_image, mask=resistor_mask)
        extracted_resistors = extract_resistors(resistor_only, resistor_mask)

        results = []
        for resistor in extracted_resistors:
            color_bands, annotated_image = extract_color_bands(resistor)
            resistance_value = calculate_resistance(color_bands)

            _, encoded_resistor = cv2.imencode('.jpg', resistor)
            _, encoded_annotated = cv2.imencode('.jpg', annotated_image)

            results.append({
                "bands": color_bands,
                "resistance": resistance_value,
                "resistor": base64.b64encode(encoded_resistor).decode('utf-8'),
                "annotated": base64.b64encode(encoded_annotated).decode('utf-8')
            })

        return results
    except Exception as e:
        return [{"error": str(e)}]

def filter_resistors(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(binary_image)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1750 or area > 1000000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        if 0.3 < aspect_ratio < 4:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)
    return filtered_mask

def extract_resistors(resistor_only_image, mask):
    edges = cv2.Canny(mask, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    resistor_images = []

    for contour in contours:
        if cv2.contourArea(contour) > 1500:  # ตรวจสอบขนาดพื้นที่ขั้นต่ำ
            # สร้าง Mask ว่างเปล่าขนาดเดียวกับภาพ
            contour_mask = np.zeros_like(mask)

            # วาดคอนทัวร์ลงใน Mask
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

            # ใช้ Mask เพื่อแยกตัวต้านทานออกมา
            masked_resistor = cv2.bitwise_and(resistor_only_image, resistor_only_image, mask=contour_mask)

            # ตัดส่วนที่ไม่เกี่ยวข้องออกโดยใช้ boundingRect เพื่อทำภาพให้เล็กลง
            x, y, w, h = cv2.boundingRect(contour)
            cropped_resistor = masked_resistor[y:y + h, x:x + w]

            resistor_images.append(cropped_resistor)

    return resistor_images

def find_resistor_bands(resistor_info):
    # เลือก COLOUR_TABLE ตามระดับความสว่าง
    selected_colour_table = select_colour_table(resistor_info)

    resolu_img = cv2.resize(resistor_info, (400, 200))
    app_bil = cv2.bilateralFilter(resolu_img, 5, 80, 80)
    hsv = cv2.cvtColor(app_bil, cv2.COLOR_BGR2HSV)
    thresh = cv2.adaptiveThreshold(
        cv2.cvtColor(app_bil, cv2.COLOR_BGR2GRAY),
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        59,
        5,
    )
    thresh = cv2.bitwise_not(thresh)

    bands = []
    overlay_image = resolu_img.copy()  # ภาพสำหรับวาดแถบสี
    band_positions = []  # ตำแหน่งของแถบสี

    for clr in selected_colour_table:
        mask = cv2.inRange(hsv, clr[0], clr[1])
        mask = cv2.bitwise_and(mask, thresh, mask=mask)
        
        if cv2.countNonZero(mask) > 600:  # จำนวนพิกเซลขั้นต่ำ
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 500:  # กรองขนาดแถบสีที่เล็กเกินไป
                    x, y, w, h = cv2.boundingRect(contour)
                    band_positions.append((x, y, w, h, clr[2], clr[4]))  # บันทึก x, y, w, h, ชื่อสี, และสีกรอบ

    # (ดำเนินการประมวลผลเหมือนเดิม)

    # จัดกลุ่มแถบสีที่อยู่ในคอลัมน์เดียวกัน
    column_threshold = 15  # กำหนดระยะห่างแนวนอนที่ยอมรับได้
    grouped_bands = []
    band_positions.sort(key=lambda pos: pos[0])  # จัดเรียงตามตำแหน่ง x

    current_column = [band_positions[0]] if band_positions else []  # เริ่มต้นคอลัมน์แรก
    for i in range(1, len(band_positions)):
        if abs(band_positions[i][0] - current_column[-1][0]) <= column_threshold:
            current_column.append(band_positions[i])  # อยู่ในคอลัมน์เดียวกัน
        else:
            grouped_bands.append(current_column)  # เก็บคอลัมน์ปัจจุบัน
            current_column = [band_positions[i]]  # เริ่มคอลัมน์ใหม่

    if current_column:
        grouped_bands.append(current_column)  # เก็บคอลัมน์สุดท้าย

    # วาดแถบสีและเก็บผลลัพธ์
    for column in grouped_bands:
        column.sort(key=lambda pos: pos[1])  # จัดเรียงคอลัมน์ตามตำแหน่ง y (จากบนลงล่าง)
        for band in column:
            x, y, w, h, color_name, color_box = band
            cv2.rectangle(overlay_image, (x, y), (x + w, y + h), color_box, 3)  # วาดกรอบสี
        bands.append(column[0][4])  # บันทึกชื่อสีของคอลัมน์

    return bands, overlay_image

def extract_color_bands(resistor_image):
    bands, annotated_image = find_resistor_bands(resistor_image)
    return bands, annotated_image

def calculate_resistance(color_bands):
    if len(color_bands) < 3:
        return "Invalid bands"

    try:
        significant_figures = "".join(str(next(color[3] for color in COLOUR_TABLE if color[2] == band)) for band in color_bands[:-1])
        multiplier = next(10 ** color[3] for color in COLOUR_TABLE if color[2] == color_bands[-1])
        resistance_value = int(significant_figures) * multiplier

        if resistance_value >= 1_000_000:
            resistance_formatted = f"{int(resistance_value / 1_000_000)} MΩ" if resistance_value % 1_000_000 == 0 else f"{resistance_value / 1_000_000}M Ω"
        elif resistance_value >= 1_000:
            resistance_formatted = f"{int(resistance_value / 1_000)} KΩ" if resistance_value % 1_000 == 0 else f"{resistance_value / 1_000} KΩ"
        else:
            resistance_formatted = f"{resistance_value} Ω"

        return resistance_formatted
    except Exception:
        return "Unable to calculate resistance"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        results = process_image(file_path)
        return jsonify(results)
    
    return render_template('upload.html')

@app.route('/selectcolor')
def selectcolor():
    return render_template('selectcolor.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
    # app.run(debug=True)
