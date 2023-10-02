import cv2
import os
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import tkinter as tk
from tkinter import Label, Button, Canvas, Frame
from PIL import Image, ImageTk

# Memuat model yang telah dilatih
model = tf.keras.models.load_model("trained_vgg16_model.h5")

# Inisialisasi GUI

root = tk.Tk()
root.title("Deteksi Tingkat Kematangan Nanas")
root.geometry("1000x600")
root.configure(bg="navy")

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Folder untuk menyimpan gambar yang ditangkap
save_folder = "data_gambar"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Variabel untuk menghitung berapa kali gambar nanas sudah ditangkap
captured_count = 0
max_captures = 4
results = []


# Fungsi untuk mendeteksi dan mengklasifikasikan gambar
def detect_and_classify_frame():
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = image.array_to_img(frame_rgb)
        img = img.resize((128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        img = cv2.resize(frame, (400, 300))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        video_canvas.create_image(0, 0, anchor=tk.NW, image=img)
        video_canvas.img = img

        root.after(10, detect_and_classify_frame)


# Fungsi untuk menangkap gambar
def capture_image():
    global captured_count
    if captured_count < max_captures:
        ret, frame = cap.read()
        if ret:
            img_name = f"data_gambar/captured_nanas_{captured_count}.jpg"
            cv2.imwrite(img_name, frame)

            img = Image.open(img_name)
            img = img.resize((150, 150))
            img = ImageTk.PhotoImage(img)

            captured_images[captured_count].config(image=img)
            captured_images[captured_count].image = img

            captured_count += 1
            if captured_count >= max_captures:
                process_button.config(state=tk.NORMAL)


# Fungsi untuk memproses gambar yang ditangkap
def process_images():
    global results
    results = []
    for i in range(max_captures):
        img_path = f"data_gambar/captured_nanas_{i}.jpg"
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        results.append(prediction[0][0] * 100)

    avg_result = sum(results) / len(results)
    result_text = f"Persentase Kematangan: {avg_result:.2f}%"
    processed_result_label.config(text=result_text, justify=tk.LEFT)


# Fungsi untuk mereset GUI dan gambar yang ditangkap
def reset_gui():
    global captured_count, results
    captured_count = 0
    results = []
    for img_label in captured_images:
        img_label.config(image="")
    processed_result_label.config(text="")
    process_button.config(state=tk.DISABLED)


# Frame utama
main_frame = Frame(root, bg="navy")
main_frame.pack(fill=tk.BOTH, expand=True)

# Frame untuk menampilkan video kamera
video_frame = Frame(main_frame, bg="navy")
video_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.BOTH, expand=True)

# Canvas untuk tampilan video kamera
video_canvas = Canvas(video_frame, width=400, height=300, bg="black")
video_canvas.pack()

# Frame untuk hasil tangkapan gambar
captured_images_frame = Frame(video_frame, bg="navy")
captured_images_frame.pack(pady=10)

# Array untuk menyimpan gambar-gambar yang ditangkap
captured_images = []
for _ in range(max_captures):
    img_label = Label(captured_images_frame, bg="navy")
    img_label.pack(side=tk.LEFT, padx=10, pady=10)
    captured_images.append(img_label)

# Frame untuk tombol proses dan reset
action_frame = Frame(main_frame, bg="navy")
action_frame.pack(side=tk.RIGHT, padx=20, pady=20, fill=tk.Y)

# Label judul
title_label = Label(
    action_frame,
    text="Deteksi Tingkat Kematangan Nanas",
    bg="navy",
    fg="white",
    font=("Helvetica", 18),
)
title_label.pack(pady=20, fill=tk.X)

# Tombol untuk menangkap gambar (berbentuk card hijau)
capture_button = Button(
    action_frame,
    text="Capture Image",
    command=capture_image,
    bg="green",
    fg="white",
)
capture_button.pack(pady=5, fill=tk.X)

# Label untuk keterangan kematangan
keterangan_label = Label(
    action_frame, text="Keterangan Kematangan", bg="navy", fg="white"
)
keterangan_label.pack(fill=tk.X)

# Label untuk status kematangan
status_kematangan_label = Label(
    action_frame, text="Status Kematangan:", bg="navy", fg="white"
)
status_kematangan_label.pack(fill=tk.X)

# Frame untuk hasil deteksi dari gambar yang diproses
processed_result_frame = Frame(action_frame, bg="green")
processed_result_frame.pack(fill=tk.X)

# Label untuk hasil deteksi dari gambar yang diproses
processed_result_label = Label(processed_result_frame, text="", bg="green", fg="white")
processed_result_label.pack(pady=10, fill=tk.X)

# Tombol untuk memproses gambar (berbentuk card hijau)
process_button = Button(
    action_frame,
    text="Process Images",
    command=process_images,
    state=tk.DISABLED,
    bg="green",
    fg="white",
)
process_button.pack(pady=10, fill=tk.X)

# Tombol untuk mereset GUI dan gambar yang ditangkap (berbentuk card hijau)
reset_button = Button(
    action_frame, text="Reset", command=reset_gui, bg="green", fg="white"
)
reset_button.pack(fill=tk.X)

# Memulai tampilan video dan deteksi dari kamera
detect_and_classify_frame()

root.mainloop()

# Tutup kamera setelah GUI ditutup
cap.release()
cv2.destroyAllWindows()
