import os
import cv2
from tkinter import filedialog
from tkinter import *
from tkinter import scrolledtext
from PIL import Image, ImageTk
from tkinter import  Label, Tk, Button, DISABLED, NORMAL
from video_tracker import process_video_frame

class ReaderUI(object):
    def __init__(self):
        video_path = 0  # 0 表示摄像头
        self.cap = cv2.VideoCapture(video_path)
        self.cap = None
        self.is_video_running = False
        self.image_list = []
        self.current_index = 0
        self.current_image_path = None
        self.windowSize = [1200, 800]  # 自定义窗口尺寸
        self.mainWindow = Tk()
        self.mainWindow.config(background='white')
        self.mainWindow.configure(bg="#ADD8E6")
        self.mainWindow.geometry("{}x{}".format(self.windowSize[0], self.windowSize[1]))  # 根据需求设置窗口尺寸
        self.center_window()  # 窗口居中
        self.mainWindow.resizable(False, False)  # 禁止调整窗口大小
        '''设置界面名称'''
        self.guiName = Label(
            self.mainWindow,
            text='xxx软件',
            font=('微软雅黑', 25, 'bold'),
            fg='black',
            bg="#ADD8E6"  # ⭐关键：和背景一样
        )
        self.cap = cv2.VideoCapture(video_path)
        self.guiName.place(x=500, y=10)
        '''-------------------------------相机图像显示-------------------------------'''
        # 左：原图
        self.imgShowWindow = self.set_icon_window("icon/interface.jpg", (600, 470), (282, 80))
        self.leftImageWindow = self.set_icon_window("icon/interface.jpg", (320, 320), (282, 160))
        self.leftImageWindow.place_forget()
        # 右：检测图（初始不显示）
        self.rightImageWindow = self.set_icon_window("icon/interface.jpg", (320, 320), (642, 160))
        self.rightImageWindow.place_forget()  # ⭐先隐藏
        '''----------------------------------系统信息显示栏----------------------------------'''

        self.systemInfoWindow = self.set_roll_window(144, 12, (10, 580))  # 系统信息滑动显示窗口

        '''----------------------------------组件功能----------------------------------'''
        self.darkFieldButton = self.set_text_button("图像导入", 9, 1, (1000, 100))
        self.darkFieldButton.configure(command=self.import_images)
        self.brightFieldButton = self.set_text_button("目标检测", 9, 1, (1000, 220))  # 定义暗场图像按钮
        self.brightFieldButton.configure(command=self.run_detection)
        self.processedImageButton = self.set_text_button("下一张图像", 9, 1, (1000, 340))  # 定义PSNR处理图像按钮
        self.processedImageButton.configure(command=self.show_next_image)
        self.detectAllButton = self.set_text_button("检测总览", 9, 1, (1000, 460))
        self.detectAllButton.configure(command=self.enable_overview_mode)
        self.videoButton = self.set_text_button(
            "视频导入", 9, 1, (95, 100),
            bg="#FFFFE0", hover_bg="#FFFACD"
        )
        self.videoButton.configure(command=self.open_video)

        self.detectBtn = self.set_text_button(
            "行人检测", 9, 1, (95, 220),
            bg="#FFFFE0", hover_bg="#FFFACD"
        )
        self.detectBtn.configure(command=self.enable_detection_mode)

        self.trackBtn = self.set_text_button(
            "行人轨迹", 9, 1, (95, 340),
            bg="#FFFFE0", hover_bg="#FFFACD"
        )
        self.trackBtn.configure(command=self.enable_tracking_mode)

        self.pauseBtn = self.set_text_button(
            "暂停/播放", 9, 1, (95, 460),
            bg="#FFFFE0", hover_bg="#FFFACD"
        )
        self.pauseBtn.configure(command=self.toggle_pause)
        self.current_name = None
        self.mode = "raw"  # detect / track
        self.is_paused = False
        # __init__ 里
        from ultralytics import YOLO

        # 图像检测模型（CBAM）
        self.image_model = YOLO(r'data/yolov8cbam_xyd/weights/best.pt')

        # 视频检测模型（COCO）
        self.video_model = YOLO('yolov8l.pt')
        # --- 波段信息轮换模板 -
        # 当前轮换下标

    def enable_detection_mode(self):
        self.mode = "detect"

    def enable_tracking_mode(self):
        self.mode = "track"

    def toggle_pause(self):
        self.is_paused = not self.is_paused

    def open_video(self):
        video_path = filedialog.askopenfilename(title="选择视频")

        if not video_path:
            return

        self.cap = cv2.VideoCapture(video_path)
        self.is_video_running = True
        self.mode = "raw"

        self.update_frame()

    def update_frame(self):
        if not self.is_video_running or self.cap is None:
            return

        if self.is_paused:
            self.mainWindow.after(30, self.update_frame)
            return

        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            self.is_video_running = False
            return

        # 👇 模式控制
        # 👇 模式控制
        if self.mode == "raw":
            result_frame = frame.copy()
            self.update_info("FPS: -- | 未开启检测")

        elif self.mode == "detect":
            results = self.video_model(frame, verbose=False)
            result_frame = results[0].plot()

            count = len(results[0].boxes) if results[0].boxes is not None else 0
            self.update_info(f"FPS: -- | Count: {count}")

        else:  # track
            result_frame, stats = process_video_frame(frame, self.video_model)

            fps, count, total, in_num, out_num = stats

            self.update_info(
                f"FPS: {fps:.2f} | Count: {count} | Total: {total} | In: {in_num} | Out: {out_num}"
            )

        # 显示图像
        img = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.resize_keep_ratio(img, (600, 470))
        tk_img = ImageTk.PhotoImage(img)

        self.imgShowWindow.configure(image=tk_img)
        self.imgShowWindow.image = tk_img

        self.mainWindow.after(30, self.update_frame)

    def update_info(self, text):
        self.systemInfoWindow.delete(1.0, END)
        self.systemInfoWindow.insert(END, text)

    def import_images(self):
        folder = filedialog.askdirectory(title="选择图像文件夹")
        if not folder:
            return

        self.image_list = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        self.image_list.sort()

        if not self.image_list:
            self.log("❌ 没有图像")
            return

        self.current_index = 0
        self.show_image(self.image_list[0])

    def show_image(self, path):
        img = Image.open(path)
        img = self.resize_keep_ratio(img, (600, 470))
        tk_img = ImageTk.PhotoImage(img)

        # 显示中间图
        self.imgShowWindow.place(x=280, y=80)
        self.imgShowWindow.configure(image=tk_img)
        self.imgShowWindow.image = tk_img

        # 隐藏左右图
        self.leftImageWindow.place_forget()
        self.rightImageWindow.place_forget()

        self.current_image_path = path

    def resize_keep_ratio(self, img, max_size=(600, 470)):
        w, h = img.size
        scale = min(max_size[0] / w, max_size[1] / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return img.resize((new_w, new_h))

    # 设置文字功能控件
    def set_text_button(self, text, widget_width, widget_height, place,
                        bg="#E6E6FA", hover_bg="#D8BFD8", bd=4):

        button = Button(
            self.mainWindow,
            text=text,
            bg=bg,
            fg="black",
            bd=bd,
            font=('楷体', 15, 'bold'),
            width=widget_width,
            height=widget_height,
        )

        # ===== 鼠标悬停效果 =====
        def on_enter(e):
            button['bg'] = hover_bg

        def on_leave(e):
            button['bg'] = bg

        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)

        button.place(x=place[0], y=place[1])

        return button

    def run_detection(self):
        if not self.current_image_path:
            self.log("请先导入图像")
            return

        # ❌ 隐藏中间图
        self.imgShowWindow.place_forget()

        # ===== 左图：原图 =====
        orig_img = Image.open(self.current_image_path)
        orig_img = self.resize_keep_ratio(orig_img, (320, 320))
        tk_orig = ImageTk.PhotoImage(orig_img)

        self.leftImageWindow.place(x=282, y=160)
        self.leftImageWindow.configure(image=tk_orig)
        self.leftImageWindow.image = tk_orig

        # ===== YOLO检测 =====
        results = self.image_model(self.current_image_path, verbose=False)

        result_img = results[0].plot()
        img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.resize_keep_ratio(img, (320, 320))
        tk_img = ImageTk.PhotoImage(img)

        # ===== 右图：检测结果 =====
        self.rightImageWindow.place(x=642, y=160)
        self.rightImageWindow.configure(image=tk_img)
        self.rightImageWindow.image = tk_img
        # ===== 获取检测信息 =====
        if results[0].boxes is not None:
            labels = []
            for cls in results[0].boxes.cls:
                labels.append(self.image_model.names[int(cls)])
            name = os.path.basename(self.current_image_path)
            self.log(f"{name} → 检测目标：{labels}")
        else:
            self.log("未检测到目标")

    def log(self, text):
        self.systemInfoWindow.insert(END, text + "\n")
        self.systemInfoWindow.see(END)

    def show_next_image(self):
        if not self.image_list:
            return

        self.current_index += 1
        if self.current_index >= len(self.image_list):
            self.current_index = 0

        self.show_image(self.image_list[self.current_index])

    def enable_overview_mode(self):
        save_dir = "results"
        os.makedirs(save_dir, exist_ok=True)

        for path in self.image_list:
            results = self.image_model(self.current_image_path)
            img = results[0].plot()

            save_path = os.path.join(save_dir, os.path.basename(path))
            cv2.imwrite(save_path, img)

        self.log("✅ 所有图像检测完成，已保存到 results/")

    # 设置信息滚动消息框
    def set_roll_window(self, widget_width, widget_height, place, bd=4):

        # 定义消息框内容
        text = scrolledtext.ScrolledText(self.mainWindow, width=widget_width, height=widget_height,
                                         bd=bd, font=('黑体', 12), fg="black",bg="#D3D3D3" )
        text.place(x=place[0], y=place[1])
        return text

    # 设置显示图窗
    def set_icon_window(self, icon_path, icon_size, place, color='gray', bd=4):

        # 读取图标, 设置为自定义尺寸, 并转换为可用格式
        icon = Image.open(icon_path)
        icon = icon.resize(icon_size)
        icon = ImageTk.PhotoImage(icon)

        # 定义显示控件, 设置为默认图像
        icon_window = Label(self.mainWindow, bd=bd, bg=color,relief="groove")
        icon_window.configure(image=icon)
        icon_window.image = icon

        # 自定义显示控件位置
        icon_window.place(x=place[0], y=place[1])
        return icon_window

    def info_save(self):
        pass

    def center_window(self):
        x = (self.mainWindow.winfo_screenwidth() // 2) - (self.windowSize[0] // 2)
        y = (self.mainWindow.winfo_screenheight() // 2) - (self.windowSize[1] // 2)
        self.mainWindow.geometry('+{}+{}'.format(x, y))