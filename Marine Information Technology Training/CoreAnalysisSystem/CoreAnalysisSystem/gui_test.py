import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

# 解决matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 导入优化后的分析模块
from hole_modules import analysis_holes
from crack_modules import analysis_crack
from grain_modules import analyze_grains


class CoreAnalysisApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("地质岩心图文分析系统 · 岩芯智析")
        self.root.geometry("1350x850")
        self.root.minsize(1100, 700)

        try:
            self.root.iconphoto(True, self.icon)
        except:
            pass

        # 图像数据
        self.original_image = None
        self.analysis_result = None
        self.analysis_type = None

        # 中间图像
        self.current_gray = None
        self.current_binary = None
        self.current_marked = None

        self.setup_style()
        self.setup_main_layout()

    def setup_style(self):
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('.', background='#F5F7FA', foreground='#1E2A3E', font=('Segoe UI', 10))
        style.configure('TFrame', background='#F5F7FA')
        style.configure('TLabel', background='#F5F7FA', foreground='#1E2A3E')
        style.configure('TLabelframe', background='#FFFFFF', foreground='#2C7DA0',
                        bordercolor='#D0D7DE', relief='groove', borderwidth=1)
        style.configure('TLabelframe.Label', background='#FFFFFF', foreground='#2C7DA0',
                        font=('Segoe UI', 10, 'bold'))
        style.configure('TEntry', fieldbackground='#FFFFFF', foreground='#1E2A3E',
                        borderwidth=1, relief='solid', focuscolor='#2C7DA0')
        style.map('TEntry', fieldbackground=[('focus', '#FFFFFF')])
        style.configure('TButton', background='#2C7DA0', foreground='#FFFFFF',
                        borderwidth=0, focusthickness=0, font=('Segoe UI', 9, 'bold'))
        style.map('TButton', background=[('active', '#1F6390')])
        style.configure('Accent.TButton', background='#1E7E34', foreground='white')
        style.map('Accent.TButton', background=[('active', '#16632A')])

        style.configure('TNotebook', background='#F5F7FA', borderwidth=0)
        style.configure('TNotebook.Tab', background='#E9ECEF', foreground='#1E2A3E',
                        padding=[12, 4], font=('Segoe UI', 10))
        style.map('TNotebook.Tab', background=[('selected', '#FFFFFF')], foreground=[('selected', '#2C7DA0')])

    def setup_main_layout(self):
        # 主水平分割
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ========= 左侧区域：参数面板 + 按钮 =========
        left_container = ttk.Frame(main_pane, width=320)
        main_pane.add(left_container, weight=0)
        left_container.pack_propagate(False)

        # 创建一个可滚动的 Canvas 以容纳较多参数（可选，保持简洁使用Frame）
        param_canvas = tk.Canvas(left_container, bg='#F5F7FA', highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_container, orient=tk.VERTICAL, command=param_canvas.yview)
        scrollable_frame = ttk.Frame(param_canvas)
        scrollable_frame.bind("<Configure>", lambda e: param_canvas.configure(scrollregion=param_canvas.bbox("all")))
        param_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        param_canvas.configure(yscrollcommand=scrollbar.set)

        param_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 1. 孔洞参数
        hole_frame = ttk.LabelFrame(scrollable_frame, text="🕳️ 孔洞分析参数", padding=10)
        hole_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(hole_frame, text="最小面积:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.hole_min_area = ttk.Entry(hole_frame, width=12)
        self.hole_min_area.insert(0, "1")
        self.hole_min_area.grid(row=0, column=1, padx=10, pady=2)

        ttk.Label(hole_frame, text="最大面积:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.hole_max_area = ttk.Entry(hole_frame, width=12)
        self.hole_max_area.insert(0, "1000")
        self.hole_max_area.grid(row=1, column=1, padx=10, pady=2)

        ttk.Label(hole_frame, text="阈值:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.hole_threshold = ttk.Entry(hole_frame, width=12)
        self.hole_threshold.insert(0, "100")
        self.hole_threshold.grid(row=2, column=1, padx=10, pady=2)

        # 以下孔洞参数不在界面显示，使用稳定默认值：
        # 圆形度阈值=0.5、CLAHE对比度=2.0、形态学核大小=5。

        # 2. 裂缝参数
        crack_frame = ttk.LabelFrame(scrollable_frame, text="⚡ 裂缝分析参数", padding=10)
        crack_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(crack_frame, text="最小面积:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.crack_min_area = ttk.Entry(crack_frame, width=12)
        self.crack_min_area.insert(0, "1000")
        self.crack_min_area.grid(row=0, column=1, padx=10, pady=2)

        ttk.Label(crack_frame, text="最大面积:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.crack_max_area = ttk.Entry(crack_frame, width=12)
        self.crack_max_area.insert(0, "inf")
        self.crack_max_area.grid(row=1, column=1, padx=10, pady=2)

        ttk.Label(crack_frame, text="阈值:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.crack_threshold = ttk.Entry(crack_frame, width=12)
        self.crack_threshold.insert(0, "100")
        self.crack_threshold.grid(row=2, column=1, padx=10, pady=2)

        # 以下裂缝参数不在界面显示，使用稳定默认值：
        # 实体度阈值=0.5、最小噪声面积=50。

        # 3. 粒度参数
        grain_frame = ttk.LabelFrame(scrollable_frame, text="🪨 粒度分析参数", padding=10)
        grain_frame.pack(fill=tk.X, pady=(0, 10))

        # 为了减少界面参数，以下粒度参数在程序中固定：
        # 最小面积=5、分水岭峰值比例=0.06、种子最小距离=自动、
        # 启用深色颗粒补偿=True、暗粒最低亮度=20。
        ttk.Label(grain_frame, text="最大面积:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.grain_max_area = ttk.Entry(grain_frame, width=12)
        self.grain_max_area.insert(0, "5000")
        self.grain_max_area.grid(row=0, column=1, padx=10, pady=2)

        ttk.Label(grain_frame, text="基质百分位:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.grain_matrix_percentile = ttk.Entry(grain_frame, width=12)
        self.grain_matrix_percentile.insert(0, "28")
        self.grain_matrix_percentile.grid(row=1, column=1, padx=10, pady=2)

        ttk.Label(grain_frame, text="提示：最大面积可输入 inf 表示无上限", foreground="#6B7280").grid(
            row=2, column=0, columnspan=2, sticky=tk.W, pady=(4, 0)
        )

        # 4. 操作按钮
        button_frame = ttk.LabelFrame(scrollable_frame, text="操作", padding=10)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        self.open_btn = ttk.Button(button_frame, text="📂 打开图像", command=self.open_image, style='Accent.TButton')
        self.open_btn.pack(fill=tk.X, pady=3)

        self.hole_btn = ttk.Button(button_frame, text="🕳️ 孔洞分析", command=lambda: self.run_analysis('hole'))
        self.hole_btn.pack(fill=tk.X, pady=3)

        self.crack_btn = ttk.Button(button_frame, text="⚡ 裂缝分析", command=lambda: self.run_analysis('crack'))
        self.crack_btn.pack(fill=tk.X, pady=3)

        self.grain_btn = ttk.Button(button_frame, text="🪨 粒度分析", command=lambda: self.run_analysis('grain'))
        self.grain_btn.pack(fill=tk.X, pady=3)

        self.help_btn = ttk.Button(button_frame, text="❓ 帮助", command=self.show_help)
        self.help_btn.pack(fill=tk.X, pady=(10, 0))

        # ========= 右侧工作区 =========
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=1)

        # Notebook 标签页
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.figures = {}
        self.canvases = {}
        for name, title in [('original', '📷 原图'), ('gray', '🌑 灰度图'),
                            ('binary', '⚫ 二值图'), ('marked', '🏷️ 标记图')]:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=title)
            fig = Figure(figsize=(6, 4), dpi=100, facecolor='#FFFFFF')
            ax = fig.add_subplot(111)
            ax.axis('off')
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.figures[name] = (fig, ax)
            self.canvases[name] = canvas

        # 结果显示
        result_container = ttk.LabelFrame(right_frame, text="📋 分析结果", padding=5)
        result_container.pack(fill=tk.BOTH, expand=True)

        text_frame = ttk.Frame(result_container)
        text_frame.pack(fill=tk.BOTH, expand=True)
        scrollbar_res = ttk.Scrollbar(text_frame)
        scrollbar_res.pack(side=tk.RIGHT, fill=tk.Y)

        self.result_text = tk.Text(text_frame, wrap=tk.WORD, font=('Consolas', 10),
                                   bg='#FFFFFF', fg='#1E2A3E', relief='flat',
                                   yscrollcommand=scrollbar_res.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_res.config(command=self.result_text.yview)

        self.clear_all_images()

    def clear_all_images(self):
        for name in self.figures:
            fig, ax = self.figures[name]
            ax.clear()
            ax.axis('off')
            self.canvases[name].draw()
        self.result_text.delete(1.0, tk.END)

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("图像文件", "*.jpg *.png *.bmp *.jpeg *.tif")]
        )
        if not file_path:
            return
        try:
            pil_img = Image.open(file_path)
            self.original_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            messagebox.showerror("错误", f"无法读取图像：{str(e)}")
            return

        self.clear_all_images()
        self.current_gray = self.current_binary = self.current_marked = None
        self.analysis_result = None
        self.analysis_type = None

        fig, ax = self.figures['original']
        ax.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        self.canvases['original'].draw()
        self.notebook.select(0)

        for name in ['gray', 'binary', 'marked']:
            fig, ax = self.figures[name]
            ax.clear()
            ax.axis('off')
            self.canvases[name].draw()

    def run_analysis(self, analysis_type):
        if self.original_image is None:
            messagebox.showwarning("提示", "请先打开图像")
            return

        # 禁用分析按钮，避免重复点击（可选）
        for btn in [self.hole_btn, self.crack_btn, self.grain_btn]:
            btn.config(state=tk.DISABLED)
        self.root.update()

        try:
            if analysis_type == 'hole':
                min_area = float(self.hole_min_area.get())
                max_area = float(self.hole_max_area.get()) if self.hole_max_area.get() != "inf" else np.inf
                thresh = int(self.hole_threshold.get())
                # 界面中删除的参数使用默认值，减少用户调参负担。
                circularity = 0.5
                clahe_clip = 2.0
                morph_size = 5

                result, gray, binary, marked = analysis_holes(
                    self.original_image, min_area, max_area, thresh,
                    circularity_thresh=circularity, clahe_clip=clahe_clip,
                    morph_kernel_size=morph_size
                )
                self.analysis_result = result
                self.analysis_type = 'hole'
                self.current_gray = gray
                self.current_binary = binary
                self.current_marked = marked

                self._update_image_tab('gray', gray, cmap='gray')
                self._update_image_tab('binary', binary, cmap='gray')
                self._update_image_tab('marked', marked, is_color=True)

                info = (f"孔洞数量: {result['孔洞数量']}\n"
                        f"总面积: {result['总面积']:.2f} 像素\n"
                        f"平均面积: {result['平均面积']:.2f} 像素\n"
                        f"平均圆形度: {result['平均圆形度']:.3f}")
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, info)
                self.notebook.select(3)

            elif analysis_type == 'crack':
                min_area = float(self.crack_min_area.get())
                max_area = float(self.crack_max_area.get()) if self.crack_max_area.get() != "inf" else np.inf
                thresh = int(self.crack_threshold.get())
                # 界面中删除的参数使用默认值，减少用户调参负担。
                solidity = 0.5
                min_noise = 50

                result = analysis_crack(
                    self.original_image, min_area, max_area, thresh,
                    solidity_thresh=solidity, min_noise_area=min_noise
                )
                self.analysis_result = result
                self.analysis_type = 'crack'

                if '二值图' not in result or '结果图' not in result:
                    messagebox.showerror("错误", "裂缝分析结果不完整")
                    return

                self.current_binary = result['二值图']
                self.current_marked = result['结果图']
                self._update_image_tab('binary', self.current_binary, cmap='gray')
                self._update_image_tab('marked', self.current_marked, is_color=True)
                # 灰度图
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                self._update_image_tab('gray', gray, cmap='gray')

                if '特征' in result and result['特征']:
                    f = result['特征']
                    info = (f"裂缝数量: {f['数量']}\n"
                            f"总面积: {f['总面积']:.2f} 像素\n"
                            f"平均面积: {f['平均面积']:.2f} 像素\n"
                            f"最大裂缝方向: {f['最大裂缝方向']}\n"
                            f"最大裂缝长度: {f['最大裂缝长度']:.2f} 像素\n"
                            f"最大裂缝最大宽度: {f['最大裂缝最大宽度']:.2f} 像素\n"
                            f"最大裂缝最小宽度: {f['最大裂缝最小宽度']:.2f} 像素")
                else:
                    info = "未检测到符合条件的裂缝"
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, info)
                self.notebook.select(2)

            elif analysis_type == 'grain':
                # 粒度界面只保留最大面积和基质百分位；其余参数使用固定默认值。
                # 固定值：min_area=5、dist_peak_ratio=0.06、seed_min_distance=None、
                # recover_dark_grains=True、dark_min_l=20。
                min_area = 5
                max_area_text = self.grain_max_area.get().strip().lower()
                max_area = float(max_area_text) if max_area_text != "inf" else np.inf
                matrix_percentile = float(self.grain_matrix_percentile.get())
                dist_peak_ratio = 0.06

                result, gray, binary, marked = analyze_grains(
                    self.original_image,
                    min_area=min_area,
                    max_area=max_area,
                    matrix_percentile=matrix_percentile,
                    dist_peak_ratio=dist_peak_ratio,
                    seed_min_distance=None,
                    recover_dark_grains=True,
                    dark_min_l=20,
                    reject_border_dark=True
                )
                self.analysis_result = result
                self.analysis_type = 'grain'
                self._update_image_tab('gray', gray, cmap='gray')
                self._update_image_tab('binary', binary, cmap='gray')
                self._update_image_tab('marked', marked, is_color=True)

                info = (
                    f"颗粒数量: {result['粒子数量']}\n"
                    f"总面积: {result['总面积']:.2f} 像素\n"
                    f"平均面积: {result['平均面积']:.2f} 像素\n"
                    f"平均等效直径: {result['平均等效直径']:.2f} 像素\n"
                    f"D10: {result['D10']:.2f} 像素\n"
                    f"D50: {result['D50']:.2f} 像素\n"
                    f"D90: {result['D90']:.2f} 像素\n"
                    f"平均圆形度: {result['平均圆形度']:.3f}\n"
                    f"平均长短轴比: {result['平均长短轴比']:.3f}\n"
                    f"粒度面积占比: {result['粒度面积占比'] * 100:.2f}%\n"
                    f"基质百分位: {result['基质百分位']:.2f}"
                )
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, info)
                self.notebook.select(3)

        except ValueError as e:
            messagebox.showerror("错误", f"参数输入有误：{str(e)}")
        except Exception as e:
            messagebox.showerror("错误", f"分析失败：{str(e)}")
        finally:
            for btn in [self.hole_btn, self.crack_btn, self.grain_btn]:
                btn.config(state=tk.NORMAL)

    def _update_image_tab(self, tab_name, image, cmap=None, is_color=False):
        if image is None:
            return
        fig, ax = self.figures[tab_name]
        ax.clear()
        if is_color:
            if len(image.shape) == 3:
                img_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                img_display = image
            ax.imshow(img_display)
        else:
            ax.imshow(image, cmap=cmap if cmap else 'gray')
        ax.axis('off')
        self.canvases[tab_name].draw()

    def show_help(self):
        help_text = (
            "📌 使用说明\n\n"
            "1. 点击「打开图像」按钮，选择岩心照片。\n"
            "2. 在左侧调节各项分析参数（面积、阈值、基质百分位等）。\n"
            "3. 点击对应分析按钮（孔洞/裂缝/粒度）。\n"
            "4. 分析完成后，可通过上方标签页查看原图、灰度图、二值图和标记图。\n"
            "5. 右侧文本框显示详细量化结果。\n\n"
            "💡 提示：最大面积可输入 'inf' 表示无上限。\n"
            "🔧 孔洞/裂缝分析：阈值越大，通常检出越多；阈值越小，结果越保守。\n"
            "🔧 粒度分析：基质百分位越小，粉色颗粒区域通常越多。"
        )
        messagebox.showinfo("帮助", help_text)


if __name__ == "__main__":
    app = CoreAnalysisApp()
    app.root.mainloop()