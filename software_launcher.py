import tkinter as tk
from tkinter import messagebox, scrolledtext, simpledialog
import subprocess
import threading
import sys
import os

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HỆ THỐNG NHẬN DIỆN THỦ NGỮ AI - PRO")
        self.root.geometry("900x650")
        self.root.configure(bg="#2C3E50")

        # --- HEADER ---
        header_frame = tk.Frame(root, bg="#2C3E50")
        header_frame.pack(pady=20)
        tk.Label(header_frame, text="PHẦN MỀM HỖ TRỢ NGƯỜI KHIẾM THÍNH", 
                 font=("Arial", 14), fg="#BDC3C7", bg="#2C3E50").pack()
        tk.Label(header_frame, text="AI SIGN LANGUAGE TRANSLATOR", 
                 font=("Arial", 22, "bold"), fg="#ECF0F1", bg="#2C3E50").pack()

        # --- BUTTONS ---
        btn_frame = tk.Frame(root, bg="#2C3E50")
        btn_frame.pack(pady=20)

        # Nút 1: Nhận diện
        self.btn_detect = tk.Button(btn_frame, text="📷 NHẬN DIỆN (DETECT)", font=("Arial", 11, "bold"),
                                    bg="#3498DB", fg="white", width=25, height=2, 
                                    command=self.open_detection_dialog)
        self.btn_detect.grid(row=0, column=0, padx=10)

        # Nút 2: Thu thập
        self.btn_collect = tk.Button(btn_frame, text="➕ THU THẬP (COLLECT)", font=("Arial", 11, "bold"),
                                     bg="#E67E22", fg="white", width=25, height=2, 
                                     command=self.open_collection_dialog)
        self.btn_collect.grid(row=0, column=1, padx=10)

        # Nút 3: Huấn luyện
        self.btn_train = tk.Button(btn_frame, text="🧠 HUẤN LUYỆN (TRAIN MODEL)", font=("Arial", 13, "bold"),
                                   bg="#27AE60", fg="white", width=54, height=2, 
                                   command=self.start_training_thread)
        self.btn_train.grid(row=1, column=0, columnspan=2, pady=15)

        # --- LOGGING AREA ---
        log_frame = tk.Frame(root, bg="#2C3E50")
        log_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        tk.Label(log_frame, text="NHẬT KÝ HỆ THỐNG (SYSTEM LOGS):", fg="#F1C40F", bg="#2C3E50", font=("Consolas", 10, "bold")).pack(anchor="w")
        
        # ScrolledText để hiển thị log
        self.log_area = scrolledtext.ScrolledText(log_frame, height=15, bg="#1E272E", fg="#ECF0F1", 
                                                  font=("Consolas", 10), state='disabled')
        self.log_area.pack(fill="both", expand=True)

        # Tag màu cho log đẹp hơn
        self.log_area.tag_config("INFO", foreground="#ECF0F1")
        self.log_area.tag_config("SUCCESS", foreground="#2ECC71") # Xanh lá
        self.log_area.tag_config("WARNING", foreground="#F1C40F") # Vàng
        self.log_area.tag_config("ERROR", foreground="#E74C3C")   # Đỏ

        self.log(">> Hệ thống đã sẵn sàng...", "INFO")

    def log(self, msg, tag="INFO"):
        """Hàm ghi log an toàn từ luồng khác"""
        def _log():
            self.log_area.config(state='normal')
            self.log_area.insert(tk.END, str(msg) + "\n", tag)
            self.log_area.see(tk.END)
            self.log_area.config(state='disabled')
        self.root.after(0, _log) # Đẩy vào hàng đợi của main thread

    # --- CÁC CHỨC NĂNG ---

    def open_detection_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Cài đặt nhận diện")
        dialog.geometry("300x160")
        dialog.configure(bg="#ECF0F1")
        
        tk.Label(dialog, text="Chọn chế độ:", bg="#ECF0F1", font=("Arial", 11)).pack(pady=10)
        
        def run(n):
            dialog.destroy()
            self.run_process("main_app.py", [str(n)])

        tk.Button(dialog, text="1 Tay (Chính xác cao)", bg="#2ECC71", fg="white", width=20, command=lambda: run(1)).pack(pady=5)
        tk.Button(dialog, text="2 Tay (Thử nghiệm)", bg="#F39C12", fg="white", width=20, command=lambda: run(2)).pack(pady=5)

    def open_collection_dialog(self):
        label = simpledialog.askstring("Thu thập dữ liệu", "Nhập tên KÝ TỰ muốn dạy:")
        if label:
            self.run_process("collect_extra_data.py", [label])
        else:
            self.log("Đã hủy thu thập.", "WARNING")

    def start_training_thread(self):
        if hasattr(self, "is_training") and self.is_training:
            return
        
        self.is_training = True
        self.btn_train.config(state="disabled", text="⏳ ĐANG CHẠY HUẤN LUYỆN...", bg="#7F8C8D")
        self.log("="*40, "INFO")
        self.log("BẮT ĐẦU TIẾN TRÌNH HUẤN LUYỆN...", "INFO")
        
        # Chạy trong luồng riêng
        threading.Thread(target=self.run_training_process, daemon=True).start()

    # def run_process(self, script_name, args=[]):
    #     """Hàm chạy các file main/collect (có cửa sổ riêng)"""
    #     self.log(f"🚀 Đang khởi chạy: {script_name}...", "INFO")
    #     try:
    #         cmd = [sys.executable, script_name] + args
    #         subprocess.Popen(cmd)
    #     except Exception as e:
    #         self.log(f"Lỗi khởi chạy: {e}", "ERROR")
    def run_process(self, script_name, args=[]):
        """Hàm chạy các file con (Tự động nhận diện .py hay .exe)"""
        self.log(f"🚀 Đang khởi chạy: {script_name}...", "INFO")
        try:
            # Kiểm tra xem đang chạy ở chế độ đóng gói (EXE) hay mã nguồn (Python)
            if getattr(sys, 'frozen', False):
                # Đang chạy file EXE -> Gọi file exe con
                exe_name = script_name.replace(".py", ".exe")
                cmd = [exe_name] + args
            else:
                # Đang chạy Python -> Gọi lệnh python
                cmd = [sys.executable, script_name] + args

            # Lệnh Popen giữ nguyên
            subprocess.Popen(cmd)
        except Exception as e:
            self.log(f"Lỗi khởi chạy: {e}", "ERROR")

    def run_training_process(self):
            """Hàm chạy training ngầm và bắt log"""
            try:
                # --- KHẮC PHỤC LỖI UNICODE TRÊN WINDOWS ---
                # Tạo bản sao biến môi trường và ép mã hóa UTF-8 cho luồng IO
                custom_env = os.environ.copy()
                custom_env["PYTHONIOENCODING"] = "utf-8"
                # ------------------------------------------

                # Trong hàm run_training_process:
                if getattr(sys, 'frozen', False):
                    cmd = ["train_model.exe"] # Gọi exe trực tiếp
                else:
                    cmd = [sys.executable, "-u", "train_model.py"] # Gọi qua python
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, 
                    text=True,
                    encoding='utf-8', 
                    env=custom_env, # Thêm tham số này vào
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )

                # Đọc từng dòng log
                while True:
                    # Dùng try-except nhỏ ở đây để tránh crash nếu có ký tự lạ xót lại
                    try:
                        line = process.stdout.readline()
                    except UnicodeDecodeError:
                        continue # Bỏ qua dòng lỗi mã hóa

                    if not line and process.poll() is not None:
                        break
                    if line:
                        clean_line = line.strip()
                        if clean_line:
                            if "Lỗi" in clean_line or "Error" in clean_line:
                                self.log(clean_line, "ERROR")
                            elif "Tiến độ" in clean_line:
                                self.log(clean_line, "WARNING")
                            elif "ĐỘ CHÍNH XÁC" in clean_line:
                                self.log(clean_line, "SUCCESS")
                            else:
                                self.log(clean_line, "INFO")

                rc = process.poll()
                if rc == 0:
                    self.log("✅ HUẤN LUYỆN HOÀN TẤT!", "SUCCESS")
                    messagebox.showinfo("Thành công", "Huấn luyện Model xong!")
                else:
                    self.log(f"❌ Có lỗi xảy ra. Mã lỗi: {rc}", "ERROR")

            except Exception as e:
                self.log(f"Lỗi nghiêm trọng: {e}", "ERROR")
            
            finally:
                self.is_training = False
                self.root.after(0, lambda: self.btn_train.config(state="normal", text="🧠 HUẤN LUYỆN (TRAIN MODEL)", bg="#27AE60"))
if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()