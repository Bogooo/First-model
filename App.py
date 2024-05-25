import torch
import customtkinter as tk
from PIL import ImageGrab

from ModelTrain import SoftmaxModel, transform, predict_class, BaseModel

model = BaseModel()
model.load_state_dict(torch.load("BaseModel.pth"))
model.eval()


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.CTkCanvas(master=root, bg='white', width=280, height=280)
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset_prev_coords)
        self.canvas.bind("<B3-Motion>", self.erase)

        self.buttons = tk.CTkFrame(master=root)
        self.buttons.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.clear_button = tk.CTkButton(master=self.buttons, text="Clear", command=self.clear, font=("Arial", 25))
        self.clear_button.pack(padx=5, pady=5, side=tk.LEFT)

        self.test_button = tk.CTkButton(master=self.buttons, text="Test", command=self.test, font=("Arial", 25))
        self.test_button.pack(padx=5, pady=5, side=tk.LEFT)

        self.result_label = tk.CTkLabel(master=self.buttons, text="Result: ", width=70, font=("Arial", 25))
        self.result_label.pack(padx=5, pady=5, side=tk.LEFT)
        self.prev_coords = None

    def reset_prev_coords(self, event):
        self.prev_coords = None

    def draw(self, event):
        x, y = event.x, event.y
        if self.prev_coords:
            x1, y1 = self.prev_coords
            self.canvas.create_line(x1, y1, x, y, fill="black", width=2)
        self.prev_coords = (x, y)

    def erase(self, event):
        x, y = event.x, event.y
        item = self.canvas.find_closest(x, y)
        self.canvas.delete(item)

    def test(self):
        canvas_coords = self.canvas.winfo_rootx(), self.canvas.winfo_rooty(), self.canvas.winfo_rootx() + 280, self.canvas.winfo_rooty() + 280
        image = ImageGrab.grab(bbox=canvas_coords)
        image = transform(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            self.result_label.configure(text=f'Result: {predict_class(output)}')

    def clear(self):
        self.canvas.delete("all")


root = tk.CTk()
app = DrawingApp(root)
root.mainloop()
