import torch
import random
from tkinter import *
import numpy as np
class LeNet5(torch.nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.act1  = torch.nn.ReLU()
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
       
        self.conv2 = torch.nn.Conv2d(
            in_channels=6, out_channels=64, kernel_size=5, padding=0)
        self.act2  = torch.nn.ReLU()
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.fc1   = torch.nn.Linear(5 * 5 * 64, 120)
        self.act3  = torch.nn.Tanh()
        
        self.fc2   = torch.nn.Linear(120, 84)
        self.act4  = torch.nn.Tanh()
        
        self.fc3   = torch.nn.Linear(84, 10)
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.act4(x)
        x = self.fc3(x)
        
        return x
lenet5 = torch.load('model', map_location=torch.device('cpu'))
lenet5.eval()


class Paint(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.tens = torch.zeros(28, 28)
        self.sc = 560//28
        self.create_widgets()

    def create_widgets(self):
        # print(self.tens)
        self.master.title("MNIST classifier")  # Устанавливаем название окна
        self.pack(fill=BOTH, expand=1)  # Размещаем активные элементы на родительском окне
        self.canv = Canvas(self, bg="white", width=600, height=600)  # Создаем поле для рисования, устанавливаем белый фон
        self.canv.bind("<B1-Motion>", self.print_x)
        self.canv.bind("<Button-1>", self.predict)
        self.canv.bind("<Button-3>", self.clear)
        self.canv.pack(side='left')
        self.canv1 = Canvas(self, bg="#888888", width=768-600, height=600)  # Создаем поле для рисования, устанавливаем белый фон
        self.canv1.pack(side='left')
        for x in range(1, 29):
            for y in range(1, 29):
                self.canv.create_rectangle(x*self.sc, 
                                           y*self.sc, 
                                           x*self.sc + self.sc, 
                                           y*self.sc + self.sc,
                                           fill='black',
                                           outline='white')
    def clear(self, event):
        self.canv.delete('all')
        self.canv1.delete('all')
        for x in range(1, 29):
            for y in range(1, 29):
                self.canv.create_rectangle(x*self.sc, 
                                           y*self.sc, 
                                           x*self.sc + self.sc, 
                                           y*self.sc + self.sc,
                                           fill='black',
                                           outline='white')
        self.tens = torch.zeros(28, 28)        
        
    def print_to_tens(self, x, y):
        self.tens[x, y] = 255
        if x+1 < 28:
            if self.tens[x+1, y] + 20 < 255:
                self.tens[x+1, y] += 20
        if x-1 > -1:
            if self.tens[x-1, y] + 20 < 255:
                self.tens[x-1, y] += 20
        if y+1 < 28:
            if self.tens[x, y+1] + 20 < 255:
                self.tens[x, y+1] += 20
        if y-1 > -1:
            if self.tens[x, y-1] + 20 < 255:
                self.tens[x, y-1] += 20
    
    def print_rect(self, x_cor, y_cor):
        self.canv.create_rectangle((x_cor+1)*self.sc, 
                                    (y_cor+1)*self.sc, 
                                    (x_cor+1)*self.sc + self.sc, 
                                    (y_cor+1)*self.sc + self.sc, 
                                    fill=self.fill_colour(x_cor, y_cor),
                                    outline='white')
        
    def print_to_canvas(self, x_cor, y_cor):
        self.print_rect(x_cor, y_cor)
        self.print_rect(x_cor+1, y_cor)
        self.print_rect(x_cor-1, y_cor)
        self.print_rect(x_cor, y_cor+1)
        self.print_rect(x_cor, y_cor-1)
        
    def fill_colour(self, x_cor, y_cor):
        fill_colour = min(255 ,self.tens[y_cor, x_cor].item())
        fill_colour = str(hex(int(fill_colour)))[2:]
        return '#' + fill_colour*3
    
    def print_results(self):
        self.canv1.create_rectangle(20, 
                                    20,
                                    200,
                                    600,
                                    fill="#888888",
                                    outline="#888888")
        
        for i in range(10):
            self.canv1.create_text(10, 38 + i*58, text=str(i))
            self.canv1.create_rectangle(20, 
                                        20 + i*58,
                                        20 + 160*self.preds[i],
                                        50 + i*58,
                                        fill='black',
                                        outline='white')
        max = self.preds.argmax()
        self.canv1.create_rectangle(20, 
                                    20 + max*58,
                                    20 + 160*self.preds[max],
                                    50 + max*58,
                                    fill='red',
                                    outline='white')
        
        
    
    def print_x(self, event):
        x_cor = event.x//self.sc - 1
        y_cor = event.y//self.sc - 1
        if -1<x_cor<29:
            if -1<y_cor<29:
                self.print_to_tens(y_cor, x_cor)
                self.print_to_canvas(x_cor, y_cor)
                
    def predict(self, event):
        self.preds = lenet5.forward(self.tens.resize(1, 1, 28, 28)).detach().numpy()[0]
        self.preds = (self.preds - self.preds.min())
        self.preds = self.preds/self.preds.max()
        self.print_results()
        
def main():
    root = Tk()
    root.geometry("768x600")
    app = Paint(root)
    root.mainloop()

if __name__ == "__main__":
    main()