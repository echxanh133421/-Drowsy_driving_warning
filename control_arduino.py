import serial
import tkinter as tk

arduino=serial.Serial('com5',baudrate=9600)

def open_warning():
    arduino.write(b'1')
    print('open_warning')
def close_warning():
    arduino.write(b'0')
    print('close warning')

if __name__=="__main__":
    root=tk.Tk()

    bt1=tk.Button(root,text='bat',command=open_warning)
    bt2=tk.Button(root,text='tat',command=close_warning)

    bt1.grid(row=0,column=0)
    bt2.grid(row=0,column=1)
    root.geometry('200x300')
    root.mainloop()