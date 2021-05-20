import sys
from tkinter import filedialog
import ANPRModel
import logging
import threading
import time
import process_plate

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk

    py3 = False
except ImportError:
    import tkinter.ttk as ttk

    py3 = True


def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    top = GUI(root)
    root.mainloop()


w = None


def create_GUI(rt, *args, **kwargs):
    '''Starting point when module is imported by another module.
       Correct form of call: 'create_GUI(root, *args, **kwargs)' .'''
    global w, w_win, root
    # rt = root
    root = rt
    w = tk.Toplevel(root)
    top = GUI(w)
    return (w, top)


def destroy_GUI():
    global w
    w.destroy()
    w = None


def exitButton():
    exit(0)


class GUI:
    trainDir = ""
    testDir = ""
    trainLabels = ""
    testLabels = ""

    def updateProgressbar(self, progress):
        self.TProgressbar1['value'] = progress
        root.update()

    def trainDirectory(self):
        self.trainDir = filedialog.askdirectory()
        self.Scrolledtext1.configure(state="normal")
        self.Scrolledtext1.insert(tk.INSERT, "SET TRAIN DATA DIRECTORY: " + self.trainDir + "\n")
        self.Scrolledtext1.configure(state="disabled")

    def testDirectory(self):
        self.testDir = filedialog.askdirectory()
        self.Scrolledtext1.configure(state="normal")
        self.Scrolledtext1.insert(tk.INSERT, "SET TEST DATA DIRECTORY: " + self.testDir + "\n")
        self.Scrolledtext1.configure(state="disabled")

    def loadModel(self):
        self.modelDir = filedialog.askdirectory()
        self.Scrolledtext1.configure(state="normal")
        self.Scrolledtext1.insert(tk.INSERT, "LOADED MODEL: " + self.modelDir + "\n")
        self.Scrolledtext1.configure(state="disabled")

    def initModel(self):
        self.model = ANPRModel.ANPRModel(gui=self)
        self.Scrolledtext1.configure(state="normal")
        self.Scrolledtext1.insert(tk.INSERT, "INITIATED ANPR MODEL!" + "\n")
        self.Scrolledtext1.configure(state="disabled")

    def trainModel(self):
        self.model.trainModel(self.trainDir, self.trainLabels, self.testDir, self.testLabels, self.GPU.get())

    def trainThread(self):
        x = threading.Thread(target=self.trainModel)
        x.start()

    def showImage(self, image):
        self.Display.create_image(720, 160, anchor=tk.E, image=image)
        root.mainloop()

    def stopTraining(self):
        self.model.model.stop_training = True

    def setTrainLabel(self):
        self.trainLabels = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.trainLabels:
            try:
                self.Scrolledtext1.configure(state="normal")
                self.Scrolledtext1.insert(tk.INSERT, "SET TRAIN LABELS FILE TO: " + self.trainLabels + "\n")
                self.Scrolledtext1.configure(state="disabled")
            except:
                self.Scrolledtext1.configure(state="normal")
                self.Scrolledtext1.insert(tk.INSERT, "FAILED TO LOAD TRAIN LABELS FILE\n")
                self.Scrolledtext1.configure(state="disabled")

    def setTestLabel(self):
        self.testLabels = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.testLabels:
            try:
                self.Scrolledtext1.configure(state="normal")
                self.Scrolledtext1.insert(tk.INSERT, "SET TEST LABELS FILE TO: " + self.testLabels + "\n")
                self.Scrolledtext1.configure(state="disabled")
            except:
                self.Scrolledtext1.configure(state="normal")
                self.Scrolledtext1.insert(tk.INSERT, "FAILE TO LOAD TEST LABELS FILE\n")
                self.Scrolledtext1.configure(state="disabled")

    def setTestImage(self):
        self.testImage = filedialog.askopenfilename(filetypes=[("PNG images", "*.png"), ("JPG images", "*.jpg")])
        if self.testImage:
            try:
                self.Scrolledtext1.configure(state="normal")
                self.Scrolledtext1.insert(tk.INSERT, "SET TEST IMAGE FILE TO: " + self.testImage + "\n")
                self.Scrolledtext1.configure(state="disabled")
            except:
                self.Scrolledtext1.configure(state="normal")
                self.Scrolledtext1.insert(tk.INSERT, "FAILED TO TEST IMAGE FILE\n")
                self.Scrolledtext1.configure(state="disabled")

    def testLoadedImage(self):
        process_plate.segmentLP(self.testImage, self)

    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9'  # X11 color: 'gray85'
        _ana1color = '#d9d9d9'  # X11 color: 'gray85'
        _ana2color = '#ececec'  # Closest X11 color: 'gray92'
        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')
        self.style.configure('.', background=_bgcolor)
        self.style.configure('.', foreground=_fgcolor)
        self.style.map('.', background=
        [('selected', _compcolor), ('active', _ana2color)])

        top.geometry("972x552+422+174")
        top.minsize(120, 1)
        top.maxsize(3844, 1061)
        top.resizable(1, 1)
        top.title("New Toplevel")
        top.configure(background="#d9d9d9")
        top.configure(highlightbackground="#d9d9d9")
        top.configure(highlightcolor="black")

        self.LoadTrainFolder = tk.Button(top)
        self.LoadTrainFolder.place(relx=0.823, rely=0.036, height=44, width=77)
        self.LoadTrainFolder.configure(command=self.trainDirectory)
        self.LoadTrainFolder.configure(activebackground="#ececec")
        self.LoadTrainFolder.configure(activeforeground="#000000")
        self.LoadTrainFolder.configure(background="#d9d9d9")
        self.LoadTrainFolder.configure(disabledforeground="#a3a3a3")
        self.LoadTrainFolder.configure(foreground="#000000")
        self.LoadTrainFolder.configure(highlightbackground="#d9d9d9")
        self.LoadTrainFolder.configure(highlightcolor="black")
        self.LoadTrainFolder.configure(pady="0")
        self.LoadTrainFolder.configure(text='''Train Folder''')

        self.LoadTrainLabels = tk.Button(top)
        self.LoadTrainLabels.place(relx=0.823, rely=0.145, height=44, width=77)
        self.LoadTrainLabels.configure(command=self.setTrainLabel)
        self.LoadTrainLabels.configure(activebackground="#ececec")
        self.LoadTrainLabels.configure(activeforeground="#000000")
        self.LoadTrainLabels.configure(background="#d9d9d9")
        self.LoadTrainLabels.configure(disabledforeground="#a3a3a3")
        self.LoadTrainLabels.configure(foreground="#000000")
        self.LoadTrainLabels.configure(highlightbackground="#d9d9d9")
        self.LoadTrainLabels.configure(highlightcolor="black")
        self.LoadTrainLabels.configure(pady="0")
        self.LoadTrainLabels.configure(text='''Train Labels''')

        self.Display = tk.Canvas(top)
        self.Display.place(relx=0.021, rely=0.036, relheight=0.583
                           , relwidth=0.787)
        self.Display.configure(background="#d9d9d9")
        self.Display.configure(borderwidth="2")
        self.Display.configure(highlightbackground="#d9d9d9")
        self.Display.configure(highlightcolor="black")
        self.Display.configure(insertbackground="black")
        self.Display.configure(relief="ridge")
        self.Display.configure(selectbackground="blue")
        self.Display.configure(selectforeground="white")

        self.Status = tk.Message(top)
        self.Status.place(relx=0.021, rely=0.634, relheight=0.063
                          , relwidth=0.783)
        self.Status.configure(background="#d9d9d9")
        self.Status.configure(foreground="#000000")
        self.Status.configure(highlightbackground="#d9d9d9")
        self.Status.configure(highlightcolor="black")
        self.Status.configure(text='''Message''')
        self.Status.configure(width=761)

        self.GPU = tk.IntVar()
        self.GPUbutton = tk.Checkbutton(top, variable=self.GPU)
        self.GPUbutton.place(relx=0.823, rely=0.236, relheight=0.056
                             , relwidth=0.17)
        self.GPUbutton.configure(activebackground="#ececec")
        self.GPUbutton.configure(activeforeground="#000000")
        self.GPUbutton.configure(background="#d9d9d9")
        self.GPUbutton.configure(disabledforeground="#a3a3a3")
        self.GPUbutton.configure(foreground="#000000")
        self.GPUbutton.configure(highlightbackground="#d9d9d9")
        self.GPUbutton.configure(highlightcolor="black")
        self.GPUbutton.configure(justify='left')
        self.GPUbutton.configure(text='''Train on GPU?''')

        self.StartTrain = tk.Button(top)
        self.StartTrain.place(relx=0.823, rely=0.29, height=44, width=77)
        self.StartTrain.configure(activebackground="#ececec")
        self.StartTrain.configure(command=self.trainThread)
        self.StartTrain.configure(activeforeground="#000000")
        self.StartTrain.configure(background="#d9d9d9")
        self.StartTrain.configure(disabledforeground="#a3a3a3")
        self.StartTrain.configure(foreground="#000000")
        self.StartTrain.configure(highlightbackground="#d9d9d9")
        self.StartTrain.configure(highlightcolor="black")
        self.StartTrain.configure(pady="0")
        self.StartTrain.configure(text='''Train''')

        self.LoadModel = tk.Button(top)
        self.LoadModel.place(relx=0.823, rely=0.399, height=34, width=167)
        self.LoadModel.configure(command=self.loadModel)
        self.LoadModel.configure(activebackground="#ececec")
        self.LoadModel.configure(activeforeground="#000000")
        self.LoadModel.configure(background="#d9d9d9")
        self.LoadModel.configure(disabledforeground="#a3a3a3")
        self.LoadModel.configure(foreground="#000000")
        self.LoadModel.configure(highlightbackground="#d9d9d9")
        self.LoadModel.configure(highlightcolor="black")
        self.LoadModel.configure(pady="0")
        self.LoadModel.configure(text='''Load Model''')

        self.LoadTestFolder = tk.Button(top)
        self.LoadTestFolder.place(relx=0.916, rely=0.036, height=44, width=77)
        self.LoadTestFolder.configure(command=self.testDirectory)
        self.LoadTestFolder.configure(activebackground="#ececec")
        self.LoadTestFolder.configure(activeforeground="#000000")
        self.LoadTestFolder.configure(background="#d9d9d9")
        self.LoadTestFolder.configure(disabledforeground="#a3a3a3")
        self.LoadTestFolder.configure(foreground="#000000")
        self.LoadTestFolder.configure(highlightbackground="#d9d9d9")
        self.LoadTestFolder.configure(highlightcolor="black")
        self.LoadTestFolder.configure(pady="0")
        self.LoadTestFolder.configure(text='''Test Folder''')

        self.LoadTestLabels = tk.Button(top)
        self.LoadTestLabels.place(relx=0.915, rely=0.145, height=44, width=77)
        self.LoadTestLabels.configure(command=self.setTestLabel)
        self.LoadTestLabels.configure(activebackground="#ececec")
        self.LoadTestLabels.configure(activeforeground="#000000")
        self.LoadTestLabels.configure(background="#d9d9d9")
        self.LoadTestLabels.configure(disabledforeground="#a3a3a3")
        self.LoadTestLabels.configure(foreground="#000000")
        self.LoadTestLabels.configure(highlightbackground="#d9d9d9")
        self.LoadTestLabels.configure(highlightcolor="black")
        self.LoadTestLabels.configure(pady="0")
        self.LoadTestLabels.configure(text='''Test Labels''')

        self.StartTest = tk.Button(top)
        self.StartTest.place(relx=0.916, rely=0.29, height=44, width=77)
        self.StartTest.configure(activebackground="#ececec")
        self.StartTest.configure(activeforeground="#000000")
        self.StartTest.configure(background="#d9d9d9")
        self.StartTest.configure(disabledforeground="#a3a3a3")
        self.StartTest.configure(foreground="#000000")
        self.StartTest.configure(highlightbackground="#d9d9d9")
        self.StartTest.configure(highlightcolor="black")
        self.StartTest.configure(pady="0")
        self.StartTest.configure(text='''Test''')

        self.LoadImage = tk.Button(top)
        self.LoadImage.place(relx=0.823, rely=0.471, height=34, width=167)
        self.LoadImage.configure(command=self.setTestImage)
        self.LoadImage.configure(activebackground="#ececec")
        self.LoadImage.configure(activeforeground="#000000")
        self.LoadImage.configure(background="#d9d9d9")
        self.LoadImage.configure(disabledforeground="#a3a3a3")
        self.LoadImage.configure(foreground="#000000")
        self.LoadImage.configure(highlightbackground="#d9d9d9")
        self.LoadImage.configure(highlightcolor="black")
        self.LoadImage.configure(pady="0")
        self.LoadImage.configure(text='''Load Image''')

        self.TProgressbar1 = ttk.Progressbar(top)
        self.TProgressbar1.place(relx=0.021, rely=0.707, relwidth=0.783
                                 , relheight=0.0, height=22)
        self.TProgressbar1.configure(length="722", orient=tk.HORIZONTAL, mode='determinate')

        self.Scrolledtext1 = ScrolledText(top)
        self.Scrolledtext1.place(relx=0.021, rely=0.797, relheight=0.172
                                 , relwidth=0.783)
        self.Scrolledtext1.configure(background="white")
        self.Scrolledtext1.configure(font="TkTextFont")
        self.Scrolledtext1.configure(foreground="black")
        self.Scrolledtext1.configure(highlightbackground="#d9d9d9")
        self.Scrolledtext1.configure(highlightcolor="black")
        self.Scrolledtext1.configure(insertbackground="black")
        self.Scrolledtext1.configure(insertborderwidth="3")
        self.Scrolledtext1.configure(selectbackground="blue")
        self.Scrolledtext1.configure(selectforeground="white")
        self.Scrolledtext1.configure(wrap="none")

        self.TestImage = tk.Button(top)
        self.TestImage.place(relx=0.823, rely=0.543, height=34, width=167)
        self.TestImage.configure(command=self.testLoadedImage)
        self.TestImage.configure(activebackground="#ececec")
        self.TestImage.configure(activeforeground="#000000")
        self.TestImage.configure(background="#d9d9d9")
        self.TestImage.configure(disabledforeground="#a3a3a3")
        self.TestImage.configure(foreground="#000000")
        self.TestImage.configure(highlightbackground="#d9d9d9")
        self.TestImage.configure(highlightcolor="black")
        self.TestImage.configure(pady="0")
        self.TestImage.configure(text='''Test Image''')

        self.StopExec = tk.Button(top)
        self.StopExec.place(relx=0.823, rely=0.797, height=44, width=157)
        self.StopExec.configure(command=self.stopTraining)
        self.StopExec.configure(activebackground="#ececec")
        self.StopExec.configure(activeforeground="#000000")
        self.StopExec.configure(background="#ff0000")
        self.StopExec.configure(disabledforeground="#a3a3a3")
        self.StopExec.configure(foreground="#000000")
        self.StopExec.configure(highlightbackground="#d9d9d9")
        self.StopExec.configure(highlightcolor="black")
        self.StopExec.configure(pady="0")
        self.StopExec.configure(text='''Stop Execution''')

        self.ExitProgram = tk.Button(top)
        self.ExitProgram.configure(command=exitButton)
        self.ExitProgram.place(relx=0.823, rely=0.888, height=44, width=157)
        self.ExitProgram.configure(activebackground="#ececec")
        self.ExitProgram.configure(activeforeground="#000000")
        self.ExitProgram.configure(background="#d9d9d9")
        self.ExitProgram.configure(disabledforeground="#a3a3a3")
        self.ExitProgram.configure(foreground="#000000")
        self.ExitProgram.configure(highlightbackground="#d9d9d9")
        self.ExitProgram.configure(highlightcolor="black")
        self.ExitProgram.configure(pady="0")
        self.ExitProgram.configure(text='''Exit''')

        self.ModelStatus = tk.Message(top)
        self.ModelStatus.place(relx=0.823, rely=0.616, relheight=0.06
                               , relwidth=0.165)
        self.ModelStatus.configure(anchor='w')
        self.ModelStatus.configure(background="#d9d9d9")
        self.ModelStatus.configure(foreground="#000000")
        self.ModelStatus.configure(highlightbackground="#d9d9d9")
        self.ModelStatus.configure(highlightcolor="black")
        self.ModelStatus.configure(text='''Model:''')
        self.ModelStatus.configure(width=160)

        self.menubar = tk.Menu(top, font="TkMenuFont", bg=_bgcolor, fg=_fgcolor)
        top.configure(menu=self.menubar)

        self.initModel()


# The following code is added to facilitate the Scrolled widgets you specified.
class AutoScroll(object):
    '''Configure the scrollbars for a widget.'''

    def __init__(self, master):
        #  Rozen. Added the try-except clauses so that this class
        #  could be used for scrolled entry widget for which vertical
        #  scrolling is not supported. 5/7/14.
        try:
            vsb = ttk.Scrollbar(master, orient='vertical', command=self.yview)
        except:
            pass
        hsb = ttk.Scrollbar(master, orient='horizontal', command=self.xview)
        try:
            self.configure(yscrollcommand=self._autoscroll(vsb))
        except:
            pass
        self.configure(xscrollcommand=self._autoscroll(hsb))
        self.grid(column=0, row=0, sticky='nsew')
        try:
            vsb.grid(column=1, row=0, sticky='ns')
        except:
            pass
        hsb.grid(column=0, row=1, sticky='ew')
        master.grid_columnconfigure(0, weight=1)
        master.grid_rowconfigure(0, weight=1)
        # Copy geometry methods of master  (taken from ScrolledText.py)
        if py3:
            methods = tk.Pack.__dict__.keys() | tk.Grid.__dict__.keys() \
                      | tk.Place.__dict__.keys()
        else:
            methods = tk.Pack.__dict__.keys() + tk.Grid.__dict__.keys() \
                      + tk.Place.__dict__.keys()
        for meth in methods:
            if meth[0] != '_' and meth not in ('config', 'configure'):
                setattr(self, meth, getattr(master, meth))

    @staticmethod
    def _autoscroll(sbar):
        '''Hide and show scrollbar as needed.'''

        def wrapped(first, last):
            first, last = float(first), float(last)
            if first <= 0 and last >= 1:
                sbar.grid_remove()
            else:
                sbar.grid()
            sbar.set(first, last)

        return wrapped

    def __str__(self):
        return str(self.master)


def _create_container(func):
    '''Creates a ttk Frame with a given master, and use this new frame to
    place the scrollbars and the widget.'''

    def wrapped(cls, master, **kw):
        container = ttk.Frame(master)
        container.bind('<Enter>', lambda e: _bound_to_mousewheel(e, container))
        container.bind('<Leave>', lambda e: _unbound_to_mousewheel(e, container))
        return func(cls, container, **kw)

    return wrapped


class ScrolledText(AutoScroll, tk.Text):
    '''A standard Tkinter Text widget with scrollbars that will
    automatically show/hide as needed.'''

    @_create_container
    def __init__(self, master, **kw):
        tk.Text.__init__(self, master, **kw)
        AutoScroll.__init__(self, master)


import platform


def _bound_to_mousewheel(event, widget):
    child = widget.winfo_children()[0]
    if platform.system() == 'Windows' or platform.system() == 'Darwin':
        child.bind_all('<MouseWheel>', lambda e: _on_mousewheel(e, child))
        child.bind_all('<Shift-MouseWheel>', lambda e: _on_shiftmouse(e, child))
    else:
        child.bind_all('<Button-4>', lambda e: _on_mousewheel(e, child))
        child.bind_all('<Button-5>', lambda e: _on_mousewheel(e, child))
        child.bind_all('<Shift-Button-4>', lambda e: _on_shiftmouse(e, child))
        child.bind_all('<Shift-Button-5>', lambda e: _on_shiftmouse(e, child))


def _unbound_to_mousewheel(event, widget):
    if platform.system() == 'Windows' or platform.system() == 'Darwin':
        widget.unbind_all('<MouseWheel>')
        widget.unbind_all('<Shift-MouseWheel>')
    else:
        widget.unbind_all('<Button-4>')
        widget.unbind_all('<Button-5>')
        widget.unbind_all('<Shift-Button-4>')
        widget.unbind_all('<Shift-Button-5>')


def _on_mousewheel(event, widget):
    if platform.system() == 'Windows':
        widget.yview_scroll(-1 * int(event.delta / 120), 'units')
    elif platform.system() == 'Darwin':
        widget.yview_scroll(-1 * int(event.delta), 'units')
    else:
        if event.num == 4:
            widget.yview_scroll(-1, 'units')
        elif event.num == 5:
            widget.yview_scroll(1, 'units')


def _on_shiftmouse(event, widget):
    if platform.system() == 'Windows':
        widget.xview_scroll(-1 * int(event.delta / 120), 'units')
    elif platform.system() == 'Darwin':
        widget.xview_scroll(-1 * int(event.delta), 'units')
    else:
        if event.num == 4:
            widget.xview_scroll(-1, 'units')
        elif event.num == 5:
            widget.xview_scroll(1, 'units')


if __name__ == '__main__':
    vp_start_gui()
