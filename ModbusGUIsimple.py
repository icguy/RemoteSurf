from Tkinter import *
import tkFont
from pyModbusTCP.client import ModbusClient
from threading import Thread

SERVER_HOST = "192.168.0.104"
SERVER_PORT = 502

PRINT_ALL_MEMORY_ON_WRITE = True
START_OPENCV_THREAD = True

def intToUint16(val):
    assert -32768 <= val <= 32767
    return val if val >= 0 else 65536 + val

def uintToInt16(val):
    assert 0 <= val <= 65536
    return val if val <= 32767 else val - 65536

class ClientGUI:
    def __init__(self):
        self.client = ModbusClient()
        self.register_values_widgets = {}
        self.build_ui()

    def build_ui(self):
        # ui hierarchy:
        #
        #root
        #   connectframe
        #       connectlabel
        #       connectbutton
        #   mainframe
        #       registerframe
        #           reglabel
        #           registergridframe
        #               ...
        #       outputframe
        #           outputlabel
        #           outputtext

        x_padding = 5
        y_padding = 5

        root = Tk()
        self.root = root
        root.wm_title("RemoteSurf Modbus Client")
        root.protocol("WM_DELETE_WINDOW", self.delete_window)

        self.font = tkFont.Font(root = root, family = "Helvetica", size = 12)

        connectframe = Frame(root)
        connectbutton = Button(connectframe, text = "Connect", command = self.connectbutton_click)
        connectlabel = Label(connectframe, text = "Not connected.")
        mainframe = Frame(root)
        registerframe = Frame(mainframe)
        reglabel = Label(registerframe, text = "Set registers")
        registergridframe = Frame(registerframe)
        # outputframe = Frame(mainframe)
        # outputlabel = Label(outputframe, text = "Output")
        # vscrollbar = Scrollbar(outputframe)
        # hscrollbar = Scrollbar(outputframe)
        # outputtext = ThreadSafeConsole(outputframe, root, vscrollbar, font = self.font, wrap = NONE)

        connectframe.pack(side = TOP, fill = X)
        connectbutton.pack(side = RIGHT)
        connectlabel.pack(side = LEFT)
        mainframe.pack(side = BOTTOM, fill = BOTH, expand = YES)
        registerframe.pack(side = TOP, expand = YES, anchor = W)
        # outputframe.pack(side = BOTTOM, fill = BOTH, expand = YES)
        reglabel.pack(side = TOP, anchor = CENTER)
        registergridframe.pack(side = BOTTOM, anchor = W)
        # registerframe.config(bg = "cyan")
        # mainframe.config(bg = "pink")
        # registergridframe.config(bg = "red")

        registergridframe.columnconfigure(0, weight = 1)
        registergridframe.columnconfigure(1, weight = 1)
        registergridframe.columnconfigure(2, weight = 1)
        registergridframe.columnconfigure(3, weight = 1)

        self.x_pad = 10
        registergrid_widgets = []
        titles = ["Address", "Label", "Value", ""]
        col = 0
        for title in titles:
            title_label = Label(registergridframe, text = title)
            title_label.grid(row = 0, column = col, padx = self.x_pad)
            registergrid_widgets.append(title_label)
            col += 1

        registers_data = [(500, "x"),
                     (501, "y"),
                     (502, "z"),
                     (503, "A"),
                     (504, "B"),
                     (505, "C"),
                     ]

        for i in range(len(registers_data)):
            reg_data = registers_data[i]
            row = i + 1
            self.add_register(registergridframe, reg_data, row, registergrid_widgets)

        # hscrollbar.config(orient = HORIZONTAL, command = outputtext.xview)
        # hscrollbar.pack(side = BOTTOM, fill = X)
        # outputtext.config(state = DISABLED, yscrollcommand = vscrollbar.set, xscrollcommand = hscrollbar.set)  #must change to NORMAL before writing text programmatically
        # outputtext.pack(side = LEFT, fill = BOTH, expand = YES, padx = x_padding, pady = y_padding)
        # vscrollbar.config(command = outputtext.yview)
        # vscrollbar.pack(side = RIGHT, fill = Y)

        self.connectframe = connectframe
        self.connectlabel = connectlabel
        self.connectbutton = connectbutton
        self.mainframe = mainframe
        self.registerframe = registerframe
        self.reglabel = reglabel
        self.registergridframe = registergridframe
        # self.outputframe = outputframe
        # self.outputlabel = outputlabel
        # self.vscrollbar = vscrollbar
        # self.hscrollbar = hscrollbar
        # self.outputtext = outputtext

        root.update()
        root.minsize(root.winfo_width(), root.winfo_height())
        root.mainloop()

    def add_register(self, master, data, row, widget_list):
        regaddresslabel = Label(master, text=str(data[0]))
        regaddresslabel.grid(row=row, column=0)
        reglabellabel = Label(master, text=data[1])
        reglabellabel.grid(row=row, column=1)
        regvalueentry = AccessibleEntry(master, justify = RIGHT)
        regvalueentry.set("0")
        regvalueentry.grid(row=row, column=2, padx=self.x_pad)
        regsetbtn = Button(master, text="Set", command = self.setbutton_click)
        regsetbtn.grid(row=row, column=3)
        widget_list.append(regaddresslabel)
        widget_list.append(reglabellabel)
        widget_list.append(regvalueentry)
        widget_list.append(regsetbtn)
        self.register_values_widgets[data[0]] = (0, regvalueentry)

    def connectbutton_click(self):
        if self.client.is_open():
            self.client.close()
        else:
            self.client.host(SERVER_HOST)
            self.client.port(SERVER_PORT)
            if self.client.open():
                print "Connection established"
                self.refresh_values()
                self.read_robot_pos()
            else:
                print "ERROR: Connecting failed"
        self.update_texts()

    def read_robot_pos(self):
        print "Reading robot position:"
        for i in range(1000, 1006):
            if self.client.is_open():
                real_val_uint = self.client.read_input_registers(i)[0]
                real_val_holding_uint = self.client.read_holding_registers(i)[0]
                assert real_val_uint == real_val_holding_uint
                real_val_int = uintToInt16(real_val_uint)
                print i, real_val_int
            else:
                print "ERROR: Read could not be completed, client not connected."
                self.update_texts()
                break
        print "Read done."

    def refresh_values(self):
        for address in self.register_values_widgets:
            if self.client.is_open():
                value, widget = self.register_values_widgets[address]
                real_val_uint = self.client.read_input_registers(address)[0]
                real_val_holding_uint = self.client.read_holding_registers(address)[0]
                assert real_val_uint == real_val_holding_uint
                real_val_int = uintToInt16(real_val_uint)
                widget.set(str(real_val_int))
                self.register_values_widgets[address] = (real_val_int, widget)
            else:
                print "ERROR: Read could not be completed, client not connected."
                self.update_texts()
                break
        print "Refresh done."

    def update_texts(self):
        if self.client.is_open():
            self.connectlabel.config(text = "Connected to: %s:%d" % (SERVER_HOST, SERVER_PORT))
            self.connectbutton.config(text = "Disconnect")
        else:
            self.connectbutton.config(text = "Connect")
            self.connectlabel.config(text = "Not connected.")

    def print_memory(self):
        self.refresh_values()
        print "Memory dump:"
        print "------------"
        for address in self.register_values_widgets:
            val, widget = self.register_values_widgets[address]
            print address, val
        print "------------"

    def setbutton_click(self):
        if not self.client.is_open():
            print "ERROR: Not connected to client"
            return

        for address in self.register_values_widgets:
            value, widget = self.register_values_widgets[address]
            widgetvalue_int = None
            try:
                widgetvalue_int = int(widget.get())
            except ValueError:
                print "ERROR: Wrong input format in value entry for address: %d" % address
                continue

            if value == widgetvalue_int:
                continue

            if not (-32768 <= widgetvalue_int <= 32767):
                print "ERROR: -32768 <= value <= 32767 is false for address: %d" % address
                continue

            widgetvalue_uint = intToUint16(widgetvalue_int)

            if self.client.is_open():
                retval = self.client.write_single_register(address, widgetvalue_uint)
                if retval:
                    self.register_values_widgets[address] = (widgetvalue_int, widget)
                    print "Register written. Address: %d, value: %d" % (address, widgetvalue_int)
                else:
                    print "ERROR: Write failed. Address: %d, value: %d" % (address, widgetvalue_int)
            else:
                print "ERROR: client not connected."
                self.update_texts()
        self.refresh_values()
        if PRINT_ALL_MEMORY_ON_WRITE:
            self.print_memory()

    def delete_window(self):
        self.client.close()
        self.root.quit()

class AccessibleEntry(Entry):
    def __init__(self, master, cnf = {}, **kw):
        Entry.__init__(self, master, cnf, **kw)
        self.var = StringVar()
        self.config(textvariable = self.var)

    def get(self):
        return self.var.get()

    def set(self, val):
        self.var.set(val)

def runOpencv():
    import CamGrabber
    CamGrabber.run(None)

if __name__ == '__main__':
    opencvThread = None
    if START_OPENCV_THREAD:
        opencvThread = Thread(target=runOpencv)
        opencvThread.start()
    ClientGUI()
    if opencvThread:
        opencvThread.join()